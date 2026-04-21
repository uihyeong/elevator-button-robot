"""
웹캠으로 YOLO-seg + EasyOCR 파이프라인 테스트 (ROS2 불필요).

실행:
  python3 nodes/test/webcam_ocr_test.py
  python3 nodes/test/webcam_ocr_test.py --camera 1      # 웹캠 인덱스 변경
  python3 nodes/test/webcam_ocr_test.py --image img.jpg  # 정지 이미지 테스트

조작:
  q / ESC : 종료
  s       : 현재 프레임 저장 (ocr_capture_YYYYMMDD_HHMMSS.jpg)
  r       : OCR 캐시 초기화 (강제 재인식)
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

# ─── 설정 ─────────────────────────────────────────────────────────────────────

NUM_MODEL_PATH = '/home/sejong/yolo_dataset_num/runs/segment/train/weights/best.pt'
OCR_INTERVAL   = 5      # 매 N프레임마다 OCR 실행
YOLO_CONF      = 0.3    # 웹캠 테스트용으로 threshold 낮춤


# ─── OCR 헬퍼 ─────────────────────────────────────────────────────────────────

def read_number(ocr, crop):
    h, w = crop.shape[:2]
    if h < 10 or w < 10:
        return None, 0.0

    scale   = max(64 / max(h, w), 1.0)
    resized = cv2.resize(crop, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)

    gray     = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    results   = ocr.readtext(enhanced, allowlist='0123456789Bb', detail=1)
    best_conf = 0.0
    best_num  = None

    for (_, text, conf) in results:
        text = text.strip().upper()
        if text.isdigit() and conf > best_conf:
            best_conf = conf
            best_num  = int(text)
        elif text.startswith('B') and text[1:].isdigit() and conf > best_conf:
            best_conf = conf
            best_num  = -int(text[1:])

    return best_num, best_conf


def box_key(x1, y1, x2, y2, grid=20):
    return f'{x1//grid}_{y1//grid}_{x2//grid}_{y2//grid}'


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def run(camera_idx: int, image_path: str | None):
    print('YOLO 모델 로드 중...')
    model = YOLO(NUM_MODEL_PATH)

    print('EasyOCR 초기화 중...')
    ocr = easyocr.Reader(['en'], gpu=False)
    print('준비 완료!\n')

    # ── 정지 이미지 모드 ───────────────────────────────────────────────────────
    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f'이미지를 열 수 없음: {image_path}')
            return

        results = model(frame, conf=YOLO_CONF, verbose=False)
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pad  = 5
                crop = frame[max(0,y1-pad):y2+pad, max(0,x1-pad):x2+pad]
                num, ocr_conf = read_number(ocr, crop)
                label = str(num) if num is not None else '?'
                color = (0, 255, 0) if num is not None else (180, 180, 180)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{label}  det={conf:.2f} ocr={ocr_conf:.2f}',
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                print(f'  박스 ({x1},{y1})-({x2},{y2})  →  숫자: {label}  (det={conf:.2f}, ocr={ocr_conf:.2f})')

        cv2.imshow('OCR Test (정지 이미지)', frame)
        print('아무 키나 누르면 종료')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ── 웹캠 모드 ──────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f'카메라 {camera_idx} 열기 실패. --camera 옵션으로 인덱스를 바꿔보세요.')
        return

    print(f'카메라 {camera_idx} 연결됨. 조작: [q/ESC] 종료  [s] 저장  [r] OCR 캐시 초기화')

    cache       = {}
    frame_count = 0
    fps_time    = time.time()
    fps         = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print('프레임 읽기 실패')
            break

        frame_count += 1
        run_ocr = (frame_count % OCR_INTERVAL == 0)

        # FPS 계산
        now = time.time()
        if now - fps_time >= 1.0:
            fps      = frame_count / (now - fps_time) if now > fps_time else 0
            fps_time = now

        # YOLO 추론
        results = model(frame, conf=YOLO_CONF, verbose=False)

        detected = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                pad  = 5
                cx1  = max(0, x1 - pad);  cy1 = max(0, y1 - pad)
                cx2  = min(frame.shape[1], x2 + pad);  cy2 = min(frame.shape[0], y2 + pad)
                crop = frame[cy1:cy2, cx1:cx2]

                key = box_key(x1, y1, x2, y2)
                if run_ocr or key not in cache:
                    num, ocr_conf = read_number(ocr, crop)
                    cache[key] = (num, ocr_conf)
                else:
                    num, ocr_conf = cache.get(key, (None, 0.0))

                detected.append((x1, y1, x2, y2, conf, num, ocr_conf))

        # 시각화
        for (x1, y1, x2, y2, conf, num, ocr_conf) in detected:
            label = str(num) if num is not None else '?'
            color = (0, 255, 0) if num is not None else (180, 180, 180)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame,
                        f'{label}  det={conf:.2f} ocr={ocr_conf:.2f}',
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # HUD
        h, w = frame.shape[:2]
        cv2.putText(frame, f'FPS: {fps:.1f}  buttons: {len(detected)}',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.putText(frame, 'q:quit  s:save  r:reset cache',
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Webcam OCR Test', frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):   # q 또는 ESC
            break
        elif key == ord('s'):
            fname = f'ocr_capture_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
            cv2.imwrite(fname, frame)
            print(f'저장됨: {fname}')
        elif key == ord('r'):
            cache.clear()
            print('OCR 캐시 초기화')

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='웹캠 OCR 테스트')
    parser.add_argument('--camera', type=int, default=0, help='웹캠 인덱스 (기본값: 0)')
    parser.add_argument('--image',  type=str, default=None, help='정지 이미지 경로')
    args = parser.parse_args()
    run(args.camera, args.image)
