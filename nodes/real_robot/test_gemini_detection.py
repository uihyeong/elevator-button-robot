"""
Gemini VLM 버튼 인식 단독 테스트 스크립트.
ROS2 / 로봇 연결 없이 카메라만 있으면 실행 가능.

실행:
  export GEMINI_API_KEY="your_key"
  python3 nodes/real_robot/test_gemini_detection.py

  # 이미지 파일로 테스트할 경우
  python3 nodes/real_robot/test_gemini_detection.py --image path/to/image.jpg

  # 카메라 번호 지정
  python3 nodes/real_robot/test_gemini_detection.py --camera 2

  # 숫자 버튼 인식 모드 (기본: UP/DOWN)
  python3 nodes/real_robot/test_gemini_detection.py --mode number --floor 3

조작:
  q        : 종료
  s        : 현재 프레임 저장 (saved_frame.jpg)
  SPACE    : 즉시 Gemini 호출
"""

import argparse
import json
import os
import sys
import threading
import time

import cv2
import numpy as np

from google import genai
from google.genai import types

# ─── 설정 ─────────────────────────────────────────────────────────────────────

GEMINI_MODEL  = 'gemini-2.5-flash'
CALL_INTERVAL = 3.0
WINDOW_NAME   = 'Gemini Button Detection Test'

# ─── 프롬프트 ─────────────────────────────────────────────────────────────────

def updown_prompt():
    return (
        'Find the elevator UP button and DOWN button. '
        'Return JSON only: [{"point": [y, x], "label": "UP"}, ...]. '
        'point is [y, x] normalized 0-1000. Empty array [] if not found.'
    )

def number_prompt(target_floor: int):
    hint = f'B{abs(target_floor)}' if target_floor < 0 else str(target_floor)
    return (
        f'Find all elevator floor buttons. '
        f'Return JSON only: [{{"point": [y, x], "label": "<text on button>", "floor": <int>}}, ...]. '
        f'point is [y, x] normalized 0-1000. floor is integer (basement is negative: B1=-1). '
        f'Target floor: {hint}. Empty array [] if not found.'
    )

# ─── Gemini 호출 ──────────────────────────────────────────────────────────────

def call_gemini(api_key: str, frame: np.ndarray, prompt: str) -> tuple[list, str]:
    _, buf = cv2.imencode('.jpg', frame)
    image_bytes = buf.tobytes()

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type='application/json',
        ),
    )
    raw = resp.text

    cleaned = raw.strip().lstrip('`').rstrip('`')
    if cleaned.startswith('json'):
        cleaned = cleaned[4:]
    detections = json.loads(cleaned)
    if not isinstance(detections, list):
        detections = []
    return detections, raw


# ─── 점 그리기 ────────────────────────────────────────────────────────────────

def draw_detections(frame: np.ndarray, detections: list, mode: str, target_floor: int):
    h, w = frame.shape[:2]
    for det in detections:
        pt  = det.get('point')
        lbl = det.get('label', '?')
        if not pt or len(pt) != 2:
            continue

        cy = int(float(pt[0]) / 1000.0 * h)
        cx = int(float(pt[1]) / 1000.0 * w)

        match = True if mode == 'updown' else det.get('floor') == target_floor
        color = (0, 255, 0) if match else (0, 165, 255)

        cv2.circle(frame, (cx, cy), 18, color, 3)
        cv2.circle(frame, (cx, cy), 4, color, -1)
        cv2.putText(frame, lbl, (cx + 22, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0,   help='카메라 번호 (기본: 0)')
    parser.add_argument('--image',  type=str, default='',  help='이미지 파일 경로 (지정 시 카메라 대신 사용)')
    parser.add_argument('--mode',   type=str, default='updown', choices=['updown', 'number'])
    parser.add_argument('--floor',  type=int, default=3,   help='number 모드에서 목표 층수')
    args = parser.parse_args()

    api_key = os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        print('[ERROR] GEMINI_API_KEY 환경변수가 없습니다.')
        print('  export GEMINI_API_KEY="your_key"')
        sys.exit(1)

    # ── 카메라 또는 이미지 로드 ──
    static_image = None
    cap = None
    rs_pipeline  = None

    if args.image:
        static_image = cv2.imread(args.image)
        if static_image is None:
            print(f'[ERROR] 이미지 파일을 열 수 없습니다: {args.image}')
            sys.exit(1)
        print(f'[INFO] 이미지 파일 모드: {args.image}')
    else:
        # D435 → pyrealsense2 우선 사용
        try:
            import pyrealsense2 as rs
            rs_pipeline = rs.pipeline()
            rs_config   = rs.config()
            rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            rs_pipeline.start(rs_config)
            print('[INFO] D435 RealSense 카메라 연결됨 (pyrealsense2)')
        except Exception as e:
            rs_pipeline = None
            print(f'[WARN] pyrealsense2 실패 ({e}), OpenCV VideoCapture 시도...')
            cap = cv2.VideoCapture(args.camera)
            if not cap.isOpened():
                print(f'[ERROR] 카메라 {args.camera}번을 열 수 없습니다.')
                sys.exit(1)
            for _ in range(30):
                cap.read()
            print(f'[INFO] 카메라 {args.camera}번 연결됨')

    print(f'[INFO] 모델: {GEMINI_MODEL}')
    print(f'[INFO] 모드: {args.mode}' +
          (f'  목표층: {args.floor}F' if args.mode == 'number' else ''))
    print('[INFO] 조작: q=종료  s=프레임저장  SPACE=즉시호출')

    # ── 상태 변수 ──
    detections   = []
    raw_text     = ''
    status_msg   = '대기 중... SPACE키로 즉시 호출'
    busy         = False
    last_call    = 0.0
    force_call   = threading.Event()

    def gemini_worker(frame_snap):
        nonlocal detections, raw_text, status_msg, busy, last_call
        try:
            prompt = updown_prompt() if args.mode == 'updown' else number_prompt(args.floor)
            status_msg = 'Gemini 호출 중...'
            dets, raw = call_gemini(api_key, frame_snap, prompt)
            detections = dets
            raw_text   = raw
            found = [f'{d.get("label","?")}({d.get("confidence",0):.0%})' for d in dets]
            status_msg = f'감지: {found}' if found else '감지 없음'
            print(f'[Gemini] {status_msg}')
            print(f'[Raw]    {raw[:300]}')
        except Exception as e:
            status_msg = f'오류: {e}'
            print(f'[ERROR] {e}')
        finally:
            busy = False
            last_call = time.time()

    # ── 메인 루프 ──
    while True:
        if static_image is not None:
            frame = static_image.copy()
        elif rs_pipeline is not None:
            frames = rs_pipeline.wait_for_frames()
            color  = frames.get_color_frame()
            if not color:
                continue
            frame = np.asanyarray(color.get_data())
        else:
            ret, frame = cap.read()
            if not ret:
                print('[ERROR] 카메라 프레임 읽기 실패')
                break

        now = time.time()

        # 이미지 모드: 최초 1회 + SPACE 수동만 호출
        # 카메라 모드: CALL_INTERVAL 자동 + SPACE 수동
        if static_image is not None:
            auto_trigger = (last_call == 0.0)   # 최초 1회만
        else:
            auto_trigger = (now - last_call >= CALL_INTERVAL)
        force_trigger = force_call.is_set()

        if (auto_trigger or force_trigger) and not busy:
            busy = True
            force_call.clear()
            snap = frame.copy()
            threading.Thread(target=gemini_worker, args=(snap,), daemon=True).start()

        # 결과 그리기
        display = draw_detections(frame.copy(), detections, args.mode, args.floor)

        # HUD
        mode_label = f'모드: {args.mode.upper()}' + \
                     (f'  목표: {args.floor}F' if args.mode == 'number' else '')
        cv2.putText(display, mode_label,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(display, status_msg,
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 200, 255) if busy else (200, 200, 200), 2)

        # 다음 호출까지 카운트다운
        remaining = max(0.0, CALL_INTERVAL - (now - last_call))
        cv2.putText(display, f'다음 자동 호출: {remaining:.1f}s  (SPACE=즉시)',
                    (10, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        # Gemini 응답 원문 (하단)
        if raw_text:
            snippet = raw_text[:120].replace('\n', ' ')
            cv2.putText(display, snippet,
                        (10, display.shape[0] - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = 'saved_frame.jpg'
            cv2.imwrite(fname, frame)
            print(f'[INFO] 프레임 저장: {fname}')
        elif key == ord(' '):
            force_call.set()
            print('[INFO] 즉시 호출 트리거')

    if cap:
        cap.release()
    if rs_pipeline:
        rs_pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
