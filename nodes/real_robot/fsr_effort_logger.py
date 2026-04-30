"""
FSR406 + joint effort 동시 로깅 스크립트.

FSR로 관절을 탭했을 때의 effort 값을 확인해
contact_detector.py의 COLLISION_THRESHOLD(현재 80) 튜닝에 사용.

사용법:
  # ROS2 실행 중인 상태에서
  python3 nodes/real_robot/fsr_effort_logger.py --port /dev/ttyACM0

출력 CSV:
  time_ms, fsr_value, fsr_pressed, joint2_effort, joint3_effort, joint4_effort

단축키:
  Ctrl+C  종료 (fsr_effort_log.csv 저장)
"""

import argparse
import csv
import os
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import serial

MONITOR_JOINTS  = ['joint2', 'joint3', 'joint4']
FSR_THRESHOLD   = 70    # 아두이노 코드와 동일하게 맞출 것
SERIAL_BAUD     = 9600
OUTPUT_FILE     = os.path.join(os.path.dirname(__file__), 'fsr_effort_log.csv')


class FsrEffortLogger(Node):
    def __init__(self):
        super().__init__('fsr_effort_logger')
        self._efforts = {j: 0.0 for j in MONITOR_JOINTS}
        self._lock = threading.Lock()
        self.create_subscription(JointState, '/joint_states', self._cb_joint, 10)

    def _cb_joint(self, msg: JointState):
        with self._lock:
            for j in MONITOR_JOINTS:
                if j in msg.name:
                    idx = msg.name.index(j)
                    if idx < len(msg.effort):
                        self._efforts[j] = msg.effort[idx]

    def get_efforts(self):
        with self._lock:
            return dict(self._efforts)


def serial_ros_loop(port: str):
    rclpy.init()
    node = FsrEffortLogger()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    rows = []
    start_ms = int(time.time() * 1000)

    print(f'포트 {port} 연결 중...')
    try:
        ser = serial.Serial(port, SERIAL_BAUD, timeout=1)
    except Exception as e:
        print(f'시리얼 연결 실패: {e}')
        return

    time.sleep(2.0)  # 아두이노 리셋 대기
    ser.readline()   # 헤더 줄 버리기

    print('로깅 시작! FSR로 관절을 탭하세요. Ctrl+C로 종료.\n')
    print(f"{'time_ms':>8}  {'fsr':>5}  {'pressed':>7}  "
          f"{'j2_effort':>10}  {'j3_effort':>10}  {'j4_effort':>10}")
    print('-' * 65)

    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) != 3:
                continue

            try:
                t_arduino = int(parts[0])
                fsr_val   = int(parts[1])
                pressed   = int(parts[2])
            except ValueError:
                continue

            efforts = node.get_efforts()
            t_ms    = int(time.time() * 1000) - start_ms

            row = [
                t_ms, fsr_val, pressed,
                round(efforts['joint2'], 2),
                round(efforts['joint3'], 2),
                round(efforts['joint4'], 2),
            ]
            rows.append(row)

            marker = ' ◀ TAP' if pressed else ''
            print(f"{t_ms:>8}  {fsr_val:>5}  {pressed:>7}  "
                  f"{efforts['joint2']:>10.1f}  "
                  f"{efforts['joint3']:>10.1f}  "
                  f"{efforts['joint4']:>10.1f}{marker}")

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        node.destroy_node()
        rclpy.shutdown()

        with open(OUTPUT_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['time_ms', 'fsr_value', 'fsr_pressed',
                 'joint2_effort', 'joint3_effort', 'joint4_effort'])
            writer.writerows(rows)

        print(f'\n✅ {len(rows)}행 저장 완료: {OUTPUT_FILE}')
        _print_summary(rows)


def _print_summary(rows):
    if not rows:
        return
    tap_rows = [r for r in rows if r[2] == 1]  # pressed==1
    if not tap_rows:
        print('탭 감지 없음.')
        return

    for i, name in enumerate(['joint2', 'joint3', 'joint4'], start=3):
        vals = [abs(r[i]) for r in tap_rows]
        print(f'{name}  탭 시 effort  max={max(vals):.1f}  avg={sum(vals)/len(vals):.1f}')
    print(f'\n현재 COLLISION_THRESHOLD = 80  —  위 max 값 참고해서 조정하세요.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default='/dev/ttyACM0',
                        help='아두이노 시리얼 포트 (기본값: /dev/ttyACM0)')
    args = parser.parse_args()
    serial_ros_loop(args.port)
