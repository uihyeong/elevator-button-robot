import os
import threading

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_PATH = os.path.join(_REPO_ROOT, 'yolo', 'weights', 'best_room.pt')
CONFIDENCE = 0.5


class RoomSignDetector(Node):
    def __init__(self):
        super().__init__('room_sign_detector')
        self.bridge = CvBridge()
        self.latest_frame = None
        self.lock = threading.Lock()

        self.model = YOLO(MODEL_PATH)
        self.get_logger().info('모델 로드 완료')

        self.sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.color_callback, 10)

        self.get_logger().info('실시간 감지 시작! q: 종료')

    def color_callback(self, msg):
        with self.lock:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def run(self):
        while rclpy.ok():
            with self.lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None

            if frame is None:
                continue

            results = self.model(frame, conf=CONFIDENCE, device=0, verbose=False)
            annotated = results[0].plot()

            for box in results[0].boxes:
                conf = float(box.conf)
                cls = int(box.cls)
                name = self.model.names[cls]
                self.get_logger().info(f'감지: {name} ({conf:.2f})')

            cv2.imshow('Room Sign Detector', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


def main():
    rclpy.init()
    node = RoomSignDetector()

    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    node.run()


if __name__ == '__main__':
    main()
