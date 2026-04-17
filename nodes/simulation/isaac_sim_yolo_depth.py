import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import threading


class IsaacSimYoloDepth(Node):
    def __init__(self):
        super().__init__('isaac_sim_yolo_depth')
        self.bridge = CvBridge()
        self.model = YOLO('/home/sejong/yolo_dataset_real_v2/runs/train_real_v2/weights/best.pt')
        self.lock = threading.Lock()

        self.depth_image = None

        # 카메라 파라미터 (camera_info 수신 전 기본값)
        self.fx = 615.0
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0

        self.sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10)

        self.get_logger().info('Isaac Sim YOLO Depth 테스트 시작!')

    def camera_info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.get_logger().info(f'카메라 파라미터 수신: fx={self.fx:.1f} fy={self.fy:.1f} cx={self.cx:.1f} cy={self.cy:.1f}')

    def depth_callback(self, msg):
        with self.lock:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        with self.lock:
            depth = self.depth_image.copy() if self.depth_image is not None else None

        results = self.model(frame, conf=0.5, verbose=False)
        colors = {
            'up_button':   (0, 255, 0),   # 초록
            'down_button': (0, 0, 255),   # 빨강
        }

        for result in results:
            for box in result.boxes:
                cls = result.names[int(box.cls)]
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = colors.get(cls, (255, 255, 0))

                # 박스 중심점
                cx_box = (x1 + x2) // 2
                cy_box = (y1 + y2) // 2

                # 3D 좌표 계산
                xyz_text = ''
                if depth is not None:
                    d = float(depth[cy_box, cx_box])
                    if d > 0 and not np.isnan(d):
                        X = (cx_box - self.cx) / self.fx * d
                        Y = (cy_box - self.cy) / self.fy * d
                        Z = d
                        xyz_text = f'X:{X:.3f} Y:{Y:.3f} Z:{Z:.3f}m'
                        self.get_logger().info(f'[{cls}] {xyz_text}')

                # 시각화
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx_box, cy_box), 4, color, -1)
                cv2.putText(frame, f'{cls} {conf:.2f}', (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if xyz_text:
                    cv2.putText(frame, xyz_text, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow('Isaac Sim YOLO Depth', frame)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = IsaacSimYoloDepth()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
