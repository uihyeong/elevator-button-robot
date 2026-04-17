import rclpy
import rclpy.time
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import threading
import tf2_ros
import tf2_geometry_msgs


class IsaacSimYoloTF(Node):
    def __init__(self):
        super().__init__('isaac_sim_yolo_tf')
        self.bridge = CvBridge()
        self.model = YOLO('/home/sejong/yolo_dataset_real_v2/runs/train_real_v2/weights/best.pt')
        self.lock = threading.Lock()

        self.depth_image = None

        # 카메라 파라미터 기본값
        self.fx = 615.0
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10)

        self.get_logger().info('Isaac Sim YOLO TF 테스트 시작!')

    def camera_info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def depth_callback(self, msg):
        with self.lock:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        with self.lock:
            depth = self.depth_image.copy() if self.depth_image is not None else None

        results = self.model(frame, conf=0.5, verbose=False)
        colors = {
            'up_button':   (0, 255, 0),
            'down_button': (0, 0, 255),
        }

        for result in results:
            for box in result.boxes:
                cls = result.names[int(box.cls)]
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = colors.get(cls, (255, 255, 0))

                cx_box = (x1 + x2) // 2
                cy_box = (y1 + y2) // 2

                if depth is None:
                    continue

                d = float(depth[cy_box, cx_box])
                if d <= 0 or np.isnan(d):
                    continue

                # 카메라 기준 XYZ
                X_cam = (cx_box - self.cx) / self.fx * d
                Y_cam = (cy_box - self.cy) / self.fy * d
                Z_cam = d

                # TF 변환: camera_color_optical_frame → open_manipulator_x
                point_cam = PointStamped()
                point_cam.header.frame_id = 'camera_color_optical_frame'
                point_cam.header.stamp = rclpy.time.Time().to_msg()
                point_cam.point.x = X_cam
                point_cam.point.y = Y_cam
                point_cam.point.z = Z_cam

                try:
                    point_base = self.tf_buffer.transform(
                        point_cam, 'open_manipulator_x')
                    X_base = point_base.point.x
                    Y_base = point_base.point.y
                    Z_base = point_base.point.z

                    self.get_logger().info(
                        f'[{cls}] 카메라: ({X_cam:.3f}, {Y_cam:.3f}, {Z_cam:.3f}) '
                        f'| 베이스: ({X_base:.3f}, {Y_base:.3f}, {Z_base:.3f})')

                    base_text = f'B({X_base:.2f},{Y_base:.2f},{Z_base:.2f})'
                    cv2.putText(frame, base_text, (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                except Exception as e:
                    self.get_logger().warn(f'TF 변환 실패: {e}')

                # 시각화
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx_box, cy_box), 4, color, -1)
                cv2.putText(frame, f'{cls} {conf:.2f}', (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f'C({X_cam:.2f},{Y_cam:.2f},{Z_cam:.2f})',
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow('Isaac Sim YOLO TF', frame)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = IsaacSimYoloTF()
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
