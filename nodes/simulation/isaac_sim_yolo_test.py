import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2


class IsaacSimYoloTest(Node):
    def __init__(self):
        super().__init__('isaac_sim_yolo_test')
        self.bridge = CvBridge()
        self.model = YOLO('/home/sejong/yolo_dataset_real_v2/runs/train_real_v2/weights/best.pt')

        self.sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10)

        self.get_logger().info('Isaac Sim YOLO 테스트 시작!')
        self.get_logger().info('토픽 대기 중: /camera/color/image_raw')

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        results = self.model(frame, conf=0.5, verbose=False)

        for result in results:
            for box in result.boxes:
                cls = result.names[int(box.cls)]
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                colors = {
                    'up_button':   (0, 255, 0),    # 초록
                    'down_button': (0, 0, 255),     # 빨강
                }
                color = colors.get(cls, (255, 255, 0))  # 기타: 노랑
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{cls} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                self.get_logger().info(f'감지: {cls} {conf:.2f}')

        cv2.imshow('Isaac Sim YOLO', frame)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = IsaacSimYoloTest()
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
