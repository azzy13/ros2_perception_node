import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImageSaver(Node):
    def __init__(self, output_dir, topic_name='/Sim/SceneDroneSensors/robots/Drone1/sensors/front_center1/scene_camera/image'):
        super().__init__('image_saver')
        self.subscription = self.create_subscription(Image, topic_name, self.listener_callback, 10)
        self.bridge = CvBridge()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.image_count = 0

    def listener_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Conversion failed: {e}")
            return

        # Save the image to disk
        image_path = os.path.join(self.output_dir, f"image_{self.image_count:04d}.png")
        cv2.imwrite(image_path, cv_image)
        self.get_logger().info(f"Saved {image_path}")
        self.image_count += 1

def main(args=None):
    rclpy.init(args=args)
    output_dir = '/output'
    node = ImageSaver(output_dir)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
