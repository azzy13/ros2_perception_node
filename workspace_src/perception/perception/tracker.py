import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import torch
import cv2
import os
from argparse import Namespace
from perception_interfaces.msg import DetectionFrame, DetectedObject

import sys
sys.path.append("/ros2_ws/src/perception/perception/vehicle_perception")
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess
from yolox.data.data_augment import preproc
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker

class ByteTrackNode(Node):
    def __init__(self):
        super().__init__('bytetrack_node')

        self.subscriber = self.create_subscription(
            Image,
            '/Sim/SceneDroneSensors/robots/Drone1/sensors/front_center1/scene_camera/image',
            self.callback,
            10
        )

        self.publisher = self.create_publisher(Image, '/tracking/image', 10)

        self.bridge = CvBridge()

        self.device = "cuda:0"
        self.exp = get_exp("/byteTrack_data/exps/example/mot/carla_drone.py", exp_name="carla_drone")
        self.model = self.exp.get_model().to(self.device).eval()
        ckpt = torch.load("/byteTrack_data/pretrained/ground.pth.tar", map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model = fuse_model(self.model).half()

        self.test_size = self.exp.test_size
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.conf_thres = 0.7
        self.nms_thres = 0.8

        self.tracker = BYTETracker(Namespace(
            track_thresh=0.7,
            track_buffer=300,
            match_thresh=0.8,
            aspect_ratio_thresh=4,
            min_box_area=10,
            mot20=False
        ), frame_rate=10)

        self.frame_count = 0
        self.output_file = "/output/tracking_results.csv"

        # Ensure the file is created and write the header
        with open(self.output_file, 'w') as f:
            f.write('frame,id,x1,y1,w,h,score,-1,-1,-1\n')

        # Ensure the output image directory exists
        self.image_output_dir = "/output/inference_frames"
        os.makedirs(self.image_output_dir, exist_ok=True)

    def callback(self, img_msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        img, ratio = preproc(frame, self.test_size, self.rgb_means, self.std)
        img_tensor = torch.from_numpy(img).unsqueeze(0).float().to(self.device).half()

        with torch.no_grad():
            outputs = self.model(img_tensor)
            outputs = postprocess(outputs, self.exp.num_classes, self.conf_thres, self.nms_thres)

        online_targets = []
        if outputs[0] is not None:
            online_targets = self.tracker.update(outputs[0], frame.shape[:2], self.test_size)

        online_tlwhs, online_ids, scores = [], [], []
        for t in online_targets:
            tlwh = t.tlwh
            vertical = tlwh[2] / tlwh[3] > 4
            if tlwh[2] * tlwh[3] > 10 and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(t.track_id)
                scores.append(t.score)

        tracking_img = plot_tracking(
            frame, online_tlwhs, online_ids, frame_id=self.frame_count, fps=30
        )

        # Save the inference image with bounding boxes
        image_save_path = os.path.join(self.image_output_dir, f"frame_{self.frame_count:05d}.jpg")
        cv2.imwrite(image_save_path, tracking_img)

        # Immediately write each frame's results
        with open(self.output_file, 'a') as f:
            for tlwh, track_id, score in zip(online_tlwhs, online_ids, scores):
                x1, y1, w, h = tlwh
                line = f'{self.frame_count},{track_id},{round(x1,1)},{round(y1,1)},{round(w,1)},{round(h,1)},{score:.2f},-1,-1,-1\n'
                f.write(line)

        tracking_msg = self.bridge.cv2_to_imgmsg(tracking_img, encoding='bgr8')
        tracking_msg.header = Header(stamp=self.get_clock().now().to_msg())
        self.publisher.publish(tracking_msg)

        self.get_logger().info(f"Published and saved frame {self.frame_count}")
        self.frame_count += 1

def main(args=None):
    rclpy.init(args=args)
    node = ByteTrackNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt detected, shutting down...')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
