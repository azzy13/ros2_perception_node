import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import os.path as osp
import sys
import torch
sys.path.append("/ros2_ws/src/perception/perception/vehicle_perception")

import yolox
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

import argparse
def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        #"--path", default="./demos/output", help="path to images or video"
        "--path", default="./videos/car-detection.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.7, type=float, help="test conf")
    parser.add_argument("--nms", default=0.8, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=10, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.7, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=300, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=4,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

class Perception(Node):
    def __init__(self, output_dir, args, topic_name='/Sim/SceneDroneSensors/robots/Drone1/sensors/front_center1/scene_camera/image'):
        super().__init__('perception')
        self.subscription = self.create_subscription(Image, topic_name, self.listener_callback, 10)
        self.bridge = CvBridge()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.exp = get_exp("/byteTrack_data/exps/example/mot/carla_drone.py", None)
        self.device = "cuda:0"
        self.model = self.exp.get_model().to(self.device)
        self.ckpt = torch.load("/byteTrack_data/pretrained/ground.pth.tar", map_location="cpu")
        self.model.load_state_dict(self.ckpt["model"])
        self.model = fuse_model(self.model)
        self.model = self.model.half()
        self.model.eval()
        self.predictor = Predictor(self.model, self.exp, None, self.device, True)
        self.bytetracker_args = AttrDict({
            "track_thresh": 0.7,  # tracking confidence threshold
            "track_buffer": 30,  # the frames for keep lost tracks
            "match_thresh": 0.8,  # matching threshold for tracking
            "aspect_ratio_thresh": 4,  # filter out boxes of which aspect ratio are above the given value
            "min_box_area": 10,  # filter out tiny boxes,
            "mot20": False
        })
        self.tracker = BYTETracker(self.bytetracker_args, frame_rate=30)
        self.timer = Timer()
        self.args = args
        self.frame_id = 0

    def listener_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Conversion failed: {e}")
            return
        
        outputs, img_info = self.predictor.inference(cv_image, self.timer)
        results = []
        online_im = None
        if outputs[0] is not None:
            online_targets = self.tracker.update(outputs[0], [img_info['height'], img_info['width']], self.exp.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > self.bytetracker_args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > self.bytetracker_args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    results.append(
                        f"{self.frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            self.timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=self.frame_id + 1, fps=1. / self.timer.average_time
            )
        else:
            self.timer.toc()
            online_im = img_info['raw_img']
        # Save the image to disk
        image_path = os.path.join(self.output_dir, f"image_{self.frame_id:04d}.png")
        cv2.imwrite(image_path, online_im)
        self.get_logger().info(f"Saved {image_path}")
        self.frame_id += 1

def main(args=None):
    rclpy.init(args=None)
    output_dir = '/output'
    node = Perception(output_dir, args=args)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    args = make_parser().parse_args()
    print("HALP ME", args)
    main(args=args)
