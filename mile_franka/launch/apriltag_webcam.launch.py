# launch/apriltag_webcam.launch.py
"""Bring up a USB/UVC webcam (e.g. Logitech), apriltag_ros, and the camera->base static tf.

Drop-in alternative to apriltag_realsense.launch.py for cameras without a RealSense SDK.
The webcam must be calibrated first (ros2 run camera_calibration cameracalibrator) and the
resulting YAML passed via MILE_CAMERA_CALIB or placed at config/camera_calib.yaml.

Args:
    device:      V4L2 device node (default /dev/video0).
    width:       capture width in pixels (default 1920).
    height:      capture height in pixels (default 1080).
    framerate:   frames per second (default 15).
    camera_info: URL for the camera intrinsics YAML (default: MILE_CAMERA_CALIB or
                 config/camera_calib.yaml).  Must be a file:// URL or package:// URL.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from mile_franka.pose.calibration import load_camera_calibration

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB = (os.environ.get("MILE_CAMERA_CALIB")
         or os.path.join(REPO, "config", "camera_calib.yaml"))
APRILTAG_CFG = os.path.join(REPO, "config", "apriltag.yaml")

# usb_cam publishes on <node_name>/image_raw and <node_name>/camera_info.
# The node is named "camera" below, so topics are /camera/image_raw and /camera/camera_info.
CAMERA_IMAGE_TOPIC = "/camera/image_raw"
CAMERA_INFO_TOPIC  = "/camera/camera_info"
# Must match the frame_id stamped by usb_cam (node name + "_optical_frame" by default).
CAMERA_FRAME = "camera_optical_frame"


def _base_frame() -> str:
    real_stack = os.environ.get("MILE_REAL_STACK", "multipanda").lower()
    if real_stack in ("fr3", "franka_ros2", "fr3_pose"):
        return "fr3_link0"
    return "panda_link0"


def generate_launch_description():
    device     = LaunchConfiguration("device")
    width      = LaunchConfiguration("width")
    height     = LaunchConfiguration("height")
    framerate  = LaunchConfiguration("framerate")
    camera_url = LaunchConfiguration("camera_info")

    nodes = [
        DeclareLaunchArgument("device",      default_value="/dev/video0"),
        DeclareLaunchArgument("width",       default_value="1920"),
        DeclareLaunchArgument("height",      default_value="1080"),
        DeclareLaunchArgument("framerate",   default_value="15.0"),
        DeclareLaunchArgument("camera_info", default_value=f"file://{CALIB}"),
    ]

    if os.path.exists(CALIB):
        args = load_camera_calibration(CALIB).static_transform_args(_base_frame(), CAMERA_FRAME)
        nodes.append(Node(
            package="tf2_ros", executable="static_transform_publisher",
            name="camera_to_base_static_tf",
            arguments=["--x", args[0], "--y", args[1], "--z", args[2],
                       "--qx", args[3], "--qy", args[4], "--qz", args[5], "--qw", args[6],
                       "--frame-id", args[7], "--child-frame-id", args[8]]))
    else:
        print(f"[apriltag_webcam.launch] {CALIB} not found — skipping camera->base static "
              "transform; only camera->tag tf will be available until calibration is run.")

    nodes.extend([
        Node(package="usb_cam", executable="usb_cam_node_exe", name="camera",
             parameters=[{
                 "video_device":    device,
                 "image_width":     width,
                 "image_height":    height,
                 "framerate":       framerate,
                 "pixel_format":    "mjpeg2rgb",
                 "camera_info_url": camera_url,
                 "camera_frame_id": CAMERA_FRAME,
                 # No SDK-managed TF; our eye-to-hand static tf covers base->camera.
                 "publish_camera_info_msg": True,
             }]),
        Node(package="apriltag_ros", executable="apriltag_node", name="apriltag",
             remappings=[("image_rect", CAMERA_IMAGE_TOPIC),
                         ("camera_info", CAMERA_INFO_TOPIC)],
             parameters=[APRILTAG_CFG]),
    ])

    return LaunchDescription(nodes)
