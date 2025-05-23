import yaml
import numpy as np
import cv2

from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class CameraUndistorter:
    """
    A utility class that loads camera intrinsic parameters from a ROS-style YAML file,
    precomputes undistortion maps, and provides a method to undistort 'sensor_msgs.msg.Image'
    messages into OpenCV images.
    """

    def __init__(self, yaml_file_path):
        """
        :param yaml_file_path: Path to the YAML file containing camera intrinsics.
        """
        self.bridge = CvBridge()

        # --- Load calibration data from YAML ---
        with open(yaml_file_path, 'r') as f:
            calib_data = yaml.safe_load(f)

        # Distortion model: 'equidistant' (fisheye) or 'plumb_bob' (pinhole)
        self.dist_model = calib_data["distortion_model"]

        # Intrinsic parameters
        self.image_width = calib_data["image_width"]
        self.image_height = calib_data["image_height"]
        self.camera_matrix = np.array(calib_data["camera_matrix"]["data"]).reshape((3, 3))
        self.dist_coeffs = np.array(calib_data["distortion_coefficients"]["data"]).reshape((1, 4))

        # --- Create undistortion maps once (for efficiency) ---
        self.map1, self.map2 = self.init_undistort_maps()

    def init_undistort_maps(self):
        """
        Initializes and returns the undistortion/rectification maps for remapping images.
        """
        if self.dist_model == "equidistant":
            # For equidistant/fisheye distortion, use OpenCV's fisheye APIs
            new_K = self.camera_matrix.copy()  # Copy of the original camera matrix
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                self.camera_matrix,
                self.dist_coeffs,
                R=np.eye(3),  # Identity for no additional rotation
                P=new_K,  # You can modify focal length/center if desired
                size=(self.image_width, self.image_height),
                m1type=cv2.CV_16SC2
            )
        else:
            raise NotImplementedError(self.dist_model)

        return map1, map2

    def undistort_ros_image(self, ros_image_msg):
        """
        Converts a sensor_msgs.msg.Image into an undistorted OpenCV image (NumPy array).

        :param ros_image_msg: A sensor_msgs.msg.Image (distorted).
        :return: A NumPy array (undistorted image in BGR format).
        """
        # Convert ROS Image to OpenCV (BGR) array
        distorted_cv_image = self.bridge.imgmsg_to_cv2(ros_image_msg, desired_encoding='bgr8')

        # Remap (undistort) the image
        undistorted_cv_image = cv2.remap(distorted_cv_image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

        return undistorted_cv_image
