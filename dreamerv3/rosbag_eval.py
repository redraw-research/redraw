import sys
import rosbag
from sensor_msgs.msg import CompressedImage, Image, Joy
import cv2
import os
import time

import numpy as np
from cv_bridge import CvBridge

sys.path.append("/home/author1/duckiebot_catkin_ws/devel/lib/python3/dist-packages/")
from duckietown_msgs.msg import Twist2DStamped
from dreamerv3.undistort_ros_images import CameraUndistorter
from dbc.remote_control import RemoteControlEnv
import matplotlib.pyplot as plt
from dreamerv3.train import make_replay
from dreamerv3 import embodied
import dreamerv3.agent as agt
from tqdm import tqdm
import math
from geometry_msgs.msg import TransformStamped
import tf2_ros
import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from dreamerv3.embodied.envs.duckiebots_sim import DreamerUELaneFollowingEnv

IMAGE_RAW = "/duckiebot5/camera_node/image_raw"
# IMAGE_RECT = "/duckiebot5/camera_node/image_rect_color"
VEL = "/duckiebot5/kinematics_node/velocity"
ACTION = "/duckiebot5/mdp_action"
duckiebot_frame = 'duckiebot'
track_origin_frame = 'duckiebots_track_origin'
CAMERA_CALIBRATION_YAML_PATH = "/home/author1/git/simdreamer/dreamerv3/dreamerv3/duckiebot5.yaml"


def list_bag_files(directory):
    # Get a list of all files in the directory
    all_files = os.listdir(directory)
    # Filter to include only files that end with '.bag' and do not contain '.tmp'
    bag_files = [
        os.path.join(directory, f) for f in all_files
        if f.endswith('.bag') and '.tmp' not in f
    ]
    return bag_files


def offline_tf_buffer_from_bag(bag):
    tf_buffer = tf2_ros.Buffer(debug=False)

    # Read all /tf and /tf_static messages from the bag
    for topic, msg, t in bag.read_messages(topics=['/tf', '/tf_static']):
        if topic == '/tf_static':
            # /tf_static -> static transforms
            for transform in msg.transforms:
                tf_buffer.set_transform_static(transform, "bag_static_authority")
        else:  # /tf -> dynamic transforms
            for transform in msg.transforms:
                tf_buffer.set_transform(transform, "bag_authority")

    return tf_buffer


def display_images_and_wait(image_pairs):
    """
    :param image_pairs: an iterable of tuples (ros_image_msg, np_opencv_img)
    """
    bridge = CvBridge()

    for idx, (ros_img_msg, np_img) in enumerate(image_pairs):
        # Convert ROS Image to OpenCV format
        ros_cv_img = bridge.imgmsg_to_cv2(ros_img_msg, desired_encoding='bgr8')

        # Show each image in separate windows (or combine them if you wish)
        cv2.imshow('ROS Image Msg (converted)', ros_cv_img)
        cv2.imshow('OpenCV NumPy Image', np_img)

        print(f"Showing pair {idx}. Press any key to move to next pair. Press 'q' to quit.")
        key = cv2.waitKey(0)  # 0 means "wait indefinitely for a keypress"

        if key == ord('q') or key == 27:  # 'q' or ESC
            print("Quitting...")
            break

    # Cleanup
    cv2.destroyAllWindows()


# latest_image_raw = None
# latest_image_rect = None
# latest_vel = None
# latest_act = None
#
# rect_time_deltas = []
#
# time_deltas = []
# initial_t = None
# for topic, msg, t in bag.read_messages(topics=[IMAGE_RAW, IMAGE_RECT, VEL, ACTION]):
#     if initial_t is None:
#         initial_t = t
#     if topic == IMAGE_RAW:
#         latest_image_raw = (msg, t)
#         print(f"image raw {t-initial_t}")
#     elif topic == IMAGE_RECT:
#         latest_image_rect = (msg, t)
#
#         # raw_time = latest_image_raw[1].stamp.to_sec()
#         # rect_time = latest_image_rect[1].stamp.to_sec()
#         # rect_delta = rect_time - raw_time
#         # rect_time_deltas.append(rect_delta
#         #                         )
#         # print(f"image rect {t-initial_t}")
#     elif topic == VEL:
#         latest_vel = (msg, t)
#     elif topic == ACTION:
#         if latest_image_rect is not None and latest_vel is not None:
#             image_time = latest_image_raw[0].header.stamp.to_sec()
#             action_time = msg.header.stamp.to_sec()
#             delta = action_time - image_time
#             print(f"Image is {delta} seconds older than action.")
#             time_deltas.append(delta)
#         else:
#             print(f"{ACTION} message at {t.to_sec()} doesn't have both prior {IMAGE_RECT} and {VEL} messages.")
#         latest_image_rect = (msg, t)


def eval_duckiebots_stats_for_rosbag(rosbag_file_path: str, env):
    bag = rosbag.Bag(f=rosbag_file_path)
    camera_undistorter = CameraUndistorter(yaml_file_path=CAMERA_CALIBRATION_YAML_PATH)

    tf_buffer = offline_tf_buffer_from_bag(bag)
    # Create a CvBridge to convert ROS Image messages to OpenCV images
    bridge = CvBridge()

    sim_reference_env = env
    sim_reference_env.step(action={'reset': True, 'action': sim_reference_env.act_space['action'].sample()})


    latest_img_msg = None
    for topic, msg, msg_t in tqdm(bag.read_messages(topics=[ACTION, IMAGE_RAW]),
                                      total=bag.get_message_count(topic_filters=[ACTION, IMAGE_RAW])):
        if topic == IMAGE_RAW:
            latest_img_msg = msg
            continue

        # Convert the bag time to rospy.Time if needed
        # (Often they're already in compatible format, but just in case:)
        timestamp = rospy.Time(msg_t.secs, msg_t.nsecs)

        try:
            # 3. Lookup transform at the correct time
            # transform_stamped = tf_buffer.lookup_transform(
            #     target_frame=track_origin_frame,  # where we want to transform into
            #     source_frame=duckiebot_frame,  # the frame we have
            #     time=timestamp,
            #     timeout=rospy.Duration(1.0)
            # )

            transform_stamped = tf_buffer.lookup_transform_core(target_frame=track_origin_frame,
                                                                source_frame=duckiebot_frame,
                                                                time=timestamp)
            # Do something with transform_stamped
            # transform_stamped.transform has the translation + rotation (geometry_msgs/Transform)
            # Example:
            translation = transform_stamped.transform.translation
            rotation = transform_stamped.transform.rotation

            x = translation.x
            y = translation.y
            z = translation.z

            quat = transform_stamped.transform.rotation
            # Convert Quaternion -> (roll, pitch, yaw) in radians
            roll, pitch, yaw = euler_from_quaternion([
                quat.x,
                quat.y,
                quat.z,
                quat.w
            ])
            # Convert yaw from radians to degrees
            yaw_degrees = math.degrees(yaw)

            xy_yaw = np.asarray([x * -100.0, y * 100.0, -yaw_degrees], dtype=np.float32)

            # ResetX, ResetY, ResetYaw, ResetForwardVelocity, ResetYawVelocity
            sim_obs = sim_reference_env.step(action={
                'action': np.asarray(msg.axes, np.float32),
                'reset': False,
                'reset_state': (xy_yaw[0], xy_yaw[1], xy_yaw[2], 0, 0)
            })
            # sim_reference_env.render()

            if latest_img_msg:
                cv_image = bridge.imgmsg_to_cv2(latest_img_msg, desired_encoding='bgr8')
                # Display the image
                cv2.imshow("Image", cv_image)
                # Wait for a brief moment to allow the image to render
                # and check for a user keypress.
                # Press 'Esc' (ASCII 27) to break early, for example.
                if cv2.waitKey(1) & 0xFF == 27:  # 27 == 'Esc'
                    break

            print(sim_reference_env.get_duckiebot_metrics_info())
            time.sleep(0.05)

                # normalized_xy_yaw = UEDuckiebotsPositionAndYawObservationFunction.normalize_xy_yaw(xy_yaw=xy_yaw)
                # ...
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            print(f"TF lookup failed at time {timestamp.to_sec()}: {e}")
            # Handle error (e.g., skip this message or use fallback)

    bag.close()


if __name__ == '__main__':
    logdir = "/home/author1/ava_simdreamer/duckiebots_renovated_oct_12/paper_runs/zonly_p2e/transfer_1/deploy_checkpoint_400000_20250120-022823/"

    bag_file_paths = list_bag_files(directory=logdir)

    env = DreamerUELaneFollowingEnv(None,
                              use_domain_randomization=False,
                              randomize_camera_location_for_tilted_robot=False,
                              use_wheel_bias=False,
                              reverse_actions=False,
                              use_mask=False,
                              render_game_on_screen=True
                              )
    while True:
        for i, bag_path in enumerate(bag_file_paths):

            print(f"\nparsing bag {os.path.basename(bag_path)}, {i + 1}/{len(bag_file_paths)}:")
            eval_duckiebots_stats_for_rosbag(rosbag_file_path=bag_path, env=env)

    cv2.destroyAllWindows()
    print("Done.")
