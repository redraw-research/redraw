import sys
import rosbag
from sensor_msgs.msg import CompressedImage, Image, Joy
import cv2
import os

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

IMAGE_RAW = "/duckiebot5/camera_node/image_raw"
# IMAGE_RECT = "/duckiebot5/camera_node/image_rect_color"
VEL = "/duckiebot5/kinematics_node/velocity"
ACTION = "/duckiebot5/mdp_action"

CAMERA_CALIBRATION_YAML_PATH = "/home/author1/git/simdreamer/dreamerv3/dreamerv3/duckiebot5.yaml"



def list_bag_files(directory):
    # Get a list of all files in the directory
    all_files = os.listdir(directory)
    # Filter to include only files that end with '.bag' and do not contain '.tmp'
    bag_files = [
        directory / f for f in all_files
        if f.endswith('.bag') and '.tmp' not in f
    ]
    return bag_files



def format_vel_obs_to_match_env(vel_msg, use_alt_action_processing=False):
    latest_velocities: Twist2DStamped = vel_msg

    max_abs_forward_vel = RemoteControlEnv.ALT_MAX_ABS_FORWARD_VEL if use_alt_action_processing else RemoteControlEnv.DEFAULT_MAX_ABS_FORWARD_VEL
    normalized_forward_vel = np.clip(latest_velocities.v,
                                     a_min=-max_abs_forward_vel,
                                     a_max=max_abs_forward_vel) / max_abs_forward_vel
    normalized_rot_yaw_vel = np.clip(latest_velocities.omega,
                                     a_min=-RemoteControlEnv.MAX_ABS_YAW_VEL,
                                     a_max=RemoteControlEnv.MAX_ABS_YAW_VEL) / RemoteControlEnv.MAX_ABS_YAW_VEL
    normalized_rot_yaw_vel = -normalized_rot_yaw_vel

    return np.asarray([normalized_rot_yaw_vel, normalized_forward_vel], dtype=np.float32)


def format_image_obs_to_match_env(bgr_whc_obs: np.ndarray):
    # Resize image
    bgr_whc_obs = cv2.resize(bgr_whc_obs,
                             (64, 64),
                             interpolation=cv2.INTER_AREA)
    # BGR to RGB
    rgb_whc_obs = bgr_whc_obs[:, :, ::-1]

    return rgb_whc_obs


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


def add_rosbag_to_replay_buffer(rosbag_file_path: str, replay_buffer, graph_statistics=False):
    bag = rosbag.Bag(f=rosbag_file_path)
    camera_undistorter = CameraUndistorter(yaml_file_path=CAMERA_CALIBRATION_YAML_PATH)

    # Find the closest image and vel to every action
    action_vel_img_raw_undistorted_npy_img_tuples = []
    act_img_time_deltas = []
    act_vel_time_deltas = []

    for _, act_msg, act_t in tqdm(bag.read_messages(topics=[ACTION]), total=bag.get_message_count(topic_filters=[ACTION])):
        action_stamp = act_msg.header.stamp.to_sec()

        # Find the closest image to the action
        closest_img = None
        min_delta_t_between_an_image_and_the_action = None
        for topic, img_msg, img_t in bag.read_messages(topics=[IMAGE_RAW]):
            img_stamp = img_msg.header.stamp.to_sec()
            delta = (action_stamp - img_stamp)
            if not closest_img or abs(delta) < min_delta_t_between_an_image_and_the_action:
                closest_img = img_msg
                min_delta_t_between_an_image_and_the_action = delta
        # print(f"action - image delta: {min_delta_t_between_an_image_and_the_action}")
        act_img_time_deltas.append(min_delta_t_between_an_image_and_the_action)

        # Find the closest vel before the action
        closest_vel = None
        min_delta_t_between_a_vel_and_the_action = None
        for topic, vel_msg, vel_t in bag.read_messages(topics=[VEL]):
            vel_stamp = vel_msg.header.stamp.to_sec()
            delta = (action_stamp - vel_stamp)
            if not closest_vel or abs(delta) < min_delta_t_between_a_vel_and_the_action:
                closest_vel = vel_msg
                min_delta_t_between_a_vel_and_the_action = delta
        # print(f"action - vel delta: {min_delta_t_between_a_vel_and_the_action}")
        act_vel_time_deltas.append(min_delta_t_between_a_vel_and_the_action)

        undistorted_img = camera_undistorter.undistort_ros_image(ros_image_msg=closest_img)
        action_vel_img_raw_undistorted_npy_img_tuples.append((act_msg, closest_vel, closest_img, undistorted_img))
    bag.close()

    assert len(action_vel_img_raw_undistorted_npy_img_tuples) > 0

    # Sort frames into contiguous sequences.
    # Continuity is broken if a frame is dropped because the obs and action are far apart
    # or if two frames are a weird distance in seconds from each other.
    max_acceptable_seconds_between_timesteps = 0.12
    min_acceptable_seconds_between_timesteps = 0.08
    max_acceptable_seconds_between_obs_and_actions = 0.022
    min_acceptable_sequence_length = 2
    assert min_acceptable_seconds_between_timesteps < max_acceptable_seconds_between_timesteps

    contiguous_sequences = []
    current_sequence = []
    between_act_steps_time_deltas = []
    last_act = None

    frame_drop_counter = 0
    for act_msg, vel_msg, img_msg, undistorted_img in action_vel_img_raw_undistorted_npy_img_tuples:
        end_sequence = False
        drop_frame = False

        delta_between_act_and_img = abs(act_msg.header.stamp.to_sec() - img_msg.header.stamp.to_sec())
        if delta_between_act_and_img > max_acceptable_seconds_between_obs_and_actions:
            print(f"dropping frame due to action-image time diff: {delta_between_act_and_img} seconds")
            end_sequence = True
            drop_frame = True

        delta_between_act_and_vel = abs(act_msg.header.stamp.to_sec() - vel_msg.header.stamp.to_sec())
        if delta_between_act_and_vel > max_acceptable_seconds_between_obs_and_actions:
            print(f"dropping frame due to action-vel time diff: {delta_between_act_and_vel} seconds")
            end_sequence = True
            drop_frame = True

        if last_act:
            delta_between_timesteps = act_msg.header.stamp.to_sec() - last_act.header.stamp.to_sec()
            between_act_steps_time_deltas.append(delta_between_timesteps)
            if (delta_between_timesteps > max_acceptable_seconds_between_timesteps or
                delta_between_timesteps < min_acceptable_seconds_between_timesteps):
                print(f"splitting sequences due to timestep diff: {delta_between_timesteps} seconds")
                end_sequence = True

        if end_sequence:
            if len(current_sequence) >= min_acceptable_sequence_length:
                contiguous_sequences.append(current_sequence)
            current_sequence = []
            if drop_frame:
                frame_drop_counter += 1
            else:
                current_sequence.append((act_msg, vel_msg, img_msg, undistorted_img))
        else:
            current_sequence.append((act_msg, vel_msg, img_msg, undistorted_img))

        last_act = act_msg
    contiguous_sequences.append(current_sequence)

    assert len(contiguous_sequences) > 0
    print(f"input frames: {len(action_vel_img_raw_undistorted_npy_img_tuples)}")
    print(f"number of contiguous_sequences: {len(contiguous_sequences)}")
    print(f"frame drops: {frame_drop_counter}")
    print(f"out frames: {sum([len(seq) for seq in contiguous_sequences])}")

    # Convert sequences into Dreamer replay buffer format
    dreamer_step_sequences = []
    for ros_sequence in contiguous_sequences:
        dreamer_step_sequence = []
        for i, (act_msg, vel_msg, img_msg, undistorted_img) in enumerate(ros_sequence):
            env_img_obs = format_image_obs_to_match_env(bgr_whc_obs=undistorted_img)
            env_vel_obs = format_vel_obs_to_match_env(vel_msg=vel_msg, use_alt_action_processing=False)
            env_act = np.asarray(act_msg.axes, np.float32)

            assert not (i > len(ros_sequence) - 1)
            obs = {
                "action": env_act,
                "image": env_img_obs,
                "yaw_and_forward_vel": env_vel_obs,
                "position_and_yaw": np.zeros(shape=(3,), dtype=np.float32),
                "is_real": True,
                "reward": 0.0,
                'is_first': i == 0,
                'is_last': i == (len(ros_sequence) - 1),
                'is_terminal': False,
            }
            replay_buffer.add(obs)
            dreamer_step_sequence.append(obs)
        dreamer_step_sequences.append(dreamer_step_sequence)

    if graph_statistics:
        fig, axs = plt.subplots(3, 2, figsize=(10, 8))  # 2x2 grid of subplots
        plt.hist
        # First subplot
        axs[0, 0].hist(act_img_time_deltas, bins='auto', edgecolor='black')
        axs[0, 0].set_xlabel('Value')
        axs[0, 0].set_ylabel('Frequency')
        axs[0, 0].set_title('act_img_time_deltas')

        # Second subplot
        axs[0, 1].hist(act_vel_time_deltas, bins='auto', edgecolor='black')
        axs[0, 1].set_xlabel('Value')
        axs[0, 1].set_ylabel('Frequency')
        axs[0, 1].set_title('act_vel_time_deltas')

        # Third subplot
        axs[1, 0].hist(between_act_steps_time_deltas, bins='auto', edgecolor='black')
        axs[1, 0].set_xlabel('Value')
        axs[1, 0].set_ylabel('Frequency')
        axs[1, 0].set_title('between_act_steps_time_deltas')

        # Fourth subplot
        axs[1, 1].hist([len(seq) for seq in contiguous_sequences], bins=max(len(seq) for seq in contiguous_sequences),
                       edgecolor='black')
        axs[1, 1].set_xlabel('Value')
        axs[1, 1].set_ylabel('Frequency')
        axs[1, 1].set_title('contiguous sequence lengths raw')

        # Fifth subplot
        axs[2, 0].hist([len(seq) for seq in dreamer_step_sequences], bins=max(len(seq) for seq in dreamer_step_sequences),
                       edgecolor='black')
        axs[2, 0].set_xlabel('Value')
        axs[2, 0].set_ylabel('Frequency')
        axs[2, 0].set_title('dreamer sequence lengths')

        # Sixth subplot
        seq_lengths_by_timestep = [len(seq) for seq in dreamer_step_sequences for _ in range(len(seq))]
        axs[2, 1].hist(seq_lengths_by_timestep, bins=max(len(seq) for seq in dreamer_step_sequences), edgecolor='black')
        axs[2, 1].set_xlabel('Value')
        axs[2, 1].set_ylabel('Frequency')
        axs[2, 1].set_title('Timesteps grouped into dreamer sequence lengths')

        plt.tight_layout()
        # plt.show()
        plt.savefig(str(bag_path).replace(".bag", "_stats.png"))




if __name__ == '__main__':
    # Parse configs
    parsed, other = embodied.Flags(configs=['defaults']).parse_known(None)
    config = embodied.Config(agt.Agent.configs['defaults'])
    for name in parsed.configs:
        config = config.update(agt.Agent.configs[name])
        config = config.update({
            "save_replay": True
        })
    config = embodied.Flags(config).parse(other)

    if not config.logdir:
        raise ValueError(f"Must specify logdir")
    logdir = embodied.Path(config.logdir)

    replay_out_dir = logdir / 'replay_cleaned'
    if os.path.isdir(replay_out_dir) and os.listdir(replay_out_dir):
        raise RuntimeError(f"replay out directory {replay_out_dir} is not empty.")
    replay = make_replay(config, directory=replay_out_dir)
    bag_file_paths = list_bag_files(directory=logdir)

    for i, bag_path in enumerate(bag_file_paths):
        print(f"\nparsing bag {bag_path.name}, {i+1}/{len(bag_file_paths)}:")
        add_rosbag_to_replay_buffer(rosbag_file_path=bag_path, replay_buffer=replay, graph_statistics=True)

    replay.save(wait=True)
    print("Done.")

