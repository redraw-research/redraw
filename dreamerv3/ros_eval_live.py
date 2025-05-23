#!/usr/bin/env python3

import sys
import math
import time
import numpy as np
import rospy
import tf2_ros
import cv2
from scipy.stats import sem
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from tf.transformations import euler_from_quaternion
from cv_bridge import CvBridge

# Make sure you have your custom packages in the PYTHONPATH if needed:
sys.path.append("/home/author1/duckiebot_catkin_ws/devel/lib/python3/dist-packages/")

# Import your simulation environment
from dreamerv3.embodied.envs.duckiebots_sim import DreamerUELaneFollowingEnv

# Topics and frames
IMAGE_RAW_TOPIC = "/duckiebot5/camera_node/image_raw"
DUCKIEBOT_FRAME = "duckiebot"
TRACK_ORIGIN_FRAME = "duckiebots_track_origin"

##############################################################################
# Global Variables
##############################################################################

bridge = CvBridge()               # For converting ROS Image -> OpenCV
tf_buffer = None                  # We'll fill this with a tf2_ros.Buffer
env = None                        # Your simulation environment instance
duckiebot_frame = DUCKIEBOT_FRAME
track_origin_frame = TRACK_ORIGIN_FRAME

##############################################################################
# Callbacks
##############################################################################

def image_callback(msg):
    """
    Display the latest camera image on screen.
    Press ESC to shut down the node.
    """
    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    cv2.imshow("Duckiebot View", cv_img)
    key = cv2.waitKey(1)
    if key == 27:  # 27 is ESC
        rospy.signal_shutdown("User requested quit via ESC key.")
        return

##############################################################################
# Main
##############################################################################

def main():
    global tf_buffer, env

    rospy.init_node("live_duckiebot_sim_no_actions", anonymous=True)

    # 1) Create a tf2 buffer + listener to receive live /tf
    tf_buffer = tf2_ros.Buffer(debug=False)
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    # 2) Create your environment
    env = DreamerUELaneFollowingEnv(
        None,
        use_domain_randomization=False,
        randomize_camera_location_for_tilted_robot=False,
        use_wheel_bias=False,
        reverse_actions=False,
        simulate_latency=False,
        use_mask=False,
        render_game_on_screen=True
    )

    # Optionally reset environment to a random start
    env.step(action={'reset': True, 'action': env.act_space['action'].sample()})

    # 3) Subscribe to camera topic only. We'll no longer subscribe to the action topic.
    rospy.Subscriber(IMAGE_RAW_TOPIC, Image, image_callback, queue_size=1)

    rospy.loginfo("[live_duckiebot_sim_no_actions] Subscribed to camera. Now looping for TF data...")

    # Use a Rate to control how often we query TF
    rate = rospy.Rate(10)  # 10 Hz loop

    starting_progress_along_track = None
    lap_starting_time = None
    has_been_before_finish_line = False
    invalid_lap = False
    lap_return = 0.0
    distances_from_path = []
    lap_times = []
    lap_returns = []
    rewards_this_lap = []
    lifetime_rewards = []
    total_laps = 0

    input("Press ENTER to start.")

    try:
        while not rospy.is_shutdown():
            try:
                # 4) Lookup the *latest* transform (time=0) from duckiebot to track origin
                transform_stamped = tf_buffer.lookup_transform(
                    target_frame=track_origin_frame,
                    source_frame=duckiebot_frame,
                    time=rospy.Time(0),              # 0 means "latest available transform"
                    timeout=rospy.Duration(0.1)
                )
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                rospy.logwarn_throttle(5.0, f"TF lookup failed: {e}")
                rate.sleep()
                continue

            # 5) Extract (x, y, yaw)
            translation = transform_stamped.transform.translation
            rotation = transform_stamped.transform.rotation
            x = translation.x
            y = translation.y
            # z = translation.z  # usually unused, but here if needed

            (roll, pitch, yaw) = euler_from_quaternion([
                rotation.x,
                rotation.y,
                rotation.z,
                rotation.w
            ])
            yaw_degrees = math.degrees(yaw)

            # Match your original coordinate scaling + sign conventions:
            # (negate X, convert to cm, invert yaw, etc.)
            xy_yaw = np.asarray([x * -100.0, y * 100.0, -yaw_degrees], dtype=np.float32)

            # 6) Step the environment, forcibly resetting the sim pose to that transform
            #    For the "action", you can pass zero or something minimal each loop.
            sim_obs = env.step(action={
                'action': np.array([0.0, 0.0], dtype=np.float32),  # no driving
                'reset': False,
                'reset_state': (xy_yaw[0], xy_yaw[1], xy_yaw[2], 0.0, 0.0)
            })
            reward = sim_obs['reward']

            # 7) Print environment metrics
            metrics_info = env.get_duckiebot_metrics_info()

            distance_from_path = metrics_info['distance_cm_from_path_center']
            if distance_from_path > 30.0:
                print(f"very high distance from path {distance_from_path}, dropping lap and resume when the starting line is crossed")
                invalid_lap = True

            # Commented this out to just have a live sim feed of duckiebot position. Uncommented for actaul reward lap tracking
            # else:
            #     distances_from_path.append(distance_from_path)
            #     rewards_this_lap.append(reward)
            #     lap_return += reward
            #
            # if starting_progress_along_track is None:
            #     starting_progress_along_track = metrics_info['progress_along_intended_path']
            #     lap_starting_time = rospy.Time.now()
            #     print(f"starting progress: {starting_progress_along_track} time: {lap_starting_time}")
            # if rospy.Time.now().to_sec() - lap_starting_time.to_sec() > 10.0 and metrics_info['progress_along_intended_path'] < starting_progress_along_track:
            #     has_been_before_finish_line = True
            #
            # if has_been_before_finish_line and metrics_info['progress_along_intended_path'] > starting_progress_along_track:
            #     end_time = rospy.Time.now()
            #     lap_time_sec = end_time.to_sec() - lap_starting_time.to_sec()
            #
            #     if not invalid_lap:
            #         lap_times.append(lap_time_sec)
            #         lap_returns.append(lap_return)
            #         lifetime_rewards.extend(rewards_this_lap)
            #         total_laps += 1
            #         print(f"Lap Time: {lap_time_sec} seconds")
            #         print(f"Return: {lap_return}")
            #     else:
            #         print("prev lap was invalid, starting new lap")
            #     lap_return = 0.0
            #     rewards_this_lap = []
            #     lap_starting_time = end_time
            #     has_been_before_finish_line = False
            #     invalid_lap = False
            #
            #     if total_laps >= 10:
            #         break

            rospy.loginfo_throttle(1, f"distance from path: {distance_from_path}")

            # # 8) Sleep to maintain loop rate
            # rate.sleep()
    except KeyboardInterrupt:
        pass

    print(f"Avg lap time: {np.mean(lap_times)} +/- {sem(lap_times)}")
    print(f"Avg distance from path: {np.mean(distances_from_path)} +/- {sem(distances_from_path)}")
    print(f"Avg lap return: {np.mean(lap_returns)} +/- {sem(lap_returns)}")
    print(f"Avg reward: {np.mean(lifetime_rewards)} +/- {sem(lifetime_rewards)}")
    # If we exit the loop normally:
    cv2.destroyAllWindows()
    rospy.loginfo("Shutting down.")

if __name__ == "__main__":
    main()
