from dbc.remote_control import RemoteControlEnv
import functools
import time

import gym
import numpy as np
from gym import spaces
import cv2
import os
import csv
from dreamerv3 import embodied
from dreamerv3.embodied.envs.from_gym import FromGym
from duckiebots_unreal_sim.rcan_obs_preprocessor import ONNXRCANObsPreprocessor
from duckiebots_unreal_sim.tools import ImageRenderer
from dreamerv3.embodied.envs.duckiebots_sim import DreamerUELaneFollowingEnv

checked_csv_exist_ok = False

def maybe_append_dict(my_dict):
    """
    Ask the user if they want to append `my_dict` as a row to a CSV file.
    If this is the first call, ask for the CSV file name and create it if it doesn't exist.
    """
    global checked_csv_exist_ok

    # If it's the first time, ask for CSV file name
    csv_file_name = os.path.join("/home/author1/git/simdreamer/dreamerv3/duckiebot_results", f"{os.environ.get('EXP_NAME')}_{os.environ.get('DB_SEED')}.csv")

    # If file doesn't exist, create it with headers
    if not os.path.exists(csv_file_name):
        with open(csv_file_name, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=my_dict.keys())
            writer.writeheader()
    else:
        if not checked_csv_exist_ok:
            choice = input(f"{csv_file_name} already exists? continue? (y)")
            if choice != 'y':
                print('stoppping.')
                exit(0)
            checked_csv_exist_ok = True

    # Ask if user wants to append
    choice = input(f"Would you like to append this dictionary to the {csv_file_name}? (y/n): ").strip().lower()

    if choice == 'y':
        with open(csv_file_name, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=my_dict.keys())
            writer.writerow(my_dict)
            print(f"Appended to {csv_file_name}")
        return True
    else:
        print("Skipping append.")
        return False


def average_dict_values(dict_list):
    """
    Given a list of dictionaries with identical keys and numeric values,
    return a dictionary with the average value for each key.
    Also prints the result in a human-readable format.
    """
    # Handle edge cases: if the list is empty, return empty dict
    if not dict_list:
        print("No dictionaries provided. Returning empty dictionary.")
        return {}

    # We'll assume all dictionaries have the same keys
    keys = dict_list[0].keys()
    avg_dict = {}

    # Compute sums for each key
    for key in keys:
        if not any(d[key] == None for d in dict_list):
            total = sum(d[key] for d in dict_list)
            avg_dict[key] = total / len(dict_list)
        else:
            avg_dict[key] = None

    # Print in a nice human-readable format
    print("Averages across all dictionaries:")
    for k, v in avg_dict.items():
        if v is None:
            print(f"  {k}: none")
        else:
            print(f"  {k}: {v:.2f}")

    return avg_dict

class DreamerRealLaneFollowingEnv(FromGym):

    def __init__(self, _, robot_name="duckiebot5", local_ip="192.168.1.9", target_control_hz=10,
                 use_mask=True, rcan_checkpoint_path="/home/author1/Downloads/ckpt_9_nov_17.onnx",
                 use_wheel_bias=False,
                 separately_return_mask_and_rgb=False,
                 reverse_actions=False,
                 measure_rewards_via_sim=True,
                 use_alt_action_processing=False,
                 disable_position_tracking=False
                 ):
        env = RemoteControlEnv(robot_ros_name=robot_name,
                               render_live_camera_feed=False,
                               local_hostname_or_ip=local_ip,
                               target_control_hz=target_control_hz,
                               observation_out_height=64,
                               observation_out_width=64,
                               validate_action_space=False,
                               use_alt_action_processing=use_alt_action_processing,
                               disable_position_tracking=disable_position_tracking)

        if separately_return_mask_and_rgb:
            raise NotImplementedError("need to add this functionality")

        self.use_wheel_bias = use_wheel_bias
        self.reverse_actions = reverse_actions

        self._rcan = None
        if use_mask:
            self._rcan = ONNXRCANObsPreprocessor(checkpoint_path=rcan_checkpoint_path, debug_render_predictions=False)
        self._latest_mask_obs_for_rendering = None
        self._mask_renderer = None

        self._sim_reference_env = None
        self._measure_rewards_via_sim = measure_rewards_via_sim
        if self._measure_rewards_via_sim:
            self._sim_episode_states_actions_and_times = []
            self._all_recorded_result_dicts = []
            self._sim_reference_env = DreamerUELaneFollowingEnv(_,
                                                                use_domain_randomization=False,
                                                                randomize_camera_location_for_tilted_robot=False,
                                                                use_wheel_bias=False,
                                                                use_mask=False,
                                                                render_game_on_screen=True,
                                                                simulate_latency=False,
                                                                physics_hz=100,
                                                                reward_function="penalize_turning_more_100hz"
                                                                )
            self._sim_reference_env.step(action={'reset': True, 'action': self._sim_reference_env.act_space['action'].sample()})

        super().__init__(env=env)

    @functools.cached_property
    def obs_space(self):
        obs_space = super().obs_space
        obs_space["is_real"] = embodied.Space(bool)
        return obs_space


    def step(self, action):

        if action['reset'] and 'reset_state' in action:
            raise NotImplementedError()

        control = np.asarray(action['action'])
        if self.reverse_actions:
            control = -control

        if self.use_wheel_bias:
            assert False
            orig_yaw_action = control[1]
            if orig_yaw_action > 0.9:
                new_yaw_action = ((orig_yaw_action - 0.9) / 2.0) + 0.9
            else:
                new_yaw_action = orig_yaw_action - 0.3
                if orig_yaw_action < 0.0:
                    new_yaw_action *= 2.0

            control[1] = new_yaw_action
        action['action'] = control

        if action['reset']:
            action['action'] = np.zeros_like(action['action'])
            if self._measure_rewards_via_sim:
                self.replay_trajectory_in_sim_to_get_metrics(self._sim_episode_states_actions_and_times)
                self._sim_episode_states_actions_and_times = []

        if self._measure_rewards_via_sim:
            sim_reference_state = self._env.get_reference_state_for_simulation()
            state_time = self._env.get_rospy_seconds()
            self._sim_episode_states_actions_and_times.append((sim_reference_state, action['action'], state_time))

        obs = super().step(action)

        if self._rcan and "image" in obs:
            # mask_obs = obs['image']
            mask_obs = self._rcan.preprocess_obs(rgb_obs=obs['image'])
            self._latest_mask_obs_for_rendering = cv2.resize(mask_obs, dsize=(512, 512))
            obs['image'] = mask_obs
            print(f"obs shape is {obs['image'].shape}")
        self.render()

        obs['is_real'] = True

        # if self._sim_reference_env:
        #     reset_sim_to_state = self._env.get_reference_state_for_simulation()
        #     sim_obs = self._sim_reference_env.step(action={
        #         'action': action['action'],
        #         'reset': False,
        #         'reset_state': reset_sim_to_state
        #     })
        #     self._sim_reference_env.render()
        #     obs['reward'] = sim_obs['reward']

        return obs

    def replay_trajectory_in_sim_to_get_metrics(self, sim_episode_states_and_actions_and_times):
        if len(sim_episode_states_and_actions_and_times) == 0:
            print("not simulating episode of length 0")
            return
        episode_velocities = []
        velocities_and_actions = []
        episode_rewards = []
        lap_rewards = []
        episode_offsets = []
        lap_offsets = []
        starting_progress_along_track = None
        starting_time = None
        lap_time = None
        invalid_episode_result = False
        completed_lap = False
        out_of_bounds = False
        has_been_before_finish_line = False
        lap_timesteps = None
        for i, (state, action, time) in enumerate(sim_episode_states_and_actions_and_times):
            sim_obs = self._sim_reference_env.step(action={
                'action': action,
                'reset': False,
                'reset_state': state
            })
            self._sim_reference_env.render()
            reward = sim_obs['reward']
            out_of_bounds = reward < -50.0
            metrics_info = self._sim_reference_env.get_duckiebot_metrics_info()
            distance_from_path = metrics_info['distance_cm_from_path_center']
            progress_along_track = metrics_info['progress_along_intended_path']
            velocity_along_track = metrics_info['velocity_along_intended_path']
            episode_velocities.append(velocity_along_track)
            velocities_and_actions.append((velocity_along_track, action))
            episode_rewards.append(reward)
            episode_offsets.append(distance_from_path)
            if not completed_lap:
                lap_rewards.append(reward)
                lap_offsets.append(distance_from_path)
            if out_of_bounds:
                print(f"OUT OF BOUNDS")
                break
            if distance_from_path > 50.0:
                print(f"very high distance from path {distance_from_path}, dropping lap and resume when the starting line is crossed")
                invalid_episode_result = True
                break
            if i == 0:
                starting_progress_along_track = progress_along_track
                starting_time = time

            if i > 100 and progress_along_track < starting_progress_along_track:
                has_been_before_finish_line = True

            if has_been_before_finish_line and progress_along_track > starting_progress_along_track and not completed_lap:
                completed_lap = True
                lap_time = time - starting_time
                lap_timesteps = i

        print(f"i: {i}")
        total_episode_rewards = sum(episode_rewards)
        total_lap_rewards = sum(lap_rewards)
        avg_episode_reward = np.mean(episode_rewards)
        avg_lap_reward = np.mean(lap_rewards)
        avg_episode_offset = np.mean(episode_offsets)
        avg_lap_offset = np.mean(lap_offsets)
        avg_episode_velocity_along_path = np.mean(episode_velocities)
        # print(velocities_and_actions)

        result_dict = {
            "run_start_time": starting_time,
            "lap_time": lap_time,
            "total_episode_rewards": total_episode_rewards,
            "total_lap_rewards": total_lap_rewards,
            "avg_episode_reward": avg_episode_reward,
            "avg_lap_reward": avg_lap_reward,
            "avg_episode_offset": avg_episode_offset,
            "avg_lap_offset": avg_lap_offset,
            "out_of_bounds": out_of_bounds,
            "completed_lap": completed_lap,
            "lap_timesteps": lap_timesteps,
            "avg_episode_velocity_along_path": avg_episode_velocity_along_path,
            "lap_rew_max": np.max(lap_rewards),
            "lap_rew_min": np.min(lap_rewards),
            "episode_rew_max": np.max(episode_rewards),
            "episode_rew_min": np.min(episode_rewards),
        }
        if not invalid_episode_result:
            for k, v in result_dict.items():
                if v is None:
                    print(f"  {k}: none")
                else:
                    print(f"  {k}: {v:.2f}")

            print(f"Adding this episode will result in {len(self._all_recorded_result_dicts) +1} laps")
            print(f"avg vals:")
            average_dict_values([*self._all_recorded_result_dicts, result_dict])

        if invalid_episode_result:
            print(f"\n\n\nPOTENTIALLY INVALID EPISODE RESULT!!!!!!!1\n\n\n")
        if maybe_append_dict(result_dict):
            self._all_recorded_result_dicts.append(result_dict)

        laps_to_record = 5
        if len(self._all_recorded_result_dicts) >= laps_to_record:
            print(f"{laps_to_record} laps recorded. Stopping")
            exit(0)

    def render(self):
        self._env.render()
        if self._latest_mask_obs_for_rendering is not None:
            if not self._mask_renderer:
                height = self._latest_mask_obs_for_rendering.shape[0]
                width = self._latest_mask_obs_for_rendering.shape[1]
                self._mask_renderer = ImageRenderer(height=height, width=width)
            self._mask_renderer.render_cv2_image(self._latest_mask_obs_for_rendering)


if __name__ == '__main__':
    from duckiebots_unreal_sim.tools import XboxController, get_keyboard_turning, get_keyboard_velocity

    print("Detecting gamepad (you may have to press a button on the controller)...")
    gamepad = None
    if XboxController.detect_gamepad():
        gamepad = XboxController()
    print("Gamepad found" if gamepad else "use keyboard controls")

    # env = UELaneFollowingEnv({
    #     "use_domain_randomization": True,
    #     "render_game_on_screen": True,
    #     "use_mask_observation": False,
    #     "return_rgb_and_mask_as_observation": False,z
    #     "preprocess_rgb_observations_with_rcan": False,
    #     "rcan_checkpoint_path": "/home/author1/Downloads/ckpt-91.onnx",
    #     "launch_game_process": True,
    #     "simulate_latency": True,
    # })

    # Recommended config for initially trying out this environment with Dreamer
    env = DreamerRealLaneFollowingEnv(None, use_mask=False, reverse_actions=False, measure_rewards_via_sim=True, use_alt_action_processing=True,
                                      disable_position_tracking=False,
                                      local_ip="192.168.1.4")

    print("env initialized")
    while True:
        i = 0
        env.step({"reset": True, "action": [0.0, 0.0]})
        env.render()

        done = False

        # target_frame_time = 1.0/30.0
        target_frame_time = 0
        prev_frame_time = time.time()
        while not done:
            velocity = gamepad.LeftJoystickY if gamepad else get_keyboard_velocity()
            turning = gamepad.LeftJoystickX if gamepad else get_keyboard_turning()
            action = np.asarray([velocity, turning], np.float32)

            obs = env.step(action={"action": action, "reset": False})

            # print(obs["image"])
            # print(obs["yaw_and_forward_vel"])

            # print(f"timestep {i}, rew {obs['reward']}, yaw_and_forward_vel: {obs['yaw_and_forward_vel']} sim sensor state: {env._env.base_env._latest_observation_state}")
            assert obs['reward'] <= 1.0, obs['reward']
            print(f"position_and_yaw: {obs['position_and_yaw']} ")

            current_delta = time.time() - prev_frame_time
            sleep_delta = max(target_frame_time - current_delta, 0)
            time.sleep(sleep_delta)
            now = time.time()

            # print(f"delta time: {now - prev_frame_time}")
            # print(f"rew: {rew}")

            prev_frame_time = now

            env.render()

            if gamepad and gamepad.B:
                print("randomizing")
                # env.reset()
                env.base_env.randomize()

            i += 1
