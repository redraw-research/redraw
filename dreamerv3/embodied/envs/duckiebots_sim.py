from duckiebots_unreal_sim.holodeck_lane_following_env import UEDuckiebotsHolodeckEnv

import time
import functools
from dataclasses import dataclass

import gym
import numpy as np
from duckiebots_unreal_sim.holodeck_env import UEDuckiebotsHolodeckEnv, DEFAULT_GAME_LAUNCHER_PATH, HolodeckException
from duckiebots_unreal_sim.reward_functions import VelocityAndDistanceAlongTrackRewardAndTerminationFunction, VelocityAndDistanceAlongTrackRewardAndTerminationFunction2, VelocityAndDistanceAlongTrackRewardAndTerminationFunction3, VelocityAndDistanceAlongTrackRewardAndTerminationFunctionStrictDistance, VelocityAndDistanceAlongTrackRewardAndTerminationFunction4, VelocityAndDistanceAlongTrackRewardAndTerminationFunction100Hz, VelocityAndDistanceAlongTrackRewardAndTerminationFunctionAlwaysPenalizeTurning
from gym import spaces
import cv2
from dreamerv3 import embodied

from dreamerv3.embodied.envs.from_gym import FromGym

@dataclass
class EpisodicPhysicsParams:
    forward_mean: float
    forward_std: float
    yaw_mean: float
    yaw_std: float

class ObservationBufferWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, obs_buffer_depth=3):
        super(ObservationBufferWrapper, self).__init__(env)
        obs_space_shape_list = list(self.observation_space.shape)

        # The last dimension, is used. For images, this should be the depth.
        # For vectors, the output is still a vector, just concatenated.
        self.buffer_axis = len(obs_space_shape_list) - 1
        obs_space_shape_list[self.buffer_axis] *= obs_buffer_depth
        # self.observation_space.shape = tuple(obs_space_shape_list)

        if len(self.observation_space.shape) == 3:
            limit_low = self.observation_space.low[0, 0, 0]
            limit_high = self.observation_space.high[0, 0, 0]
        elif len(self.observation_space.shape) == 1:
            # Note this was implemented for vector like observation spaces (e.g. a VAE latent vector)
            limit_low = self.observation_space.low[0]
            limit_high = self.observation_space.high[0]
        else:
            assert False, "Only 1 or 3 dimensional obs space supported!"

        self.observation_space = spaces.Box(
            limit_low,
            limit_high,
            tuple(obs_space_shape_list),
            dtype=self.observation_space.dtype)
        self.obs_buffer_depth = obs_buffer_depth
        self.obs_buffer = None

    def observation(self, obs):
        if self.obs_buffer_depth == 1:
            return obs
        if self.obs_buffer is None:
            self.obs_buffer = np.concatenate([obs for _ in range(self.obs_buffer_depth)], axis=self.buffer_axis,
                                             dtype=self.observation_space.dtype)
        else:
            self.obs_buffer = np.concatenate((self.obs_buffer[..., (obs.shape[self.buffer_axis]):], obs),
                                             axis=self.buffer_axis, dtype=self.observation_space.dtype)
        return self.obs_buffer

    def step(self, action, **kwargs):
        observation, reward, terminated, info = self.env.step(action, **kwargs)
        return self.observation(observation), reward, terminated, info

    def reset(self):
        self.obs_buffer = None
        observation = self.env.reset()
        return self.observation(observation)


class _UELaneFollowingEnv(gym.Env):
    def __init__(self, config: dict = None):
        env_config = {
            "use_domain_randomization": True,
            "randomize_mask": False,
            "render_game_on_screen": False,
            "use_mask_observation": False,
            "return_rgb_and_mask_as_observation": False,
            "use_rcan_instead_of_gt_mask": False,
            "simulate_latency": False,
            "preprocess_rgb_observations_with_rcan": False,
            "rcan_checkpoint_path": None,
            "frame_stack_amount": 1,
            "launch_game_process": True,
            "time_limit": 1000,
            "physics_hz": None,
            "use_simple_physics": False,
            "use_wheel_bias": False,
            "limit_backwards": False,
            "randomize_camera_location_for_tilted_robot": False,
            "game_path_override": None,
            "world_name": "DuckiebotsHolodeckMap",
            "reward_function": "penalize_turning_more",
            "randomize_physics_every_step": False,
            "randomize_physics_every_episode": False,
            "camera_shake_with_random_physics": False,
        }

        env_config.update(config)

        physics_hz = env_config['physics_hz'] or (6*2 if env_config["simulate_latency"] else 6)

        reward_function = {
            "penalize_turning_more": VelocityAndDistanceAlongTrackRewardAndTerminationFunction,
            "penalize_turning_less": VelocityAndDistanceAlongTrackRewardAndTerminationFunction2,
            "no_penalty": VelocityAndDistanceAlongTrackRewardAndTerminationFunction3,
            "strict_distance": VelocityAndDistanceAlongTrackRewardAndTerminationFunctionStrictDistance,
            "penalty4": VelocityAndDistanceAlongTrackRewardAndTerminationFunction4,
            "penalize_turning_more_100hz": VelocityAndDistanceAlongTrackRewardAndTerminationFunction100Hz,
            "always_penalize_turning": VelocityAndDistanceAlongTrackRewardAndTerminationFunctionAlwaysPenalizeTurning
        }[env_config['reward_function']]

        self.base_env = UEDuckiebotsHolodeckEnv(
            randomization_enabled=env_config["use_domain_randomization"],
            physics_hz=physics_hz,
            physics_ticks_between_action_and_observation=1,
            physics_ticks_between_observation_and_action=1 if env_config["simulate_latency"] else 0,
            render_game_on_screen=env_config["render_game_on_screen"],
            return_only_mask_as_observation=env_config["use_mask_observation"],
            return_rgb_and_mask_as_observation=env_config["return_rgb_and_mask_as_observation"],
            use_rcan_instead_of_gt_mask=env_config["use_rcan_instead_of_gt_mask"],
            randomize_mask=env_config["randomize_mask"],
            preprocess_rgb_with_rcan=env_config["preprocess_rgb_observations_with_rcan"],
            rcan_checkpoint_path=env_config["rcan_checkpoint_path"],
            launch_game_process=env_config["launch_game_process"],
            image_obs_out_height=64,
            image_obs_out_width=64,
            use_simple_physics=env_config["use_simple_physics"],
            use_wheel_bias=env_config["use_wheel_bias"],
            limit_backwards_movement=env_config["limit_backwards"],
            randomize_camera_location_for_tilted_robot=env_config["randomize_camera_location_for_tilted_robot"],
            game_path=env_config['game_path_override'],
            world_name=env_config['world_name'],
            reward_function=reward_function

        )

        # Observation Wrappers
        self._env = self.base_env

        if env_config["frame_stack_amount"] > 1:
            self._env = ObservationBufferWrapper(self._env, obs_buffer_depth=env_config["frame_stack_amount"])

        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

        self._step_counter = 0
        self.time_limit = env_config["time_limit"]

        self._randomize_physics_every_step = env_config['randomize_physics_every_step']
        self._randomize_physics_every_episode = env_config['randomize_physics_every_episode']
        self._current_episode_physics_parameters: Optional[EpisodicPhysicsParams] = None

        if self._randomize_physics_every_step and self._randomize_physics_every_episode:
            raise ValueError("Only one of randomize_physics_every_step and randomize_physics_every_episode "
                             "can be set to True.")
        self._camera_shake_with_random_physics = env_config['camera_shake_with_random_physics']

    def render(self, mode='human', image_scale=1.0, only_show_rgb_if_combined_obs=False):
        return self.base_env.render(mode=mode, image_scale=image_scale,
                                    only_show_rgb_if_combined_obs=only_show_rgb_if_combined_obs)

    def reset(self, episode_physics_parameters=None):
        self._step_counter = 0

        if self._randomize_physics_every_episode:
            if episode_physics_parameters:
                self._current_episode_physics_parameters = episode_physics_parameters
            else:
                # self._current_episode_physics_parameters = EpisodicPhysicsParams(
                #     forward_mean=np.random.uniform(low=0.6, high=1.4),
                #     forward_std=np.random.uniform(low=0.0, high=0.2),
                #     yaw_mean=np.random.uniform(low=0.6, high=1.4),
                #     yaw_std=np.random.uniform(low=0.0, high=0.2)
                # )

                self._current_episode_physics_parameters = EpisodicPhysicsParams(
                    forward_mean=np.random.normal(loc=1.0, scale=0.316),
                    forward_std=0.1,
                    yaw_mean=np.random.normal(loc=1.0, scale=0.316),
                    yaw_std=0.1
                )

        return self._env.reset()

    def step(self, action, override_duckiebot_state=None):
        if self._randomize_physics_every_step:
            assert False, "not for neurips"
            # self.base_env.randomize_physics(relative_vel_scale=np.random.uniform(low=0.25, high=3.0),
            #                                 relative_turn_scale=np.random.uniform(low=0.25, high=3.0),
            #                                 shake_camera=self._camera_shake_with_random_physics
            #                                 )
            if self._camera_shake_with_random_physics:
                self.base_env.randomize_physics(relative_vel_scale=1.0 + (np.random.standard_normal() / 3.0),
                                                relative_turn_scale=1.0 + (np.random.standard_normal() / 3.0),
                                                    cam_x_offset=np.random.uniform(low=-10.0, high=10.0),
                                                    cam_y_offset=np.random.uniform(low=-1.0, high=1.0),
                                                    cam_z_offset=np.random.uniform(low=-10.0, high=10.0),
                                                    cam_roll_offset=np.random.uniform(low=-2.0, high=2.0),
                                                    cam_pitch_offset=np.random.uniform(low=-5.0, high=5.0),
                                                    cam_yaw_offset=np.random.uniform(low=-0.3, high=0.3),
                                                    )
            else:
                self.base_env.randomize_physics(relative_vel_scale=1.0 + (np.random.standard_normal() / 3.0),
                                                relative_turn_scale=1.0 + (np.random.standard_normal() / 3.0))
        elif self._randomize_physics_every_episode:
            if self._camera_shake_with_random_physics:
                self.base_env.randomize_physics(
                    relative_vel_scale=np.random.normal(loc=self._current_episode_physics_parameters.forward_mean,
                                                        scale=self._current_episode_physics_parameters.forward_std),
                    relative_turn_scale=np.random.normal(loc=self._current_episode_physics_parameters.yaw_mean,
                                                     scale=self._current_episode_physics_parameters.yaw_std),
                    cam_x_offset=np.random.uniform(low=-5.0, high=5.0),
                    cam_y_offset=np.random.uniform(low=-1.0, high=1.0),
                    cam_z_offset=np.random.uniform(low=-5.0, high=5.0),
                    cam_roll_offset=np.random.uniform(low=-1.0, high=1.0),
                    cam_pitch_offset=np.random.uniform(low=-3.0, high=3.0),
                    cam_yaw_offset=np.random.uniform(low=-0.3, high=0.3),
                )
            else:
                assert False, "not for neurips"
                self.base_env.randomize_physics(
                    relative_vel_scale=np.random.normal(loc=self._current_episode_physics_parameters.forward_mean,
                                                        scale=self._current_episode_physics_parameters.forward_std),
                    relative_turn_scale=np.random.normal(loc=self._current_episode_physics_parameters.yaw_mean,
                                                     scale=self._current_episode_physics_parameters.yaw_std))

        action = np.asarray(action, dtype=np.float32)
        s, r, d, info = self._env.step(action, override_duckiebot_state=override_duckiebot_state)
        # self.render()
        # print(f"reward: {r}")
        # time.sleep(1.0/6.0)
        assert r <= (1.0 + 1e-8), r

        self._step_counter += 1
        # if not d and self._step_counter >= self.time_limit:
        #     print(f"hit time limit of {self.time_limit}")
        #     d = True
        #     info['is_terminal'] = False
        # elif d:
        #     info['is_terminal'] = True

        if d or self._step_counter >= self.time_limit:
            # non default behavior; never send terminal signal even when environment returns "done"
            # prevents robot killing itself to avoid penalties.
            d = True
            info['is_terminal'] = False

        return s, r, d, info

    def get_duckiebot_metrics_info(self):
        return self.base_env.get_duckiebot_metrics_info()

    def close(self):
        print("closing UELaneFollowingTask")
        self.base_env.close()


class DreamerUELaneFollowingEnv(FromGym):

    def __init__(self, _, time_limit=200, physics_hz=None, use_mask=True,
                 simulate_latency=True,
                 use_simple_physics=False, use_wheel_bias=False, limit_backwards=False,
                 randomize_camera_location_for_tilted_robot=False, is_real=False,
                 separately_return_mask_and_rgb=False,
                 use_rcan_instead_of_gt_mask=False,
                 reverse_actions=False,
                 use_domain_randomization=True,
                 use_randomized_synthetic_world=False,
                 use_alt_game_path=None,
                 render_game_on_screen=False,
                 reward_function="penalize_turning_more",
                 randomize_physics_every_step=False,
                 randomize_physics_every_episode=False,
                 camera_shake_with_random_physics=False):
        if isinstance(physics_hz, str) and physics_hz.lower() == "none":
            physics_hz = None

        self._env_launch_config = {
            "use_domain_randomization": use_domain_randomization,
            "render_game_on_screen": render_game_on_screen,
            "use_mask_observation": use_mask and not use_rcan_instead_of_gt_mask,
            "return_rgb_and_mask_as_observation": separately_return_mask_and_rgb,
            "use_rcan_instead_of_gt_mask": use_rcan_instead_of_gt_mask,
            "preprocess_rgb_observations_with_rcan": use_mask and use_rcan_instead_of_gt_mask and not separately_return_mask_and_rgb,
            "launch_game_process": True,
            "simulate_latency": simulate_latency,
            "frame_stack_amount": 1,
            "time_limit": time_limit,
            "physics_hz": physics_hz,
            "rcan_checkpoint_path": "/home/author1/Downloads/ckpt_9_nov_17.onnx",
            "use_simple_physics": use_simple_physics,
            "use_wheel_bias": use_wheel_bias,
            "limit_backwards": limit_backwards,
            "randomize_camera_location_for_tilted_robot": randomize_camera_location_for_tilted_robot,
            "game_path_override": DEFAULT_GAME_LAUNCHER_PATH.replace("Linux", use_alt_game_path) if use_alt_game_path else None,
            "world_name": "DuckiebotsHolodeckMapDomainRandomization" if use_randomized_synthetic_world else "DuckiebotsHolodeckMap",
            "reward_function": reward_function,
            "randomize_physics_every_step": randomize_physics_every_step,
            "randomize_physics_every_episode": randomize_physics_every_episode,
            "camera_shake_with_random_physics": camera_shake_with_random_physics
        }

        env = _UELaneFollowingEnv(self._env_launch_config)

        self._separately_return_mask_and_rgb = separately_return_mask_and_rgb
        self._is_real = is_real
        self._reverse_actions = reverse_actions
        super().__init__(env=env)

    @functools.cached_property
    def obs_space(self):
        obs_space = super().obs_space
        obs_space['is_real'] = embodied.Space(bool)

        if self._separately_return_mask_and_rgb:
            orig_img_space = obs_space['image']
            obs_space['mask_image'] = embodied.Space(dtype=orig_img_space.dtype,
                                                shape=(orig_img_space.shape[0], orig_img_space.shape[1], 3),
                                                high=orig_img_space.high[0,0,0],
                                                low=orig_img_space.low[0,0,0])
            obs_space['image'] = embodied.Space(dtype=orig_img_space.dtype,
                                                    shape=(orig_img_space.shape[0], orig_img_space.shape[1], 3),
                                                    high=orig_img_space.high[0,0,0],
                                                    low=orig_img_space.low[0,0,0])

        return obs_space

    def step(self, action):
        if self._reverse_actions:
            action['action'] = -np.asarray(action['action'])

        try:
            if 'reset_state' in action and action['reset_state']:
                if self._act_dict:
                    _act = self._unflatten(action)
                else:
                    _act = action[self._act_key]
                _obs, _reward, _done, _info = self._env.step(_act, override_duckiebot_state=action['reset_state'])
                obs = self._obs(_obs,
                                _reward,
                                is_last=bool(_done),
                                is_terminal=bool(_info.get('is_terminal', _done)))
            else:
                obs = super().step(action)
            obs['log_environment_fault'] = False
        except HolodeckException as e:
            self._env.close()  # clean up Unreal Engine Processes
            del self._env
            self._env = _UELaneFollowingEnv(self._env_launch_config)
            _obs = self._env.reset()
            obs = self._obs(_obs,
                            reward=0.0,
                            is_last=True,
                            is_terminal=False)
            obs['log_environment_fault'] = True

            # raise e

        if self._separately_return_mask_and_rgb:
            orig_image_obs = obs['image']
            obs['image'] = orig_image_obs[:, :, :3]
            obs['mask_image'] = orig_image_obs[:, :, 3:]

        obs['is_real'] = self._is_real
        return obs

    def get_duckiebot_metrics_info(self):
        return self._env.get_duckiebot_metrics_info()

    def render(self):
        self._env.render(image_scale=10.0)

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
    #     "return_rgb_and_mask_as_observation": False,
    #     "preprocess_rgb_observations_with_rcan": False,
    #     "rcan_checkpoint_path": "/home/author1/Downloads/ckpt-91.onnx",
    #     "launch_game_process": True,
    #     "simulate_latency": True,
    # })

    # Recommended config for initially trying out this environment with Dreamer
    # env = DreamerUELaneFollowingEnv(None, use_mask=False, separately_return_mask_and_rgb=True, use_randomized_synthetic_world=True, use_rcan_instead_of_gt_mask=True, use_domain_randomization=False)

    # domain randomization for sg
    config_kw = {"limit_backwards": False, "physics_hz": None, "randomize_camera_location_for_tilted_robot": True,
                    "reverse_actions": False,
                    "separately_return_mask_and_rgb": True,
                    "simulate_latency": True, "time_limit": 200, "use_alt_game_path": "", "use_domain_randomization": True,
                    "use_mask": False, "use_randomized_synthetic_world": False, "use_rcan_instead_of_gt_mask": False,
                    "use_simple_physics": False, "use_wheel_bias": False,
                 "randomize_physics_every_step": False,
                 "camera_shake_with_random_physics": False,
                 "randomize_physics_every_episode": True,
                 }

    # gaussian spat for sg
    # config_kw = {"limit_backwards": False, "physics_hz": None, "randomize_camera_location_for_tilted_robot": False,
    #                 "reverse_actions": False,
    #                 "separately_return_mask_and_rgb": False,
    #                 "simulate_latency": True, "time_limit": 200, "use_alt_game_path": None, "use_domain_randomization": False,
    #                 "use_mask": False, "use_randomized_synthetic_world": True, "use_rcan_instead_of_gt_mask": False,
    #                 "use_simple_physics": False, "use_wheel_bias": False,
    #                 "randomize_physics_every_step": True,
    #                 "camera_shake_with_random_physics": False
    #              }


    config_kw.update({
        # "use_alt_game_path": "Linux_jan_6",
        # "use_alt_game_path": "Linux_oct_16"

    })

    env = DreamerUELaneFollowingEnv(None, **config_kw)

    print("env initialized")
    while True:
        i = 0
        env.step({"reset": False, "action": [0.0, 0.0]})
        env.render()

        done = False

        target_frame_time = 1.0/10.0
        # target_frame_time = 0
        prev_frame_time = time.time()
        while not done:
            velocity = gamepad.LeftJoystickY if gamepad else get_keyboard_velocity()
            turning = gamepad.LeftJoystickX if gamepad else get_keyboard_turning()
            action = np.asarray([velocity, turning], np.float32)

            obs = env.step(action={"action": action, "reset": False})

            # print(obs["image"])
            # print(obs["yaw_and_forward_vel"])

            # print(f"timestep {i}, rew {obs['reward']}, yaw_and_forward_vel: {obs['yaw_and_forward_vel']} sim sensor state: {env._env.base_env._latest_observation_state}")
            # print(f"position_and_yaw: {obs['position_and_yaw']} ")

            assert obs['reward'] <= 1.0, obs['reward']

            current_delta = time.time() - prev_frame_time
            sleep_delta = max(target_frame_time - current_delta, 0)
            time.sleep(sleep_delta)
            now = time.time()



            # print(f"delta time: {now - prev_frame_time}")
            print(f"rew: {obs['reward']}")

            prev_frame_time = now

            env.render()

            if gamepad and gamepad.B:
                print("randomizing")
                # env.reset()
                env._env.base_env.randomize()

            i += 1
