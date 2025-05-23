import numpy as np
import functools

from dreamerv3.embodied.core.sim_env import SimulationEnv
from dreamerv3.embodied.envs.dmc import DMC
from dreamerv3.embodied.envs.dmc_envs import get_dmc_env
from dreamerv3.embodied import Space
from dm_control.rl import control
from dreamerv3 import embodied
import dm_env


class DMCSimulationEnv(SimulationEnv):
    def __init__(self, task_name: str, repeat=1, sum_rewards_between_action_repeat=True, render=True, size=(64, 64),
                 camera=-1, omit_gt_state=False, normalize_gt_state=False, omit_image=False, is_real_env=False):

        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)  # disable warnings about unstable simulation states
        # (we cause and handle unstable simulation states ourselves)

        if sum_rewards_between_action_repeat and repeat > 1:
            raise NotImplementedError(f"Summing together skipped step rewards with an action repeat > 1 "
                                      f"isn't supported for simulation envs. "
                                      f"Rewards have to be recoverable from the instantaneous state.")

        self._dm_base_env: control.Environment = get_dmc_env(task_name=task_name)

        domain, task = task_name.split('_', 1)
        if camera == -1:
            camera = DMC.DEFAULT_CAMERAS.get(domain, 0)

        self._dreamer_dmc_env = DMC(env=self._dm_base_env,
                                    repeat=repeat,
                                    sum_rewards_between_action_repeat=sum_rewards_between_action_repeat,
                                    render=render,
                                    size=size,
                                    camera=camera)
        self._wrapped_dm_base_env = self._dreamer_dmc_env.get_wrapped_dm_env()

        self._qpos_n = self._dm_base_env.physics.data.qpos.shape[0]
        self._qvel_n = self._dm_base_env.physics.data.qvel.shape[0]

        # self._qacc_n = self._dm_base_env.physics.data.qacc.shape[0]
        # self._ctrl_n = self._dm_base_env.physics.data.ctrl.shape[0]

        # if self._dm_base_env.physics.data.act.shape[0] > 0:
        #     raise NotImplementedError(f"simulation has physic.data.act of {self._dm_base_env.physics.data.act}, "
        #                               f"we may need to track this as internal state too, but it isn't implemented")

        # self._internal_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._qpos_n + self._qvel_n
        #                                                                    # + self._qacc_n + self._ctrl_n
        #                                                                    ,),
        #                                   dtype=np.float32)
        self._internal_space = Space(dtype=np.float32, shape=(self._qpos_n + self._qvel_n,), low=-np.inf, high=np.inf)

        self._omit_gt_state = omit_gt_state
        self._omit_image = omit_image
        self.fixed_normalize_states = normalize_gt_state

        self._normalizer_mean = np.asarray([2.5400023e+01, -4.4158384e-02, 2.6658438e-02, -7.1582668e-02,
                                            -1.2834652e-01, -8.2969911e-02, -6.1658498e-02, 1.9264413e-01,
                                            2.6146227e-01, 5.4118338e+00, 5.6820079e-03, 5.4936246e-03,
                                            -2.5598197e-03, -1.4519677e-02, 6.5552117e-04, -5.8187260e-03,
                                            2.1303508e-02, 3.0177634e-02])
        self._normalizer_std = np.asarray([18.380955, 0.05628293, 0.38872638, 0.27547017, 0.27865195, 0.26659146,
                                           0.11564326, 0.2883745, 0.20463553, 2.1005383, 0.48672038, 1.1180446,
                                           6.442224, 7.0485954, 7.0721116, 2.9723537, 6.52917, 4.469027])
        self._is_real_env = is_real_env

    @functools.cached_property
    def obs_space(self):
        obs_space = self._dreamer_dmc_env.obs_space
        obs_space['gt_state'] = self.internal_state_space
        obs_space['is_real'] = embodied.Space(bool)
        if self._omit_image:
            del obs_space['image']
            if 'image_complement' in obs_space:
                del obs_space['image_complement']
        return obs_space

    @functools.cached_property
    def act_space(self):
        return {**self._dreamer_dmc_env.act_space,
                'reset_state': self.internal_state_space,
                'reset_to_state': embodied.Space(bool)}

    def step(self, action, raise_on_physics_error=False):
        if action['reset']:
            print(f"env: resetting, action is {action}")
        if action['reset'] and action.get('reset_to_state'):
            print(f"env: resetting to {action['reset_state']}")
            assert not self._is_real_env
            new_obs, provided_state_is_valid, provided_state_is_usable = self.reset_to_internal_state(
                new_internal_state=action['reset_state'], return_input_internal_state_in_obs=True
            )
            if provided_state_is_usable:
                new_obs['is_valid'] = provided_state_is_valid
                new_obs['is_usable'] = provided_state_is_usable
                new_obs['is_first'] = True
                return new_obs
            else:
                raise ValueError(f"Provided state was not usable. "
                                 f"If it is expected that some states are occasionally unusable, "
                                 f"this error should be rewritten as a warning with a stat tracking "
                                 f"the frequency of this occurrence.")

        if action['action'] not in self.act_space['action']:
            raise ValueError(f"Got action outside of action space. Space: {self.act_space}, action: {action}")
        num_adjustments_applied = 0
        obs = None
        while obs is None:
            try:
                # self._dm_base_env.physics.data.qacc = np.zeros_like(self._dm_base_env.physics.data.qacc)
                # self._dm_base_env.physics.forward()
                obs = self._dreamer_dmc_env.step(action)
            except control.PhysicsError:
                if raise_on_physics_error:
                    raise
                self._dm_base_env.physics.step()  # allow physics to settle
                num_adjustments_applied += 1
                if num_adjustments_applied > 10:
                    raise ValueError(f"Couldn't step without a physics error.")
        assert 'gt_state' not in obs, list(obs.keys())
        if not self._omit_gt_state:
            obs['gt_state'] = self.get_internal_state()
        if self.fixed_normalize_states and 'vector_obs' in obs:
            obs['vector_obs'] = self.normalize_by_fixed_stats(obs['vector_obs'])
        if self._omit_image and 'image' in obs:
            del obs['image']
        if self._omit_image and 'image_complement' in obs:
            del obs['image_complement']
        obs['is_real'] = self._is_real_env
        obs['is_valid'] = True
        obs['is_usable'] = True
        return obs

    def render(self):
        return self._dreamer_dmc_env.render()

    def normalize_by_fixed_stats(self, internal_state: np.ndarray) -> np.ndarray:
        return (internal_state - self._normalizer_mean) / self._normalizer_std

    def denormalize_by_fixed_stats(self, internal_state: np.ndarray) -> np.ndarray:
        return internal_state * self._normalizer_std + self._normalizer_mean

    @functools.cached_property
    def internal_state_space(self):
        return self._internal_space

    def get_internal_state(self, no_normalize=False):
        if self._omit_gt_state:
            raise ValueError("self._omit_gt_state is set to True. Are you sure you should be calling this?")
        # out_state = np.asarray(self._dm_base_env.physics.get_state(), dtype=self.internal_state_space.dtype)

        out_state = np.concatenate([self._dm_base_env.physics.data.qpos, self._dm_base_env.physics.data.qvel,
                                   # self._dm_base_env.physics.data.qacc,
                                    # self._dm_base_env.physics.data.ctrl
                                    ], dtype=np.float32)

        if self.fixed_normalize_states and not no_normalize:
            out_state = self.normalize_by_fixed_stats(out_state)
        return out_state

    def reset_to_internal_state(self, new_internal_state, return_input_internal_state_in_obs=False) -> (dict, bool, bool):
        assert new_internal_state in self.internal_state_space, f"got {new_internal_state}, len {len(new_internal_state)}, space: {self.internal_state_space}"
        new_internal_state_input = new_internal_state

        if self.fixed_normalize_states:
            new_internal_state = self.denormalize_by_fixed_stats(new_internal_state)

        # if self._dm_base_env.physics.data.act.shape[0] > 0:
        #     raise NotImplementedError(f"simulation has physic.data.act of {self._dm_base_env.physics.data.act}, "
        #                               f"we may need to track this as internal state too, but it isn't implemented")

        if self._dm_base_env.physics.data.act.shape[0] > 0:
           print(f"WARNING: simulation has physic.data.act of {self._dm_base_env.physics.data.act}, "
                f"we may need to track this as internal state too, but it isn't implemented")
           new_internal_state = np.concatenate((new_internal_state, np.zeros_like(self._dm_base_env.physics.data.act)))

        provided_state_is_valid = True

        self._wrapped_dm_base_env.step({'reset': True})
        
        self._dm_base_env.physics.set_state(new_internal_state)
        # self._dm_base_env.physics.data.qpos = np.copy(new_internal_state[:self._qpos_n])
        # self._dm_base_env.physics.data.qvel = np.copy(new_internal_state[self._qpos_n:(self._qpos_n + self._qvel_n)])
        # self._dm_base_env.physics.data.qacc = np.copy(new_internal_state[(self._qpos_n + self._qvel_n):(self._qpos_n + self._qvel_n + self._qacc_n)])
        # self._dm_base_env.physics.data.ctrl = np.copy(new_internal_state[(self._qpos_n + self._qvel_n + self._qacc_n):])

        self._dm_base_env.physics.forward()

        current_new_state_candidate = self.get_internal_state(no_normalize=True)
        # assert np.allclose(new_internal_state, current_new_state_candidate), (
        #     new_internal_state, current_new_state_candidate)
        current_state_candidate_is_stable = False
        num_state_adjustments_applied = 0
        while not current_state_candidate_is_stable:
            try:
                self.step(action={'reset': False, 'action': self.act_space['action'].sample()},
                          raise_on_physics_error=True)
                current_state_candidate_is_stable = True
            except control.PhysicsError:
                provided_state_is_valid = False
                self._dm_base_env.physics.step()  # allow physics to settle
                current_new_state_candidate = self.get_internal_state(no_normalize=True)
                num_state_adjustments_applied += 1
                if num_state_adjustments_applied > 100:
                    # May want to lower the max number of adjustments for better speed
                    # if the simdreamer code is modified to be able to handle unusable states
                    break

        if current_state_candidate_is_stable:
            # If we reached this code, this means that current_new_state_candidate is stable,
            # but we also just stepped in the RL environment with an action, moving us out of current_new_state_candidate.
            # Undo that step call and reset to current_new_state_candidate:
            self._wrapped_dm_base_env.step({'reset': True})

            if self._dm_base_env.physics.data.act.shape[0] > 0:
                print(f"WARNING: simulation has physic.data.act of {self._dm_base_env.physics.data.act}, "
                      f"we may need to track this as internal state too, but it isn't implemented")
                current_new_state_candidate = np.concatenate(
                    (current_new_state_candidate, np.zeros_like(self._dm_base_env.physics.data.act)))

            self._dm_base_env.physics.set_state(current_new_state_candidate)
            # self._dm_base_env.physics.data.qpos = np.copy(current_new_state_candidate[:self._qpos_n])
            # self._dm_base_env.physics.data.qvel = np.copy(current_new_state_candidate[self._qpos_n:(self._qpos_n + self._qvel_n)])
            # self._dm_base_env.physics.data.qacc = np.copy(current_new_state_candidate[(self._qpos_n + self._qvel_n):(self._qpos_n + self._qvel_n + self._qacc_n)])
            # self._dm_base_env.physics.data.ctrl = np.copy(current_new_state_candidate[(self._qpos_n + self._qvel_n + self._qacc_n):])
            
            self._dm_base_env.physics.forward()
            self._dm_base_env._task.after_step(self._dm_base_env.physics)
        else:
            # Otherwise just reset the environment, so we can at least get an observation of the correct shape
            self._wrapped_dm_base_env.step({'reset': True})
        provided_state_is_usable = current_state_candidate_is_stable

        new_timestep: dm_env.TimeStep = self.get_mujoco_timestep_for_current_physics(control_env=self._dm_base_env)
        new_obs: dict = self._wrapped_dm_base_env._obs(new_timestep)
        new_obs = self._wrapped_dm_base_env.expand_obs_scalars(new_obs)

        assert 'image' not in new_obs, list(new_obs.keys())
        if self._dreamer_dmc_env._render and not self._omit_image:
            new_obs['image'] = self._dreamer_dmc_env.render()
        if self._omit_image and 'image_complement' in new_obs:
            del new_obs['image_complement']

        assert 'reward' in new_obs, list(new_obs.keys())
        assert 'is_terminal' in new_obs, list(new_obs.keys())
        assert 'is_last' in new_obs, list(new_obs.keys())
        assert 'is_first' in new_obs, list(new_obs.keys())
        assert not new_obs['is_first'], new_obs['is_first']  # TODO what was the reasoning for this?

        assert 'gt_state' not in new_obs, list(new_obs.keys())
        if return_input_internal_state_in_obs:
            new_obs['gt_state'] = new_internal_state_input
        else:
            new_obs['gt_state'] = self.get_internal_state()

        assert 'is_real' not in new_obs,  list(new_obs.keys())
        new_obs['is_real'] = False

        if not provided_state_is_usable:
            # We never successfully reset to a state. new_obs is incorrect in this context, replace it with zeros.
            # Indicate that we shouldn't use this observation with provided_state_is_valid = False
            new_obs = {k: np.zeros_like(v) for k, v in new_obs.items()}
            assert not provided_state_is_valid

        if self.fixed_normalize_states and 'vector_obs' in new_obs:
            new_obs['vector_obs'] = self.normalize_by_fixed_stats(new_obs['vector_obs'])
        return new_obs, provided_state_is_valid, provided_state_is_usable

    @staticmethod
    def get_mujoco_timestep_for_current_physics(control_env: control.Environment):
        # 1) Assumes the timestep is not the first of an episode
        # 2) Does not modify any state (physics or otherwise) of the mujoco env

        reward = control_env._task.get_reward(control_env._physics)
        observation = control_env._task.get_observation(control_env._physics)
        if control_env._flat_observation:
            observation = control.flatten_observation(observation)

        if control_env._step_count >= control_env._step_limit:
            discount = 1.0
        else:
            discount = control_env._task.get_termination(control_env._physics)

        episode_over = discount is not None

        if episode_over:
            return dm_env.TimeStep(
                dm_env.StepType.LAST, reward, discount, observation)
        else:
            return dm_env.TimeStep(dm_env.StepType.MID, reward, 1.0, observation)


# if __name__ == '__main__':
#     # testing module on mujoco envs
#     from tqdm import tqdm
#
#     # # Testing DMCwrapper:
#
#     n_tests_per_env = 1000
#     test_step_forward_n_times = 10
#
#     # from dm_control import suite
#     # domain_and_task_names = suite.BENCHMARKING
#     # domain_and_task_names = [("dmc", "cheetah_run")]
#     domain_and_task_names = [("dmc", "orig_cartpole_balance")]
#
#     for domain_name, task_name in domain_and_task_names:
#         print(f"Testing on domain {domain_name} and task {task_name}")
#         sim_env = DMCSimulationEnv(task_name=task_name)
#         action_space = sim_env.act_space
#         print(f"action space: {action_space}")
#         n_passed = 0
#         for test_i in tqdm(range(n_tests_per_env)):
#         # for test_i in range(n_tests_per_env):
#
#             obs = sim_env.step({'reset': True, 'action': sim_env.act_space['action'].sample()})
#             trajectory = []
#             done = False
#             while not done and len(trajectory) < 100:
#                 a = {'reset': False, 'action': sim_env.act_space['action'].sample()}
#                 trajectory.append({
#                     'internal_state': sim_env.get_internal_state(),
#                     'observation': obs,
#                     'action': a
#                 })
#
#                 obs = sim_env.step(a)
#                 #
#                 # print(f"qpos: {sim_env._dm_base_env.physics.data.qpos} "
#                 #       f"qvel: {sim_env._dm_base_env.physics.data.qvel} "
#                 #       f"act: {sim_env._dm_base_env.physics.data.act} "
#                 #       f"qacc: {sim_env._dm_base_env.physics.data.qacc} "
#                 #       f"ctrl: {sim_env._dm_base_env.physics.data.ctrl} "
#                 #       f"prev action: {a}")
#                 done = obs['is_terminal'] or obs['is_last']
#
#             starting_index = np.random.randint(len(trajectory))
#             current_index = starting_index
#             obs, is_valid, is_usable = sim_env.reset_to_internal_state(trajectory[current_index]['internal_state'])
#             assert np.allclose(sim_env.get_internal_state(), trajectory[current_index]['internal_state'])
#             flag = True
#             while current_index < len(trajectory) and (current_index - starting_index < test_step_forward_n_times):
#                 target_obs = trajectory[current_index]['observation']
#                 for key in obs.keys():
#                     if key in ['is_first']:
#                         # dont test this key
#                         continue
#                     if key == 'reward' and current_index == 0:
#                         # reward may differ between sim_reset and actual be actual assumes is_first to be true, whereas sim reset doesn't
#                         continue
#
#                     is_close = np.allclose(target_obs[key], obs[key], atol=1e-4)
#                     if not is_close:
#                         abs_error = np.sum(np.abs(target_obs[key] - obs[key]))
#                         print(f"\nFailed for {key} at timestep {current_index} - "
#                               f"\ntarget:\n{target_obs[key]}"
#                               f"\ngot:\n{obs[key]}"
#                               f"\ninternal_state:\n{trajectory[current_index]['internal_state']}",
#                               f"\ntotal error: {abs_error}\n"
#                               f"\n\n")
#                     flag = flag and is_close
#
#                 a = trajectory[current_index]['action']
#                 obs = sim_env.step(a)
#
#                 current_index += 1
#             if flag:
#                 n_passed += 1
#
#         print(f"Domain {domain_name} and task {task_name} \t\tTest passed: {n_passed}/{n_tests_per_env}")
#         print()


if __name__ == '__main__':
    import cv2
    # testing module on mujoco envs
    # domain_and_task_names = [("dmcsim", "finger_turn_hard_torque_applied")]
    domain_and_task_names = [("dmcsim", "cup_catch_windy")]
    # pendulum_swingup_reversed_actions

    for domain_name, task_name in domain_and_task_names:
        print(f"Testing on domain {domain_name} and task {task_name}")
        sim_env = DMCSimulationEnv(task_name=task_name)

        action_space = sim_env.act_space
        print(f"action space: {action_space}")
        sim_env.step({"reset": True, 'action': sim_env.act_space['action'].sample(), "reset_to_state": None})
        while True:
            img = sim_env.render()
            img_resized = cv2.resize(img, (600, 600), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("dmc", cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            # sim_env.step({'reset': False, 'action': [0.0]})
            sim_env.step({'reset': False, 'action': sim_env.act_space['action'].sample(), "reset_to_state": None})