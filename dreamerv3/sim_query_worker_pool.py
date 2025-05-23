import concurrent.futures
import importlib
import multiprocessing
from collections import deque

import jax
import numpy as np

from dreamerv3.embodied.core import wrappers
from dreamerv3.embodied.replay import Generic
tree_map = jax.tree_util.tree_map


class SimQueryWorkerPool:

    def __init__(self, config, output_replay_buffer: Generic):

        self._stats_rolling_windows = {
            "pred_state_is_valid": deque(maxlen=10000),
            "pred_state_is_usable": deque(maxlen=10000)
        }

        self._output_replay_buffer = output_replay_buffer
        self._generate_same_format_as_normal_experience = config.sim_query_data_same_in_format_as_normal_experience
        self._max_workers = config.num_simulator_query_workers

        self._executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=self._sim_query_worker_create_environment, initargs=(config,))
        self._futures = []

    def _convert_from_devices(self, value, devices):
        value = jax.device_get(value)
        value = tree_map(np.asarray, value)
        if len(devices) > 1:
            value = tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), value)
        return value

    def submit_world_model_train_outs(self, batch, world_model_train_outs, source_devices):
        actions = self._convert_from_devices(batch["action"],
                                             source_devices)
        symlog_grounded_states = self._convert_from_devices(world_model_train_outs["post"]["symlog_grounded"],
                                                            source_devices)
        self._submit(actions=actions, symlog_grounded_states=symlog_grounded_states)

    def submit_imagined_trajectories(self, behavior_trajectory, source_devices):
        actions = self._convert_from_devices(behavior_trajectory["action"],
                                             source_devices)
        symlog_grounded_states = self._convert_from_devices(behavior_trajectory["symlog_grounded"],
                                                            source_devices)
        self._submit(actions=actions, symlog_grounded_states=symlog_grounded_states)

    @property
    def stats(self) -> dict:
        return {f"{k}_mean": np.mean(v) for k, v in self._stats_rolling_windows.items()}

    def _submit(self, actions, symlog_grounded_states):
        if len(self._futures) >= self._max_workers:

            for future in self._futures:
                exception = None
                try:
                    exception = future.exception(timeout=0)
                except TimeoutError:
                    pass
                if exception:
                    raise exception

            print("Waiting to submit batch to SimQuery worker pool")
            # Wait for a worker to free up. We don't want to submit jobs faster than we can process them.
            concurrent.futures.wait(self._futures, return_when=concurrent.futures.FIRST_COMPLETED)

        if self._generate_same_format_as_normal_experience:
            future = self._executor.submit(self._query_sim_on_grounded_states_same_format_as_normal_experience,
                                           actions, symlog_grounded_states)
        else:
            future = self._executor.submit(self._query_sim_on_grounded_states,
                                           actions, symlog_grounded_states)
        self._futures.append(future)
        future.add_done_callback(lambda f: self._add_steps_to_replay_buffer(
            steps=f.result(), replay=self._output_replay_buffer, stats_rolling_windows=self._stats_rolling_windows))
        # TODO lets put these steps in a queue and have a decicated single thread or place in the main loop to add them to the replay?

        future.add_done_callback(lambda f: self._futures.remove(f))
    def wait(self):
        concurrent.futures.wait(self._futures)

    def close(self):
        self._executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _sim_query_worker_create_environment(_config):
        global sim_model_env
        global config
        config = _config
        sim_model_env = make_sim_model_env(config)

    @staticmethod
    def _query_sim_on_grounded_states(actions, symlog_grounded_states):
        # For every RSSM-predicted sim state ("symlog grounded"), get the corresponding observation from our sim.
        # Then add these observations to our data.

        # print(f"sim query actions: {actions}")

        steps = []
        for sequence_idx in range(actions.shape[0]):
            for step_idx in range(actions.shape[1]):
                step = {"action": actions[sequence_idx, step_idx]}
                symlog_grounded_state = symlog_grounded_states[sequence_idx, step_idx]
                action = step["action"]

                pred_state = symexp_np(symlog_grounded_state)
                assert not np.any(np.isnan(pred_state)), pred_state
                pred_state_obs, is_pred_state_valid, is_pred_state_usable = sim_model_env.reset_to_internal_state(
                    new_internal_state=pred_state)

                pred_state_obs["pred_state_symlog"] = symlog_grounded_state
                pred_state_obs["pred_state_is_valid"] = is_pred_state_valid
                pred_state_obs["pred_state_is_usable"] = is_pred_state_usable
                if not config.allow_predicting_unusable_grounded_states and not is_pred_state_usable:
                    raise ValueError(f"pred_state was unusable: {pred_state}")

                if config.use_sim_forward_dynamics:
                    if is_pred_state_usable:
                        # Get corresponding next states for RSSM-predicted sim states
                        next_state_obs_from_pred = sim_model_env.step(action={"action": action, "reset": False})
                        next_symlog_state = symlog_np(next_state_obs_from_pred['gt_state'])
                        assert not np.any(np.isnan(next_symlog_state)), next_symlog_state
                        pred_state_obs["pred_state_sim_next_symlog_state"] = next_symlog_state
                        for key, value in next_state_obs_from_pred.items():
                            pred_state_obs[f'pred_state_sim_next_{key}'] = value
                    else:
                        pred_state_obs["pred_state_sim_next_symlog_state"] = np.zeros_like(symlog_grounded_state)
                pred_state_obs = {f"pred_state_{k}" if not k.startswith("pred_state")
                                  else k: v for k, v in pred_state_obs.items()}
                step.update(pred_state_obs)
                steps.append(step)
        return steps

    @staticmethod
    def _query_sim_on_grounded_states_same_format_as_normal_experience(actions, symlog_grounded_states):
        steps = []
        for sequence_idx in range(actions.shape[0]):
            for step_idx in range(actions.shape[1]):
                step = {"action": actions[sequence_idx, step_idx]}
                action = step["action"]

                symlog_grounded_state = symlog_grounded_states[sequence_idx, step_idx]
                pred_state = symexp_np(symlog_grounded_state)
                assert not np.any(np.isnan(pred_state)), pred_state
                pred_state_obs, is_pred_state_valid, is_pred_state_usable = sim_model_env.reset_to_internal_state(
                    new_internal_state=pred_state)
                if not config.allow_predicting_unusable_grounded_states and not is_pred_state_usable:
                    raise ValueError(f"pred_state was unusable: {pred_state}")
                if not is_pred_state_usable:
                    print("skipping unusable predicted state")
                    continue
                if pred_state_obs['is_terminal'] or pred_state_obs['is_last']:
                    continue
                pred_state_obs['is_first'] = True
                pred_state_obs['is_last'] = False
                pred_state_obs['gt_state'] = pred_state
                step.update({
                    **pred_state_obs,
                    "is_valid": is_pred_state_valid,
                    "is_usable": is_pred_state_usable,
                    "pred_state_symlog": symlog_grounded_state,
                    "reset": False,
                })
                steps.append(step)

                next_state_obs_from_pred = sim_model_env.step(action={"action": action, "reset": False})

                assert not np.any(np.isnan(next_state_obs_from_pred['gt_state'])), next_state_obs_from_pred['gt_state']

                next_step = {"action": action}  # Reuse previous action as placeholder for last step in a sequence.
                next_state_obs_from_pred['is_first'] = False
                next_state_obs_from_pred['is_last'] = True
                next_step.update({
                    **next_state_obs_from_pred,
                    "is_valid": True,
                    "is_usable": True,
                    "pred_state_symlog": symlog_np(next_state_obs_from_pred['gt_state']),
                    "reset": False,
                })
                steps.append(next_step)
        return steps

    @staticmethod
    def _add_steps_to_replay_buffer(steps: list[dict], replay: Generic, stats_rolling_windows: dict[str, deque]):
        assert len(steps) % replay.length == 0, (len(steps), replay.length)
        for step in steps:
            replay.add(step=step, worker=0)
            if 'pred_state_is_valid' in step:
                stats_rolling_windows['pred_state_is_valid'].append(step['pred_state_is_valid'])
                stats_rolling_windows['pred_state_is_usable'].append(step['pred_state_is_usable'])
        print(f"pred state replay len: {len(replay)}")


def make_sim_model_env(config, **overrides):
    # Constructs the simulation environment that the world model will query privileged data from.

    # You can add custom environments by creating and returning the environment
    # instance here. Environments with different interfaces can be converted
    # using `embodied.envs.from_gym.FromGym` and `embodied.envs.from_dm.FromDM`.
    if not config.sim_query_task:
        raise ValueError("The config value for sim_query_task has not been specified.")
    suite, task = config.sim_query_task.split('_', 1)
    if suite == "metaworld":
        task = f"{'-'.join(task.split('_'))}-v2"
    ctor = {
        'gridworld': 'embodied.envs.gridworld:SimDreamerGridWorldEnv',
        'dmcsim': 'embodied.envs.dmc_simulation:DMCSimulationEnv',
        'dmcmjxsim': 'embodied.envs.dmc_mjx_simulation:DMCMjxSimulationEnv',
    }[suite]
    if isinstance(ctor, str):
        module, cls = ctor.split(':')
        module = importlib.import_module(module)
        ctor = getattr(module, cls)
    kwargs = config.env.get(suite, {})
    if "omit_gt_state" in kwargs:
        kwargs["omit_gt_state"] = False
    if "is_real_env" in kwargs:
        kwargs['is_real_env'] = False
    kwargs.update(overrides)
    env = ctor(task, **kwargs)
    return _wrap_env(env, config)


def _wrap_env(env, config):
    # Copied from the wrap_env function in Dreamer entry-point scripts.
    args = config.wrapper
    for name, space in env.act_space.items():
        if name in ['reset', 'reset_state', 'reset_to_state']:
            continue
        elif space.discrete:
            env = wrappers.OneHotAction(env, name)
        elif args.discretize:
            env = wrappers.DiscretizeAction(env, name, args.discretize)
        else:
            env = wrappers.NormalizeAction(env, name)
    env = wrappers.ExpandScalars(env)
    if args.length:
        env = wrappers.TimeLimit(env, args.length, args.reset)
    if args.checks:
        env = wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete and name != 'reset_state':
            env = wrappers.ClipAction(env, name)
    return env


def symexp_np(x):
    return np.sign(x) * (np.exp(np.abs(x)) - 1)


def symlog_np(x):
    return np.sign(x) * np.log(1 + np.abs(x))
