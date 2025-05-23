import random
import collections
from collections import deque
from typing import Optional
import numpy as np

from dreamerv3.embodied.core.basics import convert


class Driver:
    _CONVERSION = {
        np.floating: np.float32,
        np.signedinteger: np.int32,
        np.uint8: np.uint8,
        bool: bool,
    }

    def __init__(self, env=None, env_reset_initial_state_deque: Optional[deque] = None, static_reset_deque=False, **kwargs):
        assert env is None or len(env) > 0
        self._env = env
        self._env_reset_initial_state_deque = env_reset_initial_state_deque
        self._static_reset_deque = static_reset_deque
        if self._env_reset_initial_state_deque and self._static_reset_deque:
            print(f"Using static reset state deque of size {len(self._env_reset_initial_state_deque)}")
        self._kwargs = kwargs
        self._on_steps = []
        self._on_episodes = []
        self.reset()

    def _get_reset_state(self):
        if self._static_reset_deque:
            state = random.choice(self._env_reset_initial_state_deque)
        else:
            state = self._env_reset_initial_state_deque.pop()
        print(f"resetting to {state}")
        return state

    def reset(self):
        if self._env:
            self._acts = {
                k: convert(np.zeros((len(self._env),) + v.shape, v.dtype))
                for k, v in self._env.act_space.items()}
            self._acts['reset'] = np.ones(len(self._env), bool)

            if 'gt_state' in self._env.obs_space:
                if self._env_reset_initial_state_deque:
                    self._acts['reset_state'] = convert([self._get_reset_state() for _ in range(len(self._env))])
                    self._acts['reset_to_state'] = np.ones(len(self._env), bool)
                else:
                    self._acts['reset_to_state'] = np.zeros(len(self._env), bool)

            self._eps = [collections.defaultdict(list) for _ in range(len(self._env))]
        self._state = None

    def on_step(self, callback):
        self._on_steps.append(callback)

    def on_episode(self, callback):
        if not self._env:
            raise ValueError(f"on_episode callbacks will never be called because no env was provided.")
        self._on_episodes.append(callback)

    def __call__(self, policy, steps=0, episodes=0):
        step, episode = 0, 0
        while step < steps or episode < episodes:
            step, episode = self._step(policy, step, episode)

    def _step(self, policy, step, episode):
        if not self._env:
            step += 1
            [fn(None, 0, **self._kwargs) for fn in self._on_steps]
            return step, episode

        assert all(len(x) == len(self._env) for x in self._acts.values())
        acts = {k: v for k, v in self._acts.items() if not k.startswith('log_')}
        obs = self._env.step(acts)
        obs = {k: convert(v) for k, v in obs.items()}
        assert all(len(x) == len(self._env) for x in obs.values()), obs
        acts, self._state = policy(obs, self._state, **self._kwargs)

        # nan check added for simdreamer
        if np.any(np.isnan(acts['action'])):
            raise ValueError(f"actions had NaN: {acts}")

        acts = {k: convert(v) for k, v in acts.items()}
        if obs['is_last'].any():
            mask = 1 - obs['is_last']
            acts = {k: v * self._expand(mask, len(v.shape)) for k, v in acts.items()}
        acts['reset'] = obs['is_last'].copy()

        if 'gt_state' in obs:
            if self._env_reset_initial_state_deque:
                acts['reset_state'] = convert(
                    [self._get_reset_state() if acts['reset'][i]
                     else np.zeros_like(obs['gt_state'][i]) for i in range(len(self._env))]
                )
                acts['reset_to_state'] = acts['reset'].copy()
            else:
                acts['reset_state'] = np.zeros_like(obs['gt_state'])
                acts['reset_to_state'] = np.zeros(len(self._env), bool)

        self._acts = acts
        trns = {**obs, **acts}
        if obs['is_first'].any():
            for i, first in enumerate(obs['is_first']):
                if first:
                    self._eps[i].clear()
        for i in range(len(self._env)):
            trn = {k: v[i] for k, v in trns.items()}
            [self._eps[i][k].append(v) for k, v in trn.items()]
            [fn(trn, i, **self._kwargs) for fn in self._on_steps]
            step += 1
        if obs['is_last'].any():
            for i, done in enumerate(obs['is_last']):
                if done:
                    ep = {k: convert(v) for k, v in self._eps[i].items()}
                    [fn(ep.copy(), i, **self._kwargs) for fn in self._on_episodes]
                    episode += 1
        return step, episode

    def _expand(self, value, dims):
        while len(value.shape) < dims:
            value = value[..., None]
        return value
