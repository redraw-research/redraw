import functools
import time

import jax.numpy as jnp

from dreamerv3.embodied.core import base
from dreamerv3.embodied.core import space as spacelib


class TimeLimit(base.Wrapper):

    def __init__(self, env, duration, reset=True):
        super().__init__(env)
        self._duration = duration
        self._reset = reset
        self._step = 0
        self._done = False

    def step(self, action):
        if action['reset'] or self._done:
            self._step = 0
            self._done = False
            if self._reset:
                action.update(reset=True)
                return self.env.step(action)
            else:
                action.update(reset=False)
                obs = self.env.step(action)
                obs['is_first'] = True
                return obs
        self._step += 1
        obs = self.env.step(action)
        if self._duration and self._step >= self._duration:
            obs['is_last'] = True
        self._done = obs['is_last']
        return obs


class ActionRepeat(base.Wrapper):

    def __init__(self, env, repeat, sum_intra_step_rewards=True):
        super().__init__(env)
        self._repeat = repeat
        self._sum_intra_step_rewards = sum_intra_step_rewards

    def step(self, action):
        if action['reset']:
            return self.env.step(action)
        reward = 0.0
        for _ in range(self._repeat):
            obs = self.env.step(action)
            if self._sum_intra_step_rewards:
                reward += obs['reward']
            else:
                reward = obs['reward']
            if obs['is_last'] or obs['is_terminal']:
                break
        obs['reward'] = jnp.float32(reward)
        return obs


class ClipAction(base.Wrapper):

    def __init__(self, env, key='action', low=-1, high=1):
        super().__init__(env)
        self._key = key
        self._low = low
        self._high = high

    def step(self, action):
        clipped = jnp.clip(action[self._key], self._low, self._high)
        return self.env.step({**action, self._key: clipped})


class NormalizeAction(base.Wrapper):

    def __init__(self, env, key='action'):
        super().__init__(env)
        self._key = key
        self._space = env.act_space[key]
        self._mask = jnp.isfinite(self._space.low) & jnp.isfinite(self._space.high)
        self._low = jnp.where(self._mask, self._space.low, -1)
        self._high = jnp.where(self._mask, self._space.high, 1)

    @functools.cached_property
    def act_space(self):
        low = jnp.where(self._mask, -jnp.ones_like(self._low), self._low)
        high = jnp.where(self._mask, jnp.ones_like(self._low), self._high)
        space = spacelib.Space(jnp.float32, self._space.shape, low, high)
        return {**self.env.act_space, self._key: space}

    def step(self, action):
        orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
        orig = jnp.where(self._mask, orig, action[self._key])
        return self.env.step({**action, self._key: orig})


class OneHotAction(base.Wrapper):

    def __init__(self, env, key='action'):
        super().__init__(env)
        self._count = int(env.act_space[key].high)
        self._key = key

    @functools.cached_property
    def act_space(self):
        shape = (self._count,)
        space = spacelib.Space(jnp.float32, shape, 0, 1)
        space.sample = functools.partial(self._sample_action, self._count)
        space._discrete = True
        return {**self.env.act_space, self._key: space}

    def step(self, action):
        if not action['reset']:
            assert action[self._key].min() == 0.0, action
            assert action[self._key].max() == 1.0, action
            assert action[self._key].sum() == 1.0, action
        index = jnp.argmax(action[self._key])
        return self.env.step({**action, self._key: index})

    @staticmethod
    def _sample_action(count):
        index = jnp.random.randint(0, count)
        action = jnp.zeros(count, dtype=jnp.float32)
        action[index] = 1.0
        return action


class ExpandScalars(base.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._obs_expanded = []
        self._obs_space = {}
        for key, space in self.env.obs_space.items():
            if space.shape == () and key != 'reward' and not space.discrete:
                space = spacelib.Space(space.dtype, (1,), space.low, space.high)
                self._obs_expanded.append(key)
            self._obs_space[key] = space
        self._act_expanded = []
        self._act_space = {}
        for key, space in self.env.act_space.items():
            if space.shape == () and not space.discrete:
                space = spacelib.Space(space.dtype, (1,), space.low, space.high)
                self._act_expanded.append(key)
            self._act_space[key] = space

    @functools.cached_property
    def obs_space(self):
        return self._obs_space

    @functools.cached_property
    def act_space(self):
        return self._act_space

    def step(self, action):
        action = {
            key: jnp.squeeze(value, 0) if key in self._act_expanded else value
            for key, value in action.items()}
        obs = self.env.step(action)
        obs = self.expand_obs_scalars(obs)
        return obs

    def expand_obs_scalars(self, obs):
        return {
            key: jnp.expand_dims(value, 0) if key in self._obs_expanded else value
            for key, value in obs.items()
        }


class FlattenTwoDimObs(base.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._keys = []
        self._obs_space = {}
        for key, space in self.env.obs_space.items():
            if len(space.shape) == 2:
                space = spacelib.Space(
                    space.dtype,
                    (int(jnp.prod(space.shape)),),
                    space.low.flatten(),
                    space.high.flatten())
                self._keys.append(key)
            self._obs_space[key] = space

    @functools.cached_property
    def obs_space(self):
        return self._obs_space

    def step(self, action):
        obs = self.env.step(action).copy()
        for key in self._keys:
            obs[key] = obs[key].flatten()
        return obs


class FlattenTwoDimActions(base.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._origs = {}
        self._act_space = {}
        for key, space in self.env.act_space.items():
            if len(space.shape) == 2:
                space = spacelib.Space(
                    space.dtype,
                    (int(jnp.prod(space.shape)),),
                    space.low.flatten(),
                    space.high.flatten())
                self._origs[key] = space.shape
            self._act_space[key] = space

    @functools.cached_property
    def act_space(self):
        return self._act_space

    def step(self, action):
        action = action.copy()
        for key, shape in self._origs.items():
            action[key] = action[key].reshape(shape)
        return self.env.step(action)


class CheckSpaces(base.Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        for key, value in action.items():
            self._check(value, self.env.act_space[key], key)
        obs = self.env.step(action)
        for key, value in obs.items():
            self._check(value, self.env.obs_space[key], key)
        return obs

    def _check(self, value, space, key):
        if not isinstance(value, (
                jnp.ndarray, jnp.generic, list, tuple, int, float, bool)):
            raise TypeError(f'Invalid type {type(value)} for key {key}.')
        if value in space:
            return
        dtype = jnp.array(value).dtype
        shape = jnp.array(value).shape
        lowest, highest = jnp.min(value), jnp.max(value)
        raise ValueError(
            f"Value for '{key}' with dtype {dtype}, shape {shape}, "
            f"lowest {lowest}, highest {highest} is not in {space}.")


class DiscretizeAction(base.Wrapper):

    def __init__(self, env, key='action', bins=5):
        super().__init__(env)
        self._dims = jnp.squeeze(env.act_space[key].shape, 0).item()
        self._values = jnp.linspace(-1, 1, bins)
        self._key = key

    @functools.cached_property
    def act_space(self):
        shape = (self._dims, len(self._values))
        space = spacelib.Space(jnp.float32, shape, 0, 1)
        space.sample = functools.partial(
            self._sample_action, self._dims, self._values)
        space._discrete = True
        return {**self.env.act_space, self._key: space}

    def step(self, action):
        if not action['reset']:
            assert (action[self._key].min(-1) == 0.0).all(), action
            assert (action[self._key].max(-1) == 1.0).all(), action
            assert (action[self._key].sum(-1) == 1.0).all(), action
        indices = jnp.argmax(action[self._key], axis=-1)
        continuous = jnp.take(self._values, indices)
        return self.env.step({**action, self._key: continuous})

    @staticmethod
    def _sample_action(dims, values):
        indices = jnp.random.randint(0, len(values), dims)
        action = jnp.zeros((dims, len(values)), dtype=jnp.float32)
        action[jnp.arange(dims), indices] = 1.0
        return action


class ResizeImage(base.Wrapper):

    def __init__(self, env, size=(64, 64)):
        super().__init__(env)
        self._size = size
        self._keys = [
            k for k, v in env.obs_space.items()
            if len(v.shape) > 1 and v.shape[:2] != size]
        print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
        if self._keys:
            from PIL import Image
            self._Image = Image

    @functools.cached_property
    def obs_space(self):
        spaces = self.env.obs_space
        for key in self._keys:
            shape = self._size + spaces[key].shape[2:]
            spaces[key] = spacelib.Space(jnp.uint8, shape)
        return spaces

    def step(self, action):
        obs = self.env.step(action)
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def _resize(self, image):
        image = self._Image.fromarray(image)
        image = image.resize(self._size, self._Image.NEAREST)
        image = jnp.array(image)
        return image


class RenderImage(base.Wrapper):

    def __init__(self, env, key='image'):
        super().__init__(env)
        self._key = key
        self._shape = self.env.render().shape

    @functools.cached_property
    def obs_space(self):
        spaces = self.env.obs_space
        spaces[self._key] = spacelib.Space(jnp.uint8, self._shape)
        return spaces

    def step(self, action):
        obs = self.env.step(action)
        obs[self._key] = self.env.render()
        return obs


class RestartOnException(base.Wrapper):

    def __init__(
            self, ctor, exceptions=(Exception,), window=300, maxfails=2, wait=20):
        if not isinstance(exceptions, (tuple, list)):
            exceptions = [exceptions]
        self._ctor = ctor
        self._exceptions = tuple(exceptions)
        self._window = window
        self._maxfails = maxfails
        self._wait = wait
        self._last = time.time()
        self._fails = 0
        super().__init__(self._ctor())

    def step(self, action):
        try:
            return self.env.step(action)
        except self._exceptions as e:
            if time.time() > self._last + self._window:
                self._last = time.time()
                self._fails = 1
            else:
                self._fails += 1
            if self._fails > self._maxfails:
                raise RuntimeError('The env crashed too many times.')
            message = f'Restarting env after crash with {type(e).__name__}: {e}'
            print(message, flush=True)
            time.sleep(self._wait)
            self.env = self._ctor()
            action['reset'] = jnp.ones_like(action['reset'])
            return self.env.step(action)
