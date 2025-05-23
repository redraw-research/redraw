import os

import jax
import jax.numpy as jnp
import numpy as np

from dreamerv3 import embodied
from dreamerv3 import jaxutils
from dreamerv3 import ninjax as nj

tree_map = jax.tree_util.tree_map
tree_flatten = jax.tree_util.tree_flatten


def Wrapper(agent_cls):
    class Agent(JAXAgent):
        configs = agent_cls.configs
        inner = agent_cls

        def __init__(self, *args, **kwargs):
            super().__init__(agent_cls, *args, **kwargs)

    return Agent


class JAXAgent(embodied.Agent):

    def __init__(self, agent_cls, obs_space, act_space, step, config, sim_query_env=None):
        super().__init__(obs_space, act_space, step, config, sim_query_env)
        self.config = config.jax
        self.batch_size = config.batch_size
        self.batch_length = config.batch_length
        self.data_loaders = config.data_loaders
        self.data_load_prefetch_source = config.data_load_prefetch_source
        self.data_load_prefetch_batch = config.data_load_prefetch_batch

        self._setup()
        self.agent = agent_cls(obs_space, act_space, step, config, sim_query_env, name='agent')
        self.rng = np.random.default_rng(config.seed)

        available = jax.devices(self.config.platform)
        self.policy_devices = [available[i] for i in self.config.policy_devices]
        self.train_devices = [available[i] for i in self.config.train_devices]
        self.single_device = (self.policy_devices == self.train_devices) and (
                len(self.policy_devices) == 1)
        print(f'JAX devices ({jax.local_device_count()}):', available)
        print('Policy devices:', ', '.join([str(x) for x in self.policy_devices]))
        print('Train devices: ', ', '.join([str(x) for x in self.train_devices]))

        self._once = True
        self._updates = embodied.Counter()
        self._should_metrics = embodied.when.Every(self.config.metrics_every)

        self._alt_once = True
        self._alt_updates = embodied.Counter()
        self._alt_should_metrics = embodied.when.Every(self.config.metrics_every)

        self._transform()
        self.varibs = self._init_varibs(obs_space, act_space, augment_images=bool(config.apply_image_augmentations))
        self.sync()

    def policy(self, obs, state=None, mode='train'):
        obs = obs.copy()
        obs = self._convert_inps(obs, self.policy_devices)
        rng = self._next_rngs(self.policy_devices)
        varibs = self.varibs if self.single_device else self.policy_varibs
        if state is None:
            state, _ = self._init_policy(varibs, rng, obs['is_first'])
        else:
            state = tree_map(
                np.asarray, state, is_leaf=lambda x: isinstance(x, list))
            state = self._convert_inps(state, self.policy_devices)
        (outs, state), _ = self._policy(varibs, rng, obs, state, mode=mode)
        outs = self._convert_outs(outs, self.policy_devices)
        # TODO: Consider keeping policy states in accelerator memory.
        state = self._convert_outs(state, self.policy_devices)
        return outs, state

    def train(self, data, state=None, during_train_callback=None):
        rng = self._next_rngs(self.train_devices)
        if state is None:
            state, self.varibs = self._init_train(self.varibs, rng, data['is_first'])
        (outs, state, mets), self.varibs = self._train(
            self.varibs, rng, data, state)
        if during_train_callback:
            during_train_callback()

        self._updates.increment()
        if self._should_metrics(self._updates):
            mets = self._convert_mets(mets, self.train_devices)
        else:
            mets = {}
        if self._once:
            self._once = False
            assert jaxutils.Optimizer.PARAM_COUNTS
            for name, count in jaxutils.Optimizer.PARAM_COUNTS.items():
                mets[f'params_{name}'] = float(count) if count is not None else None
        return outs, state, mets

    @property
    def use_train_alt(self):
        return self.agent.use_train_alt

    def modify_data_for_train_alt(self, data, train_outs):
        data = jax.device_get(data)
        train_outs = jax.device_get(train_outs)
        modified_data = self.agent.modify_data_for_train_alt(data=data, train_outs=train_outs)
        return self._convert_inps(modified_data, self.train_devices)

    def train_alt(self, experience_data, pred_state_data, state_stub=None):
        if not self.agent.use_train_alt:
            raise AttributeError("Not calling train_alt since agent.use_train_alt returns False.")
        rng = self._next_rngs(self.train_devices)
        (outs, mets), self.varibs = self._train_alt(self.varibs, rng, experience_data, pred_state_data, state_stub)
        outs = self._convert_outs(outs, self.train_devices)
        self._alt_updates.increment()
        if self._alt_should_metrics(self._alt_updates):
            mets = self._convert_mets(mets, self.train_devices)
        else:
            mets = {}
        if self._alt_once:
            self._alt_once = False
            assert jaxutils.Optimizer.PARAM_COUNTS
            for name, count in jaxutils.Optimizer.PARAM_COUNTS.items():
                mets[f'params_{name}'] = float(count)
        return outs, mets

    def train_both(self, train_data, train_alt_experience_data, train_alt_pred_state_data, train_state=None, during_train_callback=None):
        if not self.agent.use_train_alt:
            raise AttributeError("Not calling train_both since agent.use_train_alt returns False.")
        rng = self._next_rngs(self.train_devices)
        if train_state is None:
            train_state, self.varibs = self._init_train(self.varibs, rng, train_data['is_first'])

        # (train_outs, train_state, train_metrics), self.varibs = self._train(self.varibs, rng, train_data, train_state)
        # (_, train_alt_metrics), self.varibs = self._train_alt(
        #     self.varibs, rng, train_alt_experience_data, train_alt_pred_state_data, train_state)

        (train_outs, state_out, train_metrics, train_alt_metrics), self.varibs = self._train_both(
            self.varibs, rng, train_data, train_alt_experience_data, train_alt_pred_state_data, train_state
        )
        if during_train_callback:
            during_train_callback()

        self._updates.increment()
        if self._should_metrics(self._updates):
            train_metrics = self._convert_mets(train_metrics, self.train_devices)
        else:
            train_metrics = {}
        if self._once:
            self._once = False
            assert jaxutils.Optimizer.PARAM_COUNTS
            for name, count in jaxutils.Optimizer.PARAM_COUNTS.items():
                train_metrics[f'params_{name}'] = float(count) if count is not None else None

        self._alt_updates.increment()
        if self._alt_should_metrics(self._alt_updates):
            train_alt_metrics = self._convert_mets(train_alt_metrics, self.train_devices)
        else:
            train_alt_metrics = {}
        if self._alt_once:
            self._alt_once = False
            assert jaxutils.Optimizer.PARAM_COUNTS
            for name, count in jaxutils.Optimizer.PARAM_COUNTS.items():
                train_alt_metrics[f'params_{name}'] = float(count) if count is not None else None

        return train_outs, train_state, train_metrics, train_alt_metrics

    def report(self, data, train_alt_data=None):
        rng = self._next_rngs(self.train_devices)
        mets, _ = self._report(self.varibs, rng, data, train_alt_data)
        mets = self._convert_mets(mets, self.train_devices)
        return mets

    def report_non_compiled(self, data):
        rng = self._next_rngs(self.train_devices)
        mets = {}
        if self._report_non_compiled:
            mets, _ = self._report_non_compiled(self.varibs, rng, data)
            mets = self._convert_mets(mets, self.train_devices)
        return mets

    def update_normalization_state(self, data):
        data = self._convert_inps(data, self.train_devices)
        rng = self._next_rngs(self.train_devices)
        _, self.varibs = self._update_normalization_state(self.varibs, rng, data)

    def update_normalization_state_from_replay_buffer(self, replay_buffer):
        cpu_device = jax.devices('cpu')[0]
        with jax.transfer_guard('allow'):
            with jax.default_device(cpu_device):
                varibs = jax.device_get(self.varibs)

                offline_normalization_updates = 0
                for step in replay_buffer.all_steps_iterator():
                    if offline_normalization_updates % 10000 == 0:
                        print(f"update {offline_normalization_updates}")
                    # step = self._convert_inps(step, devices=[cpu_device])
                    # rng = self._next_rngs(devices=[cpu_device])
                    rng = self.rng.integers(2 ** 63 - 1)
                    _, varibs = self._update_normalization_state_cpu(varibs, rng, step)

                    offline_normalization_updates += 1
                print(f"Updated agent normalization based on {offline_normalization_updates} offline steps")

        if len(self.train_devices) == 1:
            self.varibs = jax.device_put(varibs, self.train_devices[0])
        else:
            self.varibs = jax.device_put_replicated(varibs, self.train_devices)
        self.sync()

    def dataset(self, generator):
        batcher = embodied.Batcher(
            sources=[generator] * self.batch_size,
            workers=self.data_loaders,
            postprocess=lambda x: self._convert_inps(x, self.train_devices),
            prefetch_source=self.data_load_prefetch_source, prefetch_batch=self.data_load_prefetch_batch)
        return batcher()

    def dataset_evenly_from_two_sources(self, generator1, generator2):
        sources1 = [generator1] * ((self.batch_size // 2) + (self.batch_size % 2))
        sources2 = [generator2] * (self.batch_size // 2)
        batcher = embodied.Batcher(
            sources=[*sources1, *sources2],
            workers=self.data_loaders,
            postprocess=lambda x: self._convert_inps(x, self.train_devices),
            prefetch_source=self.data_load_prefetch_source, prefetch_batch=self.data_load_prefetch_batch)
        return batcher()

    def dataset_evenly_from_three_sources(self, generator1, generator2, generator3):
        sources1 = [generator1] * ((self.batch_size // 3) + (self.batch_size % 3))
        sources2 = [generator2] * (self.batch_size // 3)
        sources3 = [generator3] * (self.batch_size // 3)
        batcher = embodied.Batcher(
            sources=[*sources1, *sources2, *sources3],
            workers=self.data_loaders,
            postprocess=lambda x: self._convert_inps(x, self.train_devices),
            prefetch_source=self.data_load_prefetch_source, prefetch_batch=self.data_load_prefetch_batch)
        return batcher()

    def save(self):
        if len(self.train_devices) > 1:
            varibs = tree_map(lambda x: x[0], self.varibs)
        else:
            varibs = self.varibs
        varibs = jax.device_get(varibs)
        data = tree_map(np.asarray, varibs)
        return data

    def load(self, state):
        # TODO, add functionality to also only load certain prefixes

        print(f"Loading Jax Agent State: {list(state.keys())}")
        for skip_prefix in self.config.dont_load_weights_prefixes_from_checkpoint:
            if skip_prefix:
                for key in list(state.keys()):
                    if key.startswith(skip_prefix):
                        print(f"Skipping loading {key} from checkpoint.")
                        if key in self.varibs:
                            state[key] = self.varibs[key]
                        else:
                            del state[key]
                for key in list(self.varibs.keys()):
                    if key not in state:
                        print(f"Current model has {key}, "
                              f"which was not included in loaded checkpoint. Using current value for this key.")
                        state[key] = self.varibs[key]

        if len(self.train_devices) == 1:
            self.varibs = jax.device_put(state, self.train_devices[0])
        else:
            self.varibs = jax.device_put_replicated(state, self.train_devices)
        self.sync()

    def sync(self):
        if self.single_device:
            return
        if len(self.train_devices) == 1:
            varibs = self.varibs
        else:
            varibs = tree_map(lambda x: x[0].device_buffer, self.varibs)
        if len(self.policy_devices) == 1:
            self.policy_varibs = jax.device_put(varibs, self.policy_devices[0])
        else:
            self.policy_varibs = jax.device_put_replicated(
                varibs, self.policy_devices)

    def _setup(self):
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
            tf.config.set_visible_devices([], 'TPU')
        except Exception as e:
            print('Could not disable TensorFlow devices:', e)
        if not self.config.prealloc:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        if 'XLA_PYTHON_CLIENT_MEM_FRACTION' not in os.environ:
            os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
        xla_flags = []
        if self.config.logical_cpus:
            count = self.config.logical_cpus
            xla_flags.append(f'--xla_force_host_platform_device_count={count}')
        if xla_flags:
            os.environ['XLA_FLAGS'] = ' '.join(xla_flags)
        jax.config.update('jax_platform_name', self.config.platform)
        jax.config.update('jax_disable_jit', not self.config.jit)
        jax.config.update('jax_debug_nans', self.config.debug_nans)
        jax.config.update('jax_transfer_guard', 'disallow')
        if self.config.platform == 'cpu':
            jax.config.update('jax_disable_most_optimizations', self.config.debug)
        jaxutils.COMPUTE_DTYPE = getattr(jnp, self.config.precision)

    def _transform(self):
        self._init_policy = nj.pure(lambda x: self.agent.policy_initial(len(x)))
        self._init_train = nj.pure(lambda x: self.agent.train_initial(len(x)))
        self._policy = nj.pure(self.agent.policy)
        self._train = nj.pure(self.agent.train)
        self._train_alt = nj.pure(self.agent.train_alt) if self.agent.use_train_alt else None
        self._train_both = nj.pure(self.agent.train_both) if self.agent.use_train_alt else None
        self._report = nj.pure(self.agent.report)
        self._report_non_compiled = (nj.pure(self.agent.report_non_compiled)
                                     if hasattr(self.agent, "report_non_compiled") else None)

        self._update_normalization_state = nj.pure(self.agent.update_normalization_state)
        self._update_normalization_state_cpu = nj.jit(self._update_normalization_state)

        if len(self.train_devices) == 1:
            kw = dict(device=self.train_devices[0])
            self._init_train = nj.jit(self._init_train, **kw)
            self._train = nj.jit(self._train, **kw)
            self._train_alt = nj.jit(self._train_alt, **kw) if self.agent.use_train_alt else None
            self._train_both = nj.jit(self._train_both, **kw) if self.agent.use_train_alt else None
            self._report = nj.jit(self._report, **kw)
            self._update_normalization_state = nj.jit(self._update_normalization_state, **kw)
        else:
            kw = dict(devices=self.train_devices)
            self._init_train = nj.pmap(self._init_train, 'i', **kw)
            self._train = nj.pmap(self._train, 'i', **kw)
            self._train_alt = nj.pmap(self._train_alt, 'i', **kw) if self.agent.use_train_alt else None
            self._train_both = nj.pmap(self._train_both, 'i', **kw) if self.agent.use_train_alt else None
            self._report = nj.pmap(self._report, 'i', **kw)
            self._update_normalization_state = nj.pmap(self._update_normalization_state, **kw)
        if len(self.policy_devices) == 1:
            kw = dict(device=self.policy_devices[0])
            self._init_policy = nj.jit(self._init_policy, **kw)
            self._policy = nj.jit(self._policy, static=['mode'], **kw)
        else:
            kw = dict(devices=self.policy_devices)
            self._init_policy = nj.pmap(self._init_policy, 'i', **kw)
            self._policy = nj.pmap(self._policy, 'i', static=['mode'], **kw)

    @staticmethod
    def _convert_inps(value, devices):
        if len(devices) == 1:
            value = jax.device_put(value, devices[0])
        else:
            check = tree_map(lambda x: len(x) % len(devices) == 0, value)
            if not all(jax.tree_util.tree_leaves(check)):
                shapes = tree_map(lambda x: x.shape, value)
                raise ValueError(
                    f'Batch must by divisible by {len(devices)} devices: {shapes}')
            # TODO: Avoid the reshape?
            value = tree_map(
                lambda x: x.reshape((len(devices), -1) + x.shape[1:]), value)
            shards = []
            for i in range(len(devices)):
                shards.append(tree_map(lambda x: x[i], value))
            value = jax.device_put_sharded(shards, devices)
        return value

    def _convert_outs(self, value, devices):
        value = jax.device_get(value)
        value = tree_map(np.asarray, value)
        if len(devices) > 1:
            value = tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), value)
        return value

    def _convert_mets(self, value, devices):
        value = jax.device_get(value)
        value = tree_map(np.asarray, value)
        if len(devices) > 1:
            value = tree_map(lambda x: x[0], value)
        return value

    def _next_rngs(self, devices, mirror=False, high=2 ** 63 - 1):
        if len(devices) == 1:
            return jax.device_put(self.rng.integers(high), devices[0])
        elif mirror:
            return jax.device_put_replicated(
                self.rng.integers(high), devices)
        else:
            return jax.device_put_sharded(
                list(self.rng.integers(high, size=len(devices))), devices)

    def _init_varibs(self, obs_space, act_space, augment_images=False):
        varibs = {}
        rng = self._next_rngs(self.train_devices, mirror=True)
        dims = (self.batch_size, self.batch_length)
        data = self._dummy_batch({**obs_space, **act_space}, dims, augment_images=augment_images)
        data = self._convert_inps(data, self.train_devices)
        state, varibs = self._init_train(varibs, rng, data['is_first'])
        varibs = self._train(varibs, rng, data, state, init_only=True)

        # obs = self._dummy_batch(obs_space, (1,))
        # obs = self._convert_inps(obs, self.policy_devices)
        # # state, varibs = self._init_policy(varibs, rng, obs['is_first'])

        # varibs = self._policy(
        #     varibs, rng, obs, state, mode='train', init_only=True)
        return varibs

    def _dummy_batch(self, spaces, batch_dims, augment_images=False):
        spaces = list(spaces.items())
        data = {k: np.zeros(v.shape, v.dtype) for k, v in spaces}
        if augment_images and "image" in data:
            data['original_image'] = data['image']
        for dim in reversed(batch_dims):
            data = {k: np.repeat(v[None], dim, axis=0) for k, v in data.items()}
        return data
