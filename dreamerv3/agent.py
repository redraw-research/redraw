from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
import ruamel.yaml as yaml

from dreamerv3 import embodied

tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

import logging

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return 'check_types' not in record.getMessage()


logger.addFilter(CheckTypesFilter())

from dreamerv3 import behaviors
from dreamerv3 import jaxagent
from dreamerv3 import jaxutils
from dreamerv3 import nets
from dreamerv3 import simdreamer_nets
from dreamerv3 import simdreamer_nets2
from dreamerv3 import ninjax as nj

from dreamerv3.embodied.acme.running_statistics import RunningStatisticsState, init_state, update, normalize, denormalize
from dreamerv3.embodied.acme import specs

RSSMType = Union[
    nets.RSSM,
    simdreamer_nets.GroundedRSSM,
    simdreamer_nets.GroundedNonRecurrentSSM,
    simdreamer_nets.GroundedLiteralSSM
]


def _normalize_grounded(batch, grounded_state_normalizer_params):
    if grounded_state_normalizer_params is not None:
        normalizer_params = grounded_state_normalizer_params.read()
        # jax.debug.print("normalize params count {} {}", normalizer_params.count, normalizer_params)

        if not isinstance(batch, dict):
            return normalize(batch=batch, mean_std=normalizer_params)
        out = {}
        for k, v in batch.items():
            if k in ('gt_state', 'pred_state_gt_state', 'vector_obs', 'pred_state_sim_next_gt_state'):
                out[k] = normalize(batch=v, mean_std=normalizer_params)
            elif k in ('pred_state_symlog', 'pred_state_sim_next_symlog_state'):
                out[k] = jaxutils.symlog(normalize(batch=jaxutils.symexp(v), mean_std=normalizer_params))
            elif k == 'symlog_grounded':
                assert False, "We should never need to normalize symlog_grounded"
            elif 'grounded' in k or 'symlog' in k:
                raise ValueError(f"Unexpected key: {k}")
            elif isinstance(v, dict):
                out[k] = _normalize_grounded(batch=v, grounded_state_normalizer_params=grounded_state_normalizer_params)
            elif isinstance(v, tuple):
                out[k] = (_normalize_grounded(e, grounded_state_normalizer_params) for e in v)
            else:
                out[k] = v
        return out
    else:
        return batch

def _normalize_image_complement(batch, image_complement_normalizer_params):
    if image_complement_normalizer_params is not None:
        normalizer_params = image_complement_normalizer_params.read()

        # jax.debug.print("normalizer_params params count {} {}", normalizer_params.count, normalizer_params)

        if not isinstance(batch, dict):
            return normalize(batch=batch, mean_std=normalizer_params)
        out = {}
        for k, v in batch.items():
            if k == 'image_complement':
                out[k] = normalize(batch=v, mean_std=normalizer_params)
            elif isinstance(v, dict):
                out[k] = _normalize_image_complement(batch=v, image_complement_normalizer_params=image_complement_normalizer_params)
            elif isinstance(v, tuple):
                out[k] = (_normalize_image_complement(e, image_complement_normalizer_params) for e in v)
            else:
                out[k] = v
        return out
    else:
        return batch



def _denormalize_grounded(batch, grounded_state_normalizer_params):
    if grounded_state_normalizer_params is not None:
        normalizer_params = grounded_state_normalizer_params.read()
        # jax.debug.print("denormalize params count {} {}", normalizer_params.count, normalizer_params)

        if not isinstance(batch, dict):
            return denormalize(batch=batch, mean_std=normalizer_params)
        out = {}
        for k, v in batch.items():
            if k == 'gt_state':
                out[k] = denormalize(batch=v, mean_std=normalizer_params)
            elif k == 'symlog_grounded':
                out[k] = jaxutils.symlog(denormalize(batch=jaxutils.symexp(v), mean_std=normalizer_params))
            elif k in ('pred_state_sim_next_symlog_state', 'pred_state_symlog', 'pred_state_sim_next_gt_state'):
                assert False, f"We should never need to denormalize {k}"
            elif k == 'curr_state_symlog_sim_dynamics_pred':
                continue  # can omit this from any wm outputs
            elif 'loss' in k:
                out[k] = v
            elif 'grounded' in k or 'symlog' in k:
                raise ValueError(f"Unexpected key: {k}")
            elif isinstance(v, dict):
                out[k] = _denormalize_grounded(batch=v, grounded_state_normalizer_params=grounded_state_normalizer_params)
            elif isinstance(v, tuple):
                out[k] = (_denormalize_grounded(e, grounded_state_normalizer_params) for e in v)
            else:
                out[k] = v
        return out
    else:
        return batch

def _denormalize_image_complement(batch, image_complement_normalizer_params):
    if image_complement_normalizer_params is not None:
        normalizer_params = image_complement_normalizer_params.read()
        if not isinstance(batch, dict):
            return denormalize(batch=batch, mean_std=normalizer_params)
        out = {}
        for k, v in batch.items():
            if k == 'image_complement':
                out[k] = denormalize(batch=v, mean_std=normalizer_params)
            elif isinstance(v, dict):
                out[k] = _denormalize_image_complement(batch=v, image_complement_normalizer_params=image_complement_normalizer_params)
            elif isinstance(v, tuple):
                out[k] = (_denormalize_image_complement(e, image_complement_normalizer_params) for e in v)
            else:
                out[k] = v
        return out
    else:
        return batch


@jaxagent.Wrapper
class Agent(nj.Module):
    configs = yaml.YAML(typ='safe').load(
        (embodied.Path(__file__).parent / 'configs.yaml').read())

    def __init__(self, obs_space, act_space, step, config, sim_query_env=None):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space['action']
        self.step = step

        assert not (config.normalize_all_grounded_states and config.normalize_agent_grounded_input) # can only use one or the other
        assert not (config.normalize_image_complement and config.normalize_agent_grounded_input)
        assert not (config.normalize_image_complement and config.normalize_all_grounded_states)
        if config.normalize_all_grounded_states or config.normalize_agent_grounded_input:
            grounded_size = obs_space['gt_state'].shape[0]
            self.grounded_state_normalizer_params: RunningStatisticsState = nj.Variable(ctor=lambda: init_state(specs.Array((grounded_size,), jnp.float64)), name="obs_normalizer_params")
        else:
            self.grounded_state_normalizer_params = None

        if config.normalize_image_complement:
            image_complement_size = obs_space['image_complement'].shape[0]
            self.image_complement_normalizer_params: RunningStatisticsState = nj.Variable(ctor=lambda: init_state(specs.Array((image_complement_size,), jnp.float64)), name="image_complement_normalizer_params")
        else:
            self.image_complement_normalizer_params = None

        self.wm = WorldModel(obs_space, act_space, config, sim_query_env, name='wm',
                             normalize_fn=None,
                             denormalize_fn=None)

        # self.wm = WorldModel(obs_space, act_space, config, sim_query_env, name='wm',
        #                      normalize_fn=lambda batch: _normalize_grounded(batch, self.grounded_state_normalizer_params),
        #                      denormalize_fn=lambda batch: _denormalize_grounded(batch, self.grounded_state_normalizer_params))

        self.task_behavior = getattr(behaviors, config.task_behavior)(
            self.wm, self.act_space, self.config,
            grounded_state_normalizer_params=self.grounded_state_normalizer_params if config.normalize_agent_grounded_input else None,
            name='task_behavior')
        if config.expl_behavior == 'None':
            self.expl_behavior = self.task_behavior
        else:
            self.expl_behavior = getattr(behaviors, config.expl_behavior)(
                self.wm, self.act_space, self.config,
                grounded_state_normalizer_params=self.grounded_state_normalizer_params if config.normalize_agent_grounded_input else None,
                name='expl_behavior')

    def update_normalization_state(self, batch: dict):
        if self.config.normalize_all_grounded_states:
            self.grounded_state_normalizer_params.write(update(self.grounded_state_normalizer_params.read(), batch=batch['gt_state']))
        elif self.config.normalize_agent_grounded_input:
            self.task_behavior.ac.update_normalization_state(batch=batch)

        if self.config.normalize_image_complement:
            self.image_complement_normalizer_params.write(update(self.image_complement_normalizer_params.read(), batch=batch['image_complement']))

            # normalizer_params = self.image_complement_normalizer_params.read()
            # jax.debug.print("updated params count {} {}", normalizer_params.count, normalizer_params)

    def policy_initial(self, batch_size):
        return (
            self.wm.initial(batch_size),
            self.task_behavior.initial(batch_size),
            self.expl_behavior.initial(batch_size))

    def train_initial(self, batch_size):
        return self.wm.initial(batch_size)

    def policy(self, obs, state, mode='train'):
        self.config.jax.jit and print('Tracing policy function.')
        # if mode != 'eval':
        #     jax.debug.print(f"updating normalization state, mode: " + mode + ": {}", obs['gt_state'])
        #     self.update_normalization_state(batch=obs)
        obs = self.preprocess(obs)
        (prev_latent, prev_action), task_state, expl_state = state
        embed = self.wm.encoder(obs)
        latent, _ = self.wm.rssm.obs_step(
            prev_latent, prev_action, embed, obs['is_first'], with_residual=True, with_student_posterior=True)

        # (author1): commented this out. I think this duplicate call is superfluous?
        # At least one other person thinks so https://github.com/danijar/dreamerv3/pull/28
        # self.expl_behavior.policy(latent, expl_state)

        task_outs, task_state = self.task_behavior.policy(latent, task_state)
        expl_outs, expl_state = self.expl_behavior.policy(latent, expl_state)
        if mode == 'eval':
            outs = task_outs
            outs['action'] = outs['action'].sample(seed=nj.rng())
            outs['log_entropy'] = jnp.zeros(outs['action'].shape[:1])
        elif mode == 'explore':
            outs = expl_outs
            outs['log_entropy'] = outs['action'].entropy()
            outs['action'] = outs['action'].sample(seed=nj.rng())
        elif mode == 'train':
            outs = task_outs
            outs['log_entropy'] = outs['action'].entropy()
            outs['action'] = outs['action'].sample(seed=nj.rng())
        elif mode == 'random':
            raise NotImplementedError("This is implemented incorrectly, always returns zero or some constant action.")
            batch_size = len(next(iter(obs.values())))
            outs = {"action": np.stack([self.act_space.sample() for _ in range(batch_size)])}
        state = ((latent, outs['action']), task_state, expl_state)
        return outs, state

    def train(self, data, state):
        self.config.jax.jit and print('Tracing train function.')
        metrics = {}
        data = self.preprocess(data, add_is_valid_set_to_true=True)
        state, wm_outs, mets = self.wm.train(data, state)
        metrics.update(mets)
        # context = {**data, **wm_outs['post']}
        context = {**data, **wm_outs['post'], 'embed': wm_outs['embed']}  # to allow embded to be a plan2explore disag target
        start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)

        #TODO, if theres a residual, filer start and contexts to only features real rows
        if hasattr(self.wm.rssm, 'residual') and self.wm.rssm.residual is not None:
            print(f"context is_real shape: {context['is_real'].shape}") # batch size, batch len (512, 2)
            print(f"start is_real shape: {start['is_real'].shape}") # batch_size x batch len (use all states without ordering)  (1024,)
            # jax.debug.print("start['is_real'].sum() before prune: {x} shape: {y} ", x=start['is_real'].sum(), y=start['is_real'].shape)
            # only use real states as actor-critic imagine starting points
            real_indices = jnp.nonzero(start['is_real'],
                                       size=(self.config.batch_size * self.config.batch_length) // 3,
                                       fill_value=jnp.nan)
            print(f"real_indices: {real_indices}")
            assert len(real_indices) == 1, real_indices
            real_indices = real_indices[0].astype(dtype=jnp.int32)
            start = tree_map(lambda x: jax.numpy.take(x,
                                                     indices=real_indices,
                                                     axis=0), start)
            print(f"start is_real pruned shape: {start['is_real'].shape}") # batch_size x batch len (use all states without ordering)


        task_behavior_traj, mets = self.task_behavior.train(self.wm.imagine, start, context)
        metrics.update(mets)
        if self.config.expl_behavior != 'None':
            expl_behavior_traj, mets = self.expl_behavior.train(self.wm.imagine_no_residual, start, context)
            metrics.update({'expl_' + key: value for key, value in mets.items()})
        else:
            expl_behavior_traj = None

        outs = {
            "wm_outs": wm_outs,
            "task_behavior_traj": task_behavior_traj,
            "expl_behavior_traj": expl_behavior_traj
        }

        if self.config.normalize_all_grounded_states:
            outs = _denormalize_grounded(batch=outs, grounded_state_normalizer_params=self.grounded_state_normalizer_params)

        if self.config.normalize_image_complement:
            outs = _denormalize_image_complement(batch=outs, image_complement_normalizer_params=self.image_complement_normalizer_params)


        return outs, state, metrics

    @property
    def use_train_alt(self):
        return self.config.use_grounded_rssm

    def train_alt(self, experience_data, pred_state_data, state_stub):
        self.config.jax.jit and print('Tracing train alt function.')
        metrics = {}
        if not self.config.freeze_wm:
            experience_data = self.preprocess(experience_data, add_is_valid_set_to_true=True) if experience_data else None
            pred_state_data = self.preprocess(pred_state_data) if pred_state_data else None
            wm_outs, mets = self.wm.train_alt(experience_data, pred_state_data, state_stub)
            metrics.update(mets)
        outs = {}
        return outs, metrics

    def train_both(self, train_data, train_alt_experience_data, train_alt_pred_state_data, train_state=None):
        train_outs, state_out, train_metrics = self.train(data=train_data, state=train_state)
        _, train_alt_metrics = self.train_alt(experience_data=train_alt_experience_data,
                                               pred_state_data=train_alt_pred_state_data,
                                               state_stub=train_state)
        return train_outs, state_out, train_metrics, train_alt_metrics

    def report(self, data, train_alt_data):
        self.config.jax.jit and print('Tracing report function.')
        data = self.preprocess(data, add_is_valid_set_to_true=True)
        if train_alt_data:
            train_alt_data = self.preprocess(train_alt_data)
        report = {}
        report.update(self.wm.report(data, train_alt_data))
        mets = self.task_behavior.report(data)
        report.update({f'task_{k}': v for k, v in mets.items()})
        if self.expl_behavior is not self.task_behavior:
            mets = self.expl_behavior.report(data)
            report.update({f'expl_{k}': v for k, v in mets.items()})

        if self.grounded_state_normalizer_params is not None:
            normalizer_params = self.grounded_state_normalizer_params.read()
            report.update({
                'obs_normalizer count': normalizer_params.count,
            })
            jax.debug.print("obs_normalizer std: {}", normalizer_params.std)
            jax.debug.print("obs_normalizer mean: {}", normalizer_params.mean)
            jax.debug.print("obs_normalizer summed_variance: {}", normalizer_params.summed_variance)

        return report

    def report_non_compiled(self, data):
        with jax.transfer_guard("allow"):
            data = self.preprocess(data)
            return self.wm.report_non_compiled(data)

    def preprocess(self, obs, add_is_valid_set_to_true=False):
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith('log_') or key in ('key',):
                continue
            if len(value.shape) > 3 and value.dtype == jnp.uint8:
                print(f"normalizing obs key {key} as an image")
                value = jaxutils.cast_to_compute(value) / 255.0
            else:
                value = value.astype(jnp.float32)
            obs[key] = value
        if 'is_terminal' in obs:
            obs['cont'] = 1.0 - obs['is_terminal'].astype(jnp.float32)
        if 'pred_state_is_terminal' in obs:
            obs['pred_state_cont'] = 1.0 - obs['pred_state_is_terminal'].astype(jnp.float32)
        if 'pred_state_sim_next_is_terminal' in obs:
            obs['pred_state_sim_next_cont'] = 1.0 - obs['pred_state_sim_next_is_terminal'].astype(jnp.float32)
        if add_is_valid_set_to_true:
            obs['is_valid'] = jnp.ones_like(obs['is_first'])
        if self.config.normalize_all_grounded_states:
            obs = _normalize_grounded(batch=obs, grounded_state_normalizer_params=self.grounded_state_normalizer_params)
        if self.config.normalize_image_complement:
            obs = _normalize_image_complement(batch=obs, image_complement_normalizer_params=self.image_complement_normalizer_params)

        return obs


class WorldModel(nj.Module):

    def __init__(self, obs_space, act_space, config, sim_query_env=None, normalize_fn=None, denormalize_fn=None):
        self.sim_query_env = sim_query_env
        self.obs_space = obs_space
        self.act_space = act_space['action']
        self.config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        shapes = {k: v for k, v in shapes.items() if not k.startswith(('log_', 'gt_state'))}
        if self.config.encoder_outputs_grounded_symlog:
            self.encoder = nets.MultiEncoderWithGroundedSymlogHead(
                grounded_size=obs_space['gt_state'].shape[0], shapes=shapes, **config.encoder, name='enc')
        elif self.config.encoder_is_identity_function:
            self.encoder = nets.MultiEncoderIdentityFunctionWithSymlog(shapes, **config.encoder, name='enc')
        else:
            self.encoder = nets.MultiEncoder(shapes, **config.encoder, name='enc')
        if config.use_heads_from_vanilla_dreamer:
            dims = 'symlog_grounded' if self.config.use_grounded_rssm else 'deter'
            self.heads = {
                'decoder': nets.MultiDecoder(shapes, **config.decoder, name='dec', dims=dims),
                'reward': nets.MLP((), **config.reward_head, name='rew', dims=dims),
                'cont': nets.MLP((), **config.cont_head, name='cont', dims=dims)}
        else:
            self.heads = {}

        self.opt = jaxutils.Optimizer(name='model_opt', **config.model_opt)

        if config.use_grounded_rssm:
            if self.sim_query_env is None:
                raise ValueError(f"A sim_query_env needs to be provided if using a grounded rssm")
            if len(obs_space['gt_state'].shape) != 1:
                raise ValueError(f"The shape of obs_space['gt_state'] needs to be 1-dimensional, "
                                 f"but got shape {obs_space['gt_state'].shape}")
            grounded_state_size = obs_space['gt_state'].shape[0]
            self.sim_tf = nets.MLP(shape=grounded_state_size, **config.sim_tf,
                                   name='sim_tf') if config.use_sim_forward_dynamics else None
            if self.sim_tf:
                print(f"\n\n\nsim_tf has {self.sim_tf._layers} layers and {self.sim_tf._units} units\n\n\n")

            if config.grounded_rssm_is_non_recurrent:
                grounded_rssm_config = {**config.grounded_rssm}
                del grounded_rssm_config['deter']
                self.rssm: RSSMType = simdreamer_nets.GroundedNonRecurrentSSM(
                    **grounded_rssm_config,
                    grounded=grounded_state_size,
                    sim_tf=self.sim_tf,
                    name='rssm')
            elif config.grounded_rssm_replaces_h_with_s:
                grounded_rssm_config = {**config.grounded_rssm}
                del grounded_rssm_config['deter']
                self.rssm: RSSMType = simdreamer_nets.GroundedRSSMReplaceHwithS(
                    **grounded_rssm_config,
                    grounded=grounded_state_size,
                    sim_tf=self.sim_tf,
                    name='rssm')
            elif config.grounded_rssm_replaces_h_with_s_more_features:
                self.rssm: RSSMType = simdreamer_nets.GroundedRSSMReplaceHwithSMoreFeatures(
                    **config.grounded_rssm,
                    grounded=grounded_state_size,
                    sim_tf=self.sim_tf,
                    name='rssm')
            elif config.grounded_rssm_replaces_gru:
                self.rssm: RSSMType = simdreamer_nets.GroundedRSSMReplaceGRU(
                    **config.grounded_rssm,
                    grounded=grounded_state_size,
                    sim_tf=self.sim_tf,
                    name='rssm')
            elif config.grounded_rssm_replaces_gru2:
                self.rssm: RSSMType = simdreamer_nets.GroundedRSSMReplaceGRU2(
                    **config.grounded_rssm,
                    grounded=grounded_state_size,
                    sim_tf=self.sim_tf,
                    name='rssm')
            elif config.grounded_rssm_replaces_gru2_more_layers_to_get_symlog_grounded:
                self.rssm: RSSMType = simdreamer_nets.GroundedRSSMReplaceGRU2MoreLayersToGetSymlogGrounded(
                    **config.grounded_rssm,
                    grounded=grounded_state_size,
                    sim_tf=self.sim_tf,
                    name='rssm')
            elif config.grounded_rssm_replaces_gru3_more_layers_to_get_symlog_grounded:
                self.rssm: RSSMType = simdreamer_nets.GroundedRSSMReplaceGRU3MoreLayersToGetSymlogGrounded(
                    **config.grounded_rssm,
                    grounded=grounded_state_size,
                    sim_tf=self.sim_tf,
                    name='rssm')
            elif config.grounded_rssm_replaces_gru3:
                self.rssm: RSSMType = simdreamer_nets.GroundedRSSMReplaceGRU3(
                    **config.grounded_rssm,
                    grounded=grounded_state_size,
                    sim_tf=self.sim_tf,
                    name='rssm')
            elif config.grounded_rssm_is_non_recurrent2:
                grounded_rssm_config = {**config.grounded_rssm}
                del grounded_rssm_config['deter']
                self.rssm: RSSMType = simdreamer_nets.GroundedNonRecurrentSSM2(
                    **grounded_rssm_config,
                    grounded=grounded_state_size,
                    sim_tf=self.sim_tf,
                    name='rssm')
            elif config.grounded_rssm_is_non_recurrent3:
                grounded_rssm_config = {**config.grounded_rssm}
                del grounded_rssm_config['deter']
                self.rssm: RSSMType = simdreamer_nets.GroundedNonRecurrentSSM3(
                    **grounded_rssm_config,
                    grounded=grounded_state_size,
                    sim_tf=self.sim_tf,
                    name='rssm')
            elif config.grounded_rssm_is_non_recurrent4:
                grounded_rssm_config = {**config.grounded_rssm}
                del grounded_rssm_config['deter']
                self.rssm: RSSMType = simdreamer_nets.GroundedNonRecurrentSSM4(
                    **grounded_rssm_config,
                    grounded=grounded_state_size,
                    sim_tf=self.sim_tf,
                    name='rssm')
            elif config.grounded_rssm_is_literal:
                grounded_rssm_config = {**config.grounded_rssm}
                del grounded_rssm_config['deter']
                del grounded_rssm_config['stoch']
                del grounded_rssm_config['classes']
                del grounded_rssm_config['unimix']
                self.rssm: RSSMType = simdreamer_nets.GroundedLiteralSSM(
                    **grounded_rssm_config,
                    grounded=grounded_state_size,
                    sim_tf=self.sim_tf,
                    name='rssm')
            elif config.grounded_rssm_is_sim_only:
                grounded_rssm_config = {**config.grounded_rssm}
                del grounded_rssm_config['deter']
                del grounded_rssm_config['stoch']
                del grounded_rssm_config['classes']
                del grounded_rssm_config['unimix']
                self.rssm: RSSMType = simdreamer_nets.GroundedLiteralSimOnlySSM(
                    **grounded_rssm_config,
                    grounded=grounded_state_size,
                    sim_tf=self.sim_tf,
                    name='rssm')
            elif config.grounded_rssm_is_gt_sim_only:
                grounded_rssm_config = {**config.grounded_rssm}
                del grounded_rssm_config['deter']
                del grounded_rssm_config['stoch']
                del grounded_rssm_config['classes']
                del grounded_rssm_config['unimix']
                self.rssm: RSSMType = simdreamer_nets.GroundedLiteralGTSimOnlySSM(
                    **grounded_rssm_config,
                    grounded=grounded_state_size,
                    sim=self.sim_query_env,
                    normalize_fn=normalize_fn,
                    denormalize_fn=denormalize_fn,
                    name='rssm')
            elif config.grounded_rssm_s_from_z_only:
                self.rssm: RSSMType = simdreamer_nets.GroundedRSSMGroundedFromStochOnly(
                    **config.grounded_rssm,
                    grounded=grounded_state_size,
                    sim_tf=self.sim_tf,
                    name='rssm')
            elif config.grounded_rssm_s_from_z_only_orig_size_network:
                self.rssm: RSSMType = simdreamer_nets.GroundedRSSMGroundedFromStochOnlyOrigSizeNetwork(
                    **config.grounded_rssm,
                    grounded=grounded_state_size,
                    sim_tf=self.sim_tf,
                    name='rssm')
            elif config.grounded_rssm_s_from_z_non_sequential_posterior:
                self.rssm: RSSMType = simdreamer_nets.GroundedRSSMGroundedFromStochOnlyNonSequentialPosterior(
                    **config.grounded_rssm,
                    grounded=grounded_state_size,
                    sim_tf=self.sim_tf,
                    name='rssm')
            elif config.grounded_rssm_s_from_z_non_sequential_posterior_no_action:
                self.rssm: RSSMType = simdreamer_nets.GroundedRSSMGroundedFromStochOnlyNonSequentialPosteriorNoAction(
                    **config.grounded_rssm,
                    grounded=grounded_state_size,
                    sim_tf=self.sim_tf,
                    name='rssm')
            elif config.grounded_rssm_s_from_z_non_sequential_posterior_mlp_prior:
                self.rssm: RSSMType = simdreamer_nets.GroundedRSSMGroundedFromStochOnlyNonSequentialPosteriorMLPPrior(
                    **config.grounded_rssm,
                    grounded=grounded_state_size,
                    sim_tf=self.sim_tf,
                    name='rssm')
            elif config.grounded_rssm_s_from_z_non_sequential_posterior_mlp_prior_larger:
                self.rssm: RSSMType = simdreamer_nets.GroundedRSSMGroundedFromStochOnlyNonSequentialPosteriorMLPPriorLarger(
                    **config.grounded_rssm,
                    grounded=grounded_state_size,
                    sim_tf=self.sim_tf,
                    name='rssm')
            elif config.grounded_rssm_s_from_z_non_sequential_posterior_post_produces_h:
                self.rssm: RSSMType = simdreamer_nets.GroundedRSSMGroundedFromStochOnlyNonSequentialPosteriorPostProducesH(
                    **config.grounded_rssm,
                    grounded=grounded_state_size,
                    sim_tf=self.sim_tf,
                    name='rssm')
            elif config.grounded_rssm_s_from_z_non_sequential_posterior_post_produces_h2:
                self.rssm: RSSMType = simdreamer_nets.GroundedRSSMGroundedFromStochOnlyNonSequentialPosteriorPostProducesH2(
                    **config.grounded_rssm,
                    grounded=grounded_state_size,
                    sim_tf=self.sim_tf,
                    name='rssm')
            elif config.grounded_rssm_larger_get_grounded:
                self.rssm: RSSMType = simdreamer_nets.GroundedRSSMLargerGetGrounded(
                    **config.grounded_rssm,
                    grounded=grounded_state_size,
                    sim_tf=self.sim_tf,
                    name='rssm')
            elif config.grounded_rssm_stoch_only_larger_mlp_prior_with_prior_belief:
                self.rssm: RSSMType = simdreamer_nets.GroundedRSSMStochOnlyNonSequentialPosteriorMLPPriorLargerWithPriorBelief(
                    **config.grounded_rssm,
                    grounded=grounded_state_size,
                    name='rssm')
            elif config.grounded_rssm_stoch_only_larger_mlp_prior_with_prior_belief_and_residual:
                self.rssm: RSSMType = simdreamer_nets2.GroundedRSSMStochOnlyNonSequentialPosteriorMLPPriorLargerWithPriorBeliefAndResidual(
                    **config.grounded_rssm,
                    grounded=grounded_state_size,
                    name='rssm')
            else:
                self.rssm: RSSMType = simdreamer_nets.GroundedRSSM(
                    **config.grounded_rssm,
                    grounded=grounded_state_size,
                    sim_tf=self.sim_tf,
                    name='rssm')
            self.grounded_heads = {
                "grounded_decoder": nets.MultiDecoder(shapes, **config.grounded_decoder, name='dec_grounded', dims='symlog_grounded'),
                'grounded_reward': nets.MLP((), **config.grounded_reward_head, name='rew_grounded', dims='symlog_grounded'),
                'grounded_cont': nets.MLP((), **config.grounded_cont_head, name='cont_grounded', dims='symlog_grounded')
            }
            if config.grounded_heads_include_is_valid:
                self.grounded_heads["grounded_is_valid"] = nets.MLP((), **config.grounded_is_valid_head,
                                                                    name='is_valid_grounded')

            self.black_box_opt = jaxutils.Optimizer(name='black_box_model_opt', **config.bb_model_opt)
        else:
            if config.rssm_stoch_only_larger_mlp_prior_with_prior_belief:
                self.rssm: RSSMType = nets.RSSMStochOnlyNonSequentialPosteriorMLPPriorLargerWithPriorBelief(**config.rssm, name='rssm')
            # elif config.rssm_stoch_only_larger_mlp_prior:
            #     self.rssm: RSSMType = nets.RSSMStochOnlyNonSequentialPosteriorMLPPriorLarger(**config.rssm, name='rssm')
            # elif config.rssm_stoch_only_larger_mlp_prior_with_post_belief:
            #     self.rssm: RSSMType = nets.RSSMStochOnlyNonSequentialPosteriorMLPPriorLargerWithPostBelief(**config.rssm, name='rssm')
            # elif config.deterministic_z_only_dreamer_model:
            #     self.rssm: RSSMType = nets.DeterministicZOnlyDreamerModel(**config.rssm, name='rssm')
            # elif config.rssm_regularized_deter:
            #     self.rssm: RSSMType = nets.RSSMRegularizedDeter(**config.rssm, name='rssm')
            # elif config.rssm_regularized_deter_no_unimix_on_deter:
            #     self.rssm: RSSMType = nets.RSSMRegularizedDeterNoUnimixOnDeter(**config.rssm, name='rssm')
            elif config.td_deterministic_model:
                self.rssm: RSSMType = nets.TDDeterministicZOnlyDreamerModel(**config.rssm, name='rssm')
            elif config.td_deterministic_model_small:
                self.rssm: RSSMType = nets.TDDeterministicZOnlyDreamerModelSmall(**config.rssm, name='rssm')
            elif config.td_stochastic_model:
                self.rssm: RSSMType = nets.TDStochasticZOnlyDreamerModel(**config.rssm, name='rssm')
            elif config.td_compressed_stochastic_model:
                self.rssm: RSSMType = nets.TDCompressedStochasticZOnlyDreamerModel(**config.rssm, name='rssm')
            elif config.td_dummy_stochastic_model:
                self.rssm: RSSMType = nets.TDDummyStochasticZOnlyDreamerModel(**config.rssm, name='rssm')
            else:
                self.rssm: RSSMType = nets.RSSM(**config.rssm, name='rssm')
            self.sim_tf = None
            self.grounded_heads = None
            self.black_box_opt = None

        scales = self.config.loss_scales.copy()
        image, vector = scales.pop('image'), scales.pop('vector')

        if config.use_heads_from_vanilla_dreamer:
            scales.update({k: image for k in self.heads["decoder"].cnn_shapes})
            scales.update({k: vector for k in self.heads["decoder"].mlp_shapes})
        else:
            assert config.use_grounded_rssm
            scales.update({k: image for k in self.grounded_heads["grounded_decoder"].cnn_shapes})
            scales.update({k: vector for k in self.grounded_heads["grounded_decoder"].mlp_shapes})
        if config.use_sim_forward_dynamics:
            scales["sim_tf_on_experience"] = 1.0
            scales["sim_tf_on_pred_state"] = 1.0

        self.scales = scales

    def initial(self, batch_size):
        prev_latent = self.rssm.initial(batch_size)
        prev_action = jnp.zeros((batch_size, *self.act_space.shape))
        return prev_latent, prev_action

    def train(self, data, state):
        modules = []
        skip_keys_pattern = None

        if not self.config.freeze_wm:

            modules = [self.rssm]
            if self.config.freeze_posterior:
                skip_keys_pattern = r'^agent\/wm\/rssm\/obs_.*$'  # any string that starts with "agent/wm/rssm/obs_"

            if not self.config.freeze_encoder and not self.config.optimize_encoder_in_train_alt:
                modules.append(self.encoder)
            if self.sim_tf and self.config.optimize_sim_tf_with_world_model_in_train:
                modules.append(self.sim_tf)
            if self.heads and not self.config.freeze_vanilla_heads:
                modules.extend(self.heads.values())
        if hasattr(self.rssm, "residual") and self.rssm.residual is not None:
            modules.append(self.rssm.residual)

        if self.config.normal_wm_loss_for_residual_only:
            assert len(modules) == 1 and hasattr(self.rssm, "residual") and self.rssm.residual in modules, modules
            mets, (state, outs, metrics) = self.opt(
                modules, self.normal_wm_loss_for_residual_only, data, state, has_aux=True)
            print(f"optimizing modules in train: {modules}")
            print(f"optimizing normal_wm_loss_for_residual_only")
        elif len(modules) == 1 and hasattr(self.rssm, "residual") and self.rssm.residual in modules:
            assert not self.config.normal_wm_loss_for_residual_only
            modules = [self.rssm.residual]
            mets, (state, outs, metrics) = self.opt(
                modules, self.residual_loss_only, data, state, has_aux=True)
            print(f"optimizing modules in train: {modules}")
            print(f"optimizing residual loss only")
            metrics.update(mets)
        elif len(modules) > 0:
            mets, (state, outs, metrics) = self.opt(
                modules, self.loss, data, state, has_aux=True, skip_keys_pattern=skip_keys_pattern)
            print(f"optimizing modules in train: {modules}")
            metrics.update(mets)
        else:
            print(f"Training no WM modules in train().")
            embed = self.encoder(data)
            prev_latent, prev_action = state
            prev_actions = jnp.concatenate([
                prev_action[:, None], data['action'][:, :-1]], 1)
            post, prior = self.rssm.observe(
                embed, prev_actions, data['is_first'], prev_latent)
            outs = {'embed': sg(embed), 'post': sg(post), 'prior': sg(prior)}
            metrics = {}

        print(f"available rssm ninjax submodules: {self.rssm.getm()}")
        print(f"rssm submodules: {self.rssm._submodules}")

        return state, outs, metrics

    def train_alt(self, experience_data, pred_state_data, state_stub):
        modules = []
        if self.grounded_heads and not self.config.freeze_grounded_heads:
            modules.extend(self.grounded_heads.values())
        if self.sim_tf and not self.config.freeze_sim_tf and not self.config.optimize_sim_tf_with_world_model_in_train:
            modules.append(self.sim_tf)

        if (self.config.optimize_encoder_in_train_alt or self.config.optimize_rssm_in_train_alt) and not self.config.freeze_encoder:
            modules.append(self.encoder)
        if self.config.optimize_rssm_in_train_alt:
            modules.append(self.rssm)
        assert len(modules) > 0
        print(f"Train Alt modules: {modules}")

        mets, (outs, metrics) = self.black_box_opt(
            modules, self.black_box_loss, experience_data, pred_state_data, state_stub, has_aux=True)
        metrics.update(mets)
        return outs, metrics

    @property
    def is_td(self) -> bool:
        return any([self.config.td_deterministic_model,
                    self.config.td_deterministic_model_small,
                    self.config.td_stochastic_model,
                    self.config.td_compressed_stochastic_model,
                    self.config.td_dummy_stochastic_model])

    def residual_loss_only(self, data, state):
        embed = self.encoder(data)
        if self.config.freeze_encoder:
            embed = sg(embed)
        prev_latent, prev_action = state
        prev_actions = jnp.concatenate([
            prev_action[:, None], data['action'][:, :-1]], 1)

        losses = {}

        if self.is_td:
            data["t"] = jnp.repeat(jnp.arange(start=0, stop=self.config.batch_length)[jnp.newaxis, :], repeats=self.config.batch_size, axis=0)

            rssm: nets.TDDeterministicZOnlyDreamerModel = self.rssm
            head_input_states, dynamics_preds, dynamics_targets, obs_encoder_predictions = rssm.observe_td(
                embed, prev_actions, data['is_first'], prev_latent, with_residual=True, with_res_stop_gradients=True)
            post = dynamics_targets
            prior = dynamics_preds
            losses['residual_loss'] = self.rssm.residual.residual_loss(prior, post, data['is_first'], data["t"], self.config.td_loss_rho, self.config.dyn_loss.free)
        else:
            post, prior = self.rssm.observe(
                embed, prev_actions, data['is_first'], prev_latent, with_residual=True, with_res_stop_gradients=True, with_student_posterior=True)
            losses['residual_loss'] = self.rssm.residual.residual_loss(prior, post, data['is_first'], self.config.dyn_loss.free)
        scaled = losses
        model_loss = sum(scaled.values())
        out = {'embed': embed, 'post': post, 'prior': prior}
        out.update({f'{k}_loss': v for k, v in losses.items()})
        last_latent = {k: v[:, -1] for k, v in post.items()}
        last_action = data['action'][:, -1]
        state = last_latent, last_action
        metrics = {
            'residual_loss_mean': losses['residual_loss'].mean(),
            'residual_loss_std': losses['residual_loss'].std()

        }
        loss_out = model_loss.mean()
        return loss_out, (state, out, metrics)

    def normal_wm_loss_for_residual_only(self, data, state):
        if self.is_td:
            raise NotImplementedError()
        if self.config.also_apply_head_losses_from_priors:
            raise NotImplementedError()

        embed = self.encoder(data)
        if self.config.freeze_encoder:
            embed = sg(embed)
        prev_latent, prev_action = state
        prev_actions = jnp.concatenate([
            prev_action[:, None], data['action'][:, :-1]], 1)
        post, prior = self.rssm.observe(
            embed, prev_actions, data['is_first'], prev_latent, with_residual=True, with_res_stop_gradients=True)

        dists = {}
        measure_only_dists = {}
        feats = {**post, 'embed': embed}
        heads = {**self.heads, **self.grounded_heads} if self.grounded_heads else {**self.heads}

        for name, head in heads.items():
            out = head(feats if name in self.config.grad_heads else sg(feats))
            out = out if isinstance(out, dict) else {name: out}
            if name == "grounded_decoder":
                # The grounded decoder and normal decoder both decode items from our experience,
                # so make sure they have different loss keys for the same items.
                out = {f"grounded_{k}": v for k, v in out.items()}

            if name.startswith("grounded") and name not in self.config.grad_heads:
                measure_only_dists.update(out)
            else:
                dists.update(out)

        losses = {}
        measure_only_losses = {}  # losses we aren't optimizing

        losses['dyn'] = self.rssm.dyn_loss(post, prior, data['is_first'], **self.config.dyn_loss)
        losses['rep'] = self.rssm.rep_loss(post, prior, data['is_first'], **self.config.rep_loss)

        if self.config.use_grounded_rssm and not self.config.optimize_sup_grounded_loss_in_train_alt:
            if 'gt_state' in data:
                grounded_state_prediction_loss = self.rssm.grounded_state_loss(post=post, gt_state=data['gt_state'],
                                                                               is_real=data['is_real'])
                if self.config.supervise_grounded_state:
                    losses['supervise_grounded_state'] = grounded_state_prediction_loss
                else:
                    measure_only_losses['supervise_grounded_state'] = grounded_state_prediction_loss
            elif self.config.supervise_grounded_state:
                raise KeyError("Can't calculate supervised grounded_state_prediction_loss; "
                               "gt_state wasn't present in training data.")

        if 'curr_state_symlog_sim_dynamics_pred' in prior and 'gt_state' in data:
            raise NotImplementedError()

        for key, dist in dists.items():
            print(f"key: {key}")
            data_key = key.removeprefix("prior_").removeprefix("grounded_")
            loss = -dist.log_prob(data[data_key].astype(jnp.float32))
            assert loss.shape == embed.shape[:2], (key, loss.shape)

            if key.startswith("grounded") and self.config.supervise_grounded_state:
                # only apply grounded head grads where the gt_state isnt directly supervised
                loss = jnp.where(data['is_real'], loss, 0)
            losses[key] = loss

        for key, dist in measure_only_dists.items():
            data_key = key.removeprefix("prior_").removeprefix("grounded_")
            loss = -dist.log_prob(data[data_key].astype(jnp.float32))
            assert loss.shape == embed.shape[:2], (key, loss.shape)
            measure_only_losses[key] = loss

        scaled = {k: v * self.scales[k.removeprefix("prior_").removeprefix("grounded_")] for k, v in losses.items()}
        model_loss = sum(scaled.values())
        out = {'embed': embed, 'post': post, 'prior': prior}
        out.update({f'{k}_loss': v for k, v in losses.items()})
        last_latent = {k: v[:, -1] for k, v in post.items()}
        last_action = data['action'][:, -1]
        state = last_latent, last_action
        metrics = self._metrics(data, dists, measure_only_dists, post, prior, losses, measure_only_losses, model_loss)

        if self.config.disable_wm_train_loss:
            loss_out = jnp.zeros_like(model_loss.mean())
        else:
            loss_out = model_loss.mean()

        return loss_out, (state, out, metrics)

    def td_loss(self, data, state):
        embed = self.encoder(data)
        if self.config.freeze_encoder:
            embed = sg(embed)
        prev_latent, prev_action = state
        prev_actions = jnp.concatenate([
            prev_action[:, None], data['action'][:, :-1]], 1)

        data["t"] = jnp.repeat(jnp.arange(start=0, stop=self.config.batch_length)[jnp.newaxis, :], repeats=self.config.batch_size, axis=0)

        rssm: nets.TDDeterministicZOnlyDreamerModel = self.rssm

        head_input_states, dynamics_preds, dynamics_targets, obs_encoder_predictions = rssm.observe_td(
            embed, prev_actions, data['is_first'], prev_latent, with_residual=False)

        dists = {}
        measure_only_dists = {}
        feats = {**head_input_states, 'embed': embed}
        heads = {**self.heads, **self.grounded_heads} if self.grounded_heads else {**self.heads}

        for name, head in heads.items():
            out = head(feats if name in self.config.grad_heads else sg(feats))
            out = out if isinstance(out, dict) else {name: out}
            if name == "grounded_decoder":
                # The grounded decoder and normal decoder both decode items from our experience,
                # so make sure they have different loss keys for the same items.
                out = {f"grounded_{k}": v for k, v in out.items()}

            if name.startswith("grounded") and name not in self.config.grad_heads:
                measure_only_dists.update(out)
            else:
                dists.update(out)

        losses = {}
        measure_only_losses = {}  # losses we aren't optimizing

        if self.config.td_use_dyn_consistency_loss:
            consistency_loss = self.rssm.dyn_loss(post=dynamics_targets, prior=dynamics_preds, is_first=data['is_first'], **self.config.dyn_loss)
        else:
            state_pred_distance = (dynamics_preds['stoch_params'] - sg(dynamics_targets['stoch_params'])) ** 2  # MSE loss
            state_pred_distance = jnp.where(state_pred_distance < 1e-8, 0, state_pred_distance)
            consistency_loss = state_pred_distance.sum(axis=(-2, -1))

        assert consistency_loss.shape[0] == embed.shape[0], (consistency_loss.shape, embed.shape[:2])
        assert consistency_loss.shape[1] == (embed.shape[1] - 1), (consistency_loss.shape, embed.shape[:2])
        assert len(consistency_loss.shape) == 2, consistency_loss.shape
        consistency_loss *= (self.config.td_loss_rho ** data["t"][:, :-1])
        losses['consistency'] = consistency_loss

        if self.config.use_rep_loss_with_td:
            rep_loss = self.rssm.rep_loss(post=dynamics_targets, prior=dynamics_preds, is_first=data['is_first'], **self.config.dyn_loss)
            assert rep_loss.shape[0] == embed.shape[0], (rep_loss.shape, embed.shape[:2])
            assert rep_loss.shape[1] == (embed.shape[1] - 1), (rep_loss.shape, embed.shape[:2])
            assert len(rep_loss.shape) == 2, rep_loss.shape
            rep_loss *= (self.config.td_loss_rho ** data["t"][:, :-1])
            losses['rep'] = rep_loss

        for key, dist in dists.items():
            data_key = key.removeprefix("prior_").removeprefix("grounded_")
            print(f"key: {key}, dist: {dist.mean().shape}, data: {data[data_key].shape}")
            loss = -dist.log_prob(data[data_key].astype(jnp.float32))
            assert loss.shape == embed.shape[:2], (key, loss.shape)
            loss *= (self.config.td_loss_rho ** data["t"])
            if key.startswith("grounded") and self.config.supervise_grounded_state:
                # only apply grounded head grads where the gt_state isnt directly supervised
                loss = jnp.where(data['is_real'], loss, 0)
            losses[key] = loss

        for key, dist in measure_only_dists.items():
            data_key = key.removeprefix("prior_").removeprefix("grounded_")
            loss = -dist.log_prob(data[data_key].astype(jnp.float32))
            assert loss.shape == embed.shape[:2], (key, loss.shape)
            loss *= (self.config.td_loss_rho ** data["t"])
            measure_only_losses[key] = loss

        scaled = {k: v * self.scales[k.removeprefix("prior_").removeprefix("grounded_")] for k, v in losses.items()}
        scaled = {k: v.sum(axis=(0, 1)) for k, v in scaled.items()}
        model_loss = sum(scaled.values())

        if self.config.td_context_is_head_inputs:
            out = {'embed': embed, 'post': head_input_states, 'prior': head_input_states}
        else:
            # default
            out = {'embed': embed, 'post': obs_encoder_predictions, 'prior': head_input_states}

        out.update({f'{k}_loss': v for k, v in losses.items()})
        last_latent = {k: v[:, -1] for k, v in head_input_states.items()}
        last_action = data['action'][:, -1]
        state = last_latent, last_action
        metrics = self._metrics(data, dists, measure_only_dists, head_input_states, head_input_states, losses, measure_only_losses, model_loss)

        if self.config.disable_wm_train_loss:
            loss_out = jnp.zeros_like(model_loss.mean())
        else:
            loss_out = model_loss.mean()

        return loss_out, (state, out, metrics)

    def loss(self, data, state):
        if self.is_td:
            return self.td_loss(data, state)

        embed = self.encoder(data)
        if self.config.freeze_encoder:
            embed = sg(embed)
        prev_latent, prev_action = state
        prev_actions = jnp.concatenate([
            prev_action[:, None], data['action'][:, :-1]], 1)
        post, prior = self.rssm.observe(
            embed, prev_actions, data['is_first'], prev_latent, with_residual=False, with_student_posterior=False)

        dists = {}
        measure_only_dists = {}
        feats = {**post, 'embed': embed}
        heads = {**self.heads, **self.grounded_heads} if self.grounded_heads else {**self.heads}

        for name, head in heads.items():
            out = head(feats if name in self.config.grad_heads else sg(feats))
            out = out if isinstance(out, dict) else {name: out}
            if name == "grounded_decoder":
                # The grounded decoder and normal decoder both decode items from our experience,
                # so make sure they have different loss keys for the same items.
                out = {f"grounded_{k}": v for k, v in out.items()}

            if name.startswith("grounded") and name not in self.config.grad_heads:
                measure_only_dists.update(out)
            else:
                dists.update(out)

        if self.config.also_apply_head_losses_from_priors:
            prior_feats = {**prior, 'embed': embed}
            for name, head in heads.items():
                out = head(prior_feats if name in self.config.grad_heads else sg(prior_feats))
                out = out if isinstance(out, dict) else {name: out}
                if name == "grounded_decoder":
                    # The grounded decoder and normal decoder both decode items from our experience,
                    # so make sure they have different loss keys for the same items.
                    out = {f"grounded_{k}": v for k, v in out.items()}
                out = {f"prior_{k}": v for k, v in out.items()}
                if self.config.supervise_grounded_state and name.startswith("grounded"):
                    measure_only_dists.update(out)
                else:
                    dists.update(out)

        losses = {}
        measure_only_losses = {}  # losses we aren't optimizing

        if hasattr(self.rssm, "residual") and self.rssm.residual:
            _, residual_prior = self.rssm.observe(
                embed, prev_actions, data['is_first'], prev_latent, with_residual=True, with_res_stop_gradients=True)
            residual_loss = self.rssm.residual.residual_loss(prior=residual_prior,
                                                                   post=post,
                                                                   is_first=data['is_first'],
                                                                   free=self.config.dyn_loss.free)
            dynamics_loss = self.rssm.dyn_loss(post, prior, data['is_first'], **self.config.dyn_loss)

            residual_loss = jnp.where(data['is_real'], residual_loss, 0)
            dynamics_loss = jnp.where(data['is_real'], 0, dynamics_loss)

            losses['residual'] = residual_loss
            losses['dyn'] = dynamics_loss

            if self.config.train_rep_on_residual:
                print(f"1.0 - data['is_real'][:, 0] shape: {(1.0 - data['is_real'][:, 0]).shape}")
                print(f"data['is_real'][:, 0] shape: {(data['is_real'][:, 0]).shape}")
                representation_prior_target = jaxutils.tree_map(
                    lambda x: jaxutils.mask(value=x, mask=1.0 - data['is_real'][:, 0]), prior
                )
                representation_prior_target = jaxutils.tree_map(
                    lambda x, y: x + jaxutils.mask(value=y, mask=data['is_real'][:, 0]),
                    representation_prior_target, residual_prior
                )
                losses['rep'] = self.rssm.rep_loss(post, representation_prior_target, data['is_first'],  **self.config.rep_loss)
            else:
                representation_loss = self.rssm.rep_loss(post, prior, data['is_first'], **self.config.rep_loss)
                representation_loss = jnp.where(data['is_real'], 0, representation_loss)
                losses['rep'] = representation_loss

            if hasattr(self.rssm, 'train_student_posterior') and self.rssm.train_student_posterior:
                raise NotImplementedError()
        else:
            losses['dyn'] = self.rssm.dyn_loss(post, prior, data['is_first'], **self.config.dyn_loss)
            losses['rep'] = self.rssm.rep_loss(post, prior, data['is_first'], **self.config.rep_loss)

            if hasattr(self.rssm, 'train_student_posterior') and self.rssm.train_student_posterior:
                student_post = {
                    'logit': post['student_logit']
                }
                # losses['student_rep'] = self.rssm.rep_loss(student_post, post, data['is_first'], impl='kl', free=self.config.rep_loss.free)
                losses['student_rep'] = self.rssm.rep_loss(student_post, post, data['is_first'], impl='kl', free=0)
                # losses['student_rep'] = self.rssm.dyn_loss(post, student_post, data['is_first'], impl='kl', free=0)

        if self.config.use_grounded_rssm and not self.config.optimize_sup_grounded_loss_in_train_alt:
            if 'gt_state' in data:
                grounded_state_prediction_loss = self.rssm.grounded_state_loss(post=post, gt_state=data['gt_state'], is_real=data['is_real'])
                if self.config.supervise_grounded_state:
                    losses['supervise_grounded_state'] = grounded_state_prediction_loss
                else:
                    measure_only_losses['supervise_grounded_state'] = grounded_state_prediction_loss
            elif self.config.supervise_grounded_state:
                raise KeyError("Can't calculate supervised grounded_state_prediction_loss; "
                               "gt_state wasn't present in training data.")

        if 'curr_state_symlog_sim_dynamics_pred' in prior and 'gt_state' in data:
            next_gt_state = jnp.concatenate((
                data['gt_state'][:, 1:],
                jnp.zeros_like(data['gt_state'][:, 0:1])),
                axis=1)
            next_state_sim_pred_given_s_hat = jnp.concatenate((
                prior['curr_state_symlog_sim_dynamics_pred'][:, 1:],
                jnp.zeros_like(prior['curr_state_symlog_sim_dynamics_pred'][:, 0:1])),
                axis=1)
            measure_only_losses["sim_tf_predict_next_actual_state_from_s_hat"] = self.rssm.simulation_tf_loss(
                symlog_grounded_next_state_pred=next_state_sim_pred_given_s_hat,
                target_symlog_next_state=jaxutils.symlog(next_gt_state),
                is_last=data['is_last'])

        for key, dist in dists.items():
            print(f"key: {key}")
            data_key = key.removeprefix("prior_").removeprefix("grounded_")
            loss = -dist.log_prob(data[data_key].astype(jnp.float32))
            assert loss.shape == embed.shape[:2], (key, loss.shape)

            if key.startswith("grounded") and self.config.supervise_grounded_state:
                # only apply grounded head grads where the gt_state isnt directly supervised
                loss = jnp.where(data['is_real'], loss, 0)
            losses[key] = loss

        for key, dist in measure_only_dists.items():
            data_key = key.removeprefix("prior_").removeprefix("grounded_")
            loss = -dist.log_prob(data[data_key].astype(jnp.float32))
            assert loss.shape == embed.shape[:2], (key, loss.shape)
            measure_only_losses[key] = loss

        scaled = {k: v * self.scales[k.removeprefix("prior_").removeprefix("grounded_")] for k, v in losses.items()}
        scaled = {
            k: v
            for k, v in scaled.items()
            if self.scales[k.removeprefix("prior_").removeprefix("grounded_")] != 0.0
        }
        model_loss = sum(scaled.values())
        out = {'embed': embed, 'post': post, 'prior': prior}
        out.update({f'{k}_loss': v for k, v in losses.items()})
        last_latent = {k: v[:, -1] for k, v in post.items()}
        last_action = data['action'][:, -1]
        state = last_latent, last_action
        metrics = self._metrics(data, dists, measure_only_dists, post, prior, losses, measure_only_losses, model_loss)

        if self.config.disable_wm_train_loss:
            loss_out = jnp.zeros_like(model_loss.mean())
        else:
            loss_out = model_loss.mean()

        return loss_out, (state, out, metrics)

    def black_box_loss(self, experience_data, pred_state_data, state):
        losses = {}
        experience_dists = {}
        pred_state_dists = {}

        train_on_pred_state = (self.config.train_grounded_nets_on_world_model_train_pred_states or
                               self.config.train_grounded_nets_on_imagined_rollout_pred_states)

        if self.config.train_grounded_nets_on_experience:
            feats_from_experience = {
                "symlog_grounded": jaxutils.symlog(experience_data["gt_state"]),
                "cont": experience_data["cont"],
                "is_valid": experience_data["is_valid"],
                **{k: experience_data[k] for k in self.obs_space.keys()}
            }
            for name, head in self.grounded_heads.items():
                experience_out = head(sg(feats_from_experience))
                if not isinstance(experience_out, dict):
                    data_key = name.removeprefix("grounded_")
                    experience_out = {data_key: experience_out}
                assert isinstance(experience_out, dict), experience_out
                experience_dists.update(experience_out)

            for key, dist in experience_dists.items():
                loss = -dist.log_prob(experience_data[key].astype(jnp.float32))
                assert loss.shape == experience_data['cont'].shape[:2], (key, loss.shape)
                losses[key] = loss

            if self.config.optimize_sup_grounded_loss_in_train_alt:
                raise DeprecationWarning("trying to remove this old experimental code")
                # assert self.config.supervise_grounded_state
                # if 'gt_state' in feats_from_experience:
                #     symlog_state_pred = self.encoder(feats_from_experience)
                #     distance = (symlog_state_pred - sg(feats_from_experience['symlog_grounded'])) ** 2  # MSE loss
                #     distance = jnp.where(distance < 1e-8, 0, distance)
                #     grounded_state_prediction_loss = distance.sum(-1)
                #     losses['supervise_grounded_state'] = grounded_state_prediction_loss
                # else:
                #     raise KeyError("Can't calculate supervised grounded_state_prediction_loss; "
                #                    "gt_state wasn't present in training data.")

            if self.config.optimize_rssm_in_train_alt:
                embed = self.encoder(experience_data)
                if self.config.freeze_encoder:
                    embed = sg(embed)
                prev_latent, prev_action = state
                prev_actions = jnp.concatenate([
                    prev_action[:, None], experience_data['action'][:, :-1]], 1)
                post, prior = self.rssm.observe(
                    embed, prev_actions, experience_data['is_first'], prev_latent, with_residual=False)
                losses['dyn'] = self.rssm.dyn_loss(post, prior, experience_data['is_first'], **self.config.dyn_loss)
                losses['rep'] = self.rssm.rep_loss(post, prior, experience_data['is_first'], **self.config.rep_loss)
                losses['supervise_grounded_state'] = self.rssm.grounded_state_loss(post=post, gt_state=experience_data['gt_state'], is_real=experience_data['is_real'])

        if train_on_pred_state and pred_state_data:
            if self.config.sim_query_data_same_in_format_as_normal_experience:
                feats_from_pred_state = {
                    "symlog_grounded": pred_state_data['gt_state'] if self.config.pred_state_data_uses_gt_state else pred_state_data["pred_state_symlog"],
                    "cont": pred_state_data["cont"],
                    "is_valid": pred_state_data["is_valid"],
                    **{k: pred_state_data[k] for k in self.obs_space.keys()}
                }
                for name, head in self.grounded_heads.items():
                    pred_state_out = head(sg(feats_from_pred_state))
                    if not isinstance(pred_state_out, dict):
                        data_key = name.removeprefix("grounded_")
                        pred_state_out = {data_key: pred_state_out}
                    assert isinstance(pred_state_out, dict), pred_state_out
                    pred_state_dists.update(pred_state_out)

                for key, dist in pred_state_dists.items():
                    loss = -dist.log_prob(pred_state_data[key].astype(jnp.float32))
                    if key != "is_valid":
                        # Zero out loss for states that we couldn't actually reset the sim to.
                        # The grounded_is_valid head can still train on these to learn that they aren't valid.
                        loss = jnp.where(pred_state_data["is_usable"], loss, 0)

                    assert loss.shape == pred_state_data['is_usable'].shape[:2], (key, loss.shape)
                    # assert loss.shape == pred_state_data['pred_state_image'].shape[:2], (key, loss.shape)
                    losses[f"pred_state_{key}"] = loss
            else:
                raise DeprecationWarning("trying to remove old code")
                # feats_from_pred_state = {
                #     "symlog_grounded": pred_state_data["pred_state_symlog"],
                #     "pred_state_cont": pred_state_data["pred_state_cont"],
                #     "pred_state_is_valid": pred_state_data["pred_state_is_valid"],
                #     **{k: pred_state_data[f"pred_state_{k}"] for k in self.obs_space.keys()}
                # }
                # for name, head in self.grounded_heads.items():
                #     pred_state_out = head(sg(feats_from_pred_state))
                #     if not isinstance(pred_state_out, dict):
                #         data_key = name.removeprefix("grounded_")
                #         pred_state_out = {data_key: pred_state_out}
                #     assert isinstance(pred_state_out, dict), pred_state_out
                #     pred_state_out = {f"pred_state_{k}": v for k, v in pred_state_out.items()}
                #     pred_state_dists.update(pred_state_out)
                #
                # for key, dist in pred_state_dists.items():
                #     loss = -dist.log_prob(pred_state_data[key].astype(jnp.float32))
                #     if key != "pred_state_is_valid":
                #         # Zero out loss for states that we couldn't actually reset the sim to.
                #         # The grounded_is_valid head can still train on these to learn that they aren't valid.
                #         loss = jnp.where(pred_state_data["pred_state_is_usable"], loss, 0)
                #
                #     assert loss.shape == pred_state_data['pred_state_is_usable'].shape[:2], (key, loss.shape)
                #     # assert loss.shape == pred_state_data['pred_state_image'].shape[:2], (key, loss.shape)
                #     losses[key] = loss

            if self.config.optimize_sup_grounded_loss_in_train_alt:
                raise DeprecationWarning("trying to remove this old experimental code")
                # assert self.config.supervise_grounded_state
                # if 'gt_state' in feats_from_pred_state:
                #     symlog_state_pred = self.encoder(feats_from_pred_state)
                #     distance = (symlog_state_pred - sg(feats_from_pred_state['symlog_grounded'])) ** 2  # MSE loss
                #     distance = jnp.where(distance < 1e-8, 0, distance)
                #     grounded_state_prediction_loss = distance.sum(-1)
                #     losses['pred_state_supervise_grounded_state'] = grounded_state_prediction_loss
                # else:
                #     raise KeyError("Can't calculate supervised grounded_state_prediction_loss; "
                #                    "gt_state wasn't present in training data.")

            if self.config.optimize_rssm_in_train_alt:
                assert self.config.sim_query_data_same_in_format_as_normal_experience
                embed = self.encoder(pred_state_data)
                if self.config.freeze_encoder:
                    embed = sg(embed)
                prev_latent, prev_action = state
                prev_actions = jnp.concatenate([
                    prev_action[:, None], pred_state_data['action'][:, :-1]], 1)
                post, prior = self.rssm.observe(
                    embed, prev_actions, pred_state_data['is_first'], prev_latent, with_residual=False)
                losses['pred_state_dyn'] = self.rssm.dyn_loss(post, prior, pred_state_data['is_first'], **self.config.dyn_loss)
                losses['pred_state_rep'] = self.rssm.rep_loss(post, prior, pred_state_data['is_first'], **self.config.rep_loss)
                losses['pred_state_supervise_grounded_state'] = self.rssm.grounded_state_loss(post=post, gt_state=pred_state_data['gt_state'], is_real=pred_state_data['is_real'])

        if (self.config.use_sim_forward_dynamics and not self.config.optimize_sim_tf_with_world_model_in_train
                and self.config.train_grounded_nets_on_experience):
            symlog_gt_state = jaxutils.symlog(experience_data["gt_state"])
            next_symlog_gt_state = jnp.concatenate((
                symlog_gt_state[:, 1:],
                jnp.zeros_like(symlog_gt_state[:, 0:1])),
                axis=1
            )
            sim_tf_next_state_pred = self.rssm.get_sim_forward_dynamics(grounded_symlog_in=symlog_gt_state,
                                                                        action=experience_data['action'])
            # jax.debug.print("sim experience actions:\n{}", experience_data['action'])

            losses['sim_tf_on_experience'] = self.rssm.simulation_tf_loss(
                symlog_grounded_next_state_pred=sim_tf_next_state_pred,
                target_symlog_next_state=next_symlog_gt_state,
                is_last=experience_data['is_last'])

        if (self.config.use_sim_forward_dynamics and not self.config.optimize_sim_tf_with_world_model_in_train
                and train_on_pred_state and pred_state_data):
            curr_state_pred = sg(pred_state_data['pred_state_symlog'])
            next_state_given_curr_state_pred = sg(pred_state_data['pred_state_sim_next_symlog_state'])
            sim_tf_next_state_pred = self.rssm.get_sim_forward_dynamics(grounded_symlog_in=curr_state_pred,
                                                                        action=pred_state_data['action'])
            # jax.debug.print("sim pred state actions:\n{}", pred_state_data['action'])
            sim_tf_pred_state_loss = self.rssm.simulation_tf_loss(
                symlog_grounded_next_state_pred=sim_tf_next_state_pred,
                target_symlog_next_state=next_state_given_curr_state_pred,
                is_last=jnp.zeros_like(pred_state_data['pred_state_is_last']))
            sim_tf_pred_state_loss = jnp.where(pred_state_data["pred_state_is_usable"], sim_tf_pred_state_loss, 0)
            losses['sim_tf_on_pred_state'] = sim_tf_pred_state_loss

        out = {f'{k}_loss': v for k, v in losses.items()}

        scaled = {k: v * self.scales[k.removeprefix("pred_state_")] for k, v in losses.items()}
        alt_model_loss = sum(scaled.values())
        metrics = self._alt_metrics(experience_data, pred_state_data, losses, alt_model_loss)
        return alt_model_loss.mean(), (out, metrics)

    def imagine(self, policy, start, horizon, with_residual=True):
        first_cont = (1.0 - start['is_terminal']).astype(jnp.float32)
        keys = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}
        start['action'] = policy(start)

        def step(prev, _):
            prev = prev.copy()
            state = self.rssm.img_step(prev, prev.pop('action'), with_residual=with_residual, with_res_stop_gradients=False)
            return {**state, 'action': policy(state)}

        traj = jaxutils.scan(
            step, jnp.arange(horizon), start, self.config.imag_unroll)
        traj = {
            k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}

        cont_head = (self.heads['cont'] if self.config.use_heads_from_vanilla_dreamer
                     else self.grounded_heads['grounded_cont'])
        cont = cont_head(traj).mode()

        traj['cont'] = jnp.concatenate([first_cont[None], cont[1:]], 0)
        discount = 1 - 1 / self.config.horizon
        traj['weight'] = jnp.cumprod(discount * traj['cont'], 0) / discount
        return traj

    def imagine_no_residual(self, policy, start, horizon):
        return self.imagine(policy=policy, start=start, horizon=horizon, with_residual=False)

    @property
    def reward_head(self):
        return (self.heads['reward'] if self.config.use_heads_from_vanilla_dreamer
                else self.grounded_heads['grounded_reward'])

    def report(self, data, train_alt_pred_state_data):
        state = self.initial(len(data['is_first']))
        report = {}
        if self.config.normal_wm_loss_for_residual_only:
            report.update(self.normal_wm_loss_for_residual_only(data, state)[-1][-1])
        else:
            report.update(self.loss(data, state)[-1][-1])
        if hasattr(self.rssm, "residual") and self.rssm.residual:
            report.update(self.residual_loss_only(data, state)[-1][-1])

        # Log videos for default world model decoder
        embed = self.encoder(data)
        if self.config.use_heads_from_vanilla_dreamer:
            decoder = self.heads['decoder']
        else:
            decoder = self.grounded_heads['grounded_decoder']

        if embed.shape[1] >= 6:
            wm_batch_posterior, wm_batch_prior = self.rssm.observe(embed[:6, :5], data['action'][:6, :5],
                                                                   data['is_first'][:6, :5],
                                                                   with_residual=hasattr(self.rssm, "residual") and self.rssm.residual,
                                                                   with_student_posterior=True)
            start = {k: v[:, -1] for k, v in wm_batch_posterior.items()}



            wm_batch_posterior_recon = decoder(wm_batch_posterior)
            imagined_trajectory_prior = self.rssm.imagine(data['action'][:6, 5:], start)
            openl = decoder(imagined_trajectory_prior)

            # Log imagined trajectory vs actual trajectory (also showing decoded sim_tf predictions if possible)
            if self.config.use_grounded_rssm and self.config.use_sim_forward_dynamics:
                imagined_trajectory_sim_tf_prior = tree_map(jnp.copy, imagined_trajectory_prior)
                imagined_trajectory_sim_tf_prior['symlog_grounded'] = imagined_trajectory_sim_tf_prior[
                    'curr_state_symlog_sim_dynamics_pred']
                openl_sim_tf = decoder(imagined_trajectory_sim_tf_prior)

                for key in decoder.cnn_shapes.keys():
                    truth = data[key][:6].astype(jnp.float32)
                    model = jnp.concatenate([wm_batch_posterior_recon[key].mode()[:, :5], openl[key].mode()], 1)
                    sim_tf_model = jnp.concatenate([wm_batch_posterior_recon[key].mode()[:, :5], openl_sim_tf[key].mode()],
                                                   1)
                    error = (model - truth + 1) / 2
                    sim_tf_vs_wm_error = (sim_tf_model - model + 1) / 2
                    video = jnp.concatenate([truth, model, sim_tf_model, error, sim_tf_vs_wm_error, ], 2)
                    report[f'openl_with_sim_tf_prior_{key}'] = jaxutils.video_grid(video)
            else:
                for key in decoder.cnn_shapes.keys():
                    truth = data[key][:6].astype(jnp.float32)
                    model = jnp.concatenate([wm_batch_posterior_recon[key].mode()[:, :5], openl[key].mode()], 1)
                    error = (model - truth + 1) / 2
                    video = jnp.concatenate([truth, model, error], 2)

                    print(f"openl_{key} video shape: {video.shape}")
                    if video.shape[-1] > 3:
                        orig_video = video
                        video = orig_video[:, :, :, :, :3]
                        video_2 = orig_video[:, :, :, :, 3:]
                        # print(f"rendering frame stacked video")
                        # # assume frame stacking
                        # assert video.shape[-1] % 3 == 0, video.shape
                        # stack_amt = video.shape[-1] // 3
                        # target_shape = list(video.shape)
                        # target_shape[1] = target_shape[1] * stack_amt
                        # target_shape[-1] = target_shape[-1] // stack_amt
                        # video = video.reshape(target_shape)
                        # print(f"openl_{key} new video shape: {video.shape}")
                        report[f'openl_{key}_2'] = jaxutils.video_grid(video_2)

                    report[f'openl_{key}'] = jaxutils.video_grid(video)

        # Compare decoded sim_tf s prediction with world model s predictions
        if self.config.use_grounded_rssm and self.config.use_sim_forward_dynamics:
            wm_batch_posterior, wm_batch_prior = self.rssm.observe(self.encoder(data)[:6], data['action'][:6],
                                                                   data['is_first'][:6])
            wm_batch_posterior_recon = decoder(wm_batch_posterior)

            sim_tf_prior = tree_map(jnp.copy, wm_batch_prior)
            sim_tf_prior['symlog_grounded'] = sim_tf_prior['curr_state_symlog_sim_dynamics_pred']
            sim_tf_prior_recon = decoder(sim_tf_prior)
            wm_prior_recon = decoder(wm_batch_prior)

            for key in decoder.cnn_shapes.keys():
                posterior_video = wm_batch_posterior_recon[key].mode()[:, 1:]
                prior_video = wm_prior_recon[key].mode()[:, 1:]
                sim_tf_prior_video = sim_tf_prior_recon[key].mode()[:, 1:]
                post_vs_prior_error = (posterior_video - prior_video + 1) / 2
                prior_vs_sim_error = (prior_video - sim_tf_prior_video + 1) / 2
                post_vs_sim_error = (posterior_video - sim_tf_prior_video + 1) / 2
                video = jnp.concatenate([posterior_video, prior_video, sim_tf_prior_video,
                                         post_vs_prior_error, prior_vs_sim_error, post_vs_sim_error], 2)
                report[f'post_vs_prior_vs_sim_prior_decoded_{key}'] = jaxutils.video_grid(video)

        # Log videos for grounded decoder from gt_state
        if 'gt_state' in data and self.config.use_grounded_rssm:
            symlog_gt_state_input = jaxutils.symlog(data['gt_state'])[:6]
            assert "symlog_grounded" not in data
            decoder_outs = self.grounded_heads['grounded_decoder']({"symlog_grounded": symlog_gt_state_input})
            for key in self.grounded_heads['grounded_decoder'].cnn_shapes.keys():
                target = data[key][:6].astype(jnp.float32)
                model = decoder_outs[key].mode()
                error = (model - target + 1) / 2
                video = jnp.concatenate([target, model, error], 2)
                report[f'grounded_decoder_gt_state_{key}'] = jaxutils.video_grid(video)

        if "original_image" in data:
            augmented_images = data["image"][:12].astype(jnp.float32)
            original_images = data["original_image"][:12].astype(jnp.float32)
            video = jnp.concatenate([augmented_images, original_images], 2)
            report[f'augmented_images'] = jaxutils.video_grid(video)

        # Now that we include imagined trajectories as pred_state data,
        # we no longer have a gt obs to compare to in the snippet below:
        # # Log videos for grounded decoder from pred_state and compare to decoded gt_state
        # if train_alt_pred_state_data and self.config.use_grounded_rssm:
        #     symlog_state_input = train_alt_pred_state_data['pred_state_symlog'][:6]
        #     assert "symlog_grounded" not in train_alt_pred_state_data
        #     train_alt_pred_state_data["symlog_grounded"] = symlog_state_input
        #     decoder_outs = self.grounded_heads['grounded_decoder'](train_alt_pred_state_data)
        #     for key in self.grounded_heads['grounded_decoder'].cnn_shapes.keys():
        #         observed = train_alt_pred_state_data[key][:6].astype(jnp.float32)
        #         model = decoder_outs[key].mode()
        #         target = train_alt_pred_state_data[f"pred_state_{key}"][:6].astype(jnp.float32)
        #         error_vs_target = (model - target + 1) / 2
        #         error_vs_observed = (model - observed + 1) / 2
        #         video = jnp.concatenate([observed, target, model, error_vs_target, error_vs_observed], 2)
        #         report[f'grounded_decoder_pred_state_{key}'] = jaxutils.video_grid(video)

        # Log videos for grounded decoder from pred_state
        if 'gt_state' in data and self.config.use_grounded_rssm:
            wm_batch_posterior, _ = self.rssm.observe(self.encoder(data)[:6], data['action'][:6], data['is_first'][:6])
            symlog_pred_state_input = wm_batch_posterior['symlog_grounded']
            assert "symlog_grounded" not in data
            decoder_outs = self.grounded_heads['grounded_decoder']({"symlog_grounded": symlog_pred_state_input})
            for key in self.grounded_heads['grounded_decoder'].cnn_shapes.keys():
                observed = data[key][:6].astype(jnp.float32)
                model = decoder_outs[key].mode()
                error_vs_observed = (model - observed + 1) / 2
                video = jnp.concatenate([observed, model, error_vs_observed], 2)
                report[f'grounded_decoder_pred_state_{key}'] = jaxutils.video_grid(video)

        return report

    def report_non_compiled(self, data):
        # Report this separately because JAX can't compile interaction with actual sim environment.
        report = {}

        if self.config.use_sim_forward_dynamics and 'gt_state' in data:
            # Log videos of learned sim_tf producing transitions instead of actual environment
            decoder = (self.heads['decoder'] if self.config.use_heads_from_vanilla_dreamer
                       else self.grounded_heads['grounded_decoder'])

            image_keys = decoder.cnn_shapes.keys()
            num_batch_elements = 6
            seq_len = data['action'].shape[1]
            gt_videos = {k: data[k][:num_batch_elements].astype(np.float32) for k in image_keys}

            # Rollout a trajectory of grounded states using only the learned sim_tf,
            # then use the actual simulation to render the state predictions.
            predicted_videos = {k: [] for k in image_keys}
            is_first = jax.device_get(data['is_first']).astype(bool)
            assert not np.all(is_first), is_first
            for batch_element in range(num_batch_elements):
                predicted_video = {k: [] for k in image_keys}
                current_state_symlog = None
                for time_index in range(seq_len):
                    if is_first[batch_element, time_index]:
                        current_state_symlog = jax.device_get(
                            jaxutils.symlog(data['gt_state'][batch_element, time_index]))
                    assert current_state_symlog is not None
                    sim_tf_obs, _, _ = self.sim_query_env.reset_to_internal_state(
                        new_internal_state=jax.device_get(jaxutils.symexp(current_state_symlog)))
                    for k in image_keys:
                        new_sim_image = jnp.asarray(sim_tf_obs[k])
                        if new_sim_image.dtype == jnp.uint8:
                            new_sim_image = new_sim_image / 255.0
                        predicted_video[k].append(new_sim_image)
                    action = data['action'][batch_element, time_index]
                    current_state_symlog = self.rssm.get_sim_forward_dynamics(grounded_symlog_in=current_state_symlog,
                                                                              action=action)
                for k in image_keys:
                    predicted_videos[k].append(predicted_video[k])
            for k in image_keys:
                observed = jnp.asarray(gt_videos[k])
                predicted = jnp.asarray(predicted_videos[k])
                error_vs_observed = (predicted - observed + 1) / 2
                video = jnp.concatenate([observed, predicted, error_vs_observed], 2)
                report[f'trajectory_using_sim_tf_only_{k}'] = jaxutils.video_grid(video)

        if self.config.use_grounded_rssm and 'gt_state' in data:
            # Sanity check that resetting the sim to a ground-truth state results in the corresponding collected obs.
            decoder = self.heads['decoder'] if self.config.use_heads_from_vanilla_dreamer else self.grounded_heads[
                'grounded_decoder']
            image_keys = decoder.cnn_shapes.keys()
            num_batch_elements = 6
            seq_len = data['action'].shape[1]

            gt_videos = {k: data[k][:num_batch_elements].astype(np.float32) for k in image_keys}
            videos_from_resets = {k: [] for k in image_keys}

            is_first = jax.device_get(data['is_first']).astype(bool)
            assert not np.all(is_first), is_first
            for batch_element in range(num_batch_elements):
                predicted_video = {k: [] for k in image_keys}
                for time_index in range(seq_len):
                    current_state_symlog = jax.device_get(jaxutils.symlog(data['gt_state'][batch_element, time_index]))
                    assert current_state_symlog is not None
                    obs_from_reset, _, _ = self.sim_query_env.reset_to_internal_state(
                        new_internal_state=jax.device_get(jaxutils.symexp(current_state_symlog)))
                    for k in image_keys:
                        new_sim_image = jnp.asarray(obs_from_reset[k])
                        if new_sim_image.dtype == jnp.uint8:
                            new_sim_image = new_sim_image / 255.0
                        predicted_video[k].append(new_sim_image)
                for k in image_keys:
                    videos_from_resets[k].append(predicted_video[k])
            for k in image_keys:
                observed = jnp.asarray(gt_videos[k])
                from_resets = jnp.asarray(videos_from_resets[k])
                error_vs_observed = (from_resets - observed + 1) / 2
                video = jnp.concatenate([observed, from_resets, error_vs_observed], 2)
                report[f'sanity_check_observed_vs_from_resets_{k}'] = jaxutils.video_grid(video)

        return report

    def _metrics(self, data, dists, measure_only_dists, post, prior, losses, measure_only_losses, model_loss):
        metrics = {}
        try:
            entropy = lambda feat: self.rssm.get_dist(feat).entropy()
            metrics.update(jaxutils.tensorstats(entropy(prior), 'prior_ent'))
            metrics.update(jaxutils.tensorstats(entropy(post), 'post_ent'))
        except NotImplementedError:
            pass
        for k, v in losses.items():
            print(f"{k}_loss shape: {v.shape}")
        metrics.update({f'{k}_loss_mean': v.mean() for k, v in losses.items()})
        metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})
        metrics.update({f'{k}_loss_measure_only_mean': v.mean() for k, v in measure_only_losses.items()})
        metrics.update({f'{k}_loss_measure_only_std': v.std() for k, v in measure_only_losses.items()})
        metrics['model_loss_mean'] = model_loss.mean()
        metrics['model_loss_std'] = model_loss.std()
        metrics['reward_max_data'] = jnp.abs(data['reward']).max()

        if self.config.use_heads_from_vanilla_dreamer:
            metrics['reward_max_pred'] = jnp.abs(dists['reward'].mean()).max()
            if 'reward' in dists and not self.config.jax.debug_nans:
                stats = jaxutils.balance_stats(dists['reward'], data['reward'], 0.1)
                metrics.update({f'reward_{k}': v for k, v in stats.items()})
            if 'cont' in dists and not self.config.jax.debug_nans:
                stats = jaxutils.balance_stats(dists['cont'], data['cont'], 0.5)
                metrics.update({f'cont_{k}': v for k, v in stats.items()})

        if self.config.use_grounded_rssm:
            all_dists = {**dists, **measure_only_dists}
            metrics['grounded_reward_max_pred'] = jnp.abs(all_dists['grounded_reward'].mean()).max()
            if 'grounded_reward' in all_dists and not self.config.jax.debug_nans:
                stats = jaxutils.balance_stats(all_dists['grounded_reward'], data['reward'], 0.1)
                metrics.update({f'grounded_reward_{k}': v for k, v in stats.items()})
            if 'grounded_cont' in all_dists and not self.config.jax.debug_nans:
                stats = jaxutils.balance_stats(all_dists['grounded_cont'], data['cont'], 0.5)
                metrics.update({f'grounded_cont_{k}': v for k, v in stats.items()})

        return metrics

    def _alt_metrics(self, experience_data, pred_state_data, losses, alt_model_loss):
        metrics = {}
        metrics.update({f'{k}_loss_mean': v.mean() for k, v in losses.items()})
        metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})
        metrics['model_loss_mean'] = alt_model_loss.mean()
        metrics['model_loss_std'] = alt_model_loss.std()
        if experience_data and 'gt_state' in experience_data:
            metrics['gt_state_mean'] = experience_data['gt_state'].mean()
            metrics['gt_state_std'] = experience_data['gt_state'].std()
        if pred_state_data and 'pred_state_symlog' in pred_state_data:
            metrics['pred_state_mean'] = jaxutils.symexp(pred_state_data['pred_state_symlog']).mean()
            metrics['pred_state_std'] = jaxutils.symexp(pred_state_data['pred_state_symlog']).std()

        return metrics


class ImagActorCritic(nj.Module):

    def __init__(self, critics, scales, act_space, config, grounded_state_normalizer_params):
        critics = {k: v for k, v in critics.items() if scales[k]}
        for key, scale in scales.items():
            assert not scale or key in critics, key
        self.critics = {k: v for k, v in critics.items() if scales[k]}
        self.scales = scales
        self.act_space = act_space
        self.config = config
        disc = act_space.discrete
        self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
        dims = 'symlog_grounded' if self.config.use_grounded_rssm else 'deter'
        self.actor = nets.MLP(
            name='actor', dims=dims, shape=act_space.shape, **config.actor,
            dist=config.actor_dist_disc if disc else config.actor_dist_cont)
        self.retnorms = {
            k: jaxutils.Moments(**config.retnorm, name=f'retnorm_{k}')
            for k in critics}
        self.opt = jaxutils.Optimizer(name='actor_opt', **config.actor_opt)

        self.grounded_state_normalizer_params = grounded_state_normalizer_params
        if self.grounded_state_normalizer_params is not None:
            assert config.normalize_agent_grounded_input and not config.normalize_all_grounded_states

    def initial(self, batch_size):
        return {}

    # def _normalize_grounded(self, batch: dict):
    #     if self.grounded_state_normalizer_params is not None:
    #         return {k: v if k != 'symlog_grounded' else normalize(batch=jaxutils.symexp(v),
    #                                                               mean_std=self.grounded_state_normalizer_params.read())
    #                 for k, v in batch.items()}
    #     else:
    #         return batch
    #
    # def _update_normalization_state(self, batch: dict):
    #     if self.grounded_state_normalizer_params is not None:
    #         self.grounded_state_normalizer_params.write(
    #             update(self.grounded_state_normalizer_params.read(), batch=jaxutils.symexp(batch['symlog_grounded'])))

    def _normalize_grounded(self, batch: dict):
        if self.grounded_state_normalizer_params is not None:
            raise NotImplementedError(f"there's still a bug where the normaized state if provided to the unnormalized reward head in the critic train function")
            out = {}
            for k, v in batch.items():
                assert not isinstance(v, dict), k
                out[k] = v if k != 'symlog_grounded' else normalize(
                    batch=v, mean_std=self.grounded_state_normalizer_params.read())
            return out
        else:
            return batch

    def update_normalization_state(self, batch: dict):
        if self.grounded_state_normalizer_params is not None:
            self.grounded_state_normalizer_params.write(
                update(self.grounded_state_normalizer_params.read(), batch=jaxutils.symlog(batch['gt_state'])))

    def policy(self, state, carry):
        return {'action': self.actor(self._normalize_grounded(state))}, carry

    def train(self, imagine, start, context):

        # jax.debug.print("start['is_real'].sum(): {x} ", x=start['is_real'].sum())

        def loss(start):
            policy = lambda s: self.actor(sg(self._normalize_grounded(s))).sample(seed=nj.rng())
            traj = imagine(policy, start, self.config.imag_horizon)
            loss, metrics = self.loss(self._normalize_grounded(traj))
            print(f"actor loss shape: {loss.shape}")
            return loss, (traj, metrics)

        mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
        metrics.update(mets)
        for key, critic in self.critics.items():
            mets = critic.train(self._normalize_grounded(traj), self.actor)
            metrics.update({f'{key}_critic_{k}': v for k, v in mets.items()})

        # self._update_normalization_state(traj)

        return traj, metrics

    def loss(self, traj):
        metrics = {}
        advs = []
        total = sum(self.scales[k] for k in self.critics)
        for key, critic in self.critics.items():
            rew, ret, base = critic.score(traj, self.actor)
            offset, invscale = self.retnorms[key](ret)
            normed_ret = (ret - offset) / invscale
            normed_base = (base - offset) / invscale
            adv = (normed_ret - normed_base) * self.scales[key] / total
            advs.append(adv)
            metrics.update(jaxutils.tensorstats(rew, f'{key}_reward'))
            metrics.update(jaxutils.tensorstats(ret, f'{key}_return_raw'))
            metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_return_normed'))
            metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_adv'))
            metrics[f'{key}_return_rate'] = (jnp.abs(ret) >= 0.5).mean()
        adv = jnp.stack(advs).sum(0)
        policy = self.actor(sg(traj))
        logpi = policy.log_prob(sg(traj['action']))[:-1]
        loss = {'backprop': -adv, 'reinforce': -logpi * sg(adv)}[self.grad]
        ent = policy.entropy()[:-1]
        loss -= self.config.actent * ent
        loss *= sg(traj['weight'])[:-1]
        loss *= self.config.loss_scales.actor
        metrics.update(self._metrics(traj, policy, logpi, ent, adv))
        return loss.mean(), metrics

    def _metrics(self, traj, policy, logpi, ent, adv):
        metrics = {}
        ent = policy.entropy()[:-1]
        rand = (ent - policy.minent) / (policy.maxent - policy.minent)
        rand = rand.mean(range(2, len(rand.shape)))
        act = traj['action']
        act = jnp.argmax(act, -1) if self.act_space.discrete else act
        metrics.update(jaxutils.tensorstats(act, 'action'))
        metrics.update(jaxutils.tensorstats(rand, 'policy_randomness'))
        metrics.update(jaxutils.tensorstats(ent, 'policy_entropy'))
        metrics.update(jaxutils.tensorstats(logpi, 'policy_logprob'))
        metrics.update(jaxutils.tensorstats(adv, 'adv'))
        metrics['imag_weight_dist'] = jaxutils.subsample(traj['weight'])

        if self.grounded_state_normalizer_params is not None:
            normalizer_params = self.grounded_state_normalizer_params.read()
            metrics.update({
                'agent_obs_normalizer count': normalizer_params.count,
            })
            jax.debug.print("agent_obs_normalizer std: {}", normalizer_params.std)
            jax.debug.print("agent_obs_normalizer mean: {}", normalizer_params.mean)
            jax.debug.print("agent_obs_normalizer summed_variance: {}", normalizer_params.summed_variance)

        return metrics


class VFunction(nj.Module):

    def __init__(self, rewfn, config):
        self.rewfn = rewfn
        self.config = config
        dims = 'symlog_grounded' if self.config.use_grounded_rssm else 'deter'
        self.net = nets.MLP((), name='net', dims=dims, **self.config.critic)
        self.slow = nets.MLP((), name='slow', dims=dims, **self.config.critic)
        self.updater = jaxutils.SlowUpdater(
            self.net, self.slow,
            self.config.slow_critic_fraction,
            self.config.slow_critic_update)
        self.opt = jaxutils.Optimizer(name='critic_opt', **self.config.critic_opt)

    def train(self, traj, actor):
        target = sg(self.score(traj)[1])
        mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
        metrics.update(mets)
        self.updater()
        return metrics

    def loss(self, traj, target):
        metrics = {}
        traj = {k: v[:-1] for k, v in traj.items()}
        dist = self.net(traj)
        loss = -dist.log_prob(sg(target))
        if self.config.critic_slowreg == 'logprob':
            reg = -dist.log_prob(sg(self.slow(traj).mean()))
        elif self.config.critic_slowreg == 'xent':
            reg = -jnp.einsum(
                '...i,...i->...',
                sg(self.slow(traj).probs),
                jnp.log(dist.probs))
        else:
            raise NotImplementedError(self.config.critic_slowreg)
        loss += self.config.loss_scales.slowreg * reg
        print(f"critic loss before mean shape: {loss.shape}")
        loss = (loss * sg(traj['weight'])).mean()
        loss *= self.config.loss_scales.critic
        metrics = jaxutils.tensorstats(dist.mean())
        return loss, metrics

    def score(self, traj, actor=None):
        rew = self.rewfn(traj)
        assert len(rew) == len(traj['action']) - 1, (
            'should provide rewards for all but last action')
        discount = 1 - 1 / self.config.horizon
        disc = traj['cont'][1:] * discount
        value = self.net(traj).mean()
        vals = [value[-1]]
        interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
        for t in reversed(range(len(disc))):
            vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
        ret = jnp.stack(list(reversed(vals))[:-1])
        return rew, ret, value[:-1]


def symexp_np(x):
    return np.sign(x) * (np.exp(np.abs(x)) - 1)


def symlog_np(x):
    return np.sign(x) * np.log(1 + np.abs(x))
