import re

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from functools import partial
from typing import Type

f32 = jnp.float32
tfd = tfp.distributions
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

from dreamerv3 import jaxutils
from dreamerv3 import ninjax as nj

cast = jaxutils.cast_to_compute


class RSSM(nj.Module):

    def __init__(
            self, deter=1024, stoch=32, classes=32, unroll=False, initial='learned',
            unimix=0.01, action_clip=1.0, img_hidden_layers=2, obs_hidden_layers=0,
            decode_stoch_params_as_deter=False,
            posterior_takes_prior_deter_as_input=True,
            posterior_takes_prior_stoch_as_input=False,
            use_gru_with_prior_belief=False,
            stoch_params_include_unimix=False,
            stoch_params_are_raw_logits=False,
            use_half_of_stoch_as_free_variables=False,
            use_posterior_stoch_params_for_first_state=False,
            use_posterior_stoch_params_for_all_states=False,
            dynamics_takes_prev_stoch_as_input=False,
            dynamics_takes_prev_stoch_params_as_input=True,
            train_student_posterior=False,
            always_use_student_posterior=False,
            residual_should_take_prev_stoch_params_as_input=False,
            use_relaxed_categorical_dist=False,
            relaxed_categorical_temperature=1.0,
            residual=None,
            residual_ensemble_size=7,
            **kw):
        self._deter = deter
        self._stoch = stoch
        self._classes = classes
        self._unroll = unroll
        self._initial = initial
        self._unimix = unimix
        self._action_clip = action_clip
        self._kw = kw

    def initial(self, bs):
        if self._classes:
            state = dict(
                deter=jnp.zeros([bs, self._deter], f32),
                logit=jnp.zeros([bs, self._stoch, self._classes], f32),
                stoch=jnp.zeros([bs, self._stoch, self._classes], f32),
                probs=jnp.zeros([bs, self._stoch, self._classes], f32))
        else:
            state = dict(
                deter=jnp.zeros([bs, self._deter], f32),
                mean=jnp.zeros([bs, self._stoch], f32),
                std=jnp.ones([bs, self._stoch], f32),
                stoch=jnp.zeros([bs, self._stoch], f32))
        if self._initial == 'zeros':
            return cast(state)
        elif self._initial == 'learned':
            deter = self.get('initial', jnp.zeros, state['deter'][0].shape, f32)
            state['deter'] = jnp.repeat(jnp.tanh(deter)[None], bs, 0)
            state['stoch'] = self.get_stoch(cast(state['deter']))
            return cast(state)
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None, with_residual=True, with_res_stop_gradients=True, with_student_posterior=False):
        swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])
        step = lambda prev, inputs: self.obs_step(prev[0], *inputs)
        inputs = swap(action), swap(embed), swap(is_first)
        start = state, state
        post, prior = jaxutils.scan(step, inputs, start, self._unroll)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine(self, action, state=None, with_residual=True, with_res_stop_gradients=True):
        swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
        state = self.initial(action.shape[0]) if state is None else state
        assert isinstance(state, dict), state
        action = swap(action)
        prior = jaxutils.scan(self.img_step, action, state, self._unroll)
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_dist(self, state, argmax=False):
        if self._classes:
            logit = state['logit'].astype(f32)
            return tfd.Independent(jaxutils.OneHotDist(logit), 1)
        else:
            mean = state['mean'].astype(f32)
            std = state['std'].astype(f32)
            return tfd.MultivariateNormalDiag(mean, std)

    def obs_step(self, prev_state, prev_action, embed, is_first, with_residual=True, with_res_stop_gradients=True, with_student_posterior=False):
        is_first = cast(is_first)
        prev_action = cast(prev_action)
        if self._action_clip > 0.0:
            prev_action *= sg(self._action_clip / jnp.maximum(
                self._action_clip, jnp.abs(prev_action)))
        prev_state, prev_action = jax.tree_util.tree_map(
            lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))
        prev_state = jax.tree_util.tree_map(
            lambda x, y: x + self._mask(y, is_first),
            prev_state, self.initial(len(is_first)))
        prior = self.img_step(prev_state, prev_action)
        print(f"posterior embed input shape: {embed.shape}")
        x = jnp.concatenate([prior['deter'], embed], -1)
        x = self.get('obs_out', Linear, **self._kw)(x)
        stats = self._stats('obs_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())
        post = {'stoch': stoch, 'deter': prior['deter'], **stats}
        return cast(post), cast(prior)

    def img_step(self, prev_state, prev_action, with_residual=False, with_res_stop_gradients=False, with_student_posterior=False):
        prev_stoch = prev_state['stoch']
        prev_action = cast(prev_action)
        if self._action_clip > 0.0:
            prev_action *= sg(self._action_clip / jnp.maximum(
                self._action_clip, jnp.abs(prev_action)))
        if self._classes:
            shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
            prev_stoch = prev_stoch.reshape(shape)
        if len(prev_action.shape) > len(prev_stoch.shape):  # 2D actions.
            shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
            prev_action = prev_action.reshape(shape)
        x = jnp.concatenate([prev_stoch, prev_action], -1)
        x = self.get('img_in', Linear, **self._kw)(x)
        x, deter = self._gru(x, prev_state['deter'])
        x = self.get('img_out', Linear, **self._kw)(x)
        stats = self._stats('img_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())
        prior = {'stoch': stoch, 'deter': deter, **stats}
        return cast(prior)

    def get_stoch(self, deter):
        x = self.get('img_out', Linear, **self._kw)(deter)
        stats = self._stats('img_stats', x)
        dist = self.get_dist(stats)
        return cast(dist.mode())

    def _gru(self, x, deter):
        x = jnp.concatenate([deter, x], -1)
        kw = {**self._kw, 'act': 'none', 'units': 3 * self._deter}
        x = self.get('gru', Linear, **kw)(x)
        reset, cand, update = jnp.split(x, 3, -1)
        reset = jax.nn.sigmoid(reset)
        cand = jnp.tanh(reset * cand)
        update = jax.nn.sigmoid(update - 1)
        deter = update * cand + (1 - update) * deter
        return deter, deter

    def _stats(self, name, x):
        if self._classes:
            x = self.get(name, Linear, self._stoch * self._classes)(x)
            logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
            probs = jax.nn.softmax(logit, -1)
            if self._unimix:
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - self._unimix) * probs + self._unimix * uniform
                logit = jnp.log(probs)
            stats = {'logit': logit, "probs": probs}
            return stats
        else:
            x = self.get(name, Linear, 2 * self._stoch)(x)
            mean, std = jnp.split(x, 2, -1)
            std = 2 * jax.nn.sigmoid(std / 2) + 0.1
            return {'mean': mean, 'std': std}

    def _mask(self, value, mask):
        return jnp.einsum('b...,b->b...', value, mask.astype(value.dtype))

    def dyn_loss(self, post, prior, is_first, impl='kl', free=1.0):
        if impl == 'kl':
            loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
        elif impl == 'logprob':
            loss = -self.get_dist(prior).log_prob(sg(post['stoch']))
        else:
            raise NotImplementedError(impl)
        if free:
            loss = jnp.maximum(loss, free)
        return loss

    def rep_loss(self, post, prior, is_first, impl='kl', free=1.0):
        if impl == 'kl':
            loss = self.get_dist(post).kl_divergence(self.get_dist(sg(prior)))
        elif impl == 'uniform':
            uniform = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), prior)
            loss = self.get_dist(post).kl_divergence(self.get_dist(uniform))
        elif impl == 'entropy':
            loss = -self.get_dist(post).entropy()
        elif impl == 'none':
            loss = jnp.zeros(post['deter'].shape[:-1])
        else:
            raise NotImplementedError(impl)
        if free:
            loss = jnp.maximum(loss, free)
        return loss


# class RSSMStochOnlyNonSequentialPosteriorMLPPriorLarger(RSSM):
#     # not default z-only dreamer
#     def initial(self, bs):
#         if self._classes:
#             state = dict(
#                 deter=jnp.zeros([bs, self._deter], f32),
#                 logit=jnp.zeros([bs, self._stoch, self._classes], f32),
#                 stoch=jnp.zeros([bs, self._stoch, self._classes], f32))
#         else:
#             state = dict(
#                 deter=jnp.zeros([bs, self._deter], f32),
#                 mean=jnp.zeros([bs, self._stoch], f32),
#                 std=jnp.ones([bs, self._stoch], f32),
#                 stoch=jnp.zeros([bs, self._stoch], f32))
#         if self._initial == 'zeros':
#             return cast(state)
#         elif self._initial == 'learned':
#             print(f"Using all zeroes for initial state even though 'learned' is specified for this.")
#             return cast(state)
#         else:
#             raise NotImplementedError(self._initial)
# 
#     def img_step(self, prev_state, prev_action):
#         prev_stoch = prev_state['stoch']
#         prev_action = cast(prev_action)
#         if self._action_clip > 0.0:
#             prev_action *= sg(self._action_clip / jnp.maximum(
#                 self._action_clip, jnp.abs(prev_action)))
#         if self._classes:
#             shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
#             prev_stoch = prev_stoch.reshape(shape)
#         if len(prev_action.shape) > len(prev_stoch.shape):  # 2D actions.
#             shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
#             prev_action = prev_action.reshape(shape)
# 
#         img_inputs = jnp.concatenate([prev_stoch, prev_action], -1)
# 
#         print(f"img_inputs shape: {img_inputs.shape}")
#         print(f"prev_action shape: {prev_action.shape}")
# 
#         x = self.get('img_in', Linear, **self._kw)(img_inputs)
#         x = self.get('img_hidden', Linear, **self._kw)(x)
#         x = self.get('img_hidden2', Linear, **self._kw)(x)
#         x = self.get('img_out', Linear, **self._kw)(x)
#         stats = self._stats('img_stats', x)
#         dist = self.get_dist(stats)
#         stoch = dist.sample(seed=nj.rng())
#         print(f"img stoch shape: {stoch.shape}")
# 
#         prior = {'stoch': stoch,
#                  'deter': prev_state['deter'],
#                  **stats}
# 
#         return cast(prior)
# 
#     def obs_step(self, prev_state, prev_action, embed, is_first):
#         is_first = cast(is_first)
#         prev_action = cast(prev_action)
#         if self._action_clip > 0.0:
#             prev_action *= sg(self._action_clip / jnp.maximum(
#                 self._action_clip, jnp.abs(prev_action)))
#         prev_state, prev_action = jax.tree_util.tree_map(
#             lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))
#         prev_state = jax.tree_util.tree_map(
#             lambda x, y: x + self._mask(y, is_first),
#             prev_state, self.initial(len(is_first)))
#         prior = self.img_step(prev_state, prev_action)
# 
#         # x = jnp.concatenate([prev_action, embed], -1)
#         # print(f"posterior is conditioned on embedding and previous action")
# 
#         x = embed
#         x = self.get('obs_out', Linear, **self._kw)(x)
#         stats = self._stats('obs_stats', x)
#         dist = self.get_dist(stats)
#         stoch = dist.sample(seed=nj.rng())
# 
#         print(f"obs stoch shape: {stoch.shape}")
# 
#         post = {'stoch': stoch,
#                 'deter': prev_state['deter'],
#                 **stats}
# 
#         return cast(post), cast(prior)
# 
#     def dyn_loss(self, post, prior, is_first, impl='kl', free=1.0):
#         if impl == 'kl':
#             loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
#         elif impl == 'kl_masked':
#             is_first = cast(is_first)
#             # if this is the first timestep then loss should be 0
#             loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
#             loss = jnp.where(is_first, 0, loss)
#         elif impl == 'logprob':
#             loss = -self.get_dist(prior).log_prob(sg(post['stoch']))
#         else:
#             raise NotImplementedError(impl)
#         if free:
#             loss = jnp.maximum(loss, free)
#         return loss
# 
#     def rep_loss(self, post, prior, is_first, impl='kl', free=1.0):
#         if impl == 'kl':
#             loss = self.get_dist(post).kl_divergence(self.get_dist(sg(prior)))
#         elif impl == 'kl_masked':
#             is_first = cast(is_first)
#             # if this is the first timestep then loss should be 0
#             loss = self.get_dist(post).kl_divergence(self.get_dist(sg(prior)))
#             loss = jnp.where(is_first, 0, loss)
#         elif impl == 'uniform':
#             uniform = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), prior)
#             loss = self.get_dist(post).kl_divergence(self.get_dist(uniform))
#         elif impl == 'entropy':
#             loss = -self.get_dist(post).entropy()
#         elif impl == 'none':
#             loss = jnp.zeros(post['deter'].shape[:-1])
#         else:
#             raise NotImplementedError(impl)
#         if free:
#             loss = jnp.maximum(loss, free)
#         return loss


class RSSMStochOnlyNonSequentialPosteriorMLPPriorLargerWithPriorBelief(RSSM):
     # default z only dreamer
    def __init__(
            self, deter=1024, stoch=32, classes=32, unroll=False, initial='learned',
            unimix=0.01, action_clip=1.0, img_hidden_layers=2, obs_hidden_layers=0,
            decode_stoch_params_as_deter=False,
            posterior_takes_prior_deter_as_input=False,
            posterior_takes_prior_stoch_as_input=False,
            use_gru_with_prior_belief=False,
            stoch_params_include_unimix=False,
            stoch_params_are_raw_logits=False,
            use_half_of_stoch_as_free_variables=False,
            use_posterior_stoch_params_for_first_state=False,
            use_posterior_stoch_params_for_all_states=False,
            dynamics_takes_prev_stoch_as_input=True,
            dynamics_takes_prev_stoch_params_as_input=True,
            train_student_posterior=False,
            always_use_student_posterior=False,
            residual_should_take_prev_stoch_params_as_input=False,
            use_relaxed_categorical_dist=False,
            relaxed_categorical_temperature=1.0,
            residual=None,
            residual_ensemble_size=7,
            **kw):
        self._deter = deter
        self._stoch = stoch
        self._classes = classes
        self._unroll = unroll
        self._initial = initial
        self._unimix = unimix
        self._action_clip = action_clip
        self._img_hidden_layers = img_hidden_layers
        self._obs_hidden_layers = obs_hidden_layers
        self._decode_stoch_params_as_deter = decode_stoch_params_as_deter
        self._posterior_takes_prior_deter_as_input = posterior_takes_prior_deter_as_input
        self._posterior_takes_prior_stoch_as_input = posterior_takes_prior_stoch_as_input
        self._use_gru_with_prior_belief = use_gru_with_prior_belief
        self._stoch_params_include_unimix = stoch_params_include_unimix
        self._stoch_params_are_raw_logits = stoch_params_are_raw_logits
        assert not (stoch_params_include_unimix and stoch_params_are_raw_logits), "can only choose one"
        self._use_half_of_stoch_as_free_variables = use_half_of_stoch_as_free_variables
        self._stoch_sample_size = self._stoch // 2 if self._use_half_of_stoch_as_free_variables else self._stoch
        self._use_posterior_stoch_params_for_first_state = use_posterior_stoch_params_for_first_state
        self._use_posterior_stoch_params_for_all_states = use_posterior_stoch_params_for_all_states
        self._dynamics_takes_prev_stoch_as_input = dynamics_takes_prev_stoch_as_input
        self._dynamics_takes_prev_stoch_params_as_input = dynamics_takes_prev_stoch_params_as_input
        self.train_student_posterior = train_student_posterior
        self._always_use_student_posterior = always_use_student_posterior

        self._use_relaxed_categorical_dist = use_relaxed_categorical_dist
        self._relaxed_categorical_temperature = relaxed_categorical_temperature

        if isinstance(residual, str):
            residual = get_residual(residual)
            if residual:
                residual = residual(stoch=stoch, classes=classes, unimix=self._unimix,
                                    stoch_params_are_raw_logits=stoch_params_are_raw_logits,
                                    stoch_params_include_unimix=stoch_params_include_unimix,
                                    img_hidden_layers=img_hidden_layers,
                                    should_take_prev_stoch_params_as_input=residual_should_take_prev_stoch_params_as_input,
                                    name='residual',
                                    ensemble_size=residual_ensemble_size,
                                    **kw)
        self.residual = residual
        self._kw = kw

    def initial(self, bs):
        if self._classes:
            state = dict(
                deter=jnp.zeros([bs, self._deter], f32),
                logit=jnp.zeros([bs, self._stoch, self._classes], f32),
                stoch=jnp.zeros([bs, self._stoch_sample_size, self._classes], f32),
                prior_stoch=jnp.zeros([bs, self._stoch_sample_size, self._classes], f32),
                stoch_params=jnp.zeros([bs, self._stoch * self._classes], f32),
                stoch_raw_logits=jnp.zeros([bs, self._stoch * self._classes], f32),
            )
        else:
            state = dict(
                deter=jnp.zeros([bs, self._deter], f32),
                mean=jnp.zeros([bs, self._stoch], f32),
                std=jnp.ones([bs, self._stoch], f32),
                stoch=jnp.zeros([bs, self._stoch_sample_size], f32),
                prior_stoch=jnp.zeros([bs, self._stoch_sample_size], f32),
                stoch_params=jnp.zeros([bs, self._stoch * 2], f32),
            )

        if self.residual is not None:
            state.update(self.residual.initial(bs))

        if self._decode_stoch_params_as_deter:
            state['deter'] = self.decode_stoch_params(stoch_params=state['stoch_params'])

        if self.train_student_posterior:
            state['student_logit'] = state['logit']

        if self._initial == 'zeros':
            return cast(state)
        elif self._initial == 'learned':
            if self._classes:
                orig_logit = self.get('initial', jnp.zeros, state['stoch_raw_logits'][0].shape, f32)
                orig_logit = jnp.repeat(orig_logit[None], bs, 0)

                logit = orig_logit.reshape(orig_logit.shape[:-1] + (self._stoch, self._classes))
                orig_probs = jax.nn.softmax(logit, -1)
                probs = orig_probs
                if self._unimix:
                    uniform = jnp.ones_like(probs) / probs.shape[-1]
                    probs = (1 - self._unimix) * probs + self._unimix * uniform
                    logit = jnp.log(probs)

                if self._stoch_params_are_raw_logits:
                    stoch_params = orig_logit
                else:
                    stoch_params = probs if self._stoch_params_include_unimix else orig_probs
                    stoch_params = stoch_params.reshape(stoch_params.shape[:-2] + (self._stoch * self._classes,))

                stats = {'logit': logit,
                         'stoch_params': stoch_params,
                         'stoch_raw_logits': orig_logit
                         }
                dist = self.get_dist(stats)
                stoch = dist.mode()

                state['stoch'] = stoch.reshape(state['stoch'].shape)
                state['prior_stoch'] = state['stoch']
                state['stoch_params'] = stoch_params.reshape(state['stoch_params'].shape)
                state['stoch_raw_logits'] = orig_logit.reshape(state['stoch_raw_logits'].shape)
            else:
                orig_mean = self.get('initial_mean', jnp.zeros, state['mean'][0].shape, f32)
                orig_mean = jnp.repeat(orig_mean[None], bs, 0)

                orig_std = self.get('initial_std', jnp.ones, state['std'][0].shape, f32)
                orig_std = jnp.repeat(orig_std[None], bs, 0)

                stoch = tfd.MultivariateNormalDiag(orig_mean, orig_std).mode()
                state = {
                    'mean': orig_mean,
                    'std': orig_std,
                    'stoch_params': jnp.concatenate([orig_mean, orig_std], axis=-1),
                    'stoch': stoch,
                    'prior_stoch': stoch,
                    'deter': state['deter']
                }
            return cast(state)
        else:
            raise NotImplementedError(self._initial)

    def get_dist(self, state, argmax=False, use_all_variables=False):
        if self._classes:
            logit = state['logit'].astype(f32)

            if self._use_relaxed_categorical_dist:
                return tfd.Independent(jaxutils.RelaxedOnehotCategoricalDist(logit, temperature=self._relaxed_categorical_temperature), 1)
            else:
                if self._use_half_of_stoch_as_free_variables and not use_all_variables:
                    logit, free_logit = jnp.split(logit, 2, axis=-2)
                return tfd.Independent(jaxutils.OneHotDist(logit), 1)
        else:
            mean = state['mean'].astype(f32)
            std = state['std'].astype(f32)
            if self._use_half_of_stoch_as_free_variables and not use_all_variables:
                mean, free_mean = jnp.split(mean, 2, axis=-1)
                std, free_std = jnp.split(std, 2, axis=-1)
            return tfd.MultivariateNormalDiag(mean, std)

    def _stats(self, name, x):
        if self._classes:
            x = self.get(name, Linear, self._stoch * self._classes)(x)
            orig_logit = x
            logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
            orig_probs = jax.nn.softmax(logit, -1)
            probs = orig_probs
            if self._unimix:
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - self._unimix) * probs + self._unimix * uniform
                logit = jnp.log(probs)

            if self._stoch_params_are_raw_logits:
                stoch_params = orig_logit
            else:
                stoch_params = probs if self._stoch_params_include_unimix else orig_probs
                stoch_params = stoch_params.reshape(stoch_params.shape[:-2] + (self._stoch * self._classes,))

            stats = {'logit': logit,
                     'stoch_params': stoch_params,
                     'stoch_raw_logits': orig_logit
                     }
            return stats
        else:
            x = self.get(name, Linear, 2 * self._stoch)(x)
            mean, std = jnp.split(x, 2, -1)
            std = 2 * jax.nn.sigmoid(std / 2) + 0.1
            return {'mean': mean,
                    'std': std,
                    'stoch_params': jnp.concatenate([mean, std], axis=-1),
                    }

    def observe(self, embed, action, is_first, state=None, with_residual=True, with_res_stop_gradients=True, with_student_posterior=False):
        swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])
        step = lambda prev, inputs: self.obs_step(prev[0], *inputs, with_residual=with_residual, with_res_stop_gradients=with_res_stop_gradients, with_student_posterior=with_student_posterior)
        inputs = swap(action), swap(embed), swap(is_first)
        start = state, state
        post, prior = jaxutils.scan(step, inputs, start, self._unroll)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine(self, action, state=None):
        swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
        state = self.initial(action.shape[0]) if state is None else state
        assert isinstance(state, dict), state
        action = swap(action)
        step = partial(self.img_step, with_residual=True, with_res_stop_gradients=False)
        prior = jaxutils.scan(step, action, state, self._unroll)
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def decode_stoch_params(self, stoch_params):
        x = self.get('decode_stoch_in', Linear, **self._kw)(stoch_params)
        decoded_stoch_params = self.get("decode_stoch_out", Linear, self._deter, act='tanh')(x)
        return decoded_stoch_params

    def img_step(self, prev_state, prev_action, with_residual=True, with_res_stop_gradients=False):
        prev_stoch = prev_state['stoch']
        prev_stoch_params = prev_state['stoch_params']
        prev_action = cast(prev_action)
        if self._action_clip > 0.0:
            prev_action *= sg(self._action_clip / jnp.maximum(
                self._action_clip, jnp.abs(prev_action)))
        if self._classes:
            shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
            prev_stoch = prev_stoch.reshape(shape)
        if len(prev_action.shape) > len(prev_stoch.shape):  # 2D actions.
            shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
            prev_action = prev_action.reshape(shape)

        # print(f"img_inputs shape: {img_inputs.shape}")
        # print(f"prev_action shape: {prev_action.shape}")

        if self._use_gru_with_prior_belief:
            x = jnp.concatenate([prev_stoch, prev_action], -1)
            x = self.get('img_in', Linear, **self._kw)(x)
            stats = self._gru(x, deter=prev_stoch_params)
        else:

            img_inputs = [prev_action]
            if self._dynamics_takes_prev_stoch_params_as_input:
                if self._decode_stoch_params_as_deter:
                    img_inputs.append(prev_state['deter'])
                else:
                    img_inputs.append(prev_stoch_params)

            if self._dynamics_takes_prev_stoch_as_input:
                img_inputs.append(prev_stoch)
            else:
                assert False, "(author1) debugging assert, ok to remove, but I expect this case should be never used"

            x = jnp.concatenate(img_inputs, -1)

            x = self.get('img_in', Linear, **self._kw)(x)
            for i in range(self._img_hidden_layers):
                x = self.get(f'img_hidden{i+1}', Linear, **self._kw)(x)
            x = self.get('img_out', Linear, **self._kw)(x)
            stats = self._stats('img_stats', x)

        if with_residual and self.residual:
            stats = self.get_prior_residual_correction(
                with_res_stop_gradients=with_res_stop_gradients,
                prev_stoch=prev_stoch,
                prev_action=prev_action,
                prev_stoch_params=prev_state['deter'] if self._decode_stoch_params_as_deter else prev_stoch_params,
                prior_stoch_params=stats['stoch_params'],
                prior_stoch_raw_logits=stats['stoch_raw_logits']
            )
        elif self.residual:
            stats.update(self.residual.initial(bs=prev_action.shape[0]))
        
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())
        print(f"img stoch shape: {stoch.shape}")

        prior = {'stoch': stoch,
                 'prior_stoch': stoch,
                 # 'prior_stoch_params': stats['stoch_params'],
                 **stats}
        
        if self.train_student_posterior:
            prior['student_logit'] = prior['logit']
        
        if self._decode_stoch_params_as_deter:
            prior['deter'] = self.decode_stoch_params(stoch_params=stats['stoch_params'])
        else:
            assert 'deter' not in prior
            prior['deter'] = prev_state['deter']

        return cast(prior)

    def _gru(self, x, deter):
        raise NotImplementedError
        # x = jnp.concatenate([deter, x], -1)
        #
        # if self._classes:
        #     gru_size = self._stoch * self._classes
        # else:
        #     gru_size = 2 * self._stoch
        #
        # kw = {**self._kw, 'act': 'none', 'units': 3 * gru_size}
        # x = self.get('gru', Linear, **kw)(x)
        # reset, cand, update = jnp.split(x, 3, -1)
        # reset = jax.nn.sigmoid(reset)
        # # cand = jnp.tanh(reset * cand)
        #
        # if self._classes:
        #     # TODO this implementation for a discrete z dist is done totally wrong here for GRU
        #     x = cand
        #     logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
        #     probs = jax.nn.sigmoid(logit, -1)
        #     if self._unimix:
        #         uniform = jnp.ones_like(probs) / probs.shape[-1]
        #         probs = (1 - self._unimix) * probs + self._unimix * uniform
        #     cand_stoch_params = probs.reshape(probs.shape[:-2] + (self._stoch * self._classes,))
        # else:
        #     x = cand
        #     mean, std = jnp.split(x, 2, -1)
        #     std = 2 * jax.nn.sigmoid(std / 2) + 0.1
        #     cand_stoch_params = jnp.concatenate([mean, std], -1)
        #
        # cand_stoch_params = reset * cand_stoch_params
        # update = jax.nn.sigmoid(update - 1)
        # new_stoch_params = update * cand_stoch_params + (1 - update) * deter
        #
        # if self._classes:
        #     # TODO this implementation for a discrete z dist is done totally wrong here for GRU
        #     # renormalize probs
        #     probs = new_stoch_params.reshape(new_stoch_params.shape[:-1] + (self._stoch, self._classes))
        #     probs_sum = jnp.sum(probs, axis=-1)
        #     probs = probs / probs_sum
        #     logit = jnp.log(probs)
        #     new_stoch_params = probs.reshape(probs.shape[:-2] + (self._stoch * self._classes,))
        #     stats = {'logit': logit,
        #              'stoch_params': new_stoch_params,
        #              }
        #     return stats
        # else:
        #     mean, std = jnp.split(new_stoch_params, 2, -1)
        #     return {'mean': mean, 'std': std,
        #             'stoch_params': new_stoch_params
        #             }

    def obs_step(self, prev_state, prev_action, embed, is_first, with_residual=True, with_res_stop_gradients=True, with_student_posterior=False):
        is_first = cast(is_first)
        prev_action = cast(prev_action)
        if self._action_clip > 0.0:
            prev_action *= sg(self._action_clip / jnp.maximum(
                self._action_clip, jnp.abs(prev_action)))

        prev_state, prev_action = jax.tree_util.tree_map(
            lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))
        prev_state = jax.tree_util.tree_map(
            lambda x, y: x + self._mask(y, is_first),
            prev_state, self.initial(len(is_first)))
        prior = self.img_step(prev_state, prev_action,
                              with_residual=with_residual,
                              with_res_stop_gradients=with_res_stop_gradients)
        # x = jnp.concatenate([prev_action, embed], -1)
        # print(f"posterior is conditioned on embedding and previous action")
        print(f"posterior embed input shape: {embed.shape}")
        if self._posterior_takes_prior_deter_as_input:
            x = jnp.concatenate([prior['deter'], embed], -1)
        elif self._posterior_takes_prior_stoch_as_input:
            prior_stoch = prior['stoch']
            assert not self._posterior_takes_prior_deter_as_input
            if self._classes:
                prior_stoch = prior_stoch.reshape(prior_stoch.shape[:-2] + (self._stoch * self._classes,))
            x = jnp.concatenate([prior_stoch, embed], -1)
            print(f"posterior_takes_prior_stoch_as_input, posterior input shape: {x.shape}")
        else:
            x = embed

        for i in range(self._obs_hidden_layers):
            x = self.get(f'obs_hidden{i + 1}', Linear, **self._kw)(x)
        x = self.get('obs_out', Linear, **self._kw)(x)
        stats = self._stats('obs_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        print(f"obs stoch shape: {stoch.shape}")

        student_stats = None
        if self.train_student_posterior:
            # # TODO remove below ----------------------------------
            # if self._posterior_takes_prior_deter_as_input:
            #     x = jnp.concatenate([prior['deter'], embed], -1)
            # else:
            #     x = embed
            # # TODO REMOVE^^^^^^^^^^^^^^^^^^^^^^
            x = embed

            kw = {**self._kw}
            kw['units'] = kw['units'] * 2
            for i in range(self._obs_hidden_layers):
                x = self.get(f'student_obs_hidden_a_{i + 1}', Linear, **kw)(x)
            for i in range(self._obs_hidden_layers):
                x = self.get(f'student_obs_hidden_b_{i + 1}', Linear, **kw)(x)
            x = self.get('student_obs_out', Linear, **kw)(x)
            student_stats = self._stats('student_obs_stats', x)
            if with_student_posterior or self._always_use_student_posterior:
                stats = student_stats
                dist = self.get_dist(student_stats)
                stoch = dist.sample(seed=nj.rng())
        
        if self._use_posterior_stoch_params_for_first_state:
            stoch_params = jnp.where(jnp.tile(is_first[:, None], (1, stats['stoch_params'].shape[-1])),
                                           stats['stoch_params'], prior['stoch_params'])
            if self._classes:
                stoch_raw_logits = jnp.where(jnp.tile(is_first[:, None], (1, stats['stoch_raw_logits'].shape[-1])),
                                               stats['stoch_raw_logits'], prior['stoch_raw_logits'])
        elif self._use_posterior_stoch_params_for_all_states:
            stoch_params = stats['stoch_params']
            if self._classes:
                stoch_raw_logits = stats['stoch_raw_logits']
        else:
            stoch_params = prior['stoch_params']
            if self._classes:
                stoch_raw_logits = prior['stoch_raw_logits']

        del stats['stoch_params']
        if self._classes:
            del stats['stoch_raw_logits']

        post = {'stoch': stoch,
                'prior_stoch': prior['stoch'],
                # 'prior_stoch_params': stoch_params,
                'stoch_params': stoch_params,
                **stats}

        if self._classes:
            post['stoch_raw_logits'] = stoch_raw_logits

        if self.train_student_posterior:
            post['student_logit'] = student_stats['logit']

        if self._decode_stoch_params_as_deter:
            post['deter'] = self.decode_stoch_params(stoch_params=stoch_params)
        else:
            assert 'deter' not in post
            post['deter'] = prev_state['deter']


        if self.residual and self.residual.residual_stats_key():
            if with_residual:
                post[self.residual.residual_stats_key()] = prior[self.residual.residual_stats_key()]
            else:
                post.update(self.residual.initial(bs=is_first.shape[0]))

        return cast(post), cast(prior)


    def dyn_loss(self, post, prior, is_first, impl='kl', free=1.0):
        if impl == 'kl':
            loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
        elif impl == 'kl_masked':
            is_first = cast(is_first)
            # if this is the first timestep then loss should be 0
            loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
            loss = jnp.where(is_first, 0, loss)
        elif impl == 'logprob':
            loss = -self.get_dist(prior).log_prob(sg(post['stoch']))
        else:
            raise NotImplementedError(impl)
        if free:
            loss = jnp.maximum(loss, free)
        return loss

    def rep_loss(self, post, prior, is_first, impl='kl', free=1.0):
        if impl == 'kl':
            loss = self.get_dist(post, use_all_variables=True).kl_divergence(self.get_dist(sg(prior), use_all_variables=True))
        elif impl == 'kl_masked':
            is_first = cast(is_first)
            # if this is the first timestep then loss should be 0
            loss = self.get_dist(post, use_all_variables=True).kl_divergence(self.get_dist(sg(prior), use_all_variables=True))
            loss = jnp.where(is_first, 0, loss)
        elif impl == 'uniform':
            uniform = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), prior)
            loss = self.get_dist(post, use_all_variables=True).kl_divergence(self.get_dist(uniform, use_all_variables=True))
        elif impl == 'entropy':
            loss = -self.get_dist(post, use_all_variables=True).entropy()
        elif impl == 'none':
            loss = jnp.zeros(post['deter'].shape[:-1])
        else:
            raise NotImplementedError(impl)
        if free:
            print(f"loss shape before free bits: {loss.shape}")
            loss = jnp.maximum(loss, free)
        return loss

    def get_prior_residual_correction(self, with_res_stop_gradients, prev_stoch, prev_action, prev_stoch_params, prior_stoch_params, prior_stoch_raw_logits):
        if with_res_stop_gradients:
            prev_stoch = sg(prev_stoch)
            prev_action = sg(prev_action)
            prev_stoch_params = sg(prev_stoch_params)
            prior_stoch_params = sg(prior_stoch_params)
            prior_stoch_raw_logits = sg(prior_stoch_raw_logits)

        corrected_prior_stats = self.residual(
            prev_stoch=prev_stoch,
            prev_stoch_params=prev_stoch_params,
            prior_stoch_params=prior_stoch_params,
            prev_action=prev_action,
            prior_stoch_raw_logits=prior_stoch_raw_logits
        )
        return corrected_prior_stats


# class EnsembleResiduaOld(nj.Module):
# 
#     def __init__(self, stoch, classes, unimix, stoch_params_are_raw_logits, stoch_params_include_unimix, img_hidden_layers, **kw):
#         self._stoch = stoch
#         self._classes = classes
#         self._unimix = unimix
#         self._kw = kw
#         self._ensemble_size = 7
# 
#     def initial(self, bs):
#         state = dict(
#             ensemble_logits_for_each_member=jnp.zeros([bs, self._ensemble_size, self._stoch, self._classes], f32))
#         return state
# 
#     def residual_stats_key(self):
#         return 'ensemble_logits_for_each_member'
# 
#     def get_dist(self, logit):
#         logit = logit.astype(f32)
#         return tfd.Independent(jaxutils.OneHotDist(logit), 1)
# 
#     def _ensemble_member(self, i, prev_stoch, prev_stoch_params, prior_stoch_params, prev_action):
#         inputs = jnp.concatenate([
#             prev_stoch_params,
#             # prev_stoch,
#             # prior_stoch_params,
#             prev_action
#         ], -1)
#         print(f"residual Linear layer kw: {self._kw}")
#         x = self.get(f'residual_in_{i}', Linear, **self._kw)(inputs)
#         x = self.get(f'residual_hidden_{i}', Linear, **self._kw)(inputs)
#         x = self.get(f'residual_out_{i}', Linear, self._stoch * self._classes)(x)
# 
#         prev_logits = jnp.log(prior_stoch_params)
#         new_logits = prev_logits + x
#         new_logits = new_logits.reshape(x.shape[:-1] + (self._stoch, self._classes))
#         probs = jax.nn.softmax(new_logits, -1)
# 
#         if self._unimix:
#             uniform = jnp.ones_like(probs) / probs.shape[-1]
#             probs = (1 - self._unimix) * probs + self._unimix * uniform
#             # logit = jnp.log(probs)
# 
#         # jax.debug.print("prev_stoch_params has nan:\t\t{}", jnp.any(jnp.isnan(prev_stoch_params)))
#         # jax.debug.print("prev_action has nan:\t\t{}", jnp.any(jnp.isnan(prev_action)))
#         # jax.debug.print("prior_stoch_params has nan:\t\t{}", jnp.any(jnp.isnan(prior_stoch_params)))
#         # jax.debug.print("new_logits has nan:\t\t{}", jnp.any(jnp.isnan(new_logits)))
#         # jax.debug.print("probs has nan:\t\t{}", jnp.any(jnp.isnan(probs)))
#         return probs
# 
#     def __call__(self, prev_stoch, prev_stoch_params, prior_stoch_params, prev_action, prior_stoch_raw_logits):
#         map_fn = partial(self._ensemble_member,
#                          prev_stoch=prev_stoch,
#                          prev_stoch_params=prev_stoch_params,
#                          prior_stoch_params=prior_stoch_params,
#                          prev_action=prev_action)
#         ensemble_probs_for_each_member = []
#         for i in range(self._ensemble_size):
#             ensemble_probs = map_fn(i)
#             # jax.debug.print("ensemble_probs dont sum to 1:\t\t{}, error {}",
#             #                 jnp.logical_not(jnp.all(jnp.isclose(jnp.sum(ensemble_probs, axis=-1), 1.0))),
#             #                       jnp.mean(jnp.abs(jnp.sum(ensemble_probs, axis=-1) - 1.0))
#             #                 )
# 
#             ensemble_probs_for_each_member.append(ensemble_probs)
#         ensemble_probs_for_each_member = jnp.asarray(ensemble_probs_for_each_member, dtype=jaxutils.COMPUTE_DTYPE)
#         # print(f"ensemble_probs_for_each_member: {jnp.shape(ensemble_probs_for_each_member)}")
#         ensemble_logits_for_each_member = jnp.log(ensemble_probs_for_each_member)
#         # jax.debug.print("ensemble_probs_for_each_member min:\t\t{}", jnp.min(ensemble_probs_for_each_member))
#         #
#         # jax.debug.print("ensemble_probs_for_each_member is inf:\t\t{}: {}", jnp.any(jnp.isinf(ensemble_probs_for_each_member)), ensemble_probs_for_each_member[0, 0])
#         # jax.debug.print("ensemble_logits_for_each_member is inf:\t\t{}: {}", jnp.any(jnp.isinf(ensemble_logits_for_each_member)), ensemble_logits_for_each_member[0, 0])
# 
# 
# 
# 
#         # jax.debug.print("ensemble_probs_for_each_member has nan:\t\t{}", jnp.any(jnp.isnan(ensemble_probs_for_each_member)))
#         # jax.debug.print("ensemble_probs_for_each_member:\n{}", ensemble_probs_for_each_member[0, 0])
# 
#         # jax.debug.print("ensemble_logits_for_each_member has nan:\t\t{}", jnp.any(jnp.isnan(ensemble_logits_for_each_member)))
# 
#         avg_ensemble_probs = jnp.mean(ensemble_probs_for_each_member, axis=0)
# 
#         # avg_ensemble_probs = jnp.where(avg_ensemble_probs <= 1e-8, 1e-8, avg_ensemble_probs)
#         avg_ensemble_probs = avg_ensemble_probs / avg_ensemble_probs.sum(axis=-1, keepdims=True)
# 
#         # jax.debug.print("mean has nan:\t\t{}", jnp.any(jnp.isnan(avg_ensemble_probs)))
#         # jax.debug.print("avg_ensemble_probs dont sum to 1:\t\t{}",
#         #                 jnp.logical_not(jnp.all(jnp.isclose(jnp.sum(avg_ensemble_probs, axis=-1), 1.0))))
# 
#         print(f"avg_ensemble_probs: {avg_ensemble_probs.shape}")
# 
#         logit = jnp.log(avg_ensemble_probs)
# 
# 
#         stoch_params = avg_ensemble_probs
#         stoch_params = stoch_params.reshape(stoch_params.shape[:-2] + (self._stoch * self._classes,))
#         print(f"logit: {logit.shape}")
#         print(f"stoch_params: {stoch_params.shape}")
#         # jax.debug.print("logit after mean has nan:\t\t{} inf: {}", jnp.any(jnp.isnan(logit)), jnp.any(jnp.isinf(logit)))
#         # jax.debug.print("stoch_params has nan:\t\t{} inf: {}", jnp.any(jnp.isnan(stoch_params)), jnp.any(jnp.isinf(stoch_params)))
# 
#         stats = {'logit': logit, 'stoch_params': stoch_params, 'ensemble_logits_for_each_member': ensemble_logits_for_each_member.transpose((1, 0, 2, 3))}
#         return stats
# 
#     def residual_loss(self, prior, post, is_first, free=1.0):
#         is_first = cast(is_first)
#         total_loss = jnp.zeros(post['deter'].shape[:-1])
#         for ensemble_logit in prior['ensemble_logits_for_each_member'].transpose((2, 0, 1, 3, 4)):
#             print(f'ensemble_logit: {ensemble_logit.shape}')
#             # jax.debug.print("prior has nan:\t\t{}", jnp.any(jnp.isnan(ensemble_logit)))
#             # jax.debug.print("prior is inf:\t\t{}", jnp.any(jnp.isinf(ensemble_logit)))
# 
#             loss = self.get_dist(sg(post['logit'])).kl_divergence(self.get_dist(ensemble_logit))
#             # jax.debug.print("loss has nan:\t\t{}", jnp.any(jnp.isnan(loss)))
#             # jax.debug.print("loss is inf:\t\t{}", jnp.any(jnp.isinf(loss)))
# 
#             # if this is the first timestep then loss should be 0
#             loss = jnp.where(is_first, 0, loss)
# 
#             # TODO removing free bits because the representation is already fixed and we just want to match it?
#             # if free:
#             #     loss = jnp.maximum(loss, free)
#             total_loss += loss
#         return total_loss


class EnsembleResidual(nj.Module):

    def __init__(self, stoch, classes, unimix, stoch_params_are_raw_logits, stoch_params_include_unimix, img_hidden_layers,
                 should_take_prev_stoch_params_as_input=False, ensemble_size=7,
                 **kw):
        self._stoch = stoch
        self._classes = classes
        self._unimix = unimix
        self._stoch_params_are_raw_logits = stoch_params_are_raw_logits
        self._stoch_params_include_unimix = stoch_params_include_unimix
        self._img_hidden_layers = img_hidden_layers
        self._should_take_prev_stoch_params_as_input = should_take_prev_stoch_params_as_input
        self._ensemble_size = ensemble_size
        self._kw = kw
        print(f"Ensemble size: {ensemble_size}")

    def initial(self, bs):
        state = dict(
            ensemble_logits_for_each_member=jnp.zeros([bs, self._ensemble_size, self._stoch, self._classes], f32))
        return state

    def residual_stats_key(self):
        return 'ensemble_logits_for_each_member'

    def get_dist(self, logit):
        logit = logit.astype(f32)
        return tfd.Independent(jaxutils.OneHotDist(logit), 1)

    def _ensemble_member(self, i, prev_stoch, prev_stoch_params, prior_stoch_params, prev_action):
        input_list = [prev_stoch, prev_action]
        if self._should_take_prev_stoch_params_as_input:
            input_list.append(prev_stoch_params)
        inputs = jnp.concatenate(input_list, -1)

        print(f"residual Linear layer kw: {self._kw}")
        x = self.get(f'residual_in_{i}', Linear, **self._kw)(inputs)
        x = self.get(f'residual_hidden_{i}', Linear, **self._kw)(x)
        x = self.get(f'residual_out_{i}', Linear, self._stoch * self._classes)(x)
        logit_correction = x
        return logit_correction

    def __call__(self, prev_stoch, prev_stoch_params, prior_stoch_params, prev_action, prior_stoch_raw_logits):
        map_fn = partial(self._ensemble_member,
                         prev_stoch=prev_stoch,
                         prev_stoch_params=prev_stoch_params,
                         prior_stoch_params=prior_stoch_params,
                         prev_action=prev_action)
        ensemble_logit_corrections = []
        for i in range(self._ensemble_size):
            ensemble_logit_correction = map_fn(i)
            ensemble_logit_corrections.append(ensemble_logit_correction)
        ensemble_logit_corrections = jnp.asarray(ensemble_logit_corrections, dtype=jaxutils.COMPUTE_DTYPE)
        avg_ensemble_logit_corrections = jnp.mean(ensemble_logit_corrections, axis=0)

        ensemble_logits_for_each_member = ensemble_logit_corrections + prior_stoch_raw_logits
        gathered_logits = prior_stoch_raw_logits + avg_ensemble_logit_corrections

        ensemble_logits_for_each_member = ensemble_logits_for_each_member.reshape(ensemble_logits_for_each_member.shape[:-1] + (self._stoch, self._classes))
        gathered_logits = gathered_logits.reshape(gathered_logits.shape[:-1] + (self._stoch, self._classes))

        gathered_probs = jax.nn.softmax(gathered_logits, -1)

        if self._stoch_params_are_raw_logits:
            stoch_params = gathered_logits
        else:
            stoch_params = gathered_probs
        stoch_params = stoch_params.reshape(stoch_params.shape[:-2] + (self._stoch * self._classes,))

        stats = {'logit': gathered_logits,
                 'stoch_params': stoch_params,
                 'stoch_raw_logits': gathered_logits.reshape(gathered_logits.shape[:-2] + (self._stoch * self._classes,)),
                 'ensemble_logits_for_each_member': ensemble_logits_for_each_member.transpose((1, 0, 2, 3)),
                 }
        return stats

    def residual_loss(self, prior, post, is_first, free=1.0):
        is_first = cast(is_first)
        total_loss = jnp.zeros(post['deter'].shape[:-1])
        for ensemble_logit in prior['ensemble_logits_for_each_member'].transpose((2, 0, 1, 3, 4)):
            print(f'ensemble_logit: {ensemble_logit.shape}')
            # jax.debug.print("prior has nan:\t\t{}", jnp.any(jnp.isnan(ensemble_logit)))
            # jax.debug.print("prior is inf:\t\t{}", jnp.any(jnp.isinf(ensemble_logit)))

            loss = self.get_dist(sg(post['logit'])).kl_divergence(self.get_dist(ensemble_logit))
            # jax.debug.print("loss has nan:\t\t{}", jnp.any(jnp.isnan(loss)))
            # jax.debug.print("loss is inf:\t\t{}", jnp.any(jnp.isinf(loss)))

            # if this is the first timestep then loss should be 0
            loss = jnp.where(is_first, 0, loss)

            # TODO removing free bits because the representation is already fixed and we just want to match it?
            # if free:
            #     loss = jnp.maximum(loss, free)
            total_loss += loss
        return total_loss


class EnsembleResidualFreebits(EnsembleResidual):

    def residual_loss(self, prior, post, is_first, free=1.0):
        is_first = cast(is_first)
        total_loss = jnp.zeros(post['deter'].shape[:-1])
        for ensemble_logit in prior['ensemble_logits_for_each_member'].transpose((2, 0, 1, 3, 4)):
            print(f'ensemble_logit: {ensemble_logit.shape}')
            # jax.debug.print("prior has nan:\t\t{}", jnp.any(jnp.isnan(ensemble_logit)))
            # jax.debug.print("prior is inf:\t\t{}", jnp.any(jnp.isinf(ensemble_logit)))

            loss = self.get_dist(sg(post['logit'])).kl_divergence(self.get_dist(ensemble_logit))
            # jax.debug.print("loss has nan:\t\t{}", jnp.any(jnp.isnan(loss)))
            # jax.debug.print("loss is inf:\t\t{}", jnp.any(jnp.isinf(loss)))

            # if this is the first timestep then loss should be 0
            loss = jnp.where(is_first, 0, loss)
            if free:
                loss = jnp.maximum(loss, free)
            total_loss += loss
        return total_loss

class EnsembleResidualSmall(EnsembleResidual):
    def _ensemble_member(self, i, prev_stoch, prev_stoch_params, prior_stoch_params, prev_action):
        input_list = [prev_stoch, prev_action]
        if self._should_take_prev_stoch_params_as_input:
            input_list.append(prev_stoch_params)
        inputs = jnp.concatenate(input_list, -1)
        print(f"residual Linear layer kw: {self._kw}")
        x = self.get(f'residual_in_{i}', Linear, **self._kw)(inputs)
        x = self.get(f'residual_out_{i}', Linear, self._stoch * self._classes)(x)
        logit_correction = x
        return logit_correction


class EnsembleResidualExtraSmall(EnsembleResidual):

    def _ensemble_member(self, i, prev_stoch, prev_stoch_params, prior_stoch_params, prev_action):
        input_list = [prev_stoch, prev_action]
        if self._should_take_prev_stoch_params_as_input:
            input_list.append(prev_stoch_params)
        inputs = jnp.concatenate(input_list, -1)
        print(f"residual Linear layer kw: {self._kw}")
        kw = self._kw.copy()
        kw['units'] = kw['units'] // 2
        x = self.get(f'residual_in_{i}', Linear, **kw)(inputs)
        x = self.get(f'residual_out_{i}', Linear, self._stoch * self._classes)(x)
        logit_correction = x
        return logit_correction


class EnsembleResidualExtraSmall10Members(EnsembleResidualExtraSmall):

    def __init__(self, stoch, classes, unimix, stoch_params_are_raw_logits, stoch_params_include_unimix, img_hidden_layers,
                 should_take_prev_stoch_params_as_input=False, ensemble_size=10,
                 **kw):
        ensemble_size = 10
        super().__init__(stoch=stoch, classes=classes, unimix=unimix, stoch_params_are_raw_logits=stoch_params_are_raw_logits,
                         stoch_params_include_unimix=stoch_params_include_unimix, img_hidden_layers=img_hidden_layers,
                         should_take_prev_stoch_params_as_input=should_take_prev_stoch_params_as_input,
                         ensemble_size=ensemble_size,
                         **kw)
        assert self._ensemble_size == 10, self._ensemble_size


class EnsembleResidualExtraSmall4Members(EnsembleResidualExtraSmall):

    def __init__(self, stoch, classes, unimix, stoch_params_are_raw_logits, stoch_params_include_unimix,
                 img_hidden_layers,
                 should_take_prev_stoch_params_as_input=False, ensemble_size=4,
                 **kw):
        ensemble_size = 4
        super().__init__(stoch=stoch, classes=classes, unimix=unimix,
                         stoch_params_are_raw_logits=stoch_params_are_raw_logits,
                         stoch_params_include_unimix=stoch_params_include_unimix, img_hidden_layers=img_hidden_layers,
                         should_take_prev_stoch_params_as_input=should_take_prev_stoch_params_as_input,
                         ensemble_size=ensemble_size,
                         **kw)
        assert self._ensemble_size == 4, self._ensemble_size


class EnsembleResidualExtraSmall1Member(EnsembleResidualExtraSmall):

    def __init__(self, stoch, classes, unimix, stoch_params_are_raw_logits, stoch_params_include_unimix,
                 img_hidden_layers,
                 should_take_prev_stoch_params_as_input=False, ensemble_size=1,
                 **kw):
        ensemble_size = 1
        super().__init__(stoch=stoch, classes=classes, unimix=unimix,
                         stoch_params_are_raw_logits=stoch_params_are_raw_logits,
                         stoch_params_include_unimix=stoch_params_include_unimix, img_hidden_layers=img_hidden_layers,
                         should_take_prev_stoch_params_as_input=should_take_prev_stoch_params_as_input,
                         ensemble_size=ensemble_size,
                         **kw)
        assert self._ensemble_size == 1, self._ensemble_size


class EnsembleResidualExtraSmall1MemberConditionedOnPriorStochParams(EnsembleResidualExtraSmall1Member):

    def _ensemble_member(self, i, prev_stoch, prev_stoch_params, prior_stoch_params, prev_action):
        input_list = [prev_stoch, prev_action, prior_stoch_params]
        if self._should_take_prev_stoch_params_as_input:
            input_list.append(prev_stoch_params)
        inputs = jnp.concatenate(input_list, -1)
        print(f"residual Linear layer kw: {self._kw}")
        kw = self._kw.copy()
        kw['units'] = kw['units'] // 2
        x = self.get(f'residual_in_{i}', Linear, **kw)(inputs)
        x = self.get(f'residual_out_{i}', Linear, self._stoch * self._classes)(x)
        logit_correction = x
        return logit_correction

class EnsembleResidualExtraSmallConditionedOnPrevStochParamsOnly(EnsembleResidual):
    def _ensemble_member(self, i, prev_stoch, prev_stoch_params, prior_stoch_params, prev_action):
        input_list = [prev_stoch_params, prev_action]
        inputs = jnp.concatenate(input_list, -1)
        print(f"residual Linear layer kw: {self._kw}")
        kw = self._kw.copy()
        kw['units'] = kw['units'] // 2
        x = self.get(f'residual_in_{i}', Linear, **kw)(inputs)
        x = self.get(f'residual_out_{i}', Linear, self._stoch * self._classes)(x)
        logit_correction = x
        return logit_correction

class EnsembleResidual4xSmall(EnsembleResidual):
    def _ensemble_member(self, i, prev_stoch, prev_stoch_params, prior_stoch_params, prev_action):
        input_list = [prev_stoch, prev_action]
        if self._should_take_prev_stoch_params_as_input:
            input_list.append(prev_stoch_params)
        inputs = jnp.concatenate(input_list, -1)
        print(f"residual Linear layer kw: {self._kw}")
        kw = self._kw.copy()
        kw['units'] = kw['units'] // 4
        x = self.get(f'residual_in_{i}', Linear, **kw)(inputs)
        x = self.get(f'residual_out_{i}', Linear, self._stoch * self._classes)(x)
        logit_correction = x
        return logit_correction

class EnsembleResidualLarge(EnsembleResidual):
    def _ensemble_member(self, i, prev_stoch, prev_stoch_params, prior_stoch_params, prev_action):
        input_list = [prev_stoch, prev_action]
        if self._should_take_prev_stoch_params_as_input:
            input_list.append(prev_stoch_params)
        inputs = jnp.concatenate(input_list, -1)
        print(f"residual Linear layer kw: {self._kw}")
        x = self.get(f'residual_in_{i}', Linear, **self._kw)(inputs)
        x = self.get(f'residual_hidden_a_{i}', Linear, **self._kw)(x)
        x = self.get(f'residual_hidden_b_{i}', Linear, **self._kw)(x)
        x = self.get(f'residual_out_{i}', Linear, self._stoch * self._classes)(x)
        logit_correction = x
        return logit_correction


class ReplacementDynamicsFunctionExtraSmall(nj.Module):

    def __init__(self, stoch, classes, unimix, stoch_params_are_raw_logits, stoch_params_include_unimix, img_hidden_layers,
                 should_take_prev_stoch_params_as_input=False, ensemble_size=7,
                 **kw):
        self._stoch = stoch
        self._classes = classes
        self._unimix = unimix
        self._stoch_params_are_raw_logits = stoch_params_are_raw_logits
        self._stoch_params_include_unimix = stoch_params_include_unimix
        self._img_hidden_layers = img_hidden_layers
        self._should_take_prev_stoch_params_as_input = should_take_prev_stoch_params_as_input
        self._kw = kw
        self._ensemble_size = ensemble_size
        print(f"Ensemble size: {ensemble_size}")

    def initial(self, bs):
        state = dict(
            ensemble_logits_for_each_member=jnp.zeros([bs, self._ensemble_size, self._stoch, self._classes], f32))
        return state

    def residual_stats_key(self):
        return 'ensemble_logits_for_each_member'

    def get_dist(self, logit):
        logit = logit.astype(f32)
        return tfd.Independent(jaxutils.OneHotDist(logit), 1)

    def _ensemble_member(self, i, prev_stoch, prev_stoch_params, prior_stoch_params, prev_action):
        input_list = [prev_stoch, prev_action]
        if self._should_take_prev_stoch_params_as_input:
            input_list.append(prev_stoch_params)
        inputs = jnp.concatenate(input_list, -1)
        print(f"residual Linear layer kw: {self._kw}")
        kw = self._kw.copy()
        kw['units'] = kw['units'] // 2
        x = self.get(f'residual_in_{i}', Linear, **kw)(inputs)
        x = self.get(f'residual_out_{i}', Linear, self._stoch * self._classes)(x)
        logit_correction = x
        return logit_correction

    def __call__(self, prev_stoch, prev_stoch_params, prior_stoch_params, prev_action, prior_stoch_raw_logits):
        map_fn = partial(self._ensemble_member,
                         prev_stoch=prev_stoch,
                         prev_stoch_params=prev_stoch_params,
                         prior_stoch_params=prior_stoch_params,
                         prev_action=prev_action)
        ensemble_logit_corrections = []
        for i in range(self._ensemble_size):
            ensemble_logit_correction = map_fn(i)
            ensemble_logit_corrections.append(ensemble_logit_correction)
        ensemble_logit_corrections = jnp.asarray(ensemble_logit_corrections, dtype=jaxutils.COMPUTE_DTYPE)
        avg_ensemble_logit_corrections = jnp.mean(ensemble_logit_corrections, axis=0)

        # ensemble_logits_for_each_member = ensemble_logit_corrections + prior_stoch_raw_logits
        # gathered_logits = prior_stoch_raw_logits + avg_ensemble_logit_corrections
        ensemble_logits_for_each_member = ensemble_logit_corrections
        gathered_logits = avg_ensemble_logit_corrections

        ensemble_logits_for_each_member = ensemble_logits_for_each_member.reshape(ensemble_logits_for_each_member.shape[:-1] + (self._stoch, self._classes))
        gathered_logits = gathered_logits.reshape(gathered_logits.shape[:-1] + (self._stoch, self._classes))

        gathered_probs = jax.nn.softmax(gathered_logits, -1)

        if self._stoch_params_are_raw_logits:
            stoch_params = gathered_logits
        else:
            stoch_params = gathered_probs
        stoch_params = stoch_params.reshape(stoch_params.shape[:-2] + (self._stoch * self._classes,))

        stats = {'logit': gathered_logits,
                 'stoch_params': stoch_params,
                 'stoch_raw_logits': gathered_logits.reshape(gathered_logits.shape[:-2] + (self._stoch * self._classes,)),
                 'ensemble_logits_for_each_member': ensemble_logits_for_each_member.transpose((1, 0, 2, 3)),
                 }
        return stats

    def residual_loss(self, prior, post, is_first, free=1.0):
        is_first = cast(is_first)
        total_loss = jnp.zeros(post['deter'].shape[:-1])
        for ensemble_logit in prior['ensemble_logits_for_each_member'].transpose((2, 0, 1, 3, 4)):
            print(f'ensemble_logit: {ensemble_logit.shape}')
            # jax.debug.print("prior has nan:\t\t{}", jnp.any(jnp.isnan(ensemble_logit)))
            # jax.debug.print("prior is inf:\t\t{}", jnp.any(jnp.isinf(ensemble_logit)))

            loss = self.get_dist(sg(post['logit'])).kl_divergence(self.get_dist(ensemble_logit))
            # jax.debug.print("loss has nan:\t\t{}", jnp.any(jnp.isnan(loss)))
            # jax.debug.print("loss is inf:\t\t{}", jnp.any(jnp.isinf(loss)))

            # if this is the first timestep then loss should be 0
            loss = jnp.where(is_first, 0, loss)

            # TODO removing free bits because the representation is already fixed and we just want to match it?
            # if free:
            #     loss = jnp.maximum(loss, free)
            total_loss += loss
        return total_loss


class ReplacementDynamicsFunctionExtraSmallConditionedOnPriorStochParams(ReplacementDynamicsFunctionExtraSmall):

    def _ensemble_member(self, i, prev_stoch, prev_stoch_params, prior_stoch_params, prev_action):
        input_list = [prev_stoch, prev_action, prior_stoch_params]
        if self._should_take_prev_stoch_params_as_input:
            input_list.append(prev_stoch_params)
        inputs = jnp.concatenate(input_list, -1)
        print(f"residual Linear layer kw: {self._kw}")
        kw = self._kw.copy()
        kw['units'] = kw['units'] // 2
        x = self.get(f'residual_in_{i}', Linear, **kw)(inputs)
        x = self.get(f'residual_out_{i}', Linear, self._stoch * self._classes)(x)
        logit_correction = x
        return logit_correction


class ReplacementDynamicsFunctionExtraSmallConditionedOnPriorSampledStoch(ReplacementDynamicsFunctionExtraSmall):

    def _ensemble_member(self, i, prev_stoch, prev_stoch_params, prior_stoch_params, prev_action):

        prior_z_dist = tfd.Independent(jaxutils.OneHotDist(probs=prior_stoch_params), 1)
        prior_stoch = prior_z_dist.sample(seed=nj.rng())

        assert prior_stoch.shape == prev_stoch.shape, f"prior_stoch.shape: {prior_stoch.shape}, prev_stoch.shape: {prev_stoch.shape}"
        print(f"replacement dynamics prior stoch shape: {prior_stoch.shape}")

        input_list = [prev_stoch, prev_action, prior_stoch]
        if self._should_take_prev_stoch_params_as_input:
            input_list.append(prev_stoch_params)
        inputs = jnp.concatenate(input_list, -1)
        print(f"residual Linear layer kw: {self._kw}")
        kw = self._kw.copy()
        kw['units'] = kw['units'] // 2
        x = self.get(f'residual_in_{i}', Linear, **kw)(inputs)
        x = self.get(f'residual_out_{i}', Linear, self._stoch * self._classes)(x)
        logit_correction = x
        return logit_correction



def get_residual(name: str):
    if name == "ensemble_residual":
        return EnsembleResidual
    elif name == "ensemble_residual_free_bits":
        return EnsembleResidualFreebits
    # elif name == "ensemble_residual_old":
    #     return EnsembleResiduaOld
    elif name == "ensemble_residual_small":
        return EnsembleResidualSmall
    elif name == "ensemble_residual_extra_small":
        return EnsembleResidualExtraSmall
    elif name == "ensemble_residual_extra_small_10_members":
        return EnsembleResidualExtraSmall10Members
    elif name == "ensemble_residual_extra_small_4_members":
        return EnsembleResidualExtraSmall4Members
    elif name == "ensemble_residual_extra_small_1_member":
        return EnsembleResidualExtraSmall1Member
    elif name == "ensemble_residual_extra_small_1_member_conditioned_on_prior_stoch_params":
        return EnsembleResidualExtraSmall1MemberConditionedOnPriorStochParams
    elif name == "ensemble_residual_extra_small_conditioned_on_prev_stoch_params_only":
        return EnsembleResidualExtraSmallConditionedOnPrevStochParamsOnly
    elif name == "ensemble_residual_4x_small":
        return EnsembleResidual4xSmall
    elif name == "ensemble_residual_large":
        return EnsembleResidualLarge
    elif name == "replacement_dynamics_function_extra_small":
        return ReplacementDynamicsFunctionExtraSmall
    elif name == "replacement_dynamics_function_extra_small_conditioned_on_prior_stoch_params":
        return ReplacementDynamicsFunctionExtraSmallConditionedOnPriorStochParams
    elif name == "replacement_dynamics_function_extra_small_conditioned_on_sampled_prior_stoch":
        return ReplacementDynamicsFunctionExtraSmallConditionedOnPriorSampledStoch
    elif name == 'ensemble_residual_small_td':
        return EnsembleResidualSmallTDDreamer
    elif name == 'ensemble_residual_td':
        return EnsembleResidualTDDreamer
    elif name == 'none':
        return None
    else:
        raise NotImplementedError(name)


class EnsembleResidualSmallTDDreamer(nj.Module):

    def __init__(self, stoch, classes, img_hidden_layers, **kw):
        self._stoch = stoch
        self._classes = classes
        self._img_hidden_layers = img_hidden_layers
        self._kw = kw
        self._ensemble_size = 7

    def initial(self, bs):
        state = dict(
            ensemble_logits_for_each_member=jnp.zeros([bs, self._ensemble_size, self._stoch, self._classes], f32))
        return state

    def residual_stats_key(self):
        return 'ensemble_logits_for_each_member'

    def get_dist(self, logit):
        logit = logit.astype(f32)
        return tfd.Independent(jaxutils.OneHotDist(logit), 1)

    def _ensemble_member(self, i, prev_stoch_params, prev_action):
        inputs = jnp.concatenate([
            prev_stoch_params,
            prev_action
        ], -1)

        # small version
        print(f"residual Linear layer kw: {self._kw}")
        x = self.get(f'residual_in_{i}', Linear, **self._kw)(inputs)
        x = self.get(f'residual_out_{i}', Linear, self._stoch * self._classes)(x)
        logit_correction = x
        return logit_correction

    def __call__(self, prev_stoch_params, prev_action, prior_stoch_raw_logits):
        map_fn = partial(self._ensemble_member,
                         prev_stoch_params=prev_stoch_params,
                         prev_action=prev_action)
        ensemble_logit_corrections = []
        for i in range(self._ensemble_size):
            ensemble_logit_correction = map_fn(i)
            ensemble_logit_corrections.append(ensemble_logit_correction)
        ensemble_logit_corrections = jnp.asarray(ensemble_logit_corrections, dtype=jaxutils.COMPUTE_DTYPE)
        avg_ensemble_logit_corrections = jnp.mean(ensemble_logit_corrections, axis=0)

        ensemble_logits_for_each_member = ensemble_logit_corrections + prior_stoch_raw_logits
        gathered_logits = prior_stoch_raw_logits + avg_ensemble_logit_corrections

        ensemble_logits_for_each_member = ensemble_logits_for_each_member.reshape(ensemble_logits_for_each_member.shape[:-1] + (self._stoch, self._classes))
        gathered_logits = gathered_logits.reshape(gathered_logits.shape[:-1] + (self._stoch, self._classes))

        gathered_probs = jax.nn.softmax(gathered_logits, -1)

        stoch_params = gathered_probs
        # stoch_params = stoch_params.reshape(stoch_params.shape[:-2] + (self._stoch * self._classes,))

        stats = {'logit': gathered_logits,
                 'stoch_params': stoch_params,
                 'stoch_raw_logits': gathered_logits.reshape(gathered_logits.shape[:-2] + (self._stoch * self._classes,)),
                 'ensemble_logits_for_each_member': ensemble_logits_for_each_member.transpose((1, 0, 2, 3)),
                 }
        return stats

    def residual_loss(self, prior, post, is_first, t, rho, free=1.0):
        is_first = cast(is_first)
        total_loss = jnp.zeros(post['deter'].shape[:-1])
        for ensemble_logit in prior['ensemble_logits_for_each_member'].transpose((2, 0, 1, 3, 4)):
            print(f'ensemble_logit: {ensemble_logit.shape}')
            # jax.debug.print("prior has nan:\t\t{}", jnp.any(jnp.isnan(ensemble_logit)))
            # jax.debug.print("prior is inf:\t\t{}", jnp.any(jnp.isinf(ensemble_logit)))

            loss = self.get_dist(sg(post['logit'])).kl_divergence(self.get_dist(ensemble_logit))
            # jax.debug.print("loss has nan:\t\t{}", jnp.any(jnp.isnan(loss)))
            # jax.debug.print("loss is inf:\t\t{}", jnp.any(jnp.isinf(loss)))

            # assert loss.shape == is_first.shape[:2], (loss.shape, is_first.shape)
            assert loss.shape == t[:, :-1].shape[:2], (loss.shape, t[:, :-1].shape)
            loss *= (rho ** t[:, :-1])

            # if this is the first timestep then loss should be 0
            # loss = jnp.where(is_first, 0, loss)

            # if free:
            #     loss = jnp.maximum(loss, free)
            total_loss += loss
        return total_loss


class EnsembleResidualTDDreamer(EnsembleResidualSmallTDDreamer):
    def _ensemble_member(self, i, prev_stoch_params, prev_action):
        inputs = jnp.concatenate([
            prev_stoch_params,
            prev_action
        ], -1)

        # small version
        print(f"residual Linear layer kw: {self._kw}")
        x = self.get(f'residual_in_{i}', Linear, **self._kw)(inputs)
        x = self.get(f'residual_hidden_{i}', Linear, **self._kw)(x)
        x = self.get(f'residual_out_{i}', Linear, self._stoch * self._classes)(x)
        logit_correction = x
        return logit_correction

# class RSSMStochOnlyNonSequentialPosteriorMLPPriorLargerWithPostBelief(RSSM):
#     # not default z only dreamer
#     def initial(self, bs):
#         if self._classes:
#             state = dict(
#                 deter=jnp.zeros([bs, self._deter], f32),
#                 logit=jnp.zeros([bs, self._stoch, self._classes], f32),
#                 stoch=jnp.zeros([bs, self._stoch, self._classes], f32),
#                 stoch_probs=jnp.zeros([bs, self._stoch, self._classes], f32),
#             )
#         else:
#             state = dict(
#                 deter=jnp.zeros([bs, self._deter], f32),
#                 mean=jnp.zeros([bs, self._stoch], f32),
#                 std=jnp.ones([bs, self._stoch], f32),
#                 stoch=jnp.zeros([bs, self._stoch], f32))
#         if self._initial == 'zeros':
#             return cast(state)
#         elif self._initial == 'learned':
#             print(f"Using all zeroes for initial state even though 'learned' is specified for this.")
#             return cast(state)
#         else:
#             raise NotImplementedError(self._initial)
# 
#     def _stats(self, name, x):
#         if self._classes:
#             x = self.get(name, Linear, self._stoch * self._classes)(x)
#             logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
#             orig_probs = jax.nn.softmax(logit, -1)
#             probs = orig_probs
#             if self._unimix:
#                 uniform = jnp.ones_like(probs) / probs.shape[-1]
#                 probs = (1 - self._unimix) * probs + self._unimix * uniform
#                 logit = jnp.log(probs)
#             stats = {'logit': logit, 'stoch_probs': orig_probs}
#             return stats
#         else:
#             x = self.get(name, Linear, 2 * self._stoch)(x)
#             mean, std = jnp.split(x, 2, -1)
#             std = 2 * jax.nn.sigmoid(std / 2) + 0.1
#             return {'mean': mean, 'std': std}
# 
#     def img_step(self, prev_state, prev_action):
#         prev_stoch = prev_state['stoch']
#         prev_stoch_probs = prev_state['stoch_probs']
#         prev_action = cast(prev_action)
#         if self._action_clip > 0.0:
#             prev_action *= sg(self._action_clip / jnp.maximum(
#                 self._action_clip, jnp.abs(prev_action)))
#         if self._classes:
#             shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
#             prev_stoch = prev_stoch.reshape(shape)
#             prev_stoch_probs = prev_stoch_probs.reshape(shape)
#         if len(prev_action.shape) > len(prev_stoch.shape):  # 2D actions.
#             shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
#             prev_action = prev_action.reshape(shape)
# 
#         img_inputs = jnp.concatenate([prev_stoch_probs, prev_stoch, prev_action], -1)
# 
#         print(f"img_inputs shape: {img_inputs.shape}")
#         print(f"prev_action shape: {prev_action.shape}")
# 
#         x = self.get('img_in', Linear, **self._kw)(img_inputs)
#         x = self.get('img_hidden', Linear, **self._kw)(x)
#         x = self.get('img_hidden2', Linear, **self._kw)(x)
#         x = self.get('img_out', Linear, **self._kw)(x)
#         stats = self._stats('img_stats', x)
#         dist = self.get_dist(stats)
#         stoch = dist.sample(seed=nj.rng())
#         print(f"img stoch shape: {stoch.shape}")
# 
#         prior = {'stoch': stoch,
#                  'deter': prev_state['deter'],
#                  **stats}
# 
#         return cast(prior)
# 
#     def obs_step(self, prev_state, prev_action, embed, is_first):
#         is_first = cast(is_first)
#         prev_action = cast(prev_action)
#         if self._action_clip > 0.0:
#             prev_action *= sg(self._action_clip / jnp.maximum(
#                 self._action_clip, jnp.abs(prev_action)))
#         prev_state, prev_action = jax.tree_util.tree_map(
#             lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))
#         prev_state = jax.tree_util.tree_map(
#             lambda x, y: x + self._mask(y, is_first),
#             prev_state, self.initial(len(is_first)))
#         prior = self.img_step(prev_state, prev_action)
# 
#         # x = jnp.concatenate([prev_action, embed], -1)
#         # print(f"posterior is conditioned on embedding and previous action")
# 
#         x = embed
#         x = self.get('obs_out', Linear, **self._kw)(x)
#         stats = self._stats('obs_stats', x)
#         dist = self.get_dist(stats)
#         stoch = dist.sample(seed=nj.rng())
# 
#         print(f"obs stoch shape: {stoch.shape}")
# 
#         post = {'stoch': stoch,
#                 'deter': prev_state['deter'],
#                 **stats}
# 
#         return cast(post), cast(prior)
# 
#     def dyn_loss(self, post, prior, is_first, impl='kl', free=1.0):
#         if impl == 'kl':
#             loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
#         elif impl == 'kl_masked':
#             is_first = cast(is_first)
#             # if this is the first timestep then loss should be 0
#             loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
#             loss = jnp.where(is_first, 0, loss)
#         elif impl == 'logprob':
#             loss = -self.get_dist(prior).log_prob(sg(post['stoch']))
#         else:
#             raise NotImplementedError(impl)
#         if free:
#             loss = jnp.maximum(loss, free)
#         return loss
# 
#     def rep_loss(self, post, prior, is_first, impl='kl', free=1.0):
#         if impl == 'kl':
#             loss = self.get_dist(post).kl_divergence(self.get_dist(sg(prior)))
#         elif impl == 'kl_masked':
#             is_first = cast(is_first)
#             # if this is the first timestep then loss should be 0
#             loss = self.get_dist(post).kl_divergence(self.get_dist(sg(prior)))
#             loss = jnp.where(is_first, 0, loss)
#         elif impl == 'uniform':
#             uniform = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), prior)
#             loss = self.get_dist(post).kl_divergence(self.get_dist(uniform))
#         elif impl == 'entropy':
#             loss = -self.get_dist(post).entropy()
#         elif impl == 'none':
#             loss = jnp.zeros(post['deter'].shape[:-1])
#         else:
#             raise NotImplementedError(impl)
#         if free:
#             loss = jnp.maximum(loss, free)
#         return loss


class MultiEncoder(nj.Module):

    def __init__(
            self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', mlp_layers=4,
            mlp_units=512, cnn='resize', cnn_depth=48,
            cnn_blocks=2, resize='stride',
            symlog_inputs=False, minres=4, **kw):
        excluded = ('is_first', 'is_last', 'gt_state', 'is_real')
        shapes = {k: v for k, v in shapes.items() if (
                k not in excluded and not k.startswith('log_'))}
        self.cnn_shapes = {k: v for k, v in shapes.items() if (
                len(v) == 3 and re.match(cnn_keys, k))}
        self.mlp_shapes = {k: v for k, v in shapes.items() if (
                len(v) in (1, 2) and re.match(mlp_keys, k))}
        self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
        print('Encoder CNN keys:', cnn_keys)
        print('Encoder MLP keys:', mlp_keys)
        print('Encoder CNN shapes:', self.cnn_shapes)
        print('Encoder MLP shapes:', self.mlp_shapes)
        cnn_kw = {**kw, 'minres': minres, 'name': 'cnn'}
        mlp_kw = {**kw, 'symlog_inputs': symlog_inputs, 'name': 'mlp'}
        if cnn == 'resnet':
            self._cnn = ImageEncoderResnet(cnn_depth, cnn_blocks, resize, **cnn_kw)
        else:
            raise NotImplementedError(cnn)
        if self.mlp_shapes:
            self._mlp = MLP(None, mlp_layers, mlp_units, dist='none', **mlp_kw)

    def __call__(self, data):
        some_key, some_shape = list(self.shapes.items())[0]
        batch_dims = data[some_key].shape[:-len(some_shape)]
        data = {
            k: v.reshape((-1,) + v.shape[len(batch_dims):])
            for k, v in data.items()}
        outputs = []
        if self.cnn_shapes:
            inputs = jnp.concatenate([data[k] for k in self.cnn_shapes], -1)
            output = self._cnn(inputs)
            output = output.reshape((output.shape[0], -1))
            outputs.append(output)
        if self.mlp_shapes:
            inputs = [
                data[k][..., None] if len(self.shapes[k]) == 0 else data[k]
                for k in self.mlp_shapes]
            inputs = jnp.concatenate([x.astype(f32) for x in inputs], -1)
            inputs = jaxutils.cast_to_compute(inputs)
            outputs.append(self._mlp(inputs))
        outputs = jnp.concatenate(outputs, -1)
        outputs = outputs.reshape(batch_dims + outputs.shape[1:])
        return outputs


class MultiEncoderWithGroundedSymlogHead(nj.Module):

    def __init__(
            self, grounded_size, shapes, cnn_keys=r'.*', mlp_keys=r'.*', mlp_layers=4,
            mlp_units=512, cnn='resize', cnn_depth=48,
            cnn_blocks=2, resize='stride',
            symlog_inputs=False, minres=4, **kw):

        self.encoder = MultiEncoder(shapes=shapes, cnn_keys=cnn_keys, mlp_keys=mlp_keys, mlp_layers=mlp_layers,
                         mlp_units=mlp_units, cnn=cnn, cnn_depth=cnn_depth, cnn_blocks=cnn_blocks,
                         resize=resize, symlog_inputs=symlog_inputs, minres=minres, **kw, name='enc_base')

        self._grounded_size = grounded_size
        self._mlp_units = mlp_units
        distkeys = (
            'dist', 'outscale', 'minstd', 'maxstd', 'outnorm', 'unimix', 'bins')
        self._dense = {k: v for k, v in kw.items() if k not in distkeys}

    def __call__(self, data):
        embed = self.encoder(data)
        x = self.get('encoder_hidden', Linear, self._mlp_units, **self._dense)(embed)
        symlog_grounded = self.get('encoder_symlog_grounded_out', Linear, self._grounded_size)(x)

        return symlog_grounded


class MultiEncoderIdentityFunctionWithSymlog(nj.Module):

    def __init__(
            self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', mlp_layers=4,
            mlp_units=512, cnn='resize', cnn_depth=48,
            cnn_blocks=2, resize='stride',
            symlog_inputs=False, minres=4, **kw):
        excluded = ('is_first', 'is_last', 'gt_state', 'is_real')
        shapes = {k: v for k, v in shapes.items() if (
                k not in excluded and not k.startswith('log_'))}
        self.cnn_shapes = {k: v for k, v in shapes.items() if (
                len(v) == 3 and re.match(cnn_keys, k))}
        self.mlp_shapes = {k: v for k, v in shapes.items() if (
                len(v) in (1, 2) and re.match(mlp_keys, k))}
        self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
        print('Encoder CNN keys:', cnn_keys)
        print('Encoder MLP keys:', mlp_keys)
        print('Encoder CNN shapes:', self.cnn_shapes)
        print('Encoder MLP shapes:', self.mlp_shapes)

    def __call__(self, data):
        some_key, some_shape = list(self.shapes.items())[0]
        batch_dims = data[some_key].shape[:-len(some_shape)]
        data = {
            k: v.reshape((-1,) + v.shape[len(batch_dims):])
            for k, v in data.items()}
        outputs = []
        if self.cnn_shapes:
            inputs = jnp.concatenate([data[k] for k in self.cnn_shapes], -1)
            output = inputs
            output = output.reshape((output.shape[0], -1))
            outputs.append(output)
        if self.mlp_shapes:
            inputs = [
                data[k][..., None] if len(self.shapes[k]) == 0 else data[k]
                for k in self.mlp_shapes]
            inputs = jnp.concatenate([x.astype(f32) for x in inputs], -1)
            inputs = jaxutils.cast_to_compute(inputs)
            outputs.append(inputs)
        outputs = jnp.concatenate(outputs, -1)
        outputs = outputs.reshape(batch_dims + outputs.shape[1:])
        return jaxutils.symlog(outputs)



class MultiDecoder(nj.Module):

    def __init__(
            self, shapes, inputs=['tensor'], cnn_keys=r'.*', mlp_keys=r'.*',
            mlp_layers=4, mlp_units=512, cnn='resize', cnn_depth=48, cnn_blocks=2,
            image_dist='mse', vector_dist='mse', resize='stride', bins=255,
            outscale=1.0, minres=4, cnn_sigmoid=False, dims='deter', **kw):
        excluded = ('is_first', 'is_last', 'is_terminal', 'reward', 'gt_state', 'is_real')
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v for k, v in shapes.items()
            if re.match(cnn_keys, k) and len(v) == 3}
        if 'original_image' in cnn_keys and 'image' in shapes:
            self.cnn_shapes['original_image'] = shapes['image']

        self.mlp_shapes = {
            k: v for k, v in shapes.items()
            if re.match(mlp_keys, k) and len(v) == 1}
        self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
        print('Decoder CNN keys:', cnn_keys)
        print('Decoder MLP keys:', mlp_keys)
        print('Decoder CNN shapes:', self.cnn_shapes)
        print('Decoder MLP shapes:', self.mlp_shapes)
        cnn_kw = {**kw, 'minres': minres, 'sigmoid': cnn_sigmoid}
        mlp_kw = {**kw, 'dist': vector_dist, 'outscale': outscale, 'bins': bins}
        if self.cnn_shapes:
            shapes = list(self.cnn_shapes.values())
            assert all(x[:-1] == shapes[0][:-1] for x in shapes)
            shape = shapes[0][:-1] + (sum(x[-1] for x in shapes),)
            if cnn == 'resnet':
                self._cnn = ImageDecoderResnet(
                    shape, cnn_depth, cnn_blocks, resize, **cnn_kw, name='cnn')
            else:
                raise NotImplementedError(cnn)
        if self.mlp_shapes:
            self._mlp = MLP(
                self.mlp_shapes, mlp_layers, mlp_units, **mlp_kw, name='mlp')
        # self._inputs = Input(inputs, dims=inputs[0])  # Original was: self._inputs = Input(inputs, dims='deter')
        self._inputs = Input(inputs, dims=dims)
        self._image_dist = image_dist

    def __call__(self, inputs, drop_loss_indices=None):
        features = self._inputs(inputs)
        dists = {}
        if self.cnn_shapes:
            feat = features
            if drop_loss_indices is not None:
                feat = feat[:, drop_loss_indices]
            flat = feat.reshape([-1, feat.shape[-1]])
            output = self._cnn(flat)
            output = output.reshape(feat.shape[:-1] + output.shape[1:])
            split_indices = np.cumsum([v[-1] for v in self.cnn_shapes.values()][:-1])
            means = jnp.split(output, split_indices, -1)
            dists.update({
                key: self._make_image_dist(key, mean)
                for (key, shape), mean in zip(self.cnn_shapes.items(), means)})
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, name, mean):
        mean = mean.astype(f32)
        if self._image_dist == 'normal':
            return tfd.Independent(tfd.Normal(mean, 1), 3)
        if self._image_dist == 'mse':
            return jaxutils.MSEDist(mean, 3, 'sum')
        if self._image_dist == 'mse_mean_agg':
            return jaxutils.MSEDist(mean, 3, 'mean')
        if self._image_dist == 'symlog_mse_mean_agg':
            return jaxutils.SymlogDist(mean, 3, 'mse', 'mean')
        if self._image_dist == 'symlog_mse':
            return jaxutils.SymlogDist(mean, 3, 'mse', 'sum')
        raise NotImplementedError(self._image_dist)


class ImageEncoderResnet(nj.Module):

    def __init__(self, depth, blocks, resize, minres, **kw):
        self._depth = depth
        self._blocks = blocks
        self._resize = resize
        self._minres = minres
        self._kw = kw

    def __call__(self, x):
        stages = int(np.log2(x.shape[-2]) - np.log2(self._minres))
        depth = self._depth
        x = jaxutils.cast_to_compute(x) - 0.5
        # print(x.shape)
        for i in range(stages):
            kw = {**self._kw, 'preact': False}
            if self._resize == 'stride':
                x = self.get(f's{i}res', Conv2D, depth, 4, 2, **kw)(x)
            elif self._resize == 'stride3':
                s = 2 if i else 3
                k = 5 if i else 4
                x = self.get(f's{i}res', Conv2D, depth, k, s, **kw)(x)
            elif self._resize == 'mean':
                N, H, W, D = x.shape
                x = self.get(f's{i}res', Conv2D, depth, 3, 1, **kw)(x)
                x = x.reshape((N, H // 2, W // 2, 4, D)).mean(-2)
            elif self._resize == 'max':
                x = self.get(f's{i}res', Conv2D, depth, 3, 1, **kw)(x)
                x = jax.lax.reduce_window(
                    x, -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 2, 2, 1), 'same')
            else:
                raise NotImplementedError(self._resize)
            for j in range(self._blocks):
                skip = x
                kw = {**self._kw, 'preact': True}
                x = self.get(f's{i}b{j}conv1', Conv2D, depth, 3, **kw)(x)
                x = self.get(f's{i}b{j}conv2', Conv2D, depth, 3, **kw)(x)
                x += skip
                # print(x.shape)
            depth *= 2
        if self._blocks:
            x = get_act(self._kw['act'])(x)
        x = x.reshape((x.shape[0], -1))
        # print(x.shape)
        return x


class ImageDecoderResnet(nj.Module):

    def __init__(self, shape, depth, blocks, resize, minres, sigmoid, **kw):
        self._shape = shape
        self._depth = depth
        self._blocks = blocks
        self._resize = resize
        self._minres = minres
        self._sigmoid = sigmoid
        self._kw = kw

    def __call__(self, x):
        stages = int(np.log2(self._shape[-2]) - np.log2(self._minres))
        depth = self._depth * 2 ** (stages - 1)
        x = jaxutils.cast_to_compute(x)
        x = self.get('in', Linear, (self._minres, self._minres, depth))(x)
        for i in range(stages):
            for j in range(self._blocks):
                skip = x
                kw = {**self._kw, 'preact': True}
                x = self.get(f's{i}b{j}conv1', Conv2D, depth, 3, **kw)(x)
                x = self.get(f's{i}b{j}conv2', Conv2D, depth, 3, **kw)(x)
                x += skip
                # print(x.shape)
            depth //= 2
            kw = {**self._kw, 'preact': False}
            if i == stages - 1:
                kw = {}
                depth = self._shape[-1]
            if self._resize == 'stride':
                x = self.get(f's{i}res', Conv2D, depth, 4, 2, transp=True, **kw)(x)
            elif self._resize == 'stride3':
                s = 3 if i == stages - 1 else 2
                k = 5 if i == stages - 1 else 4
                x = self.get(f's{i}res', Conv2D, depth, k, s, transp=True, **kw)(x)
            elif self._resize == 'resize':
                x = jnp.repeat(jnp.repeat(x, 2, 1), 2, 2)
                x = self.get(f's{i}res', Conv2D, depth, 3, 1, **kw)(x)
            else:
                raise NotImplementedError(self._resize)
        if max(x.shape[1:-1]) > max(self._shape[:-1]):
            padh = (x.shape[1] - self._shape[0]) / 2
            padw = (x.shape[2] - self._shape[1]) / 2
            x = x[:, int(np.ceil(padh)): -int(padh), :]
            x = x[:, :, int(np.ceil(padw)): -int(padw)]
        # print(x.shape)
        assert x.shape[-3:] == self._shape, (x.shape, self._shape)
        if self._sigmoid:
            x = jax.nn.sigmoid(x)
        else:
            x = x + 0.5
        return x


class MLP(nj.Module):

    def __init__(
            self, shape, layers, units, inputs=['tensor'], dims=None,
            symlog_inputs=False, **kw):
        assert shape is None or isinstance(shape, (int, tuple, dict)), shape
        if isinstance(shape, int):
            shape = (shape,)
        self._shape = shape
        self._layers = layers
        self._units = units
        self._inputs = Input(inputs, dims=dims)
        self._symlog_inputs = symlog_inputs
        distkeys = (
            'dist', 'outscale', 'minstd', 'maxstd', 'outnorm', 'unimix', 'bins')
        self._dense = {k: v for k, v in kw.items() if k not in distkeys}
        self._dist = {k: v for k, v in kw.items() if k in distkeys}

    def __call__(self, inputs):
        feat = self._inputs(inputs)
        if self._symlog_inputs:
            feat = jaxutils.symlog(feat)
        x = jaxutils.cast_to_compute(feat)
        x = x.reshape([-1, x.shape[-1]])
        for i in range(self._layers):
            x = self.get(f'h{i}', Linear, self._units, **self._dense)(x)
        x = x.reshape(feat.shape[:-1] + (x.shape[-1],))
        if self._shape is None:
            return x
        elif isinstance(self._shape, tuple):
            return self._out('out', self._shape, x)
        elif isinstance(self._shape, dict):
            return {k: self._out(k, v, x) for k, v in self._shape.items()}
        else:
            raise ValueError(self._shape)

    def _out(self, name, shape, x):
        return self.get(f'dist_{name}', Dist, shape, **self._dist)(x)


class Dist(nj.Module):

    def __init__(
            self, shape, dist='mse', outscale=0.1, outnorm=False, minstd=1.0,
            maxstd=1.0, unimix=0.0, bins=255):
        assert all(isinstance(dim, int) for dim in shape), shape
        self._shape = shape
        self._dist = dist
        self._minstd = minstd
        self._maxstd = maxstd
        self._unimix = unimix
        self._outscale = outscale
        self._outnorm = outnorm
        self._bins = bins

    def __call__(self, inputs):
        dist = self.inner(inputs)
        assert tuple(dist.batch_shape) == tuple(inputs.shape[:-1]), (
            dist.batch_shape, dist.event_shape, inputs.shape)
        return dist

    def inner(self, inputs):
        kw = {}
        kw['outscale'] = self._outscale
        kw['outnorm'] = self._outnorm
        shape = self._shape
        if self._dist.endswith('_disc'):
            shape = (*self._shape, self._bins)
        out = self.get('out', Linear, int(np.prod(shape)), **kw)(inputs)
        out = out.reshape(inputs.shape[:-1] + shape).astype(f32)
        if self._dist in ('normal', 'trunc_normal'):
            std = self.get('std', Linear, int(np.prod(self._shape)), **kw)(inputs)
            std = std.reshape(inputs.shape[:-1] + self._shape).astype(f32)
        if self._dist == 'symlog_mse':
            return jaxutils.SymlogDist(out, len(self._shape), 'mse', 'sum')
        if self._dist == 'symlog_mse_mean_agg':
            return jaxutils.SymlogDist(out, len(self._shape), 'mse', 'mean')
        if 'scaled_symlog_mse_mean_agg' in self._dist:
            scale = float(self._dist.replace('scaled_symlog_mse_mean_agg_', ''))
            return jaxutils.ScaledSymlogDist(out, len(self._shape), dist='mse', agg='mean', scale=scale)
        if self._dist == 'symlog_disc':
            return jaxutils.DiscDist(
                out, len(self._shape), -20, 20, jaxutils.symlog, jaxutils.symexp)
        if self._dist == 'mse':
            return jaxutils.MSEDist(out, len(self._shape), 'sum')
        if self._dist == 'mse_mean_agg':
            return jaxutils.MSEDist(out, len(self._shape), 'mean')
        if self._dist == 'normal':
            lo, hi = self._minstd, self._maxstd
            std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
            dist = tfd.Normal(jnp.tanh(out), std)
            dist = tfd.Independent(dist, len(self._shape))
            dist.minent = np.prod(self._shape) * tfd.Normal(0.0, lo).entropy()
            dist.maxent = np.prod(self._shape) * tfd.Normal(0.0, hi).entropy()
            return dist
        if self._dist == 'binary':
            dist = tfd.Bernoulli(out)
            return tfd.Independent(dist, len(self._shape))
        if self._dist == 'onehot':
            if self._unimix:
                probs = jax.nn.softmax(out, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - self._unimix) * probs + self._unimix * uniform
                out = jnp.log(probs)
            dist = jaxutils.OneHotDist(out)
            if len(self._shape) > 1:
                dist = tfd.Independent(dist, len(self._shape) - 1)
            dist.minent = 0.0
            dist.maxent = np.prod(self._shape[:-1]) * jnp.log(self._shape[-1])
            return dist
        raise NotImplementedError(self._dist)


class Conv2D(nj.Module):

    def __init__(
            self, depth, kernel, stride=1, transp=False, act='none', norm='none',
            pad='same', bias=True, preact=False, winit='uniform', fan='avg'):
        self._depth = depth
        self._kernel = kernel
        self._stride = stride
        self._transp = transp
        self._act = get_act(act)
        self._norm = Norm(norm, name='norm')
        self._pad = pad.upper()
        self._bias = bias and (preact or norm == 'none')
        self._preact = preact
        self._winit = winit
        self._fan = fan

    def __call__(self, hidden):
        if self._preact:
            hidden = self._norm(hidden)
            hidden = self._act(hidden)
            hidden = self._layer(hidden)
        else:
            hidden = self._layer(hidden)
            hidden = self._norm(hidden)
            hidden = self._act(hidden)
        return hidden

    def _layer(self, x):
        if self._transp:
            shape = (self._kernel, self._kernel, self._depth, x.shape[-1])
            kernel = self.get('kernel', Initializer(
                self._winit, fan=self._fan), shape)
            kernel = jaxutils.cast_to_compute(kernel)
            x = jax.lax.conv_transpose(
                x, kernel, (self._stride, self._stride), self._pad,
                dimension_numbers=('NHWC', 'HWOI', 'NHWC'))
        else:
            shape = (self._kernel, self._kernel, x.shape[-1], self._depth)
            kernel = self.get('kernel', Initializer(
                self._winit, fan=self._fan), shape)
            kernel = jaxutils.cast_to_compute(kernel)
            x = jax.lax.conv_general_dilated(
                x, kernel, (self._stride, self._stride), self._pad,
                dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
        if self._bias:
            bias = self.get('bias', jnp.zeros, self._depth, np.float32)
            bias = jaxutils.cast_to_compute(bias)
            x += bias
        return x


class Linear(nj.Module):

    def __init__(
            self, units, act='none', norm='none', bias=True, outscale=1.0,
            outnorm=False, winit='uniform', fan='avg'):
        self._units = tuple(units) if hasattr(units, '__len__') else (units,)
        self._act = get_act(act)
        self._norm = norm
        self._bias = bias and norm == 'none'
        self._outscale = outscale
        self._outnorm = outnorm
        self._winit = winit
        self._fan = fan

    def __call__(self, x):
        shape = (x.shape[-1], np.prod(self._units))
        kernel = self.get('kernel', Initializer(
            self._winit, self._outscale, fan=self._fan), shape)
        kernel = jaxutils.cast_to_compute(kernel)
        x = x @ kernel
        if self._bias:
            bias = self.get('bias', jnp.zeros, np.prod(self._units), np.float32)
            bias = jaxutils.cast_to_compute(bias)
            x += bias
        if len(self._units) > 1:
            x = x.reshape(x.shape[:-1] + self._units)
        x = self.get('norm', Norm, self._norm)(x)
        x = self._act(x)
        return x


class Norm(nj.Module):

    def __init__(self, impl):
        self._impl = impl

    def __call__(self, x):
        dtype = x.dtype
        if self._impl == 'none':
            return x
        elif self._impl == 'layer':
            x = x.astype(f32)
            x = jax.nn.standardize(x, axis=-1, epsilon=1e-3)
            x *= self.get('scale', jnp.ones, x.shape[-1], f32)
            x += self.get('bias', jnp.zeros, x.shape[-1], f32)
            return x.astype(dtype)
        else:
            raise NotImplementedError(self._impl)


class Input:

    def __init__(self, keys=['tensor'], dims=None):
        assert isinstance(keys, (list, tuple)), keys
        self._keys = tuple(keys)
        self._dims = dims or self._keys[0]

    def __call__(self, inputs):
        if not isinstance(inputs, dict):
            inputs = {'tensor': inputs}
        inputs = inputs.copy()
        for key in self._keys:
            if key.startswith('softmax_'):
                inputs[key] = jax.nn.softmax(inputs[key[len('softmax_'):]])
        if not all(k in inputs for k in self._keys):
            needs = f'{{{", ".join(self._keys)}}}'
            found = f'{{{", ".join(inputs.keys())}}}'
            raise KeyError(f'Cannot find keys {needs} among inputs {found}.')
        values = [inputs[k] for k in self._keys]
        dims = len(inputs[self._dims].shape)
        for i, value in enumerate(values):
            if len(value.shape) > dims:
                values[i] = value.reshape(
                    value.shape[:dims - 1] + (np.prod(value.shape[dims - 1:]),))
        values = [x.astype(inputs[self._dims].dtype) for x in values]
        return jnp.concatenate(values, -1)


class Initializer:

    def __init__(self, dist='uniform', scale=1.0, fan='avg'):
        self.scale = scale
        self.dist = dist
        self.fan = fan

    def __call__(self, shape):
        if self.scale == 0.0:
            value = jnp.zeros(shape, f32)
        elif self.dist == 'uniform':
            fanin, fanout = self._fans(shape)
            denoms = {'avg': (fanin + fanout) / 2, 'in': fanin, 'out': fanout}
            scale = self.scale / denoms[self.fan]
            limit = np.sqrt(3 * scale)
            value = jax.random.uniform(
                nj.rng(), shape, f32, -limit, limit)
        elif self.dist == 'normal':
            fanin, fanout = self._fans(shape)
            denoms = {'avg': np.mean((fanin, fanout)), 'in': fanin, 'out': fanout}
            scale = self.scale / denoms[self.fan]
            std = np.sqrt(scale) / 0.87962566103423978
            value = std * jax.random.truncated_normal(
                nj.rng(), -2, 2, shape, f32)
        elif self.dist == 'ortho':
            nrows, ncols = shape[-1], np.prod(shape) // shape[-1]
            matshape = (nrows, ncols) if nrows > ncols else (ncols, nrows)
            mat = jax.random.normal(nj.rng(), matshape, f32)
            qmat, rmat = jnp.linalg.qr(mat)
            qmat *= jnp.sign(jnp.diag(rmat))
            qmat = qmat.T if nrows < ncols else qmat
            qmat = qmat.reshape(nrows, *shape[:-1])
            value = self.scale * jnp.moveaxis(qmat, 0, -1)
        else:
            raise NotImplementedError(self.dist)
        return value

    def _fans(self, shape):
        if len(shape) == 0:
            return 1, 1
        elif len(shape) == 1:
            return shape[0], shape[0]
        elif len(shape) == 2:
            return shape
        else:
            space = int(np.prod(shape[:-2]))
            return shape[-2] * space, shape[-1] * space


def get_act(name):
    if callable(name):
        return name
    elif name == 'none':
        return lambda x: x
    elif name == 'mish':
        return lambda x: x * jnp.tanh(jax.nn.softplus(x))
    elif hasattr(jax.nn, name):
        return getattr(jax.nn, name)
    else:
        raise NotImplementedError(name)






# class DeterministicZOnlyDreamerModel(RSSM):
# 
#     def __init__(
#             self, deter=1024, stoch=32, classes=32, unroll=False, initial='learned',
#             unimix=0.01, action_clip=1.0, img_hidden_layers=2,
#             use_gru_with_prior_belief=False,
#             stoch_params_include_unimix=False,
#             stoch_params_are_raw_logits=False,
#             use_half_of_stoch_as_free_variables=False,
#             use_posterior_stoch_params_for_first_state=False,
#             use_posterior_stoch_params_for_all_states=False,
#             dynamics_takes_prev_stoch_as_input=True,
#             residual=None,
#             **kw):
#         self._deter = deter
#         self._stoch = stoch
#         self._classes = classes
#         self._unroll = unroll
#         self._initial = initial
#         self._unimix = unimix
#         self._action_clip = action_clip
#         self._img_hidden_layers = img_hidden_layers
#         self._use_gru_with_prior_belief = use_gru_with_prior_belief
#         self._stoch_params_include_unimix = stoch_params_include_unimix
#         self._stoch_params_are_raw_logits = stoch_params_are_raw_logits
#         assert not (stoch_params_include_unimix and stoch_params_are_raw_logits), "can only choose one"
#         self._use_half_of_stoch_as_free_variables = use_half_of_stoch_as_free_variables
#         self._stoch_sample_size = self._stoch // 2 if self._use_half_of_stoch_as_free_variables else self._stoch
#         self._use_posterior_stoch_params_for_first_state = use_posterior_stoch_params_for_first_state
#         self._use_posterior_stoch_params_for_all_states = use_posterior_stoch_params_for_all_states
#         self._dynamics_takes_prev_stoch_as_input = dynamics_takes_prev_stoch_as_input
#         if isinstance(residual, str):
#             residual = get_residual(residual)
#             if residual:
#                 residual = residual(stoch=stoch, classes=classes, unimix=self._unimix,
#                                     stoch_params_are_raw_logits=stoch_params_are_raw_logits,
#                                     stoch_params_include_unimix=stoch_params_include_unimix,
#                                     img_hidden_layers=img_hidden_layers,
#                                     name='residual', **kw)
#         self.residual = residual
#         self._kw = kw
# 
#     def initial(self, bs):
#         if self._classes:
#             state = dict(
#                 deter=jnp.zeros([bs, self._deter], f32),
#                 logit=jnp.zeros([bs, self._stoch, self._classes], f32),
#                 stoch_params=jnp.zeros([bs, self._stoch, self._classes], f32),
#             )
#         else:
#             raise NotImplementedError
# 
#         if self.residual is not None:
#             state.update(self.residual.initial(bs))
# 
#         if self._initial == 'zeros':
#             return cast(state)
#         elif self._initial == 'learned':
#             print(f"Using all zeroes for initial state even though 'learned' is specified for this.")
#             return cast(state)
#         else:
#             raise NotImplementedError(self._initial)
# 
#     def get_dist(self, state, argmax=False, use_all_variables=False):
#         if self._classes:
#             logit = state['logit'].astype(f32)
#             return tfd.Independent(jaxutils.OneHotDist(logit), 1)
#         else:
#             raise NotImplementedError
# 
#     def _stats(self, name, x):
#         if self._classes:
#             x = self.get(name, Linear, self._stoch * self._classes)(x)
#             logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
#             probs = jax.nn.softmax(logit, -1)
#             stats = {'logit': logit,
#                      'stoch_params': probs,
#                      }
#             return stats
#         else:
#             raise NotImplementedError
# 
#     def observe(self, embed, action, is_first, state=None, with_residual=True, with_res_stop_gradients=True):
#         swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
#         if state is None:
#             state = self.initial(action.shape[0])
#         step = lambda prev, inputs: self.obs_step(prev[0], *inputs, with_residual=with_residual, with_res_stop_gradients=with_res_stop_gradients)
#         inputs = swap(action), swap(embed), swap(is_first)
#         start = state, state
#         post, prior = jaxutils.scan(step, inputs, start, self._unroll)
#         post = {k: swap(v) for k, v in post.items()}
#         prior = {k: swap(v) for k, v in prior.items()}
#         return post, prior
# 
#     def imagine(self, action, state=None):
#         swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
#         state = self.initial(action.shape[0]) if state is None else state
#         assert isinstance(state, dict), state
#         action = swap(action)
#         step = partial(self.img_step, with_residual=True, with_res_stop_gradients=False)
#         prior = jaxutils.scan(step, action, state, self._unroll)
#         prior = {k: swap(v) for k, v in prior.items()}
#         return prior
# 
#     def img_step(self, prev_state, prev_action, with_residual=True, with_res_stop_gradients=False):
#         prev_stoch_params = prev_state['stoch_params']
#         prev_action = cast(prev_action)
#         if self._action_clip > 0.0:
#             prev_action *= sg(self._action_clip / jnp.maximum(
#                 self._action_clip, jnp.abs(prev_action)))
# 
#         if self._classes:
#             shape = prev_stoch_params.shape[:-2] + (self._stoch * self._classes,)
#             prev_stoch_params = prev_stoch_params.reshape(shape)
#         if len(prev_action.shape) > len(prev_stoch_params.shape):  # 2D actions.
#             shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
#             prev_action = prev_action.reshape(shape)
# 
#         x = jnp.concatenate([prev_stoch_params, prev_action], -1)
# 
#         x = self.get('img_in', Linear, **self._kw)(x)
#         for i in range(self._img_hidden_layers):
#             x = self.get(f'img_hidden{i+1}', Linear, **self._kw)(x)
#         x = self.get('img_out', Linear, **self._kw)(x)
#         stats = self._stats('img_stats', x)
# 
#         if with_residual and self.residual:
#             stats = self.get_prior_residual_correction(
#                 with_res_stop_gradients=with_res_stop_gradients,
#                 prev_stoch=prev_stoch,
#                 prev_action=prev_action,
#                 prev_stoch_params=prev_stoch_params,
#                 prior_stoch_params=stats['stoch_params'],
#                 prior_stoch_raw_logits=stats['stoch_raw_logits']
#             )
#         elif self.residual:
#             stats.update(self.residual.initial(bs=prev_action.shape[0]))
# 
#         prior = {
#                  'deter': prev_state['deter'],
#                  **stats}
# 
#         return cast(prior)
# 
#     def _gru(self, x, deter):
#         raise NotImplementedError
# 
#     def obs_step(self, prev_state, prev_action, embed, is_first, with_residual=True, with_res_stop_gradients=True):
#         is_first = cast(is_first)
#         prev_action = cast(prev_action)
#         if self._action_clip > 0.0:
#             prev_action *= sg(self._action_clip / jnp.maximum(
#                 self._action_clip, jnp.abs(prev_action)))
# 
#         prev_state, prev_action = jax.tree_util.tree_map(
#             lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))
#         prev_state = jax.tree_util.tree_map(
#             lambda x, y: x + self._mask(y, is_first),
#             prev_state, self.initial(len(is_first)))
#         prior = self.img_step(prev_state, prev_action,
#                               with_residual=with_residual,
#                               with_res_stop_gradients=with_res_stop_gradients)
#         # x = jnp.concatenate([prev_action, embed], -1)
#         # print(f"posterior is conditioned on embedding and previous action")
# 
#         x = embed
#         x = self.get('obs_out', Linear, **self._kw)(x)
#         stats = self._stats('obs_stats', x)
# 
#         # if self._use_posterior_stoch_params_for_first_state:
#         #     stoch_params = jnp.where(jnp.tile(is_first[:, None], (1, stats['stoch_params'].shape[-1])),
#         #                                    stats['stoch_params'], prior['stoch_params'])
#         #     stoch_raw_logits = jnp.where(jnp.tile(is_first[:, None], (1, stats['stoch_raw_logits'].shape[-1])),
#         #                                    stats['stoch_raw_logits'], prior['stoch_raw_logits'])
#         # elif self._use_posterior_stoch_params_for_all_states:
#         #     stoch_params = stats['stoch_params']
#         #     stoch_raw_logits = stats['stoch_raw_logits']
#         # else:
#         #     stoch_params = prior['stoch_params']
#         #     stoch_raw_logits = prior['stoch_raw_logits']
#         #
#         # del stats['stoch_params']
#         # del stats['stoch_raw_logits']
#         #
#         # post = {'stoch': stoch,
#         #         'deter': prev_state['deter'],
#         #         # 'prior_stoch_params': stoch_params,
#         #         'stoch_params': stoch_params,
#         #         'stoch_raw_logits': stoch_raw_logits,
#         #         **stats}
# 
# 
#         post = {
#                 'deter': prev_state['deter'],
#                 **stats}
# 
#         if self.residual and self.residual.residual_stats_key():
#             if with_residual:
#                 post[self.residual.residual_stats_key()] = prior[self.residual.residual_stats_key()]
#             else:
#                 post.update(self.residual.initial(bs=is_first.shape[0]))
# 
#         return cast(post), cast(prior)
# 
# 
#     def dyn_loss(self, post, prior, is_first, impl='kl', free=1.0):
#         if impl == 'kl':
#             loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
#         elif impl == 'kl_masked':
#             is_first = cast(is_first)
#             # if this is the first timestep then loss should be 0
#             loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
#             loss = jnp.where(is_first, 0, loss)
#         elif impl == 'logprob':
#             loss = -self.get_dist(prior).log_prob(sg(post['stoch']))
#         else:
#             raise NotImplementedError(impl)
#         if free:
#             loss = jnp.maximum(loss, free)
#         return loss
# 
#     def rep_loss(self, post, prior, is_first, impl='kl', free=1.0):
#         if impl == 'kl':
#             loss = self.get_dist(post, use_all_variables=True).kl_divergence(self.get_dist(sg(prior), use_all_variables=True))
#         elif impl == 'kl_masked':
#             is_first = cast(is_first)
#             # if this is the first timestep then loss should be 0
#             loss = self.get_dist(post, use_all_variables=True).kl_divergence(self.get_dist(sg(prior), use_all_variables=True))
#             loss = jnp.where(is_first, 0, loss)
#         elif impl == 'uniform':
#             uniform = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), prior)
#             loss = self.get_dist(post, use_all_variables=True).kl_divergence(self.get_dist(uniform, use_all_variables=True))
#         elif impl == 'entropy':
#             loss = -self.get_dist(post, use_all_variables=True).entropy()
#         elif impl == 'none':
#             loss = jnp.zeros(post['deter'].shape[:-1])
#         else:
#             raise NotImplementedError(impl)
#         if free:
#             loss = jnp.maximum(loss, free)
#         return loss
# 
#     def get_prior_residual_correction(self, with_res_stop_gradients, prev_stoch, prev_action, prev_stoch_params, prior_stoch_params, prior_stoch_raw_logits):
#         if with_res_stop_gradients:
#             prev_stoch = sg(prev_stoch)
#             prev_action = sg(prev_action)
#             prev_stoch_params = sg(prev_stoch_params)
#             prior_stoch_params = sg(prior_stoch_params)
#             prior_stoch_raw_logits = sg(prior_stoch_raw_logits)
# 
#         corrected_prior_stats = self.residual(
#             prev_stoch=prev_stoch,
#             prev_stoch_params=prev_stoch_params,
#             prior_stoch_params=prior_stoch_params,
#             prev_action=prev_action,
#             prior_stoch_raw_logits=prior_stoch_raw_logits
#         )
#         return corrected_prior_stats




# class RSSMRegularizedDeter(RSSM):
# 
#     def __init__(
#             self, deter=1024, stoch=32, classes=32, unroll=False, initial='learned',
#             unimix=0.01, action_clip=1.0, img_hidden_layers=2,
#             use_gru_with_prior_belief=False,
#             stoch_params_include_unimix=False,
#             stoch_params_are_raw_logits=False,
#             use_half_of_stoch_as_free_variables=False,
#             use_posterior_stoch_params_for_first_state=False,
#             use_posterior_stoch_params_for_all_states=False,
#             dynamics_takes_prev_stoch_as_input=False,
#             residual=None,
#             **kw):
#         self._deter = deter
#         self._stoch = stoch
#         self._classes = classes
#         self._unroll = unroll
#         self._initial = initial
#         self._unimix = unimix
#         self._action_clip = action_clip
#         self._kw = kw
# 
#         self._img_hidden_layers = img_hidden_layers
#         if self._deter != (self._stoch * self._classes):
#             raise ValueError(f"self._deter must be equal to (self._stoch * self._classes) for this RSSM architecture")
# 
#     def initial(self, bs):
#         if self._classes:
#             state = dict(
#                 deter=jnp.zeros([bs, self._deter], f32),
#                 logit=jnp.zeros([bs, self._stoch, self._classes], f32),
#                 stoch=jnp.zeros([bs, self._stoch, self._classes], f32))
#         else:
#             state = dict(
#                 deter=jnp.zeros([bs, self._deter], f32),
#                 mean=jnp.zeros([bs, self._stoch], f32),
#                 std=jnp.ones([bs, self._stoch], f32),
#                 stoch=jnp.zeros([bs, self._stoch], f32))
#         if self._initial == 'zeros':
#             return cast(state)
#         elif self._initial == 'learned':
#             deter = self.get('initial', jnp.zeros, state['deter'][0].shape, f32)
#             state['deter'] = jnp.repeat(jnp.tanh(deter)[None], bs, 0)
#             state['stoch'] = self.get_stoch(cast(state['deter']))
#             return cast(state)
#         else:
#             raise NotImplementedError(self._initial)
# 
#     def observe(self, embed, action, is_first, state=None, with_residual=True, with_res_stop_gradients=True):
#         swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
#         if state is None:
#             state = self.initial(action.shape[0])
#         step = lambda prev, inputs: self.obs_step(prev[0], *inputs)
#         inputs = swap(action), swap(embed), swap(is_first)
#         start = state, state
#         post, prior = jaxutils.scan(step, inputs, start, self._unroll)
#         post = {k: swap(v) for k, v in post.items()}
#         prior = {k: swap(v) for k, v in prior.items()}
#         return post, prior
# 
#     def imagine(self, action, state=None, with_residual=True, with_res_stop_gradients=True):
#         swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
#         state = self.initial(action.shape[0]) if state is None else state
#         assert isinstance(state, dict), state
#         action = swap(action)
#         prior = jaxutils.scan(self.img_step, action, state, self._unroll)
#         prior = {k: swap(v) for k, v in prior.items()}
#         return prior
# 
#     def get_dist(self, state, argmax=False):
#         if self._classes:
#             if 'logit' in state:
#                 logit = state['logit'].astype(f32)
#                 return tfd.Independent(jaxutils.OneHotDist(logits=logit), 1)
#             else:
#                 probs = state['probs'].astype(f32)
#                 return tfd.Independent(jaxutils.OneHotDist(probs=probs), 1)
#         else:
#             mean = state['mean'].astype(f32)
#             std = state['std'].astype(f32)
#             return tfd.MultivariateNormalDiag(mean, std)
# 
#     def obs_step(self, prev_state, prev_action, embed, is_first, with_residual=True, with_res_stop_gradients=True):
#         is_first = cast(is_first)
#         prev_action = cast(prev_action)
#         if self._action_clip > 0.0:
#             prev_action *= sg(self._action_clip / jnp.maximum(
#                 self._action_clip, jnp.abs(prev_action)))
#         prev_state, prev_action = jax.tree_util.tree_map(
#             lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))
#         prev_state = jax.tree_util.tree_map(
#             lambda x, y: x + self._mask(y, is_first),
#             prev_state, self.initial(len(is_first)))
#         prior = self.img_step(prev_state, prev_action)
#         x = jnp.concatenate([prior['deter'], embed], -1)
#         x = self.get('obs_out', Linear, **self._kw)(x)
#         stats = self._stats('obs_stats', x)
#         del stats['probs']
#         dist = self.get_dist(stats)
#         stoch = dist.sample(seed=nj.rng())
#         post = {'stoch': stoch, 'deter': prior['deter'], **stats}
#         return cast(post), cast(prior)
# 
#     def img_step(self, prev_state, prev_action, with_residual=False, with_res_stop_gradients=False):
#         prev_stoch = prev_state['stoch']
#         prev_action = cast(prev_action)
#         if self._action_clip > 0.0:
#             prev_action *= sg(self._action_clip / jnp.maximum(
#                 self._action_clip, jnp.abs(prev_action)))
#         if not self._classes:
#             raise NotImplementedError
#         flattened_stoch_shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
#         prev_stoch = prev_stoch.reshape(flattened_stoch_shape)
#         if len(prev_action.shape) > len(prev_stoch.shape):  # 2D actions.
#             shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
#             prev_action = prev_action.reshape(shape)
#         x = jnp.concatenate([prev_stoch, prev_state['deter'], prev_action], -1)
#         print(f"prev_state['deter'].shape: {prev_state['deter'].shape}")
#         print(f"prev_stoch.shape: {prev_stoch.shape}")
#         x = self.get('img_in', Linear, **self._kw)(x)
#         for i in range(self._img_hidden_layers):
#             x = self.get(f'img_hidden{i + 1}', Linear, **self._kw)(x)
#         x = self.get('img_out', Linear, **self._kw)(x)
#         stats = self._stats('img_stats', x)
#         deter = stats['probs'].reshape(flattened_stoch_shape)
#         del stats['probs']
#         dist = self.get_dist(stats)
#         stoch = dist.sample(seed=nj.rng())
#         prior = {'stoch': stoch, 'deter': deter, **stats}
#         return cast(prior)
# 
#     def get_stoch(self, deter):
#         probs = deter.reshape((deter.shape[:-1] + (self._stoch, self._classes)))
#         dist = self.get_dist({'probs': probs})
#         return cast(dist.mode())
# 
#     def _gru(self, x, deter):
#         raise NotImplementedError
# 
#     def _stats(self, name, x):
#         if self._classes:
#             x = self.get(name, Linear, self._stoch * self._classes)(x)
#             logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
#             probs = jax.nn.softmax(logit, -1)
#             if self._unimix:
#                 uniform = jnp.ones_like(probs) / probs.shape[-1]
#                 probs = (1 - self._unimix) * probs + self._unimix * uniform
#                 logit = jnp.log(probs)
#             stats = {'logit': logit, "probs": probs}
#             return stats
#         else:
#             raise NotImplementedError
#             # x = self.get(name, Linear, 2 * self._stoch)(x)
#             # mean, std = jnp.split(x, 2, -1)
#             # std = 2 * jax.nn.sigmoid(std / 2) + 0.1
#             # return {'mean': mean, 'std': std}
# 
#     def _mask(self, value, mask):
#         return jnp.einsum('b...,b->b...', value, mask.astype(value.dtype))
# 
#     def dyn_loss(self, post, prior, is_first, impl='kl', free=1.0):
#         if impl == 'kl':
#             loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
#         elif impl == 'logprob':
#             loss = -self.get_dist(prior).log_prob(sg(post['stoch']))
#         else:
#             raise NotImplementedError(impl)
#         if free:
#             loss = jnp.maximum(loss, free)
#         return loss
# 
#     def rep_loss(self, post, prior, is_first, impl='kl', free=1.0):
#         if impl == 'kl':
#             loss = self.get_dist(post).kl_divergence(self.get_dist(sg(prior)))
#         elif impl == 'uniform':
#             uniform = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), prior)
#             loss = self.get_dist(post).kl_divergence(self.get_dist(uniform))
#         elif impl == 'entropy':
#             loss = -self.get_dist(post).entropy()
#         elif impl == 'none':
#             loss = jnp.zeros(post['deter'].shape[:-1])
#         else:
#             raise NotImplementedError(impl)
#         if free:
#             loss = jnp.maximum(loss, free)
#         return loss
# 
# class RSSMRegularizedDeterNoUnimixOnDeter(RSSMRegularizedDeter):
# 
#     def _stats(self, name, x):
#         if self._classes:
#             x = self.get(name, Linear, self._stoch * self._classes)(x)
#             logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
#             probs = jax.nn.softmax(logit, -1)
#             orig_probs = probs
#             if self._unimix:
#                 uniform = jnp.ones_like(probs) / probs.shape[-1]
#                 probs = (1 - self._unimix) * probs + self._unimix * uniform
#                 logit = jnp.log(probs)
#             stats = {'logit': logit, "probs": orig_probs}
#             return stats
#         else:
#             raise NotImplementedError
# 
# 






class TDDeterministicZOnlyDreamerModel(RSSM):

    def __init__(
            self, deter=1024, stoch=32, classes=32, unroll=False, initial='learned',
            unimix=0.01, action_clip=1.0, img_hidden_layers=2,
            use_gru_with_prior_belief=False,
            stoch_params_include_unimix=False,
            stoch_params_are_raw_logits=False,
            use_half_of_stoch_as_free_variables=False,
            use_posterior_stoch_params_for_first_state=False,
            use_posterior_stoch_params_for_all_states=False,
            dynamics_takes_prev_stoch_as_input=True,
            residual=None, # after residual, nothing here is used:
            obs_hidden_layers=None,
            decode_stoch_params_as_deter=False,
            posterior_takes_prior_deter_as_input=False,
            posterior_takes_prior_stoch_as_input=False,
            dynamics_takes_prev_stoch_params_as_input=True,
            train_student_posterior=False,
            always_use_student_posterior=False,
            residual_should_take_prev_stoch_params_as_input=False,
            use_relaxed_categorical_dist=False,
            relaxed_categorical_temperature=1.0,
            **kw):
        self._deter = deter
        self._stoch = stoch
        self._classes = classes
        self._unroll = unroll
        self._initial = initial
        self._unimix = unimix
        self._action_clip = action_clip
        self._img_hidden_layers = img_hidden_layers
        self._use_gru_with_prior_belief = use_gru_with_prior_belief
        self._stoch_params_include_unimix = stoch_params_include_unimix
        self._stoch_params_are_raw_logits = stoch_params_are_raw_logits
        assert not (stoch_params_include_unimix and stoch_params_are_raw_logits), "can only choose one"
        self._use_half_of_stoch_as_free_variables = use_half_of_stoch_as_free_variables
        self._stoch_sample_size = self._stoch // 2 if self._use_half_of_stoch_as_free_variables else self._stoch
        self._use_posterior_stoch_params_for_first_state = use_posterior_stoch_params_for_first_state
        self._use_posterior_stoch_params_for_all_states = use_posterior_stoch_params_for_all_states
        self._dynamics_takes_prev_stoch_as_input = dynamics_takes_prev_stoch_as_input
        if isinstance(residual, str):
            residual: Type[EnsembleResidualSmallTDDreamer] = get_residual(residual)
            if residual:
                residual = residual(stoch=stoch, classes=classes,
                                    img_hidden_layers=img_hidden_layers,
                                    name='residual', **kw)
        self.residual = residual
        self._kw = kw

    def initial(self, bs):
        if self._classes:
            state = dict(
                deter=jnp.zeros([bs, self._deter], f32),
                logit=jnp.zeros([bs, self._stoch, self._classes], f32),
                stoch_params=jnp.zeros([bs, self._stoch, self._classes], f32),
                stoch_raw_logits=jnp.zeros([bs, self._stoch * self._classes], f32),
            )
        else:
            raise NotImplementedError

        if self.residual is not None:
            state.update(self.residual.initial(bs))

        if self._initial == 'zeros':
            return cast(state)
        elif self._initial == 'learned':
            print(f"Using all zeroes for initial state even though 'learned' is specified for this.")
            return cast(state)
        else:
            raise NotImplementedError(self._initial)

    def get_dist(self, state, argmax=False, use_all_variables=False):
        if self._classes:
            logit = state['logit'].astype(f32)
            return tfd.Independent(jaxutils.OneHotDist(logit), 1)
        else:
            raise NotImplementedError

    def _stats(self, name, x):
        if self._classes:
            x = self.get(name, Linear, self._stoch * self._classes)(x)
            orig_logit = x
            logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
            probs = jax.nn.softmax(logit, -1)
            stats = {'logit': logit,
                     'stoch_raw_logits': orig_logit, # in case we ever add unimix, etc
                     'stoch_params': probs,

                     }
            return stats
        else:
            raise NotImplementedError

    def imagine(self, action, state=None):
        swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
        state = self.initial(action.shape[0]) if state is None else state
        assert isinstance(state, dict), state
        action = swap(action)
        step = partial(self.img_step, with_residual=True, with_res_stop_gradients=False)
        prior = jaxutils.scan(step, action, state, self._unroll)
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def img_step(self, prev_state, prev_action, with_residual=True, with_res_stop_gradients=False, with_student_posterior=False):
        prev_stoch_params = prev_state['stoch_params']
        prev_action = cast(prev_action)
        if self._action_clip > 0.0:
            prev_action *= sg(self._action_clip / jnp.maximum(
                self._action_clip, jnp.abs(prev_action)))

        if self._classes:
            shape = prev_stoch_params.shape[:-2] + (self._stoch * self._classes,)
            prev_stoch_params = prev_stoch_params.reshape(shape)
        if len(prev_action.shape) > len(prev_stoch_params.shape):  # 2D actions.
            shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
            prev_action = prev_action.reshape(shape)

        x = jnp.concatenate([prev_stoch_params, prev_action], -1)

        x = self.get('img_in', Linear, **self._kw)(x)
        for i in range(self._img_hidden_layers):
            x = self.get(f'img_hidden{i+1}', Linear, **self._kw)(x)
        x = self.get('img_out', Linear, **self._kw)(x)
        stats = self._stats('img_stats', x)

        if with_residual and self.residual:
            stats = self.get_prior_residual_correction(
                with_res_stop_gradients=with_res_stop_gradients,
                prev_action=prev_action,
                prev_stoch_params=prev_stoch_params,
                prior_stoch_raw_logits=stats['stoch_raw_logits']
            )
        elif self.residual:
            stats.update(self.residual.initial(bs=prev_action.shape[0]))

        prior = {
                 'deter': prev_state['deter'],
                 **stats}

        return cast(prior)

    def _gru(self, x, deter):
        raise NotImplementedError

    def observe_td(self, embed, action, is_first, state=None, with_residual=True, with_res_stop_gradients=True, with_student_posterior=False):
        swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])

        obs_step = lambda prev, inputs: self.obs_step(prev[0], *inputs, with_residual=with_residual, with_res_stop_gradients=with_res_stop_gradients, calculate_prior=False)
        inputs = swap(action), swap(embed), swap(is_first)
        start = state, None
        post, _ = jaxutils.scan(obs_step, inputs, start, self._unroll)
        encoder_predictions = {k: swap(v) for k, v in post.items()}

        first_encoder_state = {k: v[:, 0] for k, v in encoder_predictions.items()}
        dynamics_targets = {k: v[:, 1:] for k, v in encoder_predictions.items()}

        img_step = lambda prev, inputs: self.img_step(prev, *inputs, with_residual=with_residual, with_res_stop_gradients=with_res_stop_gradients)
        inputs = (swap(action[:, 1:]), )
        start = first_encoder_state
        dynamics_preds = jaxutils.scan(img_step, inputs, start, self._unroll)
        dynamics_preds = {k: swap(v) for k, v in dynamics_preds.items()}

        head_input_states = {k: jnp.concatenate((encoder_predictions[k][:, 0:1], dynamics_preds[k]), axis=1) for k in dynamics_preds.keys()}

        return head_input_states, dynamics_preds, dynamics_targets, encoder_predictions

    def obs_step(self, prev_state, prev_action, embed, is_first, with_residual=True, with_res_stop_gradients=True, calculate_prior=True, with_student_posterior=False):
        is_first = cast(is_first)
        prev_action = cast(prev_action)
        if self._action_clip > 0.0:
            prev_action *= sg(self._action_clip / jnp.maximum(
                self._action_clip, jnp.abs(prev_action)))

        prev_state, prev_action = jax.tree_util.tree_map(
            lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))
        prev_state = jax.tree_util.tree_map(
            lambda x, y: x + self._mask(y, is_first),
            prev_state, self.initial(len(is_first)))

        if calculate_prior:
            prior = self.img_step(prev_state, prev_action,
                                  with_residual=with_residual,
                                  with_res_stop_gradients=with_res_stop_gradients)
        else:
            prior = None

        x = embed
        x = self.get('obs_out', Linear, **self._kw)(x)
        stats = self._stats('obs_stats', x)

        post = {
                'deter': prev_state['deter'],
                **stats}

        if self.residual and self.residual.residual_stats_key():
            if with_residual and calculate_prior:
                post[self.residual.residual_stats_key()] = prior[self.residual.residual_stats_key()]
            else:
                post.update(self.residual.initial(bs=is_first.shape[0]))

        return cast(post), cast(prior)


    def dyn_loss(self, post, prior, is_first, impl='kl', free=1.0):
        if impl == 'kl':
            loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
        elif impl == 'kl_masked':
            is_first = cast(is_first)
            # if this is the first timestep then loss should be 0
            loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
            loss = jnp.where(is_first, 0, loss)
        elif impl == 'logprob':
            loss = -self.get_dist(prior).log_prob(sg(post['stoch']))
        else:
            raise NotImplementedError(impl)
        if free:
            loss = jnp.maximum(loss, free)
        return loss

    def rep_loss(self, post, prior, is_first, impl='kl', free=1.0):
        if impl == 'kl':
            loss = self.get_dist(post, use_all_variables=True).kl_divergence(self.get_dist(sg(prior), use_all_variables=True))
        elif impl == 'kl_masked':
            is_first = cast(is_first)
            # if this is the first timestep then loss should be 0
            loss = self.get_dist(post, use_all_variables=True).kl_divergence(self.get_dist(sg(prior), use_all_variables=True))
            loss = jnp.where(is_first, 0, loss)
        elif impl == 'uniform':
            uniform = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), prior)
            loss = self.get_dist(post, use_all_variables=True).kl_divergence(self.get_dist(uniform, use_all_variables=True))
        elif impl == 'entropy':
            loss = -self.get_dist(post, use_all_variables=True).entropy()
        elif impl == 'none':
            loss = jnp.zeros(post['deter'].shape[:-1])
        else:
            raise NotImplementedError(impl)
        if free:
            loss = jnp.maximum(loss, free)
        return loss

    def get_prior_residual_correction(self, with_res_stop_gradients, prev_action, prev_stoch_params, prior_stoch_raw_logits):
        if with_res_stop_gradients:
            prev_action = sg(prev_action)
            prev_stoch_params = sg(prev_stoch_params)
            prior_stoch_raw_logits = sg(prior_stoch_raw_logits)

        corrected_prior_stats = self.residual(
            prev_stoch_params=prev_stoch_params,
            prev_action=prev_action,
            prior_stoch_raw_logits=prior_stoch_raw_logits
        )
        return corrected_prior_stats

class TDDeterministicZOnlyDreamerModelSmall(TDDeterministicZOnlyDreamerModel):
    def obs_step(self, prev_state, prev_action, embed, is_first, with_residual=True, with_res_stop_gradients=True, calculate_prior=True):
        is_first = cast(is_first)
        prev_action = cast(prev_action)
        if self._action_clip > 0.0:
            prev_action *= sg(self._action_clip / jnp.maximum(
                self._action_clip, jnp.abs(prev_action)))

        prev_state, prev_action = jax.tree_util.tree_map(
            lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))
        prev_state = jax.tree_util.tree_map(
            lambda x, y: x + self._mask(y, is_first),
            prev_state, self.initial(len(is_first)))

        if calculate_prior:
            prior = self.img_step(prev_state, prev_action,
                                  with_residual=with_residual,
                                  with_res_stop_gradients=with_res_stop_gradients)
        else:
            prior = None

        x = embed
        # x = self.get('obs_out', Linear, **self._kw)(x)
        stats = self._stats('obs_stats', x)

        post = {
                'deter': prev_state['deter'],
                **stats}

        if self.residual and self.residual.residual_stats_key():
            if with_residual:
                post[self.residual.residual_stats_key()] = prior[self.residual.residual_stats_key()]
            else:
                post.update(self.residual.initial(bs=is_first.shape[0]))

        return cast(post), cast(prior)



class TDStochasticZOnlyDreamerModel(RSSM):

    def __init__(
            self, deter=1024, stoch=32, classes=32, unroll=False, initial='learned',
            unimix=0.01, action_clip=1.0, img_hidden_layers=2,
            use_gru_with_prior_belief=False,
            stoch_params_include_unimix=False,
            stoch_params_are_raw_logits=False,
            use_half_of_stoch_as_free_variables=False,
            use_posterior_stoch_params_for_first_state=False,
            use_posterior_stoch_params_for_all_states=False,
            dynamics_takes_prev_stoch_as_input=True,
            residual=None,
            **kw):
        self._deter = deter
        self._stoch = stoch
        self._classes = classes
        self._unroll = unroll
        self._initial = initial
        self._unimix = unimix
        self._action_clip = action_clip
        self._img_hidden_layers = img_hidden_layers
        self._use_gru_with_prior_belief = use_gru_with_prior_belief
        self._stoch_params_include_unimix = stoch_params_include_unimix
        self._stoch_params_are_raw_logits = stoch_params_are_raw_logits
        assert not (stoch_params_include_unimix and stoch_params_are_raw_logits), "can only choose one"
        self._use_half_of_stoch_as_free_variables = use_half_of_stoch_as_free_variables
        self._stoch_sample_size = self._stoch // 2 if self._use_half_of_stoch_as_free_variables else self._stoch
        self._use_posterior_stoch_params_for_first_state = use_posterior_stoch_params_for_first_state
        self._use_posterior_stoch_params_for_all_states = use_posterior_stoch_params_for_all_states
        self._dynamics_takes_prev_stoch_as_input = dynamics_takes_prev_stoch_as_input
        if isinstance(residual, str):
            residual = get_residual(residual)
            if residual:
                residual = residual(stoch=stoch, classes=classes, unimix=self._unimix,
                                    stoch_params_are_raw_logits=stoch_params_are_raw_logits,
                                    stoch_params_include_unimix=stoch_params_include_unimix,
                                    img_hidden_layers=img_hidden_layers,
                                    name='residual', **kw)
        self.residual = residual
        self._kw = kw

    def initial(self, bs):
        if self._classes:
            state = dict(
                deter=jnp.zeros([bs, self._deter], f32),
                logit=jnp.zeros([bs, self._stoch, self._classes], f32),
                stoch_params=jnp.zeros([bs, self._stoch, self._classes], f32),
                stoch=jnp.zeros([bs, self._stoch, self._classes], f32)
            )
        else:
            raise NotImplementedError

        if self.residual is not None:
            state.update(self.residual.initial(bs))

        if self._initial == 'zeros':
            return cast(state)
        elif self._initial == 'learned':
            print(f"Using all zeroes for initial state even though 'learned' is specified for this.")
            return cast(state)
        else:
            raise NotImplementedError(self._initial)

    def get_dist(self, state, argmax=False, use_all_variables=False):
        if self._classes:
            logit = state['logit'].astype(f32)
            return tfd.Independent(jaxutils.OneHotDist(logit), 1)
        else:
            raise NotImplementedError

    # def _stats(self, name, x): # TODO left off here
    #     if self._classes:
    #         x = self.get(name, Linear, self._stoch * self._classes)(x)
    #         logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
    #         probs = jax.nn.softmax(logit, -1)
    #         stats = {'logit': logit,
    #                  'stoch_params': probs,
    #                  }
    #         return stats
    #     else:
    #         raise NotImplementedError

    def _stats(self, name, x):
        if self._classes:
            x = self.get(name, Linear, self._stoch * self._classes)(x)
            logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
            probs = jax.nn.softmax(logit, -1)
            if self._unimix:
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - self._unimix) * probs + self._unimix * uniform
                logit = jnp.log(probs)
            stats = {'logit': logit, "stoch_params": probs}
            return stats
        else:
            raise NotImplementedError()

    def imagine(self, action, state=None):
        swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
        state = self.initial(action.shape[0]) if state is None else state
        assert isinstance(state, dict), state
        action = swap(action)
        step = partial(self.img_step, with_residual=True, with_res_stop_gradients=False)
        prior = jaxutils.scan(step, action, state, self._unroll)
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def img_step(self, prev_state, prev_action, with_residual=True, with_res_stop_gradients=False):
        prev_stoch_params = prev_state['stoch_params']
        prev_stoch = prev_state['stoch']
        prev_action = cast(prev_action)
        if self._action_clip > 0.0:
            prev_action *= sg(self._action_clip / jnp.maximum(
                self._action_clip, jnp.abs(prev_action)))

        if self._classes:
            shape = prev_stoch_params.shape[:-2] + (self._stoch * self._classes,)
            prev_stoch_params = prev_stoch_params.reshape(shape)
            prev_stoch = prev_stoch.reshape(shape)
        if len(prev_action.shape) > len(prev_stoch_params.shape):  # 2D actions.
            shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
            prev_action = prev_action.reshape(shape)

        x = jnp.concatenate([prev_stoch_params, prev_stoch, prev_action], -1)

        x = self.get('img_in', Linear, **self._kw)(x)
        for i in range(self._img_hidden_layers):
            x = self.get(f'img_hidden{i+1}', Linear, **self._kw)(x)
        x = self.get('img_out', Linear, **self._kw)(x)
        stats = self._stats('img_stats', x)

        if with_residual and self.residual:
            stats = self.get_prior_residual_correction(
                with_res_stop_gradients=with_res_stop_gradients,
                prev_stoch=prev_stoch,
                prev_action=prev_action,
                prev_stoch_params=prev_stoch_params,
                prior_stoch_params=stats['stoch_params'],
                prior_stoch_raw_logits=stats['stoch_raw_logits']
            )
        elif self.residual:
            stats.update(self.residual.initial(bs=prev_action.shape[0]))

        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        prior = {
                 'stoch': stoch,
                 'deter': prev_state['deter'],
                 **stats}

        return cast(prior)

    def _gru(self, x, deter):
        raise NotImplementedError

    def observe_td(self, embed, action, is_first, state=None, with_residual=True, with_res_stop_gradients=True):
        swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])

        obs_step = lambda prev, inputs: self.obs_step(prev[0], *inputs, with_residual=with_residual, with_res_stop_gradients=with_res_stop_gradients, calculate_prior=False)
        inputs = swap(action), swap(embed), swap(is_first)
        start = state, None
        post, _ = jaxutils.scan(obs_step, inputs, start, self._unroll)
        encoder_predictions = {k: swap(v) for k, v in post.items()}

        first_encoder_state = {k: v[:, 0] for k, v in encoder_predictions.items()}
        dynamics_targets = {k: v[:, 1:] for k, v in encoder_predictions.items()}

        img_step = lambda prev, inputs: self.img_step(prev, *inputs, with_residual=with_residual, with_res_stop_gradients=with_res_stop_gradients)
        inputs = (swap(action[:, 1:]), )
        start = first_encoder_state
        dynamics_preds = jaxutils.scan(img_step, inputs, start, self._unroll)
        dynamics_preds = {k: swap(v) for k, v in dynamics_preds.items()}

        head_input_states = {k: jnp.concatenate((encoder_predictions[k][:, 0:1], dynamics_preds[k]), axis=1) for k in dynamics_preds.keys()}

        return head_input_states, dynamics_preds, dynamics_targets, encoder_predictions

    def obs_step(self, prev_state, prev_action, embed, is_first, with_residual=True, with_res_stop_gradients=True, calculate_prior=True):
        is_first = cast(is_first)
        prev_action = cast(prev_action)
        if self._action_clip > 0.0:
            prev_action *= sg(self._action_clip / jnp.maximum(
                self._action_clip, jnp.abs(prev_action)))

        prev_state, prev_action = jax.tree_util.tree_map(
            lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))
        prev_state = jax.tree_util.tree_map(
            lambda x, y: x + self._mask(y, is_first),
            prev_state, self.initial(len(is_first)))

        if calculate_prior:
            prior = self.img_step(prev_state, prev_action,
                                  with_residual=with_residual,
                                  with_res_stop_gradients=with_res_stop_gradients)
        else:
            prior = None

        x = embed
        x = self.get('obs_out', Linear, **self._kw)(x)
        stats = self._stats('obs_stats', x)

        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        post = {
                'stoch': stoch,
                'deter': prev_state['deter'],
                **stats}

        if self.residual and self.residual.residual_stats_key():
            if with_residual:
                post[self.residual.residual_stats_key()] = prior[self.residual.residual_stats_key()]
            else:
                post.update(self.residual.initial(bs=is_first.shape[0]))

        return cast(post), cast(prior)


    def dyn_loss(self, post, prior, is_first, impl='kl', free=1.0):
        if impl == 'kl':
            loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
        elif impl == 'kl_masked':
            is_first = cast(is_first)
            # if this is the first timestep then loss should be 0
            loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
            loss = jnp.where(is_first, 0, loss)
        elif impl == 'logprob':
            loss = -self.get_dist(prior).log_prob(sg(post['stoch']))
        else:
            raise NotImplementedError(impl)
        if free:
            loss = jnp.maximum(loss, free)
        return loss

    def rep_loss(self, post, prior, is_first, impl='kl', free=1.0):
        if impl == 'kl':
            loss = self.get_dist(post, use_all_variables=True).kl_divergence(self.get_dist(sg(prior), use_all_variables=True))
        elif impl == 'kl_masked':
            is_first = cast(is_first)
            # if this is the first timestep then loss should be 0
            loss = self.get_dist(post, use_all_variables=True).kl_divergence(self.get_dist(sg(prior), use_all_variables=True))
            loss = jnp.where(is_first, 0, loss)
        elif impl == 'uniform':
            uniform = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), prior)
            loss = self.get_dist(post, use_all_variables=True).kl_divergence(self.get_dist(uniform, use_all_variables=True))
        elif impl == 'entropy':
            loss = -self.get_dist(post, use_all_variables=True).entropy()
        elif impl == 'none':
            loss = jnp.zeros(post['deter'].shape[:-1])
        else:
            raise NotImplementedError(impl)
        if free:
            loss = jnp.maximum(loss, free)
        return loss

    def get_prior_residual_correction(self, with_res_stop_gradients, prev_stoch, prev_action, prev_stoch_params, prior_stoch_params, prior_stoch_raw_logits):
        if with_res_stop_gradients:
            prev_stoch = sg(prev_stoch)
            prev_action = sg(prev_action)
            prev_stoch_params = sg(prev_stoch_params)
            prior_stoch_params = sg(prior_stoch_params)
            prior_stoch_raw_logits = sg(prior_stoch_raw_logits)

        corrected_prior_stats = self.residual(
            prev_stoch=prev_stoch,
            prev_stoch_params=prev_stoch_params,
            prior_stoch_params=prior_stoch_params,
            prev_action=prev_action,
            prior_stoch_raw_logits=prior_stoch_raw_logits
        )
        return corrected_prior_stats


class TDCompressedStochasticZOnlyDreamerModel(TDStochasticZOnlyDreamerModel):

    def initial(self, bs):
        if self._classes:
            state = dict(
                deter=jnp.zeros([bs, self._deter], f32),
                logit=jnp.zeros([bs, self._stoch, self._classes], f32),
                stoch_params=jnp.zeros([bs, self._stoch, self._classes], f32),
                stoch=jnp.zeros([bs, self._stoch, self._classes], f32),
                compressed_stoch=jnp.zeros([bs, self._kw['units'] // self._classes], f32)
            )
        else:
            raise NotImplementedError

        if self.residual is not None:
            state.update(self.residual.initial(bs))

        if self._initial == 'zeros':
            return cast(state)
        elif self._initial == 'learned':
            print(f"Using all zeroes for initial state even though 'learned' is specified for this.")
            return cast(state)
        else:
            raise NotImplementedError(self._initial)

    def img_step(self, prev_state, prev_action, with_residual=True, with_res_stop_gradients=False):
        prev_stoch_params = prev_state['stoch_params']
        prev_stoch = prev_state['stoch']
        prev_action = cast(prev_action)
        if self._action_clip > 0.0:
            prev_action *= sg(self._action_clip / jnp.maximum(
                self._action_clip, jnp.abs(prev_action)))

        if self._classes:
            shape = prev_stoch_params.shape[:-2] + (self._stoch * self._classes,)
            prev_stoch_params = prev_stoch_params.reshape(shape)
            prev_stoch = prev_stoch.reshape(shape)
        if len(prev_action.shape) > len(prev_stoch_params.shape):  # 2D actions.
            shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
            prev_action = prev_action.reshape(shape)


        x = jnp.concatenate([prev_stoch_params, prev_state['compressed_stoch'], prev_action], -1)

        x = self.get('img_in', Linear, **self._kw)(x)
        for i in range(self._img_hidden_layers):
            x = self.get(f'img_hidden{i + 1}', Linear, **self._kw)(x)
        x = self.get('img_out', Linear, **self._kw)(x)
        stats = self._stats('img_stats', x)

        if with_residual and self.residual:
            stats = self.get_prior_residual_correction(
                with_res_stop_gradients=with_res_stop_gradients,
                prev_stoch=prev_stoch,
                prev_action=prev_action,
                prev_stoch_params=prev_stoch_params,
                prior_stoch_params=stats['stoch_params'],
                prior_stoch_raw_logits=stats['stoch_raw_logits']
            )
        elif self.residual:
            stats.update(self.residual.initial(bs=prev_action.shape[0]))

        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        stoch_kw = {**self._kw}
        stoch_kw['units'] = stoch_kw['units'] // self._classes
        compressed_stoch = self.get('compress_stoch', Linear, **stoch_kw)(stoch.reshape(stoch.shape[:-2] + (self._stoch * self._classes,)))

        prior = {
            'stoch': stoch,
            'compressed_stoch': compressed_stoch,
            'deter': prev_state['deter'],
            **stats}

        return cast(prior)

    def obs_step(self, prev_state, prev_action, embed, is_first, with_residual=True, with_res_stop_gradients=True, calculate_prior=True):
        is_first = cast(is_first)
        prev_action = cast(prev_action)
        if self._action_clip > 0.0:
            prev_action *= sg(self._action_clip / jnp.maximum(
                self._action_clip, jnp.abs(prev_action)))

        prev_state, prev_action = jax.tree_util.tree_map(
            lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))
        prev_state = jax.tree_util.tree_map(
            lambda x, y: x + self._mask(y, is_first),
            prev_state, self.initial(len(is_first)))

        if calculate_prior:
            prior = self.img_step(prev_state, prev_action,
                                  with_residual=with_residual,
                                  with_res_stop_gradients=with_res_stop_gradients)
        else:
            prior = None

        x = embed
        x = self.get('obs_out', Linear, **self._kw)(x)
        stats = self._stats('obs_stats', x)

        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        stoch_kw = {**self._kw}
        stoch_kw['units'] = stoch_kw['units'] // self._classes
        compressed_stoch = self.get('compress_stoch', Linear, **stoch_kw)(stoch.reshape(stoch.shape[:-2] + (self._stoch * self._classes,)))

        post = {
                'stoch': stoch,
                'compressed_stoch': compressed_stoch,
                'deter': prev_state['deter'],
                **stats}

        if self.residual and self.residual.residual_stats_key():
            if with_residual:
                post[self.residual.residual_stats_key()] = prior[self.residual.residual_stats_key()]
            else:
                post.update(self.residual.initial(bs=is_first.shape[0]))

        return cast(post), cast(prior)


class TDDummyStochasticZOnlyDreamerModel(RSSM):

    def __init__(
            self, deter=1024, stoch=32, classes=32, unroll=False, initial='learned',
            unimix=0.01, action_clip=1.0, img_hidden_layers=2,
            use_gru_with_prior_belief=False,
            stoch_params_include_unimix=False,
            stoch_params_are_raw_logits=False,
            use_half_of_stoch_as_free_variables=False,
            use_posterior_stoch_params_for_first_state=False,
            use_posterior_stoch_params_for_all_states=False,
            dynamics_takes_prev_stoch_as_input=True,
            residual=None,
            **kw):
        self._deter = deter
        self._stoch = stoch
        self._classes = classes
        self._unroll = unroll
        self._initial = initial
        self._unimix = unimix
        self._action_clip = action_clip
        self._img_hidden_layers = img_hidden_layers
        self._use_gru_with_prior_belief = use_gru_with_prior_belief
        self._stoch_params_include_unimix = stoch_params_include_unimix
        self._stoch_params_are_raw_logits = stoch_params_are_raw_logits
        assert not (stoch_params_include_unimix and stoch_params_are_raw_logits), "can only choose one"
        self._use_half_of_stoch_as_free_variables = use_half_of_stoch_as_free_variables
        self._stoch_sample_size = self._stoch // 2 if self._use_half_of_stoch_as_free_variables else self._stoch
        self._use_posterior_stoch_params_for_first_state = use_posterior_stoch_params_for_first_state
        self._use_posterior_stoch_params_for_all_states = use_posterior_stoch_params_for_all_states
        self._dynamics_takes_prev_stoch_as_input = dynamics_takes_prev_stoch_as_input
        if isinstance(residual, str):
            residual = get_residual(residual)
            if residual:
                residual = residual(stoch=stoch, classes=classes, unimix=self._unimix,
                                    stoch_params_are_raw_logits=stoch_params_are_raw_logits,
                                    stoch_params_include_unimix=stoch_params_include_unimix,
                                    img_hidden_layers=img_hidden_layers,
                                    name='residual', **kw)
        self.residual = residual
        self._kw = kw

    def initial(self, bs):
        if self._classes:
            state = dict(
                deter=jnp.zeros([bs, self._deter], f32),
                logit=jnp.zeros([bs, self._stoch, self._classes], f32),
                stoch_params=jnp.zeros([bs, self._stoch, self._classes], f32),
                stoch=jnp.zeros([bs, self._stoch, self._classes], f32)
            )
        else:
            raise NotImplementedError

        if self.residual is not None:
            state.update(self.residual.initial(bs))

        if self._initial == 'zeros':
            return cast(state)
        elif self._initial == 'learned':
            print(f"Using all zeroes for initial state even though 'learned' is specified for this.")
            return cast(state)
        else:
            raise NotImplementedError(self._initial)

    def get_dist(self, state, argmax=False, use_all_variables=False):
        if self._classes:
            logit = state['logit'].astype(f32)
            return tfd.Independent(jaxutils.OneHotDist(logit), 1)
        else:
            raise NotImplementedError

    # def _stats(self, name, x): # TODO left off here
    #     if self._classes:
    #         x = self.get(name, Linear, self._stoch * self._classes)(x)
    #         logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
    #         probs = jax.nn.softmax(logit, -1)
    #         stats = {'logit': logit,
    #                  'stoch_params': probs,
    #                  }
    #         return stats
    #     else:
    #         raise NotImplementedError

    def _stats(self, name, x):
        if self._classes:
            x = self.get(name, Linear, self._stoch * self._classes)(x)
            logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
            probs = jax.nn.softmax(logit, -1)
            # if self._unimix:
            #     uniform = jnp.ones_like(probs) / probs.shape[-1]
            #     probs = (1 - self._unimix) * probs + self._unimix * uniform
            #     logit = jnp.log(probs)
            stats = {'logit': logit, "stoch_params": probs}
            return stats
        else:
            raise NotImplementedError()

    def imagine(self, action, state=None):
        swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
        state = self.initial(action.shape[0]) if state is None else state
        assert isinstance(state, dict), state
        action = swap(action)
        step = partial(self.img_step, with_residual=True, with_res_stop_gradients=False)
        prior = jaxutils.scan(step, action, state, self._unroll)
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def img_step(self, prev_state, prev_action, with_residual=True, with_res_stop_gradients=False):
        prev_stoch_params = prev_state['stoch_params']
        # prev_stoch = prev_state['stoch']
        prev_action = cast(prev_action)
        if self._action_clip > 0.0:
            prev_action *= sg(self._action_clip / jnp.maximum(
                self._action_clip, jnp.abs(prev_action)))

        if self._classes:
            shape = prev_stoch_params.shape[:-2] + (self._stoch * self._classes,)
            prev_stoch_params = prev_stoch_params.reshape(shape)
            # prev_stoch = prev_stoch.reshape(shape)
        if len(prev_action.shape) > len(prev_stoch_params.shape):  # 2D actions.
            shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
            prev_action = prev_action.reshape(shape)

        x = jnp.concatenate([prev_stoch_params, prev_action], -1)

        x = self.get('img_in', Linear, **self._kw)(x)
        for i in range(self._img_hidden_layers):
            x = self.get(f'img_hidden{i+1}', Linear, **self._kw)(x)
        x = self.get('img_out', Linear, **self._kw)(x)
        stats = self._stats('img_stats', x)

        if with_residual and self.residual:
            raise NotImplementedError()
            stats = self.get_prior_residual_correction(
                with_res_stop_gradients=with_res_stop_gradients,
                prev_stoch=prev_stoch,
                prev_action=prev_action,
                prev_stoch_params=prev_stoch_params,
                prior_stoch_params=stats['stoch_params'],
                prior_stoch_raw_logits=stats['stoch_raw_logits']
            )
        elif self.residual:
            stats.update(self.residual.initial(bs=prev_action.shape[0]))

        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        prior = {
                 'stoch': stoch,
                 'deter': prev_state['deter'],
                 **stats}

        return cast(prior)

    def _gru(self, x, deter):
        raise NotImplementedError

    def observe_td(self, embed, action, is_first, state=None, with_residual=True, with_res_stop_gradients=True):
        swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])

        obs_step = lambda prev, inputs: self.obs_step(prev[0], *inputs, with_residual=with_residual, with_res_stop_gradients=with_res_stop_gradients, calculate_prior=False)
        inputs = swap(action), swap(embed), swap(is_first)
        start = state, None
        post, _ = jaxutils.scan(obs_step, inputs, start, self._unroll)
        encoder_predictions = {k: swap(v) for k, v in post.items()}

        first_encoder_state = {k: v[:, 0] for k, v in encoder_predictions.items()}
        dynamics_targets = {k: v[:, 1:] for k, v in encoder_predictions.items()}

        img_step = lambda prev, inputs: self.img_step(prev, *inputs, with_residual=with_residual, with_res_stop_gradients=with_res_stop_gradients)
        inputs = (swap(action[:, 1:]), )
        start = first_encoder_state
        dynamics_preds = jaxutils.scan(img_step, inputs, start, self._unroll)
        dynamics_preds = {k: swap(v) for k, v in dynamics_preds.items()}

        head_input_states = {k: jnp.concatenate((encoder_predictions[k][:, 0:1], dynamics_preds[k]), axis=1) for k in dynamics_preds.keys()}

        return head_input_states, dynamics_preds, dynamics_targets, encoder_predictions

    def obs_step(self, prev_state, prev_action, embed, is_first, with_residual=True, with_res_stop_gradients=True, calculate_prior=True):
        is_first = cast(is_first)
        prev_action = cast(prev_action)
        if self._action_clip > 0.0:
            prev_action *= sg(self._action_clip / jnp.maximum(
                self._action_clip, jnp.abs(prev_action)))

        prev_state, prev_action = jax.tree_util.tree_map(
            lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))
        prev_state = jax.tree_util.tree_map(
            lambda x, y: x + self._mask(y, is_first),
            prev_state, self.initial(len(is_first)))

        if calculate_prior:
            prior = self.img_step(prev_state, prev_action,
                                  with_residual=with_residual,
                                  with_res_stop_gradients=with_res_stop_gradients)
        else:
            prior = None

        x = embed
        x = self.get('obs_out', Linear, **self._kw)(x)
        stats = self._stats('obs_stats', x)

        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        post = {
                'stoch': stoch,
                'deter': prev_state['deter'],
                **stats}

        if self.residual and self.residual.residual_stats_key():
            if with_residual:
                post[self.residual.residual_stats_key()] = prior[self.residual.residual_stats_key()]
            else:
                post.update(self.residual.initial(bs=is_first.shape[0]))

        return cast(post), cast(prior)


    def dyn_loss(self, post, prior, is_first, impl='kl', free=1.0):
        if impl == 'kl':
            loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
        elif impl == 'kl_masked':
            is_first = cast(is_first)
            # if this is the first timestep then loss should be 0
            loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
            loss = jnp.where(is_first, 0, loss)
        elif impl == 'logprob':
            loss = -self.get_dist(prior).log_prob(sg(post['stoch']))
        else:
            raise NotImplementedError(impl)
        if free:
            loss = jnp.maximum(loss, free)
        return loss

    def rep_loss(self, post, prior, is_first, impl='kl', free=1.0):
        if impl == 'kl':
            loss = self.get_dist(post, use_all_variables=True).kl_divergence(self.get_dist(sg(prior), use_all_variables=True))
        elif impl == 'kl_masked':
            is_first = cast(is_first)
            # if this is the first timestep then loss should be 0
            loss = self.get_dist(post, use_all_variables=True).kl_divergence(self.get_dist(sg(prior), use_all_variables=True))
            loss = jnp.where(is_first, 0, loss)
        elif impl == 'uniform':
            uniform = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), prior)
            loss = self.get_dist(post, use_all_variables=True).kl_divergence(self.get_dist(uniform, use_all_variables=True))
        elif impl == 'entropy':
            loss = -self.get_dist(post, use_all_variables=True).entropy()
        elif impl == 'none':
            loss = jnp.zeros(post['deter'].shape[:-1])
        else:
            raise NotImplementedError(impl)
        if free:
            loss = jnp.maximum(loss, free)
        return loss

    def get_prior_residual_correction(self, with_res_stop_gradients, prev_stoch, prev_action, prev_stoch_params, prior_stoch_params, prior_stoch_raw_logits):
        if with_res_stop_gradients:
            prev_stoch = sg(prev_stoch)
            prev_action = sg(prev_action)
            prev_stoch_params = sg(prev_stoch_params)
            prior_stoch_params = sg(prior_stoch_params)
            prior_stoch_raw_logits = sg(prior_stoch_raw_logits)

        corrected_prior_stats = self.residual(
            prev_stoch=prev_stoch,
            prev_stoch_params=prev_stoch_params,
            prior_stoch_params=prior_stoch_params,
            prev_action=prev_action,
            prior_stoch_raw_logits=prior_stoch_raw_logits
        )
        return corrected_prior_stats
