from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from functools import partial

f32 = jnp.float32
tfd = tfp.distributions
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

from dreamerv3 import jaxutils
from dreamerv3 import ninjax as nj
from dreamerv3.nets import Linear, MLP
from dreamerv3.simdreamer_nets import GroundedRSSM

cast = jaxutils.cast_to_compute

#
#
# class GroundedRSSMStochOnlyNonSequentialPosteriorMLPPriorLargerWithPriorBeliefAndResidual(GroundedRSSM):
#
#
#     def initial(self, bs):
#         if self._classes:
#             state = dict(
#                 deter=jnp.zeros([bs, self._deter], f32),
#                 logit=jnp.zeros([bs, self._stoch, self._classes], f32),
#                 stoch=jnp.zeros([bs, self._stoch_sample_size, self._classes], f32),
#                 stoch_params=jnp.zeros([bs, self._stoch * self._classes], f32),
#                 symlog_grounded=jnp.zeros([bs, self._grounded_size], f32)
#             )
#         else:
#             state = dict(
#                 deter=jnp.zeros([bs, self._deter], f32),
#                 mean=jnp.zeros([bs, self._stoch], f32),
#                 std=jnp.ones([bs, self._stoch], f32),
#                 stoch=jnp.zeros([bs, self._stoch_sample_size], f32),
#                 stoch_params=jnp.zeros([bs, self._stoch * 2], f32),
#                 symlog_grounded=jnp.zeros([bs, self._grounded_size], f32)
#             )
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
#             if self._use_half_of_stoch_as_free_variables and not use_all_variables:
#                 logit, free_logit = jnp.split(logit, 2, axis=-2)
#             return tfd.Independent(jaxutils.OneHotDist(logit), 1)
#         else:
#             mean = state['mean'].astype(f32)
#             std = state['std'].astype(f32)
#             if self._use_half_of_stoch_as_free_variables and not use_all_variables:
#                 mean, free_mean = jnp.split(mean, 2, axis=-1)
#                 std, free_std = jnp.split(std, 2, axis=-1)
#             return tfd.MultivariateNormalDiag(mean, std)
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
#
#             stoch_params = probs if self._stoch_params_include_unimix else orig_probs
#             stoch_params = stoch_params.reshape(stoch_params.shape[:-2] + (self._stoch * self._classes,))
#
#             stats = {'logit': logit,
#                      'stoch_params': stoch_params
#                      }
#             return stats
#         else:
#             x = self.get(name, Linear, 2 * self._stoch)(x)
#             mean, std = jnp.split(x, 2, -1)
#             std = 2 * jax.nn.sigmoid(std / 2) + 0.1
#             return {'mean': mean, 'std': std,
#                     'stoch_params': jnp.concatenate([mean, std], axis=-1)
#                     }
#
#     def get_grounded_from_stoch(self, stoch):
#         if self._classes:
#             # flatten stoch if the state representation is multi-dimensional
#             new_shape = stoch.shape[:-2] + (self._stoch * self._classes,)
#             stoch = stoch.reshape(new_shape)
#
#         # x = jnp.concatenate([deter, stoch], -1)
#         x = stoch
#         x = self.get('grounded_in', Linear, **self._kw)(x)
#         x = self.get('grounded_hidden', Linear, **self._kw)(x)
#         grounded_symlog_out = self.get('grounded_symlog_out', Linear, self._grounded_size)(x)
#         return cast(grounded_symlog_out)
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
#     def img_step(self, prev_state, prev_action, with_residual=True, with_res_stop_gradients=True):
#         assert self._sim_tf is None
#
#         prev_stoch = prev_state['stoch']
#         prev_stoch_params = prev_state['stoch_params']
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
#         # print(f"img_inputs shape: {img_inputs.shape}")
#         # print(f"prev_action shape: {prev_action.shape}")
#
#         if self._use_gru_with_prior_belief:
#             x = jnp.concatenate([prev_stoch, prev_action], -1)
#             x = self.get('img_in', Linear, **self._kw)(x)
#             stats = self._gru(x, deter=prev_stoch_params)
#         else:
#             x = jnp.concatenate([prev_stoch_params, prev_stoch, prev_action], -1)
#
#             x = self.get('img_in', Linear, **self._kw)(x)
#             for i in range(self._img_hidden_layers):
#                 x = self.get(f'img_hidden{i+1}', Linear, **self._kw)(x)
#             x = self.get('img_out', Linear, **self._kw)(x)
#             stats = self._stats('img_stats', x)
#
#
#         if with_residual:
#             stats = self.get_prior_residual_correction(
#                 with_res_stop_gradients=with_res_stop_gradients,
#                 prev_stoch=prev_stoch,
#                 prev_action=prev_action,
#                 prev_stoch_params=prev_stoch_params,
#                 prior_stoch_params=stats['stoch_params'])
#
#         dist = self.get_dist(stats)
#         stoch = dist.sample(seed=nj.rng())
#         print(f"img stoch shape: {stoch.shape}")
#         symlog_grounded = self.get_grounded_from_stoch(stoch)
#
#         prior = {'stoch': stoch,
#                  'deter': prev_state['deter'],
#                  'symlog_grounded': symlog_grounded,
#                  # 'prior_stoch_params': stats['stoch_params'],
#                  **stats}
#
#         return cast(prior)
#
#     def _gru(self, x, deter):
#         x = jnp.concatenate([deter, x], -1)
#
#         if self._classes:
#             gru_size = self._stoch * self._classes
#         else:
#             gru_size = 2 * self._stoch
#
#         kw = {**self._kw, 'act': 'none', 'units': 3 * gru_size}
#         x = self.get('gru', Linear, **kw)(x)
#         reset, cand, update = jnp.split(x, 3, -1)
#         reset = jax.nn.sigmoid(reset)
#         # cand = jnp.tanh(reset * cand)
#
#         if self._classes:
#             # TODO this implementation for a discrete z dist is done totally wrong here for GRU
#             x = cand
#             logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
#             probs = jax.nn.sigmoid(logit, -1)
#             if self._unimix:
#                 uniform = jnp.ones_like(probs) / probs.shape[-1]
#                 probs = (1 - self._unimix) * probs + self._unimix * uniform
#             cand_stoch_params = probs.reshape(probs.shape[:-2] + (self._stoch * self._classes,))
#         else:
#             x = cand
#             mean, std = jnp.split(x, 2, -1)
#             std = 2 * jax.nn.sigmoid(std / 2) + 0.1
#             cand_stoch_params = jnp.concatenate([mean, std], -1)
#
#         cand_stoch_params = reset * cand_stoch_params
#         update = jax.nn.sigmoid(update - 1)
#         new_stoch_params = update * cand_stoch_params + (1 - update) * deter
#
#         if self._classes:
#             # TODO this implementation for a discrete z dist is done totally wrong here for GRU
#             # renormalize probs
#             probs = new_stoch_params.reshape(new_stoch_params.shape[:-1] + (self._stoch, self._classes))
#             probs_sum = jnp.sum(probs, axis=-1)
#             probs = probs / probs_sum
#             logit = jnp.log(probs)
#             new_stoch_params = probs.reshape(probs.shape[:-2] + (self._stoch * self._classes,))
#             stats = {'logit': logit,
#                      'stoch_params': new_stoch_params,
#                      }
#             return stats
#         else:
#             mean, std = jnp.split(new_stoch_params, 2, -1)
#             return {'mean': mean, 'std': std,
#                     'stoch_params': new_stoch_params
#                     }
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
#         prior = self.img_step(prev_state, prev_action,
#                               with_residual=with_residual,
#                               with_res_stop_gradients=with_res_stop_gradients)
#
#         # x = jnp.concatenate([prev_action, embed], -1)
#         # print(f"posterior is conditioned on embedding and previous action")
#
#         x = embed
#         x = self.get('obs_out', Linear, **self._kw)(x)
#         stats = self._stats('obs_stats', x)
#         dist = self.get_dist(stats)
#         stoch = dist.sample(seed=nj.rng())
#         symlog_grounded = self.get_grounded_from_stoch(stoch)
#
#         print(f"obs stoch shape: {stoch.shape}")
#
#         if self._use_posterior_stoch_params_for_first_state:
#             stoch_params = jnp.where(jnp.tile(is_first[:, None], (1, stats['stoch_params'].shape[-1])),
#                                            stats['stoch_params'], prior['stoch_params'])
#         elif self._use_posterior_stoch_params_for_all_states:
#             stoch_params = stats['stoch_params']
#         else:
#             stoch_params = prior['stoch_params']
#
#         del stats['stoch_params']
#         post = {'stoch': stoch,
#                 'deter': prev_state['deter'],
#                 'symlog_grounded': symlog_grounded,
#                 # 'prior_stoch_params': stoch_params,
#                 'stoch_params': stoch_params,
#                 **stats}
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
#     @staticmethod
#     def grounded_state_loss(post, gt_state):
#         symlog_grounded = post['symlog_grounded']
#         distance = (symlog_grounded - jaxutils.symlog(gt_state)) ** 2  # MSE loss
#         distance = jnp.where(distance < 1e-8, 0, distance)
#         loss = distance.sum(-1)
#         return loss
#
#
#     def get_prior_residual_correction(self, with_res_stop_gradients, prev_stoch, prev_action, prev_stoch_params, prior_stoch_params):
#
#         if with_res_stop_gradients:
#             prev_stoch = sg(prev_stoch)
#             prev_stoch_params = sg(prev_stoch_params)
#             prior_stoch_params = sg(prior_stoch_params)
#
#         x = jnp.concatenate([
#             prev_stoch_params,
#             prev_stoch,
#             prior_stoch_params,
#             prev_action
#         ], -1)
#         x = self.get('residual_in', Linear, **self._kw)(x)
#         corrected_prior_stats = self._stats('residual_stats', x)
#         return corrected_prior_stats
#
#
#
#

