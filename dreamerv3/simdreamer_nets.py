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
from dreamerv3.nets import Linear, MLP, get_residual

cast = jaxutils.cast_to_compute


class GroundedRSSM(nj.Module):

    def __init__(
            self, deter=1024, stoch=32, classes=32, grounded=32,
            sim_tf=None, gradients_through_sim_tf=False,
            unroll=False, initial='learned', unimix=0.01, action_clip=1.0, img_is_deterministic=False, sg_in_img=False,
            img_hidden_layers=2,
            use_gru_with_prior_belief=False,
            stoch_params_include_unimix=False,
            stoch_params_are_raw_logits=False,
            use_half_of_stoch_as_free_variables=False,
            use_posterior_stoch_params_for_first_state=False,
            use_posterior_stoch_params_for_all_states=False,
            residual=None,
            **kw):
        self._deter = deter
        self._stoch = stoch
        self._grounded_size = grounded
        self._classes = classes
        self._sim_tf: Optional[MLP] = sim_tf
        self._gradients_through_sim_tf = gradients_through_sim_tf
        self._unroll = unroll
        self._initial = initial
        self._unimix = unimix
        self._action_clip = action_clip
        self._kw = kw

        self._img_hidden_layers = img_hidden_layers
        self._use_gru_with_prior_belief = use_gru_with_prior_belief
        self._stoch_params_include_unimix = stoch_params_include_unimix
        self._stoch_params_are_raw_logits = stoch_params_are_raw_logits
        assert not (stoch_params_include_unimix and stoch_params_are_raw_logits), "can only choose one"
        self._use_half_of_stoch_as_free_variables = use_half_of_stoch_as_free_variables
        self._stoch_sample_size = self._stoch // 2 if self._use_half_of_stoch_as_free_variables else self._stoch
        self._use_posterior_stoch_params_for_first_state = use_posterior_stoch_params_for_first_state
        self._use_posterior_stoch_params_for_all_states = use_posterior_stoch_params_for_all_states

        if isinstance(residual, str):
            residual = get_residual(residual)
            if residual:
                residual = residual(stoch=stoch, classes=classes, unimix=self._unimix,
                                    stoch_params_are_raw_logits=stoch_params_are_raw_logits,
                                    stoch_params_include_unimix=stoch_params_include_unimix,
                                    img_hidden_layers=img_hidden_layers,
                                    name='residual', **kw)
        self.residual = residual

    def initial(self, bs):
        if self._classes:
            state = dict(
                deter=jnp.zeros([bs, self._deter], f32),
                logit=jnp.zeros([bs, self._stoch, self._classes], f32),
                stoch=jnp.zeros([bs, self._stoch, self._classes], f32),
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))
        else:
            state = dict(
                deter=jnp.zeros([bs, self._deter], f32),
                mean=jnp.zeros([bs, self._stoch], f32),
                std=jnp.ones([bs, self._stoch], f32),
                stoch=jnp.zeros([bs, self._stoch], f32),
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))

        if self._sim_tf:
            state['curr_state_symlog_sim_dynamics_pred'] = jnp.zeros([bs, self._grounded_size], f32)

        if self._initial == 'zeros':
            return cast(state)
        elif self._initial == 'learned':
            deter = self.get('initial', jnp.zeros, state['deter'][0].shape, f32)
            state['deter'] = jnp.repeat(jnp.tanh(deter)[None], bs, 0)
            state['stoch'] = self.get_stoch(cast(state['deter']))
            state['symlog_grounded'] = self.get_grounded(cast(state['deter']), cast(state['stoch']))
            return cast(state)
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None, with_residual=False, with_res_stop_gradients=False):
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

    def imagine(self, action, state=None, with_residual=True, with_res_stop_gradients=False):
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

    def obs_step(self, prev_state, prev_action, embed, is_first):
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
        x = jnp.concatenate([prior['deter'], embed], -1)
        x = self.get('obs_out', Linear, **self._kw)(x)
        stats = self._stats('obs_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded = self.get_grounded(deter=prior['deter'], stoch=stoch)

        post = {'stoch': stoch,
                'deter': prior['deter'],
                'symlog_grounded': symlog_grounded,
                **stats}
        if self._sim_tf is not None:
            post['curr_state_symlog_sim_dynamics_pred'] = prior['curr_state_symlog_sim_dynamics_pred']
        return cast(post), cast(prior)

    def img_step(self, prev_state, prev_action):
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

        if self._sim_tf is not None:
            curr_state_symlog_sim_dynamics_pred = self.get_sim_forward_dynamics(
                grounded_symlog_in=prev_state['symlog_grounded'],
                action=prev_action)
            if not self._gradients_through_sim_tf:
                curr_state_symlog_sim_dynamics_pred = sg(curr_state_symlog_sim_dynamics_pred)
            img_inputs = jnp.concatenate([prev_stoch, prev_action, curr_state_symlog_sim_dynamics_pred], -1)
        else:
            curr_state_symlog_sim_dynamics_pred = None
            img_inputs = jnp.concatenate([prev_stoch, prev_action], -1)

        x = self.get('img_in', Linear, **self._kw)(img_inputs)
        x, deter = self._gru(x, prev_state['deter'])
        x = self.get('img_out', Linear, **self._kw)(x)
        stats = self._stats('img_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded = self.get_grounded(deter=deter, stoch=stoch)

        prior = {'stoch': stoch,
                 'deter': deter,
                 'symlog_grounded': symlog_grounded,
                 **stats}
        if curr_state_symlog_sim_dynamics_pred is not None:
            prior['curr_state_symlog_sim_dynamics_pred'] = curr_state_symlog_sim_dynamics_pred
        return cast(prior)

    def get_stoch(self, deter):
        x = self.get('img_out', Linear, **self._kw)(deter)
        stats = self._stats('img_stats', x)
        dist = self.get_dist(stats)
        return cast(dist.mode())

    def get_grounded(self, deter, stoch):
        if self._classes:
            # flatten stoch if the state representation is multi-dimensional
            new_shape = stoch.shape[:-2] + (self._stoch * self._classes,)
            stoch = stoch.reshape(new_shape)

        x = jnp.concatenate([deter, stoch], -1)
        x = self.get('grounded_in', Linear, **self._kw)(x)

        grounded_symlog_out = self.get('grounded_symlog_out', Linear, self._grounded_size)(x)
        return cast(grounded_symlog_out)

    def get_sim_forward_dynamics(self, grounded_symlog_in, action):
        action = cast(action)
        if self._action_clip > 0.0:
            action *= sg(self._action_clip / jnp.maximum(self._action_clip, jnp.abs(action)))
        x = jnp.concatenate([grounded_symlog_in, action], -1)
        next_grounded_symlog = self._sim_tf(x).mode()
        return cast(next_grounded_symlog)

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
            if self._unimix:
                probs = jax.nn.softmax(logit, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - self._unimix) * probs + self._unimix * uniform
                logit = jnp.log(probs)
            stats = {'logit': logit}
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

    @staticmethod
    def grounded_state_loss(post, gt_state, is_real):
        symlog_grounded = post['symlog_grounded']
        distance = (symlog_grounded - jaxutils.symlog(gt_state)) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)
        loss = jnp.where(is_real, 0, loss)
        return loss

    @staticmethod
    def simulation_tf_loss(symlog_grounded_next_state_pred, target_symlog_next_state, is_last):
        distance = (symlog_grounded_next_state_pred - target_symlog_next_state) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)  # sum over state dimensions, shape is still (batch x sequence len)
        # Only calculate loss for next state prediction when the current state is not the last state in a sequence.
        loss = jnp.where(is_last, 0, loss)
        return loss


class GroundedRSSMLargerGetGrounded(GroundedRSSM):

    def get_grounded(self, deter, stoch):
        if self._classes:
            # flatten stoch if the state representation is multi-dimensional
            new_shape = stoch.shape[:-2] + (self._stoch * self._classes,)
            stoch = stoch.reshape(new_shape)

        x = jnp.concatenate([deter, stoch], -1)
        x = self.get('grounded_in', Linear, **self._kw)(x)
        x = self.get('grounded_hidden', Linear, **self._kw)(x)

        grounded_symlog_out = self.get('grounded_symlog_out', Linear, self._grounded_size)(x)
        return cast(grounded_symlog_out)

class GroundedRSSMGroundedFromStochOnly(GroundedRSSM):

    def get_grounded(self, deter, stoch):
        if self._classes:
            # flatten stoch if the state representation is multi-dimensional
            new_shape = stoch.shape[:-2] + (self._stoch * self._classes,)
            stoch = stoch.reshape(new_shape)

        x = stoch
        x = self.get('grounded_in', Linear, **self._kw)(x)
        x = self.get('grounded_hidden', Linear, **self._kw)(x)
        grounded_symlog_out = self.get('grounded_symlog_out', Linear, self._grounded_size)(x)
        return cast(grounded_symlog_out)


class GroundedRSSMGroundedFromStochOnlyNonSequentialPosterior(GroundedRSSM):

    def get_grounded(self, deter, stoch):
        if self._classes:
            # flatten stoch if the state representation is multi-dimensional
            new_shape = stoch.shape[:-2] + (self._stoch * self._classes,)
            stoch = stoch.reshape(new_shape)

        x = stoch
        x = self.get('grounded_in', Linear, **self._kw)(x)
        x = self.get('grounded_hidden', Linear, **self._kw)(x)
        grounded_symlog_out = self.get('grounded_symlog_out', Linear, self._grounded_size)(x)
        return cast(grounded_symlog_out)

    def obs_step(self, prev_state, prev_action, embed, is_first):
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

        x = jnp.concatenate([prev_action, embed], -1) # TODO should there be a stop_gradient on prev_action for the posterior (ok for prior)? - probably dont need an sg since it originaly still went through prior
        print(f"posterior is conditioned on embedding and previous action")
        x = self.get('obs_out', Linear, **self._kw)(x)
        stats = self._stats('obs_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded = self.get_grounded(deter=prior['deter'], stoch=stoch)

        post = {'stoch': stoch,
                'deter': prior['deter'],
                'symlog_grounded': symlog_grounded,
                **stats}
        if self._sim_tf is not None:
            post['curr_state_symlog_sim_dynamics_pred'] = prior['curr_state_symlog_sim_dynamics_pred']
        return cast(post), cast(prior)

class GroundedRSSMGroundedFromStochOnlyNonSequentialPosteriorNoAction(GroundedRSSMGroundedFromStochOnlyNonSequentialPosterior):

    def obs_step(self, prev_state, prev_action, embed, is_first):
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

        # x = jnp.concatenate([prev_action, embed], -1) # TODO should there be a stop_gradient on prev_action for the posterior (ok for prior)? - probably dont need an sg since it originaly still went through prior
        x = embed
        print(f"posterior is conditioned on embedding and previous action")
        x = self.get('obs_out', Linear, **self._kw)(x)
        stats = self._stats('obs_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded = self.get_grounded(deter=prior['deter'], stoch=stoch)

        post = {'stoch': stoch,
                'deter': prior['deter'],
                'symlog_grounded': symlog_grounded,
                **stats}
        if self._sim_tf is not None:
            post['curr_state_symlog_sim_dynamics_pred'] = prior['curr_state_symlog_sim_dynamics_pred']
        return cast(post), cast(prior)


class GroundedRSSMGroundedFromStochOnlyNonSequentialPosteriorMLPPrior(GroundedRSSM):

    def initial(self, bs):
        if self._classes:
            state = dict(
                logit=jnp.zeros([bs, self._stoch, self._classes], f32),
                stoch=jnp.zeros([bs, self._stoch, self._classes], f32),
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))
        else:
            state = dict(
                mean=jnp.zeros([bs, self._stoch], f32),
                std=jnp.ones([bs, self._stoch], f32),
                stoch=jnp.zeros([bs, self._stoch], f32),
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))

        if self._sim_tf:
            state['curr_state_symlog_sim_dynamics_pred'] = jnp.zeros([bs, self._grounded_size], f32)

        if self._initial == 'zeros':
            return cast(state)
        elif self._initial == 'learned':
            print(f"Using all zeroes for initial state even though 'learned' is specified for this.")
            # deter = self.get('initial', jnp.zeros, state['deter'][0].shape, f32)
            # state['stoch'] = self.get_stoch(deter)
            # state['symlog_grounded'] = self.get_grounded(cast(state['stoch']))
            return cast(state)
        else:
            raise NotImplementedError(self._initial)

    def img_step(self, prev_state, prev_action):
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

        if self._sim_tf is not None:
            curr_state_symlog_sim_dynamics_pred = self.get_sim_forward_dynamics(
                grounded_symlog_in=prev_state['symlog_grounded'],
                action=prev_action)
            if not self._gradients_through_sim_tf:
                curr_state_symlog_sim_dynamics_pred = sg(curr_state_symlog_sim_dynamics_pred)
            img_inputs = jnp.concatenate([prev_stoch, prev_action, curr_state_symlog_sim_dynamics_pred], -1)
        else:
            curr_state_symlog_sim_dynamics_pred = None
            img_inputs = jnp.concatenate([prev_stoch, prev_action], -1)

        x = self.get('img_in', Linear, **self._kw)(img_inputs)
        x = self.get('img_hidden', Linear, **self._kw)(x)
        x = self.get('img_out', Linear, **self._kw)(x)
        stats = self._stats('img_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded = self.get_grounded(deter=None, stoch=stoch)

        prior = {'stoch': stoch,
                 'symlog_grounded': symlog_grounded,
                 **stats}
        if curr_state_symlog_sim_dynamics_pred is not None:
            prior['curr_state_symlog_sim_dynamics_pred'] = curr_state_symlog_sim_dynamics_pred
        return cast(prior)

    def obs_step(self, prev_state, prev_action, embed, is_first):
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

        x = jnp.concatenate([prev_action, embed],
                            -1)  # TODO should there be a stop_gradient on prev_action for the posterior (ok for prior)? - probably dont need an sg since it originaly still went through prior
        print(f"posterior is conditioned on embedding and previous action")
        x = self.get('obs_out', Linear, **self._kw)(x)
        stats = self._stats('obs_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded = self.get_grounded(deter=None, stoch=stoch)

        post = {'stoch': stoch,
                'symlog_grounded': symlog_grounded,
                **stats}
        if self._sim_tf is not None:
            post['curr_state_symlog_sim_dynamics_pred'] = prior['curr_state_symlog_sim_dynamics_pred']
        return cast(post), cast(prior)

    def get_grounded(self, deter, stoch):
        if self._classes:
            # flatten stoch if the state representation is multi-dimensional
            new_shape = stoch.shape[:-2] + (self._stoch * self._classes,)
            stoch = stoch.reshape(new_shape)

        x = stoch
        x = self.get('grounded_in', Linear, **self._kw)(x)
        x = self.get('grounded_hidden', Linear, **self._kw)(x)
        grounded_symlog_out = self.get('grounded_symlog_out', Linear, self._grounded_size)(x)
        return cast(grounded_symlog_out)


class GroundedRSSMGroundedFromStochOnlyNonSequentialPosteriorMLPPriorLarger(GroundedRSSMGroundedFromStochOnlyNonSequentialPosteriorMLPPrior):
    def img_step(self, prev_state, prev_action):
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

        if self._sim_tf is not None:
            curr_state_symlog_sim_dynamics_pred = self.get_sim_forward_dynamics(
                grounded_symlog_in=prev_state['symlog_grounded'],
                action=prev_action)
            if not self._gradients_through_sim_tf:
                curr_state_symlog_sim_dynamics_pred = sg(curr_state_symlog_sim_dynamics_pred)
            img_inputs = jnp.concatenate([prev_stoch, prev_action, curr_state_symlog_sim_dynamics_pred], -1)
        else:
            curr_state_symlog_sim_dynamics_pred = None
            img_inputs = jnp.concatenate([prev_stoch, prev_action], -1)

        x = self.get('img_in', Linear, **self._kw)(img_inputs)
        x = self.get('img_hidden', Linear, **self._kw)(x)
        x = self.get('img_hidden2', Linear, **self._kw)(x)
        x = self.get('img_out', Linear, **self._kw)(x)
        stats = self._stats('img_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded = self.get_grounded(deter=None, stoch=stoch)

        prior = {'stoch': stoch,
                 'symlog_grounded': symlog_grounded,
                 **stats}
        if curr_state_symlog_sim_dynamics_pred is not None:
            prior['curr_state_symlog_sim_dynamics_pred'] = curr_state_symlog_sim_dynamics_pred
        return cast(prior)






class GroundedRSSMStochOnlyNonSequentialPosteriorMLPPriorLargerWithPriorBelief(GroundedRSSM):



    def initial(self, bs):
        if self._classes:
            state = dict(
                deter=jnp.zeros([bs, self._deter], f32),
                logit=jnp.zeros([bs, self._stoch, self._classes], f32),
                stoch=jnp.zeros([bs, self._stoch_sample_size, self._classes], f32),
                stoch_params=jnp.zeros([bs, self._stoch * self._classes], f32),
                stoch_raw_logits=jnp.zeros([bs, self._stoch * self._classes], f32),
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32)
            )
        else:
            state = dict(
                deter=jnp.zeros([bs, self._deter], f32),
                mean=jnp.zeros([bs, self._stoch], f32),
                std=jnp.ones([bs, self._stoch], f32),
                stoch=jnp.zeros([bs, self._stoch_sample_size], f32),
                stoch_params=jnp.zeros([bs, self._stoch * 2], f32),
                stoch_raw_logits=jnp.zeros([bs, self._stoch * 2], f32),
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32)
            )

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
            orig_logit = x
            mean, std = jnp.split(x, 2, -1)
            std = 2 * jax.nn.sigmoid(std / 2) + 0.1
            return {'mean': mean, 'std': std,
                    'stoch_params': jnp.concatenate([mean, std], axis=-1),
                    'stoch_raw_logits': orig_logit
                    }

    def observe(self, embed, action, is_first, state=None, with_residual=True, with_res_stop_gradients=True):
        swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])
        step = lambda prev, inputs: self.obs_step(prev[0], *inputs, with_residual=with_residual, with_res_stop_gradients=with_res_stop_gradients)
        inputs = swap(action), swap(embed), swap(is_first)
        start = state, state
        post, prior = jaxutils.scan(step, inputs, start, self._unroll)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine(self, action, state=None, with_residual=True, with_res_stop_gradients=False):
        swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
        state = self.initial(action.shape[0]) if state is None else state
        assert isinstance(state, dict), state
        action = swap(action)
        step = partial(self.img_step, with_residual=with_residual, with_res_stop_gradients=with_res_stop_gradients)
        prior = jaxutils.scan(step, action, state, self._unroll)
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_grounded_from_stoch(self, stoch):
        if self._classes:
            # flatten stoch if the state representation is multi-dimensional
            new_shape = stoch.shape[:-2] + (self._stoch * self._classes,)
            stoch = stoch.reshape(new_shape)

        # x = jnp.concatenate([deter, stoch], -1)
        x = stoch
        x = self.get('grounded_in', Linear, **self._kw)(x)
        x = self.get('grounded_hidden', Linear, **self._kw)(x)
        grounded_symlog_out = self.get('grounded_symlog_out', Linear, self._grounded_size)(x)
        return cast(grounded_symlog_out)

    def img_step(self, prev_state, prev_action, with_residual=True, with_res_stop_gradients=False):
        assert self._sim_tf is None
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
        print(f"img stoch shape: {stoch.shape}")

        symlog_grounded = self.get_grounded_from_stoch(stoch)

        prior = {'stoch': stoch,
                 'deter': prev_state['deter'],
                 'symlog_grounded': symlog_grounded,
                 # 'prior_stoch_params': stats['stoch_params'],
                 **stats}

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

    def obs_step(self, prev_state, prev_action, embed, is_first, with_residual=True, with_res_stop_gradients=True):
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

        x = embed
        x = self.get('obs_out', Linear, **self._kw)(x)
        stats = self._stats('obs_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        print(f"obs stoch shape: {stoch.shape}")

        if self._use_posterior_stoch_params_for_first_state:
            stoch_params = jnp.where(jnp.tile(is_first[:, None], (1, stats['stoch_params'].shape[-1])),
                                           stats['stoch_params'], prior['stoch_params'])
            stoch_raw_logits = jnp.where(jnp.tile(is_first[:, None], (1, stats['stoch_raw_logits'].shape[-1])),
                                           stats['stoch_raw_logits'], prior['stoch_raw_logits'])
        elif self._use_posterior_stoch_params_for_all_states:
            stoch_params = stats['stoch_params']
            stoch_raw_logits = stats['stoch_raw_logits']
        else:
            stoch_params = prior['stoch_params']
            stoch_raw_logits = prior['stoch_raw_logits']

        del stats['stoch_params']
        del stats['stoch_raw_logits']

        symlog_grounded = self.get_grounded_from_stoch(stoch)

        post = {'stoch': stoch,
                'deter': prev_state['deter'],
                # 'prior_stoch_params': stoch_params,
                'stoch_params': stoch_params,
                'stoch_raw_logits': stoch_raw_logits,
                'symlog_grounded': symlog_grounded,
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

    @staticmethod
    def grounded_state_loss(post, gt_state, is_real):
        symlog_grounded = post['symlog_grounded']
        distance = (symlog_grounded - jaxutils.symlog(gt_state)) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)
        loss = jnp.where(is_real, 0, loss)
        return loss









class GroundedRSSMGroundedFromStochOnlyNonSequentialPosteriorPostProducesH(GroundedRSSM):

    def obs_step(self, prev_state, prev_action, embed, is_first):
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

        x = jnp.concatenate([prev_action, embed], -1) # TODO should there be a stop_gradient on prev_action for the posterior (ok for prior)? - probably dont need an sg since it originaly still went through prior
        print(f"posterior is conditioned on embedding and previous action")
        x = self.get('obs_out', Linear, **self._kw)(x)
        stats = self._stats('obs_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        obs_deter = jnp.tanh(self.get('obs_deter_out', Linear, self._deter)(x))

        symlog_grounded = self.get_grounded(deter=obs_deter, stoch=stoch)

        post = {'stoch': stoch,
                'deter': obs_deter,
                'symlog_grounded': symlog_grounded,
                **stats}
        if self._sim_tf is not None:
            post['curr_state_symlog_sim_dynamics_pred'] = prior['curr_state_symlog_sim_dynamics_pred']
        return cast(post), cast(prior)

    def dyn_loss(self, post, prior, is_first, impl='kl', free=1.0):
        kl_loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
        if free:
            kl_loss = jnp.maximum(kl_loss, free)

        # TODO NOTE that the sg is switched here
        distance = (post['deter'] - sg(prior['deter'])) ** 2  # MSE loss
        distance = jnp.maximum(distance, 1e-4)
        deter_loss = distance.sum(-1)

        loss = kl_loss + deter_loss

        return loss

    def rep_loss(self, post, prior, is_first, impl='kl', free=1.0):
        kl_loss = self.get_dist(post).kl_divergence(self.get_dist(sg(prior)))
        if free:
            kl_loss = jnp.maximum(kl_loss, free)

        # TODO NOTE that the sg is switched here
        distance = (sg(post['deter']) - prior['deter']) ** 2  # MSE loss
        distance = jnp.maximum(distance, 1e-4)
        deter_loss = distance.sum(-1)

        loss = kl_loss + deter_loss

        return loss



class GroundedRSSMGroundedFromStochOnlyNonSequentialPosteriorPostProducesH2(GroundedRSSM):


    def obs_step(self, prev_state, prev_action, embed, is_first):
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

        x = jnp.concatenate([prev_action, embed], -1) # TODO should there be a stop_gradient on prev_action for the posterior (ok for prior)? - probably dont need an sg since it originaly still went through prior
        print(f"posterior is conditioned on embedding and previous action")
        x = self.get('obs_out', Linear, **self._kw)(x)
        stats = self._stats('obs_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        obs_deter_logits = self.get('obs_deter_out', Linear, self._deter)(x)
        obs_deter = jnp.tanh(obs_deter_logits + 0.05 * jax.random.normal(key=nj.rng(), shape=obs_deter_logits.shape))

        symlog_grounded = self.get_grounded(deter=obs_deter, stoch=stoch)

        post = {'stoch': stoch,
                'deter': obs_deter,
                'symlog_grounded': symlog_grounded,
                **stats}
        if self._sim_tf is not None:
            post['curr_state_symlog_sim_dynamics_pred'] = prior['curr_state_symlog_sim_dynamics_pred']
        return cast(post), cast(prior)

    def dyn_loss(self, post, prior, is_first, impl='kl', free=1.0):
        kl_loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
        if free:
            kl_loss = jnp.maximum(kl_loss, free)


        distance = (sg(post['deter']) - prior['deter']) ** 2  # MSE loss
        distance = jnp.maximum(distance, 1e-4)
        deter_loss = distance.sum(-1)

        loss = kl_loss + 0.1 * deter_loss

        return loss

    def rep_loss(self, post, prior, is_first, impl='kl', free=1.0):
        kl_loss = self.get_dist(post).kl_divergence(self.get_dist(sg(prior)))
        if free:
            kl_loss = jnp.maximum(kl_loss, free)

        distance = (post['deter'] - sg(prior['deter'])) ** 2  # MSE loss
        distance = jnp.maximum(distance, 1e-4)
        deter_loss = distance.sum(-1)

        loss = kl_loss + deter_loss

        return loss

class GroundedRSSMGroundedFromStochOnlyOrigSizeNetwork(GroundedRSSM):

    def get_grounded(self, deter, stoch):
        if self._classes:
            # flatten stoch if the state representation is multi-dimensional
            new_shape = stoch.shape[:-2] + (self._stoch * self._classes,)
            stoch = stoch.reshape(new_shape)

        x = stoch
        x = self.get('grounded_in', Linear, **self._kw)(x)
        grounded_symlog_out = self.get('grounded_symlog_out', Linear, self._grounded_size)(x)
        return cast(grounded_symlog_out)

class GroundedNonRecurrentSSM2(nj.Module):

    def __init__(
            self, stoch=32, classes=32, grounded=32,
            sim_tf=None, gradients_through_sim_tf=False,
            unroll=False, initial='learned', unimix=0.01, action_clip=1.0, supervise_encoder_state=True,
            **kw):
        if sim_tf is None:
            raise ValueError(f"sim_tf must be provided for this class.")
        self._stoch = stoch
        self._grounded_size = grounded
        self._classes = classes
        self._sim_tf: Optional[MLP] = sim_tf
        self._gradients_through_sim_tf = gradients_through_sim_tf
        self._unroll = unroll
        self._initial = initial
        self._unimix = unimix
        self._action_clip = action_clip
        self._supervise_encoder_state = supervise_encoder_state
        self._kw = kw

    def initial(self, bs):
        if self._classes:
            state = dict(
                logit=jnp.zeros([bs, self._stoch, self._classes], f32),
                stoch=jnp.zeros([bs, self._stoch, self._classes], f32),
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))
        else:
            state = dict(
                mean=jnp.zeros([bs, self._stoch], f32),
                std=jnp.ones([bs, self._stoch], f32),
                stoch=jnp.zeros([bs, self._stoch], f32),
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))

        if self._sim_tf:
            state['curr_state_symlog_sim_dynamics_pred'] = jnp.zeros([bs, self._grounded_size], f32)
        state['deter_symlog_grounded'] = state['symlog_grounded']

        if self._initial == 'zeros':
            return cast(state)
        elif self._initial == 'learned':
            stoch = self.get('initial_stoch', jnp.zeros, state['stoch'][0].shape, f32)
            state['stoch'] = jnp.repeat(jnp.tanh(stoch)[None], bs, 0)
            state['symlog_grounded'] = self.get_grounded_from_stoch_only(cast(state['stoch']))
            state['deter_symlog_grounded'] = state['symlog_grounded']
            return cast(state)
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
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

    def imagine(self, action, state=None):
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

    def obs_step(self, prev_state, prev_action, embed, is_first):
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

        encoder_obs_symlog_grounded = self.get('obs_grounded_symlog_out', Linear, self._grounded_size)(embed)

        x = self.get('s_to_z_in', Linear, **self._kw)(encoder_obs_symlog_grounded)
        x = self.get('s_to_z_out', Linear, **self._kw)(x)
        stats = self._stats('s_to_z_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded = self.get_grounded_from_stoch_only(stoch=stoch)

        post = {'stoch': stoch,
                'symlog_grounded': symlog_grounded,
                # 'encoder_obs_symlog_grounded': encoder_obs_symlog_grounded,
                'deter_symlog_grounded': encoder_obs_symlog_grounded,
                **stats}
        if self._sim_tf is not None:
            post['curr_state_symlog_sim_dynamics_pred'] = prior['curr_state_symlog_sim_dynamics_pred']
        return cast(post), cast(prior)

    def img_step(self, prev_state, prev_action):
        prev_grounded = prev_state['symlog_grounded']
        prev_action = cast(prev_action)
        if self._action_clip > 0.0:
            prev_action *= sg(self._action_clip / jnp.maximum(
                self._action_clip, jnp.abs(prev_action)))
        if len(prev_action.shape) > len(prev_grounded.shape):  # 2D actions.
            shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
            prev_action = prev_action.reshape(shape)

        prev_grounded = prev_state['symlog_grounded']

        curr_state_symlog_sim_dynamics_pred = self.get_sim_forward_dynamics(
            grounded_symlog_in=prev_grounded,
            action=prev_action)
        if not self._gradients_through_sim_tf:
            curr_state_symlog_sim_dynamics_pred = sg(curr_state_symlog_sim_dynamics_pred)

        x = self.get('s_to_z_in', Linear, **self._kw)(curr_state_symlog_sim_dynamics_pred)
        x = self.get('s_to_z_out', Linear, **self._kw)(x)
        stats = self._stats('s_to_z_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded = self.get_grounded_from_stoch_only(stoch=stoch)

        prior = {'stoch': stoch,
                 'symlog_grounded': symlog_grounded,
                 'deter_symlog_grounded': curr_state_symlog_sim_dynamics_pred,
                 'curr_state_symlog_sim_dynamics_pred': curr_state_symlog_sim_dynamics_pred,
                 **stats}
        return cast(prior)

    def get_stoch(self, deter):
        raise NotImplementedError

    def get_grounded_from_stoch_only(self, stoch):
        if self._classes:
            # flatten stoch if the state representation is multi-dimensional
            new_shape = stoch.shape[:-2] + (self._stoch * self._classes,)
            stoch = stoch.reshape(new_shape)

        x = self.get('grounded_in', Linear, **self._kw)(stoch)

        grounded_symlog_out = self.get('grounded_symlog_out', Linear, self._grounded_size)(x)
        return cast(grounded_symlog_out)

    def get_sim_forward_dynamics(self, grounded_symlog_in, action):
        action = cast(action)
        if self._action_clip > 0.0:
            action *= sg(self._action_clip / jnp.maximum(self._action_clip, jnp.abs(action)))
        x = jnp.concatenate([grounded_symlog_in, action], -1)
        next_grounded_symlog = self._sim_tf(x).mode()
        return cast(next_grounded_symlog)

    def _gru(self, x, deter):
        raise NotImplementedError

    def _stats(self, name, x):
        if self._classes:
            x = self.get(name, Linear, self._stoch * self._classes)(x)
            logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
            if self._unimix:
                probs = jax.nn.softmax(logit, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - self._unimix) * probs + self._unimix * uniform
                logit = jnp.log(probs)
            stats = {'logit': logit}
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
            # loss = jnp.zeros(post['deter'].shape[:-1])
            raise NotImplementedError(impl)
        else:
            raise NotImplementedError(impl)
        if free:
            loss = jnp.maximum(loss, free)
        return loss

    def grounded_state_loss(self, post, gt_state):
        symlog_grounded = post['symlog_grounded']
        distance = (symlog_grounded - jaxutils.symlog(gt_state)) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)

        if self._supervise_encoder_state:
            symlog_grounded = post['deter_symlog_grounded']
            distance = (symlog_grounded - jaxutils.symlog(gt_state)) ** 2  # MSE loss
            distance = jnp.where(distance < 1e-8, 0, distance)
            additional_loss = distance.sum(-1)
            loss = loss + additional_loss

        return loss

    @staticmethod
    def simulation_tf_loss(symlog_grounded_next_state_pred, target_symlog_next_state, is_last):
        distance = (symlog_grounded_next_state_pred - target_symlog_next_state) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)  # sum over state dimensions, shape is still (batch x sequence len)
        # Only calculate loss for next state prediction when the current state is not the last state in a sequence.
        loss = jnp.where(is_last, 0, loss)
        return loss



class GroundedNonRecurrentSSM3(nj.Module):

    def __init__(
            self, stoch=32, classes=32, grounded=32,
            sim_tf=None, gradients_through_sim_tf=False,
            unroll=False, initial='learned', unimix=0.01, action_clip=1.0, supervise_encoder_state=True, img_is_deterministic=False, sg_in_img=False,
            **kw):
        if sim_tf is None:
            raise ValueError(f"sim_tf must be provided for this class.")
        self._stoch = stoch
        self._grounded_size = grounded
        self._classes = classes
        self._sim_tf: Optional[MLP] = sim_tf
        self._gradients_through_sim_tf = gradients_through_sim_tf
        self._unroll = unroll
        self._initial = initial
        self._unimix = unimix
        self._action_clip = action_clip
        self._supervise_encoder_state = supervise_encoder_state
        self._kw = kw

    def initial(self, bs):
        if self._classes:
            state = dict(
                logit=jnp.zeros([bs, self._stoch, self._classes], f32),
                stoch=jnp.zeros([bs, self._stoch, self._classes], f32),
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))
        else:
            state = dict(
                mean=jnp.zeros([bs, self._stoch], f32),
                std=jnp.ones([bs, self._stoch], f32),
                stoch=jnp.zeros([bs, self._stoch], f32),
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))

        if self._sim_tf:
            state['curr_state_symlog_sim_dynamics_pred'] = jnp.zeros([bs, self._grounded_size], f32)
        state['symlog_grounded_from_z'] = state['symlog_grounded']

        if self._initial == 'zeros':
            return cast(state)
        elif self._initial == 'learned':
            stoch = self.get('initial_stoch', jnp.zeros, state['stoch'][0].shape, f32)
            state['stoch'] = jnp.repeat(jnp.tanh(stoch)[None], bs, 0)
            state['symlog_grounded_from_z'] = self.get_grounded_from_stoch_only(cast(state['stoch']))
            state['symlog_grounded'] = state['symlog_grounded_from_z']
            return cast(state)
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
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

    def imagine(self, action, state=None):
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

    def obs_step(self, prev_state, prev_action, embed, is_first):
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

        encoder_obs_symlog_grounded = self.get('obs_grounded_symlog_out', Linear, self._grounded_size)(embed)

        x = self.get('s_to_z_in', Linear, **self._kw)(encoder_obs_symlog_grounded)
        x = self.get('s_to_z_out', Linear, **self._kw)(x)
        stats = self._stats('s_to_z_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded_from_z = self.get_grounded_from_stoch_only(stoch=stoch)

        post = {'stoch': stoch,
                'symlog_grounded_from_z': symlog_grounded_from_z,
                # 'encoder_obs_symlog_grounded': encoder_obs_symlog_grounded,
                'symlog_grounded': encoder_obs_symlog_grounded,
                **stats}
        if self._sim_tf is not None:
            post['curr_state_symlog_sim_dynamics_pred'] = prior['curr_state_symlog_sim_dynamics_pred']
        return cast(post), cast(prior)

    def img_step(self, prev_state, prev_action):
        prev_grounded = prev_state['symlog_grounded']
        prev_action = cast(prev_action)
        if self._action_clip > 0.0:
            prev_action *= sg(self._action_clip / jnp.maximum(
                self._action_clip, jnp.abs(prev_action)))
        if len(prev_action.shape) > len(prev_grounded.shape):  # 2D actions.
            shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
            prev_action = prev_action.reshape(shape)

        prev_grounded = prev_state['symlog_grounded']

        curr_state_symlog_sim_dynamics_pred = self.get_sim_forward_dynamics(
            grounded_symlog_in=prev_grounded,
            action=prev_action)
        if not self._gradients_through_sim_tf:
            curr_state_symlog_sim_dynamics_pred = sg(curr_state_symlog_sim_dynamics_pred)

        x = self.get('s_to_z_in', Linear, **self._kw)(curr_state_symlog_sim_dynamics_pred)
        x = self.get('s_to_z_out', Linear, **self._kw)(x)
        stats = self._stats('s_to_z_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded_from_z = self.get_grounded_from_stoch_only(stoch=stoch)

        prior = {'stoch': stoch,
                 'symlog_grounded_from_z': symlog_grounded_from_z,
                 'symlog_grounded': curr_state_symlog_sim_dynamics_pred,
                 'curr_state_symlog_sim_dynamics_pred': curr_state_symlog_sim_dynamics_pred,
                 **stats}

        #TODO make sure whole stack from encoder to s prediction is optimize in train alt

        return cast(prior)

    def get_stoch(self, deter):
        raise NotImplementedError

    def get_grounded_from_stoch_only(self, stoch):
        if self._classes:
            # flatten stoch if the state representation is multi-dimensional
            new_shape = stoch.shape[:-2] + (self._stoch * self._classes,)
            stoch = stoch.reshape(new_shape)

        x = self.get('grounded_in', Linear, **self._kw)(stoch)

        grounded_symlog_out = self.get('grounded_symlog_out', Linear, self._grounded_size)(x)
        return cast(grounded_symlog_out)

    def get_sim_forward_dynamics(self, grounded_symlog_in, action):
        action = cast(action)
        if self._action_clip > 0.0:
            action *= sg(self._action_clip / jnp.maximum(self._action_clip, jnp.abs(action)))
        x = jnp.concatenate([grounded_symlog_in, action], -1)
        next_grounded_symlog = self._sim_tf(x).mode()
        return cast(next_grounded_symlog)

    def _gru(self, x, deter):
        raise NotImplementedError

    def _stats(self, name, x):
        if self._classes:
            x = self.get(name, Linear, self._stoch * self._classes)(x)
            logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
            if self._unimix:
                probs = jax.nn.softmax(logit, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - self._unimix) * probs + self._unimix * uniform
                logit = jnp.log(probs)
            stats = {'logit': logit}
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
        elif impl == 'none':
            loss = jnp.zeros(post['symlog_grounded'].shape[:-1])
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
            loss = jnp.zeros(post['symlog_grounded'].shape[:-1])
        else:
            raise NotImplementedError(impl)
        if free:
            loss = jnp.maximum(loss, free)
        return loss

    def grounded_state_loss(self, post, gt_state):
        symlog_grounded = post['symlog_grounded']
        distance = (symlog_grounded - jaxutils.symlog(gt_state)) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)

        if self._supervise_encoder_state:
            symlog_grounded = post['symlog_grounded_from_z']
            distance = (symlog_grounded - jaxutils.symlog(gt_state)) ** 2  # MSE loss
            distance = jnp.where(distance < 1e-8, 0, distance)
            additional_loss = distance.sum(-1)
            loss = loss + additional_loss

        return loss

    @staticmethod
    def simulation_tf_loss(symlog_grounded_next_state_pred, target_symlog_next_state, is_last):
        distance = (symlog_grounded_next_state_pred - target_symlog_next_state) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)  # sum over state dimensions, shape is still (batch x sequence len)
        # Only calculate loss for next state prediction when the current state is not the last state in a sequence.
        loss = jnp.where(is_last, 0, loss)
        return loss


class GroundedNonRecurrentSSM4(nj.Module):

    def __init__(
            self, stoch=32, classes=32, grounded=32,
            sim_tf=None, gradients_through_sim_tf=False,
            unroll=False, initial='learned', unimix=0.01, action_clip=1.0, img_is_deterministic=False,
            **kw):
        if sim_tf is None:
            raise ValueError(f"sim_tf must be provided for this class.")
        self._stoch = stoch
        self._grounded_size = grounded
        self._classes = classes
        self._sim_tf: Optional[MLP] = sim_tf
        self._gradients_through_sim_tf = gradients_through_sim_tf
        self._unroll = unroll
        self._initial = initial
        self._unimix = unimix
        self._action_clip = action_clip
        self._img_is_deterministic = img_is_deterministic
        self._kw = kw

    def initial(self, bs):
        if self._classes:
            state = dict(
                logit=jnp.zeros([bs, self._stoch, self._classes], f32),
                stoch=jnp.zeros([bs, self._stoch, self._classes], f32),
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))
        else:
            state = dict(
                mean=jnp.zeros([bs, self._stoch], f32),
                std=jnp.ones([bs, self._stoch], f32),
                stoch=jnp.zeros([bs, self._stoch], f32),
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))

        if self._sim_tf:
            state['curr_state_symlog_sim_dynamics_pred'] = jnp.zeros([bs, self._grounded_size], f32)
        state['symlog_grounded_deter'] = state['symlog_grounded']

        if self._initial == 'zeros':
            return cast(state)
        elif self._initial == 'learned':
            stoch = self.get('initial_stoch', jnp.zeros, state['stoch'][0].shape, f32)
            state['stoch'] = jnp.repeat(jnp.tanh(stoch)[None], bs, 0)
            state['symlog_grounded_deter'] = self.get_grounded_from_stoch_only(cast(state['stoch']))
            state['symlog_grounded'] = state['symlog_grounded_deter']
            return cast(state)
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
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

    def imagine(self, action, state=None):
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

    def obs_step(self, prev_state, prev_action, embed, is_first):
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

        encoder_obs_symlog_grounded = embed

        x = self.get('s_to_z_in', Linear, **self._kw)(encoder_obs_symlog_grounded)
        x = self.get('s_to_z_out', Linear, **self._kw)(x)
        stats = self._stats('s_to_z_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded_from_z = self.get_grounded_from_stoch_only(stoch=stoch)

        post = {'stoch': stoch,
                'symlog_grounded': symlog_grounded_from_z,
                # 'encoder_obs_symlog_grounded': encoder_obs_symlog_grounded,
                'symlog_grounded_deter': encoder_obs_symlog_grounded,
                **stats}
        if self._sim_tf is not None:
            post['curr_state_symlog_sim_dynamics_pred'] = prior['curr_state_symlog_sim_dynamics_pred']
        return cast(post), cast(prior)

    def img_step(self, prev_state, prev_action):
        prev_grounded = prev_state['symlog_grounded']
        prev_action = cast(prev_action)
        if self._action_clip > 0.0:
            prev_action *= sg(self._action_clip / jnp.maximum(
                self._action_clip, jnp.abs(prev_action)))
        if len(prev_action.shape) > len(prev_grounded.shape):  # 2D actions.
            shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
            prev_action = prev_action.reshape(shape)

        prev_grounded = prev_state['symlog_grounded_deter'] if self._img_is_deterministic else prev_state['symlog_grounded']

        curr_state_symlog_sim_dynamics_pred = self.get_sim_forward_dynamics(
            grounded_symlog_in=prev_grounded,
            action=prev_action)
        if not self._gradients_through_sim_tf:
            curr_state_symlog_sim_dynamics_pred = sg(curr_state_symlog_sim_dynamics_pred)

        x = self.get('s_to_z_in', Linear, **self._kw)(curr_state_symlog_sim_dynamics_pred)
        x = self.get('s_to_z_out', Linear, **self._kw)(x)
        stats = self._stats('s_to_z_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded_from_z = self.get_grounded_from_stoch_only(stoch=stoch)

        prior = {'stoch': stoch,
                 'symlog_grounded': symlog_grounded_from_z,
                 'symlog_grounded_deter': curr_state_symlog_sim_dynamics_pred,
                 'curr_state_symlog_sim_dynamics_pred': curr_state_symlog_sim_dynamics_pred,
                 **stats}
        return cast(prior)

    def get_stoch(self, deter):
        raise NotImplementedError

    def get_grounded_from_stoch_only(self, stoch):
        if self._classes:
            # flatten stoch if the state representation is multi-dimensional
            new_shape = stoch.shape[:-2] + (self._stoch * self._classes,)
            stoch = stoch.reshape(new_shape)

        x = self.get('z_to_s_in', Linear, **self._kw)(stoch)

        grounded_symlog_out = self.get('z_to_s_out', Linear, self._grounded_size)(x)
        return cast(grounded_symlog_out)

    def get_sim_forward_dynamics(self, grounded_symlog_in, action):
        action = cast(action)
        if self._action_clip > 0.0:
            action *= sg(self._action_clip / jnp.maximum(self._action_clip, jnp.abs(action)))
        x = jnp.concatenate([grounded_symlog_in, action], -1)
        next_grounded_symlog = self._sim_tf(x).mode()
        return cast(next_grounded_symlog)

    def _gru(self, x, deter):
        raise NotImplementedError

    def _stats(self, name, x):
        if self._classes:
            x = self.get(name, Linear, self._stoch * self._classes)(x)
            logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
            if self._unimix:
                probs = jax.nn.softmax(logit, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - self._unimix) * probs + self._unimix * uniform
                logit = jnp.log(probs)
            stats = {'logit': logit}
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
        elif impl == 'none':
            loss = jnp.zeros(post['symlog_grounded'].shape[:-1])
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
            loss = jnp.zeros(post['symlog_grounded'].shape[:-1])
        else:
            raise NotImplementedError(impl)
        if free:
            loss = jnp.maximum(loss, free)

        return loss

    def grounded_state_loss(self, post, gt_state):
        symlog_grounded = post['symlog_grounded']
        distance = (symlog_grounded - jaxutils.symlog(gt_state)) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)

        return loss

    @staticmethod
    def simulation_tf_loss(symlog_grounded_next_state_pred, target_symlog_next_state, is_last):
        distance = (symlog_grounded_next_state_pred - target_symlog_next_state) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)  # sum over state dimensions, shape is still (batch x sequence len)
        # Only calculate loss for next state prediction when the current state is not the last state in a sequence.
        loss = jnp.where(is_last, 0, loss)
        return loss


class GroundedLiteralSSM(nj.Module):

    def __init__(
            self, grounded=32,
            sim_tf=None, gradients_through_sim_tf=False,
            unroll=False, initial='learned', action_clip=1.0, **kw):
        self._grounded_size = grounded
        self._sim_tf: Optional[MLP] = sim_tf
        self._gradients_through_sim_tf = gradients_through_sim_tf
        self._unroll = unroll
        self._initial = initial
        self._action_clip = action_clip
        self._kw = kw

    def initial(self, bs):
        state = dict(symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))
        if self._sim_tf:
            state['curr_state_symlog_sim_dynamics_pred'] = jnp.zeros([bs, self._grounded_size], f32)

        if self._initial == 'zeros':
            return cast(state)
        elif self._initial == 'learned':
            symlog_grounded = self.get('initial_symlog_grounded', jnp.zeros, state['symlog_grounded'][0].shape, f32)
            state['symlog_grounded'] = jnp.repeat(jnp.tanh(symlog_grounded)[None], bs, 0)
            return cast(state)
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
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

    def imagine(self, action, state=None):
        swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
        state = self.initial(action.shape[0]) if state is None else state
        assert isinstance(state, dict), state
        action = swap(action)
        prior = jaxutils.scan(self.img_step, action, state, self._unroll)
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_dist(self, state, argmax=False):
        return tfd.Deterministic(state['symlog_grounded'].astype(f32))

    def obs_step(self, prev_state, prev_action, embed, is_first):
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

        x = self.get('obs_out', Linear, **self._kw)(embed)
        symlog_grounded = self.get('obs_grounded_symlog_out', Linear, self._grounded_size)(x)

        post = {'symlog_grounded': symlog_grounded}
        if self._sim_tf is not None:
            post['curr_state_symlog_sim_dynamics_pred'] = prior['curr_state_symlog_sim_dynamics_pred']
        return cast(post), cast(prior)

    def img_step(self, prev_state, prev_action):
        prev_grounded = prev_state['symlog_grounded']
        prev_action = cast(prev_action)
        if self._action_clip > 0.0:
            prev_action *= sg(self._action_clip / jnp.maximum(
                self._action_clip, jnp.abs(prev_action)))
        if len(prev_action.shape) > len(prev_grounded.shape):  # 2D actions.
            shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
            prev_action = prev_action.reshape(shape)

        if self._sim_tf is not None:
            curr_state_symlog_sim_dynamics_pred = self.get_sim_forward_dynamics(
                grounded_symlog_in=prev_grounded,
                action=prev_action)
            if not self._gradients_through_sim_tf:
                curr_state_symlog_sim_dynamics_pred = sg(curr_state_symlog_sim_dynamics_pred)
            img_inputs = jnp.concatenate([prev_grounded, prev_action, curr_state_symlog_sim_dynamics_pred], -1)
        else:
            curr_state_symlog_sim_dynamics_pred = None
            img_inputs = jnp.concatenate([prev_grounded, prev_action], -1)

        x = self.get('img_in', Linear, **self._kw)(img_inputs)
        x = self.get('img_hidden', Linear, **self._kw)(x)
        x = self.get('img_out', Linear, **self._kw)(x)
        symlog_grounded = self.get('img_grounded_symlog_out', Linear, self._grounded_size)(x)

        prior = {'symlog_grounded': symlog_grounded}
        if curr_state_symlog_sim_dynamics_pred is not None:
            prior['curr_state_symlog_sim_dynamics_pred'] = curr_state_symlog_sim_dynamics_pred
        return cast(prior)

    def get_sim_forward_dynamics(self, grounded_symlog_in, action):
        action = cast(action)
        if self._action_clip > 0.0:
            action *= sg(self._action_clip / jnp.maximum(self._action_clip, jnp.abs(action)))
        x = jnp.concatenate([grounded_symlog_in, action], -1)
        next_grounded_symlog = self._sim_tf(x).mode()
        return cast(next_grounded_symlog)

    def _mask(self, value, mask):
        return jnp.einsum('b...,b->b...', value, mask.astype(value.dtype))

    def dyn_loss(self, post, prior, impl='mse', free=1.0):
        if impl == 'mse':
            distance = (sg(post['symlog_grounded']) - prior['symlog_grounded']) ** 2  # MSE loss
            distance = jnp.where(distance < 1e-8, 0, distance)
            loss = distance.sum(-1)  # sum over state dimensions, shape is still (batch x sequence len)
        else:
            raise NotImplementedError(impl)
        return loss

    def rep_loss(self, post, prior, impl='none', free=1.0):
        if impl == 'none':
            loss = jnp.zeros(post['symlog_grounded'].shape[:-1])
        else:
            raise NotImplementedError(impl)
        return loss

    @staticmethod
    def grounded_state_loss(post, gt_state):
        symlog_grounded = post['symlog_grounded']
        distance = (symlog_grounded - jaxutils.symlog(gt_state)) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)
        return loss

    @staticmethod
    def simulation_tf_loss(symlog_grounded_next_state_pred, target_symlog_next_state, is_last):
        distance = (symlog_grounded_next_state_pred - target_symlog_next_state) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)  # sum over state dimensions, shape is still (batch x sequence len)
        # Only calculate loss for next state prediction when the current state is not the last state in a sequence.
        loss = jnp.where(is_last, 0, loss)
        return loss



class GroundedLiteralSimOnlySSM(nj.Module):

    def __init__(
            self, grounded=32,
            sim_tf=None, gradients_through_sim_tf=False,
            unroll=False, initial='learned', action_clip=1.0, **kw):
        if sim_tf is None:
            raise ValueError(f"sim_tf must be provided for this class.")
        self._grounded_size = grounded
        self._sim_tf: Optional[MLP] = sim_tf
        self._gradients_through_sim_tf = gradients_through_sim_tf
        self._unroll = unroll
        self._initial = initial
        self._action_clip = action_clip
        self._kw = kw

    def initial(self, bs):
        state = dict(symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))
        if self._sim_tf:
            state['curr_state_symlog_sim_dynamics_pred'] = jnp.zeros([bs, self._grounded_size], f32)

        if self._initial == 'zeros':
            return cast(state)
        elif self._initial == 'learned':
            symlog_grounded = self.get('initial_symlog_grounded', jnp.zeros, state['symlog_grounded'][0].shape, f32)
            state['symlog_grounded'] = jnp.repeat(jnp.tanh(symlog_grounded)[None], bs, 0)
            return cast(state)
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
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

    def imagine(self, action, state=None):
        swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
        state = self.initial(action.shape[0]) if state is None else state
        assert isinstance(state, dict), state
        action = swap(action)
        prior = jaxutils.scan(self.img_step, action, state, self._unroll)
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_dist(self, state, argmax=False):
        return tfd.Deterministic(state['symlog_grounded'].astype(f32))

    def obs_step(self, prev_state, prev_action, embed, is_first):
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

        # x = self.get('obs_out', Linear, **self._kw)(embed)
        # symlog_grounded = self.get('obs_grounded_symlog_out', Linear, self._grounded_size)(x)

        symlog_grounded = embed

        post = {'symlog_grounded': symlog_grounded}
        if self._sim_tf is not None:
            post['curr_state_symlog_sim_dynamics_pred'] = prior['curr_state_symlog_sim_dynamics_pred']
        return cast(post), cast(prior)

    def img_step(self, prev_state, prev_action):
        prev_grounded = prev_state['symlog_grounded']
        prev_action = cast(prev_action)
        if self._action_clip > 0.0:
            prev_action *= sg(self._action_clip / jnp.maximum(
                self._action_clip, jnp.abs(prev_action)))
        if len(prev_action.shape) > len(prev_grounded.shape):  # 2D actions.
            shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
            prev_action = prev_action.reshape(shape)

        curr_state_symlog_sim_dynamics_pred = self.get_sim_forward_dynamics(
            grounded_symlog_in=prev_grounded,
            action=prev_action)
        if not self._gradients_through_sim_tf:
            curr_state_symlog_sim_dynamics_pred = sg(curr_state_symlog_sim_dynamics_pred)

        symlog_grounded = curr_state_symlog_sim_dynamics_pred

        prior = {'symlog_grounded': symlog_grounded}
        if curr_state_symlog_sim_dynamics_pred is not None:
            prior['curr_state_symlog_sim_dynamics_pred'] = curr_state_symlog_sim_dynamics_pred
        return cast(prior)

    def get_sim_forward_dynamics(self, grounded_symlog_in, action):
        action = cast(action)
        if self._action_clip > 0.0:
            action *= sg(self._action_clip / jnp.maximum(self._action_clip, jnp.abs(action)))
        x = jnp.concatenate([grounded_symlog_in, action], -1)
        next_grounded_symlog = self._sim_tf(x).mode()
        return cast(next_grounded_symlog)

    def _mask(self, value, mask):
        return jnp.einsum('b...,b->b...', value, mask.astype(value.dtype))

    def dyn_loss(self, post, prior, is_first, impl='mse', free=1.0):
        if impl == 'none':
            loss = jnp.zeros(post['symlog_grounded'].shape[:-1])
        else:
            raise NotImplementedError(impl)
        return loss
        # if impl == 'mse':
        #     distance = (sg(post['symlog_grounded']) - prior['symlog_grounded']) ** 2  # MSE loss
        #     distance = jnp.where(distance < 1e-8, 0, distance)
        #     loss = distance.sum(-1)  # sum over state dimensions, shape is still (batch x sequence len)
        # else:
        #     raise NotImplementedError(impl)
        # return loss

    def rep_loss(self, post, prior, is_first, impl='none', free=1.0):
        if impl == 'none':
            loss = jnp.zeros(post['symlog_grounded'].shape[:-1])
        else:
            raise NotImplementedError(impl)
        return loss

    @staticmethod
    def grounded_state_loss(post, gt_state):
        symlog_grounded = post['symlog_grounded']
        distance = (symlog_grounded - jaxutils.symlog(gt_state)) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)
        return loss

    @staticmethod
    def simulation_tf_loss(symlog_grounded_next_state_pred, target_symlog_next_state, is_last):
        distance = (symlog_grounded_next_state_pred - target_symlog_next_state) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)  # sum over state dimensions, shape is still (batch x sequence len)
        # Only calculate loss for next state prediction when the current state is not the last state in a sequence.
        loss = jnp.where(is_last, 0, loss)
        return loss



class GroundedNonRecurrentSSM(nj.Module):

    def __init__(
            self, stoch=32, classes=32, grounded=32,
            sim_tf=None, gradients_through_sim_tf=False,
            unroll=False, initial='learned', unimix=0.01, action_clip=1.0, img_is_deterministic=False, sg_in_img=False, **kw):
        self._stoch = stoch
        self._grounded_size = grounded
        self._classes = classes
        self._sim_tf: Optional[MLP] = sim_tf
        self._gradients_through_sim_tf = gradients_through_sim_tf
        self._unroll = unroll
        self._initial = initial
        self._unimix = unimix
        self._action_clip = action_clip
        self._kw = kw

    def initial(self, bs):
        if self._classes:
            state = dict(
                logit=jnp.zeros([bs, self._stoch, self._classes], f32),
                stoch=jnp.zeros([bs, self._stoch, self._classes], f32),
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))
        else:
            state = dict(
                mean=jnp.zeros([bs, self._stoch], f32),
                std=jnp.ones([bs, self._stoch], f32),
                stoch=jnp.zeros([bs, self._stoch], f32),
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))

        if self._sim_tf:
            state['curr_state_symlog_sim_dynamics_pred'] = jnp.zeros([bs, self._grounded_size], f32)

        if self._initial == 'zeros':
            return cast(state)
        elif self._initial == 'learned':
            stoch = self.get('initial_stoch', jnp.zeros, state['stoch'][0].shape, f32)
            state['stoch'] = jnp.repeat(jnp.tanh(stoch)[None], bs, 0)
            state['symlog_grounded'] = self.get_grounded_from_stoch_only(cast(state['stoch']))
            return cast(state)
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
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

    def imagine(self, action, state=None):
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

    def obs_step(self, prev_state, prev_action, embed, is_first):
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
        x = self.get('obs_out', Linear, **self._kw)(embed)
        stats = self._stats('obs_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded = self.get_grounded_from_stoch_only(stoch=stoch)

        post = {'stoch': stoch,
                'symlog_grounded': symlog_grounded,
                **stats}
        if self._sim_tf is not None:
            post['curr_state_symlog_sim_dynamics_pred'] = prior['curr_state_symlog_sim_dynamics_pred']
        return cast(post), cast(prior)

    def img_step(self, prev_state, prev_action):
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

        if self._sim_tf is not None:
            curr_state_symlog_sim_dynamics_pred = self.get_sim_forward_dynamics(
                grounded_symlog_in=prev_state['symlog_grounded'],
                action=prev_action)
            if not self._gradients_through_sim_tf:
                curr_state_symlog_sim_dynamics_pred = sg(curr_state_symlog_sim_dynamics_pred)
            img_inputs = jnp.concatenate([prev_stoch, prev_action, curr_state_symlog_sim_dynamics_pred], -1)
        else:
            curr_state_symlog_sim_dynamics_pred = None
            img_inputs = jnp.concatenate([prev_stoch, prev_action], -1)

        x = self.get('img_in', Linear, **self._kw)(img_inputs)
        x = self.get('img_hidden', Linear, **self._kw)(x)
        x = self.get('img_out', Linear, **self._kw)(x)

        stats = self._stats('img_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded = self.get_grounded_from_stoch_only(stoch=stoch)

        prior = {'stoch': stoch,
                 'symlog_grounded': symlog_grounded,
                 **stats}
        if curr_state_symlog_sim_dynamics_pred is not None:
            prior['curr_state_symlog_sim_dynamics_pred'] = curr_state_symlog_sim_dynamics_pred
        return cast(prior)

    def get_stoch(self, deter):
        raise NotImplementedError

    def get_grounded_from_stoch_only(self, stoch):
        if self._classes:
            # flatten stoch if the state representation is multi-dimensional
            new_shape = stoch.shape[:-2] + (self._stoch * self._classes,)
            stoch = stoch.reshape(new_shape)

        x = self.get('grounded_in', Linear, **self._kw)(stoch)

        grounded_symlog_out = self.get('grounded_symlog_out', Linear, self._grounded_size)(x)
        return cast(grounded_symlog_out)

    def get_sim_forward_dynamics(self, grounded_symlog_in, action):
        action = cast(action)
        if self._action_clip > 0.0:
            action *= sg(self._action_clip / jnp.maximum(self._action_clip, jnp.abs(action)))
        x = jnp.concatenate([grounded_symlog_in, action], -1)
        next_grounded_symlog = self._sim_tf(x).mode()
        return cast(next_grounded_symlog)

    def _gru(self, x, deter):
        raise NotImplementedError

    def _stats(self, name, x):
        if self._classes:
            x = self.get(name, Linear, self._stoch * self._classes)(x)
            logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
            if self._unimix:
                probs = jax.nn.softmax(logit, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - self._unimix) * probs + self._unimix * uniform
                logit = jnp.log(probs)
            stats = {'logit': logit}
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
            # loss = jnp.zeros(post['deter'].shape[:-1])
            raise NotImplementedError(impl)
        else:
            raise NotImplementedError(impl)
        if free:
            loss = jnp.maximum(loss, free)
        return loss

    @staticmethod
    def grounded_state_loss(post, gt_state):
        symlog_grounded = post['symlog_grounded']
        distance = (symlog_grounded - jaxutils.symlog(gt_state)) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)
        return loss

    @staticmethod
    def simulation_tf_loss(symlog_grounded_next_state_pred, target_symlog_next_state, is_last):
        distance = (symlog_grounded_next_state_pred - target_symlog_next_state) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)  # sum over state dimensions, shape is still (batch x sequence len)
        # Only calculate loss for next state prediction when the current state is not the last state in a sequence.
        loss = jnp.where(is_last, 0, loss)
        return loss





class GroundedRSSMReplaceHwithS(nj.Module):

    def __init__(
            self, stoch=32, classes=32, grounded=32,
            sim_tf=None, gradients_through_sim_tf=False,
            unroll=False, initial='learned', unimix=0.01, action_clip=1.0, img_is_deterministic=False, sg_in_img=False, **kw):
        assert sim_tf is not None
        self._stoch = stoch
        self._grounded_size = grounded
        self._classes = classes
        self._sim_tf: Optional[MLP] = sim_tf
        self._gradients_through_sim_tf = gradients_through_sim_tf
        self._unroll = unroll
        self._initial = initial
        self._unimix = unimix
        self._action_clip = action_clip
        self._kw = kw

    def initial(self, bs):
        if self._classes:
            state = dict(
                deter_symlog_grounded=jnp.zeros([bs, self._grounded_size], f32),
                logit=jnp.zeros([bs, self._stoch, self._classes], f32),
                stoch=jnp.zeros([bs, self._stoch, self._classes], f32),
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))
        else:
            state = dict(
                deter_symlog_grounded=jnp.zeros([bs, self._grounded_size], f32),
                mean=jnp.zeros([bs, self._stoch], f32),
                std=jnp.ones([bs, self._stoch], f32),
                stoch=jnp.zeros([bs, self._stoch], f32),
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))

        if self._sim_tf:
            state['curr_state_symlog_sim_dynamics_pred'] = jnp.zeros([bs, self._grounded_size], f32)

        if self._initial == 'zeros':
            return cast(state)
        elif self._initial == 'learned':
            deter_symlog_grounded = self.get('initial', jnp.zeros, state['deter_symlog_grounded'][0].shape, f32)
            state['deter_symlog_grounded'] = jnp.repeat(jnp.tanh(deter_symlog_grounded)[None], bs, 0)
            state['stoch'] = self.get_stoch(cast(state['deter_symlog_grounded']))
            state['symlog_grounded'] = self.get_grounded_from_deter_grounded_and_stoch(cast(state['deter_symlog_grounded']), cast(state['stoch']))
            return cast(state)
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
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

    def imagine(self, action, state=None):
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

    def obs_step(self, prev_state, prev_action, embed, is_first):
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
        x = jnp.concatenate([prior['deter_symlog_grounded'], embed], -1)
        x = self.get('obs_out', Linear, **self._kw)(x)
        stats = self._stats('obs_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded = self.get_grounded_from_deter_grounded_and_stoch(deter_grounded=prior['deter_symlog_grounded'],
                                                                          stoch=stoch)

        post = {'stoch': stoch,
                'deter_symlog_grounded': prior['deter_symlog_grounded'],
                'symlog_grounded': symlog_grounded,
                **stats}
        if self._sim_tf is not None:
            post['curr_state_symlog_sim_dynamics_pred'] = prior['curr_state_symlog_sim_dynamics_pred']
        return cast(post), cast(prior)

    def img_step(self, prev_state, prev_action):
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

        curr_state_symlog_sim_dynamics_pred = self.get_sim_forward_dynamics(
            grounded_symlog_in=prev_state['symlog_grounded'],
            action=prev_action)
        if not self._gradients_through_sim_tf:
            curr_state_symlog_sim_dynamics_pred = sg(curr_state_symlog_sim_dynamics_pred)
        # img_inputs = jnp.concatenate([prev_stoch, prev_action, curr_state_symlog_sim_dynamics_pred], -1)

        deter_symlog_grounded = curr_state_symlog_sim_dynamics_pred

        x = self.get('img_in', Linear, **self._kw)(deter_symlog_grounded)
        # x = self.get('img_out', Linear, **self._kw)(x)


        # x = self.get('img_in', Linear, **self._kw)(img_inputs)
        # x, deter = self._gru(x, prev_state['deter'])
        # x = self.get('img_out', Linear, **self._kw)(x)
        stats = self._stats('img_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded = self.get_grounded_from_deter_grounded_and_stoch(deter_grounded=deter_symlog_grounded, stoch=stoch)

        prior = {'stoch': stoch,
                 'deter_symlog_grounded': deter_symlog_grounded,
                 'symlog_grounded': symlog_grounded,
                 **stats}
        if curr_state_symlog_sim_dynamics_pred is not None:
            prior['curr_state_symlog_sim_dynamics_pred'] = curr_state_symlog_sim_dynamics_pred
        return cast(prior)

    def get_stoch(self, deter_symlog_grounded):
        x = self.get('img_in', Linear, **self._kw)(deter_symlog_grounded)
        stats = self._stats('img_stats', x)
        dist = self.get_dist(stats)
        return cast(dist.mode())

    def get_grounded_from_deter_grounded_and_stoch(self, deter_grounded, stoch):
        if self._classes:
            # flatten stoch if the state representation is multi-dimensional
            new_shape = stoch.shape[:-2] + (self._stoch * self._classes,)
            stoch = stoch.reshape(new_shape)

        x = jnp.concatenate([deter_grounded, stoch], -1)
        x = self.get('grounded_in', Linear, **self._kw)(x)

        grounded_symlog_out = self.get('grounded_symlog_out', Linear, self._grounded_size)(x)
        return cast(grounded_symlog_out)

    def get_sim_forward_dynamics(self, grounded_symlog_in, action):
        action = cast(action)
        if self._action_clip > 0.0:
            action *= sg(self._action_clip / jnp.maximum(self._action_clip, jnp.abs(action)))
        x = jnp.concatenate([grounded_symlog_in, action], -1)
        next_grounded_symlog = self._sim_tf(x).mode()
        return cast(next_grounded_symlog)

    # def _gru(self, x, deter):
    #     x = jnp.concatenate([deter, x], -1)
    #     kw = {**self._kw, 'act': 'none', 'units': 3 * self._deter}
    #     x = self.get('gru', Linear, **kw)(x)
    #     reset, cand, update = jnp.split(x, 3, -1)
    #     reset = jax.nn.sigmoid(reset)
    #     cand = jnp.tanh(reset * cand)
    #     update = jax.nn.sigmoid(update - 1)
    #     deter = update * cand + (1 - update) * deter
    #     return deter, deter

    def _stats(self, name, x):
        if self._classes:
            x = self.get(name, Linear, self._stoch * self._classes)(x)
            logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
            if self._unimix:
                probs = jax.nn.softmax(logit, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - self._unimix) * probs + self._unimix * uniform
                logit = jnp.log(probs)
            stats = {'logit': logit}
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

    @staticmethod
    def grounded_state_loss(post, gt_state):
        symlog_grounded = post['symlog_grounded']
        distance = (symlog_grounded - jaxutils.symlog(gt_state)) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)
        return loss

    @staticmethod
    def simulation_tf_loss(symlog_grounded_next_state_pred, target_symlog_next_state, is_last):
        distance = (symlog_grounded_next_state_pred - target_symlog_next_state) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)  # sum over state dimensions, shape is still (batch x sequence len)
        # Only calculate loss for next state prediction when the current state is not the last state in a sequence.
        loss = jnp.where(is_last, 0, loss)
        return loss


class GroundedRSSMReplaceHwithSMoreFeatures(nj.Module):

    def __init__(
            self, deter=1024, stoch=32, classes=32, grounded=32,
            sim_tf=None, gradients_through_sim_tf=False,
            unroll=False, initial='learned', unimix=0.01, action_clip=1.0, img_is_deterministic=False, sg_in_img=False, **kw):
        assert sim_tf is not None
        self._deter = deter
        self._stoch = stoch
        self._grounded_size = grounded
        self._classes = classes
        self._sim_tf: Optional[MLP] = sim_tf
        self._gradients_through_sim_tf = gradients_through_sim_tf
        self._unroll = unroll
        self._initial = initial
        self._unimix = unimix
        self._action_clip = action_clip
        self._kw = kw

    def initial(self, bs):
        if self._classes:
            state = dict(
                deter_symlog_grounded=jnp.zeros([bs, self._deter], f32),
                logit=jnp.zeros([bs, self._stoch, self._classes], f32),
                stoch=jnp.zeros([bs, self._stoch, self._classes], f32),
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))
        else:
            state = dict(
                deter_symlog_grounded=jnp.zeros([bs, self._deter], f32),
                mean=jnp.zeros([bs, self._stoch], f32),
                std=jnp.ones([bs, self._stoch], f32),
                stoch=jnp.zeros([bs, self._stoch], f32),
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))

        if self._sim_tf:
            state['curr_state_symlog_sim_dynamics_pred'] = jnp.zeros([bs, self._grounded_size], f32)

        if self._initial == 'zeros':
            return cast(state)
        elif self._initial == 'learned':
            deter_symlog_grounded = self.get('initial', jnp.zeros, state['deter_symlog_grounded'][0].shape, f32)
            state['deter_symlog_grounded'] = jnp.repeat(jnp.tanh(deter_symlog_grounded)[None], bs, 0)
            state['stoch'] = self.get_stoch(cast(state['deter_symlog_grounded']))
            state['symlog_grounded'] = self.get_grounded_from_deter_grounded_and_stoch(cast(state['deter_symlog_grounded']), cast(state['stoch']))
            return cast(state)
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
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

    def imagine(self, action, state=None):
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

    def obs_step(self, prev_state, prev_action, embed, is_first):
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
        x = jnp.concatenate([prior['deter_symlog_grounded'], embed], -1)
        x = self.get('obs_out', Linear, **self._kw)(x)
        stats = self._stats('obs_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded = self.get_grounded_from_deter_grounded_and_stoch(deter_grounded=prior['deter_symlog_grounded'],
                                                                          stoch=stoch)

        post = {'stoch': stoch,
                'deter_symlog_grounded': prior['deter_symlog_grounded'],
                'symlog_grounded': symlog_grounded,
                **stats}
        if self._sim_tf is not None:
            post['curr_state_symlog_sim_dynamics_pred'] = prior['curr_state_symlog_sim_dynamics_pred']
        return cast(post), cast(prior)

    def img_step(self, prev_state, prev_action):
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

        curr_state_symlog_sim_dynamics_pred = self.get_sim_forward_dynamics(
            grounded_symlog_in=prev_state['symlog_grounded'],
            action=prev_action)
        if not self._gradients_through_sim_tf:
            curr_state_symlog_sim_dynamics_pred = sg(curr_state_symlog_sim_dynamics_pred)
        # img_inputs = jnp.concatenate([prev_stoch, prev_action, curr_state_symlog_sim_dynamics_pred], -1)

        x = self.get('deter_grounded_in', Linear, **self._kw)(curr_state_symlog_sim_dynamics_pred)
        deter_symlog_grounded = self.get('deter_grounded_out', Linear, self._deter, act='tanh')(x)



        x = self.get('img_in', Linear, **self._kw)(deter_symlog_grounded)
        # x = self.get('img_out', Linear, **self._kw)(x)


        # x = self.get('img_in', Linear, **self._kw)(img_inputs)
        # x, deter = self._gru(x, prev_state['deter'])
        # x = self.get('img_out', Linear, **self._kw)(x)
        stats = self._stats('img_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded = self.get_grounded_from_deter_grounded_and_stoch(deter_grounded=deter_symlog_grounded, stoch=stoch)

        prior = {'stoch': stoch,
                 'deter_symlog_grounded': deter_symlog_grounded,
                 'symlog_grounded': symlog_grounded,
                 **stats}
        if curr_state_symlog_sim_dynamics_pred is not None:
            prior['curr_state_symlog_sim_dynamics_pred'] = curr_state_symlog_sim_dynamics_pred
        return cast(prior)

    def get_stoch(self, deter_symlog_grounded):
        x = self.get('img_in', Linear, **self._kw)(deter_symlog_grounded)
        stats = self._stats('img_stats', x)
        dist = self.get_dist(stats)
        return cast(dist.mode())

    def get_grounded_from_deter_grounded_and_stoch(self, deter_grounded, stoch):
        if self._classes:
            # flatten stoch if the state representation is multi-dimensional
            new_shape = stoch.shape[:-2] + (self._stoch * self._classes,)
            stoch = stoch.reshape(new_shape)

        x = jnp.concatenate([deter_grounded, stoch], -1)
        x = self.get('grounded_in', Linear, **self._kw)(x)

        grounded_symlog_out = self.get('grounded_symlog_out', Linear, self._grounded_size)(x)
        return cast(grounded_symlog_out)

    def get_sim_forward_dynamics(self, grounded_symlog_in, action):
        action = cast(action)
        if self._action_clip > 0.0:
            action *= sg(self._action_clip / jnp.maximum(self._action_clip, jnp.abs(action)))
        x = jnp.concatenate([grounded_symlog_in, action], -1)
        next_grounded_symlog = self._sim_tf(x).mode()
        return cast(next_grounded_symlog)

    # def _gru(self, x, deter):
    #     x = jnp.concatenate([deter, x], -1)
    #     kw = {**self._kw, 'act': 'none', 'units': 3 * self._deter}
    #     x = self.get('gru', Linear, **kw)(x)
    #     reset, cand, update = jnp.split(x, 3, -1)
    #     reset = jax.nn.sigmoid(reset)
    #     cand = jnp.tanh(reset * cand)
    #     update = jax.nn.sigmoid(update - 1)
    #     deter = update * cand + (1 - update) * deter
    #     return deter, deter

    def _stats(self, name, x):
        if self._classes:
            x = self.get(name, Linear, self._stoch * self._classes)(x)
            logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
            if self._unimix:
                probs = jax.nn.softmax(logit, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - self._unimix) * probs + self._unimix * uniform
                logit = jnp.log(probs)
            stats = {'logit': logit}
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

    @staticmethod
    def grounded_state_loss(post, gt_state):
        symlog_grounded = post['symlog_grounded']
        distance = (symlog_grounded - jaxutils.symlog(gt_state)) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)
        return loss

    @staticmethod
    def simulation_tf_loss(symlog_grounded_next_state_pred, target_symlog_next_state, is_last):
        distance = (symlog_grounded_next_state_pred - target_symlog_next_state) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)  # sum over state dimensions, shape is still (batch x sequence len)
        # Only calculate loss for next state prediction when the current state is not the last state in a sequence.
        loss = jnp.where(is_last, 0, loss)
        return loss


class GroundedRSSMReplaceGRU(nj.Module):

    def __init__(
            self, deter=1024, stoch=32, classes=32, grounded=32,
            sim_tf=None, gradients_through_sim_tf=False,
            unroll=False, initial='learned', unimix=0.01, action_clip=1.0, img_is_deterministic=False,  sg_in_img=False, **kw):
        self._deter = deter
        self._stoch = stoch
        self._grounded_size = grounded
        self._classes = classes
        self._sim_tf: Optional[MLP] = sim_tf
        self._gradients_through_sim_tf = gradients_through_sim_tf
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
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))
        else:
            state = dict(
                deter=jnp.zeros([bs, self._deter], f32),
                mean=jnp.zeros([bs, self._stoch], f32),
                std=jnp.ones([bs, self._stoch], f32),
                stoch=jnp.zeros([bs, self._stoch], f32),
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))

        if self._sim_tf:
            state['curr_state_symlog_sim_dynamics_pred'] = jnp.zeros([bs, self._grounded_size], f32)

        if self._initial == 'zeros':
            return cast(state)
        elif self._initial == 'learned':
            deter = self.get('initial', jnp.zeros, state['deter'][0].shape, f32)
            state['deter'] = jnp.repeat(jnp.tanh(deter)[None], bs, 0)
            state['stoch'] = self.get_stoch(cast(state['deter']))
            state['symlog_grounded'] = self.get_grounded(cast(state['deter']), cast(state['stoch']))
            return cast(state)
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
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

    def imagine(self, action, state=None):
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

    def obs_step(self, prev_state, prev_action, embed, is_first):
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
        x = jnp.concatenate([prior['deter'], embed], -1)
        x = self.get('obs_out', Linear, **self._kw)(x)
        stats = self._stats('obs_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded = self.get_grounded(deter=prior['deter'], stoch=stoch)

        post = {'stoch': stoch,
                'deter': prior['deter'],
                'symlog_grounded': symlog_grounded,
                **stats}
        if self._sim_tf is not None:
            post['curr_state_symlog_sim_dynamics_pred'] = prior['curr_state_symlog_sim_dynamics_pred']
        return cast(post), cast(prior)

    def img_step(self, prev_state, prev_action):
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

        assert self._sim_tf is not None
        curr_state_symlog_sim_dynamics_pred = self.get_sim_forward_dynamics(
            grounded_symlog_in=prev_state['symlog_grounded'],
            action=prev_action)
        if not self._gradients_through_sim_tf:
            curr_state_symlog_sim_dynamics_pred = sg(curr_state_symlog_sim_dynamics_pred)

        x = self.get('prev_symlog_to_prev_deter_in', Linear, **self._kw)(prev_state['symlog_grounded'])
        prev_deter = self.get('prev_symlog_to_prev_deter_out', Linear, self._deter, act='tanh')(x)

        img_inputs = jnp.concatenate([prev_action, prev_deter], -1)

        x = self.get('img_in', Linear, **self._kw)(img_inputs)
        # x, deter = self._gru(x, prev_state['deter'])
        x, deter = self._gru(x, prev_deter)
        x = self.get('img_out', Linear, **self._kw)(x)
        stats = self._stats('img_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded = self.get_grounded(deter=deter, stoch=stoch)

        prior = {'stoch': stoch,
                 'deter': deter,
                 'symlog_grounded': symlog_grounded,
                 **stats}
        if curr_state_symlog_sim_dynamics_pred is not None:
            prior['curr_state_symlog_sim_dynamics_pred'] = curr_state_symlog_sim_dynamics_pred
        return cast(prior)

    def get_stoch(self, deter):
        x = self.get('img_out', Linear, **self._kw)(deter)
        stats = self._stats('img_stats', x)
        dist = self.get_dist(stats)
        return cast(dist.mode())

    def get_grounded(self, deter, stoch):
        if self._classes:
            # flatten stoch if the state representation is multi-dimensional
            new_shape = stoch.shape[:-2] + (self._stoch * self._classes,)
            stoch = stoch.reshape(new_shape)

        x = jnp.concatenate([deter, stoch], -1)
        x = self.get('grounded_in', Linear, **self._kw)(x)

        grounded_symlog_out = self.get('grounded_symlog_out', Linear, self._grounded_size)(x)
        return cast(grounded_symlog_out)

    def get_sim_forward_dynamics(self, grounded_symlog_in, action):
        action = cast(action)
        if self._action_clip > 0.0:
            action *= sg(self._action_clip / jnp.maximum(self._action_clip, jnp.abs(action)))
        x = jnp.concatenate([grounded_symlog_in, action], -1)
        next_grounded_symlog = self._sim_tf(x).mode()
        return cast(next_grounded_symlog)

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
            if self._unimix:
                probs = jax.nn.softmax(logit, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - self._unimix) * probs + self._unimix * uniform
                logit = jnp.log(probs)
            stats = {'logit': logit}
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

    @staticmethod
    def grounded_state_loss(post, gt_state):
        symlog_grounded = post['symlog_grounded']
        distance = (symlog_grounded - jaxutils.symlog(gt_state)) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)
        return loss

    @staticmethod
    def simulation_tf_loss(symlog_grounded_next_state_pred, target_symlog_next_state, is_last):
        distance = (symlog_grounded_next_state_pred - target_symlog_next_state) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)  # sum over state dimensions, shape is still (batch x sequence len)
        # Only calculate loss for next state prediction when the current state is not the last state in a sequence.
        loss = jnp.where(is_last, 0, loss)
        return loss



class GroundedRSSMReplaceGRU2(nj.Module):

    def __init__(
            self, deter=1024, stoch=32, classes=32, grounded=32,
            sim_tf=None, gradients_through_sim_tf=False,
            unroll=False, initial='learned', unimix=0.01, action_clip=1.0, img_is_deterministic=False, sg_in_img=False, **kw):
        self._deter = deter
        self._stoch = stoch
        self._grounded_size = grounded
        self._classes = classes
        self._sim_tf: Optional[MLP] = sim_tf
        self._gradients_through_sim_tf = gradients_through_sim_tf
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
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))
        else:
            state = dict(
                deter=jnp.zeros([bs, self._deter], f32),
                mean=jnp.zeros([bs, self._stoch], f32),
                std=jnp.ones([bs, self._stoch], f32),
                stoch=jnp.zeros([bs, self._stoch], f32),
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))

        if self._sim_tf:
            state['curr_state_symlog_sim_dynamics_pred'] = jnp.zeros([bs, self._grounded_size], f32)

        if self._initial == 'zeros':
            return cast(state)
        elif self._initial == 'learned':
            deter = self.get('initial', jnp.zeros, state['deter'][0].shape, f32)
            state['deter'] = jnp.repeat(jnp.tanh(deter)[None], bs, 0)
            state['stoch'] = self.get_stoch(cast(state['deter']))
            state['symlog_grounded'] = self.get_grounded(cast(state['deter']), cast(state['stoch']))
            return cast(state)
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
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

    def imagine(self, action, state=None):
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

    def obs_step(self, prev_state, prev_action, embed, is_first):
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
        x = jnp.concatenate([prior['deter'], embed], -1)
        x = self.get('obs_out', Linear, **self._kw)(x)
        stats = self._stats('obs_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded = self.get_grounded(deter=prior['deter'], stoch=stoch)

        post = {'stoch': stoch,
                'deter': prior['deter'],
                'symlog_grounded': symlog_grounded,
                **stats}
        if self._sim_tf is not None:
            post['curr_state_symlog_sim_dynamics_pred'] = prior['curr_state_symlog_sim_dynamics_pred']
        return cast(post), cast(prior)

    def img_step(self, prev_state, prev_action):
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

        assert self._sim_tf is not None
        curr_state_symlog_sim_dynamics_pred = self.get_sim_forward_dynamics(
            grounded_symlog_in=prev_state['symlog_grounded'],
            action=prev_action)
        if not self._gradients_through_sim_tf:
            curr_state_symlog_sim_dynamics_pred = sg(curr_state_symlog_sim_dynamics_pred)

        x = self.get('prev_symlog_to_prev_deter_in', Linear, **self._kw)(prev_state['symlog_grounded'])
        prev_deter = self.get('prev_symlog_to_prev_deter_out', Linear, self._deter, act='tanh')(x)

        img_inputs = jnp.concatenate([prev_action, prev_state['symlog_grounded']], -1)

        x = self.get('img_in', Linear, **self._kw)(img_inputs)
        # x, deter = self._gru(x, prev_state['deter'])
        x, deter = self._gru(x, prev_deter)
        x = self.get('img_out', Linear, **self._kw)(x)
        stats = self._stats('img_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded = self.get_grounded(deter=deter, stoch=stoch)

        prior = {'stoch': stoch,
                 'deter': deter,
                 'symlog_grounded': symlog_grounded,
                 **stats}
        if curr_state_symlog_sim_dynamics_pred is not None:
            prior['curr_state_symlog_sim_dynamics_pred'] = curr_state_symlog_sim_dynamics_pred
        return cast(prior)

    def get_stoch(self, deter):
        x = self.get('img_out', Linear, **self._kw)(deter)
        stats = self._stats('img_stats', x)
        dist = self.get_dist(stats)
        return cast(dist.mode())

    def get_grounded(self, deter, stoch):
        if self._classes:
            # flatten stoch if the state representation is multi-dimensional
            new_shape = stoch.shape[:-2] + (self._stoch * self._classes,)
            stoch = stoch.reshape(new_shape)

        x = jnp.concatenate([deter, stoch], -1)

        x = self.get('grounded_in', Linear, **self._kw)(x)

        grounded_symlog_out = self.get('grounded_symlog_out', Linear, self._grounded_size)(x)
        return cast(grounded_symlog_out)

    def get_sim_forward_dynamics(self, grounded_symlog_in, action):
        action = cast(action)
        if self._action_clip > 0.0:
            action *= sg(self._action_clip / jnp.maximum(self._action_clip, jnp.abs(action)))
        x = jnp.concatenate([grounded_symlog_in, action], -1)
        next_grounded_symlog = self._sim_tf(x).mode()
        return cast(next_grounded_symlog)

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
            if self._unimix:
                probs = jax.nn.softmax(logit, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - self._unimix) * probs + self._unimix * uniform
                logit = jnp.log(probs)
            stats = {'logit': logit}
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
        elif impl == 'none':
            loss = jnp.zeros(post['deter'].shape[:-1])
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

    @staticmethod
    def grounded_state_loss(post, gt_state):
        symlog_grounded = post['symlog_grounded']
        distance = (symlog_grounded - jaxutils.symlog(gt_state)) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)
        return loss

    @staticmethod
    def simulation_tf_loss(symlog_grounded_next_state_pred, target_symlog_next_state, is_last):
        distance = (symlog_grounded_next_state_pred - target_symlog_next_state) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)  # sum over state dimensions, shape is still (batch x sequence len)
        # Only calculate loss for next state prediction when the current state is not the last state in a sequence.
        loss = jnp.where(is_last, 0, loss)
        return loss

class GroundedRSSMReplaceGRU3(nj.Module):

    def __init__(
            self, deter=1024, stoch=32, classes=32, grounded=32,
            sim_tf=None, gradients_through_sim_tf=False,
            unroll=False, initial='learned', unimix=0.01, action_clip=1.0, img_is_deterministic=False,  sg_in_img=False, **kw):
        self._deter = deter
        self._stoch = stoch
        self._grounded_size = grounded
        self._classes = classes
        self._sim_tf: Optional[MLP] = sim_tf
        self._gradients_through_sim_tf = gradients_through_sim_tf
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
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))
        else:
            state = dict(
                deter=jnp.zeros([bs, self._deter], f32),
                mean=jnp.zeros([bs, self._stoch], f32),
                std=jnp.ones([bs, self._stoch], f32),
                stoch=jnp.zeros([bs, self._stoch], f32),
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))

        if self._sim_tf:
            state['curr_state_symlog_sim_dynamics_pred'] = jnp.zeros([bs, self._grounded_size], f32)

        if self._initial == 'zeros':
            return cast(state)
        elif self._initial == 'learned':
            deter = self.get('initial', jnp.zeros, state['deter'][0].shape, f32)
            state['deter'] = jnp.repeat(jnp.tanh(deter)[None], bs, 0)
            state['stoch'] = self.get_stoch(cast(state['deter']))
            state['symlog_grounded'] = self.get_grounded(cast(state['deter']), cast(state['stoch']))
            return cast(state)
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
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

    def imagine(self, action, state=None):
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

    def obs_step(self, prev_state, prev_action, embed, is_first):
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
        x = jnp.concatenate([prior['deter'], embed], -1)
        x = self.get('obs_out', Linear, **self._kw)(x)
        stats = self._stats('obs_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded = self.get_grounded(deter=prior['deter'], stoch=stoch)

        post = {'stoch': stoch,
                'deter': prior['deter'],
                'symlog_grounded': symlog_grounded,
                **stats}
        if self._sim_tf is not None:
            post['curr_state_symlog_sim_dynamics_pred'] = prior['curr_state_symlog_sim_dynamics_pred']
        return cast(post), cast(prior)

    def img_step(self, prev_state, prev_action):
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

        assert self._sim_tf is not None
        curr_state_symlog_sim_dynamics_pred = self.get_sim_forward_dynamics(
            grounded_symlog_in=prev_state['symlog_grounded'],
            action=prev_action)
        if not self._gradients_through_sim_tf:
            curr_state_symlog_sim_dynamics_pred = sg(curr_state_symlog_sim_dynamics_pred)

        x = self.get('prev_symlog_to_prev_deter_in', Linear, **self._kw)(prev_state['symlog_grounded'])
        # prev_deter = self.get('prev_symlog_to_prev_deter_out', Linear, self._deter, act='tanh')(x)

        img_inputs = jnp.concatenate([prev_action, prev_state['symlog_grounded']], -1)

        x = self.get('img_in', Linear, **self._kw)(img_inputs)

        deter = self.get('img_hidden', Linear, **self._kw)(x)
        # x, deter = self._gru(x, prev_deter)
        x = self.get('img_out', Linear, **self._kw)(deter)
        stats = self._stats('img_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded = self.get_grounded(deter=deter, stoch=stoch)

        prior = {'stoch': stoch,
                 'deter': deter,
                 'symlog_grounded': symlog_grounded,
                 **stats}
        if curr_state_symlog_sim_dynamics_pred is not None:
            prior['curr_state_symlog_sim_dynamics_pred'] = curr_state_symlog_sim_dynamics_pred
        return cast(prior)

    def get_stoch(self, deter):
        x = self.get('img_out', Linear, **self._kw)(deter)
        stats = self._stats('img_stats', x)
        dist = self.get_dist(stats)
        return cast(dist.mode())

    def get_grounded(self, deter, stoch):
        if self._classes:
            # flatten stoch if the state representation is multi-dimensional
            new_shape = stoch.shape[:-2] + (self._stoch * self._classes,)
            stoch = stoch.reshape(new_shape)

        x = jnp.concatenate([deter, stoch], -1)

        x = self.get('grounded_in', Linear, **self._kw)(x)

        grounded_symlog_out = self.get('grounded_symlog_out', Linear, self._grounded_size)(x)
        return cast(grounded_symlog_out)

    def get_sim_forward_dynamics(self, grounded_symlog_in, action):
        action = cast(action)
        if self._action_clip > 0.0:
            action *= sg(self._action_clip / jnp.maximum(self._action_clip, jnp.abs(action)))
        x = jnp.concatenate([grounded_symlog_in, action], -1)
        next_grounded_symlog = self._sim_tf(x).mode()
        return cast(next_grounded_symlog)

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
            if self._unimix:
                probs = jax.nn.softmax(logit, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - self._unimix) * probs + self._unimix * uniform
                logit = jnp.log(probs)
            stats = {'logit': logit}
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
        elif impl == 'none':
            loss = jnp.zeros(post['deter'].shape[:-1])
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

    @staticmethod
    def grounded_state_loss(post, gt_state):
        symlog_grounded = post['symlog_grounded']
        distance = (symlog_grounded - jaxutils.symlog(gt_state)) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)
        return loss

    @staticmethod
    def simulation_tf_loss(symlog_grounded_next_state_pred, target_symlog_next_state, is_last):
        distance = (symlog_grounded_next_state_pred - target_symlog_next_state) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)  # sum over state dimensions, shape is still (batch x sequence len)
        # Only calculate loss for next state prediction when the current state is not the last state in a sequence.
        loss = jnp.where(is_last, 0, loss)
        return loss


class GroundedRSSMReplaceGRU2MoreLayersToGetSymlogGrounded(nj.Module):

    def __init__(
            self, deter=1024, stoch=32, classes=32, grounded=32,
            sim_tf=None, gradients_through_sim_tf=False,
            unroll=False, initial='learned', unimix=0.01, action_clip=1.0, img_is_deterministic=False, sg_in_img=False, **kw):
        self._deter = deter
        self._stoch = stoch
        self._grounded_size = grounded
        self._classes = classes
        self._sim_tf: Optional[MLP] = sim_tf
        self._gradients_through_sim_tf = gradients_through_sim_tf
        self._unroll = unroll
        self._initial = initial
        self._unimix = unimix
        self._action_clip = action_clip
        self._kw = kw

        self._sg_in_img = sg_in_img

    def initial(self, bs):
        if self._classes:
            state = dict(
                deter=jnp.zeros([bs, self._deter], f32),
                logit=jnp.zeros([bs, self._stoch, self._classes], f32),
                stoch=jnp.zeros([bs, self._stoch, self._classes], f32),
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))
        else:
            state = dict(
                deter=jnp.zeros([bs, self._deter], f32),
                mean=jnp.zeros([bs, self._stoch], f32),
                std=jnp.ones([bs, self._stoch], f32),
                stoch=jnp.zeros([bs, self._stoch], f32),
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))

        if self._sim_tf:
            state['curr_state_symlog_sim_dynamics_pred'] = jnp.zeros([bs, self._grounded_size], f32)

        if self._initial == 'zeros':
            return cast(state)
        elif self._initial == 'learned':
            deter = self.get('initial', jnp.zeros, state['deter'][0].shape, f32)
            state['deter'] = jnp.repeat(jnp.tanh(deter)[None], bs, 0)
            state['stoch'] = self.get_stoch(cast(state['deter']))
            state['symlog_grounded'] = self.get_grounded(cast(state['deter']), cast(state['stoch']))
            return cast(state)
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
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

    def imagine(self, action, state=None):
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

    def obs_step(self, prev_state, prev_action, embed, is_first):
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
        x = jnp.concatenate([prior['deter'], embed], -1)
        x = self.get('obs_out', Linear, **self._kw)(x)
        stats = self._stats('obs_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded = self.get_grounded(deter=prior['deter'], stoch=stoch)

        post = {'stoch': stoch,
                'deter': prior['deter'],
                'symlog_grounded': symlog_grounded,
                **stats}
        # if self._sim_tf is not None:
        #     post['curr_state_symlog_sim_dynamics_pred'] = prior['curr_state_symlog_sim_dynamics_pred']
        return cast(post), cast(prior)

    def img_step(self, prev_state, prev_action):
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

        # curr_state_symlog_sim_dynamics_pred = self.get_sim_forward_dynamics(
        #     grounded_symlog_in=prev_state['symlog_grounded'],
        #     action=prev_action)
        # if not self._gradients_through_sim_tf:
        #     curr_state_symlog_sim_dynamics_pred = sg(curr_state_symlog_sim_dynamics_pred)
        curr_state_symlog_sim_dynamics_pred = None

        prev_symlog_grounded = prev_state['symlog_grounded']

        if self._sg_in_img:
            prev_symlog_grounded = sg(prev_symlog_grounded)

        x = self.get('prev_symlog_to_prev_deter_in', Linear, **self._kw)(prev_symlog_grounded)
        prev_deter = self.get('prev_symlog_to_prev_deter_out', Linear, self._deter, act='tanh')(x)

        img_inputs = jnp.concatenate([prev_action, prev_symlog_grounded], -1)

        x = self.get('img_in', Linear, **self._kw)(img_inputs)
        # x, deter = self._gru(x, prev_state['deter'])
        x, deter = self._gru(x, prev_deter)
        x = self.get('img_out', Linear, **self._kw)(x)
        stats = self._stats('img_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded = self.get_grounded(deter=deter, stoch=stoch)

        prior = {'stoch': stoch,
                 'deter': deter,
                 'symlog_grounded': symlog_grounded,
                 **stats}
        if curr_state_symlog_sim_dynamics_pred is not None:
            prior['curr_state_symlog_sim_dynamics_pred'] = curr_state_symlog_sim_dynamics_pred
        return cast(prior)

    def get_stoch(self, deter):
        x = self.get('img_out', Linear, **self._kw)(deter)
        stats = self._stats('img_stats', x)
        dist = self.get_dist(stats)
        return cast(dist.mode())

    def get_grounded(self, deter, stoch):
        if self._classes:
            # flatten stoch if the state representation is multi-dimensional
            new_shape = stoch.shape[:-2] + (self._stoch * self._classes,)
            stoch = stoch.reshape(new_shape)

        x = jnp.concatenate([deter, stoch], -1)

        x = self.get('grounded_in', Linear, **self._kw)(x)
        x = self.get('grounded_hidden', Linear, **self._kw)(x)

        grounded_symlog_out = self.get('grounded_symlog_out', Linear, self._grounded_size)(x)
        return cast(grounded_symlog_out)

    def get_sim_forward_dynamics(self, grounded_symlog_in, action):
        action = cast(action)
        if self._action_clip > 0.0:
            action *= sg(self._action_clip / jnp.maximum(self._action_clip, jnp.abs(action)))
        x = jnp.concatenate([grounded_symlog_in, action], -1)
        next_grounded_symlog = self._sim_tf(x).mode()
        return cast(next_grounded_symlog)

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
            if self._unimix:
                probs = jax.nn.softmax(logit, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - self._unimix) * probs + self._unimix * uniform
                logit = jnp.log(probs)
            stats = {'logit': logit}
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
        elif impl == 'none':
            loss = jnp.zeros(post['deter'].shape[:-1])
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

    @staticmethod
    def grounded_state_loss(post, gt_state):
        symlog_grounded = post['symlog_grounded']
        distance = (symlog_grounded - jaxutils.symlog(gt_state)) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)
        return loss

    @staticmethod
    def simulation_tf_loss(symlog_grounded_next_state_pred, target_symlog_next_state, is_last):
        distance = (symlog_grounded_next_state_pred - target_symlog_next_state) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)  # sum over state dimensions, shape is still (batch x sequence len)
        # Only calculate loss for next state prediction when the current state is not the last state in a sequence.
        loss = jnp.where(is_last, 0, loss)
        return loss



class GroundedRSSMReplaceGRU3MoreLayersToGetSymlogGrounded(nj.Module):

    def __init__(
            self, deter=1024, stoch=32, classes=32, grounded=32,
            sim_tf=None, gradients_through_sim_tf=False,
            unroll=False, initial='learned', unimix=0.01, action_clip=1.0, img_is_deterministic=False,  sg_in_img=False, **kw):
        self._deter = deter
        self._stoch = stoch
        self._grounded_size = grounded
        self._classes = classes
        self._sim_tf: Optional[MLP] = sim_tf
        self._gradients_through_sim_tf = gradients_through_sim_tf
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
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))
        else:
            state = dict(
                deter=jnp.zeros([bs, self._deter], f32),
                mean=jnp.zeros([bs, self._stoch], f32),
                std=jnp.ones([bs, self._stoch], f32),
                stoch=jnp.zeros([bs, self._stoch], f32),
                symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))

        if self._sim_tf:
            state['curr_state_symlog_sim_dynamics_pred'] = jnp.zeros([bs, self._grounded_size], f32)

        if self._initial == 'zeros':
            return cast(state)
        elif self._initial == 'learned':
            deter = self.get('initial', jnp.zeros, state['deter'][0].shape, f32)
            state['deter'] = jnp.repeat(jnp.tanh(deter)[None], bs, 0)
            state['stoch'] = self.get_stoch(cast(state['deter']))
            state['symlog_grounded'] = self.get_grounded(cast(state['deter']), cast(state['stoch']))
            return cast(state)
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
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

    def imagine(self, action, state=None):
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

    def obs_step(self, prev_state, prev_action, embed, is_first):
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
        x = jnp.concatenate([prior['deter'], embed], -1)
        x = self.get('obs_out', Linear, **self._kw)(x)
        stats = self._stats('obs_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded = self.get_grounded(deter=prior['deter'], stoch=stoch)

        post = {'stoch': stoch,
                'deter': prior['deter'],
                'symlog_grounded': symlog_grounded,
                **stats}
        if self._sim_tf is not None:
            post['curr_state_symlog_sim_dynamics_pred'] = prior['curr_state_symlog_sim_dynamics_pred']
        return cast(post), cast(prior)

    def img_step(self, prev_state, prev_action):
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

        assert self._sim_tf is not None
        curr_state_symlog_sim_dynamics_pred = self.get_sim_forward_dynamics(
            grounded_symlog_in=prev_state['symlog_grounded'],
            action=prev_action)
        if not self._gradients_through_sim_tf:
            curr_state_symlog_sim_dynamics_pred = sg(curr_state_symlog_sim_dynamics_pred)

        x = self.get('prev_symlog_to_prev_deter_in', Linear, **self._kw)(prev_state['symlog_grounded'])
        # prev_deter = self.get('prev_symlog_to_prev_deter_out', Linear, self._deter, act='tanh')(x)

        img_inputs = jnp.concatenate([prev_action, prev_state['symlog_grounded']], -1)

        x = self.get('img_in', Linear, **self._kw)(img_inputs)

        deter = self.get('img_hidden', Linear, **self._kw)(x)
        # x, deter = self._gru(x, prev_deter)
        x = self.get('img_out', Linear, **self._kw)(deter)
        stats = self._stats('img_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        symlog_grounded = self.get_grounded(deter=deter, stoch=stoch)

        prior = {'stoch': stoch,
                 'deter': deter,
                 'symlog_grounded': symlog_grounded,
                 **stats}
        if curr_state_symlog_sim_dynamics_pred is not None:
            prior['curr_state_symlog_sim_dynamics_pred'] = curr_state_symlog_sim_dynamics_pred
        return cast(prior)

    def get_stoch(self, deter):
        x = self.get('img_out', Linear, **self._kw)(deter)
        stats = self._stats('img_stats', x)
        dist = self.get_dist(stats)
        return cast(dist.mode())

    def get_grounded(self, deter, stoch):
        if self._classes:
            # flatten stoch if the state representation is multi-dimensional
            new_shape = stoch.shape[:-2] + (self._stoch * self._classes,)
            stoch = stoch.reshape(new_shape)

        x = jnp.concatenate([deter, stoch], -1)

        x = self.get('grounded_in', Linear, **self._kw)(x)
        x = self.get('grounded_hidden', Linear, **self._kw)(x)

        grounded_symlog_out = self.get('grounded_symlog_out', Linear, self._grounded_size)(x)
        return cast(grounded_symlog_out)

    def get_sim_forward_dynamics(self, grounded_symlog_in, action):
        action = cast(action)
        if self._action_clip > 0.0:
            action *= sg(self._action_clip / jnp.maximum(self._action_clip, jnp.abs(action)))
        x = jnp.concatenate([grounded_symlog_in, action], -1)
        next_grounded_symlog = self._sim_tf(x).mode()
        return cast(next_grounded_symlog)

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
            if self._unimix:
                probs = jax.nn.softmax(logit, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - self._unimix) * probs + self._unimix * uniform
                logit = jnp.log(probs)
            stats = {'logit': logit}
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
        elif impl == 'none':
            loss = jnp.zeros(post['deter'].shape[:-1])
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

    @staticmethod
    def grounded_state_loss(post, gt_state):
        symlog_grounded = post['symlog_grounded']
        distance = (symlog_grounded - jaxutils.symlog(gt_state)) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)
        return loss

    @staticmethod
    def simulation_tf_loss(symlog_grounded_next_state_pred, target_symlog_next_state, is_last):
        distance = (symlog_grounded_next_state_pred - target_symlog_next_state) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)  # sum over state dimensions, shape is still (batch x sequence len)
        # Only calculate loss for next state prediction when the current state is not the last state in a sequence.
        loss = jnp.where(is_last, 0, loss)
        return loss



class GroundedLiteralGTSimOnlySSM(nj.Module):

    def __init__(
            self, sim, grounded=32,
            gradients_through_sim_tf=False,
            unroll=False, initial='learned', action_clip=1.0, normalize_fn=None, denormalize_fn=None, **kw):
        from dreamerv3.embodied.envs.dmc_mjx_simulation import DMCMjxSimulationEnv
        if sim is None:
            raise ValueError(f"sim_tf must be provided for this class.")
        self._grounded_size = grounded
        self._sim: DMCMjxSimulationEnv = sim
        self._gradients_through_sim_tf = gradients_through_sim_tf
        self._unroll = unroll
        self._initial = initial
        self._action_clip = action_clip
        self._kw = kw

        self._normalize_fn = normalize_fn
        self._denormalize_fn = denormalize_fn

    def initial(self, bs):
        state = dict(symlog_grounded=jnp.zeros([bs, self._grounded_size], f32))

        state['curr_state_symlog_sim_dynamics_pred'] = jnp.zeros([bs, self._grounded_size], f32)

        if self._initial == 'zeros':
            return cast(state)
        elif self._initial == 'learned':
            symlog_grounded = self.get('initial_symlog_grounded', jnp.zeros, state['symlog_grounded'][0].shape, f32)
            state['symlog_grounded'] = jnp.repeat(jnp.tanh(symlog_grounded)[None], bs, 0)
            return cast(state)
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
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

    def imagine(self, action, state=None):
        swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
        state = self.initial(action.shape[0]) if state is None else state
        assert isinstance(state, dict), state
        action = swap(action)
        prior = jaxutils.scan(self.img_step, action, state, self._unroll)
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_dist(self, state, argmax=False):
        return tfd.Deterministic(state['symlog_grounded'].astype(f32))

    def obs_step(self, prev_state, prev_action, embed, is_first):
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

        # x = self.get('obs_out', Linear, **self._kw)(embed)
        # symlog_grounded = self.get('obs_grounded_symlog_out', Linear, self._grounded_size)(x)

        symlog_grounded = embed

        post = {'symlog_grounded': symlog_grounded}
        post['curr_state_symlog_sim_dynamics_pred'] = prior['curr_state_symlog_sim_dynamics_pred']
        return cast(post), cast(prior)

    def img_step(self, prev_state, prev_action):
        prev_grounded = prev_state['symlog_grounded']
        prev_action = cast(prev_action)
        if self._action_clip > 0.0:
            prev_action *= sg(self._action_clip / jnp.maximum(
                self._action_clip, jnp.abs(prev_action)))
        if len(prev_action.shape) > len(prev_grounded.shape):  # 2D actions.
            shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
            prev_action = prev_action.reshape(shape)

        curr_state_symlog_sim_dynamics_pred = self.get_sim_forward_dynamics(
            grounded_symlog_in=prev_grounded,
            action=prev_action)
        if not self._gradients_through_sim_tf:
            curr_state_symlog_sim_dynamics_pred = sg(curr_state_symlog_sim_dynamics_pred)

        symlog_grounded = curr_state_symlog_sim_dynamics_pred

        prior = {'symlog_grounded': symlog_grounded}
        prior['curr_state_symlog_sim_dynamics_pred'] = curr_state_symlog_sim_dynamics_pred
        return cast(prior)

    def get_sim_forward_dynamics(self, grounded_symlog_in, action):
        action = cast(action)
        if self._action_clip > 0.0:
            action *= sg(self._action_clip / jnp.maximum(self._action_clip, jnp.abs(action)))
        grounded_state = jaxutils.symexp(grounded_symlog_in)
        if self._denormalize_fn:
            grounded_state = self._denormalize_fn(grounded_state)
        grounded_next_state = self._sim.get_forward_dynamics(internal_state_batch=grounded_state,
                                                             action_batch=action)
        if self._normalize_fn:
            grounded_next_state = self._normalize_fn(grounded_next_state)
        next_grounded_symlog = jaxutils.symlog(grounded_next_state)
        return cast(next_grounded_symlog)

    def _mask(self, value, mask):
        return jnp.einsum('b...,b->b...', value, mask.astype(value.dtype))

    def dyn_loss(self, post, prior, is_first, impl='mse', free=1.0):
        if impl == 'none':
            loss = jnp.zeros(post['symlog_grounded'].shape[:-1])
        else:
            raise NotImplementedError(impl)
        return loss
        # if impl == 'mse':
        #     distance = (sg(post['symlog_grounded']) - prior['symlog_grounded']) ** 2  # MSE loss
        #     distance = jnp.where(distance < 1e-8, 0, distance)
        #     loss = distance.sum(-1)  # sum over state dimensions, shape is still (batch x sequence len)
        # else:
        #     raise NotImplementedError(impl)
        # return loss

    def rep_loss(self, post, prior, is_first, impl='none', free=1.0):
        if impl == 'none':
            loss = jnp.zeros(post['symlog_grounded'].shape[:-1])
        else:
            raise NotImplementedError(impl)
        return loss

    @staticmethod
    def grounded_state_loss(post, gt_state):
        symlog_grounded = post['symlog_grounded']
        distance = (symlog_grounded - jaxutils.symlog(gt_state)) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)
        return loss

    @staticmethod
    def simulation_tf_loss(symlog_grounded_next_state_pred, target_symlog_next_state, is_last):
        distance = (symlog_grounded_next_state_pred - target_symlog_next_state) ** 2  # MSE loss
        distance = jnp.where(distance < 1e-8, 0, distance)
        loss = distance.sum(-1)  # sum over state dimensions, shape is still (batch x sequence len)
        # Only calculate loss for next state prediction when the current state is not the last state in a sequence.
        loss = jnp.where(is_last, 0, loss)
        return loss
