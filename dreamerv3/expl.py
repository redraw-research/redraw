import jax
import jax.numpy as jnp

tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

from dreamerv3 import nets
from dreamerv3 import jaxutils
from dreamerv3 import ninjax as nj


class Disag(nj.Module):

    def __init__(self, wm, act_space, config):
        self.config = config.update({'disag_head.inputs': ['tensor']})
        self.opt = jaxutils.Optimizer(name='disag_opt', **config.expl_opt)
        self.inputs = nets.Input(config.disag_head.inputs, dims='deter')
        self.target = nets.Input(self.config.disag_target, dims='deter')
        self.nets = [
            nets.MLP(shape=None, **self.config.disag_head, name=f'disag{i}')
            for i in range(self.config.disag_models)]

    def __call__(self, traj):
        inp = self.inputs(traj)
        preds = jnp.array([net(inp).mode() for net in self.nets])
        return preds.std(0).mean(-1)[1:]

    def train(self, data):
        return self.opt(self.nets, self.loss, data)

    def loss(self, data):
        inp = sg(self.inputs(data)[:, :-1])
        tar = sg(self.target(data)[:, 1:])
        losses = []

        if self.config.apply_rho_to_expl_loss:
            data["t"] = jnp.repeat(jnp.arange(start=0, stop=self.config.batch_length)[jnp.newaxis, :], repeats=self.config.batch_size, axis=0)
            for net in self.nets:
                print(f"disag inputs shape: {inp.shape}")
                print(f"disag targets shape: {tar.shape}")
                net._shape = tar.shape[2:]
                loss = -net(inp).log_prob(tar)
                print(f"expl loss shape before rho: {loss.shape}")
                loss *= (self.config.td_loss_rho ** data["t"][:, :-1])
                print(f"expl loss shape after rho: {loss.shape}")
                # only train disagreement on sim trajectories without residual dynamics predictions
                # if 'is_real' in data:
                #     # jax.debug.print("is any explore data real: {}", jnp.any(data['is_real']))
                #     loss = jnp.where(data['is_real'][:, :-1], 0, loss)
                losses.append(loss.mean())
        else:
            for net in self.nets:
                net._shape = tar.shape[2:]
                loss = -net(inp).log_prob(tar)
                # only train disagreement on sim trajectories without residual dynamics predictions
                # if 'is_real' in data:
                #     # jax.debug.print("is any explore data real: {}", jnp.any(data['is_real']))
                #     loss = jnp.where(data['is_real'][:, :-1], 0, loss)
                losses.append(loss.mean())
        return jnp.array(losses).sum() * self.config['disag_loss_scale']
