from dm_control.rl import control
from dm_control.suite import common as dm_common
from dm_control import suite
from dm_control.utils import containers

from dm_control import mjcf
import numpy as np

from dreamerv3.embodied.envs.dmc_envs import common
import collections

# How long the simulation will run, in seconds.
_DEFAULT_TIME_LIMIT = 10
SUITE = containers.TaggedTasks()


class BalanceBiasedDoubleActionMagnitude(suite.cartpole.Balance):

    def before_step(self, action, physics):
        action = np.asarray(getattr(action, "continuous_actions", action))

        # Rescale/Compress the original action space to be defined in [-0.5, 0.5].
        # In the remaining spaces [-1.0, -0.5) and (0.5, 1.0] add "Danger" zones
        # where the action is overridden to always be -1.0
        action = np.where(np.abs(action) < 0.5, action * 2.0, -1.0)
        physics.set_control(action)

        body_name = 'pole_1'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 0, 0, 0
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 0, 0])

class BalanceBiasedDoubleActionMagnitude2(suite.cartpole.Balance):

    def before_step(self, action, physics):
        action = np.asarray(getattr(action, "continuous_actions", action))
        action = (action * 2.0) + 0.25  # apply a bias and re-scale to the action
        physics.set_control(action)

        body_name = 'pole_1'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 0, 0, 0
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 0, 0])


def cartpole_balance(task=None, time_limit=_DEFAULT_TIME_LIMIT, random=None,
            environment_kwargs=None):
  """Returns the Cartpole Balance task."""
  physics = suite.cartpole.Physics.from_xml_string(*suite.cartpole.get_model_and_assets())
  if task is None:
      task = suite.cartpole.Balance(swing_up=False, sparse=False, random=random)
  else:
      task = task(swing_up=False, sparse=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)