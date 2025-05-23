from dm_control.rl import control
from dm_control.suite import common as dm_common
from dm_control import suite
from dm_control.utils import containers

from dm_control import mjcf
import numpy as np

from dreamerv3.embodied.envs.dmc_envs import common
import collections

_DEFAULT_TIME_LIMIT = 20
_BIG_TARGET = .05
_SMALL_TARGET = .015



class Reacher(suite.reacher.Reacher):

    def get_observation(self, physics):
        """Returns an observation of the state and the target position."""
        obs = collections.OrderedDict()
        obs['position'] = physics.position()
        obs['to_target'] = physics.finger_to_target()
        obs['velocity'] = physics.velocity()
        obs['image_complement'] = physics.velocity()
        return obs

class DoubleActionMagnitudeReacher(Reacher):

    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        action[0] = (action[0] - 0.75) * 4.0  # Rescale and shift action magnitudes

        if action[0] < 0.5:
            action[0] = (action[0] - 0.5) / 1.5
        else:
            action[0] = (action[0] - 0.5) * 2.0

        # if action[0] < -0.9:
        #     action[0] = 0.0
        # if action[0] > 0.9:
        #     action[0] = 0.0
        #
        
        if action[1] < 0.5:
            action[1] = (action[1] - 0.5) / 1.5
        else:
            action[1] = (action[1] - 0.5) * 2.0



        # action[1] = (action[1] - 0.75) * 4.0  # Rescale and shift action magnitudes
        physics.set_control(action)

        # # Apply a rotational torque to the body of the walker along with a horizontal "wind".
        # body_name = 'torso'
        # idx = physics.model.name2id(body_name, "body")
        # fx, fy, fz = 0, 0, -50
        # physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 10, 0])


class ReacherReversedActions(Reacher):

    def before_step(self, action, physics):
        action = -1.0 * np.asarray(getattr(action, "continuous_actions", action))
        physics.set_control(action)

def reacher_easy(task=None, time_limit=_DEFAULT_TIME_LIMIT, random=None,
            environment_kwargs=None):
  physics = suite.reacher.Physics.from_xml_string(*suite.reacher.get_model_and_assets())
  if task is None:
      task = Reacher(target_size=_BIG_TARGET, random=random)
  else:
      task = task(target_size=_BIG_TARGET, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


def reacher_hard(task=None, time_limit=_DEFAULT_TIME_LIMIT, random=None,
            environment_kwargs=None):
  physics = suite.reacher.Physics.from_xml_string(*suite.reacher.get_model_and_assets())
  if task is None:
      task = Reacher(target_size=_SMALL_TARGET, random=random)
  else:
      task = task(target_size=_SMALL_TARGET, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)