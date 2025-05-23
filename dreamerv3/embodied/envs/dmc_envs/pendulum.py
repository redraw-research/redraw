from dm_control.rl import control
from dm_control.suite import common as dm_common
from dm_control import suite
from dm_control.utils import containers

from dm_control import mjcf
import numpy as np

from dreamerv3.embodied.envs.dmc_envs import common
import collections

_DEFAULT_TIME_LIMIT = 20
_ANGLE_BOUND = 8
_COSINE_BOUND = np.cos(np.deg2rad(_ANGLE_BOUND))
SUITE = containers.TaggedTasks()


class SwingUp(suite.pendulum.SwingUp):
    def get_observation(self, physics):
        """Returns an observation.

        Observations are states concatenating pole orientation and angular velocity
        and pixels from fixed camera.

        Args:
          physics: An instance of `physics`, Pendulum physics.

        Returns:
          A `dict` of observation.
        """
        obs = collections.OrderedDict()
        obs['orientation'] = physics.pole_orientation()
        obs['velocity'] = physics.angular_velocity()
        obs['image_complement'] = physics.velocity()
        return obs


class SwingUpReversedActions(SwingUp):
    def before_step(self, action, physics):
        action = -1.0 * np.asarray(getattr(action, "continuous_actions", action))
        physics.set_control(action)


class SwingUpDampened(SwingUp):
    def before_step(self, action, physics):
        # action = np.clip(action, a_min=-0.7, a_max=0.7)
        action = action * 0.8
        physics.set_control(action)


class DoubleActionMagnitudePendulumSwingUp(SwingUp):

    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        # action = (action - 0.75) * 4.0  # Rescale and shift action magnitudes

        if action[0] < 0.5:
            action[0] = (action[0] - 0.5) / 1.5
        else:
            action[0] = (action[0] - 0.5) * 2.0
        physics.set_control(action)

        # # Apply a rotational torque to the body of the walker along with a horizontal "wind".
        # body_name = 'torso'
        # idx = physics.model.name2id(body_name, "body")
        # fx, fy, fz = 0, 0, -50
        # physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 10, 0])


def pendulum_swingup(task=None, time_limit=_DEFAULT_TIME_LIMIT, random=None,
            environment_kwargs=None):
  physics = suite.pendulum.Physics.from_xml_string(*suite.pendulum.get_model_and_assets())
  if task is None:
      task = SwingUp(random=random)
  else:
      task = task(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)