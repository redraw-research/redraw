from dm_control import suite
from dm_control.rl import control
from dm_control.utils import containers
import numpy as np
import collections


_DEFAULT_TIME_LIMIT = 20  # (seconds)
_CONTROL_TIMESTEP = .02   # (seconds)
# For TURN tasks, the 'tip' geom needs to enter a spherical target of sizes:
_EASY_TARGET_SIZE = 0.07
_HARD_TARGET_SIZE = 0.03
# Initial spin velocity for the Stop task.
_INITIAL_SPIN_VELOCITY = 100
# Spinning slower than this value (radian/second) is considered stopped.
_STOP_VELOCITY = 1e-6
# Spinning faster than this value (radian/second) is considered spinning.
_SPIN_VELOCITY = 15.0


class Turn(suite.finger.Turn):

  def get_observation(self, physics):
    """Returns state, touch sensors, and target info."""
    obs = collections.OrderedDict()
    obs['position'] = physics.bounded_position()
    obs['velocity'] = physics.velocity()
    obs['touch'] = physics.touch()
    obs['target_position'] = physics.target_position()
    obs['dist_to_target'] = physics.dist_to_target()

    obs['image_complement'] = physics.velocity()
    return obs


class TurnReversedActions(Turn):
    def before_step(self, action, physics):
        action = -1.0 * np.asarray(getattr(action, "continuous_actions", action))
        physics.set_control(action)


class TurnTorqueApplied(Turn):

    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        physics.set_control(action)
        body_name = 'spinner'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 0, 0, 0
        rx, ry, rz = 0, 4, 0
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, rx, ry, rz])

def turn_hard(task=None, generator=None, time_limit=_DEFAULT_TIME_LIMIT, random=None,
              environment_kwargs=None):
  """Returns the hard Turn task."""
  if generator is None:
    physics = suite.finger.Physics.from_xml_string(*suite.finger.get_model_and_assets())
  else:
      physics = suite.finger.Physics.from_xml_string(*generator())
  if task is None:
      task = Turn(target_radius=_HARD_TARGET_SIZE, random=random)
  else:
     task = task(target_radius=_HARD_TARGET_SIZE, random=random)

  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


class Spin(suite.finger.Spin):
  def get_observation(self, physics):
    """Returns state and touch sensors, and target info."""
    obs = collections.OrderedDict()
    obs['position'] = physics.bounded_position()
    obs['velocity'] = physics.velocity()
    obs['touch'] = physics.touch()

    obs['image_complement'] = physics.velocity()
    return obs


def spin(task=None, generator=None, time_limit=_DEFAULT_TIME_LIMIT, random=None,
              environment_kwargs=None):
  """Returns the hard Turn task."""
  if generator is None:
    physics = suite.finger.Physics.from_xml_string(*suite.finger.get_model_and_assets())
  else:
      physics = suite.finger.Physics.from_xml_string(*generator())
  if task is None:
      task = Spin(random=random)
  else:
     task = task(target_radius=_HARD_TARGET_SIZE, random=random)

  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)
