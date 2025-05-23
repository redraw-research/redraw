from dm_control import suite
from dm_control.rl import control
from dm_control.utils import containers
import numpy as np
import collections

_DEFAULT_TIME_LIMIT = 20  # (seconds)
_CONTROL_TIMESTEP = .02   # (seconds)


class BallInCup(suite.ball_in_cup.BallInCup):
    def get_observation(self, physics):
        """Returns an observation of the state."""
        obs = collections.OrderedDict()
        obs['position'] = physics.position()
        obs['velocity'] = physics.velocity()
        obs['image_complement'] = physics.velocity()
        return obs


class BallInCupWithWind(BallInCup):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        physics.set_control(action)

        body_name = 'ball'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = -2, 0, 0
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 0, 0])

class HighGravityBallInCup(BallInCup):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        physics.set_control(action)

        body_name = 'ball'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 0, 0, -0.5
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 0, 0])

class LowGravity0p5BallInCup(BallInCup):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        physics.set_control(action)

        body_name = 'ball'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 0, 0, 0.5
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 0, 0])

class LowGravity0p25BallInCup(BallInCup):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        physics.set_control(action)

        body_name = 'ball'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 0, 0, 0.25
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 0, 0])

class LowGravity0p1BallInCup(BallInCup):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        physics.set_control(action)

        body_name = 'ball'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 0, 0, 0.1
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 0, 0])

class LowGravity0p05BallInCup(BallInCup):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        physics.set_control(action)

        body_name = 'ball'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 0, 0, 0.05
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 0, 0])

class LowGravity1BallInCup(BallInCup):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        physics.set_control(action)

        body_name = 'ball'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 0, 0, 1.0
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 0, 0])

class LowGravity2BallInCup(BallInCup):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        physics.set_control(action)

        body_name = 'ball'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 0, 0, 2.0
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 0, 0])

def ball_in_cup_catch(task=None, generator=None, time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Ball-in-Cup task."""
  if generator is None:
     physics = suite.ball_in_cup.Physics.from_xml_string(*suite.ball_in_cup.get_model_and_assets())
  else:
     physics = suite.ball_in_cup.Physics.from_xml_string(*generator())
  if task is None:
     task = BallInCup(random=random)
  else:
     task = task(random=random)

  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)