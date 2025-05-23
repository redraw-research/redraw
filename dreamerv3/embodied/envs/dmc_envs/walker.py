import collections

from dm_control.rl import control
from dm_control.suite import common as dm_common
from dm_control import suite
from dm_control import mjcf
import numpy as np

from dreamerv3.embodied.envs.dmc_envs import common

_DEFAULT_TIME_LIMIT = 25
_CONTROL_TIMESTEP = .025

# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 1.2

# Horizontal speeds (meters/second) above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 8

RGBA_COLOR = "0.9 0.4 0.6 1"

def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return dm_common.read_model('walker.xml'), dm_common.ASSETS

def generate_pink_theme():
    model = mjcf.from_xml_string(dm_common.read_model('walker.xml'), assets=dm_common.ASSETS)
    model.asset.texture['skybox'].rgb1 = ".8 .4 .6"
    model.asset.texture['skybox'].rgb2 = ".1 .1 .1"
    model.worldbody.geom['floor'].rgba = "1 0 0.449 1"
    return model.to_xml_string(), dm_common.ASSETS

def generate_1_short_block():
    model = mjcf.from_xml_string(dm_common.read_model('walker.xml'), assets=dm_common.ASSETS)
    blocks = [
       {"pos": "5 0 .15", "size":".1 .8 .15", "rgba": RGBA_COLOR},
    ]
    return common._add_blocks(model, blocks), dm_common.ASSETS

def generate_1_tall_block():
    model = mjcf.from_xml_string(dm_common.read_model('walker.xml'), assets=dm_common.ASSETS)
    blocks = [
       {"pos": "5 0 .3", "size":".1 .8 .3", "rgba": RGBA_COLOR},
    ]
    return common._add_blocks(model, blocks), dm_common.ASSETS

def generate_5_short_blocks():
    model = mjcf.from_xml_string(dm_common.read_model('walker.xml'), assets=dm_common.ASSETS)
    blocks = [
       {"pos": "5 0 .15", "size":".1 .8 .15", "rgba": RGBA_COLOR},
       {"pos": "12 0 .15", "size":".1 .8 .15", "rgba": RGBA_COLOR},
       {"pos": "20 0 .15", "size":".1 .8 .15", "rgba": RGBA_COLOR},
       {"pos": "24 0 .15", "size":".1 .8 .15", "rgba": RGBA_COLOR},
       {"pos": "30 0 .15", "size":".1 .8 .15", "rgba": RGBA_COLOR},
    ]
    return common._add_blocks(model, blocks), dm_common.ASSETS

def generate_5_tall_blocks():
    model = mjcf.from_xml_string(dm_common.read_model('walker.xml'), assets=dm_common.ASSETS)
    blocks = [
       {"pos": "5 0 .3", "size":".1 .8 .3", "rgba": RGBA_COLOR},
       {"pos": "12 0 .3", "size":".1 .8 .3", "rgba": RGBA_COLOR},
       {"pos": "20 0 .3", "size":".1 .8 .3", "rgba": RGBA_COLOR},
       {"pos": "24 0 .3", "size":".1 .8 .3", "rgba": RGBA_COLOR},
       {"pos": "30 0 .3", "size":".1 .8 .3", "rgba": RGBA_COLOR},
    ]
    return common._add_blocks(model, blocks), dm_common.ASSETS

class TorsoPerturbedWalkerStand(suite.walker.PlanarWalker):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        physics.set_control(action)
        body_name = 'torso'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 0, 0, -20
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 0, 0])

class TorsoPerturbedWalkerWalk(suite.walker.PlanarWalker):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        physics.set_control(action)
        body_name = 'torso'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 0, 0, 100
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 0, 0])

class TorsoPerturbedWalkerRun(suite.walker.PlanarWalker):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        physics.set_control(action)
        body_name = 'torso'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 0, 0, 100
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 0, 0])

class Walker(suite.walker.PlanarWalker):
  """A planar walker task."""

  def get_observation(self, physics):
    """Returns an observation of body orientations, height and velocites."""
    obs = collections.OrderedDict()
    obs['orientations'] = physics.orientations()
    obs['height'] = physics.torso_height()
    obs['velocity'] = physics.velocity()
    obs['image_complement'] = np.concatenate([physics.data.qpos[0:1].copy(), physics.velocity()])
    return obs

class WalkerRelative(Walker):
    def get_observation(self, physics):
        """Returns an observation of body orientations, height and velocites."""
        obs = collections.OrderedDict()
        obs['orientations'] = physics.orientations()
        obs['height'] = physics.torso_height()
        obs['velocity'] = physics.velocity()
        obs['image_complement'] = physics.velocity()
        return obs

class WalkerRelativeDoubleActionMagnitude(WalkerRelative):

    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        action = (action - 0.75) * 4.0  # Rescale and shift action magnitudes
        physics.set_control(action)

        # Apply a rotational torque to the body of the walker along with a horizontal "wind".
        body_name = 'torso'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 0, 0, -50
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 10, 0])

class WalkerWalkReversedActions(Walker):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        action = action * -1.0
        physics.set_control(action)

class WalkerWalkDampened(Walker):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        action = action * 0.6
        physics.set_control(action)

class DoubleActionMagnitudeWalkerWalk(Walker):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        action = (action - 0.75) * 4.0  # Rescale and shift action magnitudes
        physics.set_control(action)

        # Apply a rotational torque to the body of the walker along with a horizontal "wind".
        body_name = 'torso'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 0, 0, -50
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 10, 0])

class DoubleActionMagnitudeNoForcesWalkerWalk(Walker):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        action = (action - 0.75) * 4.0  # Rescale and shift action magnitudes
        physics.set_control(action)

        # # Apply a rotational torque to the body of the walker along with a horizontal "wind".
        # body_name = 'torso'
        # idx = physics.model.name2id(body_name, "body")
        # fx, fy, fz = 0, 0, -50
        # physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 10, 0])


def walker_stand(task=None, generator=None, time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  if generator is None:
     physics = suite.walker.Physics.from_xml_string(*get_model_and_assets())
  else:
     physics = suite.walker.Physics.from_xml_string(*generator())
  if task is None:
     task = Walker(move_speed=0, random=random)
  else:
     task = task(move_speed=0, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)

def walker_walk(task=None, generator=None, time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  if generator is None:
     physics = suite.walker.Physics.from_xml_string(*get_model_and_assets())
  else:
     physics = suite.walker.Physics.from_xml_string(*generator())
  if task is None:
     task = Walker(move_speed=_WALK_SPEED, random=random)
  else:
     task = task(move_speed=_WALK_SPEED, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)

def walker_run(task=None, generator=None, time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  if generator is None:
     physics = suite.walker.Physics.from_xml_string(*get_model_and_assets())
  else:
     physics = suite.walker.Physics.from_xml_string(*generator())
  if task is None:
     task = Walker(move_speed=_RUN_SPEED, random=random)
  else:
     task = task(move_speed=_RUN_SPEED, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)
