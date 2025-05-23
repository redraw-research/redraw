from dm_control.rl import control
from dm_control.suite import common as dm_common
from dm_control import suite

from dm_control import mjcf
import numpy as np

from dreamerv3.embodied.envs.dmc_envs import common

_DEFAULT_TIME_LIMIT = 25
_CONTROL_TIMESTEP = .025

# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.4

# Horizontal speeds above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 10

RGBA_COLOR = "0.9 0.4 0.6 1"

def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return dm_common.read_model('humanoid.xml'), dm_common.ASSETS

def generate_pink_theme():
    model = mjcf.from_xml_string(dm_common.read_model('humanoid.xml'), assets=dm_common.ASSETS)
    model.asset.texture['skybox'].rgb1 = ".8 .4 .6"
    model.asset.texture['skybox'].rgb2 = ".1 .1 .1"
    model.worldbody.geom['floor'].rgba = "1 0 0.449 1"
    return model.to_xml_string(), dm_common.ASSETS

def generate_closed_area():
    model = mjcf.from_xml_string(dm_common.read_model('humanoid.xml'), assets=dm_common.ASSETS)
    blocks = [
       {"pos": "-4 0 .1", "euler": "0 0 90", "size":"4.1 .1 .1", "rgba": RGBA_COLOR},
       {"pos": "4 0 .1", "euler": "0 0 90", "size":"4.1 .1 .1", "rgba": RGBA_COLOR},
       {"pos": "0 4 .1", "euler": "0 0 90", "size":".1 3.9 .1", "rgba": RGBA_COLOR},
       {"pos": "0 -4 .1", "euler": "0 0 90", "size":".1 3.9 .1", "rgba": RGBA_COLOR},
    ]
    return common._add_blocks(model, blocks), dm_common.ASSETS

def generate_hexagonal_gates():
    model = mjcf.from_xml_string(dm_common.read_model('humanoid.xml'), assets=dm_common.ASSETS)
    blocks = [
       {"pos": "0 4 .5", "euler": "0 0 90", "size":".1 1 .5", "rgba": RGBA_COLOR},
       {"pos": "-3 2 .5", "euler": "0 0 -30", "size":".1 1 .5", "rgba": RGBA_COLOR},
       {"pos": "-3 -2 .5", "euler": "0 0 30", "size":".1 1 .5", "rgba": RGBA_COLOR},
       {"pos": "3 -2 .5", "euler": "0 0 -30", "size":".1 1 .5", "rgba": RGBA_COLOR},
       {"pos": "3 2 .5", "euler": "0 0 30", "size":".1 1 .5", "rgba": RGBA_COLOR},
       {"pos": "0 -4 .5", "euler": "0 0 90", "size":".1 1 .5", "rgba": RGBA_COLOR},
    ]
    return common._add_blocks(model, blocks), dm_common.ASSETS

def generate_square_gates():
    model = mjcf.from_xml_string(dm_common.read_model('humanoid.xml'), assets=dm_common.ASSETS)
    blocks = [
       {"pos": "-4 0 .5", "euler": "0 0 90", "size": "1.8 .1 .5", "rgba": RGBA_COLOR},
       {"pos": "4 0 .5", "euler": "0 0 90", "size": "1.8 .1 .5", "rgba": RGBA_COLOR},
       {"pos": "0 4 .5", "euler": "0 0 0", "size": "1.8 .1 .5", "rgba": RGBA_COLOR},
       {"pos": "0 -4 .5", "euler": "0 0 0", "size": "1.8 .1 .5", "rgba": RGBA_COLOR},
    ]
    return common._add_blocks(model, blocks), dm_common.ASSETS

def generate_dense_objects():
    model = mjcf.from_xml_string(dm_common.read_model('humanoid.xml'), assets=dm_common.ASSETS)
    blocks = common._block_generator(20)
    return common._add_blocks(model, blocks), dm_common.ASSETS

class TorsoPerturbedHumanoidStand(suite.humanoid.Humanoid):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        physics.set_control(action)
        body_name = 'torso'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 0, 0, -20
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 0, 0])

class TorsoPerturbedHumanoidWalk(suite.humanoid.Humanoid):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        physics.set_control(action)
        body_name = 'torso'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 50, 50, 50
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 0, 0])

class TorsoPerturbedHumanoidRun(suite.humanoid.Humanoid):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        physics.set_control(action)
        body_name = 'torso'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 50, 50, 50
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 0, 0])

def stand(task=None, generator=None, time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  if generator is None:
     physics = suite.humanoid.Physics.from_xml_string(*get_model_and_assets())
  else:
     physics = suite.humanoid.Physics.from_xml_string(*generator())
  if task is None:
     task = suite.humanoid.Humanoid(move_speed=0, pure_state=False, random=random)
  else:
     task = task(move_speed=0, pure_state=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)

def walk(task=None, generator=None, time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  if generator is None:
     physics = suite.humanoid.Physics.from_xml_string(*get_model_and_assets())
  else:
     physics = suite.humanoid.Physics.from_xml_string(*generator())
  if task is None:
     task = suite.humanoid.Humanoid(move_speed=_WALK_SPEED, pure_state=False, random=random)
  else:
     task = task(move_speed=_WALK_SPEED, pure_state=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


def humanoid_run(task=None, generator=None, time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  if generator is None:
     physics = suite.humanoid.Physics.from_xml_string(*get_model_and_assets())
  else:
     physics = suite.humanoid.Physics.from_xml_string(*generator())
  if task is None:
     task = suite.humanoid.Humanoid(move_speed=_RUN_SPEED, pure_state=False, random=random)
  else:
     task = task(move_speed=_RUN_SPEED, pure_state=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)

def humanoid_run_pure_state(task=None, generator=None, time_limit=_DEFAULT_TIME_LIMIT, random=None,
                   environment_kwargs=None):
  """Returns the Run task."""
  if generator is None:
     physics = suite.humanoid.Physics.from_xml_string(*get_model_and_assets())
  else:
     physics = suite.humanoid.Physics.from_xml_string(*generator())
  if task is None:
     task = suite.humanoid.Humanoid(move_speed=_RUN_SPEED, pure_state=True, random=random)
  else:
     task = task(move_speed=_RUN_SPEED, pure_state=True, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)