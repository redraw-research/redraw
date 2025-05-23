import collections
import os

from dm_control.rl import control
from dm_control.suite import common as dm_common
from dm_control import suite
from dm_control.utils import io as resources

from dm_control import mjcf
import numpy as np

from dreamerv3.embodied.envs.dmc_envs import common

# How long the simulation will run, in seconds.
_CONTROL_TIMESTEP = .02  # (Seconds)

# Default duration of an episode, in seconds.
_DEFAULT_TIME_LIMIT = 20

# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 0.6

# Hopping speed above which hop reward is 1.
_HOP_SPEED = 2
RGBA_COLOR = "0.9 0.4 0.6 1"


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return dm_common.read_model('hopper.xml'), dm_common.ASSETS

def generate_pink_theme():
    model = mjcf.from_xml_string(dm_common.read_model('hopper.xml'), assets=dm_common.ASSETS)
    model.asset.texture['skybox'].rgb1 = ".8 .4 .6"
    model.asset.texture['skybox'].rgb2 = ".1 .1 .1"
    model.worldbody.geom['floor'].rgba = "1 0 0.449 1"
    return model.to_xml_string(), dm_common.ASSETS


def generate_1_short_block():
    model = mjcf.from_xml_string(dm_common.read_model('hopper.xml'), assets=dm_common.ASSETS)
    blocks = [
       {"pos": "5 0 .15", "size":".1 1 .15", "rgba": RGBA_COLOR},
    ]
    return common._add_blocks(model, blocks), dm_common.ASSETS


def generate_1_tall_block():
    model = mjcf.from_xml_string(dm_common.read_model('hopper.xml'), assets=dm_common.ASSETS)
    blocks = [
       {"pos": "5 0 .3", "size":".1 1 .3", "rgba": RGBA_COLOR},
    ]
    return common._add_blocks(model, blocks), dm_common.ASSETS

def generate_5_short_blocks():
    model = mjcf.from_xml_string(dm_common.read_model('hopper.xml'), assets=dm_common.ASSETS)
    blocks = [
       {"pos": "5 0 .15", "size":".1 1 .15", "rgba": RGBA_COLOR},
       {"pos": "12 0 .15", "size":".1 1 .15", "rgba": RGBA_COLOR},
       {"pos": "20 0 .15", "size":".1 1 .15", "rgba": RGBA_COLOR},
       {"pos": "24 0 .15", "size":".1 1 .15", "rgba": RGBA_COLOR},
       {"pos": "30 0 .15", "size":".1 1 .15", "rgba": RGBA_COLOR},
    ]
    return common._add_blocks(model, blocks), dm_common.ASSETS

def generate_5_tall_blocks():
    model = mjcf.from_xml_string(dm_common.read_model('hopper.xml'), assets=dm_common.ASSETS)
    blocks = [
       {"pos": "5 0 .3", "size":".1 1 .3", "rgba": RGBA_COLOR},
       {"pos": "12 0 .3", "size":".1 1 .3", "rgba": RGBA_COLOR},
       {"pos": "20 0 .3", "size":".1 1 .3", "rgba": RGBA_COLOR},
       {"pos": "24 0 .3", "size":".1 1 .3", "rgba": RGBA_COLOR},
       {"pos": "30 0 .3", "size":".1 1 .3", "rgba": RGBA_COLOR},
    ]
    return common._add_blocks(model, blocks), dm_common.ASSETS


class Hopper(suite.hopper.Hopper):
    def get_observation(self, physics):
        """Returns an observation of positions, velocities and touch sensors."""
        obs = collections.OrderedDict()
        # Ignores horizontal position to maintain translational invariance:
        obs['position'] = physics.data.qpos[1:].copy()
        obs['velocity'] = physics.velocity()
        obs['touch'] = physics.touch()

        obs['image_complement'] = physics.velocity()
        return obs


class TorsoPerturbedHopperHop(Hopper):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        physics.set_control(action)
        body_name = 'torso'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 0, 0, -10
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 0, 0])

class TorsoPerturbedHopperStand50(Hopper):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        physics.set_control(action)
        body_name = 'torso'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 0, 0, -50
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 0, 0])

class TorsoPerturbedHopperStand40(Hopper):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        physics.set_control(action)
        body_name = 'torso'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 0, 0, -40
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 0, 0])

class TorsoPerturbedHopperStand30(Hopper):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        physics.set_control(action)
        body_name = 'torso'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 0, 0, -30
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 0, 0])

class TorsoPerturbedHopperStand15(Hopper):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        physics.set_control(action)
        body_name = 'torso'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 0, 0, -15
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 0, 0])

class TorsoPerturbedHopperStand10(Hopper):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        physics.set_control(action)
        body_name = 'torso'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 0, 0, -10
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 0, 0])

class TorsoPerturbedHopperStand5(Hopper):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        physics.set_control(action)
        body_name = 'torso'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 0, 0, -5
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 0, 0])

def get_ice_hopper_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  ice_hopper_xml_path = os.path.join(os.path.dirname(__file__), 'hopper_ice.xml')
  return resources.GetResource(ice_hopper_xml_path), dm_common.ASSETS


def hopper_stand(task=None, generator=None, time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  if generator is None:
     physics = suite.hopper.Physics.from_xml_string(*get_model_and_assets())
  else:
     physics = suite.hopper.Physics.from_xml_string(*generator())
  if task is None:
     task = Hopper(hopping=False, random=random)
  else:
     task = task(hopping=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)

def hopper_hop(task=None, generator=None, time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  if generator is None:
     physics = suite.hopper.Physics.from_xml_string(*get_model_and_assets())
  else:
     physics = suite.hopper.Physics.from_xml_string(*generator())
  if task is None:
     task = Hopper(hopping=True, random=random)
  else:
     task = task(hopping=True, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)
