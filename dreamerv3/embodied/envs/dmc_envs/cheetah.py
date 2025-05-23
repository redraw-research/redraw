from dm_control.rl import control
from dm_control.suite import common as dm_common
from dm_control import suite

from dm_control import mjcf
import numpy as np

from dreamerv3.embodied.envs.dmc_envs import common
import collections

# How long the simulation will run, in seconds.
_DEFAULT_TIME_LIMIT = 10
RGBA_COLOR = "0.9 0.4 0.6 1"
# Running speed above which reward is 1.

def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return dm_common.read_model('cheetah.xml'), dm_common.ASSETS

def generate_pink_theme():
    model = mjcf.from_xml_string(dm_common.read_model('cheetah.xml'), assets=dm_common.ASSETS)
    model.asset.texture['skybox'].rgb1 = ".8 .4 .6"
    model.asset.texture['skybox'].rgb2 = ".1 .1 .1"
    model.worldbody.geom['ground'].rgba = "1 0 0.449 1"
    return model.to_xml_string(), dm_common.ASSETS

def generate_1_short_block():
    model = mjcf.from_xml_string(dm_common.read_model('cheetah.xml'), assets=dm_common.ASSETS)
    blocks = [
       {"pos": "5 0 .15", "size":".1 .8 .15", "rgba": RGBA_COLOR},
    ]
    return common._add_blocks(model, blocks), dm_common.ASSETS

def generate_1_tall_block():
    model = mjcf.from_xml_string(dm_common.read_model('cheetah.xml'), assets=dm_common.ASSETS)
    blocks = [
       {"pos": "5 0 .3", "size":".1 .8 .3", "rgba": RGBA_COLOR},
    ]
    return common._add_blocks(model, blocks), dm_common.ASSETS

def generate_5_short_blocks():
    model = mjcf.from_xml_string(dm_common.read_model('cheetah.xml'), assets=dm_common.ASSETS)
    blocks = [
       {"pos": "5 0 .15", "size":".1 .8 .15", "rgba": RGBA_COLOR},
       {"pos": "12 0 .15", "size":".1 .8 .15", "rgba": RGBA_COLOR},
       {"pos": "20 0 .15", "size":".1 .8 .15", "rgba": RGBA_COLOR},
       {"pos": "24 0 .15", "size":".1 .8 .15", "rgba": RGBA_COLOR},
       {"pos": "30 0 .15", "size":".1 .8 .15", "rgba": RGBA_COLOR},
    ]
    return common._add_blocks(model, blocks), dm_common.ASSETS

def generate_5_tall_blocks():
    model = mjcf.from_xml_string(dm_common.read_model('cheetah.xml'), assets=dm_common.ASSETS)
    blocks = [
       {"pos": "5 0 .3", "size":".1 .8 .3", "rgba": RGBA_COLOR},
       {"pos": "12 0 .3", "size":".1 .8 .3", "rgba": RGBA_COLOR},
       {"pos": "20 0 .3", "size":".1 .8 .3", "rgba": RGBA_COLOR},
       {"pos": "24 0 .3", "size":".1 .8 .3", "rgba": RGBA_COLOR},
       {"pos": "30 0 .3", "size":".1 .8 .3", "rgba": RGBA_COLOR},
    ]
    return common._add_blocks(model, blocks), dm_common.ASSETS

class Cheetah(suite.cheetah.Cheetah):
   def get_observation(self, physics):
    """Returns an observation of the state, ignoring horizontal position."""
    obs = collections.OrderedDict()
    # Ignores horizontal position to maintain translational invariance.
    obs['position'] = physics.data.qpos[1:].copy()
    obs['velocity'] = physics.velocity()
    # obs['image_complement'] = np.concatenate([physics.data.qpos[0:1].copy(), physics.velocity()])
    obs['image_complement'] = np.concatenate([physics.velocity(), physics.activation()])
    obs['control'] = physics.control()
    return obs

class CheetahOrigObs(suite.cheetah.Cheetah):
   def get_observation(self, physics):
    """Returns an observation of the state, ignoring horizontal position."""
    obs = collections.OrderedDict()
    # Ignores horizontal position to maintain translational invariance.
    obs['position'] = physics.data.qpos[1:].copy()
    obs['velocity'] = physics.velocity()
    obs['image_complement'] = np.concatenate([physics.data.qpos[0:1].copy(), physics.velocity()])
    return obs

class CheetahScaledUpImageComplement(suite.cheetah.Cheetah):
   def get_observation(self, physics):
    """Returns an observation of the state, ignoring horizontal position."""
    obs = collections.OrderedDict()
    # Ignores horizontal position to maintain translational invariance.
    obs['position'] = physics.data.qpos[1:].copy()
    obs['velocity'] = physics.velocity()
    obs['image_complement'] = np.concatenate([physics.data.qpos[0:1].copy(), physics.velocity()]) * 1000.0
    return obs


   
class TorsoPerturbedCheetah(Cheetah):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        physics.set_control(action)
        body_name = 'torso'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 0, 0, 100
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 0, 0])

class DoubleActionMagnitudeCheetah(Cheetah):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        action = action * 2.0
        physics.set_control(action)

        body_name = 'torso'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 0, 0, -150
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 10, 0])

class CheetahGTStateIsVectorObs(suite.cheetah.Cheetah):
   def get_observation(self, physics):
    """Returns an observation of the state, ignoring horizontal position."""
    obs = collections.OrderedDict()
    obs['vector_obs'] = np.concatenate([physics.data.qpos, physics.data.qvel])
    # obs['vector_obs'] = np.concatenate([physics.data.qpos, physics.data.qvel, physics.data.qacc, physics.data.ctrl])
    return obs

def cheetah_run(task=None, generator=None, time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  if generator is None:
     physics = suite.cheetah.Physics.from_xml_string(*get_model_and_assets())
  else:
     physics = suite.cheetah.Physics.from_xml_string(*generator())
  if task is None:
     task = Cheetah(random=random)
  else:
     task = task(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)
