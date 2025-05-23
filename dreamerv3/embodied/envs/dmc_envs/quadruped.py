import collections

from dm_control.rl import control
from dm_control.suite import common as dm_common
from dm_control import suite
from dm_control.utils import xml_tools
from lxml import etree


from dm_control import mjcf
import numpy as np
from dreamerv3.embodied.envs.dmc_envs import common

from lxml import etree
_DEFAULT_TIME_LIMIT = 20
_CONTROL_TIMESTEP = .02

# Horizontal speeds above which the move reward is 1.
_RUN_SPEED = 5
_WALK_SPEED = 0.5

# Constants related to terrain generation.
_HEIGHTFIELD_ID = 0
_TERRAIN_SMOOTHNESS = 0.15  # 0.0: maximally bumpy; 1.0: completely smooth.
_TERRAIN_BUMP_SCALE = 2  # Spatial scale of terrain bumps (in meters).

# Named model elements.
_TOES = ['toe_front_left', 'toe_back_left', 'toe_back_right', 'toe_front_right']
_WALLS = ['wall_px', 'wall_py', 'wall_nx', 'wall_ny']

RGBA_COLOR = "0.9 0.4 0.6 1"

def make_model(generator, floor_size=None, terrain=False, rangefinders=False,
               walls_and_ball=False):
  """Returns the model XML string."""
  xml_string = dm_common.read_model('quadruped.xml')
  parser = etree.XMLParser(remove_blank_text=True)
  mjcf = etree.XML(xml_string, parser)
  
  if generator is not None:
    mjcf.find('visual').getchildren()[1].attrib['znear'] = '0.01'
    xml_string = etree.tostring(mjcf, pretty_print=True)
    xml_string = generator(xml_string)
    parser = etree.XMLParser(remove_blank_text=True)
    mjcf = etree.XML(xml_string, parser)
  
  # Set floor size.
  if floor_size is not None:
    floor_geom = mjcf.find('.//geom[@name=\'floor\']')
    floor_geom.attrib['size'] = f'{floor_size} {floor_size} .5'

  # Remove walls, ball and target.
  if not walls_and_ball:
    for wall in _WALLS:
      wall_geom = xml_tools.find_element(mjcf, 'geom', wall)
      wall_geom.getparent().remove(wall_geom)

    # Remove ball.
    ball_body = xml_tools.find_element(mjcf, 'body', 'ball')
    ball_body.getparent().remove(ball_body)

    # Remove target.
    target_site = xml_tools.find_element(mjcf, 'site', 'target')
    target_site.getparent().remove(target_site)

  # Remove terrain.
  if not terrain:
    terrain_geom = xml_tools.find_element(mjcf, 'geom', 'terrain')
    terrain_geom.getparent().remove(terrain_geom)

  # Remove rangefinders if they're not used, as range computations can be
  # expensive, especially in a scene with heightfields.
  if not rangefinders:
    rangefinder_sensors = mjcf.findall('.//rangefinder')
    for rf in rangefinder_sensors:
      rf.getparent().remove(rf)

  return etree.tostring(mjcf, pretty_print=True)


def generate_pink_theme(model):
    model = mjcf.from_xml_string(model, assets=dm_common.ASSETS)
    model.asset.texture['skybox'].rgb1 = ".8 .4 .6"
    model.asset.texture['skybox'].rgb2 = ".1 .1 .1"
    model.worldbody.geom['floor'].rgba = "1 0 0.449 1"
    return model.to_xml_string()

def generate_closed_area(model):
    model = mjcf.from_xml_string(model, assets=dm_common.ASSETS)
    blocks = [
       {"pos": "-4 0 .1", "euler": "0 0 90", "size":"4.1 .1 .1", "rgba": RGBA_COLOR},
       {"pos": "4 0 .1", "euler": "0 0 90", "size":"4.1 .1 .1", "rgba": RGBA_COLOR},
       {"pos": "0 4 .1", "euler": "0 0 90", "size":".1 3.9 .1", "rgba": RGBA_COLOR},
       {"pos": "0 -4 .1", "euler": "0 0 90", "size":".1 3.9 .1", "rgba": RGBA_COLOR},
    ]
    return common._add_blocks(model, blocks)

def generate_hexagonal_gates(xml_string):
    model = mjcf.from_xml_string(xml_string, assets=dm_common.ASSETS)
    blocks = [
       {"pos": "0 4 .5", "euler": "0 0 90", "size":".1 1 .5", "rgba": RGBA_COLOR},
       {"pos": "-3 2 .5", "euler": "0 0 -30", "size":".1 1 .5", "rgba": RGBA_COLOR},
       {"pos": "-3 -2 .5", "euler": "0 0 30", "size":".1 1 .5", "rgba": RGBA_COLOR},
       {"pos": "3 -2 .5", "euler": "0 0 -30", "size":".1 1 .5", "rgba": RGBA_COLOR},
       {"pos": "3 2 .5", "euler": "0 0 30", "size":".1 1 .5", "rgba": RGBA_COLOR},
       {"pos": "0 -4 .5", "euler": "0 0 90", "size":".1 1 .5", "rgba": RGBA_COLOR},
    ]
    return common._add_blocks(model, blocks)

def generate_square_gates(xml_string):
    model = mjcf.from_xml_string(xml_string, assets=dm_common.ASSETS)
    blocks = [
       {"pos": "-4 0 .5", "euler": "0 0 90", "size": "1.8 .1 .5", "rgba": RGBA_COLOR},
       {"pos": "4 0 .5", "euler": "0 0 90", "size": "1.8 .1 .5", "rgba": RGBA_COLOR},
       {"pos": "0 4 .5", "euler": "0 0 0", "size": "1.8 .1 .5", "rgba": RGBA_COLOR},
       {"pos": "0 -4 .5", "euler": "0 0 0", "size": "1.8 .1 .5", "rgba": RGBA_COLOR},
    ]
    return common._add_blocks(model, blocks)

def generate_dense_objects(xml_string):
    model = mjcf.from_xml_string(dm_common.read_model('quadruped.xml'), assets=dm_common.ASSETS)
    blocks = common._block_generator(20)
    return common.add_blocks(model, blocks), dm_common.ASSETS


class QuadrupedMove(suite.quadruped.Move):

    @staticmethod
    def _egocentric_velocity_and_actuators(physics):
        if not physics._hinge_names:
            [hinge_ids] = np.nonzero(physics.model.jnt_type ==
                                     enums.mjtJoint.mjJNT_HINGE)
            physics._hinge_names = [physics.model.id2name(j_id, 'joint')
                                 for j_id in hinge_ids]
        return np.hstack((physics.named.data.qvel[physics._hinge_names],
                          physics.data.act))

    def get_observation(self, physics):
        """Returns an observation of the state and the target position."""
        obs = collections.OrderedDict()
        obs['egocentric_state'] = physics.egocentric_state()
        obs['torso_velocity'] = physics.torso_velocity()
        obs['torso_upright'] = physics.torso_upright()
        obs['imu'] = physics.imu()
        obs['force_torque'] = physics.force_torque()

        obs['image_complement'] = np.concatenate([
            self._egocentric_velocity_and_actuators(physics),
            obs['torso_velocity'],
            obs['imu'],
            obs['force_torque']
        ])
        obs['control'] = physics.control()

        return obs

class TorsoPerturbedQuadrupedWalk(QuadrupedMove):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        physics.set_control(action)
        body_name = 'torso'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 200, 200, 200
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 0, 0])

class TorsoPerturbedQuadrupedRun(QuadrupedMove):
    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        physics.set_control(action)
        body_name = 'torso'
        idx = physics.model.name2id(body_name, "body")
        fx, fy, fz = 200, 200, 200
        physics.data.xfrc_applied[idx] = np.array([fx, fy, fz, 0, 0, 0])


class QuadrupedWalkLegDisabled(QuadrupedMove):

    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        assert len(action) == 12, action
        action[0:3] = 0.0
        physics.set_control(action)


def quadruped_walk(task=None, generator=None, time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  xml_string = make_model(generator, floor_size=_DEFAULT_TIME_LIMIT * _WALK_SPEED)
  physics = suite.quadruped.Physics.from_xml_string(xml_string, dm_common.ASSETS)
  if task is None:
     task = QuadrupedMove(desired_speed=_WALK_SPEED, random=random)
  else:
     task = task(desired_speed=_WALK_SPEED, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             control_timestep=_CONTROL_TIMESTEP,
                             **environment_kwargs)

def quadruped_run(task=None, generator=None, time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  xml_string = make_model(generator, floor_size=_DEFAULT_TIME_LIMIT * _RUN_SPEED)
  physics = suite.quadruped.Physics.from_xml_string(xml_string, dm_common.ASSETS)
  if task is None:
     task = QuadrupedMove(desired_speed=_RUN_SPEED, random=random)
  else:
     task = task(desired_speed=_RUN_SPEED, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             control_timestep=_CONTROL_TIMESTEP,
                             **environment_kwargs)

def quadruped_escape(task=None, generator=None, time_limit=_DEFAULT_TIME_LIMIT, random=None,
           environment_kwargs=None):
  raise NotImplementedError()
  # TO DO: task & generator for escape
  xml_string = make_model(generator, floor_size=40, terrain=True, rangefinders=True)
  physics = suite.quadruped.Physics.from_xml_string(xml_string, dm_common.ASSETS)
  task = suite.quadruped.Escape(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             control_timestep=_CONTROL_TIMESTEP,
                             **environment_kwargs)

def quadruped_fetch(task=None, generator=None, time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  # TO DO: task & generator for fetch
  raise NotImplementedError()
  xml_string = make_model(generator, walls_and_ball=True)
  physics = suite.quadruped.Physics.from_xml_string(xml_string, dm_common.ASSETS)
  task = suite.quadruped.Fetch(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             control_timestep=_CONTROL_TIMESTEP,
                             **environment_kwargs)