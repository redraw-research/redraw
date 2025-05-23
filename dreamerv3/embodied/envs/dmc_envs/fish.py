from dm_control import suite
from dm_control.rl import control
from dm_control.utils import containers
import numpy as np
import collections


_DEFAULT_TIME_LIMIT = 40
_CONTROL_TIMESTEP = .04
_JOINTS = ['tail1',
           'tail_twist',
           'tail2',
           'finright_roll',
           'finright_pitch',
           'finleft_roll',
           'finleft_pitch']


class Swim(suite.fish.Swim):
  """A Fish `Task` for swimming with smooth reward."""

  def get_observation(self, physics):
    """Returns an observation of joints, target direction and velocities."""
    obs = collections.OrderedDict()
    obs['joint_angles'] = physics.joint_angles()
    obs['upright'] = physics.upright()
    obs['target'] = physics.mouth_to_target()
    obs['velocity'] = physics.velocity()

    obs['image_complement'] = physics.velocity()
    return obs


def swim(task=None, generator=None, time_limit=_DEFAULT_TIME_LIMIT, random=None,
              environment_kwargs=None):
  """Returns the hard Turn task."""
  if generator is None:
    physics = suite.fish.Physics.from_xml_string(*suite.fish.get_model_and_assets())
  else:
      physics = suite.fish.Physics.from_xml_string(*generator())
  if task is None:
      task = Swim(random=random)
  else:
      task = task(random=random)

  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)
