from dm_control import suite
from dm_control import viewer

import numpy as np
import cheetah
import hopper
import walker
import humanoid
import quadruped
import ball_in_cup


#-----------------------------------------------------------
#                  Cheetah variations
#-----------------------------------------------------------
# env = cheetah.run(generator=cheetah.generate_pink_theme)
# env = cheetah.run(task=cheetah.TorsoPerturbedCheetah)

#-----------------------------------------------------------
#                  Hopper variations
#-----------------------------------------------------------

# env = hopper.stand(generator=hopper.generate_pink_theme)
# env = hopper.stand(task=hopper.TorsoPerturbedHopperStand)

# env = hopper.hop(hopper.generate_5_tall_blocks)
# env = hopper.hop(task=hopper.TorsoPerturbedHopperHop)


#-----------------------------------------------------------
#                  Walker variations
#-----------------------------------------------------------

# env = walker.stand(generator=walker.generate_pink_theme)
# env = walker.stand(task=walker.TorsoPerturbedWalkerStand)
# env = walker.walk(generator=walker.generate_1_short_block)
# env = walker.walk(task=walker.TorsoPerturbedWalkerWalk)
# env = walker.run(generator=walker.generate_1_tall_block)
# env = walker.run(task=walker.TorsoPerturbedWalkerRun)

#-----------------------------------------------------------
#                  Humanoid variations
#-----------------------------------------------------------
# env = humanoid.stand(generator=humanoid.generate_pink_theme)
# env = humanoid.stand(task=humanoid.TorsoPerturbedHumanoidStand)
# env = humanoid.walk(generator=humanoid.generate_closed_area)
# env = humanoid.walk(task=humanoid.TorsoPerturbedHumanoidWalk)
# env = humanoid.run(generator=humanoid.generate_dense_objects)
# env = humanoid.run(task=humanoid.TorsoPerturbedHumanoidRun)
# env = humanoid.run_pure_state(generator=humanoid.generate_hexagonal_gates)
# env = humanoid.run_pure_state(task=humanoid.TorsoPerturbedHumanoidRun)

#-----------------------------------------------------------
#                  Quadruped variations
#-----------------------------------------------------------
# env = quadruped.walk(generator=quadruped.generate_pink_theme)
# env = quadruped.walk(task=quadruped.TorsoPerturbedQuadrupedWalk)
# env = quadruped.run(generator=quadruped.generate_hexagonal_gates)
# env = quadruped.run(task=quadruped.TorsoPerturbedQuadrupedRun)

#-----------------------------------------------------------
#                  Ball In Cup variations
#-----------------------------------------------------------
# env = ball_in_cup.ball_in_cup_catch()
env = ball_in_cup.ball_in_cup_catch(task=ball_in_cup.DoubleGravityBallInCup)

action_spec = env.action_spec()

def random_policy(time_step):
  del time_step  # Unused.
  tmp = np.random.uniform(low=action_spec.minimum,
                           high=action_spec.maximum,
                           size=action_spec.shape)
  return tmp

viewer.launch(env, policy=random_policy)
