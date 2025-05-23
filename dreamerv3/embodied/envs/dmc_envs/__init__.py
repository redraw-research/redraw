from dm_control.rl import control

from dreamerv3.embodied.envs.dmc_envs.cheetah import cheetah_run, DoubleActionMagnitudeCheetah, TorsoPerturbedCheetah, CheetahGTStateIsVectorObs, CheetahScaledUpImageComplement, CheetahOrigObs
from dreamerv3.embodied.envs.dmc import DMC
from dreamerv3.embodied.envs.dmc_envs.cartpole import cartpole_balance, BalanceBiasedDoubleActionMagnitude, BalanceBiasedDoubleActionMagnitude2
from dreamerv3.embodied.envs.dmc_envs.walker import walker_walk, DoubleActionMagnitudeWalkerWalk, DoubleActionMagnitudeNoForcesWalkerWalk, WalkerWalkReversedActions, WalkerWalkDampened, WalkerRelative, WalkerRelativeDoubleActionMagnitude
from dreamerv3.embodied.envs.dmc_envs.reacher import reacher_easy, reacher_hard, DoubleActionMagnitudeReacher, ReacherReversedActions
from dreamerv3.embodied.envs.dmc_envs.pendulum import pendulum_swingup, DoubleActionMagnitudePendulumSwingUp, SwingUpReversedActions, SwingUpDampened
from dreamerv3.embodied.envs.dmc_envs.quadruped import quadruped_walk, QuadrupedWalkLegDisabled
from dreamerv3.embodied.envs.dmc_envs.ball_in_cup import ball_in_cup_catch, HighGravityBallInCup, LowGravity0p5BallInCup, LowGravity2BallInCup, LowGravity1BallInCup, LowGravity0p25BallInCup, LowGravity0p1BallInCup, LowGravity0p05BallInCup, BallInCupWithWind
from dreamerv3.embodied.envs.dmc_envs.finger import turn_hard, Turn, TurnReversedActions, TurnTorqueApplied
from dreamerv3.embodied.envs.dmc_envs.hopper import hopper_hop, hopper_stand, Hopper, TorsoPerturbedHopperStand50, TorsoPerturbedHopperStand40, TorsoPerturbedHopperStand30, TorsoPerturbedHopperStand10, TorsoPerturbedHopperStand5, TorsoPerturbedHopperStand15, get_ice_hopper_model_and_assets
from dreamerv3.embodied.envs.dmc_envs.fish import swim, Swim

def get_dmc_env(task_name: str) -> control.Environment:
    if task_name == "cheetah_run":
        return cheetah_run()
    elif task_name == "cheetah_run_orig_obs":
        return cheetah_run(task=CheetahOrigObs)
    elif task_name == "cheetah_run_scaled_up_image_complement":
        return cheetah_run(task=CheetahScaledUpImageComplement)
    elif task_name == "cheetah_run_gt_state_is_vector_obs":
        return cheetah_run(task=CheetahGTStateIsVectorObs)
    elif task_name == "cheetah_run_double_action_magnitude":
        return cheetah_run(task=DoubleActionMagnitudeCheetah)
    elif task_name == "cheetah_run_perturbed_torso":
        return cheetah_run(task=TorsoPerturbedCheetah)

    elif task_name == "cartpole_balance":
        return cartpole_balance()
    elif task_name == "cartpole_balance_biased_double_action_magnitude":
        return cartpole_balance(task=BalanceBiasedDoubleActionMagnitude)
    elif task_name == "cartpole_balance_biased_double_action_magnitude2":
        return cartpole_balance(task=BalanceBiasedDoubleActionMagnitude2)

    elif task_name == "walker_walk":
        return walker_walk()
    elif task_name == "walker_walk_double_action_magnitude":
        return walker_walk(task=DoubleActionMagnitudeWalkerWalk)
    elif task_name == "walker_walk_dbl_action_mg_no_forces":
        return walker_walk(task=DoubleActionMagnitudeNoForcesWalkerWalk)
    elif task_name == "walker_walk_reversed_actions":
        return walker_walk(task=WalkerWalkReversedActions)
    elif task_name == "walker_walk_dampened":
        return walker_walk(task=WalkerWalkDampened)

    elif task_name == "walker_walk_relative":
        return walker_walk(task=WalkerRelative)
    elif task_name == "walker_walk_relative_double_action_magnitude":
        return walker_walk(task=WalkerRelativeDoubleActionMagnitude)

    elif task_name == "reacher_easy":
        return reacher_easy()
    elif task_name == "reacher_easy_double_action_magnitude":
        return reacher_easy(task=DoubleActionMagnitudeReacher)
    elif task_name == "reacher_easy_reversed_actions":
        return reacher_easy(task=ReacherReversedActions)

    elif task_name == "reacher_hard":
        return reacher_hard()
    elif task_name == "reacher_hard_double_action_magnitude":
        return reacher_hard(task=DoubleActionMagnitudeReacher)

    elif task_name == "pendulum_swingup":
        return pendulum_swingup()
    elif task_name == "pendulum_swingup_dampened":
        return pendulum_swingup(task=SwingUpDampened)
    elif task_name == "pendulum_swingup_double_action_magnitude":
        return pendulum_swingup(task=DoubleActionMagnitudePendulumSwingUp)
    elif task_name == "pendulum_swingup_reversed_actions":
        return pendulum_swingup(task=SwingUpReversedActions)

    elif task_name == "quadruped_walk":
        return quadruped_walk()
    elif task_name == "quadruped_walk_leg_disabled":
        return quadruped_walk(task=QuadrupedWalkLegDisabled)

    elif task_name == "cup_catch":
        return ball_in_cup_catch()
    elif task_name == "cup_catch_high_gravity":
        return ball_in_cup_catch(task=HighGravityBallInCup)
    elif task_name == "cup_catch_low_gravity_0p5":
        return ball_in_cup_catch(task=LowGravity0p5BallInCup)
    elif task_name == "cup_catch_low_gravity_0p25":
        return ball_in_cup_catch(task=LowGravity0p25BallInCup)
    elif task_name == "cup_catch_low_gravity_0p1":
        return ball_in_cup_catch(task=LowGravity0p1BallInCup)
    elif task_name == "cup_catch_low_gravity_0p05":
        return ball_in_cup_catch(task=LowGravity0p05BallInCup)
    elif task_name == "cup_catch_low_gravity_2":
        return ball_in_cup_catch(task=LowGravity2BallInCup)
    elif task_name == "cup_catch_low_gravity_1":
        return ball_in_cup_catch(task=LowGravity1BallInCup)

    elif task_name == "cup_catch_windy":
        return ball_in_cup_catch(task=BallInCupWithWind)

    elif task_name == "finger_turn_hard":
        return turn_hard()

    elif task_name == "finger_turn_hard_reversed_actions":
        return turn_hard(task=TurnReversedActions)

    elif task_name == "finger_turn_hard_torque_applied":
        return turn_hard(task=TurnTorqueApplied)

    elif task_name == "hopper_hop":
        return hopper_hop()

    elif task_name == "hopper_stand":
        return hopper_stand()

    elif task_name == "hopper_stand_ice":
        return hopper_stand(generator=get_ice_hopper_model_and_assets)

    elif task_name == "hopper_stand_ice_perturbed_10":
        return hopper_stand(generator=get_ice_hopper_model_and_assets, task=TorsoPerturbedHopperStand10)
    elif task_name == "hopper_stand_ice_perturbed_15":
        return hopper_stand(generator=get_ice_hopper_model_and_assets, task=TorsoPerturbedHopperStand15)

    elif task_name == "hopper_stand_perturbed_50":
        return hopper_stand(task=TorsoPerturbedHopperStand50)
    elif task_name == "hopper_stand_perturbed_40":
        return hopper_stand(task=TorsoPerturbedHopperStand40)
    elif task_name == "hopper_stand_perturbed_30":
        return hopper_stand(task=TorsoPerturbedHopperStand30)
    elif task_name == "hopper_stand_perturbed_10":
        return hopper_stand(task=TorsoPerturbedHopperStand10)
    elif task_name == "hopper_stand_perturbed_5":
        return hopper_stand(task=TorsoPerturbedHopperStand5)

    elif task_name == "fish_swim":
        return swim()

    elif task_name.startswith("orig_"):
        env = task_name.replace("orig_", "")

        domain, task = env.split('_', 1)
        if domain == 'cup':  # Only domain with multiple words.
            domain = 'ball_in_cup'
        if domain == 'manip':
            from dm_control import manipulation
            env = manipulation.load(task + '_vision')
        elif domain == 'locom':
            from dm_control.locomotion.examples import basic_rodent_2020
            env = getattr(basic_rodent_2020, task)()
        else:
            from dm_control import suite
            print(f"loading orig dmc domain: {domain}, task: {task} from task name: {task_name}")
            env = suite.load(domain, task)
        return env

    raise NotImplementedError(task_name)
