import importlib
import os
import ast
import pathlib
import sys
import warnings
import random
import collections
from datetime import datetime
from typing import Optional
from functools import partial as bind

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

import dreamerv3
from dreamerv3 import embodied
from dreamerv3.embodied import wrappers
from dreamerv3.sim_query_worker_pool import SimQueryWorkerPool, make_sim_model_env

def main(argv=None):
    import dreamerv3.agent as agt

    # Parse configs
    parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
    config = embodied.Config(agt.Agent.configs['defaults'])
    for name in parsed.configs:
        config = config.update(agt.Agent.configs[name])
    config = embodied.Flags(config).parse(other)
    print(config)
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length,
        batch_size=config.batch_size,
        submit_every_nth_pred_state_batch_as_training_data=config.submit_every_nth_pred_state_batch_as_training_data,
        train_grounded_nets_on_experience=config.train_grounded_nets_on_experience,
        train_grounded_nets_on_world_model_train_pred_states=config.train_grounded_nets_on_world_model_train_pred_states,
        train_grounded_nets_on_imagined_rollout_pred_states=config.train_grounded_nets_on_imagined_rollout_pred_states,
        perform_train_on_all_data_sources=config.perform_train_on_all_data_sources
    )

    # Set up logging
    if not args.logdir:
        now = datetime.now()  # current date and time
        date_time_str = now.strftime("%Y%m%d_%H%M%S")
        config_names_str = "_".join(name for name in parsed.configs)
        directory_name = f"{config_names_str}_{config.task}_{date_time_str}"
        parent_directory = os.getenv("DREAMER_LOGDIR")
        if not parent_directory:
            dreamer_module_path = os.path.abspath(dreamerv3.__file__)
            default_logs_directory = os.path.join(os.path.dirname(os.path.dirname(dreamer_module_path)), "logs")
            parent_directory = default_logs_directory
        logdir_str = os.path.join(parent_directory, directory_name)
        args = args.update(logdir=logdir_str)

    logdir = embodied.Path(args.logdir)
    print(f"Logdir: {str(logdir)}")
    logdir.mkdirs()

    with open(os.path.join(logdir, "configs.txt"), "+w") as f:
        # save the names of config presets used for quick reference
        f.write(" ".join(name for name in parsed.configs))

    config.save(logdir / 'config.yaml')
    step = embodied.Counter()
    logger = make_logger(parsed, logdir, step, config)

    if config.replay_directory_override:
        if not any(fname.endswith('.npz')
                   for fname in os.listdir(config.replay_directory_override)):
            raise FileNotFoundError(f"No .npz files were found in replay_directory_override: "
                                    f"{config.replay_directory_override}")

    if config.grounded_nets_experience_replay_directory_override:
        if not any(fname.endswith('.npz')
                   for fname in os.listdir(config.grounded_nets_experience_replay_directory_override)):
            raise FileNotFoundError(f"No .npz files were found in grounded_nets_experience_replay_directory_override: "
                                    f"{config.grounded_nets_experience_replay_directory_override}")

    cleanup = []
    try:
        if args.script in ['train', 'train_eval', 'train_offline', 'train_sim_eval_real']:
            offline = args.script == 'train_offline'
            use_residual = bool(config.rssm.residual)
            fault_tolerant_episodic_data_collection = bool(config.fault_tolerant_episodic_data_collection)
            assert use_residual or not offline, (use_residual, offline)

            # if offline and (config.normalize_agent_grounded_input or config.normalize_all_grounded_states or config.normalize_image_complement):
            #     raise NotImplementedError(
            #     "# TODO How should we update normalization for offline data? Also check if normalization stats are retained in checkpoints.")

            if config.replay_directory_override:
                replay_directory = config.replay_directory_override
            else:
                replay_directory = logdir / 'replay' if config.save_replay else None
            replay = make_replay(config, directory=replay_directory, include_in_each_sample={'is_real': True} if config.force_replay_as_real else None)
            if len(replay) == 0 and offline:
                raise ValueError("Replay buffer has to be prefilled if training offline.")
            elif len(replay) > 0 and config.save_replay:
                input(f"Are you sure you want to add data to the replay buffer at {replay_directory} ?\n"
                      f"(Press enter to continue, otherwise kill this process to stop.)")

            if config.reset_initial_states_from_buffer_directory:
                reset_replay = make_replay(config, directory=config.reset_initial_states_from_buffer_directory, capacity=config.reset_initial_states_replay_capacity)
                print("Filling env_reset_initial_state_deque from config.reset_initial_states_from_buffer_directory")
                env_reset_initial_state_deque = collections.deque(maxlen=int(config.reset_initial_states_replay_capacity))
                for step in reset_replay.load_steps_in_order():
                    env_reset_initial_state_deque.append(step['gt_state'])
                del reset_replay
            else:
                env_reset_initial_state_deque = None


            black_box_exp_replay = None
            black_box_pred_state_replay = None
            sim_query_worker_pool = None
            local_sim_query_env = None

            if config.use_grounded_rssm:
                black_box_pred_state_replay = make_replay(
                    config=config,
                    directory=logdir / 'secondary_replay' if config.save_replay else None,
                    is_pred_state=True)
                if config.grounded_nets_experience_replay_directory_override:
                    black_box_exp_replay = make_replay(
                        config=config,
                        capacity=1e24,  # always load entire grounded experience replay buffer
                        directory=config.grounded_nets_experience_replay_directory_override,
                        required_to_have_gt_state=True,
                        include_in_each_sample={'is_real': False}
                    )

                sim_query_worker_pool = SimQueryWorkerPool(config=config,
                                                           output_replay_buffer=black_box_pred_state_replay)
                local_sim_query_env = make_sim_model_env(config=config)
                cleanup.append(local_sim_query_env)

            if args.script == 'train':
                env = make_envs(config)
                cleanup.append(env)
                agent = agt.Agent(env.obs_space, env.act_space, step, config, local_sim_query_env)
                embodied.run.train(agent=agent, env=env, replay=replay,
                                   secondary_exp_replay=black_box_exp_replay,
                                   secondary_pred_state_replay=black_box_pred_state_replay,
                                   sim_query_worker_pool=sim_query_worker_pool,
                                   logger=logger,
                                   args=args)

            elif args.script in ['train_eval', 'train_offline', 'train_sim_eval_real']:
                use_wm_state_preds_for_sim_exploration = args.script == 'train_sim_eval_real'
                eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
                env = make_envs(config, use_sim_query_task=(args.script == 'train_sim_eval_real'))
                eval_env = make_envs(config)  # mode='eval'
                cleanup += [env, eval_env]
                agent = agt.Agent(env.obs_space, env.act_space, step, config, local_sim_query_env)
                embodied.run.train_eval(
                    agent=agent, train_env=env, eval_env=eval_env, train_replay=replay,
                    secondary_exp_train_replay=black_box_exp_replay,
                    secondary_pred_state_train_replay=black_box_pred_state_replay,
                    eval_replay=eval_replay,
                    sim_query_worker_pool=sim_query_worker_pool,
                    logger=logger,
                    offline=offline,
                    args=args,
                    use_wm_state_preds_for_sim_exploration=use_wm_state_preds_for_sim_exploration,
                    env_reset_initial_state_deque=env_reset_initial_state_deque,
                    use_residual=use_residual,
                    fault_tolerant_episodic_data_collection=fault_tolerant_episodic_data_collection
                )
            else:
                raise NotImplementedError(args.script)


        elif args.script == "human_demo":
            if config.replay_directory_override:
                replay_directory = config.replay_directory_override
            else:
                replay_directory = logdir / 'replay' if config.save_replay else None
            replay = make_replay(config, directory=replay_directory, include_in_each_sample={'is_real': True} if config.force_replay_as_real else None,
                                 chunks=401)

            if len(replay) > 0 and config.save_replay:
                input(f"Are you sure you want to add data to the replay buffer at {replay_directory} ?\n"
                      f"(Press enter to continue, otherwise kill this process to stop.)")
            env = make_envs(config)
            cleanup.append(env)
            agent = agt.Agent(env.obs_space, env.act_space, step, config, sim_query_env=None)
            embodied.run.human_demo(agent=agent, env=env, target_replay=replay, logger=logger, args=args)

        elif args.script == "human_demo_ros":
            if config.replay_directory_override:
                replay_directory = config.replay_directory_override
            else:
                replay_directory = logdir / 'replay' if config.save_replay else None
            replay = make_replay(config, directory=replay_directory, include_in_each_sample={'is_real': True} if config.force_replay_as_real else None,
                                 chunks=401)

            if len(replay) > 0 and config.save_replay:
                input(f"Are you sure you want to add data to the replay buffer at {replay_directory} ?\n"
                      f"(Press enter to continue, otherwise kill this process to stop.)")
            env = make_envs(config)
            cleanup.append(env)
            agent = agt.Agent(env.obs_space, env.act_space, step, config, sim_query_env=None)
            embodied.run.human_demo_ros(agent=agent, env=env, target_replay=replay, logger=logger, args=args)

        elif args.script in ["train_sim_and_offline_real", "train_offline_sim_and_offline_real"]:
            sim_is_offline = args.script == 'train_offline_sim_and_offline_real'

            if config.replay_directory_override:
                real_replay_directory = config.replay_directory_override
            else:
                real_replay_directory = logdir / 'real_replay' if config.save_replay else None
            real_replay = make_replay(config, directory=real_replay_directory,
                                      include_in_each_sample={"is_sim": False})
            if len(real_replay) == 0:
                raise ValueError("Real Replay buffer has to be prefilled if training sim offline.")
            elif config.save_replay:
                input(f"Are you sure you want to add data to the real data replay buffer at {real_replay_directory} ?\n"
                      f"(Press enter to continue, otherwise kill this process to stop.)")

            if config.sim_replay_directory_override:
                sim_replay_directory = config.sim_replay_directory_override
            else:
                sim_replay_directory = logdir / 'sim_replay' if config.save_replay else None
            sim_replay = make_replay(config, directory=sim_replay_directory,
                                     include_in_each_sample={"is_sim": True})
            if len(sim_replay) == 0 and sim_is_offline:
                raise ValueError("Sim Replay buffer has to be prefilled if training sim offline.")
            elif len(sim_replay) > 0 and config.save_replay:
                input(f"Are you sure you want to add data to the sim data replay buffer at {sim_replay_directory} ?\n"
                      f"(Press enter to continue, otherwise kill this process to stop.)")

            black_box_exp_replay = None
            black_box_pred_state_replay = None
            sim_query_worker_pool = None
            local_sim_query_env = None

            if config.use_grounded_rssm:
                black_box_pred_state_replay = make_replay(
                    config=config,
                    directory=logdir / 'secondary_replay' if config.save_replay else None,
                    is_pred_state=True)
                if config.grounded_nets_experience_replay_directory_override:
                    black_box_exp_replay = make_replay(
                        config=config,
                        directory=config.grounded_nets_experience_replay_directory_override,
                        required_to_have_gt_state=True)

                sim_query_worker_pool = SimQueryWorkerPool(config=config,
                                                           output_replay_buffer=black_box_pred_state_replay)
                local_sim_query_env = make_sim_model_env(config=config)
                cleanup.append(local_sim_query_env)

            eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
            sim_train_env = make_sim_model_env(config=config)
            eval_env = make_envs(config)  # mode='eval'
            cleanup += [sim_train_env, eval_env]
            agent = agt.Agent(sim_train_env.obs_space, sim_train_env.act_space, step, config, local_sim_query_env)
            embodied.run.train_sim_and_offline_real(
                agent=agent, sim_train_env=sim_train_env, eval_env=eval_env,
                sim_train_replay=sim_replay, real_train_replay=real_replay,
                secondary_exp_train_replay=black_box_exp_replay,
                secondary_pred_state_train_replay=black_box_pred_state_replay,
                eval_replay=eval_replay,
                sim_query_worker_pool=sim_query_worker_pool,
                logger=logger,
                sim_is_offline=sim_is_offline,
                args=args)

        elif args.script == 'train_save':
            if config.use_grounded_rssm:
                raise NotImplementedError("This script hasn't been modified to work with SimDreamer yet.")
            replay = make_replay(config, logdir / 'replay')
            env = make_envs(config)
            cleanup.append(env)
            agent = agt.Agent(env.obs_space, env.act_space, step, config)
            embodied.run.train_save(agent, env, replay, logger, args)

        elif args.script == 'train_holdout':
            if config.use_grounded_rssm:
                raise NotImplementedError("This script hasn't been modified to work with SimDreamer yet")
            replay = make_replay(config, logdir / 'replay')
            if config.eval_dir:
                assert not config.train.eval_fill
                eval_replay = make_replay(config, config.eval_dir, is_eval=True)
            else:
                assert 0 < args.eval_fill <= config.replay_size // 10, args.eval_fill
                eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
            env = make_envs(config)
            cleanup.append(env)
            agent = agt.Agent(env.obs_space, env.act_space, step, config)
            embodied.run.train_holdout(
                agent, env, replay, eval_replay, logger, args)

        elif args.script == 'eval_only':
            if config.use_grounded_rssm:
                raise NotImplementedError("This script hasn't been modified to work with SimDreamer yet")
            env = make_envs(config)  # mode='eval'
            cleanup.append(env)
            agent = agt.Agent(env.obs_space, env.act_space, step, config)
            embodied.run.eval_only(agent, env, logger, args)

        elif args.script == 'eval_only_ros_record':
            if config.use_grounded_rssm:
                raise NotImplementedError("This script hasn't been modified to work with SimDreamer yet")
            env = make_envs(config)  # mode='eval'
            cleanup.append(env)
            agent = agt.Agent(env.obs_space, env.act_space, step, config)
            embodied.run.eval_only_ros_record(agent, env, logger, args)

        elif args.script == 'eval_only_gen_replay':
            if config.use_grounded_rssm:
                raise NotImplementedError("This script hasn't been modified to work with SimDreamer yet")
            env = make_envs(config)  # mode='eval'
            cleanup.append(env)
            agent = agt.Agent(env.obs_space, env.act_space, step, config)
            eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
            embodied.run.eval_only_gen_replay(agent, env, eval_replay, logger, args)
        
        elif args.script == 'parallel':
            if config.use_grounded_rssm:
                raise NotImplementedError("This script hasn't been modified to work with SimDreamer yet")
            assert config.run.actor_batch <= config.envs.amount, (
                config.run.actor_batch, config.envs.amount)
            step = embodied.Counter()
            env = make_env(config)
            agent = agt.Agent(env.obs_space, env.act_space, step, config)
            env.close()
            replay = make_replay(config, logdir / 'replay', rate_limit=True)
            embodied.run.parallel(
                agent, replay, logger, bind(make_env, config),
                num_envs=config.envs.amount, args=args)

        else:
            raise NotImplementedError(args.script)
    finally:
        for obj in cleanup:
            obj.close()


def make_logger(parsed, logdir, step, config):
    multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
    if config.run.script == "train_offline":
        multiplier = 1

    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(config.filter, name=logdir),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score'),
        embodied.logger.TensorBoardOutput(logdir),
        # embodied.logger.WandBOutput(logdir.name, config),
        # embodied.logger.MLFlowOutput(logdir.name),
    ], multiplier)
    return logger


def make_replay(
        config, directory=None, is_eval=False, is_pred_state=False, rate_limit=False, capacity=None,
        required_to_have_gt_state=False, include_in_each_sample: Optional[dict] = None, chunks=1024, **kwargs):
    assert config.replay == 'uniform' or not rate_limit
    length = config.batch_length
    if capacity:
        size = capacity
    elif is_pred_state:
        size = config.secondary_replay_capacity
    elif is_eval:
        size = max(config.replay_size // 10, config.batch_size * config.batch_length)
    else:
        size = config.replay_size

    omit_gt_state = config.omit_gt_state_from_replays_where_optional and not required_to_have_gt_state

    if config.replay == 'uniform' or is_eval:
        kw = {'online': config.replay_online}
        if rate_limit and config.run.train_ratio > 0:
            assert False, f"debugging assert, ok to remove, rate_limit = {rate_limit}, config.run.train_ratio = {config.run.train_ratio}"
            kw['samples_per_insert'] = config.run.train_ratio / config.batch_length
            kw['tolerance'] = 10 * config.batch_size
            kw['min_size'] = config.batch_size

        can_save = (not is_eval and not is_pred_state and config.save_replay) or (is_eval and config.save_eval_replay)
        replay = embodied.replay.Uniform(length, size, directory,
                                         can_ever_add=is_eval or is_pred_state or not config.freeze_replay,
                                         can_save=can_save,
                                         omit_gt_state=omit_gt_state,
                                         include_in_each_sample=include_in_each_sample,
                                         load_earliest_first=not is_eval and not is_pred_state and config.load_earliest_replay_steps_first,
                                         augment_images=config.apply_image_augmentations,
                                         augment_image_workers=config.augment_image_workers,
                                         debug_reverse_added_actions=config.reverse_actions_added_to_replay,
                                         save_framestacked_image_as_image=config.save_framestacked_image_as_image,
                                         chunks=chunks,
                                         **kw)
    # elif config.replay == 'reverb':
    #     replay = embodied.replay.Reverb(length, size, directory)
    # elif config.replay == 'chunks':
    #     replay = embodied.replay.NaiveChunks(length, size, directory)
    else:
        raise NotImplementedError(config.replay)
    return replay


def make_envs(config, use_sim_query_task=False, **overrides):
    suite, task = config.task.split('_', 1)
    ctors = []
    for index in range(config.envs.amount):
        if use_sim_query_task:
            ctor = lambda: make_sim_model_env(config, **overrides)
        else:
            ctor = lambda: make_env(config, **overrides)
        if config.envs.parallel != 'none':
            ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
        if config.envs.restart:
            ctor = bind(wrappers.RestartOnException, ctor)
        ctors.append(ctor)
    envs = [ctor() for ctor in ctors]
    return embodied.BatchEnv(envs, parallel=(config.envs.parallel != 'none'))


def make_env(config, **overrides):
    # You can add custom environments by creating and returning the environment
    # instance here. Environments with different interfaces can be converted
    # using `embodied.envs.from_gym.FromGym` and `embodied.envs.from_dm.FromDM`.
    suite, task = config.task.split('_', 1)
    if suite == "metaworld":
        task = f"{'-'.join(task.split('_'))}-v2"
    ctor = {
        'dummy': 'embodied.envs.dummy:Dummy',
        'gym': 'embodied.envs.from_gym:FromGym',
        'dm': 'embodied.envs.from_dmenv:FromDM',
        'crafter': 'embodied.envs.crafter:Crafter',
        'dmc': 'embodied.envs.dmc:DMC',
        'atari': 'embodied.envs.atari:Atari',
        'dmlab': 'embodied.envs.dmlab:DMLab',
        'minecraft': 'embodied.envs.minecraft:Minecraft',
        'loconav': 'embodied.envs.loconav:LocoNav',
        'pinpad': 'embodied.envs.pinpad:PinPad',
        'metaworld': 'embodied.envs.mw:MetaWorldML1',
        'gridworld': 'embodied.envs.gridworld:SimDreamerGridWorldEnv',
        'dmcsim': 'embodied.envs.dmc_simulation:DMCSimulationEnv',
        'dmcmjxsim': 'embodied.envs.dmc_mjx_simulation:DMCMjxSimulationEnv',
        'duckiebotssim': 'embodied.envs.duckiebots_sim:DreamerUELaneFollowingEnv',
        'duckiebotsreal': 'embodied.envs.duckiebots_real:DreamerRealLaneFollowingEnv'
    }[suite]
    if isinstance(ctor, str):
        module, cls = ctor.split(':')
        module = importlib.import_module(module)
        ctor = getattr(module, cls)
    kwargs = config.env.get(suite, {})
    kwargs.update(overrides)
    env = ctor(task, **kwargs)
    return wrap_env(env, config)


def wrap_env(env, config):
    args = config.wrapper
    for name, space in env.act_space.items():
        if name in ['reset', 'reset_state', 'reset_to_state']:
            continue
        elif space.discrete:
            env = wrappers.OneHotAction(env, name)
        elif args.discretize:
            env = wrappers.DiscretizeAction(env, name, args.discretize)
        else:
            env = wrappers.NormalizeAction(env, name)
    env = wrappers.ExpandScalars(env)
    if args.length:
        env = wrappers.TimeLimit(env, args.length, args.reset)
    if args.checks:
        env = wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete and name != 'reset_state':
            env = wrappers.ClipAction(env, name)
    if args.framestack_image > 1:
        if args.framestack_only_image:
            env = wrappers.FrameStackOnlyImage(env, stack_size=args.framestack_image, return_as_framestacked_image=args.return_as_framestacked_image)
        else:
            env = wrappers.FrameStackImage(env, stack_size=args.framestack_image)

    if args.repeat_after_framestack > 1:
        env = wrappers.ActionRepeat(env=env, repeat=args.repeat_after_framestack, sum_intra_step_rewards=args.sum_rewards_between_action_repeat)

    return env


if __name__ == '__main__':
    main()
