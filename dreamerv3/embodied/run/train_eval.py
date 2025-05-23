import collections
import re

import numpy as np

from dreamerv3 import embodied
import jax

tree_map = jax.tree_util.tree_map


def _convert_from_devices(value, devices):
    value = jax.device_get(value)
    value = tree_map(np.asarray, value)
    if len(devices) > 1:
        value = tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), value)
    return value

def train_eval(
        agent, train_env, eval_env, train_replay, secondary_exp_train_replay, secondary_pred_state_train_replay, eval_replay,
        sim_query_worker_pool, logger, offline, args, use_wm_state_preds_for_sim_exploration, env_reset_initial_state_deque, use_residual, fault_tolerant_episodic_data_collection):
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print('Logdir', logdir)
    should_expl = embodied.when.Until(args.expl_until)
    should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps) if not offline else (lambda _: 1)
    should_log = embodied.when.Clock(args.log_every)
    should_save = embodied.when.Every(args.save_every)
    should_eval = embodied.when.Every(args.eval_every, args.eval_initial) if not offline else embodied.when.Clock(args.eval_every)
    should_sync = embodied.when.Every(args.sync_every)
    step = logger.step
    updates = embodied.Counter()
    metrics = embodied.Metrics()
    print('Observation space:', embodied.format(train_env.obs_space), sep='\n')
    print('Action space:', embodied.format(train_env.act_space), sep='\n')

    timer = embodied.Timer()
    timer.wrap('agent', agent, ['policy', 'train', 'train_alt', 'train_both', 'report', 'save'])
    timer.wrap('env', train_env, ['step'])
    if hasattr(train_replay, '_sample'):
        timer.wrap('replay', train_replay, ['_sample'])
    timer.wrap('logger', logger, ['write'])

    num_eval_episodes = max(len(eval_env), args.eval_eps)
    scores_from_current_evaluation = []
    nonzeros = set()

    checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt', parallel=False)
    max_avg_score = [None]

    def per_episode(ep, mode):
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        logger.add({
            'length': length,
            'score': score,
            'reward_rate': (ep['reward'] - ep['reward'].min() >= 0.1).mean(),
        }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
        print(f'Episode has {length} steps and return {score:.1f}.')

        if mode == 'eval':
            scores_from_current_evaluation.append(score)
            if len(scores_from_current_evaluation) >= num_eval_episodes:
                avg_score = np.mean(scores_from_current_evaluation)
                logger.add({
                    'avg_score': avg_score,
                    'num_episodes_this_eval': len(scores_from_current_evaluation)
                }, prefix='eval')
                scores_from_current_evaluation.clear()

                if not max_avg_score[0] or avg_score > max_avg_score[0]:
                    max_avg_score[0] = avg_score
                    if args.save_max_performance_checkpoint:
                        checkpoint.save(filename=logdir / f'checkpoint_max.ckpt')
                        with open(logdir / 'max_avg_score.txt', '+w') as f:
                            f.write(f"step {step}, max_avg_score: {max_avg_score[0]}")

        # metaworld metrics
        if 'success' in ep:
            logger.add({'success': ep['success'].sum()}, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
        if 'task_num' in ep:
            logger.add({'task_num': ep['task_num'][0]}, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                print(f"video stat: {key} shape: {ep[key].shape}")
                if ep[key].shape[3] > 3:
                    # more channels than just RGB, maybe frame stacking
                    stats[f'policy_{key}'] = ep[key][:, :, :, :3]
                else:
                    stats[f'policy_{key}'] = ep[key]
        for key, value in ep.items():
            if key == "log_environment_fault":
                stats['environment_fault_occurred'] = ep[key].any()
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                stats[f'sum_{key}'] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                stats[f'mean_{key}'] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                stats[f'max_{key}'] = ep[key].max(0).mean()
        metrics.add(stats, prefix=f'{mode}_stats')

    if offline:
        driver_train = embodied.Driver(None)
        if not use_residual:
            agent.update_normalization_state_from_replay_buffer(train_replay)
    else:
         # if use_wm_state_preds_for_sim_exploration:
         #    assert env_reset_initial_state_deque is None
         #    env_reset_initial_state_deque = collections.deque(maxlen=512)  #TODO UNCOMMENT THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        driver_train = embodied.Driver(train_env, env_reset_initial_state_deque=env_reset_initial_state_deque, static_reset_deque=True)
        driver_train.on_episode(lambda ep, worker: per_episode(ep, mode='train'))
        def update_normalization(trn):
            with timer.scope("update_normalization"):
                agent.update_normalization_state(trn)

        driver_train.on_step(lambda trn, _: update_normalization(trn))
        if use_wm_state_preds_for_sim_exploration:
            driver_train.on_step(secondary_pred_state_train_replay.add)
        else:
            if fault_tolerant_episodic_data_collection:
                print("Using fault-tolerant episodic data collection.")
                episode_step_buffers = collections.defaultdict(list)
                driver_train.on_step(lambda s, env_idx: episode_step_buffers[env_idx].append(s))
                def _add_or_drop_episode_steps(_, env_idx):
                    did_env_fault_occur_in_episode = False
                    for s in episode_step_buffers[env_idx]:
                        if s['log_environment_fault']:
                            did_env_fault_occur_in_episode = True
                    if not did_env_fault_occur_in_episode:
                        # dont save episode steps to replay buffer if the environment crashed during the episode
                        for s in episode_step_buffers[env_idx]:
                            train_replay.add(s, env_idx)
                    episode_step_buffers[env_idx].clear()

                driver_train.on_episode(_add_or_drop_episode_steps)
            else:
                driver_train.on_step(train_replay.add)
    driver_train.on_step(lambda tran, _: step.increment())

    driver_eval = embodied.Driver(eval_env)
    driver_eval.on_step(eval_replay.add)
    driver_eval.on_episode(lambda ep, worker: per_episode(ep, mode='eval'))

    driver_train.reset()
    driver_eval.reset()

    random_agent = embodied.RandomAgent(train_env.act_space)

    if not offline:
        print('Prefill train dataset.')
        fill_replay = secondary_pred_state_train_replay if use_wm_state_preds_for_sim_exploration else train_replay
        while len(fill_replay) < max(args.batch_steps, args.train_fill):
            driver_train(random_agent.policy, steps=100)

    print('Prefill eval dataset.')
    while len(eval_replay) < max(args.batch_steps, args.eval_fill):
        driver_eval(random_agent.policy, steps=100)

    logger.add(metrics.result())
    logger.write()

    train_alt_on_pred = agent.use_train_alt and (args.train_grounded_nets_on_world_model_train_pred_states or args.train_grounded_nets_on_imagined_rollout_pred_states)
    train_alt_on_exp = agent.use_train_alt and args.train_grounded_nets_on_experience
    should_draw_train_alt_exp_from_separate_replay = train_alt_on_exp and secondary_exp_train_replay is not None

    if agent.use_train_alt and args.perform_train_on_all_data_sources and (train_alt_on_pred or train_alt_on_exp):
        if train_alt_on_pred and train_alt_on_exp:
            dataset_train = agent.dataset_evenly_from_three_sources(train_replay.dataset,
                                                                               secondary_pred_state_train_replay.dataset,
                                                                               secondary_exp_train_replay.dataset)
        elif train_alt_on_exp and should_draw_train_alt_exp_from_separate_replay:
            dataset_train = agent.dataset_evenly_from_two_sources(train_replay.dataset, secondary_exp_train_replay.dataset)
        elif train_alt_on_pred:
            dataset_train = agent.dataset_evenly_from_two_sources(train_replay.dataset, secondary_pred_state_train_replay.dataset)
        else:
            dataset_train = agent.dataset(train_replay.dataset)
        env_only_data_set_train = agent.dataset(train_replay.dataset)
    else:
        dataset_train = agent.dataset(train_replay.dataset)
        env_only_data_set_train = dataset_train

    secondary_exp_train_dataset = agent.dataset(secondary_exp_train_replay.dataset) if secondary_exp_train_replay is not None else None
    secondary_pred_state_train_dataset = agent.dataset(secondary_pred_state_train_replay.dataset) if secondary_pred_state_train_replay is not None else None

    dataset_eval = agent.dataset(eval_replay.dataset)

    state = [None]  # To be writable from train step function below.
    batch = [None]
    train_alt_experience_batch = [None]
    train_alt_pred_state_batch = [None]

    def train_step(tran, worker):
        for _ in range(should_train(step)):
            if should_draw_train_alt_exp_from_separate_replay:
                with timer.scope('secondary_exp_dataset'):
                    train_alt_experience_batch[0] = next(secondary_exp_train_dataset)

            if train_alt_on_pred and len(secondary_pred_state_train_replay) >= args.batch_size:
                with timer.scope('secondary_pred_state_dataset'):
                    train_alt_pred_state_batch[0] = next(secondary_pred_state_train_dataset)

            has_all_required_train_alt_data = (train_alt_experience_batch[0] or not should_draw_train_alt_exp_from_separate_replay) and (train_alt_pred_state_batch[0] or not train_alt_on_pred)
            if agent.use_train_alt and has_all_required_train_alt_data:
                with timer.scope('dataset_train'):
                    batch[0] = next(dataset_train)
                if train_alt_on_exp and not should_draw_train_alt_exp_from_separate_replay:
                    train_alt_experience_batch[0] = batch[0]
                train_outs, state[0], train_metrics, train_alt_metrics = agent.train_both(
                    batch[0], train_alt_experience_batch[0], train_alt_pred_state_batch[0], state[0])
                metrics.add(train_alt_metrics, prefix='train_alt')
            else:
                with timer.scope('env_only_data_set_train'):
                    batch[0] = next(env_only_data_set_train)
                train_outs, state[0], train_metrics = agent.train(batch[0], state[0])
            metrics.add(train_metrics, prefix='train')

            if agent.use_train_alt:
                if use_wm_state_preds_for_sim_exploration:
                    pass
                    # symlog_grounded_states = _convert_from_devices(train_outs["wm_outs"]["post"]["symlog_grounded"], agent.train_devices)   #TODO UNCOMMENT THESE 3 LINES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # symlog_grounded_states = symlog_grounded_states.reshape(args.batch_steps, symlog_grounded_states.shape[-1])
                    # env_reset_initial_state_deque.extend(np.random.permutation(symlog_grounded_states))
                else:
                    if int(updates) % args.submit_every_nth_pred_state_batch_as_training_data == 0:
                        if args.train_grounded_nets_on_world_model_train_pred_states:
                            sim_query_worker_pool.submit_world_model_train_outs(batch[0], train_outs["wm_outs"],
                                                                                agent.train_devices)
                        if args.train_grounded_nets_on_imagined_rollout_pred_states:
                            task_behavior_traj = train_outs["task_behavior_traj"]
                            expl_behavior_traj = train_outs["expl_behavior_traj"]
                            if task_behavior_traj is not None:
                                sim_query_worker_pool.submit_imagined_trajectories(task_behavior_traj, agent.train_devices)
                            if expl_behavior_traj is not None:
                                sim_query_worker_pool.submit_imagined_trajectories(expl_behavior_traj, agent.train_devices)

                    if train_alt_on_pred and len(secondary_pred_state_train_replay) < args.batch_size:
                        print("Waiting for secondary_pred_state_train_replay to initially fill.")
                        sim_query_worker_pool.wait()
                        print("Filled.")
            updates.increment()

        if should_sync(updates):
            agent.sync()
        if should_log(step):
            agg = metrics.result()
            report = agent.report(batch[0], train_alt_pred_state_batch[0])
            non_compiled_report = agent.report_non_compiled(batch[0])
            print(f"report keys: {list(report.keys())}")
            print(f"non compiled report keys: {list(non_compiled_report.keys())}")

            report.update(non_compiled_report)
            report = {k: v for k, v in report.items() if 'train/' + k not in agg}
            logger.add(agg)
            logger.add(report, prefix='report')

            with timer.scope('dataset_eval'):
                eval_batch = next(dataset_eval)
            logger.add(agent.report(eval_batch, None), prefix='eval')

            logger.add(train_replay.stats, prefix='replay')
            if secondary_pred_state_train_replay:
                logger.add(secondary_pred_state_train_replay.stats, prefix='secondary_replay')
            if sim_query_worker_pool:
                logger.add(sim_query_worker_pool.stats, prefix='sim_query')
            logger.add(eval_replay.stats, prefix='eval_replay')
            logger.add(timer.stats(), prefix='timer')
            logger.write(fps=True)

    driver_train.on_step(train_step)

    timer.wrap('checkpoint', checkpoint, ['save', 'load'])
    checkpoint.step = step
    checkpoint.agent = agent
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint, skip_key_prefixes=["replay"])
        if offline:
            # reset step counter to zero when training offline,
            # we may have been counting env steps before; now we count updates
            # log the starting step first for debugging that the correct checkpoint was loaded
            with open(logdir / 'starting_checkpoint_step.txt', "w+") as f:
                f.write(str(int(step)))

            step.load(0)
            assert step == 0, step
    checkpoint.load_or_save()
    if offline:
        assert step == 0, step

    should_save(step)  # Register that we just saved.

    print('Start training loop.')
    policy_train = lambda *args: agent.policy(
        *args, mode='explore' if should_expl(step) else 'train')
    policy_eval = lambda *args: agent.policy(*args, mode='eval')
    while step < args.steps:
        if should_eval(step):
            print('Starting evaluation at step', int(step))
            with timer.scope('eval'):
                driver_eval.reset()
                driver_eval(policy_eval, episodes=num_eval_episodes)
        driver_train(policy_train, steps=100)
        if (offline or args.unique_checkpoints) and should_save(step):
            if args.unique_checkpoints:
                checkpoint.save(filename=logdir / f'checkpoint_{int(step)}.ckpt')
            else:
                checkpoint.save()
    if args.unique_checkpoints:
        checkpoint.save(filename=logdir / f'checkpoint_{int(step)}.ckpt')
    else:
        checkpoint.save()
    # checkpoint.save(filename=logdir / f'checkpoint_{int(step)}_done.ckpt')
    logger.write()
