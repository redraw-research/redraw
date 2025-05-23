import re
from functools import partial
import numpy as np

from dreamerv3 import embodied


def train(agent, env, replay, secondary_exp_replay, secondary_pred_state_replay, sim_query_worker_pool, logger, args):
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print('Logdir', logdir)
    should_expl = embodied.when.Until(args.expl_until)
    should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
    should_log = embodied.when.Clock(args.log_every)
    should_save = embodied.when.Every(args.save_every)
    should_sync = embodied.when.Every(args.sync_every)
    step = logger.step
    updates = embodied.Counter()
    metrics = embodied.Metrics()
    print('Observation space:', embodied.format(env.obs_space), sep='\n')
    print('Action space:', embodied.format(env.act_space), sep='\n')

    timer = embodied.Timer()
    timer.wrap('agent', agent, ['policy', 'train', 'train_alt', 'train_both', 'report', 'save'])
    timer.wrap('env', env, ['step'])
    timer.wrap('replay', replay, ['add', 'save'])
    if secondary_pred_state_replay:
        timer.wrap('secondary_replay', secondary_pred_state_replay, ['add', 'save'])
    timer.wrap('logger', logger, ['write'])

    nonzeros = set()

    def per_episode(ep):
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        sum_abs_reward = float(np.abs(ep['reward']).astype(np.float64).sum())
        logger.add({
            'length': length,
            'score': score,
            'sum_abs_reward': sum_abs_reward,
            'reward_rate': (np.abs(ep['reward']) >= 0.5).mean(),
        }, prefix='episode')
        # metaworld metrics
        if 'success' in ep:
            logger.add({'success': ep['success'].sum()}, prefix='episode')
        if 'task_num' in ep:
            logger.add({'task_num': ep['task_num'][0]}, prefix='episode')
        print(f'Episode has {length} steps and return {score:.1f}.')
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                print(f"video stat: {key} shape: {ep[key].shape}")
                if ep[key].shape[3] > 3:
                    # more channels than just RGB, maybe frame stacking
                    stats[f'policy_{key}'] = ep[key][:, :, :, :3]
                else:
                    stats[f'policy_{key}'] = ep[key]
                # stats[f'policy_{key}'] = ep[key]
        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                stats[f'sum_{key}'] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                stats[f'mean_{key}'] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                stats[f'max_{key}'] = ep[key].max(0).mean()
        metrics.add(stats, prefix='stats')

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, worker: per_episode(ep))
    driver.on_step(lambda tran, _: step.increment())
    def update_normalization(trn):
        with timer.scope("update_normalization"):
            agent.update_normalization_state(trn)
    driver.on_step(lambda trn, _: update_normalization(trn))
    driver.on_step(replay.add)

    driver.reset()

    print('Prefill train dataset.')
    random_agent = embodied.RandomAgent(env.act_space)
    # random_policy = lambda *args: agent.policy(*args, mode='random')
    while len(replay) < max(args.batch_steps, args.train_fill):
        driver(random_agent.policy, steps=100)
    logger.add(metrics.result())
    logger.write()

    train_alt_on_pred = agent.use_train_alt and (args.train_grounded_nets_on_world_model_train_pred_states or args.train_grounded_nets_on_imagined_rollout_pred_states)
    train_alt_on_exp = agent.use_train_alt and args.train_grounded_nets_on_experience
    should_draw_train_alt_exp_from_separate_replay = train_alt_on_exp and secondary_exp_replay is not None

    if args.perform_train_on_all_data_sources and (train_alt_on_pred or train_alt_on_exp):
        if train_alt_on_pred and train_alt_on_exp:
            dataset = agent.dataset_evenly_from_three_sources(replay.dataset,
                                                               secondary_pred_state_replay.dataset,
                                                               secondary_exp_replay.dataset)
        elif train_alt_on_exp and should_draw_train_alt_exp_from_separate_replay:
            dataset = agent.dataset_evenly_from_two_sources(replay.dataset, secondary_exp_replay.dataset)
        elif train_alt_on_pred:
            dataset = agent.dataset_evenly_from_two_sources(replay.dataset, secondary_pred_state_replay.dataset)
        else:
            dataset = agent.dataset(replay.dataset)
        env_only_data_set = agent.dataset(replay.dataset)
    else:
        dataset = agent.dataset(replay.dataset)
        env_only_data_set = dataset

    secondary_exp_dataset = agent.dataset(secondary_exp_replay.dataset) if secondary_exp_replay is not None else None
    secondary_pred_state_dataset = agent.dataset(secondary_pred_state_replay.dataset) if secondary_pred_state_replay is not None else None

    state = [None]  # To be writable from train step function below.
    batch = [None]
    train_alt_experience_batch = [None]
    train_alt_pred_state_batch = [None]

    def train_step(tran, worker):
        for _ in range(should_train(step)):

            if should_draw_train_alt_exp_from_separate_replay:
                if secondary_exp_dataset is not None:
                    with timer.scope('secondary_exp_dataset'):
                        train_alt_experience_batch[0] = next(secondary_exp_dataset)

            if train_alt_on_pred and len(secondary_pred_state_replay) >= args.batch_size:
                with timer.scope('secondary_pred_state_dataset'):
                    train_alt_pred_state_batch[0] = next(secondary_pred_state_dataset)

            has_all_required_train_alt_data = (train_alt_experience_batch[0] or not should_draw_train_alt_exp_from_separate_replay) and (train_alt_pred_state_batch[0] or not train_alt_on_pred)
            if agent.use_train_alt and has_all_required_train_alt_data:
                with timer.scope('dataset'):
                    batch[0] = next(dataset)
                if train_alt_on_exp and not should_draw_train_alt_exp_from_separate_replay:
                    train_alt_experience_batch[0] = batch[0]
                train_outs, state[0], train_metrics, train_alt_metrics = agent.train_both(
                    batch[0], train_alt_experience_batch[0], train_alt_pred_state_batch[0], state[0])
                metrics.add(train_alt_metrics, prefix='train_alt')
            else:
                with timer.scope('env_only_data_set'):
                    batch[0] = next(env_only_data_set)
                train_outs, state[0], train_metrics = agent.train(batch[0], state[0])
            metrics.add(train_metrics, prefix='train')

            if agent.use_train_alt:
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

                if train_alt_on_pred and len(secondary_pred_state_replay) < args.batch_size:
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
            logger.add(replay.stats, prefix='replay')
            if secondary_pred_state_replay:
                logger.add(secondary_pred_state_replay.stats, prefix='secondary_replay')
            if sim_query_worker_pool:
                logger.add(sim_query_worker_pool.stats, prefix='sim_query')
            logger.add(timer.stats(), prefix='timer')
            logger.write(fps=True)

    driver.on_step(train_step)

    checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
    timer.wrap('checkpoint', checkpoint, ['save', 'load'])
    checkpoint.step = step
    checkpoint.agent = agent
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint, skip_key_prefixes=["replay"])
    checkpoint.load_or_save()
    should_save(step)  # Register that we just saved.

    print('Start training loop.')
    policy = lambda *args: agent.policy(
        *args, mode='explore' if should_expl(step) else 'train')
    while step < args.steps:
        driver(policy, steps=100)
        if should_save(step):
            checkpoint.save()
    logger.write()
