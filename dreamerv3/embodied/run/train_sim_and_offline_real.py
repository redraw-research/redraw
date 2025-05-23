import re

import numpy as np

from dreamerv3 import embodied


def train_sim_and_offline_real(
        agent, sim_train_env, eval_env,
        sim_train_replay, real_train_replay, secondary_exp_train_replay, secondary_pred_state_train_replay, eval_replay,
        sim_query_worker_pool, logger, sim_is_offline, args):

    # TODO, none of this file is implemented
    raise NotImplementedError("none of this file is implemented")

    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print('Logdir', logdir)
    should_expl = embodied.when.Until(args.expl_until)
    should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
    should_log = embodied.when.Clock(args.log_every)
    should_save = embodied.when.Clock(args.save_every)
    should_eval = embodied.when.Every(args.eval_every, args.eval_initial)
    should_sync = embodied.when.Every(args.sync_every)
    step = logger.step
    updates = embodied.Counter()
    metrics = embodied.Metrics()
    print('Observation space:', embodied.format(sim_train_env.obs_space), sep='\n')
    print('Action space:', embodied.format(sim_train_env.act_space), sep='\n')

    timer = embodied.Timer()
    timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
    timer.wrap('env', sim_train_env, ['step'])
    if hasattr(sim_train_replay, '_sample'):
        timer.wrap('sim_replay', sim_train_replay, ['_sample'])
    if hasattr(real_train_replay, '_sample'):
        timer.wrap('real_replay', real_train_replay, ['_sample'])
    timer.wrap('logger', logger, ['write'])

    num_eval_episodes = max(len(eval_env), args.eval_eps)
    scores_from_current_evaluation = []
    nonzeros = set()

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
                logger.add({
                    'avg_score': np.mean(scores_from_current_evaluation),
                    'num_episodes_this_eval': len(scores_from_current_evaluation)
                }, prefix='eval')
                scores_from_current_evaluation.clear()
        
        # metaworld metrics
        if 'success' in ep:
            logger.add({'success': ep['success'].sum()}, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
        if 'task_num' in ep:
            logger.add({'task_num': ep['task_num'][0]}, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f'policy_{key}'] = ep[key]
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
        metrics.add(stats, prefix=f'{mode}_stats')

    if sim_is_offline:
        driver_train = embodied.Driver(None)
    else:
        driver_train = embodied.Driver(sim_train_env)
        driver_train.on_episode(lambda ep, worker: per_episode(ep, mode='train'))
        driver_train.on_step(sim_train_replay.add)
    driver_train.on_step(lambda tran, _: step.increment())

    driver_eval = embodied.Driver(eval_env)
    driver_eval.on_step(eval_replay.add)
    driver_eval.on_episode(lambda ep, worker: per_episode(ep, mode='eval'))

    random_agent = embodied.RandomAgent(sim_train_env.act_space)

    if not sim_is_offline:
        print('Prefill sim train dataset.')
        while len(sim_train_replay) < max(args.batch_steps, args.train_fill):
            driver_train(random_agent.policy, steps=100)

    print('Prefill eval dataset.')
    while len(eval_replay) < max(args.batch_steps, args.eval_fill):
        driver_eval(random_agent.policy, steps=100)

    logger.add(metrics.result())
    logger.write()

    dataset_train = agent.dataset_evenly_from_two_sources(sim_train_replay, real_train_replay)

    secondary_exp_train_dataset = agent.dataset(secondary_exp_train_replay.dataset) if secondary_exp_train_replay is not None else None
    secondary_pred_state_train_dataset = agent.dataset(secondary_pred_state_train_replay.dataset) if secondary_pred_state_train_replay is not None else None

    dataset_eval = agent.dataset(eval_replay.dataset)

    state = [None]  # To be writable from train step function below.
    batch = [None]
    train_alt_experience_batch = [None]
    train_alt_pred_state_batch = [None]

    def train_step(tran, worker):
        for _ in range(should_train(step)):
            with timer.scope('dataset_train'):
                batch[0] = next(dataset_train)
            if agent.use_train_alt and len(secondary_pred_state_train_replay) >= args.batch_size:
                if secondary_exp_train_dataset is not None:
                    with timer.scope('secondary_exp_dataset'):
                        train_alt_experience_batch[0] = next(secondary_exp_train_dataset)
                else:
                    train_alt_experience_batch[0] = batch[0]
                with timer.scope('secondary_pred_state_dataset'):
                    train_alt_pred_state_batch[0] = next(secondary_pred_state_train_dataset)

                train_outs, state[0], train_metrics, train_alt_metrics = agent.train_both(
                    batch[0], train_alt_experience_batch[0], train_alt_pred_state_batch[0], state[0])
                metrics.add(train_alt_metrics, prefix='train_alt')
            else:
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

                if len(secondary_pred_state_train_replay) < args.batch_size:
                    sim_query_worker_pool.wait()
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

            logger.add(sim_train_replay.stats, prefix='sim_replay')
            logger.add(real_train_replay.stats, prefix='real_replay')

            if secondary_pred_state_train_replay:
                logger.add(secondary_pred_state_train_replay.stats, prefix='secondary_replay')
            if sim_query_worker_pool:
                logger.add(sim_query_worker_pool.stats, prefix='sim_query')
            logger.add(eval_replay.stats, prefix='eval_replay')
            logger.add(timer.stats(), prefix='timer')
            logger.write(fps=True)

    driver_train.on_step(train_step)

    checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
    timer.wrap('checkpoint', checkpoint, ['save', 'load'])
    checkpoint.step = step
    checkpoint.agent = agent
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint, skip_key_prefixes=["replay"])
    checkpoint.load_or_save()
    should_save(step)  # Register that we just saved.

    print('Start training loop.')
    policy_train = lambda *args: agent.policy(
        *args, mode='explore' if should_expl(step) else 'train')
    policy_eval = lambda *args: agent.policy(*args, mode='eval')
    while step < args.steps:
        if should_eval(step):
            print('Starting evaluation at step', int(step))
            driver_eval.reset()
            driver_eval(policy_eval, episodes=num_eval_episodes)
        driver_train(policy_train, steps=100)
        if should_save(step):
            checkpoint.save()
    logger.write()
