import os.path
import re
import time

from dreamerv3 import embodied
import numpy as np
from dreamerv3.xbox_controller_policy import XboxControllerPolicy
from dreamerv3.rosbag_recorder import ROSBagRecorder

def human_demo(agent, env, target_replay, logger, args):
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print('Logdir', logdir)
    should_log = embodied.when.Clock(args.log_every)
    step = logger.step
    metrics = embodied.Metrics()
    print('Observation space:', env.obs_space)
    print('Action space:', env.act_space)

    timer = embodied.Timer()
    timer.wrap('agent', agent, ['policy'])
    timer.wrap('env', env, ['step'])
    timer.wrap('logger', logger, ['write'])

    nonzeros = set()

    episode_steps = []

    episode_num = [0]
    # ros_bag_recorder = ROSBagRecorder()
    # rosbag_topics_list = [
    #      "/duckiebot5/camera_node/image_raw",
    #      "/duckiebot5/kinematics_node/velocity",
    #      "/duckiebot5/mdp_action"
    # ]

    def per_episode(ep):
        # ros_bag_recorder.stop()

        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        logger.add({'length': length, 'score': score}, prefix='episode')
        print(f'Episode has {length} steps and return {score:.1f}.')
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
        metrics.add(stats, prefix='stats')

        user_input = input(f"Step {step}, episode len {len(episode_steps)}: n to drop episode, anything else to add episode")

        if "n" in user_input:
            # ros_bag_recorder.drop_last_recorded_file()
            print("Dropped Episode")
        else:
            for trn in episode_steps:
                target_replay.add(trn)
            # ros_bag_recorder.commit_last_recorded_file()
            print("Added episode")

        episode_steps.clear()
        episode_num[0] += 1
        # ros_bag_recorder.start(topics_list=rosbag_topics_list,
        #                        filename=os.path.join(logdir, f"episode_{episode_num[0]}"))

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, worker: per_episode(ep))
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(lambda tran, _: episode_steps.append(tran))

    driver.on_step(lambda _, __: env.render())

    print('Start evaluation loop.')
    agent_nnet_policy = lambda *args: time.sleep(0.05)

    # call nnet policy in callback to have similar latency to nnet model
    policy_fn = XboxControllerPolicy(on_step_callback=agent_nnet_policy).policy

    input(f"Press Enter to start")    # ros_bag_recorder.start(topics_list=rosbag_topics_list,
    #                        filename=os.path.join(logdir, f"episode_{episode_num[0]}"))

    while step < args.steps:
        driver(policy_fn, episodes=1)
        if should_log(step):
            logger.add(metrics.result())
            logger.add(timer.stats(), prefix='timer')
            logger.write(fps=True)
    logger.write()
