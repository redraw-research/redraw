
# Code for Adapting World Models with Latent-State Dynamics Residuals
### Revised code coming soon!

# Installation
```bash
git clone https://github.com/redraw-research/redraw.git
cd redraw
conda env create -f environment.yml
conda activate redraw311
pip install -e .
```


# DMC Experiments
(Also see [example.md](example.md) for example scripts with placeholder values filled in.)

## Step 0: Choose Environments
Choose `source environments` from:
```
1. dmcsim_cup_catch
2. dmcsim_finger_turn_hard
3. dmcsim_pendulum_swingup
4. dmcsim_reacher_easy
```
and matching `target environments` from:
```
1. dmcsim_cup_catch_windy
2. dmcsim_finger_turn_hard_torque_applied
3. dmcsim_pendulum_swingup_reversed_actions
4. dmcsim_reacher_easy_reversed_actions
```

## Step 1: Plan2Explore Pretraining
### DRAW
```bash
python dreamerv3/train.py --configs dmcsim_vision z_only_longer_horizon_preset no_post_stchprms draw_plan2explore \
--task <source environment> \
--run.script train_eval --run.eval_every 5e4 --run.eval_eps 100 --run.log_every 3000 --run.steps 4.5e6 \
--logdir <pretraining_experiment_logdir>
```
### DreamerV3
```bash
python dreamerv3/train.py --configs dmcsim_vision plan2explore rs1e7 \
--task <source environment> \
--run.script train_eval --run.eval_every 5e4 --run.eval_eps 100 --run.log_every 3000 --run.steps 4.5e6 \
--logdir <pretraining_experiment_logdir>
```

## Step 2: Transfer Learning
Expert-policy datasets for each target environment can be found in [assets/target_environment_datasets](assets/target_environment_datasets)

### ReDRAW
```bash
python dreamerv3/train.py --configs dmcsim_vision z_only_longer_horizon_preset no_post_stchprms freeze_wm ensemble_residual_extra_small_1_member 100x_wm_lr replay_size_4e4 \
--task <target environment> \
--replay_directory_override <target environment expert dataset directory>/eval_replay \
--run.from_checkpoint <pretraining_experiment_logdir>/checkpoint.ckpt \
--run.script train_offline --freeze_replay True --run.eval_every 4000 --run.eval_eps 100 --run.log_every 4000 --run.steps 3e6 \
--logdir <pretraining_experiment_logdir>
```

### DreamerV3 Finetune
```bash
python dreamerv3/train.py --configs dmcsim_vision freeze_encoder freeze_vanilla_heads no_reward_loss replay_size_4e4 \
--task <target environment> \
--replay_directory_override <target environment expert dataset directory>/eval_replay \
--run.from_checkpoint <pretraining_experiment_logdir>/checkpoint.ckpt \
--run.script train_offline --freeze_replay True --run.eval_every 4000 --run.eval_eps 100 --run.log_every 4000 --run.steps 3e6 \
--logdir <pretraining_experiment_logdir>
```

### DRAW Finetune
```bash
python dreamerv3/train.py --configs dmcsim_vision z_only_longer_horizon_preset no_post_stchprms freeze_encoder freeze_vanilla_heads freeze_posterior zero_vanilla_head_loss no_rep_loss replay_size_4e4 \
--task <target environment> \
--replay_directory_override <target environment expert dataset directory>/eval_replay \
--run.from_checkpoint <pretraining_experiment_logdir>/checkpoint.ckpt \
--run.script train_offline --freeze_replay True --run.eval_every 4000 --run.eval_eps 100 --run.log_every 4000 --run.steps 3e6 \
--logdir <pretraining_experiment_logdir>
```


