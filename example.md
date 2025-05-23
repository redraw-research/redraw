

# DMC Experiments (Example with Cup Catch)

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
--task dmcsim_cup_catch \
--run.script train_eval --run.eval_every 5e4 --run.eval_eps 100 --run.log_every 3000 --run.steps 4.5e6 \
--logdir logs/draw_cup_catch_pretraining_1
```
### DreamerV3
```bash
python dreamerv3/train.py --configs dmcsim_vision plan2explore rs1e7 \
--task dmcsim_cup_catch \
--run.script train_eval --run.eval_every 5e4 --run.eval_eps 100 --run.log_every 3000 --run.steps 4.5e6 \
--logdir logs/dreamer_cup_catch_pretraining_1
```

## Step 2: Transfer Learning
Expert-policy datasets for each target environment can be found in [assets/target_environment_datasets](assets/target_environment_datasets)

### ReDRAW
```bash
python dreamerv3/train.py --configs dmcsim_vision z_only_longer_horizon_preset no_post_stchprms freeze_wm ensemble_residual_extra_small_1_member 100x_wm_lr replay_size_4e4 \
--task dmcsim_cup_catch_windy \
--replay_directory_override assets/target_environment_datasets/dmcsim_cup_catch_windy/eval_replay \
--run.from_checkpoint logs/draw_cup_catch_pretraining_1/checkpoint.ckpt \
--run.script train_offline --freeze_replay True --run.eval_every 4000 --run.eval_eps 100 --run.log_every 4000 --run.steps 3e6 \
--logdir logs/redraw_cup_catch_windy_finetune_1
```

### DreamerV3 Finetune
```bash
python dreamerv3/train.py --configs dmcsim_vision freeze_encoder freeze_vanilla_heads no_reward_loss replay_size_4e4 \
--task dmcsim_cup_catch_windy \
--replay_directory_override assets/target_environment_datasets/dmcsim_cup_catch_windy/eval_replay \
--run.from_checkpoint logs/dreamer_cup_catch_pretraining_1/checkpoint.ckpt \
--run.script train_offline --freeze_replay True --run.eval_every 4000 --run.eval_eps 100 --run.log_every 4000 --run.steps 3e6 \
--logdir logs/dreamer_cup_catch_windy_finetune_1
```

### DRAW Finetune
```bash
python dreamerv3/train.py --configs dmcsim_vision z_only_longer_horizon_preset no_post_stchprms freeze_encoder freeze_vanilla_heads freeze_posterior zero_vanilla_head_loss no_rep_loss replay_size_4e4 \
--task dmcsim_cup_catch_windy \
--replay_directory_override assets/target_environment_datasets/dmcsim_cup_catch_windy/eval_replay \
--run.from_checkpoint logs/draw_cup_catch_pretraining_1/checkpoint.ckpt \
--run.script train_offline --freeze_replay True --run.eval_every 4000 --run.eval_eps 100 --run.log_every 4000 --run.steps 3e6 \
--logdir logs/draw_cup_catch_windy_finetune_1
```


