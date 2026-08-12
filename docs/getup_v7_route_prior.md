# V7 prone-route prior fine-tuning

This note records the controlled fine-tuning experiment that adds the reviewed
prone-to-kneel-to-crouch routes without replacing the general V6 recovery prior.

## Hypothesis

The V6 prior has broad recovery coverage but only weak density around the
reviewed prone route. Fine-tuning from V6 with route oversampling should lower
route validation loss while preserving general recovery validation loss.

This remains a local 0.2-second motion prior. Long-horizon route ordering still
requires phase-aware PPO observations, reset sampling, and rewards.

## Fixed inputs

- General data: `datasets/npz/getup_lafan6_sliced` (55,010 windows).
- Route data: `datasets/npz/getup_lafan_prone_routes_v7` (1,994 windows).
- Normalization: `datasets/norm_stats_getup_lafan6.npz`.
- Initialization: `datasets/pretrain_ckpt/pretrained_getup_lafan6_v6.pt`.
- V6 source checkpoint: epoch 1,600, `d_model=128`, two DiT blocks.
- Training mixture: 75% general and 25% route samples, sampled with replacement.
- General and route validation losses are deterministic and reported separately.
- Route splitting is by independent base motion: an original motion and its
  mirror always remain together. With seed 42, five routes (1,582 windows) are
  used for training and `fallAndGetUp1_subject5__recovery_009` plus its mirror
  (412 windows) are held out for validation.

The route subset has only six independent motions plus mirrors. It is therefore
oversampled within V6 rather than trained as a standalone prior.

## Formal command

```bash
uv run scripts/pretrain.py \
  --data-dir datasets/npz/getup_lafan6_sliced \
  --route-data-dir datasets/npz/getup_lafan_prone_routes_v7 \
  --route-train-fraction 0.25 \
  --norm-stats-file datasets/norm_stats_getup_lafan6.npz \
  --init-checkpoint datasets/pretrain_ckpt/pretrained_getup_lafan6_v6.pt \
  --d-model 128 \
  --nhead 4 \
  --num-layers 2 \
  --batch-size 1024 \
  --num-epochs 200 \
  --num-noise-samples 10 \
  --lr 0.00003 \
  --use-ema \
  --log-interval 5 \
  --save-interval 50 \
  --name pretrain-getup-lafan-route-v7-deterministic \
  --wandb-project smp \
  --device cuda
```

With the default `samples_per_epoch=0`, one epoch samples the combined number
of training windows. Checkpoints contain both the raw model and EMA weights;
SMP evaluation prefers EMA when it is present.

## Acceptance criteria

Select a checkpoint using both validation curves, not route loss alone:

1. Route validation loss improves materially from its initialization value.
2. General validation loss does not regress persistently by more than 5%.
3. The selected checkpoint is evaluated in PPO against the unchanged V6 prior.
4. PPO comparisons use fixed prone-reset seeds and report prone success,
   recovery time, peak joint speed/torque, and foot displacement.
5. If route validation continues down while general validation rises, select an
   earlier checkpoint instead of the final epoch.

The two primary ablations are:

- V6 PPO with the unchanged V6 prior.
- The same PPO configuration with the V7 mixed route prior.

A later phase-aware PPO change should be evaluated as a separate factor so the
effect of prior fine-tuning remains identifiable.

## Superseded diagnostic run

- Git commit: `0f0c2f2`.
- Host/worktree: `dsw-lyd2:/mnt/workspace/user/luyidan/smp-v6-prior`.
- W&B: <https://wandb.ai/tabletennis/smp/runs/gwn0p6j9>.
- Run ID: `gwn0p6j9`.
- Log: `run_control/v7_route_prior.log`.
- PID file: `run_control/v7_route_prior.pid`.
- Initial held-out metrics at epoch 0:
  `val/loss_general=0.121206`, `val/loss_route=0.142104`.

This run was stopped after discovering that validation sampled fresh diffusion
noise at every evaluation. It is not used for checkpoint selection.

## Deterministic formal run

- Git commit: `eeccf90`.
- W&B: <https://wandb.ai/tabletennis/smp/runs/08x2i674>.
- Run ID: `08x2i674`.
- Log: `run_control/v7_route_prior_deterministic.log`.
- PID file: `run_control/v7_route_prior_deterministic.pid`.
- Validation uses the same fixed diffusion timesteps and noise on every pass.
- Epoch 0: `val/loss_general=0.121798`,
  `val/loss_route=0.141733`.
- Epoch 5: `val/loss_general=0.121731`,
  `val/loss_route=0.141651`.

## Checkpoint selection

All saved checkpoints were re-evaluated with the same fixed validation noise.
Epoch 150 is selected instead of the final epoch:

| Checkpoint | General loss | Route loss | Mixed loss |
| --- | ---: | ---: | ---: |
| epoch 0 | 0.121798 | 0.141733 | 0.126781 |
| epoch 50 | 0.121275 | 0.140998 | 0.126206 |
| epoch 100 | 0.121031 | 0.140618 | 0.125928 |
| **epoch 150** | **0.120966** | **0.140498** | **0.125849** |
| epoch 199 | 0.121004 | 0.140545 | 0.125889 |

The selected EMA weights are exported without optimizer state to:

`datasets/pretrain_ckpt/pretrained_getup_lafan_route_v7.pt`

The runtime file retains selection metadata pointing to W&B run
[08x2i674](https://wandb.ai/tabletennis/smp/runs/08x2i674) and source
`checkpoint_00150.pt`. Record its SHA-256 here after the runtime file is
materialized on both the workstation and training host.

## Controlled PPO task

`Smp-Getup-Robust-Smooth-V7-Route-G1` inherits the complete V6 task and
changes exactly one environment field:

`events.init_smp_state.params.ckpt_path`

It therefore retains V6's 20-second episodes, procedural-reset distribution,
failure-state replay, push cohorts, rewards, observations, terminations, and
1000-iteration checkpoint interval. The word “Route” in the task name means
“route-prior replacement relative to full V6”; it does not mean the weaker
V6 prior-only environment ablation.

The PPO pilot resumes the complete V6 policy
[5n0d2i06](https://wandb.ai/tabletennis/smp/runs/5n0d2i06) from
`model_65996.pt`. First run a two-iteration, 64-environment smoke test:

```bash
uv run scripts/train.py Smp-Getup-Robust-Smooth-V7-Route-G1 \
  --env.scene.num-envs=64 \
  --agent.resume True \
  --wandb-run-path tabletennis/smp/5n0d2i06 \
  --wandb-checkpoint-name model_65996.pt \
  --agent.max-iterations=2 \
  --agent.save-interval=1000 \
  --logger tensorboard \
  --run-name v7_route_prior_smoke
```

After the smoke test passes, run a 4,000-iteration continuation:

```bash
uv run scripts/train.py Smp-Getup-Robust-Smooth-V7-Route-G1 \
  --env.scene.num-envs=4096 \
  --agent.resume True \
  --wandb-run-path tabletennis/smp/5n0d2i06 \
  --wandb-checkpoint-name model_65996.pt \
  --agent.algorithm.learning-rate=3e-4 \
  --agent.max-iterations=4000 \
  --agent.save-interval=1000 \
  --run-name smp_getup_v7_route_from_v6_65996
```

Do not add phase-aware rewards or observations to this run. Those changes form
a separate V7+phase ablation after the prior-only replacement is evaluated.
