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
  --name pretrain-getup-lafan-route-v7 \
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
