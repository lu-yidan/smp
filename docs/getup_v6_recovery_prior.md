# V6 multi-recovery SMP prior

This note records the data and training plan for the V6 recovery prior. It is
separate from the V5 PPO policy so the old baseline remains reproducible.
For a practical explanation of raw CSV trajectories, NPZ windows, the prior,
and interactive MuJoCo data playback, see
[`motion_data_and_prior.md`](motion_data_and_prior.md).

## What the prior does

`datasets/pretrain_ckpt/pretrained_getup_f2s2.pt` is a frozen diffusion
denoiser used by SMP. It assigns a natural-motion score to the policy's most
recent 10 frames (0.2 s at 50 Hz), and its data distribution also supplies GSI
reset states. It is not the PPO controller and cannot itself make the robot
recover. PPO learns recovery from resets, contacts, pushes, and task rewards;
the prior biases those solutions toward motions represented in its dataset.

The old checkpoint was trained only from `datasets/npz/getup_f2s2/`, derived
primarily from `fallAndGetUp2_subject2.csv`. V6 trains a new checkpoint from
six LAFAN recovery sequences. The old file is never overwritten and remains the
V2--V5 baseline.

## Source data

| Source | SHA-256 | Frames | Recoveries |
| --- | --- | ---: | ---: |
| `fallAndGetUp1_subject1.csv` | `97e98a9cac5150917bc5a2ca3517759a3f3c380582cde5e92ad663c1e0553e12` | 5047 | 19 |
| `fallAndGetUp1_subject4.csv` | `5e9cf9acddc62a8646151567fffac09c7e088186d87a9a8543743cac517ddf0b` | 5047 | 11 |
| `fallAndGetUp1_subject5.csv` | `55a22ca723f26170aae8529ccb26dbda9bf2cb816b4f29eed3c57260dadb05ca` | 5047 | 19 |
| `fallAndGetUp2_subject2.csv` | `49ae9726ae176b10b1213b441535946ffcca93586e1091e8398e239f4a049a47` | 4918 | 14 |
| `fallAndGetUp2_subject3.csv` | `fa6e33725a450789160c8f3220b0a5e9001798f023fd14e3ff6c230a0829d75d` | 4918 | 13 |
| `fallAndGetUp3_subject1.csv` | `fc00701411a270a9ab61c321ba836ba36437c61630c65f8fe276ac851f23ccd7` | 3066 | 12 |

The source CSVs are external data and generated CSV/NPZ files are gitignored.
They must be transferred separately to the training host.

## Deterministic slicing

`scripts/slice_recovery_motions.py` detects a sustained fallen interval from
root height/orientation and pairs it only with a sustained stand beginning no
more than 3 s after leaving the fallen set. This rejects unrelated falls that
were previously paired with standing 18--30 s later. Clips retain 0.35 s before
the detected fall and 1.0 s after standing, and are capped at 12 s.

Each clip receives a sagittal mirror. The augmentation swaps left/right joints,
changes roll/yaw signs, mirrors root y, and transforms the root quaternion.
Algebraic involution and quaternion norms are checked for every source. A
MuJoCo FK audit over 66 sampled poses measured a maximum left/right link-position
symmetry error of `1.01e-5 m`.

The final dataset contains 88 original recoveries and 176 clips after mirroring:
34,096 CSV frames and 55,010 windows of shape `(10, 59)` after interpolation
to 50 Hz. There are no NaN or Inf values. Original lying-orientation counts are
9 roll-negative, 15 roll-positive, 24 pitch-positive, and 40 pitch-negative.

```bash
uv run scripts/slice_recovery_motions.py \
  --input-dir /path/to/LAFAN1_Retargeting_Dataset/g1 \
  --output-dir datasets/csv/getup_lafan6_sliced

uv run scripts/csv_to_npz.py \
  --input-dir datasets/csv/getup_lafan6_sliced \
  --output-dir datasets/npz/getup_lafan6_sliced \
  --device cuda

uv run scripts/compute_norm_stats.py \
  --input-dir datasets/npz/getup_lafan6_sliced \
  --output datasets/norm_stats_getup_lafan6.npz
```

## Prior training

V6 keeps the old architecture (128-wide, two layers, 50 diffusion steps) so the
comparison changes the motion distribution rather than model capacity. With
49,509 training windows, 1,600 epochs are about 77,600 optimizer steps, close to
the old one-sequence run's estimated update budget. Only five periodic/final
checkpoints are retained.

```bash
uv run scripts/pretrain.py \
  --data-dir datasets/npz/getup_lafan6_sliced \
  --norm-stats-file datasets/norm_stats_getup_lafan6.npz \
  --train-split 0.9 \
  --d-model 128 \
  --num-epochs 1600 \
  --batch-size 1024 \
  --num-noise-samples 10 \
  --save-interval 400 \
  --name pretrain-getup-lafan6-v6 \
  --device cuda
```

A local one-epoch smoke test completed with 698,674 parameters, train loss
0.756905, and validation loss 0.606289. The random 10% validation split is only
a training-health signal because adjacent windows and mirrored clips are
correlated; paper results must use subject-held-out recovery evaluation.

After training, copy the final checkpoint to a new immutable name such as
`datasets/pretrain_ckpt/pretrained_getup_lafan6_v6.pt`. V6 PPO will resume
from V5 `model_61997.pt` with this new prior. Reset expansion, failure-state
replay, and stratified pushes are separate PPO changes and should be ablated
against the prior-only change.

## Formal run

The formal prior run was launched on 2026-08-10:

- W&B: <https://wandb.ai/tabletennis/smp/runs/1bui8qxw>
- branch: `codex/getup-v6-lafan-recovery-prior`
- launch commit: `82f39c5`
- server: `dsw-lyd2`, physical GPU 0 (RTX PRO 5000 72 GB)
- worktree: `/mnt/workspace/user/luyidan/smp-v6-prior`
- launcher / Python PID at launch: `76259` / `76263`
- process record: `run_control/v6_prior_train.pid`
- console log: `run_control/v6_prior_train.log`

The server independently verified all 176 NPZ files, 55,010 windows, no
non-finite values, and CUDA availability before launch. Epoch 0 reported train
loss 0.756864 and validation loss 0.607092; epoch 10 reported 0.228779 and
0.232373. These are startup health checks, not final checkpoint selection.

## PPO ablation tasks

Two registered tasks isolate the contribution of the new prior from the harder
state distribution:

| Task | Change from V5 |
| --- | --- |
| `Smp-Getup-Robust-Smooth-V6-Prior-G1` | only replace the F2S2 prior with the LAFAN6 prior |
| `Smp-Getup-Robust-Smooth-V6-G1` | new prior plus expanded resets, failure replay, and stratified pushes |

Full V6 uses a 20 s episode. Before replay warms up, resets are 35% GSI from
the new prior and 65% procedural. Procedural poses retain the four semantic
front/back/side modes but add up to 0.40 rad noise on both roll and pitch,
0.38--0.64 m root height, 0.24 rad joint noise, 0.18 m XY offsets, 0.25 m/s root
linear velocity, and 0.60 rad/s root angular velocity. This creates continuous
oblique contact states instead of only four exact 90-degree orientations.

The training-only failure recorder stores up to 8,192 GPU-resident simulator
states after recovery progress stagnates for 75 control steps (1.5 s). Once 128
states exist, 20% of resets are replaced by replay samples. The replay ring
persists across episodes; per-environment stagnation counters do not. Playback
does not record or sample the ring.

Pushes are assigned per episode rather than uniformly:

- 25% clean: no automatic knockdown;
- 50% standard: at most one 80--170 N post-stand knockdown;
- 25% intensive: up to three 120--230 N post-recovery knockdowns.

The existing `--auto-disturbances` playback switch still controls these events.
Metrics separately report replay resets, replay-ring fill, push cohort, active
wrench, and push count.

The implementation passed three local checks with the known-good old prior as
a temporary placeholder:

- 8 environments completed one full PPO rollout/update;
- 1,024 environments completed six iterations, populated 3.76% of the 8,192
  replay slots, and exercised the push scheduler without NaN/Inf;
- a targeted 256-environment test recorded and replayed all 256 hard states,
  verified finite robot state, and scheduled a second intensive-cohort push in
  every environment.

The one-epoch new-prior smoke checkpoint was intentionally not used for these
physics checks: it produced unstable GSI samples and actor-observation NaNs.
Formal V6 PPO must wait for the converged prior and pass a GSI smoke test first.

After the prior passes validation, launch both ablations from exactly the same
V5 `model_61997.pt` with a lower refinement learning rate:

```bash
CUDA_VISIBLE_DEVICES=1 uv run scripts/train.py \
  Smp-Getup-Robust-Smooth-V6-Prior-G1 \
  --env.scene.num-envs=4096 \
  --agent.resume True \
  --wandb-run-path tabletennis/smp/pkduffcs \
  --wandb-checkpoint-name model_61997.pt \
  --agent.algorithm.learning-rate=3e-4 \
  --agent.max-iterations=4000 \
  --agent.save-interval=1000

CUDA_VISIBLE_DEVICES=2 uv run scripts/train.py \
  Smp-Getup-Robust-Smooth-V6-G1 \
  --env.scene.num-envs=4096 \
  --agent.resume True \
  --wandb-run-path tabletennis/smp/pkduffcs \
  --wandb-checkpoint-name model_61997.pt \
  --agent.algorithm.learning-rate=3e-4 \
  --agent.max-iterations=4000 \
  --agent.save-interval=1000
```

## Completed prior and V6 PPO launch

The formal LAFAN6 prior completed all 1,600 epochs on 2026-08-10. The final
checkpoint is stored on `dsw-lyd2` at
`logs/pretrain/pretrain-getup-lafan6-v6/20260810_171406/pretrained.pt` and was
copied to the immutable V6 runtime path
`datasets/pretrain_ckpt/pretrained_getup_lafan6_v6.pt`. Both files have SHA-256
`909360b5d8ede4370292facd95a657fb758eaeacfb4458f85637f998a91b61f3`.
The last periodic W&B point (epoch 1,590) reported train loss 0.120836 and
validation loss 0.121656. The final `checkpoint_01599.pt` was also saved.

A 64-environment, two-iteration `Smp-Getup-Robust-Smooth-V6-G1` smoke test
completed without NaN/Inf before the formal PPO runs were launched. Its local
run directory is
`logs/rsl_rl/smp_getup_v6_g1/2026-08-11_17-02-51_v6_final_prior_smoke2`.

Both 4,096-environment ablations were launched from commit `d514668` and the
same V5 `model_61997.pt` checkpoint on 2026-08-11:

| Task | GPU | W&B run | Runtime log |
| --- | --- | --- | --- |
| prior-only | 4 | [xxo9ip1o](https://wandb.ai/tabletennis/smp/runs/xxo9ip1o) | `run_control/v6_prior_ppo.log` |
| full V6 | 5 | [5n0d2i06](https://wandb.ai/tabletennis/smp/runs/5n0d2i06) | `run_control/v6_full_ppo.log` |

The full run entered iteration 61,997 normally. Its failure replay buffer is
expected to start at zero and warm up from stagnating recovery states; the
stratified push counter was already non-zero by iteration 61,998. Checkpoints
are saved every 1,000 iterations to limit disk use.
