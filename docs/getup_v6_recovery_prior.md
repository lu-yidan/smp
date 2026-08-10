# V6 multi-recovery SMP prior

This note records the data and training plan for the V6 recovery prior. It is
separate from the V5 PPO policy so the old baseline remains reproducible.

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
