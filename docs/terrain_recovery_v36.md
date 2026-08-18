# V3.6 Proprioceptive Terrain-Recovery Fine-Tuning

V3.6 is the first trained terrain-recovery policy.  It starts from the frozen
V3.4 balanced plate-escape checkpoint `model_98000.pt`, removes the plate, and
adapts recovery to flat, slope, stair, and rough support.  The immediate goal
is to stand without rolling out of the local area; plate-plus-terrain recovery
is intentionally deferred so the terrain contribution remains attributable.

## Geometry correction

The V3.5 preview revealed that its 4 m patch was too small: a rolling robot
could reach the gray outer border before the recovery behavior was meaningfully
measured.  Both V3.5 playback and V3.6 training now use 8 m by 8 m patches.
Stairs have roughly six 30 cm rings plus about 1.9 m of flat lower apron.  The
formal patch-exit boundary is 3.5 m from the assigned terrain origin.

## Deployment observation contract

The actor remains 96-dimensional and receives only:

- base linear velocity;
- base angular velocity;
- projected gravity;
- 29 joint positions;
- 29 joint velocities;
- previous 29-dimensional action.

Terrain family, difficulty level, terrain origin, surface normal, reset pose,
contact labels, and curriculum state never enter actor observations.  Surface
normals and terrain labels are used only to construct valid training resets and
to update the curriculum.

## Curriculum and resets

Training cohorts are 30% flat, 30% slope, 25% stairs, and 15% rough terrain.
All non-flat environments start at level 0 and may advance through four levels:

| Family | Level 0 | Level 1 | Level 2 | Level 3 |
|---|---:|---:|---:|---:|
| slope | 5 deg | 10 deg | 15 deg | 20 deg |
| stairs | 5 cm | 10 cm | 15 cm | 20 cm |
| rough grid | +/-2 cm | +/-4 cm | +/-6 cm | +/-8 cm |

An environment advances after 25 consecutive stable-standing steps while the
root remains within 1.5 m of its terrain origin.  A valid failed episode moves
down one level.  The flat cohort stays at level 0 as anti-forgetting replay.
Every reset is a physically grounded procedural prone, supine, left-side, or
right-side pose; no plate, automatic push, or failure-state replay is active.

## Objective changes from V3.4

Existing SMP, staged recovery, hand/knee-support, joint-speed, joint-power, and
smoothness objectives are retained.  Height and recovery-stage calculations
are terrain-origin-relative.  V3.6 adds small outcome penalties for planar
drift outside a 0.4 m free radius, contact-conditioned foot slip, and excessive
upright foot separation.  Recovery success requires terrain-relative head
height, uprightness, bounded linear/angular speed, and at least 25 stable steps.
Leaving the assigned patch or producing an invalid simulation state terminates
the rollout.

## Reproducibility

- Branch: `codex/terrain-recovery-v36`
- Source checkpoint: `tabletennis/smp/x3xkcqro/model_98000.pt`
- Checkpoint SHA-256:
  `fa54ac58f09a1a0ed0b46f96fb920f18de20422190c9ee92207f3080a3cbe393`
- Checkpoint interval: 1,000 additional iterations
- Planned duration: 8,000 additional PPO iterations
- Learning rate: `1e-4`
- Requested environment seed: `20260818`
- Actual runner seed: `42` (the resumed PPO runner applies `agent.seed` to the
  environment; future multi-seed runs must override `--agent.seed` explicitly)

```bash
uv run scripts/train.py Smp-Getup-Terrain-V36-G1 \
  --env.scene.num-envs=4096 \
  --env.seed=20260818 \
  --wandb-run-path tabletennis/smp/x3xkcqro \
  --wandb-checkpoint-name model_98000.pt \
  --agent.resume True \
  --agent.algorithm.learning-rate=1e-4 \
  --agent.max-iterations=8000 \
  --agent.save-interval=1000 \
  --agent.run-name terrain_v36_from_v34_98000
```

Local verification completed before launch: Ruff and bytecode checks passed;
a 32-environment CUDA run confirmed finite two-iteration training, exact
96-dimensional actor input, all four curriculum cohorts, grounded resets, and
successful loading of the V3.4 checkpoint.  The formal W&B run and selected
checkpoint are recorded below after launch.

## Formal run

- Status: running from 2026-08-18 16:42 Asia/Taipei
- Source code commit: `1198651`
- Server/GPU: `dsw-lyd2`, GPU 0
- tmux: `smp_v36`
- W&B: `tabletennis/smp/de2kit7e`
- Server log: `run_control/v36_terrain_from_v34_98000.log`
- Initial throughput: about 27,858 steps/s
- Initial ETA: about 7 h 50 min
- Iteration range: 98,000 through 105,999 (8,000 additional updates)
- Selected checkpoint: pending evaluation

## Checkpoint selection

Do not select by training reward alone.  Evaluate all saved checkpoints on four
terrain families, four levels, and four reset poses.  Report stable recovery,
recovery time, patch exit, secondary fall, planar drift/descent, foot slip,
joint speed, torque, power, and final foot separation.  Include the frozen V3.4
zero-shot baseline and a flat-only regression check.
