# V3.5 Zero-Shot Terrain Recovery Benchmark

V3.5 is an evaluation task, not a new trained policy.  It measures how far the
frozen V3.4 plate-escape checkpoint generalizes from a flat floor to terrain
with non-coplanar support.  The policy actor remains exactly 96-dimensional:
base linear/angular velocity, projected gravity, joint position/velocity, and
the previous action.  Terrain family, terrain level, reset-pose label, local
height, and benchmark success are never actor observations.

## Frozen baseline

- Branch before V3.5: `codex/escape-multipose-v34`
- W&B run: `tabletennis/smp/x3xkcqro`
- Selected checkpoint: `model_98000.pt`
- SHA-256: `fa54ac58f09a1a0ed0b46f96fb920f18de20422190c9ee92207f3080a3cbe393`
- Local archive:
  `logs/rsl_rl/smp_getup_escape_plate_v34_g1/manual_checkpoints/model_98000.pt`

## Benchmark families

| Family | Level 0 | Level 1 | Level 2 | Level 3 |
|---|---:|---:|---:|---:|
| slope | 5 deg | 10 deg | 15 deg | 20 deg |
| stairs | 5 cm | 10 cm | 15 cm | 20 cm |
| rough grid | +/-2 cm | +/-4 cm | +/-6 cm | +/-8 cm |
| flat control | flat | flat | flat | flat |

The directed slope is a rotated collision box, avoiding MuJoCo heightfield
triangle-contact overflow.  Stairs use 30 cm treads.  Rough terrain uses 30 cm
cells with deterministic generation seeds.  All non-flat reset footprints are
centred at the terrain origin; the robot collision model is translated along
the local support normal to positive clearance before simulation begins.  This
prevents both initial penetration and misleading floating resets.

Procedural resets cover prone, supine, left-side, and right-side poses with
random yaw, small oblique orientation noise, and joint perturbations.  A reset
label is used only by the benchmark sampler.

## Interactive playback

Example: prone recovery on 10 cm stairs, without plate or automatic pushes.

```bash
cd /home/d080/workspace/smp

uv run scripts/play.py Smp-Getup-Terrain-V35-G1 \
  --checkpoint-file \
  logs/rsl_rl/smp_getup_escape_plate_v34_g1/manual_checkpoints/model_98000.pt \
  --num-envs 1 \
  --viewer native \
  --no-terminations True \
  --auto-disturbances False \
  --terrain-type stairs \
  --terrain-level 1 \
  --terrain-reset-pose prone
```

Valid terrain types are `flat`, `slope`, `stairs`, `rough`, and `mixed`.
Valid levels are `0` through `3`.  Valid forced poses are `prone`, `supine`,
`left-side`, `right-side`, and `mixed`.

## Reproducible evaluation

The evaluator disables termination, automatic pushes, hard-state recording,
and replay.  It reports stable-stand success and recovery time together with
secondary falls, leaving the terrain patch, root displacement, descent below
the spawn support, contact-conditioned foot speed, joint speed, torque, and
mechanical power.  Leaving a 1.75 m radius is a benchmark failure; that rollout
is re-anchored internally so an escaped body cannot free-fall and contaminate
the numerical statistics of the remaining batch.

```bash
uv run scripts/evaluate_terrain_recovery.py \
  --checkpoint \
  logs/rsl_rl/smp_getup_escape_plate_v34_g1/manual_checkpoints/model_98000.pt \
  --terrain-types flat slope stairs rough \
  --levels 0 1 2 3 \
  --reset-modes prone supine left_side right_side \
  --num-envs 64 \
  --steps 750 \
  --output logs/evaluation/terrain_v35_model98000_full.jsonl
```

## Preliminary boundary preview

These numbers are diagnostic, not paper results.  They use level 1, prone
resets, a 12 s horizon, and seed `20260818`.

| Terrain | Environments | Stable-stand success | Terrain-exit rate |
|---|---:|---:|---:|
| flat | 8 | 100% | 0% |
| 10 deg directed slope | 8 | 0% | 100% |
| 10 cm stairs | 8 | 0% | 87.5% |
| +/-4 cm rough grid | 8 | 75% | 25% |

The flat result confirms checkpoint compatibility, while the slope and stair
results show that V3.4's hand-supported recovery does not yet control tangential
drift or edge transitions.  This is the intended zero-shot baseline rather than
a reason to tune the test until the baseline passes.

## Next experiment: V3.6

Train terrain recovery from `model_98000.pt` with a curriculum that first mixes
flat, level-0 slope/stairs/rough terrain and then admits higher levels according
to per-family success.  Preserve the 96-D actor.  Candidate training rewards
must target outcomes rather than a scripted motion: stable stand, progress in
terrain-relative height, bounded root displacement/descent, contact-conditioned
foot slip, secondary-fall avoidance, and the existing speed/power/torque safety
terms.  V3.7 should combine terrain with the plate only after V3.6 ablations
separate terrain robustness from entrapment escape.
