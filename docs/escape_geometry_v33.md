# V3.3 geometry-aware guided-board escape

## Why V3.3 is separate

V3.2 fixed the reset physics: a support-ready supine G1 starts grounded with a
positive 1 mm plate gap, and an 8 kg board can move only along its passive
vertical slide. The completed actor learned substantial lateral translation,
but the objective still measured pelvis-to-board centre distance. It could
therefore earn almost all progress while a head, hand, foot, or part of the
torso remained under a board edge. V3.3 is a new task so V3.2 remains a frozen,
reproducible baseline.

Task ID: `Smp-Getup-Escape-Plate-V33-G1`.

## Geometry-aware success

At every control step V3.3 projects every G1 collision geom into the board's
local planar frame. Conservative projected AABBs are used deliberately: a
rotated hand, foot, or capsule must be completely outside the board footprint.
The task records:

- current and best number of covered robot collision geoms;
- summed distance of covered geoms to their nearest board edge;
- minimum all-body planar clearance after the last geom leaves the footprint;
- robot-board contact, penetration, force, and hand-supported progress.

Success now requires all of the following for 15 consecutive control steps:

1. no robot-board contact;
2. zero robot collision geoms covered by the plate footprint;
3. at least 25 mm minimum planar clearance;
4. the existing minimum hand-support and supported-translation evidence.

Pelvis-to-board centre separation remains a diagnostic and receives only a
small directional reward. It no longer decides success. The actor observation
is unchanged: the deployable policy still receives the same 96-dimensional
proprioceptive vector and no plate pose, collision identity, or contact sensor.

## Curriculum

The physical plate remains 0.90 x 0.64 x 0.07 m with fixed x/y and one passive
vertical slide. Over the first 100,000 control steps:

- longitudinal offset amplitude narrows from 0.18 m to 0.04 m;
- lateral offset amplitude narrows from 0.22 m to 0.05 m;
- plate mass is sampled from 4--6 kg initially, then from 4--12 kg.

Mass and box inertia are scaled together per world. In play mode the curriculum
is disabled: the board is centred over the prepared supine body and fixed at
8 kg so visual tests always show the intended pinned scenario.

## Implementation checks

- Ruff and Python compilation pass.
- A one-iteration, 16-environment CUDA smoke test completes.
- A deterministic 256-environment reset/20-step audit produced 207 active
  boards; initial covered-geom count was 15 median (8--21 range), sampled mass
  was 4.01--5.99 kg, all active geometry trackers initialized, and actor
  observations remained finite.
- The actor input remains 96-D; geometry/contact signals appear only in reward,
  metrics, termination, and deterministic evaluation.
- Plate and automatic wrench controls remain independent in `scripts/play.py`.

## Proposed fine-tuning

Fine-tune from the completed V3.2 actor rather than from a no-obstacle recovery
policy:

```bash
uv run scripts/train.py Smp-Getup-Escape-Plate-V33-G1 \
  --env.scene.num-envs=4096 \
  --agent.resume True \
  --wandb-run-path tabletennis/smp/kazawsxg \
  --wandb-checkpoint-name model_88999.pt \
  --agent.algorithm.learning-rate=1e-4 \
  --agent.max-iterations=8000 \
  --agent.save-interval=1000 \
  --agent.run-name escape_plate_v33_geometry_from_v32_88999 \
  --enable-nan-guard True
```

Before the full run, use 64 environments for a short resume smoke test. During
checkpoint selection, require true all-body clearance rather than centre
separation alone:

```bash
uv run scripts/evaluate_escape_checkpoint.py \
  --checkpoint /path/to/model.pt \
  --task Smp-Getup-Escape-Plate-V33-G1 \
  --num-envs 512 \
  --steps 1000 \
  --seed 20260817
```

Reject a checkpoint if its apparent escape gain comes with increased invalid
contact, penetration, force, torque, or power. Also retain a separate no-plate
V8 evaluation; V3.3 is an escape specialist and is not evidence that general
fall recovery improved.

## Formal run (started 2026-08-17)

The 64-environment two-iteration resume smoke test completed from the V3.2
checkpoint. A 256-environment compatibility evaluation also loaded the old
actor unchanged and observed genuine 15-step geometry-clear successes, proving
that the stricter success state is reachable before fine-tuning.

- branch: `codex/escape-geometry-v33`
- source commit: `12534b5`
- server/GPU: `dsw-lyd2`, GPU 0
- tmux session: `smp_v33`
- W&B: `tabletennis/smp/owzoec67`
- source checkpoint: V3.2 `model_88999.pt`
- duration: 8,000 additional PPO iterations
- expected final checkpoint: `model_96998.pt`
- checkpoint interval: 1,000 iterations
- initial throughput: about 39,000 steps/s; initial ETA about 5 h 23 min
- log: `run_control/v33_geometry_from_v32_88999.log`

## Visual test

After training, enable the plate without enabling random play pushes:

```bash
uv run scripts/play.py Smp-Getup-Escape-Plate-V33-G1 \
  --wandb-run-path tabletennis/smp/<run> \
  --wandb-checkpoint-name model_<iteration>.pt \
  --num-envs 1 \
  --viewer native \
  --no-terminations True \
  --escape-obstacle True \
  --auto-disturbances False
```

## Completed run and checkpoint selection (2026-08-17)

The run completed at `model_96998.pt`. Checkpoint selection was performed with
three deterministic seeds (20260817--20260819), 512 environments per seed and
1,000 control steps per environment under the same centred 8 kg plate
condition. The evaluator additionally reports post-escape stable standing,
stance width and safety metrics. Stable standing requires the head to remain
above 1.10 m with upright projection at least 0.85, base linear speed below
0.5 m/s and base angular speed below 1.0 rad/s for 25 consecutive steps.

| checkpoint | escape + stable stand | invalid setup | median escape time | median escape-to-stand | stable foot separation | stable foot speed |
|---|---:|---:|---:|---:|---:|---:|
| 94000 | 91.73% | 8.07% | 1.71 s | 3.91 s | 0.651 m | 0.0071 m/s |
| **95000** | **95.64%** | **4.30%** | 1.80 s | 3.50 s | **0.456 m** | **0.0029 m/s** |
| 96000 | 93.75% | 6.18% | 1.71 s | **3.19 s** | 0.764 m | 0.0059 m/s |
| 96998 | 92.06% | 7.88% | 1.70 s | 4.12 s | 0.939 m | 0.0082 m/s |

`model_95000.pt` is therefore the frozen **balanced V3.3 baseline**. It has the
highest standard-condition success rate, the lowest invalid rate, the
narrowest stable stance and the lowest residual foot motion. The final
checkpoint is not selected merely because it is later.

A one-seed, 512-environment mass sweep shows a real robustness--stance
trade-off:

| plate mass | 94000 | **95000** | 96000 | 96998 |
|---|---:|---:|---:|---:|
| 4 kg | 93.75% | **95.90%** | 94.34% | 89.06% |
| 12 kg | 78.12% | 85.35% | **89.65%** | 88.28% |
| 16 kg (OOD) | 48.05% | 57.23% | 66.41% | **70.51%** |

The 16 kg result is outside the 4--12 kg training curriculum and is retained
as an OOD diagnostic, not as the headline result. `model_96998.pt` is archived
as a heavy-load specialist, while its approximately 0.94 m stable foot
separation makes it unsuitable as the balanced default.

Frozen balanced checkpoint:

- W&B: `tabletennis/smp/owzoec67`, `model_95000.pt`
- SHA-256: `38063879c144bd29af8e792bb7547b0e6c99e4043ba9b4c3c08219dab16ef81a`
- tag: `baseline/v33-escape-model-95000-balanced`
- server archive: `/mnt/workspace/user/luyidan/baselines/G1_Recovery_Below_Block/v33/model_95000.pt`

The next stance experiment must not penalize wide feet while the robot is
pinned, crawling or transitioning through kneeling. A stance-band regularizer
should activate only after full geometry clearance and stable upright entry.
For later stair recovery, the target should be terrain-relative feasible foot
placement rather than a globally narrow stance.

## Initial OOD boundary scan (2026-08-17)

The evaluator now supports explicit plate length, width, thickness, friction,
longitudinal offset and lateral offset. The physical plate spec, reset overlap
geometry and success geometry are changed together, avoiding a mismatch
between visible collision and the clearance metric. A 16-environment custom
geometry smoke test passed before the server sweep.

The first boundary scan used the balanced `model_95000.pt`, seed 20260822,
512 environments per condition, 1,000 steps, 8 kg and no reset jitter. These
are single-seed diagnostics rather than final paper statistics.

| condition | escape + stable stand | invalid | stable foot separation |
|---|---:|---:|---:|
| standard 0.90 x 0.64 m, friction 1.2 | 95.31% | 4.69% | 0.455 m |
| small 0.75 x 0.54 m | 96.88% | 3.12% | 0.455 m |
| large 1.05 x 0.74 m | 79.10% | 19.53% | 0.458 m |
| extra-large 1.20 x 0.80 m | 66.60% | 29.10% | 0.457 m |
| lateral offset -0.16 m | 81.84% | 17.97% | 0.457 m |
| lateral offset +0.16 m | 70.70% | 28.12% | 0.456 m |
| friction 0.4 | 85.94% | 14.06% | 0.455 m |
| friction 0.8 | 92.77% | 7.23% | 0.456 m |
| friction 1.8 | 91.21% | 8.20% | 0.457 m |

Longitudinal offsets of -0.20, 0.00 and +0.10 m achieved 95.70%, 90.04% and
84.57%, respectively. The left/right lateral result exposes an asymmetric
failure region that should be examined with mirrored failure states and more
seeds before changing training. Across every tested geometry, friction and
offset, the stable foot-separation median stayed within 0.455--0.458 m. This
strongly indicates that the stance is a learned checkpoint terminal state,
not a temporary response to a particular plate condition.

Priority for the next method iteration is therefore: (1) reproduce the
lateral asymmetry over three seeds, (2) train with mirrored hard-state replay
and wider-size curriculum, and only then (3) add a post-clearance-only stance
band. Do not entangle the stance penalty with the pinned escape phase.
