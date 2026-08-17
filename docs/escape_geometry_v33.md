# V3.3 geometry-aware guided-board escape

## Why V3.3 is separate

V3.2 fixed the reset physics: a crawl-ready prone G1 starts grounded with a
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
is disabled: the board is centred over the prepared prone body and fixed at
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
