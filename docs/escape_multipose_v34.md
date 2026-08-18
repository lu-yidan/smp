# V3.4 supine-and-prone guided-board escape

## Scope

V3.3 remains the frozen supine-only baseline. Its `mixed_fall_reset` selected
only reset type 2 (procedural supine), and the plate event accepted only that
type. V3.4 adds reset type 1 (procedural prone) without changing the frozen
task or its checkpoint.

This naming was corrected on 2026-08-18 after a rendered reset and a direct
torso-frame audit exposed an old label inversion. For G1, reset type 1 points
the torso forward/chest axis down (`mean z = -0.997`, prone), while type 2
points it up (`mean z = +0.997`, supine). The physical trajectories and frozen
checkpoints are unchanged; only their earlier pose labels were wrong.

Task ID: `Smp-Getup-Escape-Plate-V34-G1`.

## Reset contract

Training samples 50% supine and 50% prone episodes before replay. Both poses:

- are procedural rather than GSI resets;
- are lowered until the lowest collision surface is 4 mm above the ground;
- receive the same conservative positive 1 mm plate gap;
- use the same 4--12 kg plate-mass and overlap curriculum;
- retain the 96-dimensional deployable actor observation with no plate pose,
  mass, geometry clearance or simulator contact labels.

Only the supine subset receives the support-ready symmetric arm pose inherited
from V3.2. Prone actors retain their sampled arm configuration. This avoids
silently collapsing two orientation families into one joint pose.

The first deterministic 256-environment, two-step reset audit sampled 131
prone and 125 supine scenes. All 256 plate episodes were active; initial peak
penetration was below 2.4 mm, peak force was below 718 N, and neither contact
invalidity nor setup invalidity was observed. A longer 64-environment smoke
rollout showed that the frozen supine expert does not already solve prone: the
prone invalid-contact rate rose as the old policy acted under the unfamiliar
constraint. This is expected evidence for fine-tuning, not a reset failure.

A separate 512-environment, 1,000-step source-checkpoint evaluation quantified
the starting gap before any V3.4 learning:

| reset pose | episodes | escape + stable stand | invalid contact | stable foot separation |
|---|---:|---:|---:|---:|
| supine | 246 | 94.72% | 5.28% | 0.456 m |
| prone | 266 | 0.00% | 36.84% | not reached |

The mixed aggregate was 45.51%. This is a useful controlled baseline: V3.4
must gain prone escape without sacrificing the already strong supine stratum.

## Evaluation

The checkpoint evaluator accepts `--reset-pose prone`, `supine`, or `mixed`
and emits separate active counts, escape-and-stable-stand rates, invalid rates
and stable foot separations for the supine and prone strata.

Playback can also force the reset family:

```bash
uv run scripts/play.py Smp-Getup-Escape-Plate-V34-G1 \
  --checkpoint-file /path/to/model.pt \
  --num-envs 1 \
  --viewer native \
  --no-terminations True \
  --escape-obstacle True \
  --auto-disturbances False \
  --escape-reset-pose mixed
```

Use `supine` or `prone` in place of `mixed` for a deterministic pose family;
press Enter in the native viewer to resample within that family.

## Fine-tuning protocol

Resume from the balanced V3.3 `model_95000.pt`, not from the wider-stance
heavy-load checkpoint. A 256-environment, two-iteration resume smoke test
completed with finite metrics and an exact 50/50 pose split. The formal run used
8,000 iterations with a 1e-4 learning rate and 1,000-step checkpoint interval.
Keep the first V3.4 experiment focused on the pose expansion: do not
simultaneously add the wider-board or stance-band changes.

Checkpoint selection must report supine and prone success separately. A model
that improves the average by sacrificing the frozen supine skill is rejected.
After this controlled experiment, mirrored hard-state replay and wider-board
curriculum can be introduced as the next attributable change.

## Formal run (completed 2026-08-18)

- branch: codex/escape-multipose-v34
- source commit: 8e5ccd9
- server/GPU: dsw-lyd2, GPU 0
- tmux session: smp_v34
- W&B: tabletennis/smp/x3xkcqro
- source checkpoint: V3.3 balanced model_95000.pt
- duration: 8,000 additional PPO iterations
- final checkpoint: model_102999.pt
- wall time: 5 h 29 min 51 s
- checkpoint interval: 1,000 iterations
- initial throughput: about 41,600 steps/s; initial ETA about 5 h 42 min
- log: run_control/v34_supine_prone_from_v33_95000.log

## Corrected-label checkpoint preview

At `model_96000.pt`, a deterministic 256-environment evaluation per pose
(seed `20260818`, 1,000 steps, 8 kg centred plate) found:

| physical pose | escape + stable stand | invalid | median stable foot separation |
|---|---:|---:|---:|
| prone | 94.53% | 5.47% | 0.482 m |
| supine | 81.64% | 17.58% | 0.488 m |

The new true-prone skill emerged quickly, but the frozen supine skill initially
regressed. Later checkpoints oscillated rather than improving monotonically.

## Three-seed checkpoint selection

After screening every saved checkpoint, models 98000, 99000, and 100000 were
evaluated with seeds 20260818--20260820, 256 environments per pose and seed,
1,000 steps, and the centred 8 kg plate:

| checkpoint | prone success | supine success | prone stance | supine stance |
|---|---:|---:|---:|---:|
| 98000 | 86.20% | 90.10% | 0.463 m | 0.503 m |
| 99000 | 89.45% | 90.63% | 0.678 m | 0.622 m |
| 100000 | 86.72% | 87.76% | 0.507 m | 0.516 m |

Freeze `model_98000.pt` as the balanced V3.4 baseline. It is the narrow-stance
Pareto point while retaining balanced physical-pose recovery. Preserve
`model_99000.pt` as a robust-wide candidate: it gains about 3.3 percentage
points in prone success but expands the prone stance by about 0.214 m. Do not
use the final `model_102999.pt`; its prone screening success fell to about 61%
and its stable stance widened to about 0.81 m.
