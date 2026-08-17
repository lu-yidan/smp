# V3.4 supine-and-prone guided-board escape

## Scope

V3.3 remains the frozen prone-only baseline. Its `mixed_fall_reset` selected
only reset type 2 (procedural prone), and the plate event accepted only that
type. V3.4 adds reset type 1 (procedural supine) without changing the frozen
task or its checkpoint.

Task ID: `Smp-Getup-Escape-Plate-V34-G1`.

## Reset contract

Training samples 50% supine and 50% prone episodes before replay. Both poses:

- are procedural rather than GSI resets;
- are lowered until the lowest collision surface is 4 mm above the ground;
- receive the same conservative positive 1 mm plate gap;
- use the same 4--12 kg plate-mass and overlap curriculum;
- retain the 96-dimensional deployable actor observation with no plate pose,
  mass, geometry clearance or simulator contact labels.

Only the prone subset receives the crawl-ready symmetric arm pose inherited
from V3.2. Supine actors retain their sampled arm configuration. This avoids
silently collapsing two orientation families into one joint pose.

The first deterministic 256-environment, two-step reset audit sampled 131
supine and 125 prone scenes. All 256 plate episodes were active; initial peak
penetration was below 2.4 mm, peak force was below 718 N, and neither contact
invalidity nor setup invalidity was observed. A longer 64-environment smoke
rollout showed that the frozen prone expert does not already solve supine: the
supine invalid-contact rate rose as the old policy acted under the unfamiliar
constraint. This is expected evidence for fine-tuning, not a reset failure.

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

## Fine-tuning plan

Resume from the balanced V3.3 `model_95000.pt`, not from the wider-stance
heavy-load checkpoint. First run a short 64-environment resume smoke test,
then fine-tune for 8,000 iterations with a 1e-4 learning rate and 1,000-step
checkpoint interval. Keep the first V3.4 experiment focused on the pose
expansion: do not simultaneously add the wider-board or stance-band changes.

Checkpoint selection must report supine and prone success separately. A model
that improves the average by sacrificing the frozen prone skill is rejected.
After this controlled experiment, mirrored hard-state replay and wider-board
curriculum can be introduced as the next attributable change.
