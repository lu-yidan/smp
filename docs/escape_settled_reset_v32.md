# V3.2 settled-reset board escape

## Problem isolated from the V3.1 preview

The larger V3.1 board fixed partial body coverage, but its reset was still a
dynamic loading experiment: the board began above the robot and descended while
the policy was already trying to recover.  The resulting trajectory could push
the robot sideways until only a board edge remained in contact.

The underlying prone reset also mattered.  It rotated the nominal standing pose
onto its front without preparing the arms.  In 1024 deterministic samples, the
highest collision geom under the board was a hand in 891 cases (87.0%), a thigh
in 91 cases, and the torso in only 14 cases.  A collision-safe board therefore
had to hover above raised hands rather than begin against the trunk.

V3.2 is a separate task, `Smp-Getup-Escape-Plate-V32-G1`, so V3 and V3.1 remain
reproducible baselines.

## Reset definition

For each active board episode V3.2 now:

1. samples the same prone body orientation and lower-body variation;
2. places both arms in a symmetric crawl-ready pose, with small joint noise;
3. shifts the complete robot vertically until its lowest collision surface is
   4 mm above the flat ground and zeros reset velocity;
4. aligns the 0.90 x 0.64 m board with the torso-to-head direction;
5. computes exact vertical support for G1's sphere/capsule collision geoms and
   places the board bottom 1 mm above the highest overlapping surface.

No physics steps are hidden inside reset.  MjLab advances all vectorized worlds
together, so settling only newly reset environments would corrupt other active
episodes.  The geometry-based placement works for asynchronous resets and keeps
the actor's episode clock honest.

The board remains an 8 kg physical rigid body with one passive vertical slide.
Its x/y anchor is fixed at reset and never follows the robot.  Plate contacts use
locally stiffer solver parameters; robot-ground and other contacts are unchanged.
The actor observation is still the deployable 96-dimensional proprioceptive
vector.  Reset geometry, board pose, and contact state are not actor inputs.

## Pre-training checks (2026-08-14)

With 512 environments, seed `20260814`, and zero actions, 428 episodes contained
an active board.  Over the first 12 control steps:

- all 428 boards contacted the robot;
- first-contact step min/median/p99/max: 1/3/8/12;
- zero setup-invalid episodes;
- initial head-height median/p99/max: 0.121/0.199/0.218 m;
- peak penetration median/p99/max: 4.51/21.76/24.34 mm;
- peak contact force median/p99/max: 451/1576/2749 N;
- 32/428 samples crossed a physical-invalid threshold after the nominal
  zero-action controller began driving the prepared pose toward nominal stand.

A 4096-environment initialization and stepping smoke test also passed.  The
remaining physical-invalid samples are behavior failures, not initial overlap:
the board starts with a positive 1 mm geometric gap.  Training should terminate
and penalize policies that violently push into the board.

The frozen V3 `model_78993.pt` was evaluated for 500 steps in V3.2.  It contacted
all 428 active boards but escaped none, and 23.8% became physical-invalid.  This
is expected: V3 learned the earlier delayed/small-board route and is useful only
as a preview initialization, not evidence that V3.2 is solved.

## Visual preview before training

```bash
uv run scripts/play.py Smp-Getup-Escape-Plate-V32-G1 \
  --wandb-run-path tabletennis/smp/wwbgq95n \
  --wandb-checkpoint-name model_78993.pt \
  --num-envs 1 \
  --viewer native \
  --no-terminations True \
  --auto-disturbances True
```

The expected reset is already prone with hands near the ground and the board
almost touching.  The old policy is not expected to escape.  Confirm visually
that the initial coverage matches the intended pinned scenario before launching
fine-tuning.

## Fine-tuning run (started 2026-08-14)

The first V3.2 run fine-tunes the completed V3 actor for 10,000 additional PPO
iterations on 4096 environments:

```bash
uv run scripts/train.py Smp-Getup-Escape-Plate-V32-G1 \
  --env.scene.num-envs=4096 \
  --agent.resume True \
  --wandb-run-path tabletennis/smp/wwbgq95n \
  --wandb-checkpoint-name model_78993.pt \
  --agent.algorithm.learning-rate=3e-4 \
  --agent.max-iterations=10000 \
  --agent.save-interval=1000 \
  --agent.run-name escape_plate_v32_settled_from_v3_78993
```

- W&B run: `tabletennis/smp/7pgs6r86`
- server: `dsw-lyd2`, GPU 0
- tmux session: `smp_v32`
- expected final iteration: `88993`
- initial throughput: about 51,000 steps/s; estimated duration about 5 h 45 min
- startup checkpoint: V3 `model_78993.pt`

## Interrupted run audit and stable continuation (2026-08-14)

The first run did not reach its intended final iteration.  At iteration 84317 a
rare contact sample produced a non-finite actor observation and stopped the
whole 4096-environment job.  Immediately before the exception, mean value loss
jumped from `0.0088` to `2.0e13` and then `3.2e16`.  The last complete checkpoint
is `model_84000.pt`; the W&B run must therefore be treated as interrupted, not
successfully completed.

Deterministic 512-environment/1000-step comparisons showed that `model_84000.pt`
was the strongest saved actor: 3/428 active plate episodes completed, median
hand-supported progress was 0.414 m, and 238/428 reached 0.50 m center
separation.  However, only three samples maintained the required contact-free
hold.  This distinguishes the remaining bottleneck from a lack of crawling:
many policies move far enough laterally but do not fully clear the large board.

Commit `0cb4cf4` adds a pre-observation termination that resets only a world with
non-finite or runaway raw MuJoCo state.  It also rejects non-finite failure-replay
snapshots.  Normal actor observations are unchanged.  Training was resumed from
`model_84000.pt` with a lower learning rate and MuJoCo NaN dumps enabled:

```bash
uv run scripts/train.py Smp-Getup-Escape-Plate-V32-G1 \
  --env.scene.num-envs=4096 \
  --agent.resume True \
  --agent.load-run 2026-08-14_11-50-50_escape_plate_v32_settled_from_v3_78993 \
  --agent.load-checkpoint model_84000.pt \
  --agent.algorithm.learning-rate=1e-4 \
  --agent.max-iterations=5000 \
  --agent.save-interval=1000 \
  --agent.run-name escape_plate_v32_nan_safe_from_84000 \
  --enable-nan-guard True
```

- W&B run: `tabletennis/smp/kazawsxg`
- server/GPU: `dsw-lyd2`, GPU 0
- tmux session: `smp_v32_resume`
- run log: `run_control/v32_resume_from_84000.log`
- expected final iteration: `89000`
