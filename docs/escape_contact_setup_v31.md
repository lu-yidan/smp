# V3.1 prompt-load body-aligned board

## Why V3.1 exists

Visual inspection of the completed V3 policy exposed two task-definition
loopholes that were not represented by the final escape-rate table:

1. the 0.52 x 0.52 m plate was too small and could cover only part of the body;
2. the heavily damped plate could descend slowly enough for the policy to begin
   standing before the first contact.

V3 remains a recorded intermediate baseline. V3.1 is a separate task and does
not rewrite its results.

## Task changes

`Smp-Getup-Escape-Plate-V31-G1` uses a 0.90 x 0.64 x 0.07 m, 8 kg board. At
reset, its long axis is aligned with the supine torso-to-head direction and its
centre is shifted 0.10 m toward the pelvis. The board still has only one passive
vertical slide degree of freedom; its x/y anchor never follows the robot.

The reset keeps a collision-safe 0.22 m margin above the highest robot body
origin, while reducing slide damping from 120 to 60 N s/m. This preserves
positive initial clearance but establishes contact promptly.

An episode is setup-invalid when the first contact:

- takes longer than 35 control steps (0.70 s), or
- occurs after head height exceeds 0.75 m.

Physical validity remains unchanged: penetration above 20 mm or contact force
above 1500 N is invalid. Either invalid condition terminates training samples.

Escape completion additionally requires:

- at least five constrained control steps with hand-ground support;
- at least 0.04 m new separation accumulated during hand-supported steps;
- 0.50 m robot-board planar separation;
- 15 consecutive contact-free control steps.

The actor still receives only the deployable 96-dimensional proprioceptive
observation. Board pose, setup validity, contact, and support counters are
reward/metric information, not actor input.

## Pre-training validation (2026-08-14)

Zero-action settling with 256 environments and seed `20260814` produced 214
active board episodes:

- zero initial contacts and zero initial penetration;
- every board contacted the robot;
- first-contact step median/p99/max: 26/29.9/30;
- first-contact head-height p99/max: 0.114/0.154 m;
- zero physical-contact invalid samples.

The frozen V3 `model_78993.pt` was then evaluated in V3.1 for 1000 steps with
512 environments (428 active boards):

- every active board contacted the robot;
- first-contact step median: 16 (0.32 s);
- first-contact head-height median/max: 0.458/0.538 m;
- zero setup-invalid samples;
- escape: 28/428 (6.54%);
- hand-support time fraction: 55.15%;
- 19/428 (4.44%) physical-invalid samples under the stricter, heavier board;
- peak penetration p99/max: 16.21/22.81 mm;
- peak contact force p99/max: 1630/1721 N.

The low frozen-policy escape rate is intentional evidence that V3.1 removed the
easy V3 route. The large increase in hand support (V3 audit: about 2.4%) shows
that the new board actually forces a supported low-pose strategy. Fine-tuning
must now improve genuine contact release while reducing the 4.44% invalid rate.

## Preview before training

Use the completed V3 policy only to inspect board placement and task behaviour:

```bash
uv run scripts/play.py Smp-Getup-Escape-Plate-V31-G1 \
  --wandb-run-path tabletennis/smp/wwbgq95n \
  --wandb-checkpoint-name model_78993.pt \
  --num-envs 1 \
  --viewer native \
  --no-terminations True \
  --auto-disturbances True
```

This checkpoint is not expected to solve V3.1. Confirm that the board lands
across chest/pelvis while the robot is still low before starting the long run.
