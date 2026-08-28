# V3.8.7 deployable scratch-recovery gates

## Objective

Quickly determine whether a randomly initialized four-frame policy can first
learn clean flat-ground recovery before adding plates and terrain. These gates
are configuration tests, not final three-seed paper experiments.

The actor interface is unchanged across all gates:

- four frames, 96 values per frame (384 values total);
- zero-conditioned simulator base linear velocity;
- deployable IMU angular velocity and projected gravity;
- joint positions, joint velocities, and previous actions;
- no reset label, contact truth, terrain identity, height map, plate state, or
  simulator-only base linear velocity.

## V3.8.7 adaptive-LR gate

- Task: `Smp-Getup-Plate-Terrain-V387-Scratch-S0-Deploy-G1`
- W&B: `tabletennis/smp/2tosophn`
- 4096 environments, seed 387, 100 updates, flat ground, no plate.
- The adaptive KL scheduler reduced the configured `3e-4` learning rate to
  `1e-5` early.
- Frozen model 25, 50, and 75 evaluations all scored 0% on prone, supine,
  left-side, and right-side resets.

## V3.8.7.1 fixed-LR gate

- Task: `Smp-Getup-Plate-Terrain-V3871-Scratch-S0-Fixed-Deploy-G1`
- W&B: `tabletennis/smp/mgmguzh3`
- Commit: `ec6f686`
- 4096 environments, seed 3871, 200 updates, fixed `1e-4` learning rate.
- Frozen models 50, 100, 150, and 199 all scored 0% for all four reset poses.
- The SMP product increased from about 0.016 to 0.035, while upright fell to
  0.0028 and recovery initiation fell to 0.0024. This is a static-fall prior
  optimum rather than recovery learning.
- Do not spend additional seeds on this configuration.

Audit JSONL files are stored under:

`/mnt/workspace/user/luyidan/baselines/G1_Recovery_Below_Block/v3871/audit`

## V3.8.7.2 task-first dense gate

- Task: `Smp-Getup-Plate-Terrain-V3872-Scratch-S0-Dense-Deploy-G1`
- W&B: `tabletennis/smp/ki2lo46z`
- Commit: `6b7f20d`
- 4096 environments, seed 3872, 200 updates, fixed `1e-4` learning rate.
- Reduce `task_smp_product` weight from 1.0 to 0.05.
- Increase recovery initiation to 0.60, ordered stage progress to 1.00, and
  stable standing to 2.00.
- Add dense head-height (0.25) and upright (0.20) progress.
- Retain joint-speed, joint-power, vertical-speed, action-smoothing, and torque
  constraints.

The same seed was resumed from model 199 to iteration 1998 in W&B run
`tabletennis/smp/utng5pwo`. Upright increased to 0.966 and recovery initiation
to 0.955, but ordered-stage completion and frozen success remained zero for all
four poses. Mean knee flexion was only 0.479 rad, below the 1.0 rad seated
waypoint. Frozen model 1998 also reached 10.59 rad/s and 168.8 W in prone
evaluation. This configuration learned an upright-torso, straight-knee
shortcut and must not be expanded to additional seeds.

Audit JSONL files are stored under:

`/mnt/workspace/user/luyidan/baselines/G1_Recovery_Below_Block/v3872`

## V3.8.7.3 stage bridge

- Task: `Smp-Getup-Plate-Terrain-V3873-Scratch-Stage-Bridge-Deploy-G1`
- W&B: `tabletennis/smp/m355n8gg`
- Commits: `41a6ac3`, `22eb17f`
- Resume source: V3.8.7.2 model 1998 with matching SHA-256 copies.
- 4096 environments, seed 3872, 1000 additional updates, fixed `1e-4` rate.
- Remove the standalone head-height and upright rewards that admitted the
  straight-knee shortcut.
- Add the existing seated-crouched-standing staged pose at weight 1.20 and the
  slow staged head-velocity profile at weight 0.15.
- Raise ordered-stage progress to 2.00 and stable standing to 3.00.
- Tighten joint-speed and power excess penalties without reducing physical
  actuator limits.

Gate rule: frozen model 2248/2498/2748/2997 evaluations must show stage
advancement, non-zero formal recovery, and reduced prone speed/power before
this curriculum can add the motion prior, terrain, plates, or extra seeds.
