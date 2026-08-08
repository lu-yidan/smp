# FIRM-style G1 reproduction

This branch implements a paper-faithful reimplementation of FIRM for comparison
with SMP. It is not an exact reproduction: the authors have not released
training code, the paper uses Isaac Gym and a 23-DoF G1, while this repository
uses MJLab/MuJoCo and the 29-DoF G1.

The comparison will keep the simulator, embodiment, action space, reset
distribution, disturbances, and evaluation protocol shared with SMP. No SMP
score or frozen SMP denoiser is used by the FIRM task.

## Pipeline

1. Extract a small set of sparse fall/recovery seed demonstrations.
2. Train one sparse-keyframe PPO augmentation expert per demonstration.
3. Collect expert rollouts and post-trajectory-stitching rollouts as
   observation/goal/action sequences.
4. Distill those sequences into a conditional action diffusion policy.
5. Train the online adapter that retrieves a keyframe goal from observation
   history.

The first milestone is recovery-only FIRM-R. Unified fall prevention, impact
mitigation, and recovery will follow after FIRM-R is evaluated against the
current SMP recovery policy.

## Seed dataset

Initial source:

~~~text
/home/d080/workspace/LAFAN1_Retargeting_Dataset/g1/
  fallAndGetUp2_subject2.csv
~~~

The source is a headerless 30 Hz CSV. Each frame contains root position (3),
root quaternion in xyzw order (4), and 29 G1 joint positions. Retargeting only
enforces kinematic constraints; it does not guarantee dynamically feasible
motion or respect actuator limits.

The source is licensed CC BY-NC-ND 4.0. Derived clips stay in the ignored
`datasets/` directory. Git tracks only source frame indices, checksums,
statistics, and the processing code. Review the license before distributing
any modified motion data.

Generate the candidate manifest and selected local artifact:

~~~bash
uv run scripts/firm/prepare_lafan.py
~~~

The default detector marks candidate 003, centered on the fallen interval at
approximately 47.13--51.90 seconds. Its exported window includes two seconds
before and three seconds after the detected fallen interval:

~~~text
source frames: [1354, 1647)
duration:      9.77 s
direction:     forward-family (forward with a left component)
~~~

This is a provisional seed. Numerical checks must be followed by visual and
physics replay before it is accepted as an expert-training reference.

## Data acceptance gates

A seed clip must pass all of the following:

- upright, low-velocity start and end;
- a complete contact-rich fall or fallen posture and recovery;
- valid normalized quaternion and 29-joint ordering;
- no obvious ground penetration or discontinuity after ground alignment;
- joint positions within model limits;
- tolerable joint velocity and acceleration after 30-to-50 Hz interpolation;
- plausible tracking under the G1 actuator and PD model;
- approximately 25 informative sparse keyframes after manual review.

The selected candidate currently has these kinematic statistics:

| Quantity | Value |
| --- | ---: |
| pre-window mean root height | 0.737 m |
| pre-window mean torso up-z | 0.947 |
| post-window mean root height | 0.753 m |
| post-window mean torso up-z | 0.980 |
| 95th-percentile absolute joint speed | 2.16 rad/s |
| minimum fallen root height | 0.147 m |

### Candidate 003 validation

MJLab replay interpolates the 293 source frames to 487 frames at 50 Hz. The
motion has no joint-limit violations. Its interpolated joint-speed and
joint-acceleration statistics are:

| Quantity | Value |
| --- | ---: |
| maximum absolute joint speed | 9.06 rad/s |
| 95th-percentile absolute joint speed | 2.11 rad/s |
| maximum absolute joint acceleration | 142.10 rad/s2 |
| 95th-percentile absolute joint acceleration | 24.01 rad/s2 |

The visual sequence is coherent: crouch, controlled ground contact, side roll,
prone support, kneeling rise, and stable standing. This makes candidate 003 a
suitable first recovery-expert seed. It is a controlled descent rather than a
strong example of sudden fall mitigation, so later full-FIRM work still needs
abrupt forward, backward, and lateral fall demonstrations.

Per-frame rigid-body ground alignment requires a vertical correction between
-0.113 m and 0.094 m. This is too large and too rapidly varying for dense root
tracking. The sparse expert will therefore use the aligned poses only as
occasional posture anchors, and will append one final default-standing anchor
instead of assigning multiple keyframes to the repeated standing tail.

## Planned task stages

### Sparse-keyframe expert

The first task will be `Firm-Keyframe-G1`. It will use a 10-second episode,
random dense-frame initialization, small state perturbations, and a random
0.04--1.0 second actuator-disable interval. The actor observes root angular
velocity, joint position/velocity, last action, joint-position error to the next
keyframe, and phase. Only velocity-limit safety terminations are used.

### Rollout dataset

Each step records:

~~~text
observation = root angular velocity, q, qdot, previous action
goal        = target keyframe joint positions
action      = expert joint-position offset
~~~

A small pilot dataset will validate the action diffusion loop before scaling
toward the paper's 4.5 million samples.

### Action diffusion and adapter

The diffusion policy predicts a 12-action horizon and executes only the first
action. A Siamese goal/current-joint encoder produces a 64-dimensional relative
goal latent. The online adapter consumes 50 observation steps and refreshes the
retrieved keyframe goal every five control steps.

## Reproduction milestones

- [x] Isolate work on branch `repro/firm-g1`.
- [x] Add deterministic candidate detection and manifest generation.
- [x] Visually inspect and physically replay candidate 003.
- [ ] Train one sparse-keyframe expert.
- [ ] Scale to five directional experts.
- [ ] Collect and validate the pilot rollout dataset.
- [ ] Train action diffusion without adapter.
- [ ] Train the adapter and evaluate full FIRM-R.
