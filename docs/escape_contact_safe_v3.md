# V3 contact-safe guided-plate escape

## Scope

`Smp-Getup-Escape-Plate-V3-G1` is the first controlled physical-contact stage
after the invalid V2 free-plate pilot. It answers one narrow question:

> Can a prone G1 establish hand support and translate laterally out from under
> a real load without exploiting reset interpenetration?

The blue plate is an experimental fixture, not the final debris model. It has a
single passive vertical slide joint. Its horizontal anchor is placed over the
reset torso once and then remains fixed in world coordinates; it does **not**
follow the robot. Gravity and robot contact can move it vertically, including
letting the robot lift it. The robot itself must create horizontal separation.

This deliberate restriction prevents two confounds during the first skill:

- a free plate sliding away by itself and producing false escape success;
- the policy solving the task only by throwing or rotating the object instead
  of learning hand-supported crawling.

## Physical and reset configuration

- plate footprint: 0.52 x 0.52 m;
- thickness: 0.07 m;
- mass: 5 kg (approximately 49 N static load);
- high sliding friction;
- passive z slide with 120 N s/m damping and no actuator;
- 95% procedural resets, restricted to noisy prone poses;
- plate applied to 90% of eligible prone episodes, leaving clean controls;
- actor observation remains the same deployable 96 dimensions: base velocity,
  projected gravity, joint position/velocity, and previous action;
- plate pose, contact truth, task phase, and penetration are used only by the
  reward, termination, and metrics. They are not actor observations.

Unlike V2, reset height is not a fixed offset from the torso centre. V3 places
the plate above the highest robot body origin with a conservative 0.26 m margin,
then lets gravity establish contact. One sensor update is ignored after every
reset so a terminated episode's stale contact sample cannot contaminate the new
episode.

## Contact validity envelope

Robot-plate contact records both distance and force. An episode is marked
invalid and terminated when either:

- penetration exceeds 0.020 m; or
- contact-force norm exceeds 1500 N.

Metrics log the persistent invalid flag and per-episode peak penetration/force.
The task also penalizes force above 300 N, so a policy is not rewarded for
striking the plate upward.

## Escape objective

Phases are clean, waiting for contact, constrained, escaped, and invalid. During
the constrained phase, the original upright/SMP objective is reduced to 10%.
The main dense term is now **hand-supported escape progress**: new planar
robot-anchor separation earns reward only while the robot is low and at least
one hand contacts the ground. This closes the V2 loophole where hand motion or
low-pose movement earned reward without producing escape progress.

Escape requires at least 0.38 m planar separation and 15 consecutive control
steps without plate contact. Only then does the full V7 get-up objective resume.

## Validation record (2026-08-13)

Zero-action settling used 256 environments with seed `20260813` (226 active
plate episodes):

- first five control steps: zero robot-plate contacts and zero penetration;
- all active plates reached physical contact by step 83;
- at step 150, maximum penetration was 0.657 mm and p99 was 0.640 mm;
- at step 150, maximum contact force was 48.7 N and p99 was 47.3 N;
- no episode crossed the V3 validity envelope.

A 500-step (10 s) frozen-policy baseline used constrained V1
`model_73994.pt`, with validity termination disabled only so invalid samples
could be audited:

- all 226 active episodes established physical contact;
- 51/226 (22.6%) escaped;
- 1/226 (0.44%) crossed the penetration threshold;
- peak penetration median/p99/max: 4.46/16.06/33.39 mm;
- peak contact force median/p99/max: 580/981/1065 N.

The baseline is intentionally difficult and somewhat impact-heavy. It shows
that the task is not solved by waiting, while the force penalty gives PPO a
direct incentive to replace the V1 upward strike with controlled support and
translation.

## Training

Use the clean constrained V1 checkpoint, not the invalid V2 run:

```bash
uv run scripts/train.py Smp-Getup-Escape-Plate-V3-G1 \
  --env.scene.num-envs=4096 \
  --agent.resume True \
  --wandb-run-path tabletennis/smp/fiykfruo \
  --wandb-checkpoint-name model_73994.pt \
  --agent.algorithm.learning-rate=3e-4 \
  --agent.max-iterations=5000 \
  --agent.save-interval=1000 \
  --agent.run-name escape_plate_v3_from_constrained_73994
```

Playback with the guided plate enabled:

```bash
uv run scripts/play.py Smp-Getup-Escape-Plate-V3-G1 \
  --wandb-run-path <org/project/run> \
  --wandb-checkpoint-name <model.pt> \
  --num-envs 1 \
  --viewer native \
  --no-terminations True \
  --auto-disturbances True
```

For quantitative evaluation, do not pass `--no-terminations True`; invalid
contact episodes must remain rejected and reported.

### Active training record

- branch/implementation commit: `codex/escape-contact-safe-v3`, `97912ca`;
- server workspace: `/mnt/workspace/user/luyidan/smp-v6-prior` on `dsw-lyd2`;
- 4096-environment server smoke test completed successfully on RTX PRO 5000;
- formal seed: constrained V1 `tabletennis/smp/fiykfruo/model_73994.pt`;
- formal run: <https://wandb.ai/tabletennis/smp/runs/wwbgq95n>;
- log/PID files: `run_control/escape_plate_v3_ppo.log` and
  `run_control/escape_plate_v3_ppo.pid`;
- configuration: 4096 environments, learning rate `3e-4`, 5000 additional
  iterations, checkpoint interval 1000.

## Staged mobility roadmap

Do not claim V3 as general movable-object recovery. Advance only after V3
achieves a high valid conditional escape rate with reduced peak force:

1. **V3 guided load:** fixed x/y/yaw, passive z; learn hand support and crawl.
2. **V4 bounded mobility:** unlock limited x/y and yaw, randomize mass/friction,
   and require robot displacement as well as object-relative separation.
3. **V5 free debris:** fully free randomized boxes/boards and held-out shapes;
   reject passive object departure as success.
4. **Complex terrain:** repeat the recovery envelope on steps, slopes, and
   compliant-ground approximations before real-robot transfer.

Each stage should retain the previous stage as a separate evaluation cohort so
new object mobility cannot hide regression in the core crawling skill.
