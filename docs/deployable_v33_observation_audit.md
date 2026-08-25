# Deployable V3.3 Recovery Observation Audit

## Finding

The frozen V3.3 actor is not directly proprioceptive-deployable as previously
recorded.  Its 96-dimensional actor observation is:

1. base linear velocity in the body frame (3);
2. base angular velocity (3);
3. projected gravity (3);
4. relative joint position (29);
5. relative joint velocity (29);
6. previous action (29).

The G1 low-level state provides orientation, angular velocity, and joint
encoders, but no reliable body-frame base linear velocity.  The MuJoCo
`robot/imu_lin_vel` term is simulator state, not a directly available IMU
measurement.  Integrating accelerometer measurements is not an equivalent
substitute during contact-rich recovery because bias, gravity leakage, impacts,
and changing support contacts cause rapid drift.

## Frozen zero-observation ablation

Checkpoint: V3.3 balanced `model_95000.pt`.
Task: `Smp-Getup-Escape-Plate-V33-G1`.
Protocol: play configuration, fixed 8 kg plate, three seeds, 512 environments
per seed, 1,000 steps, deterministic policy.  Only actor observation indices
0:3 are changed.

| actor base linear velocity | escape rate | valid escape | invalid rate | mean peak power | mean peak torque |
|---|---:|---:|---:|---:|---:|
| simulator truth | 95.05% | 99.86% | 4.82% | 361.9 W | 86.4 Nm |
| forced zero | 93.42% | 99.72% | 6.32% | 376.8 W | 103.5 Nm |

Zero filling preserves most escape capability but raises peak torque by 19.7%
and peak power by 4.1%.  It is suitable for simulation and supported,
low-authority commissioning only; it is not accepted as the final real-robot
policy.

## Recommended V3.3-Deploy transition

Keep the actor input at 96 dimensions temporarily so the frozen checkpoint can
load exactly.  Replace the actor's first three dimensions with deterministic
zeros during training and deployment.  Keep simulator base linear velocity only
in the asymmetric critic.  Fine-tune from `model_95000.pt` with a low learning
rate and explicit torque, power, joint-speed, and action-rate limits.

This is preferable to a plug-in velocity estimator as the first step: recovery
contains hand, knee, torso, and mixed contacts, where leg odometry and
accelerometer integration are least reliable.  After the zero-conditioned
policy passes frozen safety evaluation, export a true 93-dimensional actor by
removing the constant inputs or distill into a short-history deployable student.

## Acceptance gate

- fixed 8 kg plate escape rate no lower than 94%;
- invalid rate no higher than 5%;
- mean peak torque no higher than the 86.4 Nm oracle baseline plus 5%;
- mean peak power no higher than 370 W;
- zero invalid dynamics;
- supported MuJoCo deployment first, then suspended real-robot commissioning
  with reduced gains and an immediate damping-mode trigger.

The ablation tool is `scripts/evaluate_escape_checkpoint.py`; use
`--base-lin-vel-mode zero` and `--play` for the deployment condition.
## V3.3-Deploy implementation and formal run

- Branch: `codex/deployable-recovery-observations`
- Training source commit: `37b5d1e`
- Task: `Smp-Getup-Escape-Plate-V33-Deploy-G1`
- Source checkpoint SHA-256:
  `38063879c144bd29af8e792bb7547b0e6c99e4043ba9b4c3c08219dab16ef81a`
- Prepared seed SHA-256:
  `6ff05dd6e0a9e1a46e67fa18677aca7486944310b88cc2a53a9b642635b7d062`
- Prepared-seed change: only `distribution.std_param`, from 0.30 to 0.15;
  deterministic actor weights are byte-identical.
- Environments: 4,096
- Runner seed: `20260825`
- Learning rate: `1e-5`
- Additional updates: 1,500
- Save interval: 250
- W&B: `tabletennis/smp/e41yaljt`
- Server tmux: `smp_v33_deploy`
- Server log: `run_control/v33_deploy_from_95000.log`

A local two-update strict-resume smoke and a server 256-environment, 20-update
smoke both passed.  The actor remains 96-dimensional with deterministic-zero
indices 0:3; the critic remains 960-dimensional and retains the simulator
velocity history.  The server smoke confirmed action standard deviation 0.15,
zero unstable simulation terminations, and finite torque/power metrics.

No checkpoint is promoted automatically.  Evaluate 95000, 95250, 95500, 95750,
96000, 96250, 96500, and the final checkpoint under the fixed 8 kg three-seed
protocol.
