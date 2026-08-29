# RA-L real-G1 recovery protocol

This protocol is frozen before selecting the final checkpoint. It applies to
the 93D deployable SMP actor without true base linear velocity. The recovery
policy remains active for the full trial; success must not depend on switching
to a separate balance policy.

## Immutable trial identity

Before the first trial, record the policy-training seed, checkpoint and ONNX
SHA-256, observation schema, deployment Git commit, logger schema, controller
period, PD gains and torque clamps, robot identifier, surface, operator, video
identifier, and randomization seed. Final paper trials require a clean
deployment worktree. A dirty-worktree trial can be used for debugging but not
for a reported result.

Use logger schema 2 from the `codex/smp-hardware-evidence-logging` deployment
branch. It records measured joint position/velocity and motor `tau_est`, the
PD-command torque estimate, position targets, kp/kd, projected gravity, IMU
angular velocity, and FSM command. `tau_cmd_est` is a controller-side estimate,
not a calibrated torque sensor measurement, and must be labeled accordingly.

Every valid trial must bind its binary log, metadata JSON, and synchronized
video by SHA-256 in the ledger. Keep all three as local files while running the
analysis; a URL without a locally verifiable video artifact is insufficient for
final evidence. The analyzer recomputes every digest and includes the 80-trial
artifact map in `analysis.json`. Invalid initializations that never activate
the policy may leave artifact paths and hashes empty, but their ledger rows
must remain.

## Flat-ground core matrix

Run one preregistered block of 80 trials for the frozen final policy:

Before any policy activation, generate and commit the randomized plan. The
timestamp, policy/checkpoint/ONNX identity, deployment commit, robot, surface,
and seed are part of the immutable file:

```bash
uv run scripts/generate_smp_hardware_trial_plan.py \
  --output-json results/real_g1_flat_core/trial_plan.json \
  --block-id BLOCK_ID \
  --frozen-before-trial-utc 2026-09-01T00:00:00Z \
  --randomization-seed SEED \
  --policy-seed POLICY_SEED \
  --checkpoint-sha256 CHECKPOINT_SHA256 \
  --onnx-sha256 ONNX_SHA256 \
  --deploy-git-commit DEPLOY_COMMIT \
  --robot-id ROBOT_ID \
  --surface SURFACE_ID
```

Record the printed plan SHA-256 in every ledger row. `planned_slot` is the
0--79 slot in this plan, whereas `order_index` is the chronological index of
every physical attempt, including invalid initializations.

| Stratum | Trials | Placement |
| --- | ---: | --- |
| Prone | 15 | chest down, limbs randomized within the safe envelope |
| Supine | 15 | back down, limbs randomized within the safe envelope |
| Left side | 15 | left torso contact, limbs randomized |
| Right side | 15 | right torso contact, limbs randomized |
| Random fall state | 20 | scripted selection from the four strata plus intermediate orientations |

Randomize the execution order once using the recorded seed. Do not repeat a
failed trial merely because its initialization looked unfavorable. If an
initialization violates the preregistered physical envelope, mark it
`invalid_initialization` before policy activation and rerun it at the end with
the same stratum and `planned_slot`; preserve both records. Each planned slot
must end with exactly one valid trial. The analyzer rejects a modified plan,
an unplanned pose, a plan frozen after trial start, a missing slot, or more than
one valid result for a slot.

Use the same flat test surface and slack safety tether throughout the core
matrix. Plate, stair, slope, rough-terrain, and push tests belong to separately
stratified matrices and must not be pooled with the flat baseline.

## Outcomes

Start time is the first recovery-policy command after the operator releases
support. A trial is a success only when, within 10 s, the robot reaches an
upright two-foot stance without human contact, remains upright and locally
stationary for 1 s, and then remains upright for a 3 s post-success window.
The upright decision combines synchronized video with projected gravity and
joint velocity. Until calibrated external pose tracking is available, video
time is the authoritative recovery-time annotation and the logger supplies
the synchronized dynamics evidence.

Count a secondary fall during the 3 s window as failure in the primary
intent-to-treat success rate, while also reporting `first_stand_rate` and
`secondary_fall_rate`. Pressing passive mode, tether loading that materially
assists motion, human contact, controller fault, or a safety-limit trigger is a
failure for intent-to-treat analysis and is additionally reported as a safety
abort. Equipment failure before policy activation is excluded but retained in
the trial ledger.

Report by pose and overall:

- successes/attempts and Wilson 95% intervals;
- macro success and worst-pose success;
- median and 90th-percentile recovery time among successes;
- first-stand and secondary-fall rates;
- peak absolute joint velocity, measured `tau_est`, PD-command torque estimate,
  IMU angular speed, action first difference, and action second difference;
- failure taxonomy and all safety aborts.

The failure taxonomy is fixed as `no_progress`, `repeated_struggle`,
`pelvis_slip`, `leg_flailing`, `rearward_fall`, `lateral_fall`,
`small_step_instability`, `joint_or_torque_limit`, `operator_abort`,
`estimator_fault`, `invalid_initialization`, or `other` with text explanation.
Keep every binary log, metadata JSON, synchronized video, and ledger row.

## Safety and stopping

Two people are required: one operates the robot and one owns the passive-mode
trigger. Use a slack overhead tether, clear exclusion zone, charged battery,
and verified passive-mode transition before each block. Stop the block after
any hard-limit event, two consecutive unexplained controller faults, visible
hardware damage, thermal warning, or loss of reliable IMU/joint telemetry.
Resume only after documenting the cause and creating a new block identifier.

Motor limits in mjlab are simulation parameters and are not automatically
hardware-certified limits. Final per-joint warning and abort thresholds must
be copied from the deployed Unitree configuration or validated manufacturer
documentation and recorded with their source. The Unitree G1 developer guide
is the primary operational reference:
https://support.unitree.com/home/en/G1_developer/basic_motion_development

Before checkpoint selection and before the first reported trial, copy
`docs/ral_hardware_safety_limits_template.json` into the result directory,
replace every placeholder, and commit the completed file. Its `robot_id` must
match the ledger, its source configuration or document must be bound by
SHA-256, and all 29 joints must have positive finite velocity, measured-torque,
and command-torque-estimate limits. The IMU angular-speed, action first-
difference, and action second-difference thresholds must also be frozen. Pilot
trials may inform the action-smoothness thresholds, but the pilot block and the
rule used to choose them must be disclosed and must not overlap the final 80
trials.

The analyzer reads each schema-2 binary log and recomputes the 29 per-joint
velocity and torque peaks plus the scalar IMU/action metrics. A ledger value
that does not match the raw log is rejected. A threshold exceedance is retained
as evidence and yields `COMPLETE_WITH_SAFETY_LIMIT_EXCEEDANCE`, never silently
discarded or converted into an invalid trial. Only `safety_gate.pass=true` can
support the C09 safety claim.

Analyze the completed ledger with:

```bash
uv run scripts/analyze_smp_hardware_trials.py \
  --trials results/real_g1_flat_core/trials.csv \
  --trial-plan results/real_g1_flat_core/trial_plan.json \
  --safety-limits results/real_g1_flat_core/safety_limits.json \
  --output-json results/real_g1_flat_core/analysis.json
```

The analyzer rejects duplicate or incomplete trials, dirty deployment code,
old logger schemas, inconsistent hashes, missing raw logs, invalid success
labels, non-finite safety values, and outcomes outside the frozen taxonomy.


The row schema is frozen in `docs/ral_hardware_trial_template.csv`. Completed
rows and raw logs become `result` evidence for C09/C10 only after validation;
this protocol and an empty template are implementation evidence, not results.
