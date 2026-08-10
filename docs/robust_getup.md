# Robust G1 get-up experiment notes

This document records the task lineage, design choices, commands, and evaluation
criteria for the robust and smooth get-up policies. The baseline
`Smp-Getup-G1` remains unchanged so results stay reproducible.

## Task lineage

| Stage | Task | Initialization | Main purpose |
| --- | --- | --- | --- |
| v1 | `Smp-Getup-G1` | GSI only | Original fall-to-stand baseline |
| v2 | `Smp-Getup-Robust-G1` | 50% GSI, 50% procedural falls | Recover from arbitrary falls and physical pushes |
| v3 | `Smp-Getup-Robust-Smooth-G1` | Same as v2 | Slower rising, quieter feet, and lower joint/action acceleration |
| v4 | `Smp-Getup-Robust-Smooth-V4-G1` | 40% GSI, 30% prone, 10% each other fall | Ordered seated-crouched-standing recovery without vertical launch |
| v5 | `Smp-Getup-Robust-Smooth-V5-G1` | 30% GSI, 35% prone, 15% supine, 10% each side | Preserve V4 smoothness while recovering from a second dynamic fall |

Checkpoint lineage used for this experiment:

| Stage | W&B run | Starting checkpoint |
| --- | --- | --- |
| v1 | `tabletennis/smp/hqzmfkkg` | trained from scratch |
| v2 | `tabletennis/smp/si4gfklo` | v1 `model_29999.pt` |
| v3 | `tabletennis/smp/rr9sxcmu` | v2 `model_38000.pt` |
| v4 | `tabletennis/smp/65x7bde7` | v3 `model_47999.pt` |
| v5 | `tabletennis/smp/pkduffcs` | v4 `model_53998.pt` |

Formal v3 training was launched on 2026-08-08 on `dsw-lyd2` from branch
`codex/robust-getup-smooth-v3`, commit `62e89c5`.

- W&B: <https://wandb.ai/tabletennis/smp/runs/rr9sxcmu>
- Remote GPU: physical GPU 1
- Launcher / Python PID at launch: `42448` / `42451`
- Captured stdout: `/tmp/smp_getup_smooth_v3_train.log`

The get-up diffusion prior is still
`datasets/pretrain_ckpt/pretrained_getup_f2s2.pt`. No new motion dataset is
required for v2 or v3: robustness and smoothness are learned by PPO through
reset coverage, disturbances, and reward shaping.

## Reset and disturbance distribution

For each reset, v2/v3 first performs the normal GSI reset. Half of the
environments keep that sampled prior state. The other half are replaced by one
of four uniformly sampled procedural poses:

- supine
- prone
- left side
- right side

Therefore the expected probabilities are 50% GSI and 12.5% for each procedural
pose. Procedural poses randomize yaw, root height, small root velocity, and joint
angles. Their SMP history buffer is re-primed from the actual simulator state.

Training applies finite world-frame forces and torques to either the pelvis or
torso. v3 uses a moderate curriculum from 30 to 160 N and 3 to 18 N m. A wrench
lasts 5--15 control steps and starts every 50--150 steps. Quiet-standing rewards
are disabled during the wrench and for 50 additional steps (about 1 second), so
the policy may step to recover.

Playback disables automatic disturbances by default. Use
`--auto-disturbances True` to reproduce them. Manual dragging in the native
viewer remains possible in either mode.

`cfg.sim.nconmax = 64` reserves space for up to 64 contacts per simulated
world. Arbitrary lying poses create more simultaneous contacts than the original
task; 64 provides headroom over the observed 38--40-contact broadphase overflow.
It does not change the policy or apply extra physical forces.

## v3 reward design

The positive task score remains bounded in `[0, 1]` and is multiplied once by
the SMP score:

| Component | Weight | Purpose |
| --- | ---: | --- |
| height-dependent head velocity | 0.15 | target 0.15 m/s low down, taper to zero near standing |
| head height | 0.15 | reach 1.15 m |
| upright posture | 0.20 | align the torso with gravity |
| quiet foot speed | 0.10 | suppress small steps only when tall and upright |
| quiet base XY speed | 0.10 | reduce drift only when tall and upright |
| low base angular velocity | 0.10 | reduce whole-body rocking |
| low joint velocity | 0.10 | reduce violent articulation |
| smooth consecutive actions | 0.10 | reduce high-frequency control changes |

The head-velocity term is symmetric: being too fast is penalized as well as
being too slow. This removes the v2 incentive to rise as quickly as possible.

Four small negative terms sit outside the SMP product, so a low SMP score cannot
hide jerky control:

| Penalty | Weight |
| --- | ---: |
| action-rate L2 | -0.0015 |
| action-acceleration L2 | -0.001 |
| joint-acceleration L2 | -2.5e-8 |
| joint-limit violation | -0.1 |

A checkpoint smoke test with the first draft showed that stronger penalties
nearly cancelled the positive task reward. The final weights above reduce the
measured penalty budget to roughly one third while preserving a clear smoothing
gradient.

## Train

A local checkpoint must be under the new experiment's log root when using
`--agent.load-run`. W&B resume is simpler:

```bash
CUDA_VISIBLE_DEVICES=1 uv run scripts/train.py Smp-Getup-Robust-Smooth-G1 \
  --env.scene.num-envs=4096 \
  --agent.resume True \
  --wandb-run-path tabletennis/smp/si4gfklo \
  --wandb-checkpoint-name model_38000.pt \
  --agent.max-iterations=10000 \
  --agent.save-interval=250
```

`max-iterations` is additional when resuming. For example, starting from
`model_38000.pt` and running 10,000 iterations produces checkpoints through
approximately `model_48000.pt`.

## Play

Clean playback:

```bash
uv run scripts/play.py Smp-Getup-Robust-Smooth-G1 \
  --wandb-run-path tabletennis/smp/<v3-run-id> \
  --wandb-checkpoint-name model_<iteration>.pt \
  --num-envs 1 \
  --viewer native \
  --no-terminations True \
  --auto-disturbances False
```

Playback with the same automatic physical disturbances:

```bash
uv run scripts/play.py Smp-Getup-Robust-Smooth-G1 \
  --wandb-run-path tabletennis/smp/<v3-run-id> \
  --wandb-checkpoint-name model_<iteration>.pt \
  --num-envs 1 \
  --viewer native \
  --no-terminations True \
  --auto-disturbances True
```

The policy control period is 0.02 s (50 Hz). If native playback appears faster
than wall time, use the viewer's real-time synchronization controls; changing
the control timestep would change policy behavior and is not an evaluation-only
speed adjustment.

## v4 staged prone recovery

Manual evaluation found that v2 recovers reliably but can be violent, while v3
substantially reduces small corrective steps but sometimes fails from prone.
v4 preserves v3's continuous smoothing penalties and adds an ordered recovery
state; it does not simply remove low-height safety constraints.

The state machine exposes one waypoint at a time:

| Stage | Head | Upright | Mean knee flexion | Target vertical speed | Hold |
| --- | ---: | ---: | ---: | ---: | ---: |
| seated / kneeling support | 0.62 m | 0.60 | >= 1.00 rad | 0.06 m/s | 0.20 s |
| crouched support | 0.86 m | 0.76 | >= 0.80 rad | 0.08 m/s | 0.20 s |
| standing | 1.15 m | 0.90 | no minimum | 0.10 m/s | 0.50 s |

The next waypoint is unavailable until the current pose is held below its
vertical-speed threshold. A separate reward penalty begins above 0.20 m/s head
vertical speed. Action rate, action acceleration, joint acceleration, joint
limits, and actuator effort remain penalized in every stage. This is intended to
discourage using the feet and knees as a simultaneous ballistic launch.

The initial reset distribution is deliberately prone-heavy:

```text
40% GSI
10% supine
30% prone
10% left side
10% right side
```

Training keeps moderate physical disturbances but reduces their maximum during
focused prone refinement. Checkpoints are saved every 1,000 iterations to limit
disk and W&B storage.

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/train.py \
  Smp-Getup-Robust-Smooth-V4-G1 \
  --env.scene.num-envs=4096 \
  --agent.resume True \
  --wandb-run-path tabletennis/smp/rr9sxcmu \
  --wandb-checkpoint-name model_47999.pt \
  --agent.max-iterations=6000 \
  --agent.save-interval=1000 \
  --agent.run-name=smp_getup_robust_smooth_v4_from_47999
```

Formal v4 training was launched on 2026-08-09 on `dsw-lyd2`, GPU 0, from
branch `codex/robust-getup-smooth-v4-staged`, code commit `03e6488`.

- W&B: `tabletennis/smp/65x7bde7`
- Server worktree: `/mnt/workspace/user/luyidan/smp-v4`
- Run directory:
  `logs/rsl_rl/smp_getup_robust_smooth_v4_g1/2026-08-09_16-31-47_smp_getup_robust_smooth_v4_from_47999`
- Process record: `run_control/v4_train.pid`
- Console log: `run_control/v4_train.log` (W&B's `files/output.log` is
  unbuffered and is preferred for live progress)

The run completed normally after 6,000 refinement iterations. Seven checkpoints
were retained, for approximately 74 MiB total:

```text
model_48000.pt
model_49000.pt
model_50000.pt
model_51000.pt
model_52000.pt
model_53000.pt
model_53998.pt
```

The final training-window snapshot at iteration 53998 reported task score
0.6613, product score 0.2444, upright 0.9282, head vertical overspeed 0.0068,
mean foot speed 0.2675, max joint speed 3.9095, joint-acceleration RMS 83.36,
and action-rate RMS 0.4976. Recovery-stage completion averaged 0.2163 over the
mixed reset and disturbance window.

These aggregate training metrics show that vertical overspeed fell during
refinement, but they do not establish prone recovery success or prove the
ordered motion is visually natural. Checkpoint selection still requires the
evaluation below, starting with `model_49000.pt` and `model_53998.pt`.

## v5 recoverability and effort design

V4 visually improved the crouch-to-stand phase, but manual tests exposed two
coverage gaps: some prone or supine states never reached the seated waypoint,
and falling again after standing was less reliable than a reset-time fall. V5
addresses those gaps without restoring V2's ballistic knee/foot launch.

The safety objective is not implemented as a very low joint-speed cap. Turning
over from a contact-rich prone pose sometimes needs moderate joint speed, while
high static support torque can be necessary at the knee. The task instead uses
several complementary controls:

| Control | Lying | Seated | Crouched | Standing |
| --- | ---: | ---: | ---: | ---: |
| action-rate/acceleration multiplier | 0.30 | 0.60 | 1.00 | 1.00 |
| joint-acceleration multiplier | 0.35 | 0.65 | 1.00 | 1.00 |
| torque-cost multiplier | 0.50 | 0.75 | 1.00 | 1.00 |
| joint-speed soft limit | 6.0 | 5.0 | 4.0 | 3.5 rad/s |
| per-joint power soft limit | 140 | 110 | 90 | 75 W |

All actuator effort limits are also physically derated to 90% of the stock G1
simulation limits: 22.5 N m for arms, 79.2 N m for the 88 N m hip group,
125.1 N m for hip-roll/knees, 4.5 N m for wrists, and 45 N m for waist/ankles.
The configuration is deep-copied before derating so constructing V5 cannot
mutate V2-V4 or compound the reduction across repeated builds.

The distinction is important:

- effort limits cap peak actuator force;
- power limits penalize large force applied at large joint speed;
- joint-speed limits catch flailing and impacts;
- head vertical overspeed directly suppresses whole-body launch;
- stage-dependent smoothing leaves enough low-pose authority to begin recovery
  and becomes strict again during crouch-to-stand.

A small ungated recovery-initiation reward based on head height and uprightness
prevents a low-SMP prone state from making lying still optimal. It does not
reward vertical velocity and therefore does not directly encourage launch.

V5 replaces periodic generic pushes with one targeted knockdown after the robot
has held the stable-standing waypoint. The horizontal torso/pelvis wrench lasts
10--18 control steps, ramps from 90 to 190 N, and is followed by a 100-step
recovery window. This creates a second fall and recovery in the same episode,
including impact velocity and contact history. As in earlier versions, playback
is clean by default and `--auto-disturbances True` enables the event.

The implementation smoke test used 8 environments and one PPO iteration on an
RTX 4090. Environment construction, a 192-transition rollout, reward/metric
evaluation, and the PPO update all completed successfully. Checkpoints remain
spaced every 1,000 iterations.

Formal refinement command:

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/train.py \
  Smp-Getup-Robust-Smooth-V5-G1 \
  --env.scene.num-envs=4096 \
  --agent.resume True \
  --wandb-run-path tabletennis/smp/65x7bde7 \
  --wandb-checkpoint-name model_53998.pt \
  --agent.max-iterations=8000 \
  --agent.save-interval=1000 \
  --agent.run-name=smp_getup_robust_smooth_v5_from_53998
```

Formal v5 training was launched on 2026-08-10 on `dsw-lyd2`, GPU 0, from
branch `codex/robust-getup-smooth-v5-knockdown`, code commit `95444b3`.

- W&B: <https://wandb.ai/tabletennis/smp/runs/pkduffcs>
- Server worktree: `/mnt/workspace/user/luyidan/smp-v5`
- Run directory: `logs/rsl_rl/smp_getup_robust_smooth_v5_g1/2026-08-10_11-30-05_smp_getup_robust_smooth_v5_from_53998`
- Launcher / Python PID at launch: `73209` / `73212`
- Process record: `run_control/v5_train.pid`
- Console log: `run_control/v5_train.log`; W&B's
  `wandb/run-20260810_113024-pkduffcs/files/output.log` is unbuffered.

At iteration 54035 the run was healthy at about 78k environment steps/s. The
first distribution-shift window reported max joint speed 3.81 rad/s, mean max
joint torque 21.28 N m, mean max joint power 39.03 W, head overspeed cost
0.0137, and recovery-stage completion 0.1583. These are startup diagnostics,
not checkpoint-selection results; the policy must adapt to the harder reset and
second-fall distribution before evaluation.

## Evaluation checklist

Compare v2 and v3 from the same reset seeds, first with disturbances off and
then on. Do not select a checkpoint from total reward alone.

- Preserve get-up success and post-push recovery.
- Reduce `mean_foot_speed`, `action_rate_rms`, `joint_acc_rms`, and
  `max_joint_speed`.
- Keep `stable_stand`, `upright`, and `product_score` from regressing
  materially.
- Visually check all five reset sources and both disturbance modes.
- Prefer the earliest checkpoint that meets the behavior targets; excessive
  smoothing can eventually make recovery passive or too slow.

For v4, additionally require:

- prone recovery success no more than five percentage points below v2;
- ordered recovery without skipping the seated and crouched holds;
- head vertical speed normally below 0.20 m/s;
- no material regression in supine or side recovery;
- max joint speed and action-rate RMS remain near v3 rather than v2;
- visually reject knee/foot launch even if final standing succeeds.

For v5, evaluate both the first recovery and the post-standing knockdown:

- report success separately for GSI, prone, supine, left side, and right side;
- report first-recovery and second-recovery success instead of only aggregate
  reward;
- keep second-recovery success within five percentage points of first-recovery
  success;
- compare peak joint torque, peak joint power, max joint speed, head overspeed,
  and action-rate RMS against V2 and V4;
- reject a checkpoint that succeeds by simultaneous foot/knee launch;
- reject a checkpoint that stays down, even if its smoothness metrics improve;
- select the earliest checkpoint meeting recovery and safety constraints.
