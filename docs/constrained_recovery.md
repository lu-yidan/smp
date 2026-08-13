# Constraint-aware recovery roadmap

## Scope

`Smp-Getup-Constrained-G1` is the first controlled baseline for recovery under
sustained external constraints. It freezes the V7 route prior, policy
observations, rewards, reset mixture, and failure-state replay. The only task
change is the disturbance distribution: a clean, trunk-loaded, or limb-loaded
cohort is sampled at reset and a downward-biased wrench persists for 2--7 s.

This first environment is a vectorized force curriculum, not a substitute for
rigid-object pinning. Movable objects, fixed obstacles, terrain perception, and
active probing are follow-up stages once the force baseline is reproducible.

## Deployment-information boundary

The actor receives exactly the V7 observations: noisy base linear/angular
velocity, projected gravity, joint position/velocity, and the previous action.
The sampled constraint body, load, duration, cohort, and remaining time are
never actor observations. They are available only as simulator state and
training/evaluation metrics. Future teacher observations must preserve this
boundary and be distilled into a student using measurable proprioception,
motor-current/torque estimates, history, and optionally egocentric depth.

## V1 cohorts and metrics

- clean: 25%
- trunk constraint (pelvis or torso): 50%
- limb constraint (elbow or knee): 25%
- downward load curriculum: 20--120 N with a 20% lateral component
- duration: 100--350 policy steps; release occurs within the same episode

Logged metrics are `constraint_active`, `constraint_cohort`,
`constraint_load_n`, and `constraint_release_progress`, in addition to all V7
recovery and safety metrics. Results must be stratified by cohort and load; a
single average success rate is not sufficient.

Trial-level evaluation CSVs use these required columns:
`pose_bin,terrain,constraint_body,load_n,duration_s,success,recovery_time_s,`
`stall_time_s,max_joint_speed,max_joint_torque,max_joint_power`. Summarize the
same format for simulation and hardware with:

```bash
uv run scripts/summarize_recovery_envelope.py trials.csv envelope.csv
```

## Training

Start from the selected V7 PPO checkpoint because actor and critic dimensions
are unchanged. The completed V7 run is `tabletennis/smp/uta5t00i` and its final
server checkpoint is `model_69995.pt`:

```bash
uv run scripts/train.py Smp-Getup-Constrained-G1 \
  --env.scene.num-envs=4096 \
  --agent.resume True \
  --wandb-run-path tabletennis/smp/uta5t00i \
  --wandb-checkpoint-name model_69995.pt \
  --agent.algorithm.learning-rate=3e-4 \
  --agent.max-iterations=4000 \
  --agent.save-interval=1000 \
  --agent.run-name constrained_v1_from_v7_69995
```

Checkpoints remain spaced at 1000 iterations. Use a new W&B run and retain the
V7 checkpoint as an immutable baseline.

Playback defaults to no automatic constraint, matching other robust tasks:

```bash
uv run scripts/play.py Smp-Getup-Constrained-G1 \
  --checkpoint-file <model.pt> \
  --num-envs 1 \
  --viewer native \
  --no-terminations True \
  --auto-disturbances False
```

Set `--auto-disturbances True` to visualize the sampled sustained constraint.

## Planned ablations

1. Frozen V7 versus V7 plus the sustained-constraint curriculum.
2. Add a history encoder that estimates a latent constraint belief from
   deployable observations; privileged contact/load truth is critic/teacher
   only.
3. Add bounded active-probing actions and an escape-progress objective.
4. Gate the SMP reward using predicted route feasibility, relaxing it only
   while the nominal motion is contact-infeasible.
5. Replace force constraints with randomized movable and fixed padded objects,
   then add stairs, gaps, slopes, and held-out geometry.

The primary evaluation is a recovery envelope over initial-pose bin, constrained
body, normalized load, duration, and terrain severity. Report recovery success,
post-release success, escape/stall time, peak joint speed/torque/power, action
jerk, contact impulse, slip, energy, and post-stand stability with matched seeds.

## V1 training record

- Branch/implementation commit: `codex/constrained-recovery-v1` / `9280a43`.
- Server workspace: `/mnt/workspace/user/luyidan/smp-v6-prior` on `dsw-lyd2`.
- V7 seed: `model_69995.pt`, copied without modification into
  `logs/rsl_rl/smp_getup_constrained_g1/v7_seed_69995/` (SHA-256
  `1324d3cbfe71d3896cd502e2cc381b839a678a17a357183528dd6153f6f0f0da`).
- A 64-environment, two-iteration checkpoint-resume smoke test passed on
  2026-08-13.
- The 4096-environment V1 continuation started on RTX PRO 5000 GPU 1 with
  learning rate `3e-4`, 4000 additional iterations, and checkpoint interval
  1000. W&B: <https://wandb.ai/tabletennis/smp/runs/fiykfruo>.
- Server log/PID: `run_control/constrained_v1_ppo.log` and
  `run_control/constrained_v1_ppo.pid`.

The continuation completed at `model_73994.pt` after 1 h 33 min. W&B contains
all five intended checkpoints (`70000`, `71000`, `72000`, `73000`, and
`73994`). At the final iteration, mean task/SMP/product scores were
`0.7015/0.3484/0.2601`; recovery-stage completion was `0.1173`. The sampled
constraint load averaged `75.37 N`, and constraints were active for `0.1772`
of episode samples. Peak-per-episode joint speed, torque, and power metrics
averaged `3.419 rad/s`, `20.98 Nm`, and `33.77 W`. These aggregate training
metrics are only a health check: they do not establish conditional recovery
success and must be followed by stratified recovery-envelope evaluation.

Visualize the final policy with the automatic sustained constraint enabled:

```bash
uv run scripts/play.py Smp-Getup-Constrained-G1 \
  --wandb-run-path tabletennis/smp/fiykfruo \
  --wandb-checkpoint-name model_73994.pt \
  --num-envs 1 \
  --viewer native \
  --no-terminations True \
  --auto-disturbances True
```
