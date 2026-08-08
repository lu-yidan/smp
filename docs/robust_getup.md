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

Checkpoint lineage used for this experiment:

| Stage | W&B run | Starting checkpoint |
| --- | --- | --- |
| v1 | `tabletennis/smp/hqzmfkkg` | trained from scratch |
| v2 | `tabletennis/smp/si4gfklo` | v1 `model_29999.pt` |
| v3 | `tabletennis/smp/rr9sxcmu` | v2 `model_38000.pt` |

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
