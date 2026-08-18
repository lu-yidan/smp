# V3.6.1 Exploration-Robust Terrain Curriculum

V3.6.1 corrects the curriculum-gating defect found after V3.6 completed.  V3.6
produced an excellent level-0 deterministic policy, but the PPO rollout's fixed
action standard deviation interrupted a separate 25-consecutive-step counter,
so every terrain cohort remained at level zero for the full run.

## Controlled change

Only the curriculum success gate changes.  At episode reset an environment may
advance when either:

1. the existing stable-standing counter reaches its threshold; or
2. recovery stage three has been completed and the root remains within 1.5 m of
   the assigned terrain origin.

Stage three is not a privileged actor input.  It is a training-only achievement
latch produced after the policy holds the seated, crouched, and standing stages
in order, and it resets after a substantial fall.  The actor remains the same
96-dimensional proprioceptive network.  The strict frozen evaluator still
requires 25 consecutive stable-standing steps.

## Source selection

V3.6 `model_102000.pt` is selected instead of the final checkpoint.  On level 0
it achieves 99.4% aggregate recovery with about 134 W mean peak power and 0.057
m/s contact-conditioned foot slip.  The final model reaches 99.6% but increases
these safety costs to about 157 W and 0.095 m/s.  On level 1 the selected source
already reaches 86.3% overall, establishing a useful warm start without adopting
the final model's more aggressive stair strategy.

## Planned run

- Branch: `codex/terrain-recovery-v361`
- Task: `Smp-Getup-Terrain-V361-G1`
- Source W&B run: `tabletennis/smp/de2kit7e`
- Source checkpoint: `model_102000.pt`
- Environments: 4,096
- Additional updates: 8,000
- Checkpoint interval: 1,000
- Learning rate: `1e-4`
- Runner seed: `20260818`

```bash
uv run scripts/train.py Smp-Getup-Terrain-V361-G1 \
  --env.scene.num-envs=4096 \
  --agent.seed=20260818 \
  --wandb-run-path tabletennis/smp/de2kit7e \
  --wandb-checkpoint-name model_102000.pt \
  --agent.resume True \
  --agent.algorithm.learning-rate=1e-4 \
  --agent.max-iterations=8000 \
  --agent.save-interval=1000 \
  --agent.run-name terrain_v361_curriculum_from_v36_102000
```

The launch gate is explicit: a server smoke run must show non-zero
`Curriculum/terrain_levels/stage_success` and at least one non-flat terrain
level above zero.  If this does not occur, do not start the formal run.

## Verification

A 256-environment, 120-update CUDA resume smoke test passed the launch gate.
The actor remained 96-dimensional, the V3.6 `model_102000.pt` checkpoint loaded
without mismatch, `stage_success` reached 0.20, mean terrain level reached
0.0977, maximum terrain level reached 2, and no patch-exit termination occurred.
The same smoke test with the original V3.6 gate had level mean and maximum equal
to zero, so this test directly verifies the controlled curriculum correction.
