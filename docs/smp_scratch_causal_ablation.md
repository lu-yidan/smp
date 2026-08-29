# SMP from-scratch causal ablation

## Goal

Identify the smallest change that expands a deployable, velocity-free SMP
policy beyond its native GSI manifold while preserving reliable policy
learning from a random initialization.

This screen uses flat ground only. It deliberately excludes stairs, slopes,
rough terrain, and the plate obstacle so failures can be attributed to the
motion prior, reset coverage, off-manifold termination, or reward sparsity.

## Shared configuration

- actor: one frame, no true base linear velocity (93 dimensions);
- critic: original privileged 10-frame observation (960 dimensions);
- random actor, critic, and observation normalizers;
- original adaptive PPO, learning rate 1e-3, 24 steps per update;
- original task-SMP reward and automatic velocity pushes unless an arm
  explicitly changes the procedural cohort;
- original 5 s episode and success termination;
- 4096 environments, seed 20260830, 30,000 updates;
- checkpoint every 1,000 updates;
- contact capacity raised uniformly to 64 for safe lying-pose simulation.

The pretrained motion prior is part of SMP and is not a policy checkpoint.
Every policy in this experiment is trained from scratch.

## Eight paired arms

| GPU | Task suffix | Prior | Reset | Low-SMP termination | Procedural reward |
| ---: | --- | --- | --- | --- | --- |
| 0 | A0 F2S2 GSI | F2S2 | 100% GSI | original | exact product |
| 1 | A1 V7 GSI | LAFAN V7 | 100% GSI | original | exact product |
| 2 | A2 F2S2 strict | F2S2 | 80% GSI + 20% procedural | original | exact product |
| 3 | A3 V7 strict | LAFAN V7 | 80% GSI + 20% procedural | original | exact product |
| 4 | A4 F2S2 reset-aware | F2S2 | 80/20 | GSI only | exact product |
| 5 | A5 V7 reset-aware | LAFAN V7 | 80/20 | GSI only | exact product |
| 6 | A6 F2S2 bridge | F2S2 | 80/20 | GSI only | 10% floor on procedural only |
| 7 | A7 V7 bridge | LAFAN V7 | 80/20 | GSI only | 10% floor on procedural only |

The procedural 20% is split uniformly across prone, supine, left-side, and
right-side resets. GSI samples always retain the original exact task-times-SMP
reward. The bridge arms use

`task * (0.10 + 0.90 * SMP)`

only for procedural reset labels. This prevents the bridge from changing the
positive-control GSI objective.

## Causal interpretation

- A0 vs A1 isolates the prior.
- A0 vs A2 and A1 vs A3 isolate the addition of procedural resets.
- A2 vs A4 and A3 vs A5 test whether low-SMP termination kills exploration.
- A4 vs A6 and A5 vs A7 test whether exact SMP gating removes the task
  gradient in off-prior states.

No safety shaping, symmetry loss, terrain, plate, replay buffer, LAFAN
milestone reset, or staged recovery reward is included.

## Checkpoint gates

Evaluate updates 8k, 15k, 25k, and 29999 with
`scripts/evaluate_smp_baseline.py`:

1. native GSI with original automatic pushes;
2. clean fixed prone, supine, left-side, and right-side resets;
3. strict stable-standing success and recovery time;
4. peak joint speed, torque, and power.

Rapid screen gates:

- native GSI success >= 95%;
- fixed-pose macro success >= 40%;
- worst fixed pose >= 20%;
- finite actions and no late numerical collapse.

Safety is reported but is not optimized in this first causal screen. The
winning recovery configuration must later receive a separate speed/power
ablation before real-robot use.

## Follow-up

After selecting one or two arms:

1. rerun the winner with three independent policy-training seeds;
2. introduce flat/rough/slope/stair terrain via a from-scratch curriculum;
3. validate the plate on flat ground;
4. combine a small terrain-plus-plate cohort only after both components pass
   independently.

Continuing a policy that originally began randomly through a curriculum is
still a from-scratch training pipeline. Loading V3.3, V8, or another external
policy checkpoint is finetuning and is excluded from the baseline evidence.
