# SMP from-scratch observation factorial

## Purpose

The original `Smp-Getup-G1` can learn recovery from a random policy. Its
`model_29999.pt` reached 100% formal success on native F2S2 GSI resets
(512 environments), but only about 23--30% on fixed prone, supine, and side
poses. This confirms that the original method trains, while also showing that
its reset manifold is narrow.

Later deployable recovery tasks changed several factors together: the motion
prior, reset distribution, reward decomposition, terminations, optimizer,
symmetry losses, disturbances, and actor observations. Some long from-scratch
runs then collapsed. This experiment changes only the actor observation, so it
can identify whether removing `base_lin_vel`, using observation history, or
their interaction prevents learning.

## Controlled 2 x 2 experiment

| GPU | Task | Actor input | Expected actor dim |
| --- | --- | --- | ---: |
| 4 | `Smp-Getup-G1` | 1 frame, true `base_lin_vel` | 96 |
| 5 | `Smp-Getup-Obs-F1-NoLinVel-G1` | 1 frame, no `base_lin_vel` | 93 |
| 6 | `Smp-Getup-Obs-F4-Vel-G1` | 4 frames, true `base_lin_vel` | 384 |
| 7 | `Smp-Getup-Obs-F4-NoLinVel-G1` | 4 frames, no `base_lin_vel` | 372 |

All four runs use the same:

- `pretrained_getup_f2s2.pt` state-dependent motion prior;
- 100% F2S2 GSI reset distribution;
- original velocity disturbances every 1--3 seconds;
- sole `task_smp_product` reward;
- original `smp_too_low`, `stood_up`, and timeout terminations;
- adaptive PPO learning rate starting at `1e-3`;
- no symmetry augmentation or symmetry loss;
- seed 3883, 4096 environments, 30,000 updates, checkpoint every 500 updates.

The critic remains the original privileged 10-frame, 960-dimensional input in
all variants. Only the actor interface is changed.

## Decision gates

Do not select checkpoints by training reward alone. For each run:

1. Check numerical health and policy statistics throughout training.
2. Evaluate saved checkpoints around the best window, not only the last one.
3. Require at least 95% formal success on native GSI resets.
4. Then evaluate fixed prone, supine, left-side, and right-side resets without
   disturbances.
5. For deployment candidates, verify that actor inputs contain no unavailable
   state such as true base linear velocity.

Interpretation:

- If F1+velocity learns but F1-no-velocity fails, velocity removal is the main
  observability gap.
- If F4-no-velocity learns, short proprioceptive history is sufficient for a
  deployable velocity-free policy.
- If F4+velocity fails while F1+velocity learns, frame stacking itself is
  changing optimization enough to require normalization or architecture work.
- If all four learn, the later failures came from the task/prior/reward/reset
  changes rather than observations.

After this gate, introduce changes one axis at a time: procedural pose resets,
the LAFAN route prior, safety shaping, terrain, then the plate obstacle.
