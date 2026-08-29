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

Formal runs started from commit `72daf14` on 2026-08-29:

| GPU | W&B run | Run ID |
| --- | --- | --- |
| 4 | [F1 + velocity](https://wandb.ai/tabletennis/smp/runs/12ov66ix) | `12ov66ix` |
| 5 | [F1 without velocity](https://wandb.ai/tabletennis/smp/runs/3yjov31t) | `3yjov31t` |
| 6 | [F4 + velocity](https://wandb.ai/tabletennis/smp/runs/zukwvixl) | `zukwvixl` |
| 7 | [F4 without velocity](https://wandb.ai/tabletennis/smp/runs/axme25ls) | `axme25ls` |

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

## Frozen evaluation (2026-08-29)

All four runs completed 30,000 updates without numerical collapse. Evaluation
used `scripts/evaluate_smp_baseline.py`, 512 environments, 500 control steps
(10 s), and seed 20260829. Native GSI evaluation retained the original automatic
velocity pushes. Fixed prone, supine, left-side, and right-side evaluation used
clean procedural resets without pushes.

Strict success required head height >= 1.10 m, uprightness >= 0.85, base linear
speed < 0.50 m/s, base angular speed < 1.0 rad/s, held for 25 consecutive
control steps.

Best fixed-pose checkpoint for each observation variant:

| Actor observation | Checkpoint | GSI | Prone | Supine | Left | Right | Fixed mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 + velocity | 29999 | 100.0% | 20.1% | 85.9% | 71.3% | 20.9% | 49.6% |
| F1, no velocity | 25000 | 100.0% | 35.2% | 37.7% | 51.8% | 22.3% | 36.7% |
| F4 + velocity | 8000 | 99.4% | 0.0% | 0.0% | 0.0% | 2.0% | 0.5% |
| F4, no velocity | 15000 | 100.0% | 18.6% | 17.0% | 12.9% | 21.1% | 17.4% |

Checkpoint drift was substantial outside the training distribution. For
example, F4 without velocity fell from 17.4% fixed-pose mean at update 15000 to
4.5% at update 25000 and 1.5% at update 29999, even though its native GSI
success stayed at 100% and its training SMP reward increased. Training reward
or the final checkpoint is therefore not a valid checkpoint selector.

The two velocity-free candidates were repeated with seeds 20260830 and
20260831 (1536 total environments per reset mode including the first seed):

| Candidate | GSI | Prone | Supine | Left | Right | Fixed mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 no-velocity, model 25000 | 100.00% | 34.96% | 37.11% | 48.70% | 24.02% | 36.20% |
| F4 no-velocity, model 15000 | 99.93% | 17.51% | 17.97% | 14.06% | 21.16% | 17.68% |

On native GSI, F1 no-velocity/model 25000 recovered in 1.01 s median with mean
peak joint speed 14.4 rad/s and mean peak power 501 W. F4
no-velocity/model 15000 recovered in 0.96 s median with 13.8 rad/s and 417 W.
Off-distribution fixed resets remained much more violent: mean peak joint speed
was roughly 38--54 rad/s depending on pose, and mean peak power reached 2.6 kW
for the F1 candidate and 3.7 kW for the F4 candidate in prone resets.

### Conclusions

1. Original SMP can reliably train a policy from a random initialization when
   its original F2S2 prior, GSI reset distribution, reward, pushes,
   terminations, and adaptive PPO schedule are retained.
2. Removing true base linear velocity does not prevent learning: both
   velocity-free actors reached approximately 100% native GSI success.
3. Four-frame history is not automatically better. It reduced fixed-pose
   generalization in this controlled experiment.
4. The recommended from-scratch breadth baseline is F1 no-velocity/model 25000.
   F4 no-velocity/model 15000 remains the history-based deployment comparison.
5. Neither is ready as an arbitrary-pose real-robot recovery policy. The next
   controlled change should broaden resets while keeping the successful
   original optimization recipe, followed by explicit speed/power constraints.

Raw evaluation outputs remain on `dsw-lyd2` under
`run_control/obs_factorial_eval_20260829` and
`run_control/obs_factorial_candidate_confirm_20260829`.
