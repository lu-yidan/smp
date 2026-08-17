# V8 natural prone recovery

## Motivation

V7 changes the local 0.2-second SMP prior but leaves the long-horizon PPO route
unchanged. Its ordered stage objective rewards head height, uprightness, knee
flexion, and low speed, but it does not require hand/knee support or penalize the
large hip abduction and yaw observed in failed prone recoveries. A policy can
therefore raise its pelvis by spreading and swinging the legs without following
the reviewed LAFAN prone-to-kneel route.

V8 is a no-obstacle natural-recovery task. Plate escape remains a separate task
and experiment so motion quality and obstacle-clearance improvements can be
attributed independently.

The play switch `--escape-obstacle True` only enables a plate entity already
defined by an escape task; it does not add a plate to V8. To stress-test a V8
checkpoint under the plate, load that checkpoint with the
`Smp-Getup-Escape-Plate-V33-G1` task. This is a cross-task baseline only: the V8
policy was not trained to escape the plate.

## Task definition

Task ID: `Smp-Getup-Robust-Smooth-V8-Natural-G1`.

V8 inherits the full V7 route-prior task and adds:

- ground-contact sensors for the hands and knee/shin collision groups;
- a prone-only support-route reward that requires hand contact for dense torso
  elevation credit and gives additional credit for knee/shin support;
- a prone early-stage penalty for hip roll beyond 0.65 rad and hip yaw beyond
  0.75 rad;
- low-stage joint-speed limits of 5.0/4.5 rad/s and mechanical-power limits of
  120/100 W before the existing crouched/standing limits take over.

The support reward saturates after the first ordered recovery waypoint. Holding
a quadruped pose therefore cannot outperform continuing toward crouch and stand.
The actor observation remains the same 96-dimensional deployable proprioceptive
vector. Contact sensors are used only by training rewards and evaluation metrics.

## Controlled initialization

Fine-tune from the completed V7 route policy rather than from the plate-escape
policy:

- W&B source: `tabletennis/smp/uta5t00i`;
- source checkpoint: `model_69995.pt`;
- proposed learning rate: `1e-4`;
- proposed duration: 5,000 iterations on 4,096 environments;
- checkpoint interval: 1,000 iterations.

```bash
uv run scripts/train.py Smp-Getup-Robust-Smooth-V8-Natural-G1 \
  --env.scene.num-envs=4096 \
  --agent.resume True \
  --wandb-run-path tabletennis/smp/uta5t00i \
  --wandb-checkpoint-name model_69995.pt \
  --agent.algorithm.learning-rate=1e-4 \
  --agent.max-iterations=5000 \
  --agent.save-interval=1000 \
  --agent.run-name v8_natural_from_v7_69995
```

## Evaluation

`scripts/evaluate_natural_recovery.py` forces one reset stratum at a time and
reports recovery success/time, hand and knee support, early-prone leg splay,
foot excursion, joint speed, torque, and power. Automatic pushes, replay, and
terminations are disabled during this clean-policy evaluation.

```bash
uv run scripts/evaluate_natural_recovery.py \
  --checkpoint /path/to/model.pt \
  --task Smp-Getup-Robust-Smooth-V8-Natural-G1 \
  --reset-mode prone \
  --num-envs 512 \
  --steps 1000 \
  --seed 20260817
```

Compare V7 `model_69995.pt` and all V8 checkpoints under identical prone,
supine, left-side, and right-side seeds. Select by the full safety/success
envelope, not final PPO reward. The intended acceptance criteria are:

1. reduce prone leg-splay p95 by at least 30%;
2. increase supported prone progress without creating a static quadruped mode;
3. preserve prone recovery success within two percentage points or improve it;
4. preserve non-prone success within five percentage points;
5. do not increase p95 joint speed, mean peak torque, or mean peak power.

## V7 baseline envelope (2026-08-17)

The deterministic evaluator was run for 1,000 control steps on 512 environments
per reset stratum with seed `20260817`. A stable stand means 25 consecutive
steps with head height at least 1.10 m, upright score at least 0.85, base linear
speed below 0.50 m/s, and angular speed below 1.0 rad/s.

| Reset | Stable success | Median time | Ordered-stage success |
| --- | ---: | ---: | ---: |
| prone | 512/512 | 2.50 s | 0/512 |
| supine | 57/512 | 4.56 s | 0/512 |
| left side | 512/512 | 2.86 s | 0/512 |
| right side | 512/512 | 2.70 s | 0/512 |

The prone policy is successful by final posture but not by the intended route:
mean hand-support occupancy is 0.85%, knee/shin support is 0.012%, leg-splay
excess p95 is 0.157, and foot excursion is 0.543 m median / 0.824 m p95. Joint
speed p95 is 12.14 rad/s, while mean peak torque and power are 85.15 Nm and
202.40 W. These values are the frozen V7 comparison targets for checkpoint
selection. The weak 11.1% supine result must also be monitored; a visually
better prone policy is not accepted if it further damages supine recovery.

## Formal run (started 2026-08-17)

The 64-environment fresh-policy smoke test and two-iteration resume smoke test
both completed. The V7 source and local V8 seed copy have identical SHA-256:
`1324d3cbfe71d3896cd502e2cc381b839a678a17a357183528dd6153f6f0f0da`.

- W&B: `tabletennis/smp/xyxaybwi`
- server/GPU: `dsw-lyd2`, GPU 0
- tmux session: `smp_v8_natural`
- log: `run_control/v8_natural_from_v7_69995.log`
- source checkpoint: V7 `model_69995.pt`
- expected checkpoints: `70000`, `71000`, `72000`, `73000`, `74000`, `74994`
- initial throughput: about 68,000 steps/s; ETA about 2 h 5 min

## Completed checkpoint evaluation (2026-08-17)

The run completed at `model_74994.pt`. All six saved checkpoints were evaluated
deterministically on 512 prone and supine resets for 1,000 control steps with
seed `20260817`. The three strongest prone candidates were also evaluated on
both side-lying strata.

| Checkpoint | Prone success | Leg-splay p95 | Hand support | Joint-speed p95 | Mean peak torque | Mean peak power | Supine success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V7 `69995` | 100.0% | 0.157 | 0.85% | 12.14 rad/s | 85.15 Nm | 202.40 W | 11.1% |
| `70000` | 100.0% | 0.157 | 0.83% | 12.00 rad/s | 84.49 Nm | 203.46 W | 10.7% |
| `71000` | 100.0% | 0.184 | 1.68% | 12.00 rad/s | 80.46 Nm | 200.91 W | 8.2% |
| `72000` | 100.0% | 0.111 | 2.03% | 11.56 rad/s | 81.62 Nm | 227.58 W | 6.6% |
| `73000` | 100.0% | 0.103 | 1.81% | 10.07 rad/s | 75.73 Nm | 221.49 W | 3.1% |
| `74000` | 99.8% | 0.108 | 1.89% | 11.43 rad/s | 80.55 Nm | 195.46 W | 4.5% |
| `74994` | 100.0% | 0.133 | 2.16% | 12.04 rad/s | 76.76 Nm | 185.16 W | 6.25% |

`74000` meets the intended prone splay reduction and improves the three prone
safety metrics, but loses 6.6 percentage points of supine success. `74994` is
the lowest-power and strongest-hand-support checkpoint and stays just inside
the five-point supine regression allowance, but its splay reduction is only
about 15%. Side-lying recovery remains at least 99.6% for `73000`, `74000`, and
`74994`.

Therefore no V8 checkpoint satisfies every predeclared acceptance criterion.
For visual review, use `74000` as the natural-prone candidate and `74994` as the
lower-power compromise; do not label either as a fully accepted replacement
for V7. A follow-up should retain a fixed supine replay fraction while
fine-tuning the prone support route.
