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
