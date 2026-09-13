# Recovery 10k → 12k continuation and quality ablations

Branch: `codex/recovery-quality-continuation`. Parent: `016ea25`.
Source checkpoints: `logs/rsl_rl/fixed_low/formal_20260913_112049/{L3,L4,L5}/model_9999.pt`.

| Arm | GPU | Source | New reward weights (effort, speed, target slew) |
|---|---:|---|---|
| L3 | 0 | L3 | 0, 0, 0 |
| L4 | 1 | L4 | 0, 0, 0 |
| L5 | 2 | L5 | 0, 0, 0 |
| L4-Q1 | 3 | L4 | -0.2, -0.2, 0 |
| L4-Q2 | 4 | L4 | -0.2, -0.2, -0.002 |

4096 environments per arm; 2000 additional PPO updates; parent iteration 9999 → iterations 10000–11999. Save roughly every 500, validate both natural and procedural banks for 20 seconds at saves. Videos every 1000 and stage end. End-of-stage regular 128-pose audit samples joint loads every 2ms and reports peak and P95 across episode maxima. P95 is not the 95th percentile over all time samples.

L3/4/5 retain fixed 20% low reset environments, equal four directions, with respectively 0%/5%/10% of all environments assigned procedural low resets. Remaining low resets are natural LAFAN. Late and middle are approximately 40% each. All low episodes retain exemption from low SMP termination; other groups retain original check. Training remains 10 seconds with standing termination; 20 seconds refers to independent validation. Actor 93D, single frame; existing actor noise, fixed exploration std 0.3, push and domain randomization remain enabled. No terrain change in this ablation.

## Costs

All are additive outside the original task × SMP product. All arms collect identical new telemetry. Q weights ramp linearly over the first 250 updates (24 control steps per update).

- Effort: average over joints and physics substeps of `[max(|tau|/tau_limit - 0.7, 0)/0.3]^2`. Existing actuator hard limits remain unchanged.
- Speed: average over joints and physics substeps of `min(max(|dq|/(0.5*v_ref)-1,0)^2,25)`. Reference speeds are mjlab G1 nominal motor values: hip roll/knee 20, hip pitch/yaw/waist yaw 32, wrist pitch/yaw 22, remaining joints 37 rad/s. Coupled ankle/waist nominal ratio is 1:1. These are experimental thresholds, not validated physical motor limits or a torque-speed envelope.
- Target slew: joint mean of `min((delta q_target/0.1)^2,25)` per 20ms control step, using processed targets after deployment projection/entry blend. First step after reset is excluded to avoid penalizing reset discontinuities. This is a soft penalty, not a hard rate limiter.

Telemetry: `Quality/{all,low,supine,prone,left,right}/...` contains raw costs; `full_weight_cost_to_task_ratio` compares the full-weight penalty with the original task-SMP reward before the warmup multiplier. `Episode_Reward/*` contains applied rewards. `LoadValidation/*` is separate nominal evaluation. Torque/velocity/power peaks and head net contact force are simulator diagnostics, not measured real-robot safety guarantees.

## Restore boundary

Legacy checkpoints contain actor, critic, observation normalizers, optimizer moments and parameter-group learning rate, iteration, and environment step counter. They **omit the SMP reward running-reference mean/count and simulator/RNG state**. Therefore these runs are not bitwise or fully objective-identical continuations.

A reproducible 500-control-step stochastic rollout of each saved initial policy, with its observation normalizer updating and the same reset/DR setup, reconstructs a reference mean. This is an approximation to the missing historical reference, not its recovery. Freeze it for continuation; L4/Q1/Q2 use the identical reference file. It must be checked for reward-scale shifts and behavior retention during preflight. Future checkpoints also save this mean/count, the separate adaptive learning-rate scalar and RNG state for better recovery, though full simulator state is still not saved.

Explicitly restore PPO's separate learning-rate scalar from the optimizer: L3/L5 start at 5.0625e-5 and L4/Q1/Q2 at 3.375e-5. Adaptive scheduling remains active. Assert actor/critic and optimizer tensors equal the parent before learning and record source/reference hashes and reset-state hashes in `launch.json`.

## Decision after 2k

Compare L4-Q1/Q2 against same-budget L4, using per-direction recovery, continuous 10-second quiet standing, loads, and videos. A reduced load with degraded recovery is a tradeoff, not automatic improvement. L3/L5 continuation tests whether their late learning continues. Longer 20k runs remain a subsequent stage; this launch does not silently schedule another 8k. No hardware deployment is performed.

Preflight calibration reduced Q2 target weight from -0.02 to -0.002 after the full-weight penalty/task ratio measured about 60% in low resets. Effort/speed were raised from -0.05 to -0.2 each; original combined ratio was about 0.55% low / 1.07% prone. These choices target a moderate penalty scale; final behavior requires the 2k comparison.
