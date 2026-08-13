# V2 physical-object escape task

> **Invalid physical experiment (2026-08-13).** Do not use this V2 run as a
> paper result or physical-object baseline. The obstacle reset teleported the
> plate to a fixed offset from the torso body centre without collision-aware
> surface placement or a settling phase. A 256-environment contact-distance
> diagnostic measured median initial penetration of 4.60 cm and maximum
> penetration of 13.37 cm. Initial contact force had median 630 N and 99th
> percentile 8.28 kN; after 10 simulation steps the maximum reached 13.29 kN.
> The W&B run `ml3n9wb6` and checkpoint table below are retained only as a
> negative implementation record. They do not measure valid pinned recovery.

## Motivation

The V1 sustained-wrench task trains load rejection and post-release recovery,
but its force follows the selected robot body. The robot cannot physically
slide out from under it. `Smp-Getup-Escape-G1` replaces that abstraction with a
visible free rigid body that can be pushed, rolled aside, or crawled out from.

V2 intentionally focuses on one failure mode: prone, chest-loaded recovery on
flat ground. Complex terrain, pelvis/limb pinning, and deformable objects remain
held-out extensions until this mechanism is validated.

## Environment and phases

The orange padded-plate proxy is 0.40 x 0.84 x 0.09 m, weighs 8 kg, and has
high sliding friction. Ninety percent of resets are procedural prone poses;
the plate is sampled on 90% of those poses. Other resets remain clean controls.

Escape state is simulator-only:

- 0: clean episode;
- 1: plate placed, waiting for first physical contact;
- 2: constrained after contact;
- 3: escaped after 0.3 s without plate contact and at least 0.24 m planar
  robot-plate separation.

The actor still receives the unchanged 96-dimensional V7 observation. It never
receives phase, plate pose, contact flags, or target-body identity. Thus V1/V7
checkpoints remain shape-compatible. Contact truth is used only by task reward
and metrics in this feasibility pilot.

## Hand-supported escape objective

During phase 2, the original upright/SMP product is scaled to 15%. Separate
ungated objectives reward:

- controlled 0.10 m/s low-pose planar motion while either hand supports on the
  floor;
- only newly achieved robot-plate separation, preventing oscillation farming;
- persistent escape completion.

After phase 3, the complete V7 get-up objective resumes automatically. Existing
joint-speed, torque, power, vertical-speed, and action-smoothing penalties stay
active throughout escape.

The initial design used a smaller 5 kg plate. A 64-environment zero-action
diagnostic produced 18 false escapes out of 52 loaded episodes because the
plate slid away under gravity. The final 8 kg cross-body plate reduced this to
1 out of 46 over 300 policy steps, so escape cannot normally be obtained by
waiting.

## Training and playback

Resume from constrained V1 `model_73994.pt`; network dimensions are unchanged:

```bash
uv run scripts/train.py Smp-Getup-Escape-G1 \
  --env.scene.num-envs=4096 \
  --agent.resume True \
  --wandb-run-path tabletennis/smp/fiykfruo \
  --wandb-checkpoint-name model_73994.pt \
  --agent.algorithm.learning-rate=3e-4 \
  --agent.max-iterations=4000 \
  --agent.save-interval=1000 \
  --agent.run-name escape_v2_from_constrained_73994
```

Visualize the plate with automatic disturbances enabled:

```bash
uv run scripts/play.py Smp-Getup-Escape-G1 \
  --wandb-run-path <org/project/run> \
  --wandb-checkpoint-name <model.pt> \
  --num-envs 1 \
  --viewer native \
  --no-terminations True \
  --auto-disturbances True
```

## Next ablations

1. Frozen V1 policy in the plate environment versus V2 reward fine-tuning.
2. Remove hand-support reward; remove separation reward; restore fixed SMP
   weight.
3. Add deployable action/IMU/joint/estimated-torque history and a bounded
   residual escape adapter.
4. Distill a privileged contact/plate-state teacher into that history encoder.
5. Hold out plate mass, footprint, friction, chest offset, then add pelvis and
   single-limb pinning.

## V2 pilot record

- Implementation branch/commits: `codex/constrained-escape-v2`, `11dc4d7`
  plus final-metric fix `5b57f41`.
- Server workspace: `/mnt/workspace/user/luyidan/smp-v6-prior` on `dsw-lyd2`.
- Seed: constrained V1 `model_73994.pt` (local copied SHA-256
  `83763a19fcc6a13e6f40d835c60ee74172128fc3694663f46e3f3c96eb10ea7d`).
- Frozen-policy baseline: 512 environments x 1000 steps. Loaded episodes were
  71.34% of the batch and 15.19% of the whole batch ended escaped, giving a
  conditional loaded-episode escape rate of approximately 21.3%. Mean hand
  support contact was 9.31%; mean peak joint torque/power were 33.49 Nm and
  67.27 W. The baseline log is `run_control/escape_v2_baseline_final.log`.
- The 4096-environment V2 continuation started on RTX PRO 5000 GPU 1 with
  learning rate `3e-4`, 4000 additional iterations, and checkpoint interval
  1000. W&B: <https://wandb.ai/tabletennis/smp/runs/ml3n9wb6>.
- Server log/PID: `run_control/escape_v2_ppo.log` and
  `run_control/escape_v2_ppo.pid`.

## Completed V2 result

The run completed at `model_77993.pt` after 2 h 3 min. All intended
checkpoints (`74000`, `75000`, `76000`, `77000`, `77993`) are present in W&B.
Training increased hand-ground support and reduced contact effort, but it did
not improve physical escape success.

Every checkpoint and the frozen V1 seed were re-evaluated with the same seed
(`20260813`), 512 environments, and 1000-step rollouts. Conditional escape is
`escape_completion / escape_obstacle_episode`:

| checkpoint | conditional escape | hand support | peak torque (Nm) | peak power (W) |
| --- | ---: | ---: | ---: | ---: |
| V1 `73994` | 25.1% | 9.2% | 33.31 | 67.04 |
| V2 `74000` | 21.8% | 8.9% | 33.48 | 67.32 |
| V2 `75000` | 10.1% | 24.0% | 28.57 | 59.61 |
| V2 `76000` | 10.3% | 32.5% | 29.97 | 60.79 |
| V2 `77000` | 13.2% | 36.0% | 29.85 | 63.49 |
| V2 `77993` | 14.9% | 35.0% | 30.27 | 60.82 |

Because the initial contacts are invalid, the checkpoint comparison cannot
isolate reward design from collision artifacts. The observed increase in hand
support remains a debugging clue, not a valid ablation result. A replacement
task must spawn the plate with guaranteed positive clearance, settle it under
physics, reject/resample excessive penetration, and log contact distance and
force before any reward comparison. Only after that validation should hand
support be gated by positive separation and the reward balance be revisited.

Evaluation logs are under `run_control/escape_v2_eval/` on the server.
