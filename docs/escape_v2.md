# V2 physical-object escape task

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
