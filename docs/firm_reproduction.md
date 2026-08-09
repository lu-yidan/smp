# FIRM-style G1 reproduction

This branch implements a paper-faithful reimplementation of FIRM for comparison
with SMP. It is not an exact reproduction: the authors have not released
training code, the paper uses Isaac Gym and a 23-DoF G1, while this repository
uses MJLab/MuJoCo and the 29-DoF G1.

The comparison will keep the simulator, embodiment, action space, reset
distribution, disturbances, and evaluation protocol shared with SMP. No SMP
score or frozen SMP denoiser is used by the FIRM task.

## Pipeline

1. Extract a small set of sparse fall/recovery seed demonstrations.
2. Train one sparse-keyframe PPO augmentation expert per demonstration.
3. Collect expert rollouts and post-trajectory-stitching rollouts as
   observation/goal/action sequences.
4. Distill those sequences into a conditional action diffusion policy.
5. Train the online adapter that retrieves a keyframe goal from observation
   history.

The first milestone is recovery-only FIRM-R. Unified fall prevention, impact
mitigation, and recovery will follow after FIRM-R is evaluated against the
current SMP recovery policy.

## Seed dataset

Initial source:

~~~text
/home/d080/workspace/LAFAN1_Retargeting_Dataset/g1/
  fallAndGetUp2_subject2.csv
~~~

The source is a headerless 30 Hz CSV. Each frame contains root position (3),
root quaternion in xyzw order (4), and 29 G1 joint positions. Retargeting only
enforces kinematic constraints; it does not guarantee dynamically feasible
motion or respect actuator limits.

The source is licensed CC BY-NC-ND 4.0. Derived clips stay in the ignored
`datasets/` directory. Git tracks only source frame indices, checksums,
statistics, and the processing code. Review the license before distributing
any modified motion data.

Generate the candidate manifest and selected local artifact:

~~~bash
uv run scripts/firm/prepare_lafan.py
~~~

The default detector marks candidate 003, centered on the fallen interval at
approximately 47.13--51.90 seconds. Its exported window includes two seconds
before and three seconds after the detected fallen interval:

~~~text
source frames: [1354, 1647)
duration:      9.77 s
direction:     forward-family (forward with a left component)
~~~

This is a provisional seed. Numerical checks must be followed by visual and
physics replay before it is accepted as an expert-training reference.

## Data acceptance gates

A seed clip must pass all of the following:

- upright, low-velocity start and end;
- a complete contact-rich fall or fallen posture and recovery;
- valid normalized quaternion and 29-joint ordering;
- no obvious ground penetration or discontinuity after ground alignment;
- joint positions within model limits;
- tolerable joint velocity and acceleration after 30-to-50 Hz interpolation;
- plausible tracking under the G1 actuator and PD model;
- approximately 25 informative sparse keyframes after manual review.

The selected candidate currently has these kinematic statistics:

| Quantity | Value |
| --- | ---: |
| pre-window mean root height | 0.737 m |
| pre-window mean torso up-z | 0.947 |
| post-window mean root height | 0.753 m |
| post-window mean torso up-z | 0.980 |
| 95th-percentile absolute joint speed | 2.16 rad/s |
| minimum fallen root height | 0.147 m |

### Candidate 003 validation

MJLab replay interpolates the 293 source frames to 487 frames at 50 Hz. The
motion has no joint-limit violations. Its interpolated joint-speed and
joint-acceleration statistics are:

| Quantity | Value |
| --- | ---: |
| maximum absolute joint speed | 9.06 rad/s |
| 95th-percentile absolute joint speed | 2.11 rad/s |
| maximum absolute joint acceleration | 142.10 rad/s2 |
| 95th-percentile absolute joint acceleration | 24.01 rad/s2 |

The visual sequence is coherent: crouch, controlled ground contact, side roll,
prone support, kneeling rise, and stable standing. This makes candidate 003 a
suitable first recovery-expert seed. It is a controlled descent rather than a
strong example of sudden fall mitigation, so later full-FIRM work still needs
abrupt forward, backward, and lateral fall demonstrations.

Per-frame rigid-body ground alignment requires a vertical correction between
-0.113 m and 0.094 m. This is too large and too rapidly varying for dense root
tracking. The sparse expert will therefore use the aligned poses only as
occasional posture anchors, and will append one final default-standing anchor
instead of assigning multiple keyframes to the repeated standing tail.

The accepted schedule contains 25 targets: 24 recovery anchors from output
frames 0 through 405, followed by one final stable-standing anchor at output
frame 486. Source frame 1597 is the manually reviewed start of the stable tail.
The choice and the generated frame indices are recorded in the validation JSON.

## Sparse-keyframe expert

The first task is `Firm-Keyframe-G1`. At reset, each environment samples a
dense frame uniformly from the validated 50 Hz motion and writes that physical
state to the simulator with small pose, velocity, and joint perturbations. The
command then exposes the next strictly later sparse keyframe; after the final
anchor it holds the stable-standing target.

The policy runs at 50 Hz. Its actor observation has 120 dimensions:

| Term | Dimension |
| --- | ---: |
| root angular velocity | 3 |
| joint position | 29 |
| joint velocity | 29 |
| previous action | 29 |
| current joint position minus next-keyframe position | 29 |
| normalized motion phase | 1 |

The asymmetric critic additionally observes root linear velocity and therefore
has 123 dimensions. The actor and critic MLPs use hidden sizes 512 and 256.

The main tracking reward weights follow the FIRM paper:

| Term | Weight |
| --- | ---: |
| rigid-body position | 1.25 |
| rigid-body orientation | 0.5 |
| rigid-body linear velocity | 0.125 |
| rigid-body angular velocity | 0.125 |
| joint position | 0.5 |
| joint velocity | 0.125 |
| joint-position limit | -10 |
| joint-velocity limit | -5 |
| action rate | -0.001 |
| torque | -1e-6 |
| joint acceleration | -2.5e-7 |
| self collision | -1e-7 |

This first runnable version is explicitly **Stage 0**. It keeps startup mass,
encoder-bias, and friction randomization, but disables automatic pushes and
actuator dropout. Rough terrain, momentum/yank penalties, the 0.04--1.0 second
actuator-disable curriculum, and physical disturbances will be introduced as
separate ablations only after the clean expert learns. This staging makes it
possible to attribute a failure to one change instead of several simultaneous
curricula.

The only non-timeout termination is a numerical safety guard for extreme joint,
root-linear, or root-angular velocity. Playback starts from the first frame,
disables observation corruption, and has no automatic disturbance in Stage 0.

### Build the local motion artifact

```bash
uv run scripts/firm/prepare_lafan.py
uv run scripts/firm/validate_lafan.py
```

The generated NPZ and contact sheet stay under the ignored
`datasets/firm/lafan/` directory. They must be copied to a training host
separately from Git.

### Train

```bash
uv run scripts/train.py Firm-Keyframe-G1 \
  --env.scene.num-envs=4096 \
  --agent.max-iterations=30000 \
  --agent.save-interval=500
```

The installed MJLab training wrapper passes a tracking-only `registry_name`
keyword that its own base runner version does not accept. This task registers a
small compatibility runner that discards the keyword only after the wrapper has
already resolved the local motion file. No third-party package is patched.

### Play

```bash
uv run scripts/play.py Firm-Keyframe-G1 \
  --checkpoint-file logs/rsl_rl/firm_keyframe_g1_c003/<run>/model_<iteration>.pt \
  --num-envs 1 \
  --viewer native \
  --no-terminations True
```

### Smoke record

On 2026-08-09, a local 32-environment, 2-iteration tensorboard smoke completed
1,536 simulation steps and both PPO updates. The constructed actor and critic
were 120 and 123 dimensions, respectively. This verifies runtime wiring only;
the reported early rewards are not evidence of expert quality.

### Formal Stage 0 run

Formal candidate 003 training was launched on 2026-08-09:

| Field | Value |
| --- | --- |
| code branch / commit | `repro/firm-g1` / `6af66fb` |
| server workspace | `/mnt/workspace/user/luyidan/smp-firm` |
| device | physical GPU 2, NVIDIA RTX PRO 5000 72GB Blackwell |
| environments | 4096 |
| iterations / checkpoint interval | 30,000 / 500 |
| W&B run | [`tabletennis/smp/j0q8fell`](https://wandb.ai/tabletennis/smp/runs/j0q8fell) |
| run name | `firm_keyframe_g1_c003_stage0` |
| launcher / uv PID at launch | `45906` / `45907` |
| captured stdout | `/tmp/firm_keyframe_g1_c003_stage0_train.log` |
| motion NPZ SHA256 | `27657fe40642245b3c1d0362895c4f5a04988a6d59e05e7c0c29ab41b0f9fb1f` |

The server clone was created independently from `smp-robust`, and the
git-ignored NPZ was copied separately and checksum-verified. A 4096-environment
capacity smoke completed 98,304 steps without broadphase overflow or unsafe
velocity termination before the formal launch.

At iteration 19, the formal run reached approximately 168k steps/s, mean reward
4.73, mean episode length 474.5, and zero unsafe-velocity terminations. These
startup values confirm health only; checkpoint selection will use full-episode
tracking metrics and visual playback rather than reward alone.

## Stage 0 evaluation and rollout capture

`scripts/firm/evaluate_expert.py` evaluates the expert from 25 evenly spaced
dense reference frames rather than replaying only frame zero. Each frame is
replicated across independently randomized environments. An episode succeeds
only if it reaches the time limit without the numerical safety termination and
ends with 25 consecutive stable-standing control steps.

The evaluator reports both tracking and safety quantities:

- success, timeout, and unsafe-termination rates overall and per start frame;
- world and root-relative MPKPE;
- joint-position RMSE and action-rate RMS;
- peak joint speed, joint acceleration, actuator force, and root vertical speed.

Use the local validated motion explicitly; the Stage 0 W&B run did not upload a
motion artifact.

```bash
uv run scripts/firm/evaluate_expert.py \
  --wandb-run-path tabletennis/smp/j0q8fell \
  --wandb-checkpoint-name model_29999.pt \
  --motion-file datasets/firm/lafan/fallAndGetUp2_subject2_candidate_003_validated.npz \
  --num-start-frames 25 \
  --episodes-per-frame 32 \
  --output-file datasets/firm/evaluation/c003_stage0_model_29999.json
```

The pilot recorder uses the same fixed-start protocol. With 25 starts, eight
replicas, and 500 steps, it writes at most 100,000 transitions:

```bash
uv run scripts/firm/collect_rollouts.py \
  --wandb-run-path tabletennis/smp/j0q8fell \
  --wandb-checkpoint-name model_29999.pt \
  --motion-file datasets/firm/lafan/fallAndGetUp2_subject2_candidate_003_validated.npz \
  --num-start-frames 25 \
  --episodes-per-frame 8 \
  --output-dir datasets/firm/rollouts/c003_stage0_model_29999_pilot
```

Rollout shards are compressed NPZ files with:

| Field | Shape | Meaning |
| --- | ---: | --- |
| `observation` | 90 | root angular velocity, joint position/velocity, previous action |
| `goal` | 29 | target sparse-keyframe joint position |
| `action` | 29 | clipped expert policy action |
| `episode_id`, `episode_step` | scalar | sequence and history boundaries |
| `start_frame`, `motion_frame`, `goal_frame` | scalar | reference provenance |
| `done`, `unsafe`, `timeout` | scalar | transition outcome |

`manifest.json` records checkpoint and motion SHA256 values, field layout,
per-episode success, and each shard's checksum. Failed episodes remain in the
pilot dataset but are explicitly labeled; downstream diffusion training must
filter or deliberately weight them.

### Formal Stage 0 evaluation record

The 800-episode evaluation (25 dense starts x 32 replicas, seed 42, observation
corruption enabled) completed on 2026-08-09:

| Metric | Result |
| --- | ---: |
| success rate | 99.00% |
| unsafe termination rate | 0.00% |
| mean MPKPE | 0.0351 m |
| mean root-relative MPKPE | 0.0715 m |
| mean joint-position RMSE | 0.0940 rad |
| mean action-rate RMS | 0.2707 |
| p95 peak joint speed | 22.80 rad/s |
| maximum joint speed | 24.49 rad/s |
| p95 peak root vertical speed | 1.40 m/s |

Frame 162 was the weakest dense start at 93.75% success. Frames 0, 81, 324,
364, 405, and 446 each reached 96.875%; every other sampled start reached
100%. Candidate 003 is therefore accepted as a functional pilot teacher, but
not as the final safety teacher: its peak joint and root speeds are too high for
a claim of gentle recovery.

The ignored evaluation JSON is stored at
`datasets/firm/evaluation/c003_stage0_model_29999.json` on `dsw-lyd2`;
its SHA256 is
`9624209fd13710a81b89f896e0eac34b902c21f69f65011359b79f33c42cdf31`.

### Pilot rollout record

The pilot capture contains 100,000 transitions from 200 completed episodes.
All 200 episodes met the stable-standing criterion and none reached the unsafe
termination. Two 50,000-transition shards occupy 43 MiB:

| Artifact | SHA256 |
| --- | --- |
| `manifest.json` | `edbb6fbb10a473600285187539572ba422b7e6dde175cbc7b37c8aea2c159a1c` |
| `shard_0000.npz` | `eaeaecf9f3f1270806fe8063d92bdacc7a654c65252c655c03bebd9141f6898d` |
| `shard_0001.npz` | `c23e6b53e0fc8d19b9a38b5c9a3056dcdb8137fd5d63d1cfe0ef5152e4e2ce22` |

The dataset remains local to
`/mnt/workspace/user/luyidan/smp-firm/datasets/firm/rollouts/c003_stage0_model_29999_pilot`.

## Planned task stages

### Rollout dataset

Each step records:

~~~text
observation = root angular velocity, q, qdot, previous action
goal        = target keyframe joint positions
action      = expert joint-position offset
~~~

A small pilot dataset will validate the action diffusion loop before scaling
toward the paper's 4.5 million samples.

### Action diffusion and adapter

The diffusion policy predicts a 12-action horizon and executes only the first
action. A Siamese goal/current-joint encoder produces a 64-dimensional relative
goal latent. The online adapter consumes 50 observation steps and refreshes the
retrieved keyframe goal every five control steps.

The pilot action model is implemented before the adapter. It uses:

- a 12-action prediction horizon;
- a shared 29-to-64 joint encoder for the current and goal poses;
- the difference between goal and current embeddings as the relative-goal
  latent;
- a separate 90-D state encoder;
- a conditional DiT epsilon-prediction denoiser with a 50-step cosine schedule.

Windows never cross episode boundaries and are retained only when the same
sparse-keyframe goal is valid for all 12 actions. Train/validation splitting is
performed by episode rather than by window. Normalization statistics are
computed from training episodes only.

```bash
CUDA_VISIBLE_DEVICES=2 uv run scripts/firm/train_action_diffusion.py \
  --manifest-file datasets/firm/rollouts/c003_stage0_model_29999_pilot/manifest.json \
  --horizon 12 \
  --batch-size 512 \
  --num-epochs 100 \
  --save-interval 50 \
  --run-name firm_action_diffusion_c003_pilot
```

Checkpoints are saved at epoch 0 and every 50 epochs, plus one final model; the
Adam optimizer state is intentionally omitted. With the default 5.24M-parameter
model this keeps the three expected files near 120 MiB total. Each contains the
online and EMA models, train-only normalization statistics, the dataset
manifest checksum, and window counts. This stage validates conditional
imitation only; closed-loop simulator recovery and autonomous keyframe
selection are separate acceptance gates.

### Pilot action diffusion record

Training completed on 2026-08-09 on `dsw-lyd2`, physical GPU 2, from code
commit `c366440`.

| Field | Result |
| --- | --- |
| W&B | [`tabletennis/smp/o0koto4w`](https://wandb.ai/tabletennis/smp/runs/o0koto4w) |
| model parameters | 5,237,842 |
| valid 12-step windows | 76,216 |
| train / validation windows | 68,316 / 7,900 |
| epoch 0 train / validation L1 | 0.35799 / 0.87668 |
| epoch 50 train / validation L1 | 0.15022 / 0.14491 |
| epoch 99 train / validation L1 | 0.14678 / 0.14326 |
| peak checkpoint disk use | 121 MiB |

The final optimizer-free checkpoint is:

```text
logs/firm_action_diffusion/firm_action_diffusion_c003_pilot/
  2026-08-09_21-48-33/firm_action_diffusion.pt
```

Its SHA256 is
`715623b036a379e1049fff358ac01fd3e620edee3957c57387d293999e316aba`.
The checkpoint embeds the pilot manifest SHA256
`edbb6fbb10a473600285187539572ba422b7e6dde175cbc7b37c8aea2c159a1c`,
the train-only normalization tensors, and the online and EMA weights.

After W&B reported all three files as uploaded, the local epoch-0 and epoch-50
copies were deleted when the shared CPFS filled. The server retains the 40 MiB
final model; both intermediate checkpoints remain recoverable from W&B run
`o0koto4w`.

The denoising loss converged without a train/validation gap. This accepts the
offline conditional-imitation pipeline, not the policy: DDPM sampling quality,
first-action error, and closed-loop recovery remain to be evaluated.

## Reproduction milestones

- [x] Isolate work on branch `repro/firm-g1`.
- [x] Add deterministic candidate detection and manifest generation.
- [x] Visually inspect and physically replay candidate 003.
- [x] Implement and smoke-test one sparse-keyframe expert.
- [x] Train the Stage 0 candidate 003 expert.
- [x] Quantitatively evaluate the Stage 0 candidate 003 expert.
- [ ] Scale to five directional experts.
- [x] Collect and validate the pilot rollout dataset.
- [x] Train action diffusion without adapter.
- [ ] Evaluate action diffusion sampling and closed-loop recovery.
- [ ] Train the adapter and evaluate full FIRM-R.
