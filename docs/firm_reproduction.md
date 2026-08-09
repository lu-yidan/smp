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

Because the shared CPFS reached capacity, the durable working copies for later
stages were moved to the default-login root filesystem:

- rollout: `/root/workspace/smp-firm-artifacts/c003_stage0_model_29999_pilot`;
- final action model:
  `/root/workspace/smp-firm-artifacts/checkpoints/firm_action_diffusion_c003_pilot.pt`;
- evaluation results: `/root/workspace/smp-firm-artifacts/evaluation`.

The directory occupies 83 MiB; the root filesystem had 62 GiB free after the
move. File hashes match the original CPFS copies.

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
first-action error, and closed-loop recovery are separate gates.

### Full DDPM sampling record

`scripts/firm/evaluate_action_diffusion.py` reconstructs the episode-level
held-out split, performs all 50 ancestral DDPM steps using EMA weights, and
compares sampled 12-step actions with expert targets. The complete 7,900-window
evaluation produced:

| Metric | Result |
| --- | ---: |
| finite-window rate | 100.00% |
| first-action RMSE / MAE | 0.0759 / 0.0401 |
| first-action per-window p50 / p95 RMSE | 0.0411 / 0.1447 |
| 12-step action RMSE / MAE | 0.1797 / 0.1144 |
| normalized 12-step RMSE | 0.2852 |
| target / predicted action RMS | 1.3675 / 1.3661 |
| target / predicted action-rate RMS | 0.1921 / 0.1742 |

The output is
`/root/workspace/smp-firm-artifacts/evaluation/action_diffusion_heldout_full.json`
with SHA256
`b404757d003d58b77fc7411f314c62368aeaa17f75f4e9e0ebb10bcf25ef4863`.
The matched action magnitude and lower predicted rate accept offline sampling
and rule out numerical scheduler instability for the trained checkpoint.

### Closed-loop action-policy record

`scripts/firm/evaluate_diffusion_policy.py` uses exactly the Stage 0 fixed-start
environment and metrics, samples a horizon from the current 90-D observation and
goal, and executes the first action before replanning. A 100-episode baseline
(25 dense starts x 4 replicas) produced:

| Metric | Expert | Diffusion |
| --- | ---: | ---: |
| episodes | 800 | 100 |
| success rate | 99.00% | 31.00% |
| unsafe termination rate | 0.00% | 0.00% |
| mean MPKPE | 0.0351 m | 0.3422 m |
| mean joint-position RMSE | 0.0940 rad | 0.1308 rad |
| mean action-rate RMS | 0.2707 | 0.2724 |
| p95 peak joint speed | 22.80 rad/s | 22.46 rad/s |
| p95 peak root vertical speed | 1.40 m/s | 3.07 m/s |

The diffusion result is
`/root/workspace/smp-firm-artifacts/evaluation/diffusion_policy_closed_loop_100ep.json`
with SHA256
`dfb49ba6027c97700a2c35152d774b3424b613bbdded87000afa14f914a373c2`.

A 20-episode diagnostic over frames 0, 122, 243, 364, and 486 reached 40%
success: all late-stage replicas at frames 364 and 486 succeeded, while every
early/middle recovery replica failed. Executing four sampled actions before
replanning reduced success to 0% and increased peak acceleration, so action
chunking is retained only as an ablation and the default remains one-step
receding-horizon execution.

### Corrective expert rollout and fine-tuning

The large offline/closed-loop gap is state-distribution shift, not insufficient
fitting of the original dataset. `scripts/firm/collect_rollouts.py` therefore
supports an opt-in contact-mediated torso wrench while the Stage 0 expert acts.
It records whether each transition was directly forced; the following states
contain the expert's corrective response.

The selected setting applies a 3--5 control-step wrench every 100--150 steps,
with up to 40 N per world-frame force component and 5 N m per torque component,
followed by a 40-step recovery period. A more frequent 20--40-step smoke reached
only 60% expert success and was rejected. The selected sparse setting reached
90% on its 10-episode smoke and was scaled to the same 25 x 8 schedule as the
pilot:

| Field | Result |
| --- | ---: |
| total transitions | 100,000 |
| episodes / successful | 200 / 187 |
| unsafe episodes | 0 |
| directly forced transitions | 2,617 (2.617%) |
| valid successful 12-step windows | 71,563 |
| train / validation windows | 64,338 / 7,225 |
| dataset size | 43 MiB |

The dataset is stored at
`/root/workspace/smp-firm-artifacts/c003_stage0_model_29999_corrective_force40_seed42`.
Its immutable hashes are:

| Artifact | SHA256 |
| --- | --- |
| `manifest.json` | `1e9918bc0349926a096a5ef180772e9be09a92198985b5faa953c9ed93225c9c` |
| `shard_0000.npz` | `f48c900f0e95dbaf0e82dbed10807930667cc0c0c37eff6f9e5d17040594bf08` |
| `shard_0001.npz` | `a3d0e4e99ac63de722a1fc998309f78dfdd11ca49abae3b3c70e8cb849c3a982` |

The corrective dataset already includes nominal motion between disturbances and
approximately 40 feedback steps after each wrench. Fine-tuning therefore starts
from the pilot EMA weights rather than duplicating the original nominal windows.
The trainer reuses the pilot normalization tensors exactly and writes the
source checkpoint path, epoch, weight type, and SHA256 into every new
checkpoint.

Corrective fine-tuning v1 launched from code commit `1eafb21` with 50 epochs,
batch size 512, learning rate `1e-4`, and EMA. `save_interval=100` keeps only
epoch 0 and the final checkpoint. Artifacts are written below
`/root/workspace/smp-firm-artifacts/training`; the W&B run is
[`tabletennis/smp/w9aneczd`](https://wandb.ai/tabletennis/smp/runs/w9aneczd).
Acceptance requires evaluation on both original and corrective held-out
episodes followed by the unchanged 100-episode closed-loop gate.

Fine-tuning v1 completed at epoch 49 with train/validation L1
`0.14540 / 0.14699`; its W&B run is
[`w9aneczd`](https://wandb.ai/tabletennis/smp/runs/w9aneczd), and the final
checkpoint SHA256 is
`04ea0eb4bfb7194b54c1d86b51c002135180f5d89b3137c6e1532c6dcd8f12bd`.
It reduced original held-out first-action RMSE from 0.0759 to 0.0648 without
changing the 12-step RMSE, but improved closed-loop success only from 31% to
36%. Four-sample action averaging reduced action rate and acceleration but
reached 35%, showing that sampling variance was not the primary failure mode.

The second corrective pass targets only the 17 failed early/middle start frames
from 0 through 324. It uses 24 replicas per frame and the same sparse wrench:

| Field | Result |
| --- | ---: |
| total transitions | 204,000 |
| episodes / successful | 408 / 359 |
| unsafe episodes | 0 |
| directly forced transitions | 5,592 (2.741%) |
| valid successful 12-step windows | 120,681 |
| train / validation windows | 108,834 / 11,847 |
| dataset size | 88 MiB |

The targeted dataset lives at
`/root/workspace/smp-firm-artifacts/c003_earlymid_corrective_force40_seed42`.
Its manifest SHA256 is
`3b9fe74897c7707818f524a30ebd604b3634e64a7f78e059688c190c1182d88f`;
the five shard hashes are:

| Shard | SHA256 |
| --- | --- |
| `shard_0000.npz` | `231f5362c66f6e59c7635bd527a6e3012cd47c64eba081db44bdd8de759bacb6` |
| `shard_0001.npz` | `fa890e6ec4282f4b6886d1e7ce984cd16720d3b625b23aadcd8950d2b082527c` |
| `shard_0002.npz` | `d265a8f728fcbad29e3f05a1a044befb738fd248924bea345de3c47835d2770a` |
| `shard_0003.npz` | `e02faf84cf386e5c1e72a14e9b7e3d374229a155ef2e4d0f20d097ac90873f85` |
| `shard_0004.npz` | `adc5d1ee134a3949a5cebb2ea09b68fe0e6eddaad7e17a8153f1d8fd31dfcd41` |

Fine-tuning v2 warm-started from v1, used the same optimizer settings for 50
epochs, and ended at train/validation L1 `0.14586 / 0.14589`. Its W&B run is
[`pfv50i6w`](https://wandb.ai/tabletennis/smp/runs/pfv50i6w). The final
checkpoint is
`/root/workspace/smp-firm-artifacts/training/firm_action_diffusion_c003_earlymid_ft_v2/2026-08-09_22-56-46/firm_action_diffusion.pt`
with SHA256
`ed1ef872eae8e00fecd9be975f9d56ecbf3753ee744009e76dc10122cdf900f4`.

V2 retained the original distribution: original held-out first-action RMSE
improved to 0.0597 and 12-step RMSE remained 0.1792. On the targeted held-out
set its first-action and 12-step RMSE were 0.0691 and 0.1829.

The unchanged 100-episode closed-loop comparison is:

| Metric | Pilot | V1 | V2 | V2 ensemble-4 |
| --- | ---: | ---: | ---: | ---: |
| success rate | 31% | 36% | 54% | **55%** |
| unsafe termination rate | 0% | 0% | 1% | **0%** |
| mean MPKPE (m) | 0.342 | 0.301 | 0.217 | **0.213** |
| joint-position RMSE (rad) | 0.131 | 0.121 | 0.115 | **0.114** |
| action-rate RMS | 0.272 | 0.282 | 0.278 | **0.266** |
| p95 peak joint speed (rad/s) | 22.46 | 23.29 | 22.73 | **22.56** |
| p95 peak acceleration (rad/s2) | 3731 | 4540 | 4441 | **3803** |
| p95 peak root vertical speed (m/s) | 3.07 | 3.04 | 3.09 | **2.99** |

Four-sample averaging is therefore a useful safety ablation, not the source of
the recovery gain. V2 establishes that targeted corrective data materially
helps, but 55% remains far below the 99% expert and four-sample inference is
costlier. Additional offline expert rollouts now have diminishing returns. The
next stage should collect states visited by the diffusion policy itself and
attach corrective expert labels (DAgger-style or state cloning), then train an
adapter or residual policy on those true failure states.

The V2 result JSON hashes are:

| Evaluation | SHA256 |
| --- | --- |
| targeted held-out | `9feafe6e356ed19c8045a46fbda1eb69357b5b7e730ac9e7a14c091c05af4697` |
| original held-out | `1ac72be4bb9aee1f2bb747e30cb901fc686f5ecb53915d66c29f2d3e28581fba` |
| closed-loop single | `6a7e6ab9916c78a9cdbea64acbb8d155faff2c1a76db0c18ee1d480275c94a9c` |
| closed-loop ensemble-4 | `820731d268a20e7457f3159e6ad93b72e4f07d5f1127887c917984aebe65e8c6` |
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
- [x] Evaluate full DDPM held-out sampling.
- [x] Establish the first closed-loop diffusion baseline.
- [x] Aggregate perturbed corrective expert data.
- [x] Fine-tune v1/v2 on corrective expert data.
- [ ] Aggregate on-policy diffusion failure states and close the policy gap.
- [ ] Train the adapter and evaluate full FIRM-R.
