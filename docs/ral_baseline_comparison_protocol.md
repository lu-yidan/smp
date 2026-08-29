# RA-L baseline comparison protocol

This protocol is frozen before selecting the proposed method's final
checkpoint. It prevents privileged observations, unmatched resets, larger
training budgets, or method-specific success definitions from silently making
one baseline easier.

## Method taxonomy

SMP is a task-independent diffusion prior reused as a frozen reward during
downstream policy learning. FIRM combines sparse fall/recovery demonstrations,
augmented tracking experts, an action diffusion policy, and online keyframe
adaptation. They therefore answer different questions and require both a
matched deployment comparison and a native-method reference.

The quantitative table has three tiers:

### Tier A: matched deployable recovery policies

All Tier-A actors receive the same 93D single-frame observation: IMU angular
velocity, projected gravity, joint position, joint velocity, and previous
action. True base linear velocity, terrain labels, plate pose/contact, motion
phase, reference identity, future keyframes, and simulator contact forces are
forbidden at deployment.

1. **Task-only PPO:** identical task, reset distribution, network, safety
   limits, and curriculum, with no motion-prior reward, termination, actor
   input, or deployed runtime. It may read the shared offline reset-state bank
   defined below; this is held constant across every Tier-A method.
2. **Original-product SMP:** the reproduced task-times-SMP objective, frozen
   prior, and original GSI reset/termination recipe, using the deployable actor.
3. **Proposed SMP recovery:** the checkpoint-stable winner selected by the
   eight-arm causal screen, before terrain or plate extensions.
4. **FIRM-R deployable adaptation:** the accepted FIRM-R reproduction with no
   phase, reference identity, future state, or privileged velocity exposed to
   the deployed policy. Any deterministic rescue, expert override, or
   hand-authored goal selector is a separately named ablation, not FIRM.
5. **Recovery tracking baseline:** sparse/dense recovery keyframe tracking with
   the same deployable proprioception. It measures whether demonstrations alone
   explain performance without diffusion adaptation or SMP guidance.

### Tier B: native-method references, reported separately

- Native SMP/MimicKit-style observation and reward configuration.
- Native FIRM-style goal/adapter configuration required by the closest
  paper-faithful reproduction.
- Historical 96D V3.3 plate policy and other finetuned checkpoints.

These references may use information absent from the real robot or different
training data. They are useful upper bounds and historical controls but must
not appear in the main deployable ranking or support a sim-to-real claim.

### Tier C: recent external methods

StableMimic and UniReLo are included in the related-work and capability table.
Direct numerical comparison is permitted only if official code/checkpoints can
be evaluated on the same G1 embodiment and frozen reset protocol. Otherwise
their published numbers remain clearly labeled as external, unmatched results.
An in-house approximation must be named `StableMimic-style` or
`UniReLo-style`, with every divergence listed; it cannot be labeled an exact
reproduction.

Primary references:

- SMP: https://arxiv.org/abs/2512.03028
- FIRM: https://arxiv.org/abs/2511.07407
- StableMimic: https://arxiv.org/abs/2608.02385
- UniReLo: https://arxiv.org/abs/2606.08922

## Frozen training budget

For every Tier-A method and policy seed:

- same MuJoCo G1 model, action scale, PD controller, control/physics period,
  joint limits, collision model, episode length, and domain randomization;
- same actor/critic width unless a method intrinsically requires another
  architecture, in which case parameter count and inference latency are
  reported;
- random actor/critic initialization; external recovery-policy checkpoints are
  prohibited;
- three independent policy-training seeds after the rapid one-seed screen;
- maximum 30,000 PPO updates, 4,096 environments, and 24 transitions per
  environment per update (2,949,120,000 transitions per full run);
- immutable checkpoints at 8k, 15k, 25k, and final, with SHA-256, code commit,
  task, training seed, and W&B run recorded.

Report success versus environment transitions at every gate. A method that
meets the frozen promotion threshold earlier retains that sample-efficiency
advantage; training all methods to the maximum budget does not erase it. Do not
retune one method after viewing the held-out evaluation. Any method-specific
hyperparameter search receives the same declared trial budget and separate
selection/evaluation seeds.

Motion-data budgets are reported separately from simulator transitions. SMP
prior windows, FIRM recovery demonstrations, augmented expert rollouts, and
any StableMimic-style references must list source clips, duration, frame count,
processing code, license, and whether test states were used during collection.

## Shared reset-state contract

The GSI reset distribution is generated by a motion prior, so simply removing
the prior from Task-only PPO would silently change its training states. Before
Tier-A training, generate one immutable reset-state bank from the selected
flat arm's exact GSI/procedural mixture. The frozen contract is:

- 262,144 states, generation seed 20260920, with SHA-256 and source-prior hash;
- root pose, root velocity, joint position, joint velocity, reset family, and
  the exact 10-frame, 59D SMP history used at that state;
- identical bank and sampling weights for every Tier-A method, with only the
  per-policy-seed permutation differing;
- reset family is logged for analysis but never exposed to the actor;
- held-out frozen evaluation states are disjoint from this training bank.

The bank is an input artifact, not a rollout result. Its manifest must bind the
selected three-seed flat promotion, source-prior hash, generator code commit,
state/history tensor shapes, reset-family counts, bank hash, and frozen
registry hash. A current-state-only bank is invalid because it would give SMP
methods a different history from the state that Task-only PPO receives.

Task-only PPO therefore receives the same state distribution without using a
motion prior in its objective, termination, policy input, or deployment. If a
method cannot consume this reset interface, it is reported as an unmatched
native reference rather than being placed in the main Tier-A ranking.

The native Task-only, Original-product SMP, and Proposed SMP tasks are
preregistered for every candidate arm as
`Smp-Getup-RAL-B-{TaskOnly,OriginalSMP,ProposedSMP}-A{0..7}-G1`. The launch
process must fill the bank path and SHA placeholders from the immutable runtime
registry; an empty placeholder is intentionally non-runnable. Task-only removes
the SMP startup, objective, metrics, and low-prior termination. Original-product
restores the exact product and global low-SMP termination while retaining the
selected prior. Proposed preserves the selected arm's reset-aware termination
and procedural bridge. All three replace GSI/procedural reset events with the
same bank loader, and only SMP methods consume the matched 10-frame history.
Reset sampling uses a dedicated policy-seed permutation and per-environment
cursor, not the global Torch RNG consumed by SMP inference. Thus the same
environment's nth reset is identical across native methods even when their
termination times differ. Common friction, encoder-bias, and centre-of-mass
randomization also runs before method-specific SMP initialization.

The machine-readable preregistration is
`docs/ral_baseline_registry.json`. Before the three native methods launch,
`scripts/audit_smp_baseline_registry.py` must validate the reset bank and mark
their individual rows `ready_for_training`; the full-registry status correctly
remains `BASELINES_BLOCKED` while FIRM-R or tracking still lacks an accepted
adapter. A full `BASELINES_READY_FOR_TRAINING` report is required only when all
five Tier-A rows enter one launch campaign.
After the three-seed flat promotion and only when all training GPUs are idle,
`advance_smp_ral_pipeline.py --launch-baseline-bank-when-ready` backgrounds
the bank generator, records its PID/log/plan ID, and refuses T/P smoke until
the bank, manifest, runtime registry, hashes, state shapes, and 10-frame
history all validate. A dead process with partial artifacts remains an alert;
it is never silently relaunched or regenerated.

After the reset bank is ready, `launch_smp_native_baselines.py` creates one
immutable plan for Task-only PPO, Original-product SMP, and Proposed SMP over
the three registered policy/environment seeds. It binds the flat-promotion,
runtime-registry, reset-bank, bank-manifest, and code hashes; forces random
actor/critic initialization; and records every command, GPU, log, seed, task,
and run name. Nine jobs are assigned round-robin to at most eight physical-GPU
workers, so a worker may run a second job only after its first job exits
successfully. The launcher refuses any active GPU process. T/P specialists have
queue priority; native baselines are launched only after the specialist jobs
and their GPU processes have exited. FIRM-R and tracking remain adapter-blocked
and are never silently substituted by these three native tasks.

When all nine native jobs finish, `build_smp_native_baseline_manifests.py`
verifies the full method-by-policy-seed factorial, saved random-init flag,
agent/environment seeds, 4,096 environments, 24-step rollout, 30k budget,
1,000-step checkpoint interval, exact 93D one-frame actor terms, and the saved
matched-bank path/SHA. It emits 12 immutable manifests for the four gates and
three policy seeds. These checkpoint manifests deliberately remain
`CHECKPOINTS_READY_EVALUATION_BANK_BLOCKED`: the five-pose comparison cannot run
until a separate SHA-locked held-out reset bank, disjoint from the training
bank, is generated and bound. Training checkpoints or their manifests are not
performance evidence.

`generate_smp_matched_eval_banks.py` materializes that held-out fixture only
after flat promotion and the training bank are SHA-valid. It creates 512 states
for each of native GSI, prone, supine, left-side, and right-side using frozen
seed 20260829 and the selected prior/procedural parameters. Every bank carries
the exact 10x59 history, mode counts, tensor shapes, and source hashes. The
generator verifies the reset-family counts separately, then compares root
state, joints, and velocities against the entire training bank and rejects any
exact physical-state overlap or duplicate held-out state. Partial outputs,
changed hashes, a live GPU process, or a different promotion fail closed.

`bind_smp_native_eval_banks.py` is the only promotion path from the 12 blocked
native checkpoint manifests to formal matrix manifests. It verifies every
checkpoint and held-out-bank hash, the complete 3-seed by 4-gate factorial, the
method set, protocol dimensions, selected-promotion lineage, training-bank SHA,
and zero exact overlap. It then emits a second immutable 12-manifest set with
status `READY_WITH_MATCHED_HELD_OUT_RESET_BANK`. The source checkpoint
manifests remain blocked and are never rewritten. Only the bound READY
manifests may enter `run_smp_frozen_eval_matrix.py`; the binding itself is still
not performance evidence.

The main evidence controller advances this branch only after the T/P specialist
queue reaches a terminal evaluated state. With no GPU process active, it first
freezes the held-out bank (before native baseline optimization), then launches
the nine native jobs. After all immutable worker queues finish, it builds and
binds the 12 seed-by-gate manifests, evaluates one bound manifest at a time, and
aggregates each gate with independently trained policy seed as the sampling
unit. A dead generator, worker, evaluator, partial artifact, or changed hash is
an alert state; the controller never silently restarts it. T/P retains queue
priority, and adapter baselines remain separately blocked.

After all 12 native matrices are complete, the preregistered paired analysis
uses the same policy seed as the pairing unit. It reports per-pose, macro,
worst-pose, and safety deltas for Proposed-minus-Original,
Proposed-minus-Task-only, and Original-minus-Task-only at every gate. The
primary support rule is frozen in the registry: final Proposed-minus-Original
worst-pose bootstrap lower bound above zero, macro non-inferiority within five
percentage points, Task-only worst-pose non-inferiority within five points, and
all Proposed seeds passing the 15k/25k/final stability contract. A complete
null comparison is retained and reported; it does not trigger threshold changes
or selective reruns.

## Frozen flat evaluation

Use `evaluate_smp_baseline.py` schema 2 for every compatible policy:

- `native_gsi`, `prone`, `supine`, `left_side`, and `right_side`;
- 512 environments per pose and held-out evaluation seed 20260829;
- 500 control steps;
- strict stable-standing success and identical timeout/termination handling;
- success counts and Wilson intervals;
- macro and worst-pose success;
- foot slip, root drift, secondary falls, final foot separation, action first
  and second difference, qvel, torque, and power distributions.

For policies whose runtime interface is not compatible with the PPO evaluator,
write a method adapter that emits the same schema and explicitly records extra
inputs, history, diffusion samples, inference rate, and interventions. The
analyzer must reject a row with a different success definition or missing
safety fields rather than filling it with zeros.

## Terrain and plate extensions

Only policies passing the flat three-seed gate advance. Evaluate flat, slope,
rough, stair-center, and stair-edge strata first without a plate; then evaluate
the physical plate on flat ground; finally evaluate combined terrain-plus-plate
conditions. Use identical pose, difficulty, mass, coverage, friction, and
geometry strata for every promoted method. Do not pool environments or report
only the easiest terrain average. Report macro, worst stratum, recovery time,
secondary fall/roll-off, hand support, displacement, and safety distributions.

## Claim boundary

The current interactive FIRM-R playback failure under mouse dragging is a
diagnostic, not a publication result. FIRM enters the main table only after its
adapter, persistent post-reference behavior, deployable observation audit,
checkpoint provenance, and frozen matrix pass. If this cannot be completed, the
paper reports the reproduction faithfully as incomplete and uses Tier-A
task-only PPO, original-product SMP, tracking, and causal ablations as the
quantitative baselines instead of overstating a broken implementation.

No method is declared superior from training reward, a single policy seed,
selected videos, or unmatched published numbers.
