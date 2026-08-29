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
   limits, and curriculum, with no frozen motion prior.
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
