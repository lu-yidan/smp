# RA-L terrain and plate progression protocol

This protocol is frozen before the flat causal screen and its three independent
policy-seed confirmation are complete. Its machine-readable source of truth is
`docs/ral_terrain_plate_protocol.json`.

## Why the progression is factorial

Historical V3.6--V3.8 experiments established useful task mechanics, but they
used 96D or 384D actors, warm-started historical policies, and showed terrain
or plate forgetting. They are engineering evidence and failure-boundary
references, not quantitative evidence for the current 93D actor.

The new experiment separates two independent environment factors:

| phase | complex terrain | physical plate | purpose |
| --- | :---: | :---: | --- |
| T | yes | no | measure terrain recovery and edge robustness |
| P | no | yes | measure contact-rich escape without terrain confounding |
| U | yes | yes | test whether one deployable actor retains both capabilities |

T and P may run concurrently only after the flat three-seed aggregate passes.
U remains blocked until the same seeds pass both specialist matrices. This
ordering prevents a failed joint policy from hiding whether exploration,
terrain contact, or plate mechanics caused the failure.

## Observation and initialization contract

Every actor remains the selected 93D, one-frame policy: angular velocity,
projected gravity, joint position, joint velocity, and previous action. It
does not receive true base linear velocity, terrain type/level, a height map,
plate state, reset phase, contact truth, or future reference motion. The critic
may retain simulator-only observations during training, but actor exports and
deployment logs must pass the observation audit.

Each T/P/U seed continues from its matched independently trained flat-policy
checkpoint. This is a curriculum continuation of a policy lineage that began
randomly; it is not a new random initialization and must not be described as
one. Historical V3.3/V3.8 checkpoints are prohibited. U restarts from the
matched flat checkpoint rather than selecting whichever T or P specialist is
easier to merge.

## Frozen compute and selection budget

T and P each use three policy seeds, 4,096 environments, 24 transitions per
update, and at most 20,000 updates. Checkpoints at 2k, 5k, 10k, and final are
evaluated with a held-out rollout seed. Training reward cannot select a model.
The maximum is 1,966,080,000 simulator transitions per seed and phase.

The flat aggregate first must retain mean native-GSI success of 95%, fixed-pose
macro success of 80%, and worst-pose success of 60%. In addition, every policy
seed must reach 70% macro and 45% worst pose. A deterministic candidate ranking
allocates compute but is not a statistical superiority claim when policy-level
intervals overlap.

## Phase T: terrain without a plate

Training retains 30% flat replay and uses 20% slope, 35% stairs, and 15% rough
terrain. Fall poses are balanced. Stairs explicitly sample center, near-edge,
straddle, and lower-tread resets; levels 0/1/2 remain present instead of relying
on a success-only curriculum that can forget easier support regions.

Promotion requires flat retention, level-0 and level-1 nonflat success, stair
edge coverage, less than 2% terrain exit, zero invalid dynamics, low secondary
falls, and safety distributions no more than 20% above the matched flat seed.
Every terrain, pose, level, and edge cohort is reported separately.

## Phase P: a physical plate on flat ground

Half of training remains unpinned flat recovery. The pinned half balances true
prone and supine resets under the audited 0.90 x 0.64 x 0.07 m plate, with
4/8/12 kg mass, friction, and planar coverage offsets. The plate is a passive
vertical-prismatic body; after contact neither the robot nor plate may be
teleported. A positive initial gap and invalid-setup metric are mandatory.

Promotion requires unpinned retention, stratified escape-and-stable-stand
success, at most 2% invalid setup, zero invalid dynamics, low secondary falls,
and matched safety distributions. Side-lying remains in the unpinned retention
set but is not labeled as a pinned result until a physically valid side-contact
fixture is separately defined.

## Phase U: unified recovery

U uses 25% unpinned flat, 45% unpinned terrain, 20% flat plate, and 10% plate on
stair centers. A first paper version deliberately excludes plate placement on
slope, rough terrain, or stair edges because a horizontal vertical-only plate
can collide with the terrain and turn setup failure into apparent policy
failure. Those combinations are future OOD tests only after their physics is
audited.

One actor must pass all T and P gates without a privileged selector. Its macro
performance may regress by at most five percentage points from each matched
specialist. Results are stratified rather than pooled into an easy overall
average.

## Claim boundary

Passing T, P, or U in simulation does not establish real-robot validity. The
RA-L claim additionally needs three policy seeds, policy-level uncertainty,
strong matched baselines, safety thresholds grounded in the G1 deployment
configuration, and the preregistered real-robot trial matrix. Failure to pass a
frozen gate produces `NO_PROMOTION`; it does not authorize post-hoc threshold
relaxation.

## Implemented task composition (not experimental evidence)

The code registers candidate-specific task IDs
`Smp-Getup-RAL-T-A{0..7}-G1` and `Smp-Getup-RAL-P-A{0..7}-G1`.
They are configuration templates only: no T/P run may start until the flat
three-policy-seed prerequisite passes and stable selection identifies the
corresponding arm.

Both templates begin from that arm's 93D one-frame flat configuration, remove
automatic pushes, and preserve its prior, off-manifold termination choice, and
procedural SMP floor. T composes generated terrain, frozen family/level
sampling, stair-edge sampling, terrain-aware grounding, and terrain safety
metrics. P composes the audited passive vertical plate, a joint reset
distribution giving 50% unpinned balanced poses and 50% pinned prone/supine
poses, categorical 4/8/12 kg loads, friction and planar-offset randomization,
and physical-validity metrics.

The registered tasks deliberately do not use the historical V3.6/V3.8 actor
configurations. Structural tests in `tests/test_ral_progression_cfg.py` verify
the observation contract, distribution parameters, event order, and reward
bridge inheritance. A successful structural test is prerequisite engineering
evidence, not a terrain or plate performance result.

T/P use `SmpCurriculumWarmStartRunner`: actor, critic, and observation
normalizers are initialized from each seed's matched confirmed-flat checkpoint,
while optimizer state, PPO iteration, and environment curriculum time restart
at zero. This makes the frozen specialist learning rate and relative
2k/5k/10k/final gates effective; ordinary `resume` would silently restore the
flat optimizer and shift every checkpoint by 30k updates.

Automatic promotion is fail-closed. `select_smp_confirmed_flat_arm.py` checks
the mean and every-seed flat gates, finite actions, source-summary hashes, seed
provenance, and checkpoint hashes before selecting at most one arm for compute
allocation. `run_smp_tp_physics_smoke.py` must then step the registered T and P
environments from the matched flat checkpoint on an otherwise idle GPU and
produce a commit- and promotion-bound `PASS` artifact. Only then may
`launch_smp_tp_specialists.py` start the six matched-seed jobs. This selection
is an allocation rule, not a statistical superiority claim.

After all six specialist processes exit, the controller does not infer success
from training reward or process exit. It first audits all six logs, verifies the
saved policy and environment seeds, re-hashes the matched flat source, and
requires every frozen 2k/5k/10k/final checkpoint. Only then does
`build_smp_specialist_manifests.py` atomically record the full
T/P x three-seed x four-gate factorial as 24 immutable manifests. Missing,
duplicate, seed-mismatched, or changed checkpoints produce
`TP_SPECIALIST_ALERT`; a valid set produces `TP_SPECIALIST_READY_FOR_EVAL`.
This status means only that evaluation inputs are auditable. It is not a T/P
performance pass and does not unblock U until the separately stratified
matrices and frozen promotion gates are implemented and satisfied.
