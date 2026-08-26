# V3.7 Edge-Aware Stair Recovery

## Objective

V3.7 targets the failure mode exposed by V3.6.3: training resets were almost
always within 4 cm of the terrain origin, so the robot recovered near the
centre of the stair platform but degraded near an edge. V3.7 changes only the
stair reset distribution and terrain-relative bookkeeping. It does not add
terrain labels, edge distance, reset cohort, support height, contacts, or any
other privileged signal to the actor.

The primary policy is the deployable task
`Smp-Getup-Terrain-V37-Deploy-G1`. Its 96-dimensional actor input keeps the
three base-linear-velocity slots for checkpoint compatibility, but they are
identically zero in both training and deployment. The critic still receives
the simulator velocity during asymmetric actor-critic training. The Oracle
task retains the true actor velocity only for a controlled ablation and is not
the default training target.

## Stair reset cohorts

Only stair environments are stratified. Flat, slope, and rough resets retain
the V3.6.3 distribution and provide anti-forgetting replay.

| cohort | training weight | radial anchor range | purpose |
|---|---:|---:|---|
| centre | 50% | original reset, about 0–0.06 m | retain the selected V3.6.3 capability |
| near edge | 25% | 0.18–0.24 m | recover close to the top-platform edge |
| straddle | 15% | 0.27–0.34 m | body footprint spans the top edge and first tread |
| lower tread | 10% | 0.38–0.52 m | begin one or more support regions below the top |

The radial direction is sampled over positive and negative x/y axes with up to
12 cm tangential variation. The exact pyramid-stair height profile is evaluated
at nine conservative points of every collision-geometry AABB. The complete
robot is then translated vertically so no sampled support point starts below a
stair surface. This avoids both centre-plane assumptions and intentional
initial interpenetration.

The selected anchor and local support height are simulator-side reset state.
They are used only for grounding, rewards, terminations, metrics, and frozen
evaluation. They never enter actor observations.

## Local validation before training

The deploy checkpoint selected by V3.6.3 is the only training seed:

- path: `logs/rsl_rl/smp_getup_terrain_v363_deploy_g1/baseline_v363_deploy_strict_102049/model_102049.pt`;
- SHA-256: `7fe15d4214dca185d0b97ce0550184923c4e2ba89835573f45b13fd8656c7c80`;
- actor shape: 96, with base linear velocity replaced by exactly three zeros;
- critic shape: 960, retaining true simulator observations.

A 64-environment reset audit at 10 cm stairs produced the following actual
anchor offsets. All cases had zero invalid dynamics after stepping.

| cohort | min | median | max | median local support relative to top |
|---|---:|---:|---:|---:|
| centre | 0.009 m | 0.032 m | 0.054 m | 0.00 m |
| near edge | 0.181 m | 0.225 m | 0.262 m | 0.00 m |
| straddle | 0.271 m | 0.317 m | 0.356 m | -0.05 m |
| lower tread | 0.382 m | 0.458 m | 0.530 m | -0.05 m |

The frozen V3.6.3 policy also completed a short 8-environment rollout for all
four cohorts without invalid dynamics or patch exits. Its median displacement
rose to 0.35–0.42 m in only 1.2 s, confirming that the new distribution is a
real out-of-distribution test rather than a relabelling of centre resets.

## Training protocol

V3.7 inherits the conservative V3.6.3 optimizer: learning rate `1e-5`, PPO
clip 0.10, desired KL 0.005, two learning epochs, maximum gradient norm 0.5,
and checkpoint interval 50. Stage A is exactly 50 additional updates from the
selected Deploy `model_102049.pt`, using 4,096 environments. No Oracle policy
is trained unless the zero-velocity ablation is later required for the paper.

Training stops after Stage A for frozen evaluation. Further training is allowed
only in 50-update increments while both the old-distribution retention gates
and the new edge gates pass. This prevents the false-positive curriculum and
catastrophic forgetting observed in V3.6.2.

## Promotion gates

The unchanged V3.6.3 retention benchmark remains mandatory:

- level-0 aggregate success at least 95%;
- level-1 aggregate success at least 78%;
- level-1 stairs success at least 35%;
- level-1 rough success at least 90%;
- mean peak power below 190 W;
- terrain exit below 2% and invalid dynamics exactly zero.

The new frozen edge benchmark reports every combination of stair level 0/1,
four fall poses, and four reset cohorts separately. Initial Stage-A promotion
requires zero invalid dynamics and less than 2% terrain exits in every cohort,
no more than a five-percentage-point loss on the centre cohort relative to the
V3.6.3 seed, and a measurable gain in both straddle and lower-tread aggregate
success. The untrained seed establishes the absolute edge-success baseline;
the first trained checkpoint does not define its own acceptance threshold.

## Playback and evaluation

Select a deterministic cohort during playback:

```bash
uv run scripts/play.py Smp-Getup-Terrain-V37-Deploy-G1 \
  --checkpoint-file <checkpoint.pt> \
  --num-envs 1 --viewer native --no-terminations True \
  --terrain-edge-cohort straddle
```

Accepted playback values are `mixed`, `center`, `near-edge`, `straddle`, and
`lower-tread`. Formal evaluation uses underscore names:

```bash
uv run scripts/evaluate_terrain_recovery.py \
  --checkpoint <checkpoint.pt> \
  --task Smp-Getup-Terrain-V37-Deploy-G1 \
  --terrain-types stairs --levels 0 1 \
  --reset-modes prone supine left_side right_side \
  --edge-cohorts center near_edge straddle lower_tread \
  --num-envs 32 --steps 500 \
  --output logs/evaluation/v37_edge.jsonl
```

Each result records the requested cohort, actual reset-offset range, local
support-height change, success and recovery time, secondary falls, terrain
exit, invalid dynamics, planar drift/descent, foot slip, joint speed, torque,
and power.
## Stage-A result and controlled retry

The first 50-update Stage-A run, W&B `psj7jtvi`, produced
`model_102098.pt` (SHA-256
`4abb07b0accb41fb256589b015132267bf07100a0c6623c92d707335cf30d2bc`).
It is rejected and must not be deployed. It had zero invalid dynamics, but the
frozen benchmark showed edge regression rather than learning:

| level | cohort | seed 102049 | Stage A 102098 | delta |
|---|---|---:|---:|---:|
| 0 | centre | 95.31% | 85.16% | -10.16 pp |
| 0 | near edge | 91.41% | 77.34% | -14.06 pp |
| 0 | straddle | 93.75% | 79.69% | -14.06 pp |
| 0 | lower tread | 87.50% | 78.91% | -8.59 pp |
| 1 | centre | 32.03% | 29.69% | -2.34 pp |
| 1 | near edge | 25.78% | 15.62% | -10.16 pp |
| 1 | straddle | 25.00% | 17.19% | -7.81 pp |
| 1 | lower tread | 7.03% | 3.12% | -3.91 pp |

The old-distribution retention result was 94.92% at level 0 and 76.76% at
level 1, so both aggregate gates also failed. Training telemetry explains the
failure: only 25% of environments were stairs, hard edge cohorts were a small
fraction of those, while successful episodes quickly advanced the curriculum
to levels 2 and 3. Fifty on-policy updates therefore caused general policy
drift before sufficient edge data was collected.

The controlled retry restarts from the untouched `model_102049.pt`, never from
the rejected checkpoint. It makes the following pre-registered changes:

- train with 25% flat, 20% slope, 40% stairs, and 15% rough environments;
- within stairs use 40% centre and 20% for each edge cohort;
- cap adaptive training at level 1 with a 75% level-0 / 25% level-1 floor;
- reduce the learning rate to `5e-6`;
- run only 10 updates and save at that boundary.

This increases useful edge exposure while keeping 60% non-stair replay and a
large centre cohort. The frozen evaluator and promotion gates are unchanged.
## Controlled-retry result and selection

The controlled retry is W&B `yezkyihk`. Its `model_102058.pt` has SHA-256
`d66014d8ab9370f48002cff135a449e9d4665c7884e78a3543d77123ef62a36c`.
It restored all old-distribution retention gates:

| level | aggregate | stairs | rough | mean peak power | exit | invalid |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 98.83% | 97.66% | 100.00% | 138.9 W | 0.00% | 0.00% |
| 1 | 79.30% | 39.84% | 95.31% | 167.5 W | 0.78% | 0.00% |

However, it did not pass the new edge-improvement gate. Level-0 edge results
were retained, but the difficult level-1 straddle and lower-tread results did
not improve:

| level | cohort | seed 102049 | retry 102058 | delta |
|---|---|---:|---:|---:|
| 0 | centre | 95.31% | 96.09% | +0.78 pp |
| 0 | near edge | 91.41% | 92.19% | +0.78 pp |
| 0 | straddle | 93.75% | 93.75% | 0.00 pp |
| 0 | lower tread | 87.50% | 87.50% | 0.00 pp |
| 1 | centre | 32.03% | 33.59% | +1.56 pp |
| 1 | near edge | 25.78% | 25.00% | -0.78 pp |
| 1 | straddle | 25.00% | 20.31% | -4.69 pp |
| 1 | lower tread | 7.03% | 6.25% | -0.78 pp |

Consequently neither `model_102098.pt` nor `model_102058.pt` is promoted.
The selected deployable baseline remains V3.6.3 `model_102049.pt`. V3.7 is a
validated task and benchmark contribution, not yet an improved policy result.
The next experiment must target level-1 edge exposure explicitly and should
consider deployable contact/proprioceptive history or a seed-policy anchoring
loss; simply extending PPO is prohibited by these results.
