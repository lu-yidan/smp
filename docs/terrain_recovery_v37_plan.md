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
