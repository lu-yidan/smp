# V3.6.3 Conservative Terrain Adaptation Plan

## Why V3.6.2 is rejected

V3.6.2 finished without invalid dynamics, but strict deterministic evaluation
showed catastrophic forgetting. Level-1 success fell from 82.03%/80.47% for
the frozen Oracle/Deploy seeds to 1.95%/1.37% at `model_103499.pt`.

The online curriculum metric was a false positive. At the final checkpoint:

| run | curriculum success | stage success | strict stand success |
|---|---:|---:|---:|
| Oracle | 37.63% | 37.63% | 1.45% |
| Deploy | 50.90% | 50.90% | 2.74% |

`accept_completed_recovery_stage=True` therefore allowed a persistent stage
latch to advance curriculum without a stable standing recovery. The 0.15 SMP
floor and level-2/3 replay then accelerated policy drift. No V3.6.2 checkpoint
is promoted; frozen V3.6 `model_102000.pt` remains the baseline.

## V3.6.3 controlled changes

1. Inherit directly from V3.6, not V3.6.1 or V3.6.2.
2. Require the real 25-step stable-standing counter for curriculum success;
   stage completion is telemetry only and cannot advance terrain.
3. Restrict non-flat replay floors to 70% level 0 and 30% level 1. Levels 2 and
   3 receive no forced cohort until level-1 capability is retained.
4. Reduce the SMP floor from 0.15 to 0.03.
5. Use action standard deviation 0.10, learning rate `1e-5`, PPO clip 0.10,
   desired KL 0.005, two learning epochs, and maximum gradient norm 0.5.
6. Train only 50 updates in stage A and then stop for strict evaluation. Extend
   in 50-update increments only while the retention gate passes.

## Stage-A retention gate

Evaluate level 0 and level 1 with 32 environments per terrain/pose pair, four
reset poses, 500 control steps, and the unchanged 25-step success criterion.

- Level-0 aggregate success must remain at least 95%.
- Level-1 aggregate success must remain at least 78%.
- Level-1 stairs must remain at least 35%.
- Rough terrain must remain at least 90%.
- Mean peak power must remain below 190 W.
- Terrain exit must remain below 2%; invalid dynamics must remain zero.

Oracle and zero-velocity Deploy tasks use the same source checkpoint, seed, and
optimizer settings. This keeps deployment cost measurable without mixing it
with the terrain-adaptation changes.

## Local validation

- Fixed-std seed SHA-256:
  `497d0119acd9817c8f15c4548c759249c91b664794fa3f7af9f3700d9880a474`.
- Oracle and Deploy both loaded the unchanged 96-dimensional actor checkpoint.
- In a 64-environment, 50-update Oracle smoke test, curriculum success and
  strict stand success were both 87.5%; stage success was 0%.
- Replay-floor mean was 0.2188 (expected about 0.21), maximum terrain level was
  3, terrain exit was 0%, and invalid dynamics were 0%.

## Formal paired training and early stopping

Both policies resumed from the same fixed-std V3.6 seed for 50 updates. The
formal Stage-A W&B runs are:

- Oracle: `bploru7k`, final checkpoint `model_102049.pt`.
- zero-velocity Deploy: `v8y4i559`, final checkpoint `model_102049.pt`.

The frozen 32-environment-per-cell evaluation passed every retention gate:

| policy | level | success | stairs | rough | mean peak power | terrain exit |
|---|---:|---:|---:|---:|---:|---:|
| Oracle 102049 | 0 | 99.02% | 96.88% | 100.00% | 136.34 W | 0.00% |
| Oracle 102049 | 1 | 83.59% | 50.78% | 96.88% | 155.24 W | 0.00% |
| Deploy 102049 | 0 | 99.22% | 98.44% | 100.00% | 138.86 W | 0.00% |
| Deploy 102049 | 1 | 78.52% | 38.28% | 94.53% | 166.64 W | 0.59% |

One additional 50-update Stage-B interval was run as a controlled early-stop
probe (`47ebsn6e` Oracle and `0f2may0p` Deploy), producing
`model_102098.pt`. It was not promoted:

- Oracle level-0 success fell to 94.73%, below the 95% gate, and level-1 rough
  success fell to 89.06%, below the 90% gate.
- Deploy still narrowly passed the absolute gates, but level-1 aggregate fell
  from 78.52% to 78.32% and stairs fell from 38.28% to 35.16%.
- The small power and speed reductions did not compensate for the lost recovery
  success.

Training therefore stops after this probe. `model_102049.pt` is the selected
V3.6.3 checkpoint for both Oracle and deployable zero-velocity policies. The
local preserved copies are:

- `logs/rsl_rl/smp_getup_terrain_v363_g1/baseline_v363_strict_102049/model_102049.pt`
  (`b9493b9211b47557eba59c7875334dfff55233b8246004fd3ad9c43662de3912`).
- `logs/rsl_rl/smp_getup_terrain_v363_deploy_g1/baseline_v363_deploy_strict_102049/model_102049.pt`
  (`7fe15d4214dca185d0b97ce0550184923c4e2ba89835573f45b13fd8656c7c80`).

The strict raw results are retained under
`logs/evaluation/v363_stage_{a,b}/`. These generated artifacts are not committed
to Git; the code, decision rules, run IDs, checkpoint hashes, and promotion
decision are recorded here for reproduction.
