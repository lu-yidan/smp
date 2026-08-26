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
