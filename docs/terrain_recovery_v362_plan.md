# V3.6.2 Anti-Collapse Terrain Specialization Plan

## Decision after V3.6.1

V3.6.1 completed, but no checkpoint is promoted.  The curriculum initially
advanced and then collapsed to level zero after iteration 105,000.  Continue
from the frozen V3.6 `model_102000.pt`, not from a V3.6.1 checkpoint.

## Frozen capability boundary

All numbers use the deterministic evaluator with 32 environments per terrain
and reset-pose pair, four reset poses, 500 control steps, and the unchanged
25-step stable-standing success gate.

| level | aggregate | flat | slope | stairs | rough | exit rate | peak power |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 99.4% | - | - | 97.7% | - | 0.0% | 134.1 W |
| 1 | 86.3% | 100.0% | 94.5% | 51.6% | 99.2% | 0.0% | 151.5 W |
| 2 | 51.0% | 100.0% | 12.5% | 0.0% | 91.4% | 1.6% | 213.4 W |
| 3 | 45.3% | 100.0% | 0.0% | 0.0% | 81.3% | 7.4% | 229.0 W |

The next experiment therefore targets slope and stairs.  Flat and rough terrain
must remain replayed as anti-forgetting controls.

## Controlled method changes

1. Replace the zero-able multiplicative task/SMP gate with a floored gate:

   `task * (smp_floor + (1 - smp_floor) * smp)`

   Start with `smp_floor = 0.15`.  This retains SMP preference while preserving
   a recovery gradient outside the motion-prior distribution.
2. Reserve fixed terrain-level cohorts instead of allowing every environment to
   fall back to level zero.  Initial replay mixture: 50% level 0, 30% level 1,
   15% level 2, and 5% level 3.  Keep terrain-family sampling balanced.
3. Resume only from frozen V3.6 `model_102000.pt`.
4. Reduce PPO learning rate to `3e-5`.
5. Run 1,500 updates initially, save every 250 updates, and evaluate at every
   saved checkpoint.  Extend only if the strict selection gate is still
   improving.

## Selection gate

A candidate is promoted only if all of the following hold:

- level 0 aggregate success is at least 98%;
- level 1 aggregate success is at least the 86.3% frozen baseline;
- level 2 slope and stairs improve without reducing rough below 90%;
- mean peak power does not exceed 190 W at level 1;
- invalid dynamics remain zero and terrain exit remains below 2%;
- no single reset pose loses more than five percentage points at level 1.

The first run is an ablation-quality experiment: the SMP floor and fixed replay
must be logged independently so the failure mechanism can be reported, even if
the candidate is not promoted.
