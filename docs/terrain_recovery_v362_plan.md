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
2. Assign minimum terrain-level replay floors so hard-terrain environments
   cannot all collapse to level zero. Flat remains at level zero; within each
   non-flat family the floor mixture is 50% level 0, 30% level 1, 15% level 2,
   and 5% level 3. Environments may still progress above their floor.
3. Resume only from frozen V3.6 `model_102000.pt`.
4. Reduce PPO learning rate to `3e-5`.
5. Run 1,500 updates initially, save every 250 updates, and evaluate at every
   saved checkpoint.  Extend only if the strict selection gate is still
   improving.
6. Prepare the shared seed with action standard deviation 0.15. Only
   `distribution.std_param` changes; all deterministic actor and critic weights
   remain byte-for-byte equal to the frozen source.

## Paired deployability experiment

Train two otherwise identical policies with the same seed and source checkpoint:

- `Smp-Getup-Terrain-V362-G1` is the oracle control. Its actor receives the
  simulator base linear velocity.
- `Smp-Getup-Terrain-V362-Deploy-G1` replaces only the actor's three
  `base_lin_vel_b` values with zeros in training and play. The observation
  remains 96-dimensional, and the asymmetric critic keeps the simulator velocity.

The Deploy task matches the current real-robot observation contract. Reporting
both variants separates terrain-learning changes from the cost of removing an
unavailable deployment signal.

## Implementation validation

- Source checkpoint SHA-256:
  `2ebfa32eea590a40ebaec4bad237180c95d334fbe507ea7544e055ede70a2b55`.
- Fixed-std seed SHA-256:
  `aea692a3d65355f9ea25d980364fa4dbce08fbd8de4c59406f306a373ff86b03`.
- Both tasks loaded the 96-dimensional actor and 960-dimensional critic
  checkpoint in local GPU smoke tests.
- A 64-environment, 50-update curriculum smoke reached maximum level 3 and
  logged replay-floor mean 0.5156. The finite-sample expectation is about 0.525:
  30% flat at floor zero plus 70% non-flat with mean floor 0.75.
- Replay floors are applied only when an environment resets, so another active
  environment is never moved to a different terrain origin mid-episode.

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

## Formal paired runs (2026-08-25)

Launched on `dsw-lyd2` from server worktree `/mnt/workspace/user/luyidan/smp-v362`:

- source commit: `400edc0` on `codex/terrain-recovery-v362`;
- shared fixed-std seed SHA-256:
  `aea692a3d65355f9ea25d980364fa4dbce08fbd8de4c59406f306a373ff86b03`;
- oracle run: `tabletennis/smp/kw5w8bd2` on GPU 3;
- zero-velocity Deploy run: `tabletennis/smp/3utesmeb` on GPU 4;
- both use 4,096 environments, seed `20260825`, learning rate `3e-5`, 1,500
  updates, and save interval 250;
- frozen level-1 oracle/Deploy evaluations run concurrently on GPUs 1 and 2
  before either policy has been fine-tuned.
