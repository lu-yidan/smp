# Procedural reset pose-label audit (2026-08-18)

## Finding

The first two modes of `mixed_fall_reset` were historically named in the wrong
order. A 128-environment direct torso-frame audit with zero orientation noise
measured:

| reset type | root pitch | torso forward-axis world z | physical pose |
|---:|---:|---:|---|
| 1 | +pi/2 | -0.997 mean | prone (chest down) |
| 2 | -pi/2 | +0.997 mean | supine (chest up) |

The corrected mode order is therefore **prone, supine, left side, right side**.
The actor observation and physical simulator state were always unchanged; this
was a semantic error in comments, forced-pose evaluation, stratified metrics,
and pose-gated rewards.

## Consequences

- V3--V3.3 plate escape checkpoints are physically **supine-only**, although
  their original documents called them prone. Their success numbers remain
  valid after relabeling the pose.
- The V3.4 mixed reset already sampled types 1 and 2 equally, so its physical
  50/50 coverage was correct. The run begun at commit `8e5ccd9` has swapped
  pose metric names in its training log, but PPO rewards in that task do not
  depend on those metrics.
- The pre-V3.4 source baseline must be read as 94.72% supine and 0% prone, not
  the reverse.
- Historical V8 `prone_*` pose-gated terms actually targeted physical supine.
  Those old checkpoints remain reproducible, but the old labels cannot support
  a claim about prone-specific natural recovery. Future V8 training uses the
  corrected type-1 prone mask.

## Code corrections

Forced-pose playback/evaluation, reset metrics, pose-gated natural-recovery
rewards, reset-distribution weights, and V3.2/V3.3 support-ready naming now use
the physical mapping above. Frozen checkpoints were not modified.
