# Motion data and the SMP prior

This note explains what is actually stored at each stage of the recovery-data
pipeline and how to inspect the robot motion before training.

## Mental model

The data flow is:

```text
LAFAN retargeted CSV
  -> recovery CSV clips
  -> 50 Hz sliding NPZ windows
  -> diffusion prior checkpoint
  -> local motion score + GSI during PPO
```

These objects are different:

1. **Raw CSV** is a motion trajectory. Each 30 Hz row contains root position,
   root quaternion, and 29 G1 joint angles. This can be played directly.
2. **Sliced CSV** is a shorter trajectory copied from the raw data, such as one
   fall-to-stand event. It is still directly playable.
3. **NPZ windows** are overlapping 10-frame samples after interpolation to
   50 Hz. Each window represents only 0.2 seconds and contains root, joint,
   end-effector, and velocity features.
4. **The prior checkpoint** is a neural denoiser trained on those windows. It is
   not a video, a policy, or a stored get-up trajectory.
5. **The PPO policy** produces robot actions. The prior scores whether the most
   recent 0.2 seconds resembles its training distribution and also supplies GSI
   reset states. It does not tell PPO that prone must be followed by kneeling,
   then crouching, then standing.

This distinction explains why a prior can contain natural kneeling and crouching
frames while the policy still selects the wrong long-horizon recovery route.

## Data locations

The external, read-only retargeted source is:

```text
/home/d080/workspace/LAFAN1_Retargeting_Dataset/g1/
```

The V6 general recovery dataset is generated under:

```text
datasets/csv/getup_lafan6_sliced/
datasets/npz/getup_lafan6_sliced/
datasets/pretrain_ckpt/pretrained_getup_lafan6_v6.pt
```

V7 visual-review candidates are generated under:

```text
datasets/csv/getup_lafan_prone_routes_v7_candidates/
```

That directory contains 48 CSVs (24 original front-down candidates and their
mirrors), `manifest.json`, and one flat `review.jsonl` record per clip. The
manifest deliberately contains `"approved_for_training": false`; automatic
stage labels are only navigation hints until the clips have been watched.

Regenerate the candidate directory from the immutable V6 data with:

```bash
uv run scripts/export_prone_route_candidates.py
```

The exporter refuses to write into a non-empty destination, so an earlier review
cannot be silently overwritten.

## Interactive MuJoCo playback

Play a short candidate that passed the initial automatic checks:

```bash
uv run scripts/play_csv_motion.py \
  --input-file \
  datasets/csv/getup_lafan_prone_routes_v7_candidates/fallAndGetUp1_subject1__recovery_006.csv
```

The player auto-loads a sibling `manifest.json` and displays the current
automatic stage. Controls are:

| Key | Action |
| --- | --- |
| Space | pause/resume |
| Left / Right | previous/next interpolated frame and pause |
| Up / Down | increase/decrease playback speed |
| Home / End | jump to first/last selected frame |
| L | toggle looping |

Play an exact zero-based, end-exclusive span from an original LAFAN file:

```bash
uv run scripts/play_csv_motion.py \
  --input-file \
  /home/d080/workspace/LAFAN1_Retargeting_Dataset/g1/fallAndGetUp3_subject1.csv \
  --start-frame 399 \
  --end-frame 469
```

Use `--speed 0.5` for slow motion. The displayed file frame remains the 30 Hz
CSV frame even though poses are interpolated and shown at the same 50 Hz used by
the SMP converter.

Validate a clip and its MuJoCo forward kinematics without opening a window:

```bash
uv run scripts/play_csv_motion.py \
  --input-file datasets/csv/getup_lafan_prone_routes_v7_candidates/CLIP.csv \
  --dry-run
```

## Review protocol

For each non-mirrored candidate:

1. Verify that the initial orientation is genuinely prone or oblique-prone.
2. Identify whether the torso is supported by the arms before the pelvis rises.
3. Accept either bilateral kneeling or a physically supported half-kneel.
4. Verify that at least one foot is placed under the body before crouch-to-stand.
5. Reject foot penetration, retargeting discontinuities, prolonged idle motion,
   and ballistic leg swings that are artifacts rather than useful recovery.
6. Review the mirror automatically only after its source clip passes.

The automatic stage detector uses root orientation/height, wrist and foot
heights, and individual left/right knee angles. It intentionally does not mark a
clip as training-ready.

## Recommended V7 experiment

Do not replace the general V6 prior immediately. First create a reviewed prone
route subset and compare:

1. V6 full policy and V6 prior (baseline).
2. Route-aware reward/state machine with the unchanged V6 prior.
3. The same route-aware policy plus phase-balanced GSI/reset sampling.
4. Optionally, a gated prone-route prior used only for prone and oblique-prone
   states, alongside the general V6 prior.

This separates the contribution of long-horizon route supervision from local
motion naturalness and gives a defensible ablation for a paper.
