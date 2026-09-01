"""Play SMP tasks with independent disturbance and task-obstacle switches."""

from __future__ import annotations

import os
import sys

_AUTO_DISTURBANCES_ENV = "SMP_PLAY_AUTO_DISTURBANCES"
_ESCAPE_OBSTACLE_ENV = "SMP_PLAY_ESCAPE_OBSTACLE"
_ESCAPE_RESET_POSE_ENV = "SMP_PLAY_ESCAPE_RESET_POSE"
_TERRAIN_TYPE_ENV = "SMP_PLAY_TERRAIN_TYPE"
_TERRAIN_LEVEL_ENV = "SMP_PLAY_TERRAIN_LEVEL"
_TERRAIN_RESET_POSE_ENV = "SMP_PLAY_TERRAIN_RESET_POSE"
_TERRAIN_EDGE_COHORT_ENV = "SMP_PLAY_TERRAIN_EDGE_COHORT"


def _consume_auto_disturbances_arg(argv: list[str]) -> bool:
  """Remove the wrapper-only boolean flag before mjlab's Tyro parser runs."""
  flag = "--auto-disturbances"
  count = argv.count(flag)
  if count == 0:
    return False
  if count > 1:
    raise SystemExit(f"{flag} may only be specified once")

  index = argv.index(flag)
  if index + 1 >= len(argv):
    raise SystemExit(f"{flag} requires True or False")
  value = argv[index + 1].lower()
  if value not in {"true", "false"}:
    raise SystemExit(f"{flag} requires True or False, got {argv[index + 1]!r}")
  del argv[index : index + 2]
  return value == "true"


def _consume_escape_obstacle_arg(argv: list[str]) -> bool | None:
  """Consume an optional escape-plate override from the wrapper arguments."""
  flag = "--escape-obstacle"
  count = argv.count(flag)
  if count == 0:
    return None
  if count > 1:
    raise SystemExit(f"{flag} may only be specified once")

  index = argv.index(flag)
  if index + 1 >= len(argv):
    raise SystemExit(f"{flag} requires True or False")
  value = argv[index + 1].lower()
  if value not in {"true", "false"}:
    raise SystemExit(f"{flag} requires True or False, got {argv[index + 1]!r}")
  del argv[index : index + 2]
  return value == "true"


def _consume_escape_reset_pose_arg(argv: list[str]) -> str | None:
  """Consume an optional V3.4 reset-pose selection."""
  flag = "--escape-reset-pose"
  count = argv.count(flag)
  if count == 0:
    return None
  if count > 1:
    raise SystemExit(f"{flag} may only be specified once")

  index = argv.index(flag)
  if index + 1 >= len(argv):
    raise SystemExit(f"{flag} requires mixed, prone, or supine")
  value = argv[index + 1].lower()
  if value not in {"mixed", "prone", "supine"}:
    raise SystemExit(
      f"{flag} requires mixed, prone, or supine, got {argv[index + 1]!r}"
    )
  del argv[index : index + 2]
  return value


def _consume_terrain_type_arg(argv: list[str]) -> str | None:
  """Consume an optional V3.5 terrain-family selection."""
  flag = "--terrain-type"
  count = argv.count(flag)
  if count == 0:
    return None
  if count > 1:
    raise SystemExit(f"{flag} may only be specified once")
  index = argv.index(flag)
  if index + 1 >= len(argv):
    raise SystemExit(f"{flag} requires flat, slope, stairs, rough, or mixed")
  value = argv[index + 1].lower()
  choices = {"flat", "slope", "stairs", "rough", "mixed"}
  if value not in choices:
    raise SystemExit(f"{flag} got unsupported value {argv[index + 1]!r}")
  del argv[index : index + 2]
  return value


def _consume_terrain_level_arg(argv: list[str]) -> int | None:
  """Consume a fixed V3.5 difficulty bin from zero through three."""
  flag = "--terrain-level"
  count = argv.count(flag)
  if count == 0:
    return None
  if count > 1:
    raise SystemExit(f"{flag} may only be specified once")
  index = argv.index(flag)
  if index + 1 >= len(argv):
    raise SystemExit(f"{flag} requires 0, 1, 2, or 3")
  try:
    value = int(argv[index + 1])
  except ValueError as exc:
    raise SystemExit(f"{flag} requires 0, 1, 2, or 3") from exc
  if value not in range(4):
    raise SystemExit(f"{flag} requires 0, 1, 2, or 3")
  del argv[index : index + 2]
  return value


def _consume_terrain_reset_pose_arg(argv: list[str]) -> str | None:
  """Consume an optional V3.5 physical reset-pose selection."""
  flag = "--terrain-reset-pose"
  count = argv.count(flag)
  if count == 0:
    return None
  if count > 1:
    raise SystemExit(f"{flag} may only be specified once")
  index = argv.index(flag)
  if index + 1 >= len(argv):
    raise SystemExit(f"{flag} requires mixed, prone, supine, left-side, or right-side")
  value = argv[index + 1].lower().replace("-", "_")
  choices = {"mixed", "prone", "supine", "left_side", "right_side"}
  if value not in choices:
    raise SystemExit(f"{flag} got unsupported value {argv[index + 1]!r}")
  del argv[index : index + 2]
  return value


def _consume_terrain_edge_cohort_arg(argv: list[str]) -> str | None:
  """Consume an optional V3.7 stair reset-location cohort."""
  flag = "--terrain-edge-cohort"
  count = argv.count(flag)
  if count == 0:
    return None
  if count > 1:
    raise SystemExit(f"{flag} may only be specified once")
  index = argv.index(flag)
  if index + 1 >= len(argv):
    raise SystemExit(
      f"{flag} requires mixed, center, near-edge, straddle, or lower-tread"
    )
  value = argv[index + 1].lower().replace("-", "_")
  choices = {"mixed", "center", "near_edge", "straddle", "lower_tread"}
  if value not in choices:
    raise SystemExit(f"{flag} got unsupported value {argv[index + 1]!r}")
  del argv[index : index + 2]
  return value


def _chosen_task(argv: list[str]) -> str | None:
  """Return the positional task ID expected by mjlab's play entry point."""
  if len(argv) < 2 or argv[1].startswith("-"):
    return None
  return argv[1]


def _load_inference_runner_cls(task_name: str):
  """Use a plain runner when a task custom runner is training-only."""
  from mjlab.rl import MjlabOnPolicyRunner
  from mjlab.tasks.registry import load_runner_cls

  from smp.rl.warm_start_runner import SmpCurriculumWarmStartRunner

  runner_cls = load_runner_cls(task_name) or MjlabOnPolicyRunner
  if issubclass(runner_cls, SmpCurriculumWarmStartRunner):
    return MjlabOnPolicyRunner
  return runner_cls


def main() -> None:
  auto_disturbances = _consume_auto_disturbances_arg(sys.argv)
  escape_obstacle = _consume_escape_obstacle_arg(sys.argv)
  escape_reset_pose = _consume_escape_reset_pose_arg(sys.argv)
  terrain_type = _consume_terrain_type_arg(sys.argv)
  terrain_level = _consume_terrain_level_arg(sys.argv)
  terrain_reset_pose = _consume_terrain_reset_pose_arg(sys.argv)
  terrain_edge_cohort = _consume_terrain_edge_cohort_arg(sys.argv)
  if auto_disturbances:
    os.environ[_AUTO_DISTURBANCES_ENV] = "1"
  else:
    os.environ.pop(_AUTO_DISTURBANCES_ENV, None)
  if escape_obstacle is None:
    os.environ.pop(_ESCAPE_OBSTACLE_ENV, None)
  else:
    os.environ[_ESCAPE_OBSTACLE_ENV] = "1" if escape_obstacle else "0"
  if escape_reset_pose is None:
    os.environ.pop(_ESCAPE_RESET_POSE_ENV, None)
  else:
    os.environ[_ESCAPE_RESET_POSE_ENV] = escape_reset_pose
  if terrain_type is None:
    os.environ.pop(_TERRAIN_TYPE_ENV, None)
  else:
    os.environ[_TERRAIN_TYPE_ENV] = terrain_type
  if terrain_level is None:
    os.environ.pop(_TERRAIN_LEVEL_ENV, None)
  else:
    os.environ[_TERRAIN_LEVEL_ENV] = str(terrain_level)
  if terrain_reset_pose is None:
    os.environ.pop(_TERRAIN_RESET_POSE_ENV, None)
  else:
    os.environ[_TERRAIN_RESET_POSE_ENV] = terrain_reset_pose
  if terrain_edge_cohort is None:
    os.environ.pop(_TERRAIN_EDGE_COHORT_ENV, None)
  else:
    os.environ[_TERRAIN_EDGE_COHORT_ENV] = terrain_edge_cohort

  # Task configs are constructed during this import, after the flag is known.
  import mjlab.scripts.play as mjlab_play
  from mjlab.tasks.registry import load_env_cfg

  import smp.rl.tasks  # noqa: F401

  # mjlab generic player requests an actor-only load. Curriculum warm-start
  # runners intentionally reject that training-inappropriate load contract, so
  # use the ordinary runner for inference without changing task registration.
  mjlab_play.load_runner_cls = _load_inference_runner_cls

  task_name = _chosen_task(sys.argv)
  if escape_obstacle and task_name is not None:
    task_cfg = load_env_cfg(task_name, play=True)
    if "escape_obstacle" not in task_cfg.scene.entities:
      raise SystemExit(
        f"{task_name} does not contain an escape obstacle. "
        "--escape-obstacle can enable an obstacle already defined by an escape "
        "task, but it cannot add one to a recovery-only task. Use "
        "Smp-Getup-Escape-Plate-V33-G1 to play with the plate."
      )

  state = "enabled" if auto_disturbances else "disabled"
  print(f"[INFO] Automatic physical disturbances during play: {state}")
  obstacle_state = (
    "task default"
    if escape_obstacle is None
    else ("enabled" if escape_obstacle else "disabled")
  )
  print(f"[INFO] Escape obstacle during play: {obstacle_state}")
  reset_pose_state = escape_reset_pose or "task default"
  print(f"[INFO] Escape reset pose during play: {reset_pose_state}")
  terrain_state = terrain_type or "task default"
  level_state = terrain_level if terrain_level is not None else "task default"
  terrain_pose_state = terrain_reset_pose or "task default"
  edge_state = terrain_edge_cohort or "task default"
  print(f"[INFO] Terrain during play: {terrain_state}, level: {level_state}")
  print(f"[INFO] Terrain reset pose during play: {terrain_pose_state}")
  print(f"[INFO] Terrain edge cohort during play: {edge_state}")
  mjlab_play.main()


if __name__ == "__main__":
  main()
