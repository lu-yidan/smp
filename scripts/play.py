"""Play SMP tasks with independent disturbance and task-obstacle switches."""

from __future__ import annotations

import os
import sys

_AUTO_DISTURBANCES_ENV = "SMP_PLAY_AUTO_DISTURBANCES"
_ESCAPE_OBSTACLE_ENV = "SMP_PLAY_ESCAPE_OBSTACLE"
_ESCAPE_RESET_POSE_ENV = "SMP_PLAY_ESCAPE_RESET_POSE"


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


def _chosen_task(argv: list[str]) -> str | None:
  """Return the positional task ID expected by mjlab's play entry point."""
  if len(argv) < 2 or argv[1].startswith("-"):
    return None
  return argv[1]


def main() -> None:
  auto_disturbances = _consume_auto_disturbances_arg(sys.argv)
  escape_obstacle = _consume_escape_obstacle_arg(sys.argv)
  escape_reset_pose = _consume_escape_reset_pose_arg(sys.argv)
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

  # Task configs are constructed during this import, after the flag is known.
  from mjlab.scripts.play import main as mjlab_play_main
  from mjlab.tasks.registry import load_env_cfg

  import smp.rl.tasks  # noqa: F401

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
  mjlab_play_main()


if __name__ == "__main__":
  main()
