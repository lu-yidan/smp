"""Play SMP tasks with an optional automatic-disturbance switch."""

from __future__ import annotations

import os
import sys

_AUTO_DISTURBANCES_ENV = "SMP_PLAY_AUTO_DISTURBANCES"


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


def main() -> None:
  auto_disturbances = _consume_auto_disturbances_arg(sys.argv)
  if auto_disturbances:
    os.environ[_AUTO_DISTURBANCES_ENV] = "1"
  else:
    os.environ.pop(_AUTO_DISTURBANCES_ENV, None)

  # Task configs are constructed during this import, after the flag is known.
  from mjlab.scripts.play import main as mjlab_play_main

  import smp.rl.tasks  # noqa: F401

  state = "enabled" if auto_disturbances else "disabled"
  print(f"[INFO] Automatic physical disturbances during play: {state}")
  mjlab_play_main()


if __name__ == "__main__":
  main()
