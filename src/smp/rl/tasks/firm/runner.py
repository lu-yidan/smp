"""Compatibility runner for local MJLab motion files."""

from __future__ import annotations

from typing import Any

from mjlab.rl.runner import MjlabOnPolicyRunner


class LocalMotionOnPolicyRunner(MjlabOnPolicyRunner):
  """Accept MJLab's tracking registry argument after local file resolution.

  The installed training wrapper passes registry_name to every tracking
  runner, while this MJLab release's base runner does not accept that keyword.
  The wrapper has already resolved the local motion path before construction,
  so discarding the optional registry label is safe.
  """

  def __init__(
    self,
    env,
    train_cfg: dict[str, Any],
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ) -> None:
    del registry_name
    super().__init__(env, train_cfg, log_dir, device)
