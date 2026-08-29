"""Runner for a new curriculum phase initialized from a frozen policy."""

from __future__ import annotations

from typing import Any

from mjlab.rl import MjlabOnPolicyRunner


class SmpCurriculumWarmStartRunner(MjlabOnPolicyRunner):
  """Load policy/value normalization but start a fresh optimizer and clock.

  T/P/U are new curriculum phases, not interrupted continuations of the flat
  PPO run. Loading the source optimizer would silently override their frozen
  learning rate; loading the source iteration would also shift the registered
  2k/5k/10k/final gates by 30k updates.
  """

  def load(
    self,
    path: str,
    load_cfg: dict[str, Any] | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    if load_cfg is not None:
      raise ValueError("curriculum warm start does not accept an external load_cfg")
    infos = super().load(
      path,
      load_cfg={
        "actor": True,
        "critic": True,
        "optimizer": False,
        "iteration": False,
        "rnd": True,
      },
      strict=strict,
      map_location=map_location,
    )
    self.current_learning_iteration = 0
    self.env.unwrapped.common_step_counter = 0
    return infos


__all__ = ["SmpCurriculumWarmStartRunner"]
