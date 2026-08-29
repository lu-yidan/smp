"""Preregistered flat Tier-A baselines on one SHA-locked reset bank."""

from __future__ import annotations

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from smp.rl.rewards import task_only_reward, task_smp_product
from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.ral_progression_env_cfg import (
  ACTOR_TERMS,
  SCRATCH_ARM_BUILDERS,
)

BASELINE_METHOD_TASK_NAMES = {
  "task_only_ppo": "TaskOnly",
  "original_product_smp": "OriginalSMP",
  "proposed_smp_recovery": "ProposedSMP",
}
RESET_BANK_NUM_STATES = 262144


def _replace_resets_with_matched_bank(cfg, *, include_smp_window: bool) -> None:
  """Replace GSI/procedural resets without changing dynamics or task terms."""
  startup = EventTermCfg(
    func=mdp.init_matched_reset_bank,
    mode="startup",
    params={
      "bank_path": "",
      "bank_sha256": "",
      "expected_num_states": RESET_BANK_NUM_STATES,
      "include_smp_window": include_smp_window,
      "sampling_seed": None,
    },
  )
  reset = EventTermCfg(func=mdp.matched_reset_bank_reset, mode="reset")
  source = cfg.events
  if "init_smp_state" not in source or "gsi_reset" not in source:
    raise RuntimeError("selected arm lacks the frozen SMP startup/reset anchors")
  events = {
    name: term
    for name, term in source.items()
    if name not in ("init_smp_state", "gsi_reset", "gsi_refresh", "mixed_fall_reset")
  }
  # Common domain randomization runs before method-specific SMP initialization,
  # so its global RNG draws remain comparable across methods.
  if include_smp_window:
    events["init_smp_state"] = source["init_smp_state"]
  events["init_matched_reset_bank"] = startup
  events["matched_reset_bank_reset"] = reset
  cfg.events = events


def _selected_arm_cfg(arm: str, play: bool):
  try:
    cfg = SCRATCH_ARM_BUILDERS[arm](play=play)
  except KeyError as exc:
    raise ValueError(f"unknown scratch arm {arm!r}") from exc
  actor = cfg.observations["actor"]
  if tuple(actor.terms) != ACTOR_TERMS or actor.history_length is not None:
    raise RuntimeError("Tier-A baselines require the exact 93D one-frame actor")
  return cfg


def _task_only_cfg(arm: str, play: bool):
  cfg = _selected_arm_cfg(arm, play)
  product = cfg.rewards.pop("task_smp_product")
  cfg.rewards["task_only"] = RewardTermCfg(
    func=task_only_reward,
    weight=product.weight,
    params={"task_terms": product.params["task_terms"]},
  )
  cfg.terminations.pop("smp_too_low", None)
  for name in ("smp_score", "raw_smp_score", "product_score"):
    cfg.metrics.pop(name, None)
  _replace_resets_with_matched_bank(cfg, include_smp_window=False)
  return cfg


def _original_product_cfg(arm: str, play: bool):
  cfg = _selected_arm_cfg(arm, play)
  reward = cfg.rewards["task_smp_product"]
  reward.func = task_smp_product
  reward.params.pop("procedural_smp_floor", None)
  reward.params.pop("smp_floor", None)
  cfg.terminations["smp_too_low"].func = mdp.smp_too_low
  _replace_resets_with_matched_bank(cfg, include_smp_window=True)
  return cfg


def _proposed_smp_cfg(arm: str, play: bool):
  cfg = _selected_arm_cfg(arm, play)
  _replace_resets_with_matched_bank(cfg, include_smp_window=True)
  return cfg


def g1_ral_baseline_env_cfg(method: str, arm: str, play: bool = False):
  """Build a frozen flat baseline while varying only the registered method."""
  builders = {
    "task_only_ppo": _task_only_cfg,
    "original_product_smp": _original_product_cfg,
    "proposed_smp_recovery": _proposed_smp_cfg,
  }
  try:
    return builders[method](arm, play)
  except KeyError as exc:
    raise ValueError(f"unknown Tier-A baseline method {method!r}") from exc


__all__ = [
  "BASELINE_METHOD_TASK_NAMES",
  "RESET_BANK_NUM_STATES",
  "g1_ral_baseline_env_cfg",
]
