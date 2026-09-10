"""Numerical termination for the explicit continuous-recovery ablation."""

import torch


def unstable_sim_state(
  env,
  max_abs_qpos: float = 1.0e3,
  max_abs_qvel: float = 1.0e3,
  max_abs_qacc: float = 1.0e6,
) -> torch.Tensor:
  """Reset only environments whose raw MuJoCo state has become unsafe.

  This term runs before observations are recomputed.  It therefore prevents a
  single rare contact explosion from putting NaNs into the actor observation
  and aborting an otherwise healthy many-environment training run.
  """
  data = env.sim.data
  invalid = ~torch.isfinite(data.qpos).all(dim=-1)
  invalid |= ~torch.isfinite(data.qvel).all(dim=-1)
  invalid |= ~torch.isfinite(data.qacc).all(dim=-1)
  invalid |= ~torch.isfinite(data.qacc_warmstart).all(dim=-1)
  invalid |= ~torch.isfinite(data.sensordata).all(dim=-1)
  invalid |= data.qpos.abs().amax(dim=-1) > max_abs_qpos
  invalid |= data.qvel.abs().amax(dim=-1) > max_abs_qvel
  invalid |= data.qacc.abs().amax(dim=-1) > max_abs_qacc
  invalid |= data.qacc_warmstart.abs().amax(dim=-1) > max_abs_qacc
  return invalid
