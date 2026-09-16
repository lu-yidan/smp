"""V33 escape shaping, kept outside SMP multiplication."""
from __future__ import annotations
import torch
from mjlab.envs import ManagerBasedRlEnv

def _contact_found(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  found = env.scene[sensor_name].data.found
  if found is None:
    raise RuntimeError(f"{sensor_name} must expose the 'found' contact field")
  return found > 0


def escape_separation_progress(
  env: ManagerBasedRlEnv,
  progress_scale: float = 0.025,
) -> torch.Tensor:
  """Reward only new robot-obstacle planar separation, not oscillation."""
  phase = getattr(env, "_escape_phase", None)
  delta = getattr(env, "_escape_separation_delta", None)
  if phase is None or delta is None:
    return torch.zeros(env.num_envs, device=env.device)
  return (phase == 2).float() * torch.clamp(delta / progress_scale, 0.0, 1.0)


def escape_geometry_progress(
  env: ManagerBasedRlEnv,
  sensor_name: str = "hand_ground_contact",
  coverage_scale: float = 0.025,
  clearance_scale: float = 0.02,
  max_head_height: float = 0.90,
) -> torch.Tensor:
  """Reward all-body footprint clearance gained while hand-supported.

  Unlike centre-distance progress, this term cannot be maximized while the
  head, torso, hand, or foot remains under a plate edge.
  """
  phase = getattr(env, "_escape_phase", None)
  coverage_delta = getattr(env, "_escape_coverage_delta", None)
  clearance_delta = getattr(env, "_escape_clearance_delta", None)
  if phase is None or coverage_delta is None or clearance_delta is None:
    return torch.zeros(env.num_envs, device=env.device)
  support_fraction = _contact_found(env, sensor_name).float().mean(dim=-1)
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  low_pose = robot.data.site_pos_w[:, head_idx, 2] <= max_head_height
  progress = torch.clamp(coverage_delta / coverage_scale, 0.0, 1.0)
  progress += 0.5 * torch.clamp(clearance_delta / clearance_scale, 0.0, 1.0)
  return (phase == 2).float() * low_pose.float() * support_fraction * progress


def escape_geometry_clearance_score(
  env: ManagerBasedRlEnv,
  target_clearance: float = 0.04,
) -> torch.Tensor:
  """Small dense score for reducing covered geoms and clearing the last edge."""
  phase = getattr(env, "_escape_phase", None)
  covered = getattr(env, "_escape_covered_geom_count", None)
  initial = getattr(env, "_escape_initial_covered_geom_count", None)
  clearance = getattr(env, "_escape_planar_clearance", None)
  if phase is None or covered is None or initial is None or clearance is None:
    return torch.zeros(env.num_envs, device=env.device)
  denominator = torch.clamp(initial.float(), min=1.0)
  uncovered_fraction = torch.clamp(
    1.0 - covered.float() / denominator, min=0.0, max=1.0
  )
  clearance_score = torch.clamp(clearance / target_clearance, 0.0, 1.0)
  active = (phase == 2) | (phase == 3)
  return active.float() * (0.85 * uncovered_fraction + 0.15 * clearance_score)


def escape_completion(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Persistent success signal after stable contact-free separation."""
  phase = getattr(env, "_escape_phase", None)
  if phase is None:
    return torch.zeros(env.num_envs, device=env.device)
  return (phase == 3).float()


def escape_contact_force_excess_l2(
  env: ManagerBasedRlEnv,
  sensor_name: str = "robot_obstacle_contact",
  force_limit: float = 300.0,
  force_scale: float = 300.0,
) -> torch.Tensor:
  """Penalize striking the plate rather than establishing controlled support."""
  phase = getattr(env, "_escape_phase", None)
  force = env.scene[sensor_name].data.force
  if phase is None or force is None:
    return torch.zeros(env.num_envs, device=env.device)
  peak = torch.linalg.vector_norm(force, dim=-1).amax(dim=-1)
  excess = torch.clamp(peak - force_limit, min=0.0) / max(force_scale, 1e-6)
  constrained = (phase == 1) | (phase == 2)
  return constrained.float() * torch.square(excess)

