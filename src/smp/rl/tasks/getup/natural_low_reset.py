"""Small natural reset mixture and episode-specific persistent low-SMP termination."""

import math
import numpy as np
import torch

from smp.rl.events import gsi_reset
from smp.rl.tasks.getup.master_deployment_contract import JOINT_NAMES
from smp.rl.tasks.getup.mdp.terminations import smp_too_low


def prime_static_history(env, ids):
  robot = env.scene["robot"]
  origin = env.scene.env_origins[ids]
  buf = env._smp_buffer
  w = buf.window_size
  ee = robot.data.body_link_pos_w[ids][:, env._smp_ee_indexes] - origin[:, None]
  buf.reset(ids,
    (robot.data.root_link_pos_w[ids] - origin)[:, None].expand(-1, w, -1),
    robot.data.root_link_quat_w[ids][:, None].expand(-1, w, -1),
    robot.data.root_link_lin_vel_w[ids][:, None].expand(-1, w, -1),
    robot.data.root_link_ang_vel_w[ids][:, None].expand(-1, w, -1),
    ee[:, None].expand(-1, w, -1, -1),
    robot.data.joint_pos[ids][:, None].expand(-1, w, -1),
    robot.data.joint_vel[ids][:, None].expand(-1, w, -1))


def natural_mixed_reset(env, env_ids=None, bank_path="", probability=0.1):
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if not hasattr(env, "_natural_bank"):
    b = np.load(bank_path, allow_pickle=False)
    env._natural_bank = torch.as_tensor(b["qpos"], device=env.device)
    env._natural_weights = torch.as_tensor(b["sampling_weights"], device=env.device, dtype=torch.float32)
    names = ("supine", "prone", "left_side_down", "right_side_down")
    env._natural_labels = torch.tensor([names.index(x) for x in b["labels"]], device=env.device)
    env._natural_type = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    env._natural_bad_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    env._natural_draw_counts = torch.zeros(5, device=env.device, dtype=torch.long)
    env._natural_rng = torch.Generator(device=env.device).manual_seed(int(env.cfg.seed) + 901117)
  # Consume exactly the original GSI draw before overriding the selected subset.
  gsi_reset(env, env_ids)
  env._natural_type[env_ids] = -1
  env._natural_bad_count[env_ids] = 0
  choose = torch.rand(len(env_ids), device=env.device, generator=env._natural_rng) < probability
  ids = env_ids[choose]
  env._natural_draw_counts[0] += len(env_ids) - len(ids)
  if not len(ids):
    return
  indices = torch.multinomial(env._natural_weights, len(ids), replacement=True, generator=env._natural_rng)
  pose = env._natural_bank[indices]
  robot = env.scene["robot"]
  assert tuple(robot.joint_names) == JOINT_NAMES
  root = robot.data.default_root_state[ids].clone()
  root[:, :3] = pose[:, :3] + env.scene.env_origins[ids]
  root[:, 3:7] = pose[:, 3:7]
  root[:, 7:] = 0
  robot.write_root_state_to_sim(root, env_ids=ids)
  robot.write_joint_state_to_sim(pose[:, 7:], torch.zeros_like(pose[:, 7:]), env_ids=ids)
  env.sim.forward()
  prime_static_history(env, ids)
  env._natural_type[ids] = env._natural_labels[indices]
  env._natural_draw_counts += torch.bincount(env._natural_type[ids] + 1, minlength=5)


def update_low_count(count, low, eligible):
  """Require uninterrupted bad scores; grace and good scores clear the counter."""
  return torch.where(low & eligible, count + 1, torch.zeros_like(count))


def natural_persistent_low_smp(env, threshold=0.02, ws=6.0, grace_steps=5,
                             natural_grace_seconds=0.5, natural_hold_seconds=0.5):
  ordinary = smp_too_low(env, threshold=threshold, ws=ws, grace_steps=grace_steps)
  if not hasattr(env, "_natural_type"):
    return ordinary
  natural = env._natural_type >= 0
  raw = getattr(env, "_smp_raw_err", None)
  if raw is None:
    env._natural_bad_count.zero_()
    return ordinary & ~natural
  low = torch.exp(-ws * raw) < threshold
  # First eligible check is strictly AFTER the 0.5-second warmup.
  eligible = natural & (env.episode_length_buf > math.ceil(natural_grace_seconds / env.step_dt))
  env._natural_bad_count = update_low_count(env._natural_bad_count, low, eligible)
  persistent = env._natural_bad_count >= math.ceil(natural_hold_seconds / env.step_dt)
  return (ordinary & ~natural) | (persistent & natural)
