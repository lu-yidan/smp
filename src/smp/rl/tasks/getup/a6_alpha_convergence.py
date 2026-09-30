"""Alpha sweep and a closure x anchor factorial on otherwise frozen A6."""
import torch
from mjlab.managers.event_manager import requires_model_fields, RecomputeLevel
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from smp.rl.tasks.getup import multiterrain as mt, v33_path_ablation as pa
from smp.rl.tasks.getup.a6_paper_ablation import init_eval_loads, sample_eval_loads

ARMS = ('AC0_a005', 'AC1_a020', 'AC2_a050', 'AC3_a100',
        'AC4_close', 'AC5_anchor', 'AC6_close_anchor')


def settings(arm):
  assert arm in ARMS, arm
  return dict(blocked_alpha=(.05, .20, .50, 1., 1., 1., 1.)[ARMS.index(arm)],
              close=arm in ('AC4_close', 'AC6_close_anchor'),
              anchor=arm in ('AC5_anchor', 'AC6_close_anchor'))


def scaled_task(env, task_terms, blocked_alpha=.05, **kwargs):
  result = mt.recorded_task_smp_product(env, task_terms, **kwargs)
  blocked = (env._mt_scene > 0) & ~env._mt_escaped
  env._plate_product = result * torch.where(blocked, blocked_alpha, 1.)
  return env._plate_product


def closed_separation(env, index=4):
  return mt.escape_reward(env, index) * ~env._mt_escaped


def saturated_clearance(env, index=1):
  value = pa.reward(env, index)
  clear = (env._mt_scene > 0) & env._mt_escaped & ~env._mt_invalid
  return torch.where(clear, torch.ones_like(value), value)


def position_cost(xy, anchor, gate, valid, free_radius=.25, transition=.25):
  """Bounded cost outside a free disk, never a hard safety boundary."""
  excess = ((xy-anchor).norm(dim=-1)-free_radius).clamp_min(0)
  return (excess/transition).clamp_max(1).square() * gate * valid


@requires_model_fields('body_mass', 'body_inertia', 'geom_size', recompute=RecomputeLevel.set_const)
def reset(env, env_ids=None, **kwargs):
  # Common to all seven arms: no random draws or physical writes beyond A6 reset.
  pa.reset(env, env_ids, **kwargs)
  if not hasattr(env, '_ac_anchor'):
    env._ac_anchor = torch.zeros(env.num_envs, 2, device=env.device)
    env._ac_anchor_set = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._ac_ready_time = torch.zeros(env.num_envs, device=env.device)
    env._ac_cost = torch.zeros(env.num_envs, device=env.device)
    env._ac_tick = -1
  ids = slice(None) if env_ids is None else env_ids
  env._ac_anchor[ids] = 0
  env._ac_anchor_set[ids] = False
  env._ac_ready_time[ids] = 0
  env._ac_cost[ids] = 0


def anchor_cost(env):
  if env._ac_tick == env.common_step_counter:
    return env._ac_cost
  env._ac_tick = env.common_step_counter
  r = env.scene['robot']
  z = mt.height(env)
  u = -r.data.projected_gravity_b[:, 2]
  flat = env._mt_scene == 0
  ready = flat & (z >= .85) & (u >= .70)
  ready &= (mt.ground_force(env, 'quality_feet')[..., 2].abs() > 20).all(-1)
  env._ac_ready_time = torch.where(ready, env._ac_ready_time+env.step_dt, 0.)
  escaped = env._mt_escaped & ~env._mt_invalid
  latch = ~env._ac_anchor_set & torch.where(flat, env._ac_ready_time >= .3-1e-6, escaped)
  xy = r.data.root_link_pos_w[:, :2]
  env._ac_anchor[latch] = xy[latch]
  env._ac_anchor_set |= latch
  # Pause when reblocked, but never move the stored anchor within an episode.
  valid = env._ac_anchor_set & (flat | escaped)
  gate = ((z-.85)/.30).clamp(0, 1) * ((u-.70)/.23).clamp(0, 1)
  env._ac_cost = position_cost(xy, env._ac_anchor, gate, valid)
  return env._ac_cost


def apply_intervention(cfg, arm):
  opts = settings(arm)
  cfg.rewards['task_smp_product'].func = scaled_task
  cfg.rewards['task_smp_product'].params['blocked_alpha'] = opts['blocked_alpha']
  cfg.events['gsi_reset'].func = reset
  # Compute identical bookkeeping even when the reward is disabled.
  cfg.metrics['ac_anchor_cost'] = MetricsTermCfg(func=anchor_cost)
  if opts['close']:
    cfg.rewards['plate_separation'].func = closed_separation
    cfg.rewards['plate_clearance'].func = saturated_clearance
  if opts['anchor']:
    cfg.rewards['post_clear_anchor'] = RewardTermCfg(func=anchor_cost, weight=-.05)
