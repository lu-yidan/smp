"""Blocked-task alpha x post-clear settling, with frozen A6 dynamics/reset."""
import torch
from mjlab.managers.reward_manager import RewardTermCfg
from smp.rl.tasks.getup import a6_alpha_convergence as ac
from smp.rl.tasks.getup.a6_paper_ablation import init_eval_loads, sample_eval_loads

ARMS = ('CS_a020_base', 'CS_a020_settle', 'CS_a050_base', 'CS_a050_settle',
        'CS_a100_base', 'CS_a100_settle')

def settings(arm):
    assert arm in ARMS, arm
    alpha = {'a020': .2, 'a050': .5, 'a100': 1.}[arm.split('_')[1]]
    settle = arm.endswith('_settle')
    return dict(blocked_alpha=alpha, close=settle, anchor=settle)

def settle_cost(distance, head_z, upright, valid):
    """35cm free region; smooth standing gate; cost remains graded to 1.35m."""
    def smooth(x, lo, hi):
        t = ((x-lo)/(hi-lo)).clamp(0, 1)
        return t.square()*(3-2*t)
    gate = smooth(head_z, .95, 1.15)*smooth(upright, .80, .93)
    excess = ((distance-.35).clamp_min(0)/.50).clamp_max(2)
    return excess.square()*gate*valid

def post_clear_settle(env):
    # Reuse A6's shared anchor bookkeeping; no extra RNG, reset, or observations.
    ac.anchor_cost(env)
    robot = env.scene['robot']
    xy = robot.data.root_link_pos_w[:, :2]
    valid = (env._mt_scene > 0) & env._mt_escaped & ~env._mt_invalid & env._ac_anchor_set
    return settle_cost((xy-env._ac_anchor).norm(dim=-1), ac.mt.height(env),
                       -robot.data.projected_gravity_b[:, 2], valid)

def apply_intervention(cfg, arm):
    opts = settings(arm)
    base = {.2:'AC1_a020', .5:'AC2_a050', 1.:'AC3_a100'}[opts['blocked_alpha']]
    ac.apply_intervention(cfg, base)
    if opts['close']:
        cfg.rewards['plate_separation'].func = ac.closed_separation
        cfg.rewards['plate_clearance'].func = ac.saturated_clearance
        cfg.rewards['post_clear_anchor'] = RewardTermCfg(func=post_clear_settle, weight=-.05)
