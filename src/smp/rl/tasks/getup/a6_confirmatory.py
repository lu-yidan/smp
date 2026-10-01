"""Preregistered alpha=1 confirmatory ablations; frozen A6 physics and reset."""
from smp.rl.tasks.getup import a6_alpha_convergence as ac
from smp.rl.tasks.getup.a6_paper_ablation import GEOMETRY_TERMS, init_eval_loads, sample_eval_loads

ARMS = ('C20_full', 'C20_no_geometry', 'C20_no_Q', 'C20_no_L')

def settings(arm):
    assert arm in ARMS
    return dict(blocked_alpha=1., close=False, anchor=False,
                geometry=arm != 'C20_no_geometry', quiet_feet=arm != 'C20_no_Q',
                joint_stall=arm != 'C20_no_L')

def apply_intervention(cfg, arm):
    opts = settings(arm)
    ac.apply_intervention(cfg, 'AC3_a100')
    if not opts['geometry']:
        for name in GEOMETRY_TERMS:
            cfg.rewards[name].weight = 0.
    if not opts['quiet_feet']:
        cfg.rewards['path_quiet_feet'].weight = 0.
    if not opts['joint_stall']:
        cfg.rewards['path_joint_stall'].weight = 0.
