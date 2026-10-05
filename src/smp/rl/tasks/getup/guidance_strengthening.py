"""Paired stronger baselines: common scene mix and bounded escape auxiliaries.

These auxiliaries also use privileged plate geometry, but never actor inputs.
`without geometry` in this experiment means without the three core A6 terms.
"""
import numpy as np
import torch
from mjlab.managers.event_manager import requires_model_fields, RecomputeLevel
from mjlab.utils.lab_api.math import quat_apply
from smp.rl.tasks.getup import multiterrain as mt, a6_alpha_convergence as ac
from smp.rl.tasks.getup import v33_path_ablation as pa


def scene_layout(n):
    assert n >= 32 and n % 32 == 0
    units = np.asarray([.30, .50, .20]) * (n // 4)
    counts = np.floor(units).astype(int)
    order = np.argsort(-(units-counts), kind='stable')
    counts[order[:n//4-int(counts.sum())]] += 1
    counts *= 4
    scene = np.repeat(np.arange(3), counts)
    direction = np.concatenate([np.repeat(np.arange(4), c//4) for c in counts])
    return scene, direction


def auxiliary_step(timer, best_coverage, best_clearance, best_distance,
                   best_lateral, coverage, clearance, distance, lateral,
                   eligible, prone_guided, dt):
    """No reward for oscillating within a previously visited lateral extent."""
    meaningful = ((best_coverage-coverage >= .001)
                  | (clearance-best_clearance >= .001)
                  | (distance-best_distance >= .001))
    timer = torch.where(eligible & ~meaningful, timer+dt, 0.)
    stall = ((timer-.5)/1.).clamp(0, 1) * eligible
    progress = ((lateral-best_lateral)/.025).clamp(0, 1)
    progress = progress * eligible * prone_guided
    return (timer, torch.minimum(best_coverage, coverage),
            torch.maximum(best_clearance, clearance),
            torch.maximum(best_distance, distance),
            torch.maximum(best_lateral, lateral), stall, progress)


@requires_model_fields('body_mass', 'body_inertia', 'geom_size',
                       recompute=RecomputeLevel.set_const)
def reset(env, env_ids=None, bank_path='outputs/multiterrain_bank/train.npz',
          evaluation=False, **kwargs):
    # Initialize before the existing reset; no extra random draw is consumed.
    if not hasattr(env, '_mt_bank'):
        mt._init(env, bank_path)
        if not evaluation:
            scene, direction = scene_layout(env.num_envs)
            env._mt_scene.copy_(torch.as_tensor(scene, device=env.device))
            env._mt_stratum.copy_(env._mt_scene)
            env._mt_direction.copy_(torch.as_tensor(direction, device=env.device))
            env._mt_source.zero_()
            for s in range(3):
                for d in range(4):
                    ids = torch.where((env._mt_scene == s) & (env._mt_direction == d))[0]
                    env._mt_source[ids[:round(len(ids)*.25)]] = 1
            env._fixed_group = env._mt_direction+2
    ac.reset(env, env_ids, bank_path=bank_path, evaluation=evaluation, **kwargs)
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    if not hasattr(env, '_sg_timer'):
        for name in ('timer', 'coverage', 'clearance', 'distance', 'lateral', 'stall', 'progress'):
            setattr(env, '_sg_'+name, torch.zeros(env.num_envs, device=env.device))
        env._sg_origin = torch.zeros(env.num_envs, 2, device=env.device)
        env._sg_axis = env._sg_origin.clone()
        env._sg_tick = -1
    _, coverage, clearance = pa.geometry(env)
    env._sg_coverage[ids] = coverage[ids]
    env._sg_clearance[ids] = clearance[ids]
    for name in ('timer', 'distance', 'lateral', 'stall', 'progress'):
        getattr(env, '_sg_'+name)[ids] = 0
    robot = env.scene['robot']
    env._sg_origin[ids] = robot.data.root_link_pos_w[ids, :2]
    axis = env._sg_axis.new_tensor([0., 1., 0.]).expand(len(ids), -1)
    axis = quat_apply(robot.data.root_link_quat_w[ids], axis)[:, :2]
    env._sg_axis[ids] = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    # Do not invalidate the whole-world tick on a partial reset: untouched
    # worlds must never accumulate the same control interval twice.


def update(env):
    if not hasattr(env, '_sg_timer') or env._sg_tick == env.common_step_counter:
        return
    env._sg_tick = env.common_step_counter
    mt.update(env)
    _, coverage, clearance = pa.geometry(env)
    robot = env.scene['robot']
    xy = robot.data.root_link_pos_w[:, :2]
    rows = torch.arange(env.num_envs, device=env.device)
    gids = torch.stack(env._mt_plate_ids)[(env._mt_scene == 2).long()]
    distance = (xy-env.sim.data.geom_xpos[rows, gids, :2]).norm(dim=-1)
    lateral = ((xy-env._sg_origin)*env._sg_axis).sum(-1).abs()
    eligible = (env._mt_scene > 0) & env._mt_ever_contact & ~env._mt_escaped & ~env._mt_invalid
    prone = (env._mt_scene == 1) & (env._mt_direction == 1)
    values = auxiliary_step(env._sg_timer, env._sg_coverage, env._sg_clearance,
                            env._sg_distance, env._sg_lateral, coverage,
                            clearance, distance, lateral, eligible, prone, env.step_dt)
    for name, value in zip(('timer', 'coverage', 'clearance', 'distance', 'lateral', 'stall', 'progress'), values):
        getattr(env, '_sg_'+name).copy_(value)


def reward(env, index):
    update(env)
    if not hasattr(env, '_sg_timer'):
        return torch.zeros(env.num_envs, device=env.device)
    return env._sg_stall if index == 0 else env._sg_progress
