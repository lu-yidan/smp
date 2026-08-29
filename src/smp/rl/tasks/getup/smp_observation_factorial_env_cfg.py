"""Observation-only factorial variants of the original SMP get-up task.

These variants intentionally keep the original F2S2 prior, GSI resets,
disturbances, reward, terminations, and PPO settings unchanged. Only the
deployable actor observation is varied.
"""

from smp.rl.tasks.getup.getup_env_cfg import g1_getup_smp_env_cfg


def _observation_factorial_cfg(
    *, play: bool, include_base_lin_vel: bool, history_length: int
):
    cfg = g1_getup_smp_env_cfg(play=play)
    actor = cfg.observations["actor"]
    if not include_base_lin_vel:
        actor.terms.pop("base_lin_vel", None)
    actor.history_length = history_length if history_length > 1 else None
    return cfg


def g1_getup_obs_f1_nolinvel_smp_env_cfg(play: bool = False):
    """One actor frame without privileged base linear velocity."""
    return _observation_factorial_cfg(
        play=play, include_base_lin_vel=False, history_length=1
    )


def g1_getup_obs_f4_vel_smp_env_cfg(play: bool = False):
    """Four actor frames with true base linear velocity."""
    return _observation_factorial_cfg(
        play=play, include_base_lin_vel=True, history_length=4
    )


def g1_getup_obs_f4_nolinvel_smp_env_cfg(play: bool = False):
    """Four deployable actor frames without base linear velocity."""
    return _observation_factorial_cfg(
        play=play, include_base_lin_vel=False, history_length=4
    )


__all__ = [
    "g1_getup_obs_f1_nolinvel_smp_env_cfg",
    "g1_getup_obs_f4_nolinvel_smp_env_cfg",
    "g1_getup_obs_f4_vel_smp_env_cfg",
]
