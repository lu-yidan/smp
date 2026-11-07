"""V36 observation adaptation: remove actor linear velocity only."""

from smp.rl.tasks.getup.terrain_v36_env_cfg import g1_getup_terrain_v36_smp_env_cfg


def g1_getup_terrain_v36_93d_smp_env_cfg(play: bool = False):
  cfg = g1_getup_terrain_v36_smp_env_cfg(play=play)
  if list(cfg.observations["actor"].terms)[0] != "base_lin_vel":
    raise RuntimeError("Projection requires base_lin_vel in the first three entries")
  cfg.observations["actor"].terms.pop("base_lin_vel")
  return cfg
