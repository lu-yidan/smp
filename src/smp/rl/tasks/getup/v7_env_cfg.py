"""V7 route-prior ablation on top of the full V6 recovery task."""

from smp.rl.tasks.getup.v6_env_cfg import g1_getup_v6_smp_env_cfg

V7_ROUTE_PRIOR_PATH = (
  "datasets/pretrain_ckpt/pretrained_getup_lafan_route_v7.pt"
)


def g1_getup_v7_route_smp_env_cfg(play: bool = False):
  """Full V6 behavior with only the recovery prior checkpoint replaced."""
  cfg = g1_getup_v6_smp_env_cfg(play=play)
  cfg.events["init_smp_state"].params["ckpt_path"] = V7_ROUTE_PRIOR_PATH
  return cfg
