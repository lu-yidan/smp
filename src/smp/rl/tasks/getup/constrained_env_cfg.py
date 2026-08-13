"""V7 recovery with sustained, body-localized external constraints."""

from __future__ import annotations

import os

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg

from smp.rl.tasks.getup import mdp
from smp.rl.tasks.getup.v7_env_cfg import g1_getup_v7_route_smp_env_cfg


def g1_getup_constrained_smp_env_cfg(play: bool = False):
  """Build the first constraint-recovery benchmark on the frozen V7 task.

  V1 deliberately changes only the disturbance distribution.  The actor keeps
  the V7 deployable observation contract and never receives the sampled body,
  force, duration, or cohort.  This makes it a clean curriculum baseline for
  later constraint-belief and active-probing ablations.
  """
  cfg = g1_getup_v7_route_smp_env_cfg(play=play)
  cfg.episode_length_s = 20.0
  cfg.sim.nconmax = 96

  # Isolate persistent low-pose constraints from V6's post-stand knockdowns.
  # A later combined benchmark can enable both after each component is measured.
  cfg.events.pop("stratified_post_stand_wrench", None)
  enabled = not play or os.environ.get("SMP_PLAY_AUTO_DISTURBANCES") == "1"
  if enabled:
    constraint_params = {
      "body_names": (
        "pelvis",
        "torso_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_knee_link",
        "right_knee_link",
      ),
      "cohort_weights": (0.25, 0.50, 0.25),
      "delay_steps": (0, 15),
      "duration_steps": (100, 350),
      "force_range": (20.0, 120.0),
      "torque_range": (0.0, 8.0),
      "lateral_force_fraction": 0.20,
      "curriculum_steps": 400_000,
    }
    cfg.events["reset_sustained_constraint"] = EventTermCfg(
      func=mdp.reset_sustained_constraint,
      mode="reset",
      params=constraint_params,
    )
    cfg.events["apply_sustained_constraint"] = EventTermCfg(
      func=mdp.apply_sustained_constraint,
      mode="step",
    )

  cfg.metrics.update(
    {
      "constraint_active": MetricsTermCfg(func=mdp.constraint_active_metric),
      "constraint_cohort": MetricsTermCfg(func=mdp.constraint_cohort_metric),
      "constraint_load_n": MetricsTermCfg(func=mdp.constraint_load_metric),
      "constraint_release_progress": MetricsTermCfg(
        func=mdp.constraint_release_progress_metric
      ),
    }
  )
  return cfg
