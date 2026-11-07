"""Validate projection against the actual RSL-RL inference path."""

import importlib.util
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from smp.rl.tasks.getup.terrain_v36_93d_env_cfg import (
  g1_getup_terrain_v36_93d_smp_env_cfg,
)
from smp.rl.tasks.getup.terrain_v36_env_cfg import g1_getup_terrain_v36_smp_env_cfg

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
  "projection", ROOT / "scripts/project_v36_checkpoint_to_93d.py"
)

module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class ProjectionTests(unittest.TestCase):
  def test_full_actor_equivalence_and_critic_preservation(self):
    torch.manual_seed(20261108)
    path = ROOT / "artifacts/v36_source/model_102000.pt"
    original = torch.load(path, map_location="cpu", weights_only=False)
    projected, _ = module.project_checkpoint(path)
    x = torch.randn(512, 93)
    x96 = torch.cat((torch.zeros(512, 3), x), -1)
    outputs = []
    for values, state in [(x96, original), (x, projected)]:
      obs = TensorDict({"actor": values}, batch_size=[512])
      actor = MLPModel(
        obs,
        {"actor": ["actor"]},
        "actor",
        29,
        (512, 256, 128),
        "elu",
        True,
        {
          "class_name": "GaussianDistribution",
          "init_std": 0.3,
          "std_type": "scalar",
          "learn_std": False,
        },
      ).eval()
      actor.load_state_dict(state["actor_state_dict"], strict=True)
      outputs.append(actor(obs))
    error = (outputs[0] - outputs[1]).abs().max().item()
    print("Full actor maximum absolute error:", error)
    self.assertLess(error, 5e-5)
    for key, value in original["critic_state_dict"].items():
      self.assertTrue(torch.equal(value, projected["critic_state_dict"][key]))
    self.assertEqual(projected["optimizer_state_dict"], {})

  def test_only_actor_linear_velocity_removed(self):
    for play in (False, True):
      original = asdict(g1_getup_terrain_v36_smp_env_cfg(play))
      projected = asdict(g1_getup_terrain_v36_93d_smp_env_cfg(play))
      original["observations"]["actor"]["terms"].pop("base_lin_vel")

      def canonical(v):
        if callable(v):
          return (v.__module__, v.__qualname__)
        if isinstance(v, dict):
          return {k: canonical(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
          return tuple(canonical(x) for x in v)
        return v

      self.assertEqual(canonical(original), canonical(projected))


if __name__ == "__main__":
  unittest.main()
