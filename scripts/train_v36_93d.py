"""V34-style projected warm start on the original V36 terrain task."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

import smp.rl.tasks  # noqa: F401
from smp.rl.warm_start_runner import SmpCurriculumWarmStartRunner


def main():
  p = argparse.ArgumentParser()
  p.add_argument(
    "--checkpoint",
    type=Path,
    default=Path("artifacts/v36_projected/model_102000_zero_velocity_93d.pt"),
  )
  p.add_argument("--log-dir", type=Path, required=True)
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--iterations", type=int, default=12000)
  p.add_argument("--seed", type=int, default=20261108)
  p.add_argument("--smoke", action="store_true")
  p.add_argument("--task", default="Smp-Getup-Terrain-V36-93D-G1")
  a = p.parse_args()
  if a.log_dir.exists():
    raise FileExistsError(a.log_dir)
  configure_torch_backends()
  task = a.task
  cfg, rl = load_env_cfg(task), load_rl_cfg(task)
  cfg.seed = rl.seed = a.seed
  cfg.scene.num_envs = a.num_envs
  rl.max_iterations = a.iterations
  rl.logger = "tensorboard" if a.smoke else "wandb"
  rl.upload_model = not a.smoke
  rl.run_name = a.log_dir.name
  if a.smoke:
    cfg.events["init_smp_state"].params.update(
      compile_model=False, gsi_buffer_size=64, gsi_batch_size=64
    )
  a.log_dir.mkdir(parents=True)
  dump_yaml(a.log_dir / "params/env.yaml", asdict(cfg))
  dump_yaml(a.log_dir / "params/agent.yaml", asdict(rl))
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  wrapped = RslRlVecEnvWrapper(env, clip_actions=rl.clip_actions)
  runner = SmpCurriculumWarmStartRunner(wrapped, asdict(rl), str(a.log_dir), "cuda:0")
  runner.load(str(a.checkpoint))
  obs = wrapped.get_observations()
  assert obs["actor"].shape[-1] == 93 and obs["critic"].shape[-1] == 960
  assert runner.current_learning_iteration == 0 and env.common_step_counter == 0
  source = torch.load(a.checkpoint, map_location="cuda:0", weights_only=False)
  for key, value in source["actor_state_dict"].items():
    torch.testing.assert_close(
      runner.alg.actor.state_dict()[key], value, rtol=0, atol=0
    )
  for key, value in source["critic_state_dict"].items():
    torch.testing.assert_close(
      runner.alg.critic.state_dict()[key], value, rtol=0, atol=0
    )
  assert not runner.alg.optimizer.state
  terrain = env.scene.terrain
  audit = {
    "task": task,
    "source_checkpoint": str(a.checkpoint),
    "source_sha256": hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
    "reward_weights": {name: term.weight for name, term in cfg.rewards.items()},
    "product_task_terms": [
      {"function": fn.__name__, "weight": weight, "params": params}
      for fn, weight, params in cfg.rewards["task_smp_product"].params["task_terms"]
    ],
    "actor_dim": 93,
    "critic_dim": 960,
    "fresh_optimizer": True,
    "learning_rate_schedule": rl.algorithm.schedule,
    "configured_learning_rate": rl.algorithm.learning_rate,
    "optimizer_learning_rates": [g["lr"] for g in runner.alg.optimizer.param_groups],
    "source_weights_exact": True,
    "num_envs": env.num_envs,
    "seed": a.seed,
    "terrain_type_counts": torch.bincount(terrain.terrain_types).cpu().tolist(),
    "reset_type_counts": torch.bincount(env._robust_reset_type, minlength=5)
    .cpu()
    .tolist(),
    "terminations": list(cfg.terminations),
    "physics_dt": cfg.sim.mujoco.timestep,
    "control_dt": cfg.sim.mujoco.timestep * cfg.decimation,
  }
  if hasattr(env, "_stair_region"):
    from smp.rl.tasks.getup.mdp.rewards import _head_height

    head = env.scene["robot"].find_sites(["head"], preserve_order=True)[0][0]
    expected = (
      env.scene["robot"].data.site_pos_w[:, head, 2] - env._recovery_support_height
    )
    actual = _head_height(env, relative_to_env_origin=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    audit["local_height_reference_exact"] = True
    audit["local_vs_platform_reference_max_m"] = float(
      (env._recovery_support_height - terrain.env_origins[:, 2]).abs().max()
    )
    audit["region_counts"] = (
      torch.bincount(env._stair_region, minlength=4).cpu().tolist()
    )
    audit["level_counts"] = (
      torch.bincount(terrain.terrain_levels, minlength=4).cpu().tolist()
    )
  if hasattr(env, "_support_nominal_width"):
    from smp.rl.tasks.getup.stair_support_env_cfg import support_values

    values = support_values(env)
    assert all(torch.isfinite(v).all() for v in values.values())
    assert torch.all(values["ground_loss"] == 0)
    audit["support_nominal_width_m"] = env._support_nominal_width
    audit["support_initial_ground_loss_zero"] = True
  (a.log_dir / "preflight.json").write_text(json.dumps(audit, indent=2))
  print("V36_93D_PREFLIGHT", json.dumps(audit), flush=True)
  runner.add_git_repo_to_log(__file__)
  runner.learn(num_learning_iterations=a.iterations, init_at_random_ep_len=True)
  if rl.algorithm.schedule == "fixed":
    assert all(
      g["lr"] == rl.algorithm.learning_rate for g in runner.alg.optimizer.param_groups
    ), "Fixed learning rate changed during training"
  if not a.smoke:
    runner.export_policy_to_onnx(str(a.log_dir / "exported"), "v36_93d_final.onnx")
    from onnx.reference import ReferenceEvaluator
    from tensordict import TensorDict

    onnx_path = a.log_dir / "exported/v36_93d_final.onnx"
    model = runner.alg.actor.eval().cpu()
    torch.set_float32_matmul_precision("highest")
    rng = torch.Generator().manual_seed(20261109)
    samples = torch.randn(256, 93, generator=rng)
    reference = ReferenceEvaluator(str(onnx_path))
    actual = np.concatenate(
      [
        reference.run(None, {reference.input_names[0]: x[None].numpy()})[0]
        for x in samples
      ]
    )
    with torch.inference_mode():
      expected = model(TensorDict({"actor": samples}, batch_size=[256])).numpy()
    error = float(np.max(np.abs(actual - expected)))
    assert np.isfinite(actual).all() and error < 1e-3, error
    manifest = {
      "status": "TRAINING_COMPLETE_ONNX_PARITY_PASSED_REQUIRES_SIM2SIM_VALIDATION",
      "actor_dim": 93,
      "action_dim": 29,
      "normalizer_embedded": True,
      "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
      "onnx_max_abs_error": error,
      "iterations": a.iterations,
      "source": str(a.checkpoint),
      "seed": a.seed,
    }
    (onnx_path.with_suffix(".manifest.json")).write_text(json.dumps(manifest, indent=2))
  env.close()


if __name__ == "__main__":
  main()
