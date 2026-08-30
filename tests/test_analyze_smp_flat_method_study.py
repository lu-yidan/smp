from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import analyze_smp_flat_method_study as analysis


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


class FlatMethodAnalysisTest(unittest.TestCase):
  def _fixture(self, root: Path, failed_seed: int | None = None) -> analysis.FlatMethodAnalysisCfg:
    protocol = Path(__file__).parents[1] / "docs" / "ral_flat_method_study_v1.json"
    manifest_dir = root / "manifests"
    manifest_dir.mkdir()
    formal = root / "formal"
    index_rows = []
    manifest_ids = []
    for seed in analysis._SEEDS:
      for gate in analysis._GATES:
        runs = []
        for arm in analysis._ARMS:
          checkpoint = root / "checkpoints" / arm / str(seed) / f"model_{gate}.pt"
          checkpoint.parent.mkdir(parents=True, exist_ok=True)
          checkpoint.write_bytes(f"{arm}-{seed}-{gate}".encode())
          runs.append(
            {
              "name": arm,
              "task": f"task-{arm}",
              "checkpoint": str(checkpoint.resolve()),
              "checkpoint_sha256": _sha(checkpoint),
              "policy_seed": seed,
            }
          )
        manifest_id = f"manifest-{seed}-{gate}"
        manifest_ids.append(manifest_id)
        manifest = {
          "schema_version": 1,
          "status": "READY_FOR_FROZEN_EVALUATION",
          "study_id": analysis._STUDY_ID,
          "launch_plan_id": analysis._PLAN_ID,
          "protocol_sha256": analysis._PROTOCOL_SHA256,
          "checkpoint_step": gate,
          "policy_seed": seed,
          "manifest_id": manifest_id,
          "runs": runs,
        }
        stable_manifest = {
          key: value for key, value in manifest.items() if key != "manifest_id"
        }
        manifest["manifest_id"] = hashlib.sha256(
          json.dumps(stable_manifest, sort_keys=True).encode()
        ).hexdigest()
        manifest_id = manifest["manifest_id"]
        manifest_ids[-1] = manifest_id
        manifest_path = manifest_dir / f"gate_{gate}_seed_{seed}.json"
        manifest_path.write_text(json.dumps(manifest))
        index_rows.append(
          {
            "policy_seed": seed,
            "checkpoint_step": gate,
            "path": str(manifest_path.resolve()),
            "sha256": _sha(manifest_path),
          }
        )
        matrix = formal / f"gate_{gate}" / f"seed_{seed}"
        matrix.mkdir(parents=True)
        summary_rows = []
        for run in runs:
          arm = run["name"]
          for mode in analysis._MODES:
            if arm == analysis._PROPOSED:
              successes = 492 if mode == "native_gsi" else 410
              if failed_seed == seed and gate == 29999 and mode == "right_side":
                successes = 0
            else:
              successes = 492 if mode == "native_gsi" else 256
            failure_codes = [0] * successes + [7] * (analysis._NUM_ENVS - successes)
            reason_counts = {code: 0 for code in analysis._FAILURE_CODEBOOK}
            reason_counts["0"] = successes
            reason_counts["7"] = analysis._NUM_ENVS - successes
            per_env = {
              name: (failure_codes if name == "strict_failure_reason_code" else [True] * analysis._NUM_ENVS if name == "finite_action" else [False] * analysis._NUM_ENVS if name == "invalid_dynamics" else [-1] * analysis._NUM_ENVS)
              for name in analysis._PER_ENV_GATE_ARRAYS
            }
            result = {
              "evaluation_schema_version": analysis._SCHEMA,
              "task": run["task"],
              "checkpoint_path": run["checkpoint"],
              "checkpoint_sha256": run["checkpoint_sha256"],
              "policy_seed": seed,
              "seed": analysis._EVAL_SEED,
              "num_envs": analysis._NUM_ENVS,
              "steps": analysis._STEPS,
              "reset_mode": mode,
              "strict_successes": successes,
              "strict_success_rate": successes / analysis._NUM_ENVS,
              "finite_action_rate": 1.0,
              "invalid_dynamics_rate": 0.0,
              "strict_failure_diagnosis": {
                "schema_version": analysis._FAILURE_SCHEMA,
                "does_not_change_strict_success": True,
                "reason_codebook": analysis._FAILURE_CODEBOOK,
                "reason_counts": reason_counts,
              },
              "per_env": per_env,
              "contact_foot_slip_p95_m_s": 0.1,
              "post_success_root_drift_p95_m": 0.1,
              "secondary_fall_rate_after_success": 0.0,
              "foot_separation_at_success_p95_m": 0.4,
              "action_delta_rms_p95": 0.2,
              "action_second_difference_rms_p95": 0.1,
              "max_power_mean_w": 100.0,
              "max_joint_speed_p95_rad_s": 5.0,
            }
            raw = matrix / f"{arm}__model_{gate}__{mode}__eval{analysis._EVAL_SEED}.json"
            raw.write_text(json.dumps(result))
            summary_rows.append({**{k: v for k, v in result.items() if k != "per_env"}, "arm": arm})
        (matrix / "summary.json").write_text(
          json.dumps({"metadata": {"manifest_id": manifest_id}, "evaluations": summary_rows})
        )
        (matrix / "_COMPLETE.json").write_text(
          json.dumps(
            {
              "evaluation_schema_version": analysis._SCHEMA,
              "manifest": str(manifest_path.resolve()),
              "result_count": 10,
              "modes": list(analysis._MODES),
              "eval_seeds": [analysis._EVAL_SEED],
              "num_envs": analysis._NUM_ENVS,
              "steps": analysis._STEPS,
              "devices": ["cuda:0"],
            }
          )
        )
    index = {
      "schema_version": 1,
      "status": "READY_FOR_FROZEN_EVALUATION",
      "study_id": analysis._STUDY_ID,
      "launch_plan_id": analysis._PLAN_ID,
      "protocol_sha256": analysis._PROTOCOL_SHA256,
      "arms": list(analysis._ARMS),
      "policy_seeds": list(analysis._SEEDS),
      "checkpoint_steps": list(analysis._GATES),
      "checkpoint_entry_count": 24,
      "manifest_ids": sorted(manifest_ids),
      "index_id": "test-index",
      "manifests": index_rows,
    }
    stable_index = {
      key: value for key, value in index.items() if key not in {"index_id", "manifests"}
    }
    index["index_id"] = hashlib.sha256(
      json.dumps(stable_index, sort_keys=True).encode()
    ).hexdigest()
    index_path = manifest_dir / "index.json"
    index_path.write_text(json.dumps(index))
    return analysis.FlatMethodAnalysisCfg(
      manifest_index=index_path,
      evaluation_root=formal,
      protocol=protocol,
      output_json=root / "analysis.json",
      output_markdown=root / "analysis.md",
    )

  def test_complete_supported_study_promotes_only_a8(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      cfg = self._fixture(Path(temporary))
      result = analysis.write_analysis(cfg)
      self.assertEqual(result["status"], "PROMOTE_TP_SPECIALISTS")
      self.assertTrue(result["promotion"]["pass"])
      self.assertFalse(result["promotion"]["control_arm_promotion_eligible"])
      self.assertEqual(result["evaluation_contract"]["raw_result_count"], 120)
      self.assertEqual(result["evaluation_contract"]["per_environment_rollout_count"], 61440)
      self.assertGreater(
        result["paired_causal_analysis"]["effects"]["fixed_worst"]["ci95_low"],
        0.0,
      )
      self.assertEqual(
        set(result["paired_causal_analysis"]["safety_pareto_effects"]),
        set(analysis._SAFETY_MAX),
      )

  def test_complete_negative_result_is_retained_without_promotion(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      cfg = self._fixture(Path(temporary), failed_seed=20261003)
      result = analysis.write_analysis(cfg)
      self.assertEqual(result["status"], "FLAT_METHOD_COMPLETE_NO_PROMOTION")
      self.assertFalse(result["promotion"]["pass"])
      self.assertTrue(cfg.output_json.is_file())

  def test_failure_telemetry_mismatch_fails_closed(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = self._fixture(root)
      raw = next(cfg.evaluation_root.glob("**/a8_balanced_bridge__*.json"))
      payload = json.loads(raw.read_text())
      payload["strict_failure_diagnosis"]["reason_counts"]["7"] -= 1
      raw.write_text(json.dumps(payload))
      with self.assertRaisesRegex(ValueError, "do not sum"):
        analysis.analyze(cfg)

  def test_partial_matrix_and_immutable_output_fail_closed(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = self._fixture(root)
      next(cfg.evaluation_root.glob("**/a6_replication_control__*.json")).unlink()
      with self.assertRaisesRegex(ValueError, "ten raw results"):
        analysis.analyze(cfg)
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = self._fixture(root)
      analysis.write_analysis(cfg)
      cfg.output_json.write_text("{}\n")
      with self.assertRaisesRegex(ValueError, "analysis conflicts"):
        analysis.write_analysis(cfg)


if __name__ == "__main__":
  unittest.main()
