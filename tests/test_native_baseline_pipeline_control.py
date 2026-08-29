from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import advance_smp_ral_pipeline as pipeline


def _cfg(root: Path, **overrides) -> pipeline.PipelineCfg:
  values = {
    "control_dir": root / "flat_control",
    "evidence_dir": root / "flat_eval",
    "state": root / "state.json",
    "confirmation_control_dir": root / "confirmation",
    "confirmation_evidence_dir": root / "confirmation_eval",
    "specialist_control_dir": root / "specialists",
    "specialist_evidence_dir": root / "specialist_eval",
    "baseline_control_dir": root / "baselines",
    "baseline_native_control_dir": root / "baselines/native_training",
    "baseline_native_manifest_dir": root / "baselines/native_manifests",
    "baseline_eval_bank_output_dir": root / "baselines/held_out",
    "baseline_eval_bank_manifest": root / "baselines/held_out.json",
    "baseline_formal_manifest_dir": root / "baselines/formal",
    "baseline_evidence_dir": root / "baselines/eval",
  }
  values.update(overrides)
  return pipeline.PipelineCfg(**values)


class NativeBaselinePipelineTest(unittest.TestCase):
  def test_held_out_bank_is_frozen_before_native_training(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = _cfg(root, launch_native_eval_bank_when_ready=True)
      launcher = mock.Mock(return_value={"pid": 123, "plan_id": "held-out"})
      native_launcher = mock.Mock()
      with (
        mock.patch.object(pipeline, "_active_eval_bank", return_value=None),
        mock.patch.object(pipeline, "_validate_eval_bank_artifacts", return_value=None),
        mock.patch.object(pipeline, "_launch_eval_bank", launcher),
        mock.patch.object(pipeline, "launch_baselines", native_launcher),
      ):
        state = pipeline._advance_native_baselines(
          cfg, [], root / "flat_promotion.json"
        )
      self.assertEqual(state["status"], "NATIVE_HELD_OUT_RUNNING")
      launcher.assert_called_once()
      native_launcher.assert_not_called()

  def test_gpu_activity_blocks_held_out_generation(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = _cfg(root, launch_native_eval_bank_when_ready=True)
      with (
        mock.patch.object(pipeline, "_active_eval_bank", return_value=None),
        mock.patch.object(pipeline, "_validate_eval_bank_artifacts", return_value=None),
        mock.patch.object(pipeline, "_launch_eval_bank") as launcher,
      ):
        state = pipeline._advance_native_baselines(
          cfg, ["gpu-process"], root / "flat_promotion.json"
        )
      self.assertEqual(state["status"], "WAITING_FREE_GPU")
      launcher.assert_not_called()

  def test_ready_held_out_bank_exposes_native_launch_gate(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = _cfg(root, launch_native_baselines_when_ready=False)
      held_out = {"status": "READY", "plan_id": "held-out"}
      planned = {"status": "PLANNED", "plan_id": "native-plan"}
      with (
        mock.patch.object(pipeline, "_active_eval_bank", return_value=None),
        mock.patch.object(
          pipeline, "_validate_eval_bank_artifacts", return_value=held_out
        ),
        mock.patch.object(pipeline, "launch_baselines", return_value=planned),
      ):
        state = pipeline._advance_native_baselines(
          cfg, [], root / "flat_promotion.json"
        )
      self.assertEqual(state["status"], "NATIVE_BASELINE_READY")
      self.assertEqual(state["held_out_bank"], held_out)

  def test_terminal_specialists_hand_control_to_native_pipeline(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = _cfg(root, launch_confirmation_when_ready=True)
      health = {
        "healthy": True,
        "jobs": [
          {"log": f"gpu{i}.log", "iteration": 29999, "completed": True}
          for i in range(8)
        ],
      }
      manifests = [
        {
          "gate": gate,
          "path": str(root / f"gate_{gate}.json"),
          "sha256": str(gate),
        }
        for gate in cfg.gates
      ]
      native = {
        "status": "NATIVE_HELD_OUT_READY",
        "action": "freeze held-out states",
      }
      confirmation = {
        "status": "LAUNCHED",
        "plan_id": "confirmation",
        "jobs": [{"pid": 1}],
      }
      confirmation_progress = {
        "status": "CONFIRMATION_ANALYSIS_COMPLETE",
        "action": "done",
        "aggregate": {"status": "MINIMUM_POLICY_SEEDS_MET"},
      }
      with (
        mock.patch.object(pipeline, "inspect", return_value=health),
        mock.patch.object(pipeline, "_ensure_manifests", return_value=(manifests, [])),
        mock.patch.object(pipeline, "_analysis_complete", return_value=True),
        mock.patch.object(pipeline, "_active_eval", return_value=None),
        mock.patch.object(pipeline, "_gpu_processes", return_value=[]),
        mock.patch.object(
          pipeline,
          "write_selection",
          return_value={
            "status": "PROMOTE_FOR_POLICY_SEEDS",
            "promoted_candidates": ["arm"],
          },
        ),
        mock.patch.object(pipeline, "launch_confirmation", return_value=confirmation),
        mock.patch.object(
          pipeline, "_advance_confirmation", return_value=confirmation_progress
        ),
        mock.patch.object(
          pipeline,
          "_advance_specialists",
          return_value={
            "status": "TP_SPECIALIST_NO_PROMOTION",
            "action": "T/P terminal",
          },
        ),
        mock.patch.object(
          pipeline, "_advance_native_baselines", return_value=native
        ) as advance_native,
      ):
        state = pipeline.advance(cfg)
      self.assertEqual(state["status"], "NATIVE_HELD_OUT_READY")
      self.assertEqual(state["native_baselines"], native)
      advance_native.assert_called_once()

  def test_complete_native_matrices_emit_paired_effect_evidence(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      cfg = _cfg(root)
      held_out = {"status": "READY", "plan_id": "held-out"}
      launch = {
        "status": "LAUNCHED",
        "max_updates": 30000,
        "workers": [],
        "jobs": [{"log": str(root / f"job{i}.log")} for i in range(9)],
      }
      checkpoint_index = {"status": "CHECKPOINTS_READY_EVALUATION_BANK_BLOCKED"}
      rows = []
      for gate in (8000, 15000, 25000, 29999):
        for seed in (20260901, 20260902, 20260903):
          rows.append(
            {
              "checkpoint_step": gate,
              "policy_seed": seed,
              "path": str(root / f"formal_{gate}_{seed}.json"),
            }
          )
      formal_index = {
        "status": "READY",
        "policy_seeds": [20260901, 20260902, 20260903],
        "manifests": rows,
      }

      def aggregate(cfg_arg):
        cfg_arg.output_json.parent.mkdir(parents=True, exist_ok=True)
        cfg_arg.output_json.write_text("{}")
        return {"status": "MINIMUM_POLICY_SEEDS_MET"}

      def paired(cfg_arg):
        cfg_arg.output_json.parent.mkdir(parents=True, exist_ok=True)
        cfg_arg.output_json.write_text("{}")
        return {
          "status": "PROPOSED_PAIRED_ADVANTAGE_SUPPORTED",
          "support_rule_checks": {"primary": True},
        }

      with (
        mock.patch.object(pipeline, "_active_eval_bank", return_value=None),
        mock.patch.object(
          pipeline, "_validate_eval_bank_artifacts", return_value=held_out
        ),
        mock.patch.object(pipeline, "launch_baselines", return_value=launch),
        mock.patch.object(pipeline, "_job_log_completed", return_value=True),
        mock.patch.object(
          pipeline,
          "write_native_baseline_manifests",
          return_value=checkpoint_index,
        ),
        mock.patch.object(pipeline, "write_bindings", return_value=formal_index),
        mock.patch.object(pipeline, "_active_native_evaluation", return_value=None),
        mock.patch.object(pipeline, "_analysis_complete", return_value=True),
        mock.patch.object(pipeline, "write_aggregate", side_effect=aggregate),
        mock.patch.object(pipeline, "write_native_effect_analysis", side_effect=paired),
      ):
        state = pipeline._advance_native_baselines(
          cfg, [], root / "flat_promotion.json"
        )
      self.assertEqual(state["status"], "NATIVE_BASELINE_EVIDENCE_COMPLETE")
      self.assertEqual(
        state["paired_effects"]["status"],
        "PROPOSED_PAIRED_ADVANTAGE_SUPPORTED",
      )
      self.assertEqual(len(state["aggregates"]), 4)


if __name__ == "__main__":
  unittest.main()
