from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

from embodied_memory_pilot.ai2thor_paired_task_bridge_control import (
    CLAIM_BOUNDARY,
    LIVE_PASSIVE_CLAIM_BOUNDARY,
    FORBIDDEN_POSITIVE_CLAIM_TERMS,
    REQUIRED_PAIR_FIELDS,
    REQUIRED_SUMMARY_FIELDS,
    SCHEMA_VERSION,
    _build_active_pair_row,
    _build_live_passive_pair_row,
    _build_passive_pair_row,
    _check_schema,
    _csv_fieldnames,
    _render_readme,
    _stable_case_id,
    _term_in_positive_context,
    build_live_passive_paired_control,
    build_paired_control,
    build_pairs,
    summarize_pairs,
    write_outputs,
)
from tests.test_ai2thor_live_gsam_closed_loop import (
    FakePassiveNavigationFailureController,
    FakePassiveTaskBridgeController,
)


def _make_mock_active_row(
    row_idx: int = 0,
    scene: str = "FloorPlan1",
    seed: int = 29,
    target: str = "Apple",
    verifier_used: bool = True,
    memory_mutated: bool = True,
    task_execution_evaluated: bool = True,
    downstream_task_success: bool = True,
    oracle_stale_label: bool = True,
) -> dict[str, object]:
    return {
        "row_idx": row_idx,
        "scene": scene,
        "seed": seed,
        "target": target,
        "target_object_id": f"{target}|-00.47|+01.15|+00.48",
        "target_pickupable": True,
        "target_openable": False,
        "remembered_location": f"{scene}:{target}|-00.47|+01.15|+00.48@before",
        "verifier_used": verifier_used,
        "verifier_name": "grounded_sam2",
        "decision_stale": True,
        "decision_confidence": 1.0,
        "decision_reason": "nearest_after_detection",
        "decision_matches_oracle": True,
        "oracle_stale_label": oracle_stale_label,
        "memory_mutated": memory_mutated,
        "memory_old_position": {"x": -0.465, "y": 1.151, "z": 0.476},
        "memory_new_position": {"x": -0.059, "y": 1.158, "z": -0.254},
        "memory_update_action": "propose_detector_evidence_refresh",
        "task_execution_evaluated": task_execution_evaluated,
        "downstream_task_success": downstream_task_success,
        "execute_action": "PickupObject",
        "execution_boundary": "stepwise_task_bridge_after_verified_memory_update",
        "task_bridge_navigation_mode": "stepwise_navigation",
        "used_teleportfull_for_revisit": False,
        "task_bridge_used_teleportfull": False,
        "revisit_path_steps": 2,
        "task_bridge_path_steps": 5,
        "_source_file": "test/artifact.json",
        "_source_row_offset": row_idx,
    }


class StableCaseIdTest(unittest.TestCase):
    def test_same_inputs_produce_same_id(self) -> None:
        id1 = _stable_case_id("FloorPlan1", 29, "Apple")
        id2 = _stable_case_id("FloorPlan1", 29, "Apple")
        self.assertEqual(id1, id2)
        self.assertEqual(len(id1), 8)

    def test_different_inputs_produce_different_ids(self) -> None:
        ids = {
            _stable_case_id("FloorPlan1", 29, "Apple"),
            _stable_case_id("FloorPlan1", 11, "Apple"),
            _stable_case_id("FloorPlan201", 29, "Apple"),
            _stable_case_id("FloorPlan1", 29, "Pencil"),
        }
        self.assertEqual(len(ids), 4)

    def test_id_is_hex_digest(self) -> None:
        cid = _stable_case_id("FloorPlan1", 29, "Apple")
        self.assertTrue(all(c in "0123456789abcdef" for c in cid))


class BuildActivePairRowTest(unittest.TestCase):
    def test_active_row_copies_fields(self) -> None:
        mock = _make_mock_active_row()
        result = _build_active_pair_row(mock)

        self.assertEqual(result["case_id"], _stable_case_id("FloorPlan1", 29, "Apple"))
        self.assertEqual(result["arm"], "active")
        self.assertEqual(result["scene"], "FloorPlan1")
        self.assertEqual(result["seed"], 29)
        self.assertEqual(result["target"], "Apple")
        self.assertTrue(result["verifier_used"])
        self.assertTrue(result["memory_mutated"])
        self.assertTrue(result["task_execution_evaluated"])
        self.assertTrue(result["downstream_task_success"])
        self.assertEqual(result["row_type"], "recorded_active")
        self.assertEqual(result["derived_from"], "recorded_active_closed_loop_row")

    def test_active_row_has_all_required_fields(self) -> None:
        mock = _make_mock_active_row()
        result = _build_active_pair_row(mock)
        for field in REQUIRED_PAIR_FIELDS:
            self.assertIn(field, result, f"active row missing required field: {field}")


class BuildPassivePairRowTest(unittest.TestCase):
    def test_passive_row_sets_verifier_used_false(self) -> None:
        mock = _make_mock_active_row()
        result = _build_passive_pair_row(mock)

        self.assertFalse(result["verifier_used"])
        self.assertEqual(result["verifier_name"], "none_passive_arm")
        self.assertIsNone(result["decision_stale"])
        self.assertIsNone(result["decision_confidence"])
        self.assertEqual(result["decision_reason"], "passive_arm_no_verifier_executed")
        self.assertIsNone(result["decision_matches_oracle"])

    def test_passive_row_sets_memory_mutated_false(self) -> None:
        mock = _make_mock_active_row()
        result = _build_passive_pair_row(mock)

        self.assertFalse(result["memory_mutated"])
        self.assertEqual(result["memory_old_position"], mock["memory_old_position"])
        self.assertEqual(result["memory_new_position"], mock["memory_old_position"])
        self.assertEqual(result["memory_update_action"], "none_passive_arm")

    def test_passive_row_preserves_oracle_label(self) -> None:
        mock = _make_mock_active_row(oracle_stale_label=True)
        result = _build_passive_pair_row(mock)
        self.assertTrue(result["oracle_stale_label"])

    def test_passive_row_marks_task_as_prepared_not_evaluated(self) -> None:
        mock = _make_mock_active_row(task_execution_evaluated=True, downstream_task_success=True)
        result = _build_passive_pair_row(mock)

        self.assertFalse(result["task_execution_evaluated"])
        self.assertIsNone(result["downstream_task_success"])
        self.assertEqual(result["execute_action"], "not_executed")
        self.assertIn("prepared_passive_control_reference_only", cast(str, result["execution_boundary"]))

    def test_passive_row_derived_from_is_correct(self) -> None:
        mock = _make_mock_active_row()
        result = _build_passive_pair_row(mock)

        self.assertEqual(result["row_type"], "derived_passive")
        self.assertIn("derived_from_recorded_active_row", cast(str, result["derived_from"]))
        self.assertNotEqual(result["derived_from"], "recorded_active_closed_loop_row")

    def test_passive_row_has_no_bridge_metadata(self) -> None:
        mock = _make_mock_active_row()
        result = _build_passive_pair_row(mock)

        self.assertEqual(result["bridge_mode"], "passive_no_bridge")
        self.assertFalse(result["used_teleportfull_for_revisit"])
        self.assertFalse(result["task_bridge_used_teleportfull"])
        self.assertEqual(result["revisit_path_steps"], 0)
        self.assertEqual(result["task_bridge_path_steps"], 0)

    def test_passive_row_has_all_required_fields(self) -> None:
        mock = _make_mock_active_row()
        result = _build_passive_pair_row(mock)
        for field in REQUIRED_PAIR_FIELDS:
            self.assertIn(field, result, f"passive row missing required field: {field}")


class BuildPairsTest(unittest.TestCase):
    def test_two_active_rows_produce_four_paired_rows(self) -> None:
        rows = [
            _make_mock_active_row(row_idx=0, scene="FloorPlan1", seed=29, target="Apple"),
            _make_mock_active_row(row_idx=1, scene="FloorPlan201", seed=29, target="Pencil"),
        ]
        paired = build_pairs(rows)
        self.assertEqual(len(paired), 4)

        active_count = sum(1 for r in paired if r["arm"] == "active")
        passive_count = sum(1 for r in paired if r["arm"] == "passive")
        self.assertEqual(active_count, 2)
        self.assertEqual(passive_count, 2)

    def test_active_and_passive_share_case_id(self) -> None:
        rows = [_make_mock_active_row()]
        paired = build_pairs(rows)

        self.assertEqual(paired[0]["case_id"], paired[1]["case_id"])
        self.assertNotEqual(paired[0]["arm"], paired[1]["arm"])


class SummarizePairsTest(unittest.TestCase):
    def _make_paired_rows(self, n: int = 3) -> list[dict[str, object]]:
        paired: list[dict[str, object]] = []
        for i in range(n):
            mock = _make_mock_active_row(
                row_idx=i,
                scene="FloorPlan1",
                seed=29 + i,
                target="Apple",
            )
            paired.append(_build_active_pair_row(mock))
            paired.append(_build_passive_pair_row(mock))
        return paired

    def test_summary_has_all_required_fields(self) -> None:
        paired = self._make_paired_rows(3)
        summary = summarize_pairs(paired, source_active_artifact="test.json")

        for field in REQUIRED_SUMMARY_FIELDS:
            self.assertIn(field, summary, f"summary missing required field: {field}")

    def test_summary_counts(self) -> None:
        paired = self._make_paired_rows(3)
        summary = summarize_pairs(paired, source_active_artifact="test.json")

        self.assertEqual(summary["n_pairs"], 3)
        self.assertEqual(summary["active_successes"], 3)
        self.assertEqual(summary["passive_successes"], 0)
        self.assertIsNone(summary["paired_delta"])
        self.assertFalse(summary["paired_delta_evaluable"])
        self.assertEqual(
            summary["paired_delta_not_evaluable_reason"],
            "passive_arm_not_live_executed_or_not_all_task_evaluated",
        )
        self.assertEqual(summary["active_task_evaluated"], 3)
        self.assertEqual(summary["passive_task_evaluated"], 0)
        self.assertEqual(summary["passive_prepared_rows"], 3)

    def test_passive_is_not_live_execution(self) -> None:
        paired = self._make_paired_rows(1)
        summary = summarize_pairs(paired)

        self.assertFalse(summary["passive_is_live_execution"])
        self.assertEqual(
            summary["passive_derivation_method"],
            "derived_from_recorded_active_rows_no_live_passive_controller",
        )

    def test_prepared_passive_rows_are_not_counted_as_failures(self) -> None:
        paired = self._make_paired_rows(3)
        summary = summarize_pairs(paired)

        self.assertEqual(summary["passive_stale_failures"], 0)
        self.assertEqual(summary["passive_prepared_stale_rows"], 3)

    def test_evaluated_failed_passive_rows_count_as_stale_failures(self) -> None:
        mock = _make_mock_active_row()
        passive = _build_passive_pair_row(mock)
        passive["task_execution_evaluated"] = True
        passive["downstream_task_success"] = False
        passive["execute_action"] = "PickupObject"
        paired = [_build_active_pair_row(mock), passive]

        summary = summarize_pairs(paired)
        self.assertEqual(summary["passive_stale_failures"], 1)
        self.assertEqual(summary["passive_prepared_stale_rows"], 0)

    def test_no_oracle_stale_passive_are_not_failures(self) -> None:
        mock = _make_mock_active_row(oracle_stale_label=False)
        paired = [
            _build_active_pair_row(mock),
            _build_passive_pair_row(mock),
        ]
        summary = summarize_pairs(paired)
        self.assertEqual(summary["passive_stale_failures"], 0)


class SchemaCheckTest(unittest.TestCase):
    def test_valid_pairs_pass_schema_check(self) -> None:
        mock = _make_mock_active_row()
        paired = [
            _build_active_pair_row(mock),
            _build_passive_pair_row(mock),
        ]
        summary = summarize_pairs(paired)
        result = _check_schema(paired, summary)

        self.assertTrue(result["schema_ok"])
        self.assertEqual(result["n_violations"], 0)

    def test_missing_field_detected(self) -> None:
        mock = _make_mock_active_row()
        paired = [
            _build_active_pair_row(mock),
            _build_passive_pair_row(mock),
        ]
        paired[0].pop("case_id", None)
        summary = summarize_pairs(paired)
        result = _check_schema(paired, summary)

        self.assertFalse(result["schema_ok"])
        self.assertGreater(cast(int, result["n_violations"]), 0)

    def test_active_with_false_verifier_used_detected(self) -> None:
        mock = _make_mock_active_row(verifier_used=False)
        paired = [
            _build_active_pair_row(mock),
            _build_passive_pair_row(mock),
        ]
        paired[0]["verifier_used"] = False
        summary = summarize_pairs(paired)
        result = _check_schema(paired, summary)

        self.assertFalse(result["schema_ok"])
        violations_str = " ".join(str(v) for v in cast(list[object], result["violations"]))
        self.assertIn("verifier_used=true", violations_str)

    def test_passive_with_memory_mutated_detected(self) -> None:
        mock = _make_mock_active_row()
        paired = [
            _build_active_pair_row(mock),
            _build_passive_pair_row(mock),
        ]
        paired[1]["memory_mutated"] = True
        summary = summarize_pairs(paired)
        result = _check_schema(paired, summary)

        self.assertFalse(result["schema_ok"])
        violations_str = " ".join(str(v) for v in cast(list[object], result["violations"]))
        self.assertIn("memory_mutated=false", violations_str)

    def test_unbalanced_pairs_detected(self) -> None:
        mock = _make_mock_active_row()
        paired = [
            _build_active_pair_row(mock),
        ]
        summary = summarize_pairs(paired)
        result = _check_schema(paired, summary)

        self.assertFalse(result["schema_ok"])

    def test_forbidden_claim_terms_rejected(self) -> None:
        paired = [
            _build_active_pair_row(_make_mock_active_row()),
            _build_passive_pair_row(_make_mock_active_row()),
        ]
        summary = summarize_pairs(paired)
        summary["claim_boundary"] = "This claims ObjectNav and SPL support."
        result = _check_schema(paired, summary)

        self.assertFalse(result["schema_ok"])
        violations_str = " ".join(str(v) for v in cast(list[object], result["violations"]))
        self.assertIn("positively asserts", violations_str)

    def test_passive_derived_from_must_not_be_recorded_active(self) -> None:
        mock = _make_mock_active_row()
        paired = [
            _build_active_pair_row(mock),
            _build_passive_pair_row(mock),
        ]
        paired[1]["derived_from"] = "recorded_active_closed_loop_row"
        summary = summarize_pairs(paired)
        result = _check_schema(paired, summary)

        self.assertFalse(result["schema_ok"])


class ClaimBoundaryInvariantsTest(unittest.TestCase):
    def test_claim_boundary_contains_declaration(self) -> None:
        self.assertIn("recorded/prepared", CLAIM_BOUNDARY)

    def test_claim_boundary_forbids_forbidden_terms(self) -> None:
        for term in FORBIDDEN_POSITIVE_CLAIM_TERMS:
            self.assertFalse(
                _term_in_positive_context(term, CLAIM_BOUNDARY),
                f"claim boundary positively asserts forbidden term: {term}",
            )

    def test_claim_boundary_explicitly_negates_all_required_terms(self) -> None:
        for term in FORBIDDEN_POSITIVE_CLAIM_TERMS:
            self.assertFalse(
                _term_in_positive_context(term, CLAIM_BOUNDARY),
                f"claim boundary does not explicitly forbid: {term}",
            )

    def test_claim_boundary_explicitly_names_passive_as_derived(self) -> None:
        self.assertIn("derived", CLAIM_BOUNDARY.lower())
        self.assertIn("no new live experiment", CLAIM_BOUNDARY.lower())

    def test_passive_row_execution_boundary_is_safe(self) -> None:
        mock = _make_mock_active_row(task_execution_evaluated=True)
        passive = _build_passive_pair_row(mock)
        boundary = str(passive["execution_boundary"])
        for term in FORBIDDEN_POSITIVE_CLAIM_TERMS:
            self.assertFalse(
                _term_in_positive_context(term, boundary),
                f"passive boundary positively asserts: {term}",
            )


class WriteOutputsTest(unittest.TestCase):
    def test_write_outputs_creates_all_files(self) -> None:
        mock = _make_mock_active_row()
        paired = [
            _build_active_pair_row(mock),
            _build_passive_pair_row(mock),
        ]
        summary = summarize_pairs(paired, source_active_artifact="test.json")
        schema_check = _check_schema(paired, summary)
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "claim_boundary": CLAIM_BOUNDARY,
            "source_active_artifact": "test.json",
            "paired_rows": paired,
            "summary": summary,
            "schema_check": schema_check,
        }

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            write_outputs(result, out_dir)

            self.assertTrue((out_dir / "paired_task_bridge_control.json").exists())
            self.assertTrue((out_dir / "paired_task_bridge_control.csv").exists())
            self.assertTrue((out_dir / "paired_task_bridge_control_summary.csv").exists())
            self.assertTrue((out_dir / "README.md").exists())
            self.assertTrue((out_dir / "schema_check.json").exists())

    def test_readme_contains_key_metrics(self) -> None:
        mock = _make_mock_active_row()
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "claim_boundary": CLAIM_BOUNDARY,
            "source_active_artifact": "test.json",
            "paired_rows": [_build_active_pair_row(mock), _build_passive_pair_row(mock)],
            "summary": summarize_pairs([_build_active_pair_row(mock), _build_passive_pair_row(mock)]),
            "schema_check": {"schema_ok": True, "n_violations": 0},
        }
        readme = _render_readme(result)

        self.assertIn("n_pairs", readme)
        self.assertIn("active_successes", readme)
        self.assertIn("passive_successes", readme)
        self.assertIn("paired_delta", readme)
        self.assertIn("paired_delta_evaluable", readme)
        self.assertIn("passive_stale_failures", readme)
        self.assertIn("prepared/recorded", readme.lower())
        self.assertIn("no live passive controller", readme.lower())

    def test_schema_check_json_is_valid(self) -> None:
        mock = _make_mock_active_row()
        paired = [_build_active_pair_row(mock), _build_passive_pair_row(mock)]
        summary = summarize_pairs(paired)
        schema_check = _check_schema(paired, summary)

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            result = {
                "schema_version": SCHEMA_VERSION,
                "status": "ok",
                "claim_boundary": CLAIM_BOUNDARY,
                "source_active_artifact": "test.json",
                "paired_rows": paired,
                "summary": summary,
                "schema_check": schema_check,
            }
            write_outputs(result, out_dir)
            sc_data = cast(dict[str, object], json.loads((out_dir / "schema_check.json").read_text(encoding="utf-8")))
            self.assertTrue(sc_data["schema_ok"])

    def test_csv_fieldnames_includes_all_required(self) -> None:
        mock = _make_mock_active_row()
        paired = [_build_active_pair_row(mock)]
        fieldnames = _csv_fieldnames(paired)
        for field in REQUIRED_PAIR_FIELDS:
            self.assertIn(field, fieldnames, f"CSV fieldnames missing: {field}")


class BuildPairedControlIntegrationTest(unittest.TestCase):
    def test_full_pipeline_with_minimal_artifact(self) -> None:
        rows = [
            _make_mock_active_row(row_idx=0, scene="FloorPlan1", seed=29, target="Apple"),
            _make_mock_active_row(row_idx=1, scene="FloorPlan201", seed=29, target="Pencil"),
        ]
        artifact = {
            "schema_version": "ai2thor_live_gsam_closed_loop.v1",
            "status": "ok",
            "claim_boundary": "test boundary",
            "summary": {"rows": 2},
            "rows": rows,
        }

        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "live_gsam_closed_loop.json"
            _ = artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

            result = build_paired_control(artifact_path, source_label="test.json")

            self.assertEqual(result["schema_version"], SCHEMA_VERSION)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["source_active_artifact"], "test.json")

            paired_rows = cast(list[dict[str, object]], result["paired_rows"])
            self.assertEqual(len(paired_rows), 4)

            summary = cast(dict[str, object], result["summary"])
            self.assertEqual(summary["n_pairs"], 2)

            schema_check = cast(dict[str, object], result["schema_check"])
            self.assertTrue(schema_check["schema_ok"])

    def test_empty_artifact_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "empty.json"
            _ = artifact_path.write_text(json.dumps({"rows": []}), encoding="utf-8")
            with self.assertRaises(ValueError):
                build_paired_control(artifact_path)

    def test_non_dict_artifact_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "bad.json"
            _ = artifact_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            with self.assertRaises(ValueError):
                build_paired_control(artifact_path)


class BuildLivePassivePairRowTest(unittest.TestCase):
    def test_live_passive_row_has_correct_provenance(self) -> None:
        mock = _make_mock_active_row()
        passive_task_fields: dict[str, object] = {
            "task_execution_evaluated": True,
            "downstream_task_success": True,
            "execute_action": "PickupObject",
            "execution_boundary": "live_passive_task_bridge_from_stale_memory",
            "task_bridge_navigation_mode": "stepwise_navigation_live_passive",
            "task_bridge_used_teleportfull": False,
            "task_bridge_path_steps": 3,
        }
        result = _build_live_passive_pair_row(mock, passive_task_fields)

        self.assertEqual(result["arm"], "passive")
        self.assertEqual(result["row_type"], "live_passive")
        self.assertEqual(result["derived_from"], "live_passive_controller_from_recorded_active_case")
        self.assertFalse(result["verifier_used"])
        self.assertEqual(result["verifier_name"], "none_passive_arm")
        self.assertFalse(result["memory_mutated"])
        self.assertEqual(result["memory_update_action"], "none_live_passive_arm")
        self.assertIsNone(result["decision_stale"])
        self.assertTrue(result["task_execution_evaluated"])
        self.assertTrue(result["downstream_task_success"])
        self.assertFalse(result["task_bridge_used_teleportfull"])
        self.assertFalse(result["used_teleportfull_for_revisit"])

    def test_live_passive_row_has_all_required_fields(self) -> None:
        mock = _make_mock_active_row()
        passive_task_fields: dict[str, object] = {
            "task_execution_evaluated": True,
            "downstream_task_success": False,
            "execute_action": "PickupObject",
            "execution_boundary": "live_passive",
            "task_bridge_navigation_mode": "stepwise_navigation_live_passive",
            "task_bridge_used_teleportfull": False,
            "task_bridge_path_steps": 0,
        }
        result = _build_live_passive_pair_row(mock, passive_task_fields)
        for field in REQUIRED_PAIR_FIELDS:
            self.assertIn(field, result, f"live passive row missing required field: {field}")

    def test_live_passive_row_is_evaluable_not_prepared(self) -> None:
        mock = _make_mock_active_row()
        passive_task_fields: dict[str, object] = {
            "task_execution_evaluated": True,
            "downstream_task_success": True,
            "execute_action": "PickupObject",
            "execution_boundary": "live_passive",
            "task_bridge_navigation_mode": "stepwise_navigation_live_passive",
            "task_bridge_used_teleportfull": False,
            "task_bridge_path_steps": 3,
        }
        result = _build_live_passive_pair_row(mock, passive_task_fields)
        self.assertTrue(result["task_execution_evaluated"])
        self.assertNotEqual(result["execute_action"], "not_executed")


class LivePassiveSummaryTest(unittest.TestCase):
    def test_all_live_passive_rows_produce_evaluable_delta(self) -> None:
        mock = _make_mock_active_row(downstream_task_success=True)
        active = _build_active_pair_row(mock)
        passive_task_fields: dict[str, object] = {
            "task_execution_evaluated": True,
            "downstream_task_success": False,
            "execute_action": "PickupObject",
            "execution_boundary": "live_passive",
            "task_bridge_navigation_mode": "stepwise_navigation_live_passive",
            "task_bridge_used_teleportfull": False,
            "task_bridge_path_steps": 2,
        }
        passive = _build_live_passive_pair_row(mock, passive_task_fields)
        paired = [active, passive]
        summary = summarize_pairs(paired, source_active_artifact="test.json")

        self.assertTrue(summary["passive_is_live_execution"])
        self.assertEqual(summary["passive_derivation_method"], "live_passive_controller_from_recorded_active_cases")
        self.assertTrue(summary["paired_delta_evaluable"])
        self.assertEqual(summary["paired_delta"], 1)
        self.assertEqual(summary["passive_live_rows"], 1)
        self.assertIn("live_passive", str(summary["claim_boundary"]))

    def test_mixed_prepared_and_live_passive_rows_are_not_live(self) -> None:
        mock1 = _make_mock_active_row(row_idx=0, scene="FloorPlan1", seed=29, target="Apple")
        mock2 = _make_mock_active_row(row_idx=1, scene="FloorPlan1", seed=11, target="Pencil")
        active1 = _build_active_pair_row(mock1)
        active2 = _build_active_pair_row(mock2)
        passive1 = _build_passive_pair_row(mock1)  # prepared
        passive_task_fields: dict[str, object] = {
            "task_execution_evaluated": True,
            "downstream_task_success": False,
            "execute_action": "PickupObject",
            "execution_boundary": "live_passive",
            "task_bridge_navigation_mode": "stepwise_navigation_live_passive",
            "task_bridge_used_teleportfull": False,
            "task_bridge_path_steps": 0,
        }
        passive2 = _build_live_passive_pair_row(mock2, passive_task_fields)  # live

        paired = [active1, passive1, active2, passive2]
        summary = summarize_pairs(paired)

        self.assertFalse(summary["passive_is_live_execution"])
        self.assertEqual(summary["passive_live_rows"], 1)
        self.assertIsNone(summary["paired_delta"])
        self.assertEqual(
            summary["passive_derivation_method"],
            "derived_from_recorded_active_rows_no_live_passive_controller",
        )

    def test_live_passive_zero_pairs_delta_is_zero(self) -> None:
        mock = _make_mock_active_row(downstream_task_success=True)
        passive_task_fields: dict[str, object] = {
            "task_execution_evaluated": True,
            "downstream_task_success": True,
            "execute_action": "PickupObject",
            "execution_boundary": "live_passive",
            "task_bridge_navigation_mode": "stepwise_navigation_live_passive",
            "task_bridge_used_teleportfull": False,
            "task_bridge_path_steps": 0,
        }
        active = _build_active_pair_row(mock)
        passive = _build_live_passive_pair_row(mock, passive_task_fields)
        paired = [active, passive]
        summary = summarize_pairs(paired)

        self.assertEqual(summary["paired_delta"], 0)
        self.assertTrue(summary["paired_delta_evaluable"])

    def test_live_passive_summary_has_required_fields(self) -> None:
        mock = _make_mock_active_row()
        passive_task_fields: dict[str, object] = {
            "task_execution_evaluated": True,
            "downstream_task_success": True,
            "execute_action": "PickupObject",
            "execution_boundary": "live_passive",
            "task_bridge_navigation_mode": "stepwise_navigation_live_passive",
            "task_bridge_used_teleportfull": False,
            "task_bridge_path_steps": 1,
        }
        paired = [
            _build_active_pair_row(mock),
            _build_live_passive_pair_row(mock, passive_task_fields),
        ]
        summary = summarize_pairs(paired)
        for field in REQUIRED_SUMMARY_FIELDS:
            self.assertIn(field, summary, f"live-passive summary missing required field: {field}")


class BuildLivePassivePairedControlTest(unittest.TestCase):
    def test_live_passive_build_yields_evaluable_paired_rows(self) -> None:
        rows = [
            _make_mock_active_row(row_idx=0, scene="FloorPlan1", seed=29, target="Apple"),
            _make_mock_active_row(row_idx=1, scene="FloorPlan201", seed=29, target="Pencil"),
        ]
        artifact = {
            "schema_version": "ai2thor_live_gsam_closed_loop.v1",
            "status": "ok",
            "claim_boundary": "test boundary",
            "summary": {"rows": 2},
            "rows": rows,
        }
        controllers: list[FakePassiveTaskBridgeController] = []

        def controller_factory(**kwargs: object) -> FakePassiveTaskBridgeController:
            _ = kwargs
            controller = FakePassiveTaskBridgeController()
            controllers.append(controller)
            return controller

        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "live_gsam_closed_loop.json"
            _ = artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            result = build_live_passive_paired_control(
                artifact_path,
                source_label="test.json",
                controller_factory=controller_factory,
            )

        self.assertEqual(result["status"], "ok")
        self.assertIn("live_passive", str(result["claim_boundary"]))

        paired_rows = cast(list[dict[str, object]], result["paired_rows"])
        self.assertEqual(len(paired_rows), 4)

        passive_rows = [r for r in paired_rows if r["arm"] == "passive"]
        self.assertEqual(len(passive_rows), 2)
        for pr in passive_rows:
            self.assertEqual(pr["row_type"], "live_passive")
            self.assertFalse(pr["verifier_used"])
            self.assertFalse(pr["memory_mutated"])
            self.assertTrue(pr["task_execution_evaluated"])
            self.assertEqual(pr["execute_action"], "PickupObject")
            self.assertEqual(pr["bridge_mode"], "stepwise_navigation_live_passive")
            self.assertGreater(cast(int, pr["task_bridge_path_steps"]), 0)
            self.assertFalse(pr["task_bridge_used_teleportfull"])

        action_names = [action for controller in controllers for action, _ in controller.actions]
        self.assertIn("InitialRandomSpawn", action_names)
        self.assertIn("GetReachablePositions", action_names)
        self.assertIn("PickupObject", action_names)
        self.assertNotIn("TeleportFull", action_names)
        self.assertNotIn("TeleportObject", action_names)

        summary = cast(dict[str, object], result["summary"])
        self.assertTrue(summary["passive_is_live_execution"])
        self.assertTrue(summary["paired_delta_evaluable"])
        self.assertIsNotNone(summary["paired_delta"])
        self.assertEqual(summary["passive_live_rows"], 2)

    def test_live_passive_builder_records_controller_failure_as_delta(self) -> None:
        rows = [_make_mock_active_row(row_idx=0, scene="FloorPlan1", seed=29, target="Apple")]
        artifact = {
            "schema_version": "ai2thor_live_gsam_closed_loop.v1",
            "status": "ok",
            "rows": rows,
        }
        controllers: list[FakePassiveNavigationFailureController] = []

        def controller_factory(**kwargs: object) -> FakePassiveNavigationFailureController:
            _ = kwargs
            controller = FakePassiveNavigationFailureController()
            controllers.append(controller)
            return controller

        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "live_gsam_closed_loop.json"
            _ = artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            result = build_live_passive_paired_control(
                artifact_path,
                controller_factory=controller_factory,
            )

        paired_rows = cast(list[dict[str, object]], result["paired_rows"])
        passive = [r for r in paired_rows if r["arm"] == "passive"][0]
        self.assertTrue(passive["task_execution_evaluated"])
        self.assertFalse(passive["downstream_task_success"])
        self.assertEqual(passive["task_execution_failure_reason"], "collision_stuck")
        self.assertEqual(passive["task_bridge_navigation_failure_reason"], "collision_stuck")

        summary = cast(dict[str, object], result["summary"])
        self.assertTrue(summary["paired_delta_evaluable"])
        self.assertEqual(summary["paired_delta"], 1)
        self.assertEqual(summary["passive_successes"], 0)

        action_names = [action for controller in controllers for action, _ in controller.actions]
        self.assertIn("InitialRandomSpawn", action_names)
        self.assertIn("GetReachablePositions", action_names)
        self.assertNotIn("TeleportFull", action_names)
        self.assertNotIn("TeleportObject", action_names)

        schema_check = cast(dict[str, object], result["schema_check"])
        self.assertTrue(schema_check["schema_ok"])

    def test_live_passive_empty_artifact_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "empty.json"
            _ = artifact_path.write_text(json.dumps({"rows": []}), encoding="utf-8")
            with self.assertRaises(ValueError):
                build_live_passive_paired_control(artifact_path)

    def test_prepared_only_behavior_remains_unchanged(self) -> None:
        """Verify prepared-only build_paired_control still produces non-evaluable passive rows."""
        rows = [_make_mock_active_row()]
        artifact = {
            "schema_version": "ai2thor_live_gsam_closed_loop.v1",
            "status": "ok",
            "rows": rows,
        }
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "source.json"
            _ = artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            result = build_paired_control(artifact_path)

        paired_rows = cast(list[dict[str, object]], result["paired_rows"])
        passive_rows = [r for r in paired_rows if r["arm"] == "passive"]
        self.assertEqual(len(passive_rows), 1)
        pr = passive_rows[0]
        self.assertEqual(pr["row_type"], "derived_passive")
        self.assertFalse(pr["task_execution_evaluated"])
        self.assertEqual(pr["execute_action"], "not_executed")
        self.assertIsNone(pr["downstream_task_success"])

        summary = cast(dict[str, object], result["summary"])
        self.assertFalse(summary["passive_is_live_execution"])
        self.assertIsNone(summary["paired_delta"])
        self.assertEqual(summary["passive_live_rows"], 0)


class LivePassiveClaimBoundaryTest(unittest.TestCase):
    def test_live_passive_boundary_contains_expected_markers(self) -> None:
        self.assertIn("live_passive", LIVE_PASSIVE_CLAIM_BOUNDARY)
        self.assertIn("evaluable", LIVE_PASSIVE_CLAIM_BOUNDARY)
        self.assertIn("stale memory_old_position", LIVE_PASSIVE_CLAIM_BOUNDARY)
        self.assertIn("without verifier evidence", LIVE_PASSIVE_CLAIM_BOUNDARY.lower())
        self.assertIn("without memory mutation", LIVE_PASSIVE_CLAIM_BOUNDARY.lower())

    def test_live_passive_boundary_forbids_forbidden_terms(self) -> None:
        for term in FORBIDDEN_POSITIVE_CLAIM_TERMS:
            self.assertFalse(
                _term_in_positive_context(term, LIVE_PASSIVE_CLAIM_BOUNDARY),
                f"live-passive claim boundary positively asserts forbidden term: {term}",
            )

    def test_live_passive_boundary_is_distinct_from_prepared(self) -> None:
        self.assertNotEqual(LIVE_PASSIVE_CLAIM_BOUNDARY, CLAIM_BOUNDARY)
        self.assertNotIn("recorded/prepared", LIVE_PASSIVE_CLAIM_BOUNDARY)


if __name__ == "__main__":
    _ = unittest.main()
