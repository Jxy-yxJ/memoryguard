# Live GSAM False-Negative Failure Cases

Schema: `ai2thor_live_gsam_failure_cases.v1`
Status: `ok`
Input rows: 20
False-negative rows: 5
Case rows: 5
Memory updated rows: 0

Claim boundary: CPU-only materialization of qualitative false-negative cases from a previously recorded AI2-THOR controller-backed GSAM detector artifact; oracle labels post-hoc only; not new live run/navigation/manipulation/ObjectNav/task success/recovery-search/policy superiority/Habitat transfer/memory writeback/memory repair

This table is CPU-only materialization from a previously recorded artifact and is not new live evidence.
Oracle labels are used only to select recorded false negatives, then not used for within-subset ranking.
The outputs do not launch AI2-THOR, GSAM, Habitat, GPU workloads, navigation, manipulation, recovery search, or memory writeback.

## Cases
| row_idx | scene | seed | target | visibility_error | target_visible_after_revisit_sweep | decision_reason | decision_matched_distance | decision_detections_considered | expected_verification_value | p_stale_estimate | candidate_rank_by_ev | candidate_selected_by_budget | oracle_stale_label | decision_stale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 19 | FloorPlan1 | 31 | Book | target_not_visible_after_revisit_sweep | false | nearest_after_detection_normalized_frame_distance | 0.0278 | 30 |  |  |  |  | true | false |
| 4 | FloorPlan201 | 7 | Newspaper | target_not_visible_after_revisit_sweep | false | nearest_after_detection_normalized_frame_distance | 0.0444 | 57 |  |  |  |  | true | false |
| 10 | FloorPlan201 | 11 | Newspaper | target_not_visible_after_revisit_sweep | false | nearest_after_detection_normalized_frame_distance | 0.0377 | 61 |  |  |  |  | true | false |
| 16 | FloorPlan201 | 29 | Newspaper | target_not_visible_after_revisit_sweep | false | nearest_after_detection_normalized_frame_distance | 0.0377 | 73 |  |  |  |  | true | false |
| 1 | FloorPlan1 | 7 | Book |  | true | nearest_after_detection_normalized_frame_distance | 0.0277 | 16 |  |  |  |  | true | false |
