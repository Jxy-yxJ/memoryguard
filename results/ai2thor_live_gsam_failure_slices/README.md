# Live GSAM Failure Slices

Schema: `ai2thor_live_gsam_failure_slices.v1`
Status: `ok`
Input rows: 20
Labelled rows: 20
Memory updated rows: 0

Claim boundary: CPU-only post-processing of a previously recorded AI2-THOR controller-backed GSAM detector artifact; oracle labels are evaluation-only detector outcome labels; no new live run; no AI2-THOR, GSAM, Habitat, GPU, navigation, manipulation, ObjectNav, task success, recovery-search success, live policy superiority, no memory writeback/repair, and no persistent memory update

This is CPU-only post-processing of recorded detector rows. It does not launch AI2-THOR, GSAM, Habitat, GPU workloads, navigation, manipulation, recovery search, or memory writeback.
Oracle labels are used only as post-hoc evaluation labels, never as policy inputs or ranking features.

## Detector Outcomes
- `tp`: 13
- `fp`: 0
- `tn`: 2
- `fn`: 5
- `unlabelled`: 0

## Grouped Recorded Fields
- `scene`
- `target`
- `visibility_error`
- `target_visible_after_revisit_sweep`
- `decision_reason`
- `candidate_selected_by_budget`
