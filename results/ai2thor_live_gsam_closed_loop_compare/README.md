# Live GSAM Closed Loop Compare

Schema: `ai2thor_live_gsam_closed_loop_compare.v1`
Status: `ok`
Input files: 1
Input rows: 20

Claim boundary: CPU-only recorded-row comparison of existing live_gsam_closed_loop artifacts; no new live experiment result; no AI2-THOR, GSAM, Habitat, navigation, manipulation, ObjectNav, task success, persistent memory repair/writeback, recovery-search, or Habitat object-level transfer; oracle labels are evaluation-only and never used for policy inclusion or ranking

This is a CPU-only recorded-row comparison. It does not launch AI2-THOR, GSAM, Habitat, navigation, memory writeback, or GPU workloads.
Oracle labels are used only for evaluation metrics, never for policy inclusion or ranking.

## Variants
- `passive_no_verification`: selected=0, agreement_rate=None, memory_updated_rows=0
- `recorded_ev_gate`: selected=20, agreement_rate=0.75, memory_updated_rows=0
- `recorded_budgeted_ev_gate`: selected=20, agreement_rate=0.75, memory_updated_rows=0
