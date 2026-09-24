# Paired Passive-vs-Active Task Bridge Control

Schema: `ai2thor_paired_task_bridge_control.v1`
Status: `ok`
Source active artifact: `results/ai2thor_live_gsam_mixed_challenge_budgetfix_v1/live_gsam_closed_loop.json`

## Summary

| Metric | Value |
|--------|-------|
| n_pairs | 6 |
| active_successes | 6 |
| passive_successes | 0 |
| paired_delta | None |
| paired_delta_evaluable | False |
| paired_delta_not_evaluable_reason | `passive_arm_not_live_executed_or_not_all_task_evaluated` |
| passive_stale_failures | 0 |
| passive_prepared_stale_rows | 6 |
| active_task_evaluated | 6 |
| passive_task_evaluated | 0 |
| passive_prepared_rows | 6 |
| passive_is_live_execution | False |
| passive_derivation_method | `derived_from_recorded_active_rows_no_live_passive_controller` |

## Claim Boundary

> recorded/prepared passive-vs-active paired-control artifact derived from recorded active closed-loop controller-backed GSAM stale-memory detection artifact; passive rows are derived/synthetic and do NOT represent a live passive controller execution, only the recorded active evidence from the source artifact; active rows are recorded as-is from the source active closed-loop run; no new live experiment result; no AI2-THOR, GSAM, Habitat, navigation, manipulation, ObjectNav/SPL, or GPU workloads; not a manipulation benchmark; not cross-platform transfer (Habitat/Warehouse/ObjectNav); not persistent memory writeback or repair; not broad active-maintenance generalization; passive task-execution fields reflect whether downstream task was evaluated or only prepared/skipped

## Schema Check

Schema ok: `True`
Violations: 0

This is a prepared/recorded paired-control artifact. Passive rows are derived from recorded active rows.
No live passive controller was executed. No AI2-THOR, GSAM, GPU, or external services were used.
