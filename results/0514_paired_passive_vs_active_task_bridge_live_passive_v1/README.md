# Paired Passive-vs-Active Task Bridge Control

Schema: `ai2thor_paired_task_bridge_control.v1`
Status: `ok`
Source active artifact: `results/ai2thor_live_gsam_mixed_challenge_budgetfix_v1/live_gsam_closed_loop.json`

## Summary

| Metric | Value |
|--------|-------|
| n_pairs | 6 |
| active_successes | 6 |
| passive_successes | 6 |
| paired_delta | 0 |
| paired_delta_evaluable | True |
| paired_delta_not_evaluable_reason | `` |
| passive_stale_failures | 0 |
| passive_prepared_stale_rows | 0 |
| active_task_evaluated | 6 |
| passive_task_evaluated | 6 |
| passive_prepared_rows | 0 |
| passive_live_rows | 6 |
| passive_is_live_execution | True |
| passive_derivation_method | `live_passive_controller_from_recorded_active_cases` |

## Claim Boundary

> live_passive paired-vs-active paired-control artifact; passive rows represent evaluable live controller execution derived from recorded active cases using stale memory_old_position without verifier evidence and without memory mutation; active rows are recorded as-is from source active closed-loop run; uses AI2-THOR controller only for the passive arm; no GSAM verifier, GPU verifier, Habitat, manipulation, ObjectNav/SPL, or persistent memory workload; not a manipulation benchmark; not cross-platform transfer (Habitat/Warehouse/ObjectNav); not persistent memory writeback or repair; not broad active-maintenance generalization; not passive-vs-active superiority claim; fixed-case evaluability only

## Schema Check

Schema ok: `True`
Violations: 0

This is a live-passive paired-control artifact. Passive rows are evaluable live controller
executions derived from recorded active cases using stale memory_old_position.
The passive arm uses an AI2-THOR controller only; no verifier evidence, GSAM/GPU verifier,
external services, or memory mutation are used for the passive arm.