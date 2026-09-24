# Pre-Registered Two-Case Closed-Loop Follow-Up Protocol

Date: 2026-05-25

## Purpose

Freeze a tiny follow-up protocol before executing a bounded reproduction sweep for the two already-positive FloorPlan1/Apple closed-loop cases.

## Fixed Cases

Run exactly these cases:

- FloorPlan1 / seed 29 / Apple
- FloorPlan1 / seed 11 / Apple

No additional scenes, seeds, targets, retries, target substitution, case-list expansion, or broad sweep is allowed for this artifact.

## Fixed Runtime Protocol

Use the existing live GSAM closed-loop runner with:

- `--scenes FloorPlan1`
- `--seeds 29 11`
- `--target-objects Apple`
- `--targets-per-scene-seed 1`
- `--max-rows 2`
- `--verification-budget 2`
- `--active-verification-threshold -1.0`
- `--ev-variant calibrated`
- `--revisit-mode stepwise`
- `--execute-task-bridge`

The measured revisit path and measured task-bridge path must not use `TeleportFull`.

## Pass Criteria

Each case passes only if all of the following are true:

- `spawn_success=true`
- `revisit_action=stepwise_navigation`
- `used_teleportfull_for_revisit=false`
- `revisit_navigation_failure_reason=null`
- `policy_should_verify=true`
- `verifier_used=true`
- `decision_stale=true`
- `memory_mutated=true`
- `task_execution_evaluated=true`
- `task_bridge_navigation_mode=stepwise_navigation`
- `task_bridge_used_teleportfull=false`
- `task_bridge_navigation_failure_reason=null`
- `execute_action=PickupObject`
- `downstream_task_success=true`

The artifact-level reproduction passes only if both fixed cases pass. Any missing row or extra row is a protocol failure.

## Failure Criteria

The artifact fails if any case uses `TeleportFull` on a measured path, skips GSAM verification, fails to mutate memory after detector-backed stale verification, fails stepwise task-bridge navigation, fails `PickupObject`, or if the run includes any unregistered case.

## Claim Boundary

If both cases pass, the only supported claim is that the two pre-registered FloorPlan1/Apple positives reproduced under the same closed-loop non-`TeleportFull` GSAM-memory-task-bridge protocol.

This does not establish broad scaling across seeds, rooms, objects, scenes, target categories, or blocked cases. Known blockers remain: FloorPlan1/seed7/Apple non-`TeleportFull` bridge `collision_stuck`, and FloorPlan201/Pencil stepwise bridge failures (`collision_stuck` or `max_steps_exceeded`).
