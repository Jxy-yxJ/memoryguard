# Pre-Registered Two-Case Closed-Loop Result-to-Claim

**Verdict**: `yes_preregistered_two_case_floorplan1_apple_reproduction; no_broad_scale_claim`.

The frozen two-case protocol in `FOLLOWUP_PROTOCOL.md` ran exactly `FloorPlan1/seed29/Apple` and `FloorPlan1/seed11/Apple` in one fixed live GSAM closed-loop sweep. Both rows passed the pre-registered checklist: live spawn, stepwise revisit without `TeleportFull`, GSAM stale verification, detector-backed in-memory refresh, stepwise task bridge without `TeleportFull`, and downstream `PickupObject` success.

## Outcome Table

| Scene | Seed | Target | Revisit steps | Bridge steps | Pass |
| --- | ---: | --- | ---: | ---: | --- |
| FloorPlan1 | 29 | Apple | 2 | 5 | yes |
| FloorPlan1 | 11 | Apple | 2 | 4 | yes |

## Boundary

This supports only pre-registered reproduction of the two selected FloorPlan1/Apple positives. It does not establish broad scaling across scenes, seeds, rooms, objects, or target categories; it is not ObjectNav/SPL, not a manipulation benchmark, not passive-vs-active task-performance superiority, not persistent memory writeback, not cross-platform transfer, and not broad active maintenance.

Known blockers remain: FloorPlan1/seed7/Apple non-`TeleportFull` bridge `collision_stuck`, and FloorPlan201/Pencil stepwise bridge failures.
