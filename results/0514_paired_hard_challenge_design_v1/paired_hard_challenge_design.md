# Pre-Registered Discriminative Passive-vs-Active Challenge — Design v1

Date: 2026-09-19
Status: design only, not executed
Companion spec: `paired_hard_challenge_design.json`

## Objective

Pre-register a paired passive-vs-active challenge where stale memory is expected to make the
passive downstream task attempt fail while the active verify-update-act loop can succeed, so
that active-over-passive mechanism evidence becomes evaluable.

## Why the previous paired control was not discriminative

The live passive control at
`results/0514_paired_passive_vs_active_task_bridge_live_passive_v1/` reported active 6/6 and
live passive 6/6 (`paired_delta=0`). Two protocol facts explain this:

1. **Forced interaction.** Both arms executed `PickupObject` with `forceAction=True`, which
   bypasses proximity and visibility preconditions. The passive arm therefore picked the moved
   object from the stale location even when the object was meters away.
2. **Small displacements.** Stale-to-true displacement in the fixed six cases was 0.47, 0.55,
   0.60, and 0.83 m for four rows, with only 3.05 m (FloorPlan1/Apple seed 7) and 3.51 m
   (FloorPlan201/Pencil seed 29) exceeding 2 m.

Useful positive finding for the new design: the detector-refreshed position equals the live
true object position in all six rows (detector-to-true distance 0.0), so the active arm's
refresh target is accurate under the current verifier.

## Frozen universe

| Scene | Target | Seeds |
|---|---|---|
| FloorPlan1 | Apple | 7, 11, 17, 23, 29, 37, 43, 53, 67, 73, 83, 97 |
| FloorPlan1 | Book | same |
| FloorPlan3 | Cup | same |
| FloorPlan201 | Newspaper | same |
| FloorPlan201 | Pencil | same |

60 candidates total. These five scene-target combinations have prior live spawn evidence
(`results/ai2thor_live_gsam_scale_expansion_v1/`, `results/ai2thor_live_gsam_mixed_challenge_42_case_pre_registered_v1/`).
The universe and seeds are frozen before any screening or arm execution.

## Frozen screening rule (construction only, no policy outcomes)

For each candidate:

1. `InitialRandomSpawn(randomSeed=seed, forceVisible=True, numPlacementAttempts=5)`.
2. If the target does not spawn, exclude the candidate (`not_spawned`); the rule is not relaxed.
3. `P_true` = live target position; `P_stale` = before-phase remembered position from
   `rearrangement_location(scene, target, "before")`.
4. `G_passive` = nearest reachable position to `P_stale`; `G_active` = nearest reachable
   position to `P_true`.
5. `d_passive = ||G_passive - P_true||`, `d_active = ||G_active - P_true||`.
6. Qualify iff `d_passive >= 2.0 m` **and** `d_active <= 1.5 m` **and** target pickupable.

Freeze the first `K = 8` qualifying candidates sorted by `(scene, target, seed)` into
`results/0514_paired_hard_challenge_v1/case_list_paired_hard_challenge_frozen_v1.json` and
commit it before running either arm. If fewer than 8 qualify, run all qualifying candidates and
report the shortfall; do not relax thresholds or substitute cases.

`1.5 m` is the AI2-THOR interaction range used as the geometric reference; `2.0 m` is a safety
buffer beyond it. Screening uses simulator metadata for case construction only; oracle metadata
is never a policy input.

## Arms and interaction protocol

Both arms share the scene, seed, `InitialRandomSpawn` configuration, stepwise navigation mode,
adaptive action-budget policy, and the absence of `TeleportFull`/`TeleportObject` on measured
paths.

- **Active**: verify at the stale location, GSAM stale decision, detector-backed in-memory
  refresh, stepwise task bridge to the refreshed position, then act.
- **Passive**: stepwise navigation to `memory_old_position`, no verification, no memory
  mutation, act from the stale location.

New `--honest-interaction` mode (default off, so existing artifacts stay reproducible):

- `PickupObject` with `forceAction=False`, so simulator proximity/visibility preconditions apply.
- Deterministic face-target rotation before the action (active: refreshed position; passive:
  remembered position).
- No same-type substitution: the screening rule excludes ambiguous same-type after-pairings, so
  the target instance is unique by construction. Because AI2-THOR `objectId`s encode positions and
  change after spawn, resolution falls back to the unique same-type instance near the arm's target
  position; if no matching instance exists, record `target_object_id_missing` and fail the attempt.
- Record `interaction_agent_object_distance`, `interaction_target_visible`, and the action
  `errorMessage`.

## Endpoints and pre-registered decision rule

Primary: paired downstream `PickupObject` success on the true target instance, paired by
`case_id = scene|target|seed`.

| Outcome | Rule |
|---|---|
| Discriminative support | `active_success > passive_success` and `passive_success <= 1/8`; report McNemar exact p descriptively |
| Partial | `passive_success in {2,3}/8`; exact counts and geometry only, no strong claim |
| Non-discriminative | `passive_success >= 4/8`; stop and record a negative result |

Secondary: verifier/mutation rates, stale-to-true displacement, agent-object distance at action
time, path steps/actions, TeleportFull flags, failure reasons.

## Falsification rules

- No case substitution, threshold relaxation, or re-screening after outcomes are observed.
- Active-arm failures caused by GSAM false negatives are recorded, not excluded.
- Passive successes caused by route proximity to the true object are recorded with the
  route-distance covariate, not excluded.
- Fewer than 4 qualifying cases: run them and report the shortfall.

## Implementation plan

1. `embodied_memory_pilot/ai2thor_paired_hard_challenge_screen.py` + tests — AI2-THOR-only
   screening, outputs the candidate table and the frozen case list.
2. `--honest-interaction` flag and shared face-then-pick helper in
   `embodied_memory_pilot/ai2thor_live_gsam_closed_loop.py` + tests.
3. Paired runner extension in `embodied_memory_pilot/ai2thor_paired_task_bridge_control.py` +
   tests — accepts the frozen case list, runs both arms, emits paired JSON/CSV and schema check.
4. Execution order: screen -> freeze + commit case list -> active arm -> passive arm ->
   result-to-claim artifact.

Estimated runtime: screening ~30-60 min for 60 candidates; arms ~1-2 h for 8+8 runs.

## Claim boundary

Fixed-challenge paired mechanism evidence only. No broad scale, statistical robustness, unseen
case generalization, ObjectNav/SPL, manipulation benchmark, broad passive-vs-active superiority,
persistent writeback, or cross-platform transfer claims.
