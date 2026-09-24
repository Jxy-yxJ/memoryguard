# Pre-Registration Amendment v1: Freeze All Qualifying Cases

Date: 2026-09-19
Status: approved by the user before any arm execution (no policy outcomes observed)

## Change

The original design froze the first `K = 8` qualifying cases sorted by `(scene, target, seed)`.
This amendment raises `K` to all qualifying cases (`K = 36`).

## Rationale

- The screening phase is outcome-free: it uses only spawn geometry and simulator metadata, and no
  GSAM decision, memory mutation, or task outcome was observed before this amendment.
- The rule-faithful first-8 freeze contains only FloorPlan1 (Apple seeds 7/23/37/53/67/83 and Book
  seeds 17/73) because alphabetical ordering fills `K` before FloorPlan201 and FloorPlan3.
- All 36 qualifying cases pass the same frozen geometry thresholds (`d_passive >= 2.0m`,
  `d_active <= 1.5m`, pickupable, unambiguous same-type pairing) and cover all five
  scene-target combinations: FloorPlan1 Apple (6), FloorPlan1 Book (2),
  FloorPlan201 Newspaper (10), FloorPlan201 Pencil (9), FloorPlan3 Cup (9).
- Running all qualifying cases increases paired statistical power without changing the screening
  rule, thresholds, arms, endpoints, or decision rule.

## Unchanged

- Screening thresholds and candidate universe.
- Both arm protocols, honest interaction semantics, and shared controls.
- Primary endpoint and the pre-registered decision rule.
- Falsification rules: no case substitution, no threshold relaxation, no re-screening after arm
  outcomes.

## Frozen artifact

`case_list_paired_hard_challenge_all_qualifying_v1.json` (36 cases). The original rule-faithful
8-case file (`case_list_paired_hard_challenge_frozen_v1.json`) is retained unchanged.
