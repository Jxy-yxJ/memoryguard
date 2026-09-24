# Stale Memory Verification Case

- Source probe: `results/ai2thor_rearrangement_6scene_seed31/ai2thor_rearrangement_probe.json`
- Seed: `31`
- Scene: `FloorPlan3`
- Target: `SaltShaker`
- Selected before-state memory: `FloorPlan3:SaltShaker|+00.33|+01.31|-02.83@before`
- True after-state target: `FloorPlan3:SaltShaker|+00.21|+01.32|-02.68@after`
- Verification risk score: `0.7292` at threshold `0.2`
- Salience-only outcome: `stale_error`
- Thresholded verification outcome: `stale_caught_then_scene_search_recovery`
- Costs: stale action `1.5`, direct after action `1.5`, fallback scene search `13.35`, verification `0.5`

This is a replay/oracle-style verification case over saved before/after AI2-THOR metadata, not a live closed-loop sensing trace.
