# Corrected Honest-Interaction Re-Evaluation (pre-registered)

Date: 2026-09-24
Status: design frozen before any re-run

## Motivation (diagnosis, not outcome tuning)

A simulation-level diagnosis of the 16 held-out v2 active failures and 12 original-challenge failures
showed the face-target rotation in the honest pickup used `atan2(dz, dx)` for the yaw, while AI2-THOR's
yaw convention is `atan2(dx, dz)`. In most failing cases the agent therefore faced roughly 90 degrees
away from the target and never saw it. A follow-up pose diagnostic showed several "invisible" targets
become visible from the next reachable positions (and from a corrected facing), and one shelf target is
visible only from the fourth-nearest reachable pose.

## Protocol corrections (deterministic, disclosed)

1. **Yaw fix**: face the target with `atan2(dx, dz)`.
2. **Extended horizon sweep**: down to +60 degrees and up to -30 degrees (was down 60 / up 30).
3. **Bounded approach-pose fallback** (active arm only, configurable): after a failed honest pickup,
   navigate stepwise to up to three further reachable poses nearest to the refreshed location and retry
   the honest pickup. Zero disables it; the flag defaults to zero.

## Re-evaluation protocol

- Same frozen case lists as before: the original challenge list (36 cases) and the held-out v2 list
  (24 cases). No case substitution, no threshold changes, no re-screening.
- Both arms re-run end-to-end with the corrected code for protocol consistency (the yaw and sweep
  corrections affect the passive arm too; the pose fallback applies only to the active arm because the
  passive arm has no refreshed location and does not search).
- Report old versus new results on identical cases, exact paired statistics, CIs, and failure
  decomposition.
- Boundary: this is a corrected-protocol re-evaluation; it does not add new case universes and does not
  change any claim boundary. The corrected numbers supersede the buggy-facing numbers for the honest
  interaction protocol; the buggy-protocol numbers remain archived.
