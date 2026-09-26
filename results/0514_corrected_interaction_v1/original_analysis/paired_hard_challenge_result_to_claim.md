# Paired Hard Challenge Result-to-Claim

Date: 2026-09-25

Frozen cases: 36 | Evaluable pairs: 32

## Primary endpoint (paired downstream PickupObject success)

- Active success: 29
- Passive success: 0
- Both success: 0 | Both fail: 3
- Active-only success: 29 | Passive-only success: 0
- McNemar exact two-sided p: 3.725e-09
- 95% CI (Clopper-Pearson): active (0.7498, 0.9802) evaluable, (0.6894, 0.9505) over all frozen rows; passive (0.0, 0.1028)
- Active arm not evaluated (detector false negatives): 2 ['5592b48f', '830d0b7a']

## Pre-registered decision: **discriminative_support**

Support threshold: passive success <= 4; non-discriminative threshold: passive success >= 16.

## Boundary

fixed-challenge paired mechanism evidence on a pre-registered geometry-screened case set; no broad scale, no statistical robustness beyond the exact paired test, no ObjectNav/SPL, no manipulation benchmark, no persistent writeback, no cross-platform transfer
