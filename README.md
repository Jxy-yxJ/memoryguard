**English** | [简体中文](README.zh-CN.md)

# MemoryGuard

**Bounded active memory maintenance for long-horizon embodied agents.**

MemoryGuard studies a concrete failure mode of long-horizon embodied memory: a remembered
object location silently becomes *stale* after the environment changes, and a downstream task
then fails because it acts on the outdated memory. Instead of storing memory and hoping it stays
valid, MemoryGuard treats memory as something that must be **actively maintained under a bounded
budget**: before acting, it decides whether a memory is worth verifying, verifies it with a
grounded perception/detector signal, refreshes the memory when it is stale, and only then executes
the downstream action.

The loop is a simple three-stage mechanism:

```
        verify            update             act
memory ────────► detector ────────► refresh ────────► downstream task
   ▲               (only if           stale           (pickup / open)
   │            expected value >                            
   └─────────────── budget) ◄────────── bounded revisit budget
```

The verification stage is **detector-agnostic**: any staleness signal (an oracle-metadata proxy
in replay, a Grounded-SAM2 detector in live runs, or a vision-language model) can be plugged in.
A key finding is that reliable staleness *detection* is not the same as reliable memory
*maintenance*: a strong vision-language model detects staleness in 6/6 cases yet makes the correct
update decision in only 3/6.

---

## Highlights

| Result | Setting | Numbers |
|---|---|---|
| **Paired passive-vs-active challenge** | Pre-registered, geometry-screened AI2-THOR challenge (36 frozen cases / 34 evaluable pairs) | Active verify–update–act **22/34** vs. live passive stale-memory **0/36**; McNemar exact two-sided **p = 4.768e-07**; pre-registered decision `discriminative_support` |
| **Live closed-loop detector** | 30 controller-backed rows with live `InitialRandomSpawn`, no `TeleportObject` | **22/30** agreement with offline labels; 20/30 memories mutated; non-uniform expected verification value (mean 0.7287) |
| **Stepwise task loop (no teleport shortcut)** | Fixed 6-case mixed challenge | Adaptive route-aware action budget lifts downstream `PickupObject` success from **2/6 → 6/6** |
| **VLM diagnostic baseline** | Qwen3-VL-32B on raw before/after frames | Staleness detection **6/6**, correct update **3/6**; multi-round agent completes **0/6** full chains |

These are **fixed-challenge mechanism** results with explicit boundaries (see
[Scope and limitations](#scope-and-limitations)); they are not broad-scale,
ObjectNav/SPL, or manipulation-benchmark claims.

---

## Results

### 1. Active maintenance beats passive retention on a pre-registered paired challenge

Each challenge case is a geometry-screened AI2-THOR rearrangement pair where a *passive* agent
acting on stale memory is forced to a far, wrong location (`d_passive >= 2.0m`) while an *active*
agent that verifies and refreshes can reach a nearby correct one (`d_active <= 1.5m`). All 36
qualifying cases were frozen before either arm ran.

- **Active arm:** 22 successes / 34 evaluable pairs
- **Live passive arm:** 0 successes / 36 rows (honest interaction, `forceAction=False`)
- **Paired exact test:** McNemar two-sided **p = 4.768e-07**
- 2 active rows were not evaluable (detector false negatives) and are reported, not excluded.

Artifacts: `results/0514_paired_hard_challenge_v1/`, `results/0514_paired_hard_challenge_screen_v1/`.

### 2. A live, controller-backed closed-loop detector

In a 30-row sweep the only policy-side staleness detector is a Grounded-SAM2 verifier running
after a live `InitialRandomSpawn`; oracle metadata is used **only** for offline labels. Detector
agreement with offline labels is **22/30 (0.7333)** with non-uniform expected verification values,
and 20/30 stale rows trigger a within-session memory refresh.

Artifacts: `results/ai2thor_live_gsam_closed_loop_post_05822d3/`,
`results/ai2thor_live_gsam_closed_loop_compare/`.

### 3. The full verify–update–act loop without navigation shortcuts

On a frozen 6-case mixed challenge, an adaptive, route-length-aware action budget converts a
stepwise (non-`TeleportFull`) loop from 2/6 to **6/6** downstream `PickupObject` successes, with
memory refreshed from detector evidence before acting.

Artifacts: `results/ai2thor_live_gsam_mixed_challenge_budgetfix_v1/`,
`results/ai2thor_live_gsam_complete_closed_loop_seed29_apple_v1/`.

### 4. Detection is not maintenance (VLM baseline)

A vision-language model (Qwen3-VL-32B) judged staleness correctly on all 6 mixed-challenge cases
from raw frames, but chose the wrong update direction on 3/6 — small/occluded objects were
declared "keep" because they were invisible from the changed viewpoint. A multi-round VLM agent
completed 0/6 full chains. This separates *detection* from *maintenance* and motivates the
verify–update–act loop.

Artifacts: `results/vlm_agent_multi_round_v1/`.

### Figures

| Budget / verification-value curve | Component ablation | Mixed-challenge progression |
|---|---|---|
| ![budget curve](figures/fig_0514_budget_curve.png) | ![component ablation](figures/fig_0514_component_ablation.png) | ![mixed challenge](figures/fig_0514_mixed_challenge.png) |

---

## Repository layout

```
embodied_memory_pilot/        # core library: benchmarks, verifiers, live closed-loop runners
  ai2thor_*.py                #   AI2-THOR probes, rearrangement benchmarks, live GSAM loops
  *_verifier.py               #   oracle / MLP / CLIP / Grounded-SAM2 staleness verifiers
  maintenance / stress        #   proactive-maintenance policies and controlled stress tests
tests/                        # unit tests (verifier schemas, budgets, failure accounting)
scripts/                      # batch run / analysis helpers
figures/                      # paper-ready figures (PNG)
results/                      # curated evidence artifacts (JSON/CSV/MD + selected frames)
run_b3_remaining_seeds.sh     # batch reference for the seed sweep
```

## Reproducing

The AI2-THOR experiments use a dedicated conda environment.

```bash
conda create -n memoryguard-ai2thor python=3.11 -y
conda activate memoryguard-ai2thor
pip install ai2thor grounded-sam2   # plus the project requirements for CPU-only runs

# unit tests
python -m unittest discover -s tests

# live AI2-THOR closed-loop detector sweep (needs a GL-capable display, e.g. Xvfb)
conda run -n memoryguard-ai2thor python -m embodied_memory_pilot.ai2thor_live_gsam_closed_loop \
    --out-dir results/ai2thor_live_gsam_closed_loop

# CPU-only analyses of recorded rows
python -m embodied_memory_pilot.ai2thor_live_gsam_closed_loop_compare --help
```

## Scope and limitations

This repository intentionally ships **bounded, auditable claims**. In particular, it does **not**
claim:

- broad scale or statistical robustness beyond the exact paired test;
- full navigation success rate or SPL (`TeleportFull` is used as a bounded revisit shortcut in
  parts of the live pipeline; stepwise variants are labeled as such);
- manipulation-benchmark performance or task success beyond the fixed challenges reported here;
- persistent memory writeback or cross-platform (Habitat/Gibson) transfer;
- active-over-passive superiority outside the frozen, geometry-screened case set.

Oracle simulator metadata is used **only** for case construction and offline evaluation — never as
a policy input, ranking feature, or deployable detector.

## Author

**Xinyu Jiang** ([@Jxy-yxJ](https://github.com/Jxy-yxJ))
