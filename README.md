**English** | [简体中文](README.zh-CN.md)

# MemoryGuard

**Bounded active memory maintenance for long-horizon embodied agents.**

[![tests](https://github.com/Jxy-yxJ/MemoryGuard/actions/workflows/tests.yml/badge.svg)](https://github.com/Jxy-yxJ/MemoryGuard/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

*Xinyu Jiang ([@Jxy-yxJ](https://github.com/Jxy-yxJ))*


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

## How it works

MemoryGuard keeps an object memory of `(class, pose)` records and runs a bounded
**verify → update → act** loop over it.

**1. Score — which memories are worth verifying.**
Before any observation, each memory receives an expected-verification-value score from cheap
signals: whether the target was visible on the last revisit, its distance from the remembered
pose, and a per-class visual-confusion indicator (for visually confusable classes such as Book,
Newspaper and Pencil). Under a verification budget `B`, the top-`B` memories are selected. The
score is deliberately simple and auditable so that the mechanism evidence is not confounded by a
learned ranker; the loop is signal-agnostic and any `P(stale)` estimator — including a VLM-based
one — can replace it.

**2. Verify — perception-backed staleness detection.**
A stepwise controller navigates to the remembered pose (no `TeleportFull` on measured paths) and
captures the live RGB frame. An open-vocabulary detector — Grounding DINO + SAM 2 ("GSAM") —
decides whether the remembered object is still there and, if stale, returns a detection-grounded
candidate location. Oracle simulator metadata is used **only** for offline labels and case
construction, never as a policy input.

**3. Update — detector-backed memory refresh.**
When the verifier marks a memory stale, the record is refreshed in place from the detector
evidence. Each update stores the old position, the new position, the verifier confidence, and a
machine-readable update-source string, so every memory mutation is traceable.

**4. Act — guarded, honest task execution.**
The agent navigates to the refreshed location and attempts the downstream action
(`PickupObject`). Interaction is *honest*: `forceAction=False` so the simulator enforces proximity
and visibility, a yaw-corrected face-then-pick step, a camera-horizon sweep (down 60°, up 30°),
and a bounded approach fallback over up to three nearby reachable poses when the target is not
visible from the first pose.

**Auditability.** Every row records whether the verifier was used, the stale decision, whether
memory was mutated, the measured revisit/task paths, `TeleportFull`/`TeleportObject` flags, and a
failure reason, so claims can be traced back to individual decisions.

### Evaluation protocol

Claims are tested with a **paired** design rather than aggregate success rates. Cases are
pre-registered and passed through an outcome-free geometry screen (the stale remembered location
is at least 2 m from the true object and the refreshed location within 1.5 m, the target is
pickupable, and same-type pairing is unambiguous). Active and passive arms then run the **same
frozen cases**, the same scene/seed/spawn, the same stepwise navigation, and the same honest
interaction; the passive arm acts from the stale location with no verifier and no memory mutation.
We report exact counts, McNemar exact tests, and Clopper-Pearson intervals, and replicate on
held-out scenes, target classes, and seeds.

---

## Contributions

- **Problem framing.** Treats stale object memory as a *bounded active-maintenance* problem —
  deciding which memories to verify under a limited sensing and navigation budget — instead of
  passive replay detection or oracle-metadata verification.
- **A detector-agnostic verify–update–act loop** with a bounded budget and row-level
  auditability, driven by a deliberately simple pre-verification ranking signal that any learned
  `P(stale)` estimator can replace.
- **An honest-interaction paired evaluation protocol.** Pre-registered, geometry-screened, paired
  active-vs-passive challenges with frozen case lists, exact statistics, and held-out replication —
  including the diagnosis and fix of the facing-angle failure that the first honest run exposed.
- **Empirical findings.** (i) Perception-backed active maintenance discriminates from stale
  passive execution (29/32 vs 0/34; replicated at 11/11 and 22/24 on unseen cases); (ii) budgeted
  ranking cuts selected strict detector false negatives from 6 to 1 on the FN-prone protocol;
  (iii) strong vision-language models detect staleness (6/6) but fail maintenance (3/6 correct
  updates, 0/6 full chains) — detection is not maintenance; (iv) same-type instance confusion is
  the dominant verifier error mode (8/8 false negatives) and is live-run sensitive.
- **Open artifacts.** Pre-registrations, frozen case lists, per-row run outputs, analysis code and
  statistics, and demo videos are all included in this repository.

---

## Demo

![MemoryGuard vs passive baseline: side-by-side](videos/demo_comparison.gif)

*Left: a passive agent acting on stale memory arrives at the remembered location (red X) and finds
nothing. Right: MemoryGuard verifies the memory, refreshes it, navigates to the object's new
location (green circle), and picks it up. Rendered from real simulator frames; full-resolution
clips are in [`videos/`](videos/).*

---

## Highlights

| Result | Setting | Numbers |
|---|---|---|
| **Paired passive-vs-active challenge** | Pre-registered, geometry-screened AI2-THOR challenge (36 frozen cases / 32 evaluable pairs; corrected interaction protocol) | Active verify–update–act **29/32 (91%)** vs. live passive stale-memory **0/34**; McNemar exact two-sided **p = 3.7e-09** |
| **Held-out replication** | Unseen scenes, target classes, and seeds (11-case pilot and 24-case v2) | Active **11/11** vs. passive **0/11** (p = 9.8e-04) and active **22/24** vs. passive **0/24** (p = 4.8e-07) |
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

- **Active arm:** 29 successes / 32 evaluable pairs (91%)
- **Live passive arm:** 0 successes / 34 rows (honest interaction, `forceAction=False`)
- **Paired exact test:** McNemar two-sided **p = 3.7e-09**; 95% Clopper-Pearson CI for the active arm [0.75, 0.98]
- 2 active rows were not evaluable (detector false negatives) and 2 frozen cases were lost to a
  simulator crash; both are reported, not excluded.

**Protocol correction.** An initial honest-interaction sweep faced targets with an incorrect yaw
formula and under-reported active success (22/34). The corrected protocol fixes the facing angle,
extends the camera-horizon sweep, and adds a bounded three-pose approach fallback for the active
arm; the archived pre-fix numbers are kept only as a development reference.

Artifacts: `results/0514_corrected_interaction_v1/`, `results/0514_paired_hard_challenge_screen_v1/`.

### 2. Held-out replication on unseen scenes, targets, and seeds

Two held-out challenges reuse the frozen protocol but unseen scenes (no FloorPlan1/3/201), unseen
target classes (no Apple/Book/Cup/Newspaper/Pencil), and unseen seeds. The pilot froze 11 cases;
the larger v2 froze 24 cases across six unseen scenes and eight unseen scene-target combinations,
using a disclosed richer before-state probe.

- **Held-out pilot:** active **11/11** vs. passive **0/11** (McNemar exact p = 9.8e-04)
- **Held-out v2:** active **22/24 (92%)** vs. passive **0/24** (McNemar exact p = 4.8e-07;
  95% CI for the active arm [0.73, 0.99])
- The 2 remaining active failures are occluded targets that stay invisible even after the approach
  fallback; they are reported, not excluded.

Artifacts: `results/0514_corrected_interaction_v1/holdout_v1_active/`,
`results/0514_corrected_interaction_v1/holdout_v2_active/`.

### 3. A live, controller-backed closed-loop detector

In a 30-row sweep the only policy-side staleness detector is a Grounded-SAM2 verifier running
after a live `InitialRandomSpawn`; oracle metadata is used **only** for offline labels. Detector
agreement with offline labels is **22/30 (0.7333)** with non-uniform expected verification values,
and 20/30 stale rows trigger a within-session memory refresh.

Artifacts: `results/ai2thor_live_gsam_closed_loop_post_05822d3/`,
`results/ai2thor_live_gsam_closed_loop_compare/`.

### 4. The full verify–update–act loop without navigation shortcuts

On a frozen 6-case mixed challenge, an adaptive, route-length-aware action budget converts a
stepwise (non-`TeleportFull`) loop from 2/6 to **6/6** downstream `PickupObject` successes, with
memory refreshed from detector evidence before acting.

Artifacts: `results/ai2thor_live_gsam_mixed_challenge_budgetfix_v1/`,
`results/ai2thor_live_gsam_complete_closed_loop_seed29_apple_v1/`.

### 5. Detection is not maintenance (VLM baseline)

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
  ai2thor_*.py                #   probes, paired-challenge screening, live GSAM loops, paired control
  *_verifier.py               #   oracle / MLP / CLIP / Grounded-SAM2 staleness verifiers
  maintenance / stress        #   proactive-maintenance policies and controlled stress tests
tests/                        # unit tests (verifier schemas, budgets, honest interaction, screening)
scripts/                      # batch run / analysis / demo helpers
  make_demo_video.py           #   render captioned two-panel demo videos
  analyze_paired_hard_challenge.py  # paired statistics, exact tests, Clopper-Pearson intervals
figures/                      # paper-ready figures (PNG)
videos/                       # demo videos (mp4) and the README GIF
results/                      # curated evidence artifacts (JSON/CSV/MD + selected frames)
run_b3_remaining_seeds.sh     # batch reference for the seed sweep
```

## Reproducing

The AI2-THOR experiments use a dedicated conda environment.

```bash
conda create -n memoryguard-ai2thor python=3.11 -y
conda activate memoryguard-ai2thor
pip install -r requirements.txt     # plus Grounding DINO and SAM 2 from their upstream repositories

# unit tests
python -m unittest discover -s tests

# live AI2-THOR closed-loop detector sweep (needs a GL-capable display, e.g. Xvfb)
conda run -n memoryguard-ai2thor python -m embodied_memory_pilot.ai2thor_live_gsam_closed_loop \
    --out-dir results/ai2thor_live_gsam_closed_loop

# paired passive-vs-active challenge: screen, run both arms, analyze
conda run -n memoryguard-ai2thor python -m embodied_memory_pilot.ai2thor_paired_hard_challenge_screen \
    --scene-targets FloorPlan2:Egg FloorPlan5:Bread --seeds 101 103 107 --k 4 \
    --freeze-mode round_robin --probe-mode rich --out-dir results/screen

conda run -n memoryguard-ai2thor python -m embodied_memory_pilot.ai2thor_live_gsam_closed_loop \
    --scenes FloorPlan2 FloorPlan5 --seeds 101 103 107 \
    --case-list results/screen/case_list_paired_hard_challenge_frozen_v1.json \
    --verification-budget 4 --revisit-mode stepwise --execute-task-bridge \
    --honest-interaction --max-alternate-poses 3 --rich-before-probe --out-dir results/active

conda run -n memoryguard-ai2thor python -m embodied_memory_pilot.ai2thor_paired_task_bridge_control \
    results/active/live_gsam_closed_loop.json --live-passive --honest-interaction --out-dir results/paired

python scripts/analyze_paired_hard_challenge.py --active results/active/live_gsam_closed_loop.json \
    --paired results/paired/paired_task_bridge_control.json \
    --case-list results/screen/case_list_paired_hard_challenge_frozen_v1.json --out-dir results/analysis

# render a demo video for a recorded case
python scripts/make_demo_video.py --row-source results/active/live_gsam_closed_loop.json \
    --case FloorPlan2:Potato:101 --mode active --out-dir videos
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

## Citation

If you use MemoryGuard in your research, please cite this repository:

```bibtex
@misc{jiang2026memoryguard,
  title        = {MemoryGuard: Bounded Active Memory Maintenance for Long-Horizon Embodied Agents},
  author       = {Jiang, Xinyu},
  year         = {2026},
  howpublished = {\url{https://github.com/Jxy-yxJ/MemoryGuard}},
  note         = {Open-source research software}
}
```

## Author

**Xinyu Jiang** ([@Jxy-yxJ](https://github.com/Jxy-yxJ))
