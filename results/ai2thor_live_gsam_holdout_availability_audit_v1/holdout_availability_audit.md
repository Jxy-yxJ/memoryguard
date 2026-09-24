# Holdout Availability Audit

**Verdict**: `no_eligible_unseen_evidence_grounded_holdout_cases; stop_before_new_run`.

本审计在 budget-fix 的固定 6-case mixed challenge 达到 6/6 后执行，目的不是继续挑选正例，而是检查是否还能冻结一个不 cherry-pick 的四例 holdout。

## Frozen Rule

- 候选宇宙只来自本地已有 `results/ai2thor_live_gsam_*/live_gsam_closed_loop.json`。
- 排除已用于 selected positives、two-case reproduction、mixed challenge、waypoint rerun、budgetfix rerun 的 6 个 case。
- eligible case 必须是 pickupable，且已有 detector-stale 与 detector-backed memory mutation 证据。
- 若 eligible unseen candidates 少于 4 个，则停止，不替换成手工挑选 case。

## Result

该规则得到 `eligible_unseen_candidate_count=0`，因此没有执行新的 holdout run。

## Boundary

这是 holdout availability audit，不是新实验成功证据。它支持“停止并打包当前 fixed-challenge budgetfix 证据”，不支持 broad scale、statistical robustness、unseen scene/seed/target generalization、ObjectNav/SPL、manipulation benchmark 或 broad active maintenance。
