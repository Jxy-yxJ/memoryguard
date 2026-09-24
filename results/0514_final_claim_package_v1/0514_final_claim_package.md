# 0514 Final Claim Package

## 结论

0514 目标已经从“被动 stale-memory 诊断”推进到“受限预算下的闭环 active memory maintenance 机制证据”。

## 可以说的

- MemoryGuard 在 AI2-THOR controller-backed live GSAM 轨道上，能够完成 detector-backed verification/update + non-Teleport measured task execution。
- 冻结的两类证据都成立：
  - selected positives 可复现：seed29 与 seed11 的 FloorPlan1/Apple 连续通过。
  - fixed mixed challenge 的 hard cases 暴露了明确边界，之后 budgetfix 把同一 six-case challenge 从 2/6 提升到 6/6。
- holdout 审计显示：按现有本地证据规则，没有可用的未见 holdout case，因此不应为了“扩展”而临时拼 case。

## 不能说的

- 不能说 broad scale。
- 不能说 ObjectNav/SPL、manipulation benchmark、passive-vs-active superiority、persistent writeback、cross-platform transfer。
- 不能说已经解决所有 hard cases。

## 推荐写法

把 0514 的主贡献写成：

> 在受限感知与导航预算下，MemoryGuard 通过 detector-backed verification/update 和受限的 non-Teleport closed-loop task execution，实现了一个可审计的 embodied memory maintenance 机制；在冻结 mixed challenge 上，adaptive stepwise budgeting 将 downstream success 从 2/6 提升到 6/6，但该结论仅限于固定挑战集。

## 证据链

1. 0514 synthesis / gap / metric coverage：定义问题与边界。
2. single-case closed loop + preregistered two-case reproduction：证明 selected positives 可复现。
3. mixed challenge + waypoint route + budgetfix：证明 hard cases、partial mechanism fix、以及固定挑战集的 6/6。
4. holdout audit：证明当前本地证据集下不应继续 cherry-pick 新 holdout。

## 下一步

- 写论文时，把 `fixed-challenge mechanism evidence` 和 `remaining gaps` 并列写清楚。
- 若要真正升级为广泛 claim，必须先预注册更大的 case universe 或切到 Habitat 第二平台。