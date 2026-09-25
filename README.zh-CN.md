[English](README.md) | **简体中文**

# MemoryGuard

**面向长程具身智能体的有界主动记忆维护。**

MemoryGuard 研究长程具身记忆中的一个具体失效模式：环境变化后，记忆中某个物体的位置会悄悄变*陈旧*，
下游任务因为按过时记忆行动而失败。与其存入记忆后指望它一直有效，MemoryGuard 把记忆视为必须在**有界
预算下主动维护**的对象：在行动之前，它先判断某条记忆是否值得核验，用接地的感知/检测信号去核验，在陈
旧时刷新该记忆，然后才执行下游动作。

整体是一个简单的三阶段机制：

```
        核验              刷新              行动
记忆 ────────► 检测器 ────────► 更新 ────────► 下游任务
   ▲            (仅当             陈旧          (抓取 / 打开)
   │         期望价值 >                          
   └─────────── 预算) ◄────────── 有界重访预算
```

核验阶段是**检测器无关**的：任何陈旧信号都可接入（回放中的 oracle 元数据代理、实时运行中的
Grounded-SAM2 检测器，或视觉语言模型）。一个关键发现是：可靠的陈旧*检测*不等于可靠的记忆*维护*——
一个很强的视觉语言模型在 6/6 个案例上都能检出陈旧，却只在 3/6 上给出正确的更新决策。

---

## 核心结果

| 结果 | 设置 | 数值 |
|---|---|---|
| **配对式 被动 vs 主动 挑战** | 预注册、几何筛选的 AI2-THOR 挑战（冻结 36 个案例 / 34 个可评估配对） | 主动 verify–update–act **22/34**，实时被动陈旧记忆 **0/36**；McNemar 精确双侧 **p = 4.768e-07**；预注册判定 `discriminative_support` |
| **实时闭环检测器** | 30 条控制器真实运行的行，使用实时 `InitialRandomSpawn`，不使用 `TeleportObject` | 与离线标签一致 **22/30**；20/30 条记忆被刷新；期望核验价值非均匀（均值 0.7287） |
| **逐步任务环（无传送捷径）** | 冻结的 6 案例混合挑战 | 自适应的、随路径长度缩放的行动预算，把下游 `PickupObject` 成功率从 **2/6 提升到 6/6** |
| **VLM 诊断基线** | Qwen3-VL-32B 基于原始前后帧 | 陈旧检测 **6/6**，正确更新 **3/6**；多轮智能体完成完整链路 **0/6** |

以上均为**固定挑战的机制性**结果，并附明确边界（见[适用范围与局限](#适用范围与局限)）；它们不是
大规模、ObjectNav/SPL 或操作基准层面的结论。

---

## 结果详解

### 1. 在预注册配对挑战上，主动维护优于被动保留

每个挑战案例都是几何筛选后的 AI2-THOR 重排配对：*被动* 智能体按陈旧记忆行动，会被迫走向一个很远
的错误位置（`d_passive >= 2.0m`）；而 *主动* 智能体通过核验与刷新，可以到达附近的正确位置
（`d_active <= 1.5m`）。全部 36 个合格案例在两个分支运行前就已冻结。

- **主动分支：** 34 个可评估配对中成功 22 个
- **实时被动分支：** 36 行中成功 0 个（诚实交互，`forceAction=False`）
- **配对精确检验：** McNemar 双侧 **p = 4.768e-07**
- 有 2 条主动行不可评估（检测器漏检），予以如实报告而非剔除。

证据文件：`results/0514_paired_hard_challenge_v1/`、`results/0514_paired_hard_challenge_screen_v1/`。

### 2. 实时、控制器驱动的闭环检测器

在 30 行的扫描中，策略侧唯一的陈旧检测器是运行在实时 `InitialRandomSpawn` 之后的 Grounded-SAM2 核验
器；oracle 元数据**仅**用于离线标签。检测器与离线标签的一致率为 **22/30 (0.7333)**，期望核验价值
非均匀，并且 20/30 条陈旧行触发了会话内的记忆刷新。

证据文件：`results/ai2thor_live_gsam_closed_loop_post_05822d3/`、
`results/ai2thor_live_gsam_closed_loop_compare/`。

### 3. 不依赖导航捷径的完整 verify–update–act 环

在冻结的 6 案例混合挑战上，一个自适应的、随路径长度缩放的行动预算，把逐步（不使用 `TeleportFull`）
的闭环从 2/6 提升到 **6/6** 次下游 `PickupObject` 成功，且行动前已用检测器证据刷新记忆。

证据文件：`results/ai2thor_live_gsam_mixed_challenge_budgetfix_v1/`、
`results/ai2thor_live_gsam_complete_closed_loop_seed29_apple_v1/`。

### 4. 检测不等于维护（VLM 基线）

一个视觉语言模型（Qwen3-VL-32B）仅凭原始帧就在全部 6 个混合挑战案例上正确判断了陈旧性，但在 3/6
上选错了更新方向——小物体/被遮挡物体因为从变化后的视角不可见而被判为“保留”。一个多轮 VLM 智能体
完成完整链路的比例为 0/6。这把*检测*与*维护*区分开来，也正是 verify–update–act 环的动机。

证据文件：`results/vlm_agent_multi_round_v1/`。

### 图表

| 预算 / 核验价值曲线 | 组件消融 | 混合挑战进展 |
|---|---|---|
| ![预算曲线](figures/fig_0514_budget_curve.png) | ![组件消融](figures/fig_0514_component_ablation.png) | ![混合挑战](figures/fig_0514_mixed_challenge.png) |

---

## 仓库结构

```
embodied_memory_pilot/        # 核心库：基准、核验器、实时闭环运行器
  ai2thor_*.py                #   AI2-THOR 探测、重排基准、实时 GSAM 环
  *_verifier.py               #   oracle / MLP / CLIP / Grounded-SAM2 陈旧核验器
  maintenance / stress        #   主动维护策略与受控压力测试
tests/                        # 单元测试（核验器 schema、预算、失败记账）
scripts/                      # 批量运行 / 分析辅助脚本
figures/                      # 论文级图表（PNG）
results/                      # 精选证据产物（JSON/CSV/MD + 部分帧图）
run_b3_remaining_seeds.sh     # 种子扫描的批量运行参考
```

## 复现

AI2-THOR 相关实验使用独立的 conda 环境。

```bash
conda create -n memoryguard-ai2thor python=3.11 -y
conda activate memoryguard-ai2thor
pip install ai2thor grounded-sam2   # 以及纯 CPU 运行所需的项目依赖

# 单元测试
python -m unittest discover -s tests

# 实时 AI2-THOR 闭环检测器扫描（需要支持 GL 的显示，例如 Xvfb）
conda run -n memoryguard-ai2thor python -m embodied_memory_pilot.ai2thor_live_gsam_closed_loop \
    --out-dir results/ai2thor_live_gsam_closed_loop

# 对已记录运行行做纯 CPU 分析
python -m embodied_memory_pilot.ai2thor_live_gsam_closed_loop_compare --help
```

## 适用范围与局限

本仓库刻意只给出**有界、可审计**的结论。特别地，它**不**主张：

- 超出该精确配对检验之外的大规模或统计稳健性；
- 完整导航成功率或 SPL（实时流水线部分环节使用 `TeleportFull` 作为有界重访捷径；逐步变体会被明确标注）；
- 超出本文所报固定挑战之外的操作基准性能或任务成功；
- 持久化记忆写回，或跨平台（Habitat/Gibson）迁移；
- 在冻结的几何筛选案例集之外的“主动优于被动”结论。

Oracle 仿真器元数据**仅**用于案例构造与离线评估——绝不作为策略输入、排序特征或可部署检测器。

## 作者

**Xinyu Jiang**（[@Jxy-yxJ](https://github.com/Jxy-yxJ)）
