# Mixed Challenge Budget-Fix Result-to-Claim

**Verdict**: `fixed_mixed_challenge_six_of_six_after_adaptive_stepwise_budget; no_broad_scale_claim`.

这次只检验一个窄问题：reachable-waypoint route 已经存在时，固定 12 个 primitive actions 的预算是否会把合法长路径误判为 `max_steps_exceeded`。新增回归测试先红灯确认该问题，然后将 waypoint route 的动作预算按 route 长度放大并保留硬上限。

## Evidence

- 前一版 waypoint route 固定 mixed challenge：2/6 downstream success，4/6 downstream failure，主要为 `max_steps_exceeded`。
- budget fix 后同一个固定 6-case case-list：6/6 case-list matched，6/6 verifier-used，6/6 memory-updated，6/6 downstream `PickupObject` success。
- 所有 post-fix rows 的 measured revisit 和 task bridge 都记录 `TeleportFull=false`，且 `teleport_object_used=false`。

## Boundary

这是固定 mixed challenge 上的机制修复证据，不是 broad scale、ObjectNav/SPL、manipulation benchmark、passive-vs-active superiority、persistent writeback、cross-platform transfer 或 broad active maintenance。
