# SDPX — 长时间自主自循环 Goal 契约

版本：2026-08-27
状态：long-running / self-looping（不由单一 wave 终止）

---

## 北极星

`time-to-verified-certificate at a specified tolerance and precision`：任意精度、多锥（LP/SOC/RSOC/SDP/Exp/Power）、原生多锥 HSD、原坐标证书、MOI 可验证、集群可复现、遗留冗余退役、文档收敛。

---

## 运行优先级（每轮按此选择"下一步动作"）

1. **推进 V4 当前 wave**：A0 → A1 → A2 → A3 → A4 → B → C，逐项实现、本地 gate、push、Oracle review、修 findings、集成。
2. **Loop 自改进（每 K 轮 retro）**：移除摩擦、硬化脚本/脚本化 gate、收紧证据、更新 `agent.md`。
3. **精度看门**：保持 Float64 与 BigFloat256 gate 全绿；Human 偶尔检查精度，agent 每 wave 输出一份简短精度摘要。
4. **冗余设计与文档简化（常驻）**：消除重复内核与双 verifier、退役死代码、收敛文档单一权威；**每项独立可回滚**。

## 自循环（一轮 = 一个可验证推进单元）

```text
选择下一动作（wave item / blocking finding / loop 摩擦 / 冗余简化项）
  ↓
独立 worktree/branch 原子实现（可回滚）
  ↓
本地 G0–G4 gate
  ↓
冻结 candidate SHA；push GitHub；断言 local == upstream
  ↓
一次 repo-wide pi-oracle Pro 审阅（context + archive）
  ↓ blocking findings → task cards
集成；merge result 重跑 required checks
  ↓
写 receipt + 证据；推进下一轮
```

关键：每轮结束后必然存在可选的"下一步"（还有 V4 项、或 open finding、或 loop 摩擦、或冗余项），因此**当仍有合法工作时，agent 不 idle、不询问、不停下**。

## 决策权威

- **Human Lead**：数学合同变更、大架构变更、merge、激活/暂停 goal。
- **sol-high Architect**：复杂数值裁决，输出 MERGE / FIX_REQUIRED / ABANDON。
- **GPT Pro（pi-oracle）**：只读对抗性 reviewer；不直接改代码。
- **机器 gate + 集群 holdout + oracle receipt**：是唯一能提升 status 的证据。

## 停止条件（避免无界循环）

- 外部：Human 喊停 / 改变 scope / 决定 release / 要求暂停。
- 阻塞：同一硬 blocker 连续三轮都无法用现有证据推进 → 报告 blocker 与所需输入，等待。
- 发布边界：所有 release gate 绿 → 产出 release candidate 报告，暂停等 Human Lead，然后继续。
- 有界迭代：每个 wave 都要到达可判定边界（集成或明确回退），不允许悬空。

## 精度（Human 只偶尔看）

- 严禁静默放松 tolerance 或删除失败测试来掩盖错误。
- 每轮在 receipt 里附精度摘要：status、原坐标证书、residual、gap、precision、iteration。
- 让 Human 随时可查：`/oracle-read <job-id>` + 集群 evidence。

## 冗余简化（常驻工作流）

目标：削减重复执行路径与两套 verifier，收敛文档单一源；**保留 `fixq3` 与冻结数学合同**。

每个简化单元必须：`grep` 无生产引用 + 替换 gate 全绿 + benchmark before/after + 独立可回滚 commit。若某一简化回退，仅回退该 commit，不影响其它已集成 wave。
