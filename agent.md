# SDPX.jl — 项目 Agent 工程手册

**项目目标**：打造一个**高精度的 Julia 凸优化求解器**，主要为 **numerical bootstrap**（共形自举数值计算）服务。

**一句话定位**：任意精度（Float64/BigFloat/MultiFloat 阶梯）、多锥（LP/SOC/RSOC/SDP/Exp/Power）、原生多锥 HSD 内核、原坐标证书权威、集群可复现——为 bootstrap 社区提供 time-to-verified-certificate 最优的求解器。

**详细契约**：
- 长时间自主循环 Goal：`.pi/GOAL.md`
- Oracle 审阅合同：`.pi/ORACLE_REVIEW_PROMPT.md`
- 当前 wave 状态：`.pi/WAVE.yaml`
- 当前 1.0 路线图：`docs/PLAN.md`
- 冻结数学合同：`docs/design/CANONICAL_FORM.md`、`HSD_FORMULATION.md`、`NEWTON_SYSTEM.md`
- 当前 GPT Pro 问题/性能审阅：`docs/reviews/GPTPRO_BUG_KERNEL_REVIEW_20260828.md`、`GPTPRO_PERFORMANCE_PLAN_20260828.md`
- 父级通用 Harness 模式：`../agent.md` §4.3

---

## 五角色弧（SDPX 实例化）

每个原子任务按五角色弧执行；角色间用隔离 prompt。分配原则：
**规划/批评用前沿模型，执行用商品模型，传播用脚本**。

| 角色 | 职责 | 本项目工具链 | 模型 |
|---|---|---|---|
| **Explore** 调研/根因 | 定位数值失败根因、环境差异、算法文献 | 本地 Julia probe（每个 agent 自建 testenv 并验证 Manifest 指向）；`scripts/run_test_file.sh`；`ucas-hpc` skill 集群复现（PBS，≤48gb/90min 免批）；`pi-web-access`/feynman MCP 查文献 | recon `luna-high`；深挖 `luna-max`；数值根因 `sol-high` |
| **Planner** 计划固化 | 产出显式 DAG：每节点带文件域、验收门、回滚边界 | Lead Agent 亲自写任务书；大设计咨询 `pi-oracle`（`scripts/build_oracle_archive.sh` 冻结 SHA + `oracle_receipt.sh`）或 `gpt56solpro-consult`；计划落 `docs/design/`、`docs/reviews/` | **前沿**：`sol-high`(max) 设计；大 wave GPT Pro（Pro effort） |
| **Worker** 逐节点实现 | 独立 worktree/分支实现 DAG 节点 | `pi-subagents` `runs.run`（worktree:true + 自建 testenv + `steer` 纠偏 + supervisor 决策）；长数值跑集群 PBS | **商品**：T1/T2 `deepseek-high`（`merge/deepseek/deepseek-v4-flash-0731`，比 luna 快一个量级；重任务 thinking=max，两轮未解直接升 `sol-high`）；数值内核 T3 `sol-high`（prewalk 省前沿用量） |
| **Critic** 质疑/返工（sol 任务超时预算 120 分钟，见父级 agent.md）| 机器证据复核，可触发返工 | 四层：① 机器证据（quick gate 9/9 + `test/kernel_failure_regressions.jl` + 目标值 + git SHA，零成本）② `sol-high` 交叉审 ③ council-mode（重大取舍）④ GPT Pro 全仓审阅（每 wave 1 次 + 1 follow-up） | Lead 机器验证（零）；`sol-high`；T4 Pro |
| **Promoter** 传播完成 | 合并主线、集群同步、状态存档 | 机械执行：merge → E2E/专项门复核 → `git archive`+SHA-256 → 集群 release（`ucas-hpc` release layout）→ `.pi/WAVE.yaml`/`docs/PLAN.md` 更新 → `check_release_metadata.jl` | 无模型（脚本优先） |

## 超时预算

- deepseek（T1/T2）：默认 30 分钟
- **sol-high（T3）：默认 120 分钟**（`timeoutMs: 7200000`）——sol 天然慢，勿用短超时
- spec/文档任务（sol）：60 分钟

## 硬门（不可协商）

1. **原坐标证书权威**：任何 status promotion 必须有原坐标证书；未通过证书的"更快"结果直接拒绝。
2. **quick gate 9/9 且 <60s**：`test/quick_gate.jl` 是每个 commit 的门；失败基线用 opt-in `@test_broken` 诚实可见，不静默删除。
3. **无容差放松、无 fallback**：fail-closed；已知失败显式记录。
4. **每个原子任务**：独立 worktree/分支 + 独立提交 + 可回滚 + 推送 SHA。
5. **Critic 独立于 Worker**：Worker 自报完成不算数。

## 当前执行序（详见 `docs/PLAN.md`）

legacy 调用迁移与原子删除 → PureKLU/QDLDL 生产接线 → fused residual、线程预算与固定尺寸 hot path → general benchmark findings → 黑盒 E2E 与专项 provider/MOI/precision 门 → 本地/集群 campaign → 最终独立审阅 → 一次性 push。