# SDPX.jl 统一计划 — THE PLAN（唯一权威）

> 本文件是 SDPX.jl 唯一的计划权威。其他规划文档（GPT Pro 重构审阅、成熟度审计、
> 旧 ROADMAP）降级为历史参考，与本文件冲突时以本文件为准。
> **完成态定义：本计划全部完成并过发布门 = v1.0。**

最后更新：2026-08-28（合并 P1.5 + 清理审阅后）

---

## 目标

打造一个**高精度的 Julia 凸优化求解器**，主要为 numerical bootstrap 服务。
主 KPI：`time-to-verified-certificate at a specified tolerance and precision`。

**v1.0 = 本文件所有里程碑 M0–M7 完成 + 发布门全绿。**

---

## 当前进度（2026-08-28）

| 包 | 状态 | 证据 |
|---|---|---|
| A1 混合 Reals+PSD 停滞修复 | ✅ 已合并 | CFT 10.9293 / Lattice 0.693 native optimal |
| P0 失败基线 + quick gate | ✅ 已合并 | 25/25 + 5 缺口 opt-in；quick gate 9/9 |
| P1 类型变换 + 生产集成 | ✅ 已合并 | 单一层级；Nonpositive/RSOC 经变换 lowerer |
| P2 NewtonSystem + expanded KKT（opt-in）+ KKT 冷启动 | ✅ 已合并 | `kkt_route=:bordered`(默认)/`:expanded`；KKT 67/67 |
| P1.5 稳定化（Pro 审计 8 项） | ✅ 已合并 | trial μ、终态所有权、验证排序、dkappa 双候选、RSOC O(n)、射线容差、来源链 |
| 清理审阅（deslop/verbosity） | ✅ 已合并 | 栈危害、异常收窄、路由记录、审计横幅 |
| **native_hsd_optin** | **290/6** | 0.5 时代 275/21 |

---

## 里程碑

### M0 — 稳定基线 ✅（已完成）
A1 + P1.5 + 清理审阅全部落地；quick gate 9/9；来源链（source SHA + manifest SHA
进 gate 产物）；include 唯一性测试；诚实缺口基线（5 项 opt-in @test_broken）。

### M1 — 变换层收尾（Priority 1 剩余）
- 移除生产 `CanonicalBlockMap` 的 `::Any` 字段（linear/linear_adjoint/
  coordinate_map）→ 类型化（参数化或 union），`transform::Any` 一并收编
- 变换层级支持**维度变更**：`AbstractCoordinateTransform`（保维）与
  `AbstractProgramReduction`（变维，带源/目标维度与 scratch 契约）分离
- 栈级 objective-shift 组合；RSOC 经 ReconstructionStack 组合测试
- Nonpositive 可执行冒烟测试改用非零目标（增强 stationarity 证据）
- **验收**：配对/平稳性/目标/双射线不变式经全链往返；原坐标证书绿

### M2 — NewtonSystem + expanded KKT 深化（Priority 2 剩余）
- 冻结五方程语义（M1 结构已在，补齐文档级冻结 + 一致性测试）
- :expanded 路由补齐：符号静态正则化 + 动态 pivot 正则化 + 惯性检查 + 多 RHS +
  对未正则化方程的迭代精化（完整 Pro A5 阶梯）
- 事务性锥缩放更新 + 结构化块失败原因（Pro I4：统一 `try_psd_nt_scaling!`，
  收敛 throwing 与 nonthrowing 双实现的数学）
- **验收**：Phase 0 缺口在 :expanded 下再转绿（rank-1 PSD、混合 PSD、bounded
  Nonpositive）；错误惯性永不静默接受

### M3 — equilibration + 等式策略 + 最小预解（Priority 3 剩余）
- 锥保 Ruiz 平衡（自由列/正交行/每块标量/等式行独立正缩放；存图重建）
- 等式策略：保留（混合默认）/稀疏 QR/小型 RRQR——**最多一次**数值秩变换；
  禁用第二次行空间降维（默认）
- 最小可逆预解（零行/列、singleton、重复行、固定变量）
- **验收**：混合模型初始残差显著低于 identity 起点；初始锥裕度严格正

### M4 — 单 HSD 状态机迁移 + 恢复阶梯（Priority 4）
- HSD 主循环改走 NewtonSystem/KKT 层（:expanded 成熟后评估默认化）
- 恢复阶梯：验证当前点 → 验证终态试探 → 中心化恢复 → 正则化 → 路由切换 →
  精度升级 → 类型化失败
- 替换家族回溯为**进度/邻域判据**（Pro I2：回溯数不再是隐式停滞检测器）
- 分离容差（primal/dual/gap/ray/κτ/锥边距/Newton 残差）+ 状态分类法
  （StalledUnverified/InsufficientPrecision/RankAmbiguous/InvalidModel 等）
- DualInfeasible 分类（真无界模型）+ DualInfeasible 完整状态面
- **验收**：native_hsd_optin 无 SOC/PSD 家族遗留失败；quick gate <60s 恢复或
  重新校准

### M5 — 一条稀疏高精度 KKT 路由（Priority 5）
- 恰好一个稀疏直接路由：一种排序、一个符号缓存、一个数值因子接口、
  一个正则化契约、一个精化实现、一个高精度后端
- 稠密/稀疏 setup 估算路由（阈值可配置）；Q3 特化推迟到发布后
- **验收**：中等规模稀疏门在声明内存预算内；BigFloat 语义回归同过

### M6 — MOI + 发布硬化（Priority 6）
- 修复 `ListOfVariableAttributesSet` introspection bug
- 对声明支持面运行全部适用 MOI.Test；声明面 = 绿面
- 公共设置与 native 能力对齐（拒绝即显式错误）
- **验收**：声明锥族的 feasible/边界/infeasible 回归全绿

### M7 — legacy 退役（Priority 7）
- 删除 lp_solver/lp_sparse/solver/interior_point/step/nonnegative_hsd/soc_native
  生产部分与旧 KKT/Schur 重复实现
- 门：一个 HSD 循环、一份 Newton 方程、一个证书权威、无隐藏 fallback、
  参考内核移入 test/reference/、quick gate 保持

### 1.x — 明确推迟（非 1.0）
自动精度升级、chordal、warm start/data update、fixq3 集成、多稀疏后端、
Gondzio 多校正器、分离步长、GPU/分布式、QP 目标、增量 MOI 编辑、
面归约/2×2 PSD→RSOC 预解

---

## v1.0 发布门

- [ ] 唯一可达 HSD 求解循环 + 一份 Newton 方程实现
- [ ] 每个声明锥族：feasible + 近边界 optimal 回归；适用族 primal/dual
      infeasible 回归
- [ ] 声明支持面内无 @test_broken 残留
- [ ] 每个成功状态经原坐标证书
- [ ] 无容差放松；quick gate 绿（预算重新校准后仍 <60s 量级）
- [ ] 一条稀疏中等规模门在内存预算内
- [ ] 声明面全部适用 MOI.Test 通过
- [ ] Float64 与选定高精度算术同语义回归
- [ ] legacy 引擎不编译、不可达、无隐藏 fallback
- [ ] 非 v1 特性显式列为 non-goals

---

## 并行策略（Harness 五角色，见仓库 agent.md）

- Explore（商品模型）→ Planner（前沿）→ Worker（商品/数值 sol）→
  Critic（机器证据→sol→council→Pro）→ Promoter（脚本）
- M1/M2 文件域部分重叠，串行；M3 与 M5 可并行；M6 依赖 M4；
  M7 依赖全部 parity
- 每个 milestone：独立分支 + per-fix commit + quick gate + 唤醒验证

## 历史参考（冲突以本文件为准）

- `docs/reviews/GPTPRO_KERNEL_RESTRUCTURE_REVIEW_20260827.md`（8 阶段原始计划，
  标记 HISTORICAL 前缀处见合并时间线）
- `docs/reviews/GPTPRO_MATURITY_AUDIT_20260827.md`（6.0/10 审计，B/I 发现）
- `docs/design/FIXQ3_PORT_SPEC.md`（Q3 移植 spec，1.x 执行）
- `docs/design/PHASE_B_RETIREMENT.md`（wave/v4-b-retire 分支，遗留退役就绪度）