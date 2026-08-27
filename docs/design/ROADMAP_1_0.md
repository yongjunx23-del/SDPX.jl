# SDPX.jl 1.0 路线图 — 成熟求解器对标缺口分析（2026-08-27）

对标对象：Clarabel（HSDE/齐次嵌入、presolve、equilibration、chordal、精化、
多 KKT 后端、MOI conformance、三态几乎解状态）、SDPA-GMP（多精度）、
SDPT3/SeDuMi（数值稳健性实践）、Hypatia（Julia 生态 conic 完备性）、
MOI.Test（conformance 文化）。

## 成熟度对照（SDPX 现状 vs 1.0 要求）

| 维度 | Clarabel 等成熟线 | SDPX 现状 | 缺口 |
|---|---|---|---|
| 正确性/证书 | 原坐标证书、双不可行分类 | ✅ 原坐标证书权威；❌ DualInfeasible 分类缺失 | P2/P4 |
| 数值鲁棒性 | equilibration、正则化+惯性+精化阶梯、恢复不终止 | ⚠️ A1 已修 NT/终态验证/家族回溯；❌ equilibration、单次降维、LDL 阶梯（P2 进行中） | P2/P3 |
| 性能/规模 | 符号/数值分离、多 RHS、分块装配、后端可选 | ⚠️ 每迭代重建 LU；threaded.jl 策略未接 native；❌ 多后端 | P5 |
| API/MOI | MOI.Test 全绿、JuMP/Convex 可用 | ❌ fail-closed；MOI.Test 未过 | P4 后 |
| 测试文化 | CI+codecov+分层（component/solver/interface） | ✅ 分层 + 失败基线 + quick gate；⚠️ 无 codecov 覆盖率门 | 低成本补 |
| 功能完备 | presolve/scaling/chordal/warm start/data update/JSON | ⚠️ presolve 雏形（soc_presolve）；❌ 其余 | 可选后置 |
| 多精度 | SDPA-GMP 级 | ✅ **领先项**：Float64/BigFloat/MultiFloat 原生 + owned-arithmetic | 保持 |
| 可维护性 | 单一求解循环 | ❌ 5 引擎冗余（P7 删除） | P7 |

## v1.0 最小集（MVP，Pro D 问题的预答）

1. **P2 完成**（进行中）：NewtonSystem + expanded quasidefinite KKT + KKT 冷启动
   ——治愈剩余 5 个已知缺口（微型退化 SOC/PSD、混合 native、Nonpositive、对偶不可行分类）
2. **P1 集成**（进行中）：Nonpositive/RSOC 变换进入生产 lowerer，支持声明=可审计契约
3. **P4**：单 HSD 状态机成为唯一数学执行路径（:expanded 默认化）
4. **P5 精简版**：fixq3 移植 + 零分配热路径 + 符号复用（不追求全部分块装配优化）
5. **MOI.Test LP/SOC/PSD 子集通过**（conformance 是社区采用的门槛）
6. **P7 第一批**：删除 lp_solver/interior_point/step（4.8k+ 行，parity 已达）
7. 分离容差（E6）+ 状态分类法（E7）——证书权威不变

## 明确后置到 1.1+（Clarabel 有但非 1.0 必须）

- chordal 分解（已有 quarantine 的 chordal.jl）
- warm start、data updating、JSON 桥
- 多稀疏 KKT 后端（QDLDL/CHOLMOD/HSL/Pardiso 等价物——先用 Julia 内置 + 本项目 la_backend）
- 覆盖率门、docs 生成（Documenter 站点）

## SDPX 独有优势（保持并作为卖点）

- 任意精度阶梯（BigFloat/Float64x2/x4/MultiFloat + owned-arithmetic MPFR 安全）
- 原坐标证书权威（status promotion 硬门）
- bootstrap 社区专用基准套件（7 物理问题 + Phase 0 失败基线）

## 里程碑顺序（依赖约束）

P2(进行中) → P1 集成(进行中) → P4 迁移 → P5 精简 → P7 批次1(LP/SDP legacy)
→ MOI.Test 子集 → v1.0-rc → P6 精度阶梯 → P7 收尾(chordal 等可选) → v1.0

预计：核心 3 周 + 精简 P5 约 1.5 周 + MOI 子集 1 周 ≈ **6 周内 v1.0-MVP**，
成熟求解器全量目标仍按 Pro 的 6-8 周估计。