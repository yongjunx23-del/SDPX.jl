# SDPX 科学内核路线图（R0–R6，精简版）

更新：2026-09-09。本文是唯一计划入口，接续并取代旧审计与历史交接文档。目标：以 Julia 为主体提升 MultiFloat/BigFloat 与大规模多核能力；优化对象是**获得原坐标认证解的总时间与内存**。推送、合并主分支和发布仍不在授权内。

## 当前状态（2026-09-09）

- 开发线：SDPX 0.6.1，分支 `development/scientific-core-20260907`。
- **R2 实现已提交**（`a274011`）：会话级 symbolic 复用（Cold100=1 / Warm100=0 门通过）。**窄范围测试已提交，但 R2 完整关闭尚未成立**：R2-C/D 的并发与资源边界仍为单线程/顺序窄测试，未建立多线程资格、retained-live 对象上界、RSS 或完整阶段记账。证据 `docs/evidence/R2_FULL_QUALIFICATION.md`（其中已标明窄范围）。
- **R1 实现与窄测试已提交**（`7920982`）：AccuracyContract、owned-result 隔离、原坐标证书真实性、类型化失败符号。**窄范围测试不等于 R1 完整关闭**：R1-B BigFloat 256/512 需在加载 BFLA provider 后显式执行（oracle 发现默认 `Outputs` 未保留 primal 的缺陷已修复）；R1-C provider 闭包与 R1-D 完整矩阵仍待补。`test/test_r1_full_qualification.jl`。
- **R0**：T0 跨求解器诊断完成（问题非 ill-defined，Float64 Power 缺陷已定位）；R0-P4 公开 opt-in 路由、R0-E 补偿 Exp 研究、R0-S PSD 相对谱实验均已集成；默认 Float64 Power/Exp 仍为 known-issue。
- **R3**：实验稀疏 core 已集成（LP-only、UNADMITTED）；内存准入仍 unavailable（完整峰值上界未证）。
- 完整 R0–R3 资格验收进行中（worker 子代理正在跑全量回归与分区套件）。

## 顺序与出口条件

| 顺序 | 工作包 | 出口条件 |
| --- | --- | --- |
| 1 | T0 跨求解器诊断 | 已完成首轮 |
| 2 | R0-P/R0-E/R0-S | 真实 accepted-step 与原坐标证书；默认 Float64 Power/Exp 仍失败 |
| 并行 | R1/R2 | 实现+窄测试已提交；完整关闭仍待补（见下） |
| 3 | R3 稀疏多精度 KKT | 原始 KKT 方向合格；完整峰值上界成立前内存准入继续拒绝 |
| 4 | R4 结构化核 | 核正确性、provider 兼容、代表结构族、总认证时间可复现 |
| 5 | R5 自适应精度与单节点并行 | 精度升级/拒绝正确、串并行同数值资格、真实资源归因可信 |
| 6 | R6 生产资格 | 只认证 R0–R5 已过门子集；公开 known failures；困难问题允许 Unknown |

## 已完成的增量（压缩台账，详见 docs/evidence/）

- **R0-P4 factor-pair 公开 opt-in**（`0ebf053`）：E2E `:optimal`、certificate.valid、obj err 1.48e-9；typed 拒绝、64 MiB 上限、6825/6825。
- **R0-E 补偿 Exp**（`1308d5e`+修复）：177 断言、6/6 冻结记录；配对谓词区间界修复（`e2322de` 91/91）。
- **R2 session lease 基础**（`0f58d83`–`d4438c2`）：881 断言，oracle 复核通过。
- **R3 私有 sparse pattern**（`b005c66`、`da0c244`、`10ff7d1`）：LP-only 研究资格；内存准入 unavailable。
- **R0 半参数根/仿射 epoch/高阶修正**（`035a28c` 等）：断开实验资格，非生产。
- **R1-A AccuracyContract + R2-A 真实 symbolic 计数**：已集成。
- **T0 首轮**（`6a1333c`）：MOSEK/Clarabel/SDPX BigFloat 通过；Float64 Power 缺陷定位为 scaling 构造失败。

## 架构选择与不可越过的门（保持不变）

1. MultiFloats 表示/基础算术；MFA 研究来源；MFLA/BFLA 线代 provider；SDPX 数学/结构/资源/认证。不默认增加第三套公开线代 provider。
2. 不放宽解/方向/metric 检查，不静默改精度/舍入，不用模型名分支，不把未认证 rank reduction 当优化。
3. 数值不稳的检查须推导等价稳定计算并独立审阅；不删除负例依赖的检查。
4. 同一算术网络 SIMD/尾部可逐位一致；不同 FMA 网络须明示语义并验证最坏误差。
5. 工作树/测试环境/可变 workspace 独立所有者；每次执行记录实际源码路径/HEAD/Manifest/扩展；子任务只改自己的环境副本。

## R0 · 数学契约、稳定表示与反例闭环

**保留范围**：独立 primal/dual/HSD 推导、强弱不可行、Exp 自协调性、SOC pairing、所有 RHS/scalar closure/coupled/fused/trial/refinement/ray 消费者。

**下一步**：构造不依赖 capture 的原生 pair/trial epoch → line search → 生产策略；联合处理根残差、可靠区间、停滞与有限预算；覆盖 Float64/BF256/BF512/x4 冷种子与近边界点。已确认反例全部纳入永久测试。

**验收**：独立五方程与原坐标证书一致；公开 LP/SOC/RSOC/PSD/Exp/Power 及 infeasible/unbounded 控制通过；旧算法与 standard-v1 明确区分。

### R0 实施包
- **R0-P2** 原生 pair/trial 构造器已集成（`0602a27`）。
- **R0-P3** 完整 accepted-step 实验 loop 已闭环（Float64 冷启动→terminal 27 步，obj err 1.4e-9）。
- **R0-P4** 有限 runtime 迁移 + 公开 opt-in 路由（`0ebf053`）。
- **R0-E** Exp 冻结+定位+precision ladder 已闭环；repair 路线：factor-pair 模式适配 Exp psi-scale。
- **R0-S** PSD SPD-relative 谱实验已实现（`experimental_relative2`，仅 Float64/n=2）。
- **R0-Q** 总验收：四种精度跑正常/边界/不可行/无界/非 Slater 控制，逐原坐标独立验证。

## 版本闭包与禁止集成的候选（2026-09-09）

**锁定版本**：SDPX `7ffcc416`、MFLA `50e6e0b`、BFLA `f95d3e6`、MultiFloats 3.2.6、[MultiFloatArithmetic](https://github.com/yongjunx23-del/MultiFloatArithmetic.jl) `d2bbbd8`；Julia 1.12.6、MutableArithmetics 1.8.0。MFA 是待评估材料，尚非 solver 执行依赖。

**三个候选的处置必须保留（禁止集成）**：
- PSD `5999e8f`：删除独立逆向检查，存在约 0.707 的逆向误差反例，禁止集成。
- Power `8a51043`：虽改善 Float64，BF256 与 x4 冷启动种子出现迭代耗尽，禁止按该单行变更集成；x4 完整公开求解并未完成。
- Exp `8035ecc`：在指定点改善中心比值计算，但缺永久回归、注释/范围边界仍需修订，mixed Exp 未修好。固定已发布坐标的配对失败，不构成所有同精度算法不可行的证明。

旧 CSDR 101 次迭代与 SHA `25ef57d499cb9fdaa45600bd11c7e6948df23ab063434eff126765545e529ca7` 只属于旧算法资格；新 standard-v1 必须另立基线。完整 N14、SDPB 同精度比较和新算法生产资格均未完成。

## R1 · 所有权、AccuracyContract 与算术有效域

**状态：实现与窄测试已提交（`7920982`），完整关闭未成立。** 已具备：AccuracyContract 字段对运行时核验、owned-result 隔离（Float64；BigFloat 256/512 在加载 BFLA provider 后执行）、原坐标证书摘要、类型化失败符号与 infeasibility 射线校验。

**仍缺（不得据窄测试宣称关闭）**：R1-B 覆盖 BigFloat256/512/1024 的初始化/copy/view/共享槽/重复 solve/失败恢复/外层任务并发，逐实际 backing 存储检查，并验证 precision/rounding 改变不复用旧因子；R1-C 补齐 Julia 1.10/1.11/1.12 与 LinearSolve/QDLDL 扩展缺口（不得把一个 pin/ABI 的结论迁移）；R1-D 完整矩阵含极小 tau、scaled rays、source/result mutation 与 Unknown 返回。

## R2 · 单一 Prepared Native 与资源生命周期

**状态：实现与窄测试已提交（`a274011`），完整关闭未成立。** 已具备：会话级 symbolic 复用真实计数门（Warm100=0 / Cold100=1）、结构失效事务、checkout-before-validation 租约事务、顺序会话隔离、分配波动检查。

**仍缺（不得据窄测试宣称关闭）**：多线程/任务级并发资格（当前 R2-C 为顺序会话 + 手动 busy）；retained-live 对象、actual capacity、MPFR/GMP scratch、线程 scratch 与 GC 重叠的完整上界（当前 R2-D 只测 10 次求解的分配波动）；完整阶段记账与 RSS 行为；真实 backend symbolic reuse 的规模资格。

## R3 · 稀疏多精度 KKT 与 cone-preserving scaling

**保留范围**：signed LDL/准定性、零 primal 对角、排序、正则矩阵、目标 precision、实际因子类型；逐类 cone 块存储/fill/回填映射；原系统 refinement；完整 owned-live 上界。

**下一步**：R3-A provider 数学契约 → R3-B 真实稀疏装配 → R3-C 原系统 refinement → R3-D 完整 owned-live 上界（不能证明时内存准入继续 unavailable，与仅供研究的未准入入口明确区分）。

## R4 · 结构化核（PSD 谱、trial 复用、panel/low-rank、chordal）

**保留范围**：PSD spectral provider、trial 复用、panel/low-rank Gram、真正 chordal+recovery、连续批量 Q3、采样/多项式来源结构；削减 O(k⁴) operator 存储。

**实施包**：R4-A 扩展诚实配对矩阵（小/中/大、odd tail、transpose/view、beta/对消等级）→ R4-B 每次只换一个核 → R4-C 结构先证等价（原坐标审计）→ R4-D end-to-end 归因。

## R5 · 有验证的自适应精度与单节点并行

**保留范围**：低精度因子/目标精度残差分离，precision promotion 重建状态和因子；源数据不足则重新生成，不给 rounded data 补零。

**实施包**：R5-A 精度控制器 → R5-B 单一线程预算 → R5-C 硬件矩阵（PBS 1/8/16/32 核，真实分配）→ R5-D 性能判定（同原问题/同误差目标/原坐标门）。

## R6 · Production scientific qualification

**保留范围**：支持/实验/不可用表、独立 parser/MOI 映射、取消/checkpoint、长期 held-out 对抗、可重建 release 证据。

**实施包**：R6-A 支持矩阵 → R6-B 接口与恢复 → R6-C 应用阶梯（standard-v1 → N6/新 CSDR → N14/BF512 → SDPX/SDPB 同目标 → 8/16/32 核）→ R6-D 可重建交付。

## 统一测量与第一批次

精度按实际类型与模型声明记录（MultiFloats `53N−(N−1)`；BigFloat 实测核对）。完整记账：compile/prepare/pack/预热核/factor/solve/verify；稳定预热中位数改善 ≥2% 才计 kernel speed credit；最终报告 time-to-certified-solution 与精度—时间 Pareto。N14 的 65 coefficients/9300 Q3 不足以推出约化 rank 资格；PSD k=100 下三角 Theta 为 12,753,775 标量，完整 q² 不是三角存储数。

自动执行批次：
1. 纠正证据口径与环境所有权，保留 R0 失败与反例。
2. R1 舍入/所有权判别、R2 结构缓存锁、只读结构/资源诊断（窄测试已提交；完整矩阵待补）。
3. 不改默认策略下建立 MFLA/BFLA/MFA 对照基线。
4. 按依赖推进 R3/R4/R5，保留 sparse Ruiz、chordal/recovery、MOI、checkpoint、真实模型与独立证书事项。

基础设施故障先保存工作树/环境差异与执行状态，只走明确同协议重试；数学 gate 失败先区分计算误差与实际状态错误。
