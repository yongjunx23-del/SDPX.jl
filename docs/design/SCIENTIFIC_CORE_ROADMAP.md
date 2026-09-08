# SDPX 科学内核与高精度体系：完整 R0–R6 路线图

更新：2026-09-08。本文接续 2026-09-07 项目审计，更新其执行路线图；旧审计、历史实验与证书保持原样。它是当前技术设计与验收计划，不是会话交接文档。

目标：以 Julia 为主体，提高 MultiFloat、BigFloat 和大规模多核计算能力；优化对象最终是**获得原坐标认证解的总时间与内存**，不是单个算术峰值。用户已授权按本计划自动实施、检查和交付；推送、合并主分支和发布仍不在授权内。

## 依据与当前边界

本轮只读体系审阅锁定：SDPX `7ffcc416`、MFLA `50e6e0b`、BFLA `f95d3e6`、MultiFloats 3.2.6、用户提供的 [MultiFloatArithmetic](https://github.com/yongjunx23-del/MultiFloatArithmetic.jl) `d2bbbd8`；Julia 1.12.6、MutableArithmetics 1.8.0。MFA 是待评估材料，尚非 solver 执行依赖。

四路源码调查经过独立 Astra 综合复核。原报告中的内存模型、算术资格、尺寸估算错误不得进入实现依据；该次只读审阅未包含新的性能实测。范围/散列与原报告保存在 `local-archives/high-precision-ecosystem-20260908/` 对应证据集合，独立综合报告为 `ecosystem-astra-synthesis.md`。

当前已建立 standard-HSD 数学契约、对数 Exp barrier 及独立 Newton 参考；完整 HSD 符号迁移获独立源码审阅通过。`7ffcc416` 的 1,252 项聚焦检查通过，**不等于完整公开路径资格**：Float64 Power、mixed Exp、紧容差 PSD 仍有失败。

三个候选的处置必须保留：
- PSD `5999e8f` 删除独立逆向检查，存在约 0.707 的逆向误差反例，禁止集成。
- Power `8a51043` 虽改善 Float64，BF256 与 x4 冷启动种子出现迭代耗尽，禁止按该单行变更集成；x4 完整公开求解并未完成。
- Exp `8035ecc` 在指定点改善中心比值计算，但缺永久回归、注释/范围边界仍需修订，mixed Exp 未修好。固定已发布坐标的配对失败，不构成所有同精度算法不可行的证明。

旧 CSDR 101 次迭代与 SHA `25ef57d499cb9fdaa45600bd11c7e6948df23ab063434eff126765545e529ca7` 只属于旧算法资格；新 standard-v1 必须另立基线。完整 N14、SDPB 同精度比较和新算法生产资格均未完成。

### 已完成的增量验收（不等于全部阶段完成）

- R1：BFLA `aaa71f3` 的有效舍入与 RRQR 创建语境元数据修复通过核心源码/证据复核。父流程独立执行：Julia 1.12 核心 10,960 项，Julia 1.10 核心 10,957 项加 1 项不适用，双线程 285 项；48 组同运行时默认舍入记录一致。Julia 1.11 与 LinearSolve 扩展仍未验证；QDLDL 仅有下述有限验证。
- R2：结构缓存代际/锁协议 `b82c7a3` 已集成，2/4 线程各 98 项通过；实际因子所有者诊断 `a85d24b` 已集成，239 项诊断加 57 项 Newton 检查通过。
- 新 BFLA 仅进入独立 SDPX 验证环境，96 项集成检查通过；原受保护环境与 provider 主分支不变。
- R3：BFLA/QDLDL 0.4.1 在 n=12、256/512/1024-bit、单线程下通过 231 项父流程复测，包含原矩阵残差、所有权和失效控制；加载来源绑定 fail-closed。另有 886 项精确秩/相容性及原始 KKT 残差参考检查通过；借用的正则策略和 12 次细化是实验规则，不等于生产门。内部 factor-cache 适配器已集成为 `a72d1cc`，父流程通过 287 项适配器、24 项 MFLA 共用 seam、4 项缺失扩展/默认路线检查；使用 checked solve，调用者仍须提供合格移位算子与原始残差权威。公开 BigFloat 稀疏路线仍关闭，不构成完整 symmetric-core、规模或性能资格。
- R4：MFLA/MFA 两个尺寸已有同类型 ABBA 多进程重复观察、全幅输入和可测的产品对消；仍无通用或 solver 速度资格。显式实验性 MFA 串行策略及 runner 已通过限定审阅，父流程在 Julia 1.10/1.12 各复测 87 项及来源/哈希/日志失败控制。另一个 MFA 连续视图增量 `cdb8468` 在 MF3.2.6 下通过 215,412 项父流程测试；不能把旧 pin 的资格自动转移给新 pin。默认关闭，远端 CI 未执行；ABBA 不排除全部位置效应。
- R5：MFLA 实验性并行策略的非单射目标存储拒绝与真实 limb-bit 检查经修复；父流程日志增量 `5399c0cc` 获独立窄审通过。Julia1.12 下 t1/t4 分别通过 218/234 项，实际公开调用回执观测到最多 4 个线程，串行标注 tasks_spawned=0；固定 MFA `cdb8468`、MF3.2.6，来源/环境前后散列一致。仅为这些显式实验调用的正确性资格，不是默认路线、solver 或大核数性能资格。
- R0：独立几何反例已永久接线。Power 区间参考通过独立审阅及 337 项父流程复测；Exp 导数/坐标参考经范围修复后通过 129 项复测与审阅。二者均未接入生产；完整 SAME-metric 缩放与冻结 epoch 迁移仍未完成，完整公开套件仍受既有 Power/mixed Exp 失败阻断。

### 受限集成与仍未过门的工作

- R3 主窗口修复 `da0c244` 已获 Astra high 窄审通过：冻结精度与舍入、精确 CSC/回填映射绑定、实验结构数组独立所有权及真正的 post-factor 失败恢复；父流程 1,053 项通过（212 原有 +841 新增），210 个来源/环境散列不变。后继 `10ff7d1` 将内存准入改为 unavailable 并在构造前拒绝；显式私有研究入口不授予内存资格，1,080 项通过包括 27 项拒绝检查，窄审通过。最终研究实现与下述桥接已整合到开发分支 `4701fac`，源码除路线图外与已审阅 `772b724` 完全一致。资格仍限 LP-only、UNADMITTED 小型研究；SOC、完整内存上界、公开路线与规模资格未完成。
- BFLA 隔离后继 `5fce2e6` 新增显式 `ordering=:natural`，调用同一 QDLDL 的 `perm=nothing`；省略选项保留 AMD。父流程 244 项检查及三个精度的默认路径前后进程精确对照通过，独立审阅通过。SDPX 桥接 `772b724` 窄审通过，开发分支整合后重新通过新 provider 的 1,513 项及旧 provider 的 1,400 项检查；各自 262 个散列前后一致，260 个非私有环境散列与被审版本相同。覆盖旧 BFLA 拒绝自然排序且保留 AMD、MFLA 共用 seam、256/512-bit 研究方向的原五方程验收。内存准入仍拒绝，BFLA 主分支与原 `aaa71f3` 资格不变。
- 存储诊断隔离候选 `849bb50` 只统计选定数组的逻辑存储，不由 SDPX 加载。初稿 `ce8c9bb` 被审阅发现 `precision(x)` 会修复反序列化后的 MPFR 指针，父流程在三个精度均复现原始第 4 个 limb word 被改写；现改为已锁定的只读 `x.prec`，增加原始字节不变回归。162 项通过、169 个散列不变，修复复核待完成；独立 C 头文件探针支持相应布局字段。仍不包含完整对象/阶段、MPFR/GMP scratch、allocator/GC 或 RSS 上界，不授予内存准入或 HPC Linux 资格。
- Exp 参考测试 `0c8f767` 已恢复真正跨表示的逐元素协变门，而非仅修改阈值；父流程通过 metric251/chart129 项，最大协变误差约 3.48e-14，最终独立窄审通过。资格限于 reference-only；生产共轭、fallback、epoch、corrector 或 Newton 仍未完成。
- PBS 算术 pilot `211161.node220` 的 harness `6e68fc2` 被父流程判为不能计 A/B 性能资格：所谓 ABBA/BAAB 实为输入种子交换，非调用交替；同进程三次采样也不是独立进程重复，计时插桩和执行来源绑定尚需修复。用户明确批准取消后，父流程核对目标身份与 Q 状态，仅对该 job 执行 qdel（exit0），随后核实 C 状态；原 payload/receipt 保留，未提交重复任务，held210917 不动。此结论不撤销已取得的本地 provider 正确性资格。

新增证据：`local-archives/high-precision-ecosystem-20260908/` 下的 `mfla-parallel-parent-validation/`、`sparse-adapter-parent-validation/`、`exp-covariance-parent-fix/` 、`r3-reviewed-integration/`、`julia112-logical-storage-parent/`、`power-production-next/` 和 `r5-kernel-pbs-pilot-20260908/parent-methodology-audit.md`。

## 架构选择与不可越过的门

1. MultiFloats 负责表示与基础算术；MFA 作为算术/微核研究来源；MFLA、BFLA 保持明确的线代 provider 分工；SDPX 负责数学、结构、资源预算和认证。不默认增加第三套公开线代 provider。
2. 不放宽解/方向/metric 检查以换成功，不静默改变精度或舍入，不用模型名分支，不把未认证 rank reduction 当优化。
3. 检查本身若数值不稳定，应推导等价或有明确误差预算的稳定计算，独立审阅后替换；不能删除负例所依赖的检查或重复构造表达式充当独立验证。
4. 同一算术网络的 SIMD/尾部可要求逐位一致；不同 FMA、两步乘加或融合网络之间须明示语义并验证最坏误差与累计残差，不能用平均误差或 normalized 代替完整正确性。
5. 工作树、测试环境与可变 workspace 各有独立所有者。每次执行记录实际加载源码路径/HEAD、Manifest 和扩展；子任务只能改自己的环境副本。检查受保护环境前后散列，停止任务的空 Git diff 不能证明仓库外未变更。

## R0 · 数学契约、稳定表示与反例闭环

**保留范围**：独立 primal/dual/HSD 推导，强弱不可行及非 Slater 边界；Exp 自协调性/Fenchel 关系；SOC `2μe` 与 Euclidean/svec pairing；所有 RHS、scalar closure、coupled/fused/trial/refinement/ray 消费者。

**下一步**：
- Power：先修复共享对数比值的信息丢失。父流程确认 Float64 `n=3*2^-54,d=1` 的 `log1p(fl((n-d)/d))` 及对应 Phi 误差约 0.287682，而现有 floor 仅 6.7291e-13；不据此声称它就是历史求解失败原因。完整 native 误差账本仍缺失，Cartesian 诊断也不能被当作独立重算真实梯度。随后联合处理根残差、可靠区间、停滞和有限预算，禁止仅收紧区间或盲加迭代数。覆盖 Float64、BF256/BF512、x4 冷种子和近边界点。
- Exp：补中心比值消去的精确点回归及范围/溢出负例；以正确的 Float64 ulp 和显式舍入参考区分梯度重构误差与配对求和误差。研究稳定梯度/metric 表示，保留全部独立门。
- PSD：把实际 SDPX 生成的 `S/Y/P/Pinv/R/Q/Lambda` 直接提升精度回放，不更换 Cholesky、svec 展开或重新求逆；分别处理谱相对精度、表示和检查误差。对 factor-coordinate 或 scaled-frame 方案给出双向误差传播及失效条件。
- 把所有已确认反例纳入永久测试；不把舍入上界、两个精度下的失败或局部失败推成普遍不可能性。

**验收**：独立五方程与原坐标证书一致，非法 metric/错误逆向必须拒绝；公开 LP/SOC/RSOC/PSD/Exp/Power 及 infeasible/unbounded 控制通过；旧算法与 standard-v1 明确区分。

**检查**：Mac 小型精确点、解析问题与负例；本地门通过后提交新、有界 PBS standard-v1 资格任务。未过此阶段，不晋升一般 HSD/Exp/PSD 求解能力。

## R1 · 所有权、AccuracyContract 与算术有效域

**保留范围**：独立 owned-copy、输入/输出变更隔离、统一 AccuracyContract、原坐标 evaluator、失败分类、最小 typed iteration log、native 数值 checkpoint。

**高精度增补**：
- 256/512/1024-bit × ambient/scoped precision/rounding × task/thread 传播；明确支持的舍入模式，再选择统一实现或对不支持模式提前拒绝。重点判别 MA 的 `ROUNDING_MODE[]` 与 BFLA 的 `rounding_raw(BigFloat)`。
- 明确 finite/overflow/subnormal、EFT 前提、normalized/canonical、误差预算、可复现与正确舍入的不同资格。MFA x2–x4 深对消用精确 dyadic 参考；x5–x8 safe 层保持研究/参考身份。
- Julia 1.12.6 BigFloat 为不可变包装，内部 `Memory{Limb}` 仍可变。本机 aarch64 Darwin 的具体数组内联 8-byte 包装；Memory 同时包含 32-byte MPFR 描述符及 significand，不另计逐元素 boxed BigFloat。按实际 backing capacity 和 Memory 身份去重，不能用值相等代替；`precision(x)` 也不必然是只读操作。分别计算逻辑存储、MPFR/GMP 临时分配、allocator/GC 与 RSS；不得把局部逻辑账目当完整内存上界或推广到其他 ABI。

**验收**：无错误 positive certification；source/result mutation 不影响既有结果；默认舍入数值行为保持可核对，非支持上下文明确失败；测试实际加载版本与宣称一致。

**检查**：Mac 逐标量和小矩阵、共享槽/undef/复用、极端 tau、目标常数/符号、scaled rays、NaN/Inf/边界；PBS 同种子跨 ARM/x86 与线程。所有权与证据修订可和 R0 并行。

## R2 · 单一 Prepared Native 与资源生命周期

**保留范围**：PreparedConicProblem、solve-local 数值 workspace、numeric-only update、旧兼容层隔离、required CI、tested-as-executed 选项。

**下一步**：
- 修复结构缓存 enable/disable/clear 与读取/发布的锁纪律；结构缓存不得保存共享可变数值状态。
- 明确 provider、scratch 和线程预算的所有者；精度、结构、布局改变必须失效。持久线程 scratch 不是未经验证的低风险替换。
- 补只读结构/路线诊断：真实 n/m、等式/约化 rank、fixed-trace 适用性、full/compact 维数、存储标量数、provider/扩展、请求与实际线程、各阶段时间；不修改路由决策。

**验收/检查**：同结构 100 次 c/b 更新不重复符号分析；结构改变必失效；并发独立 solver 无共享可变数据或额外内存增长；冷编译、prepare、数值更新、solve、verify 账目可核对。Mac 检查生命周期，PBS 对照外层并发与单 solve 多线程。

## R3 · 稀疏多精度 KKT 与 cone-preserving scaling

**保留范围**：真正 sparse signed-LDL/indefinite provider，原始与正则化算子分离，backward-error 驱动 refinement，按需 rank/nullspace，稀疏 Ruiz、cone-preserving scaling，fill/内存/精度联合路线政策。

**下一步**：先核实 QDLDL 扩展实际加载、数值类型、零 primal 对角与准定性前提；从小型独立 KKT 建立 provider gate，再接 SDPX。不能把可选适配器存在当作稀疏多精度已接通。

**验收**：宣称 sparse 的路径不偷偷形成 dense A/Q 或完整稠密因子；fill 和内存估算可核查；超过预算明确停止，或仅走获授权且有记录的回退；方向始终按原始方程预算验证，禁止静默 Float64 rank 权威。

**检查**：Mac 亏秩/重复等式/正则失效/预算边界；PBS 固定 nnz/行倍增、稠密列、巨大单 SOC、256/512/1024 bits，记录 fill、RSS、方向误差和失败率。依赖 R0/R1 及 R2 seam；独立 provider 调查可提前。

## R4 · Julia 密集核、PSD 与 Bootstrap 结构化加速

**保留范围**：PSD spectral provider、trial 复用、panel/low-rank Gram、真正 chordal 加 recovery、连续批量 Q3、采样/多项式来源结构；削减 O(k⁴) operator 存储，而非只扩内存。

**密集核路线**：
- 同机比较 MFLA direct/packed 与 MFA 候选，核对真实 factorization/viewfree 调用点；先建立算术兼容层和候选开关，不替换默认 FMA。
- 逐项考察 AoS/SoA gather、B/A packing、MC/KC/NC、MR/NR、余数/转置/view/alias、α/β、寄存器压力及 ARM/x86 codegen。Vec8 不等于一条硬件指令，MR 加倍也不自动减半数据流量。
- GEMM/SYRK/GEMMT/TRSM 与 LDLT/Cholesky panel、pivot 和 trailing update 一起测。BFLA 保持显式目标精度和已声明的乘加顺序；改单舍入 FMA 属数值算法变化。
- 不默认进入连续 limb 重写、CRT/硬件拆分乘法或 x5–x8 热核；这些需要单独的表示/精确重构证明和相应热点证据。

**验收/检查**：Mac 阈值两侧、奇数尾部、小矩阵与强基线 A/B；PBS PSD 16/64/128/256、clique 梯、CSDR、完整 N14 与真正 polynomial-matrix 案例。chordal/low-rank 必须有原问题等价性和 dual recovery；性能不得以错误 metric 为基线。结构族 ≥3× 仍是实验目标，不是承诺；独立微基准可提前，operator 集成依赖 R0–R3。

## R5 · 有验证的自适应精度与单节点并行

**保留范围**：低精度因子/目标精度残差分离，precision promotion 重建状态和因子；源数据不足则重新生成，不给 rounded data 补零。

**下一步**：统一叶级线程预算，避免 SDPX/MFLA/BFLA/BLAS 叠加超配；再测任务粒度、task migration、scratch 复用、false sharing、亲和性和 NUMA。只有热点与通信/计算模型支持时才研究 MPI；以 Julia 为主体，不优先 C++ 重写。

**验收/检查**：condition ladder 明确 IR 收缩/失败边界；不收缩则按受支持政策停止或晋升，不静默降精度；串行和并行达到相同数值资格。PBS 1/8/16/32 核，外层并发×内层线程、NUMA 对照；记录实际物理资源。依赖 R1/R3/R4。

## R6 · Production scientific qualification

**保留范围**：支持/实验/不可用表，独立 parser/映射与 MOI 矩阵，取消/checkpoint 语义，长期/held-out 对抗和真实应用，可重建 release 证据。

**验收**：只认证 R0–R5 已过门的子集，公开 known failures；困难问题允许 Unknown，不允许错误自信。锁定 Julia、MA、MPFR/GMP、provider 和扩展状态。未实现 MPI 不阻止边界明确的发布。

**顺序**：新 standard-v1 基线 → N6 控制 → 完整有限 N14/BF512 → SDPX/SDPB 同目标精度与独立原坐标审计 → 8/16/32 核比较。大计算只在 PBS；不重复旧的失败 campaign、不改旧 CSDR 指纹、不触碰 held job 210917。

## 统一测量与第一实施批次

精度按实际类型和模型声明记录：当前 MultiFloats 类型的 `precision` 公式为 `53N−(N−1)`；SDPX 的模型精度元数据另须实测核对，不把 x4、209/208 bits、BigFloat256 混为同一配置。分别比较“同表示配置”与“同认证误差目标”。参考精度随目标增长；1024-bit 目标不能只用 512-bit oracle。

完整记账：编译、prepare、pack、预热核、分解、solve、verify；Julia 分配、已验证口径的外部分配、采样 RSS/峰值、残差与失败率。稳定预热中位数改善 ≥2% 才计 kernel speed credit；最终同时报告 time-to-certified-solution 和精度—时间 Pareto。不能组合不同机器、尺寸、generic 对照的宣传数字得出跨包排名。

两个必须核实而不能凭猜测的规模量：
- N14 的 65 coefficients/9300 Q3 不足以推出约化 rank 或 fixed-trace 资格；先采 canonical 摘要。compact 条件是无 fixed-trace 且 `full > 4*compact`，不是给定块数就恒真。
- PSD k=100 时 q=5050，下三角 Theta 为 12,753,775 个标量；完整 q² 不是当前三角存储数。另计索引、快照、其他块和 factor fill。

自动执行从以下可隔离批次开始，而不是等待整个 R0–R6 一次性完成：
1. 纠正证据口径与环境所有权，保留 R0 失败和反例；源修复只在独立复核后集成。
2. 并行完成 R1 舍入/所有权判别测试、R2 结构缓存锁修复及只读结构/资源诊断。每个工作树只有一个 writer，完成后审阅、父流程复测。
3. 在不改默认策略下建立 MFLA 尾部/view/分解路径、BFLA TRSM/舍入、MFA 算术与微核对照基线；据实际调用与热点选下一项。
4. 按依赖推进 R3/R4/R5，保留原计划的 sparse Ruiz、chordal/recovery、MOI、checkpoint、真实模型和独立证书事项，不能被新 SIMD 工作挤掉。

任何基础设施故障先保存工作树/环境差异和执行状态，只走明确的同协议重试；任何数学 gate 失败先区分计算误差与实际状态错误，不用来源不忠实的高精度重算证明当前因子正确。
