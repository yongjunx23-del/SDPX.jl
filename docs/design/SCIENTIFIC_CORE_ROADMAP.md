# SDPX 科学内核与高精度体系：完整 R0–R6 路线图

更新：2026-09-09。本文接续 2026-09-07 项目审计，更新其执行路线图；旧审计、历史实验与证书保持原样。它是当前技术设计与验收计划，不是会话交接文档。

目标：以 Julia 为主体，提高 MultiFloat、BigFloat 和大规模多核计算能力；优化对象最终是**获得原坐标认证解的总时间与内存**，不是单个算术峰值。用户已授权按本计划自动实施、检查和交付；推送、合并主分支和发布仍不在授权内。

## 当前状态、优先顺序与唯一计划入口

本文是唯一的科学内核执行计划。旧 `HANDOVER.md` 及 `docs/HANDOVER.md` 镜像已由本计划取代；从活跃工作树移除，历史内容仍由 Git 和退休文档归档保存。`docs/evidence/`、原始日志、失败候选、数学规范和独立物理模型的来源计划不在删除范围。

**当前仍在 R0；R0–R6 没有任何完整阶段关闭。** 最近已集成冻结 combined Newton 切片 `72627a8`，整合复测1042项通过。原生 half-Power pair/trial 候选 `44bd368` 在隔离分支通过362项：两个保留输入的cold/warm构造及六个新epoch的combined检查通过，12个固定trial中8个构造成功、4个拒绝。这些是**构造/方向证据，不是line-search accepted progress或生产修复**；该候选尚待独立审阅，最后一轮完整兼容性总回执亦待补齐。

用户最新要求调整实施顺序：**先用其他求解器和精度阶梯给失败分类，再决定修什么；不继续只围绕一个内部失败点反复构造实验。**

| 顺序 | 工作包 | 现在的出口条件 |
| --- | --- | --- |
| 1 | T0：MOSEK / Clarabel.jl / 等价锥形式交叉诊断 | **已完成首轮（见下）**：同一原问题、真实精度、原坐标审计及失败分类 |
| 2 | R0-P：Power原生pair、完整步与有限生产接线 | 根据T0结论选算法；通过真实accepted-step及原五方程，而非只看局部证书 |
| 并行 | R0-E / R0-S：Exp / PSD | 各自先做外部对照和独立几何诊断，再做有限修复 |
| 并行 | R1 / R2：算术所有权、Prepared生命周期 | 可以独立验证；不因Power单例阻断整个体系 |
| 3 | R3：稀疏多精度KKT和完整资源契约 | 原始KKT方向合格，完整峰值上界成立前内存准入继续拒绝 |
| 4 | R4 / R5：结构化核与多核 | 正确性门先过，再做真实调用、同目标误差和实际资源匹配的比较 |
| 5 | R6：真实应用生产资格 | 新standard-v1控制、新CSDR基线、有限N14/BF512、SDPB和8/16/32核证据 |

每包遵循：冻结输入/来源 → 基线与负控 → 单一可证伪改动 → 相关检查 → 独立审阅 → 开发分支整合 → 同HEAD复测。包内失败不自动停止无依赖的其他包；没有过门的路线不得被默认启用。

### R0-P4 最新进度（2026-09-09，dev HEAD 44bdc32）

设计权威：`docs/design/R0P4_FACTOR_PAIR_BOUNDARY.md`（oracle 设计，含 consumer map、refusal 语义、16 步实施计划、验收标准；**不授权默认 dispatch 变更**）。

已完成切片：

1. **step 1 证据台账**（`508566f`）：修正 accepted-alpha 声明——晚期 below-floor 步是**未改动的 progress 门**在算术邻域内的真实表示进展，不是“所有 alpha ≥ 0.026”。
2. **step 3 类型化选择器 + fail-closed 准入**（`bc38f72`）：`NonsymmetricBackendChoice`（默认 `NativeNonsymmetricBackend` 行为不变；`ExperimentalHalfPowerFactorPairBackend` opt-in），`src/hsd/factor_pair_admission.jl` 对 arithmetic/engine/kkt_route/provider/formulation/sparse/scaling/threads/iteration_policy/cones 逐项 typed 拒绝，在 `_public_native_hsd_core` 数值设置前强制；in-scope 请求当前以 `:not_implemented` 失败关闭，**绝不回退默认路径**。39 项选择器/拒绝测试。
3. **step 4 内部命名空间移植**（`dd59a31`）：`src/hsd/factor_pair/*.jl`（FA/NC/HC/FC/NP + 支撑模块），与 validation 参考模块在 pair/epoch/affine/combined/证书上 **21/21 逐位一致**（唯一差异是 owner 绑定的 fingerprint）。
4. **step 6–14 内部生产适配器**（`44bdc32`）：`FactorPairHSD` 使用 `NP.epoch` 准入（非绕过）、typed `FactorPairNumericalRefusal`、未改动的接受门（分量 homotopy、raw max-inf merit+既有 scale、精确 useful-progress、0.9/0.5/64 回溯）、primal+dual+tau/kappa 边界与 `sigma=min(1,(mu_aff/mu)^3)`、**提交前准备并认证下一 epoch**、普通 terminal audit（源证书不等式 + 源 cone 谓词 + `mu/tau²`，无 tau 再除）。canonical power 12 行问题 **28/28**，27 步 terminal 与已认证 loop 逐量一致（audit.m `2.8017973855476926e-9`、obj `1.1242390972345995`、obj_gap `2.243317531736011e-9`、norm_resid `7.004493463869232e-10`、mu/tau² `5.588893961926076e-10`）。

仍未完成（不得声称公开路由合格）：canonical/equality-reduction → factor-pair 映射与可逆行置换、原坐标恢复、route-neutral terminal/public result、完整内存准入、资格矩阵与独立审阅闭环。公开默认 Float64 Power dispatch **未修复**，opt-in 公开路由仍拒绝。

## T0 · 跨求解器、精度与问题适定性诊断（立即执行）

### T0.1 冻结真正的原问题，不把迭代点当问题

- 当前优先案例：`validation/scientific_core/fixtures/factor_affine_trial_17.toml` 与 trial19。只取两份相同的 **A/b/c、锥顺序和alpha**；捕获的x/s/y/tau/kappa是算法状态，不能作为另一个求解器的建模输入。
- 规范形式为 `min sum(t_i)`，`t_i>=0`，`(t_i,1,a_i) in PowerCone(0.5)`，三个a的实际binary64值约为 `0.626678964309454, 0.3230223181314613, -0.7919401216799509`。固定原始位串，所有高精度输入都**精确提升这些存储值**；不从截短十进制重建另一道题。
- 解析校验：`t_i*=a_i^2`，最优值的精确有理数为 `91209111564668556464635313382413 / 81129638414606681695789005144064`，约 `1.1242390986454483421`。这不是任何外部求解器的运行结果。
- 严格原始可行点取 `t_i=a_i^2+1`，每个Power determinant为1。严格对偶可行点取orthant分量1/2和Power对偶块 `(1/2,(a_i^2+1)/2,-a_i)`：stationarity精确成立，`4uv-w^2=1`。本例A还满足 `A'A=2I`。因此这里有明确的可行性/有界性/适定性依据，不能把正常边界最优点的barrier病态误称为原问题ill-defined。
- 输出：可重建canonical输入、精确目标/原始与对偶witness审计、输入文件散列和两份fixture一致性检查。诊断参考不进入SDPX方向或根的生成。

### T0 首轮结果（2026-09-09，已集成 `6a1333c`）

对 trial17/19 同一 canonical问题（精确最优值 `1.124239098645448342103796599...`，有严格原始/对偶内点，`A'A=2I`）：

- MOSEK 11.1.3 Float64在native power与rotated-SOC两形式、默认与1e-12容差下均为optimal，目标误差3.93e-17，complementarity 7.87e-17。
- Clarabel 0.11.1 Float64默认目标误差2.98e-9，收紧后3.04e-14；BigFloat256/512在1e-30容差下目标误差7.29e-31。SOC形式同数量级通过。
- SDPX开发分支Float64为`numerical_breakdown`（目标0）；**同一源码BigFloat256/512为optimal**，目标误差1.48e-26（相对1.32e-26），原始/对偶残差1.88e-26/4.07e-26。

**结论：该问题并非ill-defined，也非不可解。** 唯一位于失败侧的样本是SDPX的Float64非对称Power路径；这是精度敏感的算法/实现缺陷。下一步不再向冻结失败态叠加局部证书，而是以BigFloat路径和两个外部求解器为参考，定位Float64首个差异阶段（root区间、scaling/BFGS guard、三阶contraction、projection）。旋转SOC形式仅作交叉验证；只能过SOC不能过native Power不算修好。证据：`cross-solver-power/`与`validation/scientific_core/cross_solver/CROSS_SOLVER_POWER.md`。

### T0.2 最小对照矩阵

| 求解器 | 真实算术 | 初始请求目标 | 用途与边界 |
| --- | --- | --- | --- |
| MOSEK | 双精度 | 默认约1e-8，再显式1e-12 | 容差不是precision；不称为BF256/BF512，不把near-optimal当严格optimal |
| Clarabel.jl 0.11.1 | Float64 | 默认1e-8，再1e-12 | 与MOSEK相同原输入/目标；记录默认及改变后的设置 |
| Clarabel.jl 0.11.1 | BigFloat256 | 先1e-12同目标，再1e-30 | 区分增加算术精度与要求更高输出精度 |
| Clarabel.jl 0.11.1 | BigFloat512 | 先1e-30同目标，再1e-60 | 检查残差是否继续下降，不能只有BigFloat类型标签 |
| SDPX | Float64 / BF256 / BF512 | 对应同目标 | 记录当前默认路线与显式实验路线，禁止合并两者结果 |
| 可选后继 | Float64x4（实际209 bits） | 已通过BF基线的目标 | 先确认Clarabel/provider真实支持；不假定与BF256等精度 |

两个形式分别记账：
1. **原生Power形式**：相同12×3 canonical问题，prefix3 orthants + 三个Power块。
2. **明确等价的SOC对照**：`(t,1,a)` 映射为 `(t+1,t-1,2a)`，一般块映射为 `(s1+s2,s1-s2,2s3)`。只用整数/二进制精确系数；保存线性映射并用其转置恢复原Power对偶量。也可单列MOSEK rotated-SOC形式 `(t,1/2,a)`，但不能把转换后成功计作“原生Power路径成功”。

初轮不扩成性能campaign：每个必要单元一次有界独立进程；差异、异常或声称改进时再做至少一次独立复现。MOSEK 11.1.3 的Python包、Clarabel.jl 0.11.1及本地许可证文件已经发现，**这只是可用性线索，MOSEK实际许可/求解与高精度求解结果尚未取得**。不得把安装失败或许可证失败计为数学失败。

### T0.3 高精度设置与独立审计

- Julia `setprecision` 必须包住settings、数据、workspace、solve和输出；记录实际存储precision与舍入。目标常数在目标类型内从字符串或整数幂生成，不先变成Float64再补位。
- Clarabel使用原生Julia接口和明确 `direct_solve_method=:qdldl`。先保留默认行为作基线；另列precision-adjusted arm，完整记录gap/feas/infeas/reduced tolerances、static/dynamic regularization、`eps(T)^2`、iterative refinement、step阈值和迭代上限。
- 0.11.1 的 `static_regularization_proportional` 默认来自未带T的 `eps()^2`；BigFloat不能沿用这一默认却宣称所有设置已适配精度。调整正则、equilibration、presolve、minimum-step等均是明确实验变量，不能被隐藏在“只提高precision”的标签里。
- 原输入始终相同。内部equilibration可以做独立开/关对照，但输出必须恢复原坐标；不能只检查缩放后的残差。
- 对每个返回结果保存真实x/s/y或等价的primal/dual活动量、状态、终止原因、迭代次数、目标、配置、时间和资源。独立检查 `Ax+s-b`、`A'y+c`、原锥可行性、目标/dual gap、complementarity及相对精确最优值的误差；分别报告绝对量和明确分母的归一化量。
- 参考可用exact dyadic/rational或有证明的更高精度包络；不得调用被测求解器的同一残差函数充当唯一审计。超时、没有结果和未达到目标也保留原始回执。

### T0.4 结果决定下一步，不预设结论

| 观测 | 解释边界 | 后续动作 |
| --- | --- | --- |
| MOSEK/Clarabel双精度过原坐标门，SDPX失败 | 强证据指向SDPX的算法/实现，而非模型ill-defined | 比较scaling、root、corrector、边界与终止策略；选择最小可复现差异修复 |
| 双精度失败，高精度通过 | 支持有限精度/表示/容差问题，不自动证明原问题病态 | 分开测算术与设置效应，做精度/条件阶梯并据此设计受支持政策 |
| SOC通过、原生Power失败 | 指向锥表示/非对称实现路径差异 | 审计等价映射和对偶恢复；研究可明确声明的通用转换或稳定Power算法 |
| 所有求解器失败 | 不是ill-posedness证明 | 检查canonical映射、Slater/facial结构、秩、尺度、数据误差和证书；只在证据成立后修模型或标Unknown |
| 软件、许可证、类型或provider不可用 | 操作性阻断，不是求解失败 | 精确记录原因；修私有环境或说明未测，不更换证据口径 |

单例止损规则：完成初轮外部/精度/等价形式对照后写出诊断。连续两轮实现实验若没有新的可测改进或推翻假设，应暂停该假设，转查其他solver源码/论文或重审表示，而不是继续叠加局部证书。保留良定义的失败回归；不能删除困难样本、放宽全局门或把局部停工伪装成阶段完成。只有需要新增问题规模、资源费用或生产政策时才请求新决定。

参考入口：[MOSEK power-cone cookbook](https://docs.mosek.com/modeling-cookbook/powo.html)、[数值建模与适定性](https://docs.mosek.com/modeling-cookbook/practical.html)、[MOSEK数值诊断](https://docs.mosek.com/11.1/capi/debugging-numerical.html)、[Clarabel arbitrary precision](https://clarabel.org/stable/literate/build/arbitrary_precision/)、本机Clarabel 0.11.1的 `settings.jl`、`coneops_powcone.jl`、`coneops_nonsymmetric_common.jl`、`solver.jl`。对fallback、third-order correction和reduced-accuracy终止只研究其设计理由；不能直接照搬成删除SDPX独立门的许可。

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

- R0 冻结combined RHS/Newton后继 `2f42ce6` 获独立Astra审阅无问题，整合为 `72627a8`。实际trial17/19各取显式sigma_mu=0/.25mu/.75mu，六例均通过原五方程和独立原生复合算子/高阶项/RHS证书；不通过line search选择sigma。复用原完整bordered LU，保存原始方向；metric的primal/dual/mu与原epoch显式绑定，错误尺寸/类型及重新fingerprint的不一致状态也拒绝。原affine-only门保留，未把旧dense inverse门称为通过。最大原方程归一化上界4.83e-9（原2^-17门），最大forward shift上界6.42e-17（原128gamma3门）。整合后530项combined与239/273项affine控制共1042项通过，三进程247/247/248个散列不变。证据：`combined-newton-epoch/`、`combined-reviewed-integration/summary.json`。下一片为不依赖CAP.replay/fixture的原生pair/trial构造，根成功仍须认证实际存储shadow及factor/decrement/BFGS；尚不要求或宣称accepted progress。

- R0 因子保留研究链已获分片独立审阅并整合至开发分支 `035a28c`，仅新增22个 `validation/scientific_core/` 文件，不改生产默认入口。实际L/R/scale贯穿完整有边框仿射Newton系统，不形成normal equations；独立有理数与原生EFT分别认证真实存储几何、BFGS目标、变换系数及原五方程。compensated-shadow构造的六个候选取得真实因子/metric证书，但旧log-factor门全部拒绝这些新候选，不把新证书改称旧门通过；也不声称六个旧L全部失格。原生仿射计数P2已闭合：trial17完整6383/403373、trial19完整6108/332617次TwoProd/TwoSum，旧3897/291663及3677/236349只是另列的polynomial小计。证据：`factor-affine-epoch/`、`compensated-half-factor/`、`native-affine-certificate/counter-fix-report.md`。
- R0 半参数cold/warm全gap多项式根与current-primal高阶修正已按断开实验范围审阅/集成。根保留[0,1]括区间、原容差和有限预算，166次成功调用最多5次迭代/4次中点探测，不用解析根；低于2^-40的域仍拒绝。corrector使用独立L(s)，不借用L(shadow)，保留真实H分量后验、双序三阶项、原raw-Euler目标ds_aff⋅dy_aff与有界投影。初稿advisor符号错误由父流程在编码前发现并经erratum纠正。trial17/block4原L的eta8.14e-7失格；固定27点首列相邻Float64位型网格第7次得到eta8.84e-8，其余五块保留原候选，全部满足原2^-22且不加ridge/放宽容差。full verifier缺失dual的P1及失败路径计数P2均修复并复审闭合。旧完整corrector本身在这六点也成功，不能将其描述成旧失败。证据：`half-full-gap-root/`、`factor-combined-epoch/`。
- 整合源码 `035a28c` 通过3593项具名复测：stored-affine239、compensated-affine273、full-gap-root2865、current-corrector216；四个独立有界进程分别247/247/238/243个来源/环境散列前后相同。限制仍为已验证Julia1.12.6/aarch64 Darwin原生Float64语境；该片本身不包含combined RHS/Newton；其冻结后继见上。line search、原生trial构造、strict/general-alpha/BF/x4及生产策略尚未闭合，**无R0–R6完整阶段完成**。整合回执：`continuous-factor-integration/summary.json`。

- R0 半参数根/真实存储几何后继 `0f47d73` 获独立窄审并参考性集成为 `9c60352`：两个保留的warm根在1次向外区间Newton求值内达到原相对目标，原生产根仍各失败64次。64项通过；精确有理数证明实际L与真实Newton decrement满足2^-22，但物化H/逆B的真实metric误差超过该目标，尽管H/B均SPD。原生重构在新scratch中执行，未发布任何生产有效状态。其显式、未晋升的dual-Hessian one-secant因子表示与仿射Newton后继见上；strict路径使用H(s)，不得偷换成L(shadow)。证据：`power-root-geometry/parent-report.md`。
- R2 恢复误删的cold owned约束收缩方法（非旧Schur求解器），按审阅及P2修复集成为 `186d76f`。原PreparedSolver首次求解在结果桥接处因缺失buildP_owned!报错；buildP_owned!/accumulate_v_owned!现恢复Dense/Sparse线性与Frobenius语义。最终216项内核、1475项生命周期、20项固定BF256/512结果所有权检查通过。100次c/b更新、结构失效/恢复及4个外层并发会话有独立当前输入/残差/缓冲检查；不把元数据reuse计数当backend符号分析次数，不把cache条目数当RSS上界。证据：`prepared-block-assembly/parent-report.md`。
- R3 私有LP pattern后继 `dedbbfe` 获独立窄审并集成为 `b005c66`：检查计数及canonical CSC后直接分配最终私有结构，消除shared-cache lookup/publication、临时sparse(A)及template复制；默认/shared/provider/原方程门不变。2739项结构/值/所有权、1210项既有研究数值/自然排序/准入拒绝、98项shared-cache协议检查通过。只减少已明确的构造重叠，不证明实际capacity、全对象、MPFR/GMP、GC或保留结果上界；内存准入仍拒绝。证据：`private-sparse-pattern/parent-report.md`。
- 上述三个切片在整合源码 `b005c66` 重新通过共5822项具名检查，七个独立有界进程的来源/环境前后相同；用户原始main工作树、原环境、旧失败证据未动。整合回执：`continuous-core-integration/summary.json`。
- R0 新确认实际 PSD 非有限因子假接受：n=1/2、S=2^-1070 I、Y=2^1000 I 的原默认 NT 返回 valid=true，但实际 Pinv 非有限；数学逆为2^1035 I，已超出Float64表示范围。隔离候选 `4da908a` 仅给 `_psd_nt_close` 加输入、差和原allowance的有限性前置门，不改有限网络/阈值或谱算法；两个真实案例现在拒绝，失败后算子拒用且可恢复。候选455项聚焦检查加28项既有捕获检查，共483项通过，原七案例终态/workspace/算子位型完全相同；Astra独立窄审通过，已作为 `1405207` 集成。连同已审基准代码的开发分支 `8ff6b87` 复测455项有限性/几何、199项精确参考、77项配对控制，共731项通过，来源/环境前后相同；被审生产和测试代码逐文件相同，仅后续文档另记。有限Float16输入但allowance溢出的情形也必须拒绝，不将其称为全有限计算。证据：`psd-finite-gate/parent-report.md`。
- R0 隔离 PSD 阶段捕获 `836e085` 完成父流程检查并获 Astra 独立窄审通过，仅作诊断证据，不集成观察钩子。七个 n=2 Float64 案例在 base/off/on 三进程的终态、workspace 与算子输出位型相同，28/28/181 共 237 项通过，各 230 个散列不变。dyadic 反例的首个 Jacobi 已在非零 off≈8.88e-16≤tol≈4.44e-15 时不旋转返回；实际首个 core inverse 的 DMD−I 误差已约 0.7071，不只是后续 Pinv 构造问题。最终原逆向门正确拒绝。captured_toy 也在首个绝对阈值处退出，但原坐标误差须由保留的带符号传播项分析，不能仅凭未加权 core 残差作普遍结论。参考、拷贝所有权、失败后 valid、精确存储矩阵形状与算子控制见 `psd-stage-capture/parent-report.md`；尚无数值修复或 PSD 求解资格。
- Provider 后继诊断 PBS `211443` 确认 login2/node120 的 Git 版本相同但 RPM/二进制散列不同；独立前后来源不变。严格 node120 专用散列绑定后，PBS `211446` 的身份、零测试 preflight 和末尾来源门均通过，QDLDL 240 项、四线程 285 项通过；core 为 10952 通过、0 失败、1 个 `lu` 导出歧义错误，不能计完整通过。父流程隔离 harness 修复 `faf152b` 仅将 LinearAlgebra 改为 module import 并限定两处 BLAS 名称：本地复现旧错误，新 core 10960 项及三个既有选择 660 项均通过，35 个 core 文件/入口出口标记不变，逐项前后散列相同。该修复获 Astra 独立窄审通过；后继 PBS `211448` 已在 node120 成功完成：core10960、QDLDL240、四线程285，共11485项（选择间有重叠），零测试 preflight 与三次身份门均通过，293/301/293个进程内散列前后相同且全部匹配741个预冻结物理文件。父流程复核原始日志、选择、退出码与来源；远端 Julia1.10/1.11/LinearSolve 仍未运行。期间 ASCII collector 的 Unicode 输出失败已按同一 native 协议修复，只复用有实际 exit0 回执的 setup，不重跑 setup、不改 provider/test/阈值、不覆盖旧证据。资格限该 Julia1.12.6/AMD0.5.3 闭包，不授予性能资格。证据：`provider-git-diagnostic-ebc1d91a/`、`node120-provider-retry-fb2e1748/`、`provider-namespace-fix/`。
- BFLA `5fce2e6` 的新 Mac 兼容性选择由冻结 harness `5f742c4` 执行：Julia 1.10.11 QDLDL/自然排序 240 项、Julia 1.12.6 四线程 285 项、LinearSolve 5.16.0/SciMLBase 3.53.1 扩展 135 项，共 660 项通过。父流程核对原始日志与逐项相等的前后快照（305/293/1323 个散列）；三个私有环境离线建立，无源码或 pin 变更。资格仅限这些选择，不追溯授予旧 `aaa71f3` 的 LinearSolve 资格，也不等于完整核心、求解器或性能资格。证据：`provider-qualification-parent/parent-mac-verification.json`，原始记录 `/var/tmp/sdpx-mac-compat-l5B4wQTL/`。
- 同一 harness 的 PBS `211442.node220`（node120、8 核、16GB、2 小时）已结束，但 core/QDLDL/threading 三项均在数值测试前退出 1：私有 Git shim 报 `Real Git SHA mismatch`，真实 Git 子进程退出 91。不得计为 Linux 数值通过；计划、原始日志及 8 个文件的传输散列已保留于 `provider-qualification-parent/pbs211442/`。登录节点 Git 的固定散列尚未取得计算节点兼容资格；先独立记录计算节点真实二进制身份与来源，再审定修复，不移除散列门、不覆盖原 campaign。远端 AMD 0.5.3 与 Mac 0.5.4 的闭包亦须分别标注。

- R3 主窗口修复 `da0c244` 已获 Astra high 窄审通过：冻结精度与舍入、精确 CSC/回填映射绑定、实验结构数组独立所有权及真正的 post-factor 失败恢复；父流程 1,053 项通过（212 原有 +841 新增），210 个来源/环境散列不变。后继 `10ff7d1` 将内存准入改为 unavailable 并在构造前拒绝；显式私有研究入口不授予内存资格，1,080 项通过包括 27 项拒绝检查，窄审通过。最终研究实现与下述桥接已整合到开发分支 `4701fac`，源码除路线图外与已审阅 `772b724` 完全一致。资格仍限 LP-only、UNADMITTED 小型研究；SOC、完整内存上界、公开路线与规模资格未完成。
- BFLA 隔离后继 `5fce2e6` 新增显式 `ordering=:natural`，调用同一 QDLDL 的 `perm=nothing`；省略选项保留 AMD。父流程 244 项检查及三个精度的默认路径前后进程精确对照通过，独立审阅通过。SDPX 桥接 `772b724` 窄审通过，开发分支整合后重新通过新 provider 的 1,513 项及旧 provider 的 1,400 项检查；各自 262 个散列前后一致，260 个非私有环境散列与被审版本相同。覆盖旧 BFLA 拒绝自然排序且保留 AMD、MFLA 共用 seam、256/512-bit 研究方向的原五方程验收。内存准入仍拒绝，BFLA 主分支与原 `aaa71f3` 资格不变。
- 存储诊断 `849bb50` 的只读 P1 复核通过，已按参考范围整合至开发分支 `83d73d2`；只统计选定数组的逻辑存储，不由 SDPX 加载。初稿 `ce8c9bb` 的 `precision(x)` 会修复反序列化后的 MPFR 指针，父流程在三个精度均复现原始第 4 个 limb word 被改写；改为已锁定的只读 `x.prec` 并增加原始字节不变回归。整合后重新通过 162 项检查、169 个散列不变，157 个源码/测试散列与被审版本相同；独立 C 头文件探针支持相应布局字段。仍不包含完整对象/阶段、MPFR/GMP scratch、allocator/GC 或 RSS 上界，不授予内存准入或 HPC Linux 资格。
- R0 共享对数比值修复 `6b03d3c` 已按窄审范围集成为 `366a82d`，整合后重新通过 1,841 项内核/冷点及 544 项 Exp 检查，各自 259 个散列不变。只在计算出的相对参数位于 [-1/2,1] 时用原 log1p 路径，其余用已有的分别取对数路径；value/work 共用分支，根、metric、求解阈值及迭代上限不变。原反例误差约 0.287682 降至 3.54e-15，独立固定参考目标通过，不靠新分支较大的 work floor。公开 Power：Float64 从旧 151 次 affine-boundary 失败变成 43 次 line-search 失败，仍未修好；BF256 从 86 次认证变为 76 次认证，不计速度资格。完整 native Phi 误差账本与一般 Power 几何/收敛资格仍缺失。
- R0 诊断捕获 `d00149c` 获独立 Astra 审阅通过，限证据权威，不集成观察钩子。在外层及共轭内部回滚前复制实际 Float64 对象/位串；220 条记录中 78 条为内部回滚入口，不能仅凭入口标签判为失败。五次 source-bound 对照保持终态数值位及前 142 条外层记录一致，各 254 个散列不变。bt0/2 实际中点逆矩阵确实不定，拒绝正确；bt1 在新逆构造前失败，其旧/无效 B 不作新候选证据。真实梯度与 gap 梯度仍不等价，不能靠配对或逆矩阵发布解决完整几何。
- R0 隔离候选 `d656070` 在中点拒绝后，才为 Float64 枚举原有 8 个 upper/lower 组合；所有原 native 门通过后，新增仅验证的精确 dyadic/BigInt Sylvester 符号否决。普通 native Cholesky 存在精确半正定/不定假阳性，父流程已复现简单反例及合成 L 扰动的组合反例。默认中点、既有 BigFloat 行为及 Float32/MultiFloat 范围不变；精确整数不生成矩阵值，不替代因子/求解或放宽门。固定 6400-bit 中间量上限不等于物理内存上界。父流程 355 新检查 +1841 内核 +544 Exp =2740 项通过，各 263 个散列不变；独立有理数 Schur 消元参考与真实存储因子支持首个选择 3/5/0。三个类型/精度的独立进程冷点精确值/精度/符号保持一致。公开 Float64 Power 仍在 64 次 line-search 失败，mixed Exp 仍在 22 次失败；BF256 保持 76/69 次认证，纯 Exp 保持 Float64/BF256 的 10/34 次认证。Astra high 实现/证据窄审通过，已集成为 `f62b2f5`；整合后再次通过 2740 项检查，各 263 个散列不变，262 个与被审版本相同，仅私有 Manifest 路径不同。无完整求解或速度资格。
- R0 后继捕获 `ab972f1` 保留已审观察位置，仅将窗口改为 iteration≥62；base/off/on 三次终态数值位相同，各 255 个散列不变，记录 212 条。最终两个搜索各有 33 个 scaling 失败、31 个邻域通过但进展失败；最大已采样可行步约 5.157e-8，不能放宽 useful-progress。头两个失败根的 source-body 回放与直接生产返回值相同：64 次、0 次二分，末尾在相邻浮点数之间循环；误差半径仍超过现有相对宽度门。真实输入的独立参考揭示约 ±3e-17 的 Phi 计算误差；不能仅改迭代上限或接受循环。
- R0 半参数 Phi 补偿参考 `0fd2b0e` 获独立 Astra 窄审通过，按 reference-only 范围整合，不由 SDPX 加载。仅 Float64、α 的位型精确等于 1/2、指定输入/级数域；保留 20/6 个 native FMA/TwoSum 分量，逐原语向外区间和 12 阶多项式余项，不调用原生 log 或生成解析根。父流程 2230 项、272 个散列不变，覆盖 128 个真实根回放 current、独立有理数和定向 MPFR 参考、输入/上下文拒绝及结果隔离；两个循环点的包络半径约 4.93e-32。CLI fast 负控实际映射至 user 模式，失败记录保留，不能冒称 fast-math 拒绝通过；另有非默认 IEEE 模式的范围拒绝检查。整合后复测另记；尚非生产根括区间、停机、真实存储点几何或一般 α 资格。
- Exp 参考测试 `0c8f767` 已恢复真正跨表示的逐元素协变门，而非仅修改阈值；父流程通过 metric251/chart129 项，最大协变误差约 3.48e-14，最终独立窄审通过。资格限于 reference-only；生产共轭、fallback、epoch、corrector 或 Newton 仍未完成。
- PBS 算术 pilot `211161.node220` 的 harness `6e68fc2` 被父流程判为不能计 A/B 性能资格：所谓 ABBA/BAAB 实为输入种子交换，非调用交替；同进程三次采样也不是独立进程重复，计时插桩和执行来源绑定尚需修复。用户明确批准取消后，父流程核对目标身份与 Q 状态，仅对该 job 执行 qdel（exit0），随后核实 C 状态；原 payload/receipt 保留，未提交重复任务，held210917 不动。此结论不撤销已取得的本地 provider 正确性资格。

- R4 配对基准首片 `b868b20` 已实现实际四 limb 位型 fixture、独立 exact-dyadic 整数逐项参考和诚实误差指标，Astra 独立窄审通过。Float64x4/MF3.2.6、MFLA5399c0cc、MFAcdb84680 下，1/4线程各199项通过；另一次固定(23,65,17)、paircancel、β=0.5输入/参考准备通过，三个进程各173个来源/环境散列不变。参考在2^-2148网格精确累加，不调用 BLAS/MFA/MFLA 或 MPFR 生成参考；原生输出不重归一化，3.1e-61门明确标为 max(1,‖reference‖∞) 混合尺度，并另报无floor相对误差和零分母。该参考与后继计时层 `01a2b2c` 已按审阅范围集成为 `a35ef4d`/`8ff6b87`，不由SDPX默认加载。计时层实现真正 ABBA+BAAB 公开调用、逐个输出独立保存/精确检查、非计时 Debug 回执和独立进程汇总：1/4线程各77项控制通过；每个线程预算3个独立进程，共96个计时输出及12个probe/12个allocation输出全过门，六个进程各177个散列不变，另有4个汇总负控通过。仅同一(23,65,17)、β=0.5小型对消尾部单元，计时层获Astra独立窄审通过，仅接纳这些本地观察：1线程预算的进程比值中位数约1.401，4线程约2.173且单进程范围1.766–2.884。4线程预算时 direct planner仅2workers、MFA启动4，不能称等实际worker算术比较。affinity未固定，不提交HPC性能campaign，不计普遍/求解器速度资格。证据：`paired-benchmark-repair/{parent-report,timing-parent-report}.md`。

新增证据：`local-archives/high-precision-ecosystem-20260908/` 下的 `mfla-parallel-parent-validation/`、`sparse-adapter-parent-validation/`、`exp-covariance-parent-fix/` 、`r3-reviewed-integration/`、`julia112-logical-storage-parent/`、`power-production-next/`、`logratio-reviewed-integration/`、`power-linesearch-capture/`、`float64-inverse-veto-parent/`、`inverse-veto-reviewed-integration/`、`power-after-veto-capture/`、`half-phi-parent/` 和 `r5-kernel-pbs-pilot-20260908/parent-methodology-audit.md`。

## 架构选择与不可越过的门

1. MultiFloats 负责表示与基础算术；MFA 作为算术/微核研究来源；MFLA、BFLA 保持明确的线代 provider 分工；SDPX 负责数学、结构、资源预算和认证。不默认增加第三套公开线代 provider。
2. 不放宽解/方向/metric 检查以换成功，不静默改变精度或舍入，不用模型名分支，不把未认证 rank reduction 当优化。
3. 检查本身若数值不稳定，应推导等价或有明确误差预算的稳定计算，独立审阅后替换；不能删除负例所依赖的检查或重复构造表达式充当独立验证。
4. 同一算术网络的 SIMD/尾部可要求逐位一致；不同 FMA、两步乘加或融合网络之间须明示语义并验证最坏误差与累计残差，不能用平均误差或 normalized 代替完整正确性。
5. 工作树、测试环境与可变 workspace 各有独立所有者。每次执行记录实际加载源码路径/HEAD、Manifest 和扩展；子任务只能改自己的环境副本。检查受保护环境前后散列，停止任务的空 Git diff 不能证明仓库外未变更。

## R0 · 数学契约、稳定表示与反例闭环

**保留范围**：独立 primal/dual/HSD 推导，强弱不可行及非 Slater 边界；Exp 自协调性/Fenchel 关系；SOC `2μe` 与 Euclidean/svec pairing；所有 RHS、scalar closure、coupled/fused/trial/refinement/ray 消费者。

**下一步**：
- Power：共享对数比值的信息丢失已按窄范围修复；Float64 额外逆候选的精确否决已完成窄审并集成；后续实际失败点已捕获，半参数 Phi 补偿包络仅取得参考资格。半参数全gap根、因子保留仿射epoch及current-primal高阶修正已取得上述断开实验资格；冻结combined epoch及原RHS、rho/hHat/h、scalar/orthant、原五方程已通过上述受限检查；下一步构造不依赖capture的原生pair/trial epoch，再处理line search和生产策略。禁止放宽 useful-progress 或把较准 Phi 直接套入未证明的旧停机账本。原 Float64 `n=3*2^-54,d=1` 的约 0.287682 Phi 误差不再保留，但旧 floor 并非普遍可靠误差包络；不据此声称它解释全部历史失败。完整 native 误差账本仍缺失，Cartesian 诊断不能被当作独立重算真实梯度。联合处理根残差、可靠区间、停滞和有限预算，禁止仅收紧区间或盲加迭代数；精确 SPD 否决也不能替代根/梯度几何。覆盖 Float64、BF256/BF512、x4 冷种子和近边界点。
- Exp：补中心比值消去的精确点回归及范围/溢出负例；以正确的 Float64 ulp 和显式舍入参考区分梯度重构误差与配对求和误差。研究稳定梯度/metric 表示，保留全部独立门。
- PSD：已取得 pre-overwrite 实际对象捕获资格；下一片为显式、仅 Float64/n=2 的 SPD-relative 谱实验，不改默认/generic/indefinite 路线。先限定相对pivot、范围与小特征值消去误差，检查归一化后的实际特征系统和原输入非对称缺陷；保留全部原坐标门；`1405207` 已修复有限性前置检查，不能以此替代谱误差或原坐标metric证明。实际 `S/Y/P/Pinv/R/Q/Lambda` 只用于提升精度诊断，不更换 Cholesky、svec 展开或重新求逆，不强制两个反例被接受。
- 把所有已确认反例纳入永久测试；不把舍入上界、两个精度下的失败或局部失败推成普遍不可能性。

**验收**：独立五方程与原坐标证书一致，非法 metric/错误逆向必须拒绝；公开 LP/SOC/RSOC/PSD/Exp/Power 及 infeasible/unbounded 控制通过；旧算法与 standard-v1 明确区分。

**检查**：Mac 小型精确点、解析问题与负例；本地门通过后提交新、有界 PBS standard-v1 资格任务。未过此阶段，不晋升一般 HSD/Exp/PSD 求解能力。

### R0实施包与具体交付

- **R0-P2 原生pair/trial构造器已审阅集成（`0602a27`）**：替代capture依赖的FA.build研究路径；全gap多项式根/重建shadow/compensated factor/true-BFGS链，无解析根、无shadow坐标修复、无容差/精度变更。Owner.anchor!、warm lineage、拒绝trial不覆盖anchor、typed refusal（含triangular overflow的专属FactorSeamNumericalFailure）均经两轮closure审阅关闭。367项原生+239/273/530项兼容回执通过，253/247/247/248个散列不变；生产路由未变。
- **关键负结论（父流程实测）**：在trial17首块的近边界点（L[3,3]≈6.5e-8、scale≈7.1e-4），**用Float64物化稠密Θ=S·S′不可行**：G·Θ−I≈7、Cholesky失败、secant≈2e-7。变换本身作为算子一致（WS−I≈1e-8），但稠密矩阵积在条件数≈1e14下灾难性对消。因此修复必须是把因子L/R/scale完整携带到消费者的factor-pair/whole-epoch后端，而不是把factor物化回旧dense Θ。
- **R0-P1 外部差异定位**：T0首轮已完成并集成；问题本身排除ill-defined。Float64失败定位为近边界点的scaling构造失败（`NS_SCALING_CONJUGATE_FAILED`/`NS_CONJUGATE_ITERATION_LIMIT`/`NS_CONJUGATE_HESSIAN_NOT_SPD`，fallback失败，alpha在进度下限上方全部被scaling拒绝，通过scaling的trial在2.3e-23处仅progress失败）。
- **R0-P3 完整accepted step（实验loop已闭环集成）**：`validation/scientific_core/power_runtime/`在dev HEAD含完整实验step loop：真affine边界（primal+dual power边界+orthant+tau/kappa）、`sigma=min(1,(mu_aff/mu)^3)`、原样acceptance门（homotopy/merit/progress floor 2cbrt(eps)）、atomic commit权威链（NP pair、FA epoch、NC affine cert、FC combined）。每步fresh全gap根→重建shadow→compensated factor→true-BFGS证书。经5轮closure审阅关闭（含dual boundary缺失、terminal证书非独立、mu/tau²不变式、source cone-membership谓词、double-tau除尽、同owner stale-token refusal、rejected-trial rollback、cold-rebuild 25/27为观察诊断非保证）。**Float64冷启动→terminal 27步**，每步α∈[0.026,0.90]≫floor；terminal独立dense重算通过ordinary certificate（cert_tol=1e-6，source谓词）：merit 2.80e-9≤1e-8 target、pr/dr 7.0e-10、sNy 6.7e-9、kappa/tau 5.6e-10、mu/tau² 5.6e-10、obj_err 1.4e-9，homogeneous-rescaling不变式回归；`power_epigraph_small`真实benchmark arm（harness数据，27 iter，obj_err 1.4e-9）通过。仍opt-in实验；生产dispatch未变。
- **R0-P4 有限runtime迁移**：映射 `src/cones/runtime/nonsymmetric_api.jl`、`product.jl`、`src/hsd/predictor_corrector.jl`、`linesearch.jl` 及KKT消费者；禁止一个消费者保留因子而另一个重新物化旧Theta。先明确实验one-secant策略，strict double-secant的H(s)义务单独过门；原型审阅不自动授权默认政策改变。
- **R0-E Exp（冻结+定位+precision ladder已闭环）**：纯Exp冻结 `exp_entropy_small`/`exp_logsumexp_small`（`21484f3`）——well-posed（Clarabel分别解出 -log(3) 与 logsumexp 至 ~1e-9；canonical `Ax+s=b` 形式，presolve off），SDPX dev Float64 为 `line_search_breakdown`（iter 24/42，merit ~6e-6，terminal alpha=0，backtracking=15）。Scout定位（`26ab94a`）：第一失效阶段=共轭scaling构造（`try_update_scaling!`）：先 `NS_SCALING_SHADOW_IDENTITY_FAILED`（converged conjugate，block offset 7，iter 12），后 `NS_CONJUGATE_BARRIER_FAILED`（Exp专属 `exp_logarithmic_conjugate!` rho方程，offset 4/10）；所有trial严格interior——近边界interior scaling失败，tau~6-7/kappa~1e-6；metric/predictor/corrector/homotopy/merit全部排除。Precision ladder（`394e6b7`）：同路径BigFloat **optimal+cert_valid，obj=-log(3)至5e-17**——breakdown随精度消失，确认Float64 rounding在Exp rho方程/shadow-identity（Power同族但Exp专属表现）。Repair路线（bounded next）：factor-pair模式适配Exp psi-scale（compensated求值 `L11=1/psi` 分析Cholesky），禁止容差放宽/额外fallback；生产dispatch未变。mixed Exp仍待冻结case。
- **R0-S PSD（实验loop已实现）**：显式Float64/n=2 SPD-relative谱实验已完成并审阅修复：range-safe相对门（|b|/√(ac)，frexp指数分离+fld重建，subnormal区refuse）、bounded rotation（d/β/t设计式）、t=0 unrepresentable-rotation refusal、unresolved/nonpositive-diagonal refusal；dyadic δ=2^-50 case下 production绝对阈值跳过旋转（D0·M·D0−I=0.7071）而相对路线旋转并达7.9e-31（power-of-2可精确表示）或1.1e-16（非幂情形rounding barrier）——记录的测量与门控对比。route `:experimental_relative2`显式选择仅Float64/n=2，生产dispatch未变。44项断言。下游PSD NT scaling的一般近边界行为仍是开放问题（R0-S继续）。
- **R0-Q 总验收**：Float64、BF256/BF512、x4分别运行LP/SOC/RSOC/PSD/Exp/Power/mixed的正常、边界、不可行、无界及非Slater控制；每个对外成功结果独立验证原坐标。任何尚未合格类型/锥组合留在unsupported/experimental清单。

**交付门**：T0诊断、永久失败fixture、数学契约与策略范围、独立审阅、同HEAD完整路径结果、全部原生/参考差异解释。只通过根或frozen Newton不能关闭R0。

## R1 · 所有权、AccuracyContract 与算术有效域

**保留范围**：独立 owned-copy、输入/输出变更隔离、统一 AccuracyContract、原坐标 evaluator、失败分类、最小 typed iteration log、native 数值 checkpoint。

**高精度增补**：
- 256/512/1024-bit × ambient/scoped precision/rounding × task/thread 传播；明确支持的舍入模式，再选择统一实现或对不支持模式提前拒绝。重点判别 MA 的 `ROUNDING_MODE[]` 与 BFLA 的 `rounding_raw(BigFloat)`。
- 明确 finite/overflow/subnormal、EFT 前提、normalized/canonical、误差预算、可复现与正确舍入的不同资格。MFA x2–x4 深对消用精确 dyadic 参考；x5–x8 safe 层保持研究/参考身份。
- Julia 1.12.6 BigFloat 为不可变包装，内部 `Memory{Limb}` 仍可变。本机 aarch64 Darwin 的具体数组内联 8-byte 包装；Memory 同时包含 32-byte MPFR 描述符及 significand，不另计逐元素 boxed BigFloat。按实际 backing capacity 和 Memory 身份去重，不能用值相等代替；`precision(x)` 也不必然是只读操作。分别计算逻辑存储、MPFR/GMP 临时分配、allocator/GC 与 RSS；不得把局部逻辑账目当完整内存上界或推广到其他 ABI。

**验收**：无错误 positive certification；source/result mutation 不影响既有结果；默认舍入数值行为保持可核对，非支持上下文明确失败；测试实际加载版本与宣称一致。

**检查**：Mac 逐标量和小矩阵、共享槽/undef/复用、极端 tau、目标常数/符号、scaled rays、NaN/Inf/边界；PBS 同种子跨 ARM/x86 与线程。所有权与证据修订可和 R0 并行。

### R1实施包

1. **R1-A 统一AccuracyContract**：把存储类型/有效bits、构造精度、工作精度、验证精度、舍入、finite/subnormal/overflow和允许误差分别记录；状态至少区分verified、unsupported、numerical failure和infrastructure failure。
2. **R1-B owned对象矩阵**：覆盖BigFloat256/512/1024的初始化、copy、view、共享槽、重复solve、失败恢复、输出持有及外层任务并发；逐实际backing存储检查，不按值相等推断无alias。验证precision/rounding改变不会复用旧因子。
3. **R1-C provider闭包**：分别补Julia1.10/1.11/1.12、Mac/Linux及LinearSolve/QDLDL扩展缺口；已完成选择可复用其原始证据，不能把一个pin/ABI/扩展的结论迁移到另一个。
4. **R1-D 稳定公共输出**：提供原坐标evaluator、typed iteration log、失败原因与checkpoint精度信息。测试source/result mutation、极小tau和scaled rays；参考不足时返回Unknown而非伪成功。

**交付门**：声明的配置矩阵全部有执行证据；没有错误positive certificate；不把native-core局部测试或逻辑存储探针称为通用算术/内存资格。

## R2 · 单一 Prepared Native 与资源生命周期

**保留范围**：PreparedConicProblem、solve-local 数值 workspace、numeric-only update、旧兼容层隔离、required CI、tested-as-executed 选项。

**下一步**：
- 已审结构缓存锁协议与owned收缩恢复保持回归；下一步验证真实backend symbolic reuse及完整生命周期，shared结构缓存仍不得保存可变数值状态。
- 明确 provider、scratch 和线程预算的所有者；精度、结构、布局改变必须失效。持久线程 scratch 不是未经验证的低风险替换。
- 补只读结构/路线诊断：真实 n/m、等式/约化 rank、fixed-trace 适用性、full/compact 维数、存储标量数、provider/扩展、请求与实际线程、各阶段时间；不修改路由决策。

**验收/检查**：同结构 100 次 c/b 更新不重复符号分析；结构改变必失效；并发独立 solver 无共享可变数据或额外内存增长；冷编译、prepare、数值更新、solve、verify 账目可核对。Mac 检查生命周期，PBS 对照外层并发与单 solve 多线程。

### R2实施包

1. **R2-A 真正的symbolic/numeric分离**：在provider真实symbolic分析入口计数；同结构100次c/b更新应不重复分析。现有metadata reuse只能证明metadata reuse，不满足这项出口。
2. **R2-B 失效事务**：precision、rounding、CSC结构、cone布局、provider和线程预算变化时撤销旧receipt；在构造、factor、solve和证书阶段分别注入失败，确认下一次合法更新可恢复。
3. **R2-C 并发所有者**：外层独立PreparedSolver各有数值scratch/结果；shared cache只放不可变结构。记录实际外层任务×内部线程，检查全部被写buffer的归属和任务迁移，不只看条目数量。
4. **R2-D 资源账本/CI**：分别测冷编译、prepare、update、factor、solve、verify与retained outputs；把测试过的配置放入required CI。稳定重复运行的live-object增长与RSS分别记录，不能互相替代。

**交付门**：真实symbolic计数、失效/恢复负控、所有权证明、可重复资源账目和同结构长期回归均通过。

## R3 · 稀疏多精度 KKT 与 cone-preserving scaling

**保留范围**：真正 sparse signed-LDL/indefinite provider，原始与正则化算子分离，backward-error 驱动 refinement，按需 rank/nullspace，稀疏 Ruiz、cone-preserving scaling，fill/内存/精度联合路线政策。

**下一步**：保留已完成的小型BFLA/QDLDL provider、自然排序和LP-only私有研究检查，推进完整owned-live上界、原系统refinement及其他cone的稀疏表示。新类型/provider组合仍须核实实际加载、零primal对角与准定性契约；现有适配器和未准入LP研究不等于完整公开稀疏多精度资格。

**验收**：宣称 sparse 的路径不偷偷形成 dense A/Q 或完整稠密因子；fill 和内存估算可核查；超过预算明确停止，或仅走获授权且有记录的回退；方向始终按原始方程预算验证，禁止静默 Float64 rank 权威。

**检查**：Mac 亏秩/重复等式/正则失效/预算边界；PBS 固定 nnz/行倍增、稠密列、巨大单 SOC、256/512/1024 bits，记录 fill、RSS、方向误差和失败率。依赖 R0/R1 及 R2 seam；独立 provider 调查可提前。

### R3实施包

1. **R3-A provider数学契约**：明确signed LDL/准定性、零primal对角、排序、正则矩阵与原矩阵、目标precision、实际因子类型；小型独立KKT先验证，不以扩展成功加载代替数值资格。
2. **R3-B 真实稀疏装配**：扩展已审LP-only私有pattern，逐类cone说明块存储、fill和回填映射；任何dense A/Q/Theta/因子形成都须显式披露。等式rank处理必须有原问题证书与dual recovery。
3. **R3-C 原系统refinement**：残差从未正则化的原算子生成，按backward error验证收缩；静态/动态shift、拒绝/重构和精度提升分别留回执，禁止静默Float64 rank权威或隐藏fallback。
4. **R3-D 完整owned-live上界**：分装配/分析/分解/求解/验证/结果保留阶段列出所有并存对象、actual capacity、索引/排列、MPFR/GMP scratch、临时复制、线程scratch及GC重叠；别名只计一次。现有 `8q+a+18d+ell` 是局部小计，不是峰值。不能证明完整界时内存准入继续unavailable，并与仅供研究的未准入入口明确区分。
5. **R3-E 规模阶梯**：Mac先做亏秩/重复等式/失效/预算边界；再PBS做固定nnz增长、dense列、巨大SOC、256/512/1024bits。每格保留fill、RSS、阶段峰值、原方程误差和退出原因。

**交付门**：原始KKT方向、存储/资源契约和失败恢复同时成立，才讨论公开稀疏路线；性能快不能抵消数学或预算失格。

## R4 · Julia 密集核、PSD 与 Bootstrap 结构化加速

**保留范围**：PSD spectral provider、trial 复用、panel/low-rank Gram、真正 chordal 加 recovery、连续批量 Q3、采样/多项式来源结构；削减 O(k⁴) operator 存储，而非只扩内存。

**密集核路线**：
- 同机比较 MFLA direct/packed 与 MFA 候选，核对真实 factorization/viewfree 调用点；先建立算术兼容层和候选开关，不替换默认 FMA。
- 逐项考察 AoS/SoA gather、B/A packing、MC/KC/NC、MR/NR、余数/转置/view/alias、α/β、寄存器压力及 ARM/x86 codegen。Vec8 不等于一条硬件指令，MR 加倍也不自动减半数据流量。
- GEMM/SYRK/GEMMT/TRSM 与 LDLT/Cholesky panel、pivot 和 trailing update 一起测。BFLA 保持显式目标精度和已声明的乘加顺序；改单舍入 FMA 属数值算法变化。
- 不默认进入连续 limb 重写、CRT/硬件拆分乘法或 x5–x8 热核；这些需要单独的表示/精确重构证明和相应热点证据。

**验收/检查**：Mac 阈值两侧、奇数尾部、小矩阵与强基线 A/B；PBS PSD 16/64/128/256、clique 梯、CSDR、完整 N14 与真正 polynomial-matrix 案例。chordal/low-rank 必须有原问题等价性和 dual recovery；性能不得以错误 metric 为基线。结构族 ≥3× 仍是实验目标，不是承诺；独立微基准可提前，operator 集成依赖 R0–R3。

### R4实施包

1. **R4-A 扩展诚实配对矩阵**：从现有单个23×65×17单元扩展到小/中/大、奇数tail、transpose/view、beta与对消等级；冻结实际limb输入、每次调用输出和独立参考，执行真实ABBA+BAAB及独立进程重复。
2. **R4-B 每次只换一个核策略**：按热点选择pack/layout、tile、TRSM或factor trailing update；记录请求线程与实际workers。不同FMA网络比较数值误差，不能要求无理由逐位相同或只看平均误差。
3. **R4-C 结构先证等价**：PSD谱provider、panel/low-rank、chordal+recovery、Q3批量和采样/多项式结构各有独立映射与原坐标审计；先减少不必要O(k^4)存储，不用未经证明的rank截断。
4. **R4-D end-to-end归因**：只保留在合格输入上改善实际热点且无回归的策略；单独报告compile/pack/prepare/factor/solve/verify。kernel改善至少2%的原目标不等于solver改善，更不能从一格外推到CSDR/N14。

**交付门**：核正确性、provider兼容、代表结构族和总认证时间均可复现；同预算不同实际worker的结果必须标明。

## R5 · 有验证的自适应精度与单节点并行

**保留范围**：低精度因子/目标精度残差分离，precision promotion 重建状态和因子；源数据不足则重新生成，不给 rounded data 补零。

**下一步**：统一叶级线程预算，避免 SDPX/MFLA/BFLA/BLAS 叠加超配；再测任务粒度、task migration、scratch 复用、false sharing、亲和性和 NUMA。只有热点与通信/计算模型支持时才研究 MPI；以 Julia 为主体，不优先 C++ 重写。

**验收/检查**：condition ladder 明确 IR 收缩/失败边界；不收缩则按受支持政策停止或晋升，不静默降精度；串行和并行达到相同数值资格。PBS 1/8/16/32 核，外层并发×内层线程、NUMA 对照；记录实际物理资源。依赖 R1/R3/R4。

### R5实施包

1. **R5-A 精度控制器**：明确低精度factor/目标精度residual的组合，监测IR收缩、停滞和错误界；升级时从足够精度的原数据重建所有数值状态与因子，禁止给已舍入数据补零假装提升精度。
2. **R5-B 单一线程预算**：SDPX、MFLA/BFLA、BLAS及外层任务共享明确预算；验证nested parallelism、task migration、thread-local scratch失效和取消恢复。
3. **R5-C 硬件矩阵**：本地1/4线程先过正确性，PBS再做1/8/16/32核、外层×内层、亲和性/NUMA对照。当前已准许的node120:ppn8任务不能超配后冒称16/32个物理核；后两档须先确认并取得足够的真实PBS分配。
4. **R5-D 性能判定**：同原问题、同表示或明确同误差目标、同source/provider配置；每个timed输出过原坐标门，记录真实workers、RSS和总认证时间。不得用更宽松目标或更少有效位换取多核速度。

**交付门**：精度升级/拒绝逻辑正确、串并行达到同数值资格、实际资源和改善归因可信；MPI不是当前单节点通过的前置条件。

## R6 · Production scientific qualification

**保留范围**：支持/实验/不可用表，独立 parser/映射与 MOI 矩阵，取消/checkpoint 语义，长期/held-out 对抗和真实应用，可重建 release 证据。

**验收**：只认证 R0–R5 已过门的子集，公开 known failures；困难问题允许 Unknown，不允许错误自信。锁定 Julia、MA、MPFR/GMP、provider 和扩展状态。未实现 MPI 不阻止边界明确的发布。

**顺序**：新 standard-v1 基线 → N6 控制与新 CSDR 基线 → 完整有限 N14/BF512 → SDPX/SDPB 同目标精度与独立原坐标审计 → 8/16/32 核比较。大计算只在 PBS；不重复旧的失败 campaign、不改旧 CSDR 指纹、不触碰 held job 210917。

### R6实施包

1. **R6-A 支持矩阵**：按cone、类型/有效bits、平台、provider、策略、problem class列supported/experimental/unavailable和known failures；把局部实验与公开solve支持分开。
2. **R6-B 接口与恢复**：独立parser/canonical/MOI映射，目标符号、非有限数据、infeasible/unbounded rays、取消/checkpoint、失败后恢复及长期held-out回归。
3. **R6-C 应用阶梯**：新standard-v1 → N6控制与新CSDR小基线 → 完整有限N14/BF512 → SDPX/SDPB同目标精度与原坐标审计 → 8/16/32核比较。CSDR不得沿用旧101-iteration资格或旧指纹。
4. **R6-D 可重建交付**：汇总源HEAD、Julia/MA/MPFR/GMP/provider/扩展、输入与环境散列、原始回执、审计器、known failures和复现命令。仅形成候选发布材料；push、公共main合并和release仍需另行授权。

**交付门**：仅宣称R0–R5已过门的子集；有限N14的正确性不等于所有bootstrap家族都合格。未通过的问题如实Unknown，不能靠旧结果补齐新资格。

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
