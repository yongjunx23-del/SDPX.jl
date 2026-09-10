# SDPX × Clarabel：源码对照与工程改进执行计划

**审阅日期：2026-09-10；冻结 SDPX：`2ab596fe360fc394698582e7d9cc4a4f548b5386`；冻结 Clarabel fork：`fbc5dd01576d47fda53861a992a02e93c8dbd03e`。**

读取时，SDPX 的 `main` 与 `development/scientific-core-20260907` 指向同一提交。本文讨论指定 fork，而不是假定它等于 Clarabel 上游最新开发版本。

**结论：不应把 SDPX 重写成另一个 Clarabel。应直接吸收 Clarabel 的结构化稀疏表示、精简的数值生命周期、KKT 起点和更新协议，嵌入 SDPX 已有的统一 HSD、FactorCache、MFLA/BFLA、原始坐标认证体系。**

本文是源码审阅和待实施计划，不是性能验收报告。本次没有运行 Julia、两仓库的测试套件或真实求解器性能对照；本地桥接不可达，容器也没有 Julia。附带的 Python 实验只验证合成 SOC 低秩展开及五方程恢复，不是仓库集成测试，也不证明收敛性、任意精度能力或性能收益。本文中的复杂度是结构推导；所有性能门槛均为建议的验收目标，不是已取得的加速。

文中的 `[Cxx]` / `[Sxx]` 对应文末的冻结源码索引，完整不可变定位位于 `source_manifest.json`。

---

## 勘误（2026-09-11，PR-00 执行期间核实）

执行本文时逐条核实，以下论断**在当前仓库中不成立**。正文保持原样以保留审阅记录，但任何依赖下列条目的步骤都必须先按勘误修正。

| 条目 | 正文论断 | 核实结果 | 影响 |
|---|---|---|---|
| 基线 SHA | 冻结 SDPX `2ab596fe360fc394698582e7d9cc4a4f548b5386` | 该提交在仓库中**不存在**（`git cat-file` 失败，`git log --all` 无此对象） | 实际基线为 `db42fd2` 加 30 个未提交改动；已分类为 7 个提交（`b587b1b`…`e49f88b`） |
| F04 / PR-03 落点 / S07 | `src/factor_cache/routes/experimental_sparse_core.jl` 已包装 BF `SparseQDLDLCache` | 该文件在**任何修订中都不存在**；`ExperimentalSparseCoreCache` 标识符在源码树中**零出现**（仅出现在本文第 70 行） | PR-03 无法"推进既有实验包装"。正文所述类型实为 `SparseQDLDLCache`（`src/factor_cache/routes/qdldl_sparse.jl:92`），且它目前**不可从任何公共 `Settings` 到达** |
| PR-08 落点 | `src/factor_cache/session_symbolic_lease.jl` | 该文件**不存在**（仅存在于未合并分支） | PR-08 必须**新建** lease 机制，而非扩展 |
| F06 / S10 | `docs/evidence/P3_01_BETA_EXPERIMENT.md` 记录 β 单因素对照 | 该文件**不存在**；仓库中无任何文件记录 105/107/118 这组数字 | F06 的**结论**（不默认提高 β）可独立论证，但其**所称证据不在本仓库** |
| §3.1 | 当前符号约定为 `cᵀdx + bᵀdy + dκ = r_g`，并称来自 `HSDNewtonRHS`[S09] | 可执行源码使用**负号**：`-c'*dx - b'*dy + dκ = r_g`（`src/kkt/system.jl` 的 `HSDNewtonRHS` 文档串与 `newton_residual!` 算术一致） | 已记入 `docs/design/frozen_math_contract.md`；正文 §6 的规则同样适用于正文自身 |
| §3.3 存储交叉点 | 未给出阈值，仅称"小 SOC 应保留 dense" | 精确定界为 **k = 6**：k=5 时 17 > 15（dense 更小），k=6 时 20 < 21（expanded 更小） | 任何 `dense_small` 阈值必须 **< 6**；已由 `validation/clarabel_borrowing/soc_rank2_gate.jl` 断言 |

**已独立验证的正文内容：** §3.2 的 rank-2 展开等价性与 §3.3 的字节数（`8390656×4×8 = 268500992 B` ≈ 256.06 MiB，`12290×4×8 = 393280 B` ≈ 384.06 KiB）经独立门禁核实通过（375 项断言），见 `docs/evidence/PR02_SOC_RANK2_GATE_20260911.md`。

---

## 1. 先明确哪些已经有，哪些还值得借用

| 维度 | Clarabel 实现 | 当前 SDPX 状态 | 应做的工作 |
|---|---|---|---|
| 统一锥规划迭代 | 一套主循环、锥接口、KKT 系统 | 已有统一 product-HSD 生产入口 | 不新增另一套 solver；压缩热路径重复工作 |
| HSD 消元 | 对称核、每轮常量 RHS、标量闭合 | 已有相同结构与多种已规划的等价 KKT 执行器 | 保留架构，落实真实 factor/solve 生命周期 |
| 稀疏装配 | CSC 槽位映射，锥块形状与低秩扩展 | 已有 frozen CSC、slot map、结构缓存；通用核 shape 仅 `:dense_lower` | 增加结构化锥度量形状，优先大 SOC |
| 高精度稀疏 | QDLDL 泛型接口；具体精度须实际验证 | 原生 descriptor 中 BF/MF 是 dense；BF 有实验稀疏核 | 推进现有实验路径，不重写第三个稀疏框架 |
| 起点 | 对称锥采用身份度量 KKT 起点 | 默认 bordered 自动选择 identity；另有 KKT 起点函数 | 先修启动器双分解与计数，再做起点对照 |
| 锥类型分发 | 函数屏障、锥分发 | 已有按锥族分组的具体类型 runtime | 不重复“消除所有 abstract”；处理批量小核与任务粒度 |
| 残差/缩放生命周期 | 主循环集中更新 | 一般锥试探点构建缩放，接受后下一轮再构建；step 两端刷新残差 | 使用可验证的 accepted-iterate epoch 缓存 |
| PSD | Cholesky/SVD 缩放，也显式构建 packed `Hs` | 已有矩阵 action、PSD panels 等组件 | 可对照缩放内核；不要复制大 PSD 显式 Kronecker 路线 |
| 重复求解 | 同 pattern 数据更新，拒绝不安全的变换组合 | 已有 prepared、结构缓存、session symbolic lease 相关组件 | 在现有机制上补齐明确的数值更新与失效规则 |
| 认证 | 全局终止检查，局部 refinement success 语义较宽 | 严格五方程、原始坐标最优/射线认证 | 不把 Clarabel 的局部成功条件当成 SDPX 的认证条件 |

证据：主循环与消元 `[C01–C03, S01–S04]`；结构与 provider `[C04–C06, S05–S08]`；PSD、更新和既有组件 `[C07–C08, S14, S16]`。

### 推荐优先顺序

第一批是**修正事实和消除重复工作**：真实计数、当前数学契约、残差/缩放 freshness、可运行的对照基线。第二批是**改变大问题的结构成本**：SOC rank-2 CSC、稀疏 provider 生产化、精度相关 planner。第三批才是**减少迭代和扩展能力**：起点、校正策略、PSD 专项、prepared updates。

不要把“再调 β”“把所有矩阵交给多线程 BLAS”“把所有锥转成一种统一的大稠密矩阵”当成下一阶段主线。

---

## 2. 确认的源码级发现

### F01：KKT 起点实际做了两次分解，却报告一次

`src/hsd/initialize.jl` 中，`kkt_derived_start!` 分配一个 `(n+m)×(n+m)` dense 矩阵，先调用 `factorize_pivoted_ldl!` 检查惯性，再创建 LU 因子并调用 `factorize_pivoted_lu!` 求解两个 RHS。成功返回的 `HSDKKTStartReport` 却写入 `factor_count=1, rhs_solves=2`。这是可直接从源码确认的计数不完整，也是真实的启动成本。[S13]

**范围必须说清：这不是“每个 Newton 步都双分解”。** 默认 bordered 路径的 auto 起点是 identity，这段启动成本只影响使用该 KKT 初始化器的路径。不能据此推断全部当前基准耗时。[S04]

立即修复真实调用计数，包括失败尝试；随后让一个能提供合格惯性/正则化信息的 LDL 因子完成两个 RHS。不要为了启用 KKT start，又引入一个额外 dense 启动 KKT。

### F02：接受点的残差和缩放生命周期存在可消除的重复

`product_hsd_step!` 入口刷新残差，接受步后又刷新；常规 `linesearch.jl` 为试探点构建 scaling，接受后复制点到 base，下一步入口又执行 scaling update。实际 `try_update_scaling!` 会重新复制锥点并调用 NT 构建，不含同点 freshness 快捷返回。[S02–S03, S17]

第一项低风险工作是：保留接受步后的权威残差计算，使用 freshness 标识避免下一步对未改变点重复计算。第二项是：证明接受的 scaling 仍对应完全相同的 `(s,y,μ,policy)` 后复用。

不能仅凭 `runtime.last_mu == base.mu` 复用。相同 μ 不代表相同点；Power dual-Hessian 切换、SOC conditioned replay、recenter、terminal trial 和数据更新都可能使状态不同。

**fixed-trace Q3 已经特殊处理：** 试探点只进行轻量可分辨内部性检查，不在每个试探点重复完整 HKM 构建。应保留这一优化，不能把它也报告为未实现。[S03]

### F03：通用对称核的锥块表示能力仍然不足

`SymmetricCorePattern` 已有成熟的 CSC 结构缓存和 slot map，但 `_block_shape_code` 只接受 `:dense_lower`；构造器按块长 k 预留 `k(k+1)/2` 槽位。[S05]

Clarabel 则将可展开的 SOC 度量写成对角部分加 rank-2 更新，通过两个辅助变量装入 KKT。[C04–C05]

这不是“SDPX 没有稀疏 KKT”，而是**现有稀疏 KKT 不能充分表达某些锥的结构**。大 SOC 是明确优先项。纯 orthant 的现有准备路径是否已经拆为 singleton，实施时必须先验证；不得未经确认声称所有 LP 都有平方膨胀。

### F04：高精度稀疏不是缺少所有代码，而是缺少已验证的生产闭环

`_native_hsd_kkt_descriptor` 的原生 sparse augmented 是 Float64/CHOLMOD；常规 BF/MF augmented 则对应 dense pivoted LDL。与此同时，`ExperimentalSparseCoreCache` 已包装 BF `SparseQDLDLCache`，包含 lower→upper map、精度/舍入绑定、原始矩阵快照和 epoch 失效规则，但明确未接入 native/public。[S01, S07]

下一步应处理既有包装的 admissible scope、结构化 metric、provider 能力、方向恢复、五方程认证、内存/失败行为和公共路由描述。不能仅新增一个 `Settings` 枚举就宣布任意精度稀疏完成。

### F05：维数比选择规则需要升级，但不应增加动态试错式 planner

当前 compact Schur 的选择在无 fixed-trace 条件下采用 `full_core_dimension > 4 * compact_dimension`。[S01]

维数是成本因素之一，但并不包含因子填充、锥块结构、精度、provider、内存峰值和构造 Schur 的成本。应在 setup 阶段用已冻结的数据选择，而不是每次迭代对候选后端依次尝试并隐藏成本。

### F06：β 增大不是已经获得支持的优化方向

仓库已有 CSDR Float64x4、4 线程的 β 单因素记录：默认 0.9 为 105 步，0.95 为 107 步，0.98 为 118 步；默认保持不变。[S10]

这些是仓库已有记录，**不是本次复跑，也不是 general-cone 全面结论**。仅凭零 backtracking 不能证明 β 没有影响，因为 β 本身就是乘法阻尼；不要照搬该证据文档中的全部因果解释。可靠结论是：该次实验没有支持简单提高 β。

### F07：性能与高精度发布验证存在需要收束的工程债务

当前 `test.yml` 包含平台/线程的 E2E 矩阵，但注释明确记录 allocation hard gate 已移除，provider smoke 不再作为该 workflow 的必需 job。[S11]

这不等于“当前全部 CI 失败”，本文没有做这个判断。它意味着：声称本次优化已实现 zero-allocation 或高精度 provider 全覆盖，不能只依赖普通 package CI 绿色；需要独立、短小、明确的发布门槛。

### F08：不要让旧说明继续成为新移植的数学入口

`NEWTON_SYSTEM.md` 已注明其旧 gap 符号被后续标准取代，但某些仍可见的说明和源码页头保留历史迁移语言。当前 `src/kkt/system.jl` 明确使用正号的 `c' dx + b' dy + dκ`。[S09, S15]

新测试必须从当前可执行方程独立建立参考，不从旧文档复制另一套符号。文档清理是低风险必要工作，但不应成为一次大规模数值重构。

---

## 3. 移植前冻结的数学契约

### 3.1 当前五方程

用 `r_p, r_d, r_g, h, r_t` 表示已经带有正确符号的 Newton RHS，而不是未取负的残差：

```text
A dx + ds - b dτ       = r_p
Aᵀ dy + c dτ          = r_d
cᵀ dx + bᵀ dy + dκ    = r_g
ds + H dy             = h
κ dτ + τ dκ           = r_t
```

该符号约定来自当前 `HSDNewtonRHS`。[S09]

令

```text
K = [ 0  Aᵀ ]
    [ A  -H ]

K w = [ r_d ; r_p - h ]
K u = [ -c  ; b       ]
```

在分母可分辨且系统相容的普通情形下，

```text
denom = κ - τ (cᵀu_x + bᵀu_y)
dτ    = (r_t - τr_g + τ(cᵀw_x + bᵀw_y)) / denom
dx    = w_x + dτ u_x
dy    = w_y + dτ u_y
ds    = r_p - A dx + b dτ
dκ    = r_g - cᵀdx - bᵀdy
```

上述是直接代入得到的恒等消元；rank-reduced 路径必须包含现有基映射。`denom` 不可分辨、奇异 gauge、rank-deficient 情形继续由现有严格 scalar closure/失败语义处理，不得无条件套除法。

**每个普通 scaling epoch：一次数值分解、一次齐次 RHS 求解、一次 predictor RHS 求解、一次 corrector RHS 求解。** 初始化、正则化重试、refinement、策略切换另计，不得藏入“1 factor”这个理想目标。

齐次 RHS 虽然由固定的 b,c 构成，其解 `u=K⁻¹[-c;b]` 随 K 变化；不能跨数值因子 epoch 复用。predictor 和 corrector 也不能简单一起批量求解，因为 corrector RHS 依赖 predictor。可批量的是已经独立形成的 RHS，例如初始化的两个 RHS，或本 epoch 的齐次/affine RHS。

### 3.2 SOC rank-2 展开的等价性

在 Clarabel 使用的缩放表示中，

```text
H = η² (D + u uᵀ - v vᵀ),   D 为正对角矩阵。
```

可使用对称扩展核：

```text
K_ext = [ 0       Aᵀ       0       0    ]
        [ A      -η²D     -η²v    -η²u  ]
        [ 0      -η²vᵀ    -η²     0    ]
        [ 0      -η²uᵀ     0      +η²  ]
```

消去最后两个辅助变量，第二对角块变成 `-H`。因此该变化是**线性系统的精确表示变换**，不是新增/删减原问题约束，也不是二阶锥的近似。

实际移植必须先建立 SDPX `apply_Theta!` 与该 H 的坐标/缩放对应；不要从 SDPX NT state 取一个名称相似的 w 就直接套公式。fixed-trace Q3 的 HKM 路径不在此次通用 SOC NT 移植范围内。[C04–C05, S03]

### 3.3 存储收益的边界

单个 k 维 SOC 的 packed dense 下三角需要 `k(k+1)/2` 个数值槽；上述扩展核的锥专属部分约需要 `3k+2` 个槽（k 个对角、2k 个辅助列元素、2 个辅助对角）。

取 k=4096：`8,390,656` 对 `12,290` 个槽。只计 Float64x4 的四个 Float64 limb 数值载荷，约 `256.06 MiB` 对 `384.06 KiB`。

这**不是总体内存比，也不是加速倍数**：未计索引、原始副本、工作区、ordering、填充和数值因子。必须记录 `nnz(K)`、`nnz(L)` 和实际 RSS；对小 SOC，额外辅助变量可能得不偿失，保留 dense 小块。

### 3.4 本次独立 Python 检查

附带 `soc_rank2_reference_check.py`，仅使用 NumPy，随机种子为 20260910。对 SOC 维数 3/8/16/32 与三个缩放强度组合，共 12 个合成问题，同时检查 H 的展开、完整五方程 Jacobian 解和扩展核恢复的方向。

实际结果：12/12 通过；最大 metric 相对误差 `1.914e-16`；rank-2 恢复方向的最大五方程归一化后向误差 `7.861e-17`；相对完整 Jacobian 解的最大方向差异 `8.575e-15`。

该检查只覆盖适度条件数的 Float64 合成数据。边界附近、正则化、rank reduction、PSD/Exp/Power、MF/BF、线程和生产原始坐标证书，必须由下面的 Julia gates 另外覆盖。

---

## 4. 总体设计：扩展现有部件，不新建并行求解器

建议保留：

```text
Model / MOI / 原始数据
        ↓
现有 Canonical / Transform / ExecutionPlan
        ↓
同一 ProductConeHSDState 与统一五方程
        ↓
已规划的 KKT 表示：augmented / reduced Schur / 结构特化
        ↓
现有 FactorCache + provider
        ↓
方向验证、步长接受、原始坐标结果认证
```

需扩展的是**度量表示**、**有效状态的生命周期**和**provider 能力矩阵**，不是再增加一个 `ClarabelSolver` 隐藏兜底。

建议的度量接口形状是实现目标，不是现有 API：

```julia
# 建议接口；在现有 AbstractConeLinearization / runtime 中实现。
metric_shape(block)                  # diagonal / dense_small / soc_rank2 / psd_action
refill_metric_values!(map, block)     # 不改变 CSC pattern
mul_metric!(out, block, x)            # 与真实 Θ action 一致
recover_auxiliary!(workspace, rhs)    # 仅结构展开需要
```

存储、formulation、provider、precision、ordering、threads 应保持不同的计划维度。`:sparse` 不等于 `:qdldl`；`:bigfloat` 不等于 `:dense`；“同一个 HSD”也不要求不同精度使用同一个数值分解。

---

## 5. PR 执行序列

### PR-00 — 冻结事实、修正计数、建立可比较基线

**范围：** `[S09, S11, S13, S15]`；建议新增 `validation/clarabel_borrowing/` 与 `benchmark/clarabel_borrowing/`。

实施：

1. 修复初始化计数：LDL 惯性探测和 LU 都进入实际 numerical factor attempt/success 计数。失败路径不能一律返回零而抹掉已发生的调用。
2. 冻结当前五方程、canonical cone ordering、PSD svec 定义、SOC 归一化、RSOC 映射、Power 参数与原始坐标认证阈值。
3. 建立一个小型全 Jacobian oracle，直接计算五组 residual，不复用被测生产装配或 RHS helpers。
4. 先采集旧 SHA 的 canonical benchmark receipts，再改任何优化；若基线已有失败，单独记录，不允许新版本把该行删掉。
5. 记录两个仓库、依赖、provider 的精确 SHA/版本与硬件环境。添加 Clarabel 来源及许可清单。

**验收：** baseline 可重复；计数与实际 provider 调用一致；契约测试在 Float64、可用 MF 和 BF 下通过。此 PR 不改变默认数值策略。

### PR-01 — 接受点状态复用与热循环去重

**落点：** `src/hsd/product_cone_hsd.jl`、`src/hsd/linesearch.jl`、`src/hsd/product_cone_solve.jl`、`src/cones/runtime/types.jl`。

实施：

- 先为 solve-owned 状态建立 `data_epoch / iterate_epoch / scaling_policy_epoch / residual_epoch / scaling_epoch`；可以复用现有计数器语义，不为同一事实再建立互相矛盾的 authority。
- 第一步只消除“上一步接受后权威残差已经计算，下一步入口又计算”这类重复。保留每次接受后的原算术次序，争取逐步 bit-identical。
- 第二步复用接受试探点的完整 scaling。必须证明其点值、μ、算术上下文和策略完全匹配。fixed-trace 仍使用自身 HKM lifecycle。
- terminal trial、tau recenter、conditioned SOC rescue、Power dual-Hessian switch、数据更新和异常恢复全部显式失效或提交正确 epoch。禁止仅修改标签来伪造 freshness。
- 将 debug 环境读取、字符串格式化放入 cold 配置/诊断层，不让观察工具触发数值重算。保留需要的可观测性。

**第二个独立子 PR：** 可将便宜且必需的标量内部性/残差拒绝条件前置，避免为最终必然拒绝的 trial 构建昂贵 scaling；但不能跳过接受点的原检查。该重排可能影响 shadow warm seed 和失败原因，需要独立轨迹测试，不能混入“bit-identical 去重”提交。

**验收：** 普通路径每个接受点的权威 residual/scaling 计算次数下降；所有失效场景均有 mutation tests；A*、Aᵀ*调用计数吻合。重复检查移除前后的方向、接受点与终端证书相同，或给出独立论证的数值等价证据。

### PR-02 — 结构化锥度量与 SOC 稀疏扩展

**借用：** `[C04–C05]`。**落点：** `src/kkt/symmetric_core.jl`、`src/kkt/system.jl`、`src/cones/runtime/symmetric_api.jl`、`src/cones/symmetric/soc.jl`。新建 expansion helper 的名称由现有代码组织决定。

实施：

- 将单一 `:dense_lower` 扩展为经编译/准备确定的合法 shape；最低包含 diagonal、dense_small、SOC rank-2。先确认 orthant singleton 路径，避免重复建设。
- setup 时分配全部 slots、两条辅助列、对角位置、D-sign、恢复 scratch 和 pattern signature。numeric refill 只写 nzval。
- 通过一个适配器把 SDPX 的真实 Θ 映射成 rank-2 参数，独立验证 `H*x`，再验证完整 KKT 和五方程。
- 小 SOC 保持 dense，阈值来自维数/填充/provider 的可复现微基准，不按 benchmark 名称选择。
- 不改变 cone layout 对外定义，不将辅助 KKT 变量泄漏为原问题变量。保持原变量/对偶恢复映射。

**验收：** k=3/8/32/128/512/4096 的 setup storage、nnz、slot bijection、辅助变量消元、五方程均通过；改变 cone partition/ordering/pattern 会使缓存失效；大 SOC 锥专属装配存储随 k 线性增长。因子填充与 E2E 性能另行计量。

### PR-03 — 推进既有稀疏 provider，不新增平行框架

**借用：** `[C03, C06]`。**落点：** `src/factor_cache/routes/qdldl_sparse.jl`、`src/factor_cache/routes/experimental_sparse_core.jl`、`src/factor_cache/routes/sparse_symbolic_numeric.jl`、`src/hsd/native_hsd_public.jl` 及现有 MFLA/BFLA extensions。

阶段：

1. 在 Float64 小问题上对照原 CHOLMOD 与可用的原地 QDLDL 适配路径，确保原始矩阵和 factor-view 分离。
2. 推进现有 BigFloat experimental wrapper：先限经过 rank authority、结构与内存验收的 LP/SOC/小三维锥组合，再扩大范围。
3. MF64x2/x3/x4 按真实 provider capability 分别验证；不得从 BigFloat 或 Clarabel `{T}` 泛型签名推导“已支持”。

必须保持：

- 未正则化的 `K=[0 Aᵀ; A -H]` 不能直接叫 quasi-definite。仅合格的 signed shift 后因子输入具备相应结构；原始五方程仍是接受标准。
- 同 pattern 的 symbolic analysis 复用、numeric refactor、原地多 RHS、transpose/对称语义、precision/rounding/ownership、错误后的撤销全部由既有 FactorCache 接口承载。
- 不读取或写入降精度的数值副本。Int 图结构 ordering 可以与数值算术分离；不允许数值系数悄悄转 Float64。
- 记录实际 `nnzL`、ordering、requested/executed threads、factor/refinement 时间；QDLDL 驱动的串行性质不能包装成多线程加速。[C06]
- 所有 fallback 由计划明确授权并记录；禁止 silently sparse→dense 越过内存上限。

**验收：** 每个已声明支持的精度均有 solver E2E 证据；同 pattern 多步不重复 symbolic；跨 precision/rounding/pattern/策略失效；原矩阵不被 shift 污染；failed factor 后无法 solve stale factor；中大稀疏例不 densify。

### PR-04 — 改造 KKT 起点，再决定是否改变 auto

**借用：** `[C01–C02]`。**落点：** `src/hsd/initialize.jl`、`src/hsd/product_cone_solve.jl`、`src/hsd/native_hsd_public.jl`。

该 PR 必须拆两层：

**A. 启动器工程修复。** 保留既有起点方程，使用一个满足检查要求的 provider LDL 因子求解两 RHS，消除额外 LU；将内存预检放在分配前；支持 planned sparse pattern，不为 sparse solve 隐式准备大 dense 启动矩阵。

**B. 初始化算法对照。** 把现有 identity start、修复后的 existing KKT start、Clarabel-style identity-metric start 作为明确候选。它们不是完全相同的启动方程，不能在一次 patch 中混为“仅优化实现”。先测试纯对称锥；Exp/Power 继续保留合格 central-point 初始化，不把未证明的混合起点推广为默认。

不得跨不同数值 KKT 复用启动因子，只因矩阵形状相同就当作相同。可复用的是可证明一致的 pattern、symbolic 信息和已拥有的 buffer；数值算子改变仍需要 refactor。

**验收：** 统计 `T_start + T_iteration + T_final_cert`，同时比较迭代数与总时间。只有总成本/稳健性有代表性改善后才更新 auto；未获胜的候选不进入默认。启动修复可先独立合并，默认 policy 变更必须另交证据。

### PR-05 — precision/provider/structure-aware planner

**落点：** `src/hsd/native_hsd_public.jl` 和现有 plan/formulation 组件。不要再创建第二个 route authority。

替换固定 `full > 4*compact` 的单因素判断，至少考虑：`n,m,rank`、锥 shape、`nnz(A)`、扩展后 nnz、ordering 预测填充、PSD packed dimension、当前精度和 provider kernel 特性、全部同时存活 buffer 及内存限额。

候选仍是已有统一 HSD 下的等价线性系统：symmetric augmented、compact/reduced Schur、经过结构判定的 fixed-trace Q3 等。评分可用已标定阶段模型，不能在实际求解前把所有候选都数值分解一遍来“选最快”。

建议阶段成本：

```text
T_total ≈ T_setup + N_iter × (
    T_scaling + T_assembly + T_factor
    + N_rhs × T_triangular_solve + T_direction_gate + T_line_search
) + T_final_certificate
```

起初 N_iter 使用保守基线，不声称 setup 预测器知道未来迭代数。若仅替换结构执行器而数学路线一致，先比较同等迭代成本。记录每个候选拒绝原因和预测/实际偏差。

**验收：** 稀疏大 SOC、不同比例的 dense Schur、many-small-cones、大 PSD 各有独立测试。相同数据和算术上下文产生可重现计划；memory gate 不能用开发环境 RSS override 规避。优化不以 CSDR 名称或固定行数作为分支条件。

### PR-06 — 热分配治理与有边界的多线程

**落点：** 现有 cone runtime、Schur/panel kernels、provider/thread-budget 组件。当前已经按锥族分组，不重复发明 abstract→concrete 改造。[S06]

实施：

- Float64 和 MF 固定宽度：定位每步分配，明确是 solver scratch、日志装箱还是 provider 的 `factor \\ rhs` 返回值。不能把包装层无分配当作底层无分配。
- BF：要求不重新分配随问题维数增长的工作区、所有可变 MPFR 值有明确所有权；不在未经测量时承诺每个高精度标量运算零分配。
- many-small-cones 使用足够大的任务粒度和每 worker 自有 scratch；大矩阵由 provider 接管；线程预算避免 Julia block threads × BLAS/provider threads 嵌套超额。
- 固定顺序的局部归约与最终归约，记录实际线程。不能用同一个数值结果的位级一致性假设跨 ISA、BLAS 实现或不同 reduction order；这些改变使用严格误差/证书验收。
- BigFloat 并行必须固定并核实 arithmetic context，禁止在工作任务之间竞争修改全局 precision/rounding。

**验收：** 1/4/16/64 线程的速度、RSS、分配、证书和失败分布；只在资源允许时跑对应档。恢复一个小而稳定的 allocation gate，不将全部大型性能测试塞回普通 E2E。

### PR-07 — 有证据的初始化后迭代策略

**借用候选：** Clarabel 的 affine-step centering、首步较小 Mehrotra correction、显式策略 checkpoint。[C01]

实施纪律：

- β 的现有 CSDR 负结果作为已知证据保留；不再默认“0.9 改成 0.99 就更快”。[S10]
- 先记录每步真实 `mu_aff/mu, sigma_used, alpha_aff, alpha_combined, correction norm, backtracks, retry reason`，不要只记录 requested setting。
- 对照 Clarabel 的 `sigma=(1-alpha_aff)^3` 等候选时，统一 SOC barrier/μ/Jordan product 归一化，以及 primal/dual step 的定义。不同规范下相同 σ 数值不代表相同 Newton RHS。
- 一次只改一个算法因素。首次校正阻尼、centering policy、同因子多重校正、非对称策略切换分别测试。
- corrector 扩展必须使用同一已合格因子，仅当额外 solve 的开销换来实际 E2E 改善才保留。不要将更少迭代等同更快。
- 显式 checkpoint 保留上一个合格 iterate；不覆盖 SDPX 的终端原始坐标证书、SOC replay、Power dual-Hessian 的既有 authority。

**验收：** 一般 LP/SOC/SDP/Exp/Power 全家族，含 mixed cones 与病态例；相同最终认证目标；失败和 maxiter 不能过滤。任何默认策略变化都需要跨多类问题收益，而不是单个 CSDR 最优参数。

### PR-08 — 在现有 PreparedSolver 上完善更新协议

**借用：** `[C08]`。**落点：** `src/prepared.jl`、`src/factor_cache/session_symbolic_lease.jl`、结构缓存、public update/admission 边界。

首先写出依赖失效表：

| 修改 | 可保留 | 必须失效/重新判断 |
|---|---|---|
| 仅 c/b，结构与变换已证明不变 | 符号 pattern，合格结构计划 | 残差、norm、齐次 RHS 解、terminal recovery 中依赖 b 的因子 |
| A 数值变化但 CSC pattern 不变 | 候选 symbolic pattern | numeric factor、rank authority、scaling/变换适用性、原始证书缓存 |
| cone 参数/partition、pattern、精度、rounding、provider 改变 | 仅可证明不依赖这些值的只读资料 | 所有相关缓存与计划；必要时重新 setup |

Clarabel 对 presolve、丢零、chordal 后的更新采取限制，值得借鉴其“拒绝不安全更新”原则，而不是只复制 `update_A!` 接口。SDPX 可以在能证明映射不变时比这个限制更精细，但必须有对应证明和测试。

每次更新原始数据/目标必须使原始证书信息失效。结构复用与 warm start 是两项不同能力；不能因为沿用了 pattern 就跳过新问题的起点/认证。

**验收：** 更新后的结果与全新 setup 对照，重复批量扫描可观测地减少 setup/symbolic 成本；突变/alias/同 pattern 改值触发正确失效；无跨 solve 数值污染。

### PR-09 — PSD 专项与产品化收尾

**PSD：** 可把 Clarabel 的 Cholesky/SVD 缩放作为对照实现，但必须保留高精度 provider 能力与独立测试。对于 side p、packed q=p(p+1)/2 的 PSD，显式 q×q metric 仍有 O(p⁴) 存储；不能把 Clarabel 的 `skron!(Hs,...)` 当作 SDPX 大 SDP 的通用终点。[C07]

优先验证已存在 PSD panels / matrix actions 是否实际服务生产路径；在此基础上选择 blocked Schur、稀疏结构分解或适用的 chordal 变换。任何 chordal 方案必须证明原问题类型、填充变量、重叠一致性和对偶恢复正确；不能见到零元素就随意拆成小 PSD。该项收益依赖问题结构，不默认比当前路径更好。

**可选产品能力：** Clarabel 的原生二次目标 P、generalized power、更多线性求解器等可以作为后续能力扩展；这些不是本轮修复 LP/SOC/SDP/Exp/Power 性能的前置依赖，也不能把未审阅完的 SDPX 功能直接宣布为缺失。

**发布：** 将被采纳的小型 correctness/allocation/provider smoke 加入相应发布门槛；清理历史注释、文档链接、过时性能宣称；把实验/原生/可选能力矩阵和实际测试覆盖一起发布。

---

## 6. 依赖与并行安排

```text
PR-00 基线/计数/数学/来源
 ├─ PR-01 接受点生命周期 ──────────────┐
 ├─ PR-02 结构化 metric ── PR-03 sparse provider ── PR-05 planner
 └─ PR-04A 起点单因子工程修复        │
      └─ PR-04B 起点策略对照 ── PR-07 迭代策略

PR-01 + 结构化表示稳定后 → PR-06 分配/线程
PR-01 + provider 生命周期稳定后 → PR-08 prepared updates
PR-05 + 独立 PSD evidence → PR-09 PSD 专项/发布
```

并行开发可以按测试与生命周期、结构化 KKT、provider 三条线进行；`product_cone_hsd.jl`、`native_hsd_public.jl`、公共 Settings 和核心数学契约必须由同一集成责任人控制，避免多个 agent 同时建立互不相容的 authority。

每个 PR 交付：最小 diff、源 SHA、问题假设、使用的 Clarabel 来源、测试、原始 benchmark receipts、失败列表、默认是否改变、回滚开关。不可只交“tests passed / faster”摘要。

---

## 7. 验收矩阵：不让复制后的共同错误通过互相对照

### 7.1 数学与状态正确性

- 独立完整 Jacobian 五方程 oracle；SOC dense vs expansion；每条 KKT 路线恢复到同一 canonical 方向。
- 原始坐标 primal/dual/gap/cone 证书及不可行射线；rescaling、row/column permutation、精确等价 RSOC/SOC 映射、PSD svec 内积等变性。
- 成功、失败、异常中断、rank ambiguity、near-singular scalar closure、tau collapse、精度不足、内存拒绝的状态语义。
- 修改一个值但保持对象身份、修改 pattern、跨 precision/rounding、recenter、策略切换的缓存失效。
- 各个被复制的 kernel 至少有一个不共享实现代码的参考。不能因为 SDPX 与 Clarabel 复制了同一公式后答案相同，就认为已独立验证。

### 7.2 问题家族

| 家族 | 必须覆盖的结构 |
|---|---|
| LP | 解析已知解/证书、稀疏大例、冗余/病态等式、不可行/无界 |
| SOC | 单大锥、many-small-cones、稀疏与稠密 A、near-boundary；Q3 与 generic SOC 分开统计 |
| SDP | 多小 PSD、单中大 PSD、稀疏约束和稠密约束；不得只使用 2×2 PSD |
| Exp | 解析 entropy/log-sum-exp 类、与等式/orthant 混合、near-boundary |
| Power | 多个 α（含非 1/2）、参数靠近两端、混合锥、已知最优解 |
| Mixed | 多锥族共享变量、不同尺度、非对称 fallback 和原始证书 |
| 应用 | 真实 CSDR/physics 数据；只能作为代表 workload，不能决定全局默认 |

### 7.3 算术和运行环境

Float64、MF64x2/x3/x4、BF256/BF512/BF1024。先运行每种已加载 provider 的小型完整 correctness；大规模按可用内存/时间分级。BF1024 的小例通过不等于大规模高精度可扩展性已证明。

每行记录 Julia/provider/BLAS 版本，源 SHA，precision bits，rounding，requested/executed threads，硬件，设置，effective route，输入指纹和 seed。依赖版本、CPU/ISA 或线程数改变时，性能 receipt 不能默认为同一个比较样本。

### 7.4 性能数据口径

分开计量首次编译、warm fresh-setup、prepared update，不混在一个平均值中。最终主指标是**达到同一原始坐标认证目标的端到端时间及成功率**。

至少包含：setup/rank/equilibration/scaling/assembly/symbolic/numeric-factor/solve/refinement/direction-gate/line-search/final-certificate 分项，factor/RHS 次数、拒绝试探次数、iterations、allocated bytes、GC、RSS、nnzK、nnzL。

建议每个确定性 warm 性能例重复至少五次，报告中位数和离散程度。失败、超时、maxiter、内存拒绝留在总表，另报共同成功子集的速度比；不可只展示共同成功子集掩盖成功率下降。

**建议进入默认策略的初始门槛：** correctness 和原始认证不退化；至少一个预先指定的代表工作负载获得超过噪声的明确改善（可先以目标阶段约 10% 为调查阈值），其余核心代表例端到端中位数无超过约 5% 的未解释回归。阈值须根据基线噪声调整并事先固定；这不是保证达到的收益。

---

## 8. 许可与来源管理

指定 Clarabel fork 是 Apache-2.0，SDPX 当前是 MIT。[C09, S12]

允许把合格的 Clarabel 代码/设计引入 SDPX，但复制或实质改编代码时，不能将其原归属去掉后标成仅 MIT。按 Apache-2.0 再分发条件保留许可文本和相关归属，在修改文件注明变更；上游若有适用的 NOTICE，应保留对应 notices。新增独立代码与整体分发的标识需要准确区分，不能以顶层 MIT 文件覆盖导入部分的条件。

工程交付建议：`THIRD_PARTY_NOTICES.md`、Apache 许可副本、`docs/design/clarabel_borrowing.md`。逐条记录上游/所审 fork SHA、原始路径、本地路径、原样/改编/独立重写类别、关键变更和覆盖测试。若进一步直接引入 QDLDL 或其他 provider 源码，其许可和归属另外审查，不能由 Clarabel 的许可一并代替。

---

## 9. 给执行 agent 的任务边界

```text
任务：按本计划分 PR 改善 SDPX；不要重写第二套 HSD，也不要把 Clarabel 当隐藏求解兜底。
基线：SDPX 2ab596fe360fc394698582e7d9cc4a4f548b5386。
参考：Clarabel fork fbc5dd01576d47fda53861a992a02e93c8dbd03e。

先完成 PR-00，修复真实计数并生成基线，再做 PR-01 / PR-02 / PR-04A。
已有优化不得重做：统一 product-HSD、FactorCache、frozen CSC、结构缓存、
terminal QR cache、按锥族具体类型 runtime、fixed-trace 轻量 trial gate。

数值不可变条件：当前五方程、原始坐标证书、容差、误差门槛、不可行状态语义。
不要用 relaxed_liveness、RSS override、隐藏 Float64 降精度或删除失败样本获得改进。
不要把正则化系统的成功等同于原始 Newton 系统的成功。

每个新公共 route 必须有真实执行与失败行为测试；
BF 稀疏的现有 experimental wrapper 必须逐项升级，不得仅改标签。
MF 与 BF 的支持必须分别验收。

每个提交报告：源 SHA、改动范围、准确计数、测试命令与结果、原始 receipts、
未通过/未运行项、是否改默认、来源与许可。
无性能收益或稳健性退化的候选不进入默认。
```

这里提议新增的 Julia 验收目录和 benchmark 入口**尚未在仓库创建**，应随 PR-00 提交后再运行。当前包级基线可使用原有项目的 `Pkg.test()` 与已存在的 benchmark 入口；不要声称本文的建议入口已经可运行。

附带 Python oracle 可以立即运行：

```bash
python -m pip install numpy
python soc_rank2_reference_check.py
```

它会写出 `soc_rank2_reference_results.json`，只承担第 3.4 节声明的独立代数检查。

---

## 10. 冻结源码索引

以下是本次读取、用于本文判断的主要文件；不是对两个仓库逐文件完整审计的声明。未读取的模块只作为建议实施落点，不据此断言内部缺陷。完整不可变 URL 见配套 `source_manifest.json`。

| 编号 | 仓库 | 文件 | 对照主题 |
|---|---|---|---|
| C01 | Clarabel.jl | `src/solver.jl` | 主循环、起点、centering、checkpoint |
| C02 | Clarabel.jl | `src/kktsystem.jl` | 每轮常量 RHS、变量 RHS 与标量闭合 |
| C03 | Clarabel.jl | `src/kktsolvers/kktsolver_directldl.jl` | 原始矩阵/正则化因子分离、refinement、函数屏障 |
| C04 | Clarabel.jl | `src/cones/coneops_socone.jl` | SOC 度量、O(k) action、rank-2 表示 |
| C05 | Clarabel.jl | `src/kktsolvers/direct-ldl/directldl_datamaps.jl` | SOC 两个辅助变量与 CSC 槽位映射 |
| C06 | Clarabel.jl | `src/kktsolvers/direct-ldl/directldl_qdldl.jl` | logical factorization、原地求解、实际 nnzL/线程数 |
| C07 | Clarabel.jl | `src/cones/coneops_psdtrianglecone.jl` | Cholesky/SVD 缩放及显式 skron |
| C08 | Clarabel.jl | `src/data_updating.jl` | 同 pattern 更新与 presolve/chordal 限制 |
| C09 | Clarabel.jl | `LICENSE.md` | Apache-2.0 与再分发条件 |
| C10 | Clarabel.jl | `src/Clarabel.jl` | 组件边界与预编译入口 |
| S01 | SDPX.jl | `src/hsd/native_hsd_public.jl` | 生产路由、精度后端、4 倍维数选择规则、公开调用 |
| S02 | SDPX.jl | `src/hsd/product_cone_hsd.jl` | step 生命周期、恢复缓存、方向验证与因子所有权 |
| S03 | SDPX.jl | `src/hsd/linesearch.jl` | 试探点缩放、残差、接受点与 fixed-trace 特例 |
| S04 | SDPX.jl | `src/hsd/product_cone_solve.jl` | auto 起点选择、逐步证书、terminal recovery |
| S05 | SDPX.jl | `src/kkt/symmetric_core.jl` | IdentityRankBasis、CSC pattern、仅 dense_lower |
| S06 | SDPX.jl | `src/cones/runtime/types.jl` | 按锥族具体类型分组、checkpoint、worker budget |
| S07 | SDPX.jl | `src/factor_cache/routes/experimental_sparse_core.jl` | 已有 BigFloat 实验性稀疏核包装 |
| S08 | SDPX.jl | `src/factor_cache/routes.jl` | 已有 FactorCache 协议及 provider-specific allocation |
| S09 | SDPX.jl | `src/kkt/system.jl` | 当前五方程的正负号与块线性化 |
| S10 | SDPX.jl | `docs/evidence/P3_01_BETA_EXPERIMENT.md` | 已有 CSDR β 对照负结果；非本次复跑 |
| S11 | SDPX.jl | `.github/workflows/test.yml` | CI 范围；allocation gate 与 provider smoke 状态 |
| S12 | SDPX.jl | `LICENSE` | MIT 许可及已有归属 |
| S13 | SDPX.jl | `src/hsd/initialize.jl` | KKT 起点的 dense LDL + LU 与计数 |
| S14 | SDPX.jl | `src/cones/symmetric/SymmetricCones.jl` | 对称锥内核及矩阵 action |
| S15 | SDPX.jl | `docs/design/NEWTON_SYSTEM.md` | 已标为历史的旧 gap 符号文档 |
| S16 | SDPX.jl | `src/SDPX.jl` | 已有统一入口、providers、PSD panels、prepared/cache 组件 |
| S17 | SDPX.jl | `src/cones/runtime/symmetric_api.jl` | try_update_scaling 实际重建 NT 状态；PSD scaling 的算术操作 |


---

**最终决策：最优路线是“Clarabel 的结构化稀疏工程 + SDPX 的高精度与严格认证”，不是“整仓复制 Clarabel”，也不是继续在现有长循环上无差别叠加检查和启发式。先删除可证明的重复工作，再让表示复杂度与问题结构相匹配。**
