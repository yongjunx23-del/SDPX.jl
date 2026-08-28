# SDPX.jl — THE PLAN（唯一权威计划 v2）

> **本文件是 SDPX.jl 唯一的计划权威**；其他规划/审阅文档（docs/reviews/*、
> 旧 ROADMAP_1_0.md、agent.md 中的阶段摘要）降级为历史参考，冲突以本文件为准。
> **完成态定义：Phase 0–10 全部完成 + v1.0 发布门全绿 = v1.0。**
>
> 来源：用户整体计划（wholeplan，2026-08-28）+ GPT Pro 重构审阅与成熟度审计，
> 由 Lead Agent 审阅采纳并标注当前进度。

最后更新：2026-08-28（Wave B/C/D/E/F 全部落地；Phase 0 缺口清零）

## 进度快照（与本计划 Phase 对照）

| Phase | 状态 | 证据 |
|---|---|---|
| Phase 0 冻结基线 | ✅ 完成 | P0 25/25 + quick gate 9/9 + 来源链（source SHA/manifest SHA）+ include 唯一性 |
| Phase 1 transforms 接管 | 🔄 ~70% | P1 additive + integration 已落地（单一层级、生产 Nonpositive/RSOC 接线、栈危险已修）；剩余：`::Any` 收编、变维 reduction、栈 objective 组合、PSD metric |
| Phase 2 NewtonSystem | 🔄 ~50% | P2 已实现 NewtonSystem/NewtonRHS/expanded 会话/KKT 冷启动；剩余：spec 文档 + Python/BigFloat oracle + 手工 fixture 形式化 |
| Phase 3 dense expanded KKT | 🔄 ~50% | expanded_quasidefinite.jl + 正则化/精化已在（P2）；剩余：惯性检查、动态 pivot 正则化、零分配验证 |
| Phase 4 equilibration/预解/等式/冷启动 | 🔄 ~80%（equilibration/预解/策略 prepared；接线默认路径待大模型验证；KKT 冷启动已落地） |
| Phase 5 统一 HSD 状态机 | ✅ 完成（Wave G/H：状态瘦身、统一邻域/进度、同迭代 route fallback、typed exhaustion；:bordered 仍默认） |
| Phase 6 sparse / Phase 7 多精度 / Phase 8 assembly+fixq3 | ⏳ v0.8 |
| Phase 9 API / Phase 10 legacy 删除 | ⏳ v0.9 |
| **v1.0** | 完成态 = Phase 0–10 全部 + 发布门 |

> 注意：实际执行先于本计划完成了部分 Phase 2/3 内容（P2 wave）——这使
> Phase 2 的剩余工作聚焦在**数学语义冻结与独立 oracle**，而非从零实现。

---


# SDPX.jl HSD-only 全面重构与开发总计划

## 一、总体判断

你的最终愿景是合理的，但实现路径必须进一步收敛为一句话：

> **SDPX.jl 只保留一个产品锥 HSD-IPM 状态机；LP、SOC、RSOC、SDP、Exp、Power 的差异只存在于规范化变换、锥局部算子、KKT 组装策略和线性代数后端中，不再存在多个独立求解器。**

以下计划把你已经完成的 Phase 0 快速门、typed transform 基础，以及 RSOC、Exp、Power 原生路径视为冻结基线，不再重复安排此前已经修复的问题。

当前约 9 万行 `src/` 的主要问题已经不是缺少算法，而是：

* 同一数学内容存在多套实现；
* HSD、旧 LP、旧 SDP、`soc_native` 各有状态机；
* KKT、Schur、factor cache、mixed precision、sparse LA 已经有许多好组件，但没有被一个统一的 HSD 语义层消费；
* `public/`、`pipeline/`、`midend/` 仍在围绕“选择哪个求解引擎”组织，而不是围绕“同一 HSD 的哪个 KKT/LA 路线”组织。

因此，后续不应继续“增强现有 native HSD 文件”，而应逐层建立新架构，再把现有实现迁入，最后一次性删除旧引擎。

---

# 二、先修正最终愿景中的五个技术表述

## 1. Ruiz Equilibration 应是冻结坐标变换，而不是每轮动态预条件

Ruiz scaling 应在：

1. 结构 presolve 后；
2. 数值 rank 判断前；
3. HSD 初始化前；

执行并冻结为 `EquilibrationMap`。

不建议每个 IPM 迭代重新 equilibration，因为这会：

* 改变原始坐标重建关系；
* 破坏 factor-cache symbolic reuse；
* 让迭代残差不可直接比较；
* 增加热路径分配和复杂度。

允许重新 equilibration 的时机只有：

* presolve 改变问题结构后；
* 工作精度升级后；
* KKT 路线整体重建时。

动态机制应主要用于 **KKT 对角正则化、pivot 修复和迭代求精**，而不是每轮改变问题坐标。

## 2. `eps(T)` 是可达精度预算，不是用户容差的替代品

不能简单地设置：

```julia
tol = sqrt(eps(T))
```

然后将其同时用于 primal、dual、gap、ray、step 和 factor acceptance。

正确设计应区分：

* 用户要求的证书容差；
* 当前算术可达到的误差下界；
* KKT backward error；
* 锥局部算子的舍入预算；
* 步长和停滞阈值。

推荐建立：

```julia
struct NumericalBudget{T}
    primal_target::T
    dual_target::T
    gap_target::T
    ray_target::T
    newton_backward_target::T
    scaling_backward_target::T
    step_resolution_floor::T
    attainable_certificate_floor::T
end
```

其中：

$$
\text{attainable floor}
\sim
\gamma_{\mathrm{work}}\,
\kappa_{\mathrm{estimated}}\,
\epsilon(T).
$$

规则是：

* 原坐标证书仍必须满足用户要求；
* 若预测的可达误差高于用户要求，不得偷偷放宽容差；
* 应升级工作精度，或者返回 `InsufficientPrecision`。

## 3. 不应强迫整个工程只能有一个标量类型 `T`

数值内核可以坚持 `T <: AbstractFloat`，但多精度架构最好明确三个角色：

```julia
ArithmeticPolicy{Tsource, Twork, Taccum}
```

* `Tsource`：输入系数原始精度；
* `Twork`：HSD 状态和 KKT 主计算精度；
* `Taccum`：残差、dot、certificate 或 refinement 的累加精度。

常见配置：

```text
Float64       / Float64       / Float64
Float64x4     / Float64x4     / Float64x4
BigFloat512   / BigFloat512   / BigFloat512
BigFloat512   / Float64x4     / BigFloat512    # 可选 mixed route
```

这样才能避免：

* 将 BigFloat 输入先舍入成 Float64 再“升精度”；
* mixed-precision refinement 被迫污染整个 HSD 类型；
* certificate 与 factorization 只能使用相同精度。

## 4. “全局大 Workspace”应改成分层拥有的 Workspace

不要建立一个包含数百字段的单体 `Workspace{T}`。它会再次演变为当前 `Workspace`、`HSDState`、`HotStepState`、SOC workspace、SDP workspace 并存的问题。

目标应是：

```julia
mutable struct SolveWorkspace{T,CW,KW}
    hsd::HSDState{T}
    cones::CW
    kkt::KW
    trial::TrialWorkspace{T}
    certificate::CertificateScratch{T}
end
```

每个子 workspace：

* 有唯一所有者；
* 初始化时一次分配；
* 热路径不改变形状；
* 不跨层保存不属于自己的状态。

这同样可以实现零 Julia heap allocation，但维护性显著更好。

## 5. FPAN 只适用于一部分底层验证

FPAN 在相关研究中是 **Floating-Point Accumulation Network**，主要关注固定浮点累加网络及其误差界自动验证；它适合借鉴到固定归约树、dot、GEMM、SYRK 等微内核，但不能直接替代整个 HSD 求解器的验证体系。([arXiv][1])

SDPX 应采用四层验证：

1. 数学恒等式和手工 fixture；
2. property/metamorphic tests；
3. BigFloat 或独立实现作为数值 oracle；
4. 固定浮点归约网络的 componentwise error proof。

---

# 三、终局架构

## 1. 总体数据流

```text
Model / MOI / Low-level API
            │
            ▼
SourceProgram{Tsource}
            │ compile + typed transforms
            ▼
CanonicalProgram{Tsource}
            │ presolve + equality policy + equilibration
            ▼
ExecutionProgram{Twork}
            │
            ▼
ProductConeRuntime ───────┐
                          │
HSDState ──> NewtonSystem ├──> KKTSession ──> LA Provider
                          │
                          └──> line search / recovery
            │
            ▼
Canonical candidate
            │ reverse transforms
            ▼
Original-coordinate certificate
            │
            ▼
Public Result / MOI Result
```

最重要的约束是：

> **只有 certificate 层可以发布 `Optimal`、`PrimalInfeasible` 或 `DualInfeasible`。**

HSD 中的 \(\tau,\kappa\)、内部 residual、regularized KKT residual 都只能产生“候选状态”。

---

## 2. 建议的文件布局

```text
src/
├── SDPX.jl
│
├── public/
│   ├── model.jl
│   ├── settings.jl
│   ├── optimize.jl
│   ├── result.jl
│   ├── diagnostics.jl
│   ├── low_level.jl
│   └── moi.jl
│
├── program/
│   ├── types.jl
│   ├── compile.jl
│   ├── canonicalize.jl
│   ├── layout.jl
│   ├── reconstruction.jl
│   │
│   ├── transforms/
│   │   ├── interface.jl
│   │   ├── coordinate.jl
│   │   ├── reductions.jl
│   │   ├── nonpositive.jl
│   │   ├── rsoc.jl
│   │   └── psd_metric.jl
│   │
│   ├── presolve.jl
│   ├── equilibrate.jl
│   ├── equalities.jl
│   └── route_plan.jl
│
├── cones/
│   ├── interface.jl
│   ├── product.jl
│   ├── layout.jl
│   │
│   ├── symmetric/
│   │   ├── nonnegative.jl
│   │   ├── soc.jl
│   │   ├── psd_triangle.jl
│   │   ├── eigen.jl
│   │   ├── nt_scaling.jl
│   │   └── boundary.jl
│   │
│   └── nonsymmetric/
│       ├── exponential.jl
│       ├── power.jl
│       ├── conjugate.jl
│       ├── scaling.jl
│       ├── corrector.jl
│       ├── initialization.jl
│       └── boundary.jl
│
├── hsd/
│   ├── embedding.jl
│   ├── state.jl
│   ├── residuals.jl
│   ├── initialize.jl
│   ├── rhs.jl
│   ├── predictor_corrector.jl
│   ├── neighborhood.jl
│   ├── linesearch.jl
│   ├── recovery.jl
│   ├── termination.jl
│   └── solve.jl
│
├── kkt/
│   ├── system.jl
│   ├── contribution.jl
│   ├── assembly.jl
│   ├── expanded_quasidefinite.jl
│   ├── reduced_schur.jl
│   ├── regularization.jl
│   ├── refinement.jl
│   ├── route.jl
│   └── specializations/
│       └── fixed_trace_q3.jl
│
├── la/
│   ├── api.jl
│   ├── capabilities.jl
│   ├── factorization_session.jl
│   ├── dense.jl
│   ├── sparse.jl
│   ├── standard.jl
│   ├── multifloat.jl
│   ├── bigfloat.jl
│   ├── mixed_precision.jl
│   ├── extended_precision_blas.jl
│   └── threading.jl
│
└── certificates/
    ├── canonical.jl
    ├── rays.jl
    ├── reconstruction.jl
    └── original.jl
```

`ext/` 中只放外部 provider 适配：

```text
ext/
├── SDPXMultiFloatLinearAlgebraExt.jl
├── SDPXBigFloatLinearAlgebraExt.jl
├── SDPXLinearSolveExt.jl
└── SDPXAppleAccelerateExt.jl
```

核心求解器不能依赖 extension 才能定义数学语义。

---

# 四、六层职责与所有权

## 1. Program 层

负责：

* 编译；
* canonicalization；
* Nonpositive → Nonnegative；
* RSOC → SOC；
* PSD triangle metric；
* presolve；
* equality strategy；
* equilibration；
* reconstruction chain。

它不允许：

* 保存 HSD iterate；
* 计算 NT scaling；
* 决定 `Optimal`；
* 直接选择某个旧求解器。

建议将 transform 分为两类：

```julia
abstract type AbstractCoordinateTransform end
abstract type AbstractProgramReduction end
```

`AbstractCoordinateTransform`：

* 维度不变；
* Nonpositive；
* RSOC；
* PSD coordinate metric。

`AbstractProgramReduction`：

* 维度可能变化；
* singleton elimination；
* equality elimination；
* duplicate-row removal；
* future chordal decomposition。

二者共同实现 primal、dual、ray 和 objective reconstruction，但不能假装 scratch dimension 完全相同。

## 2. Cone 层

统一接口不应只包含 barrier、gradient、hessian，而应包含 HSD 真正需要的操作：

```julia
initialize_primal_dual!
strictly_interior
update_scaling!
apply_Theta!
apply_G!
affine_shift!
corrector_shift!
max_step_primal!
max_step_dual!
barrier_degree
checkpoint_scaling!
restore_scaling!
```

对称锥：

* Nonnegative；
* SOC；
* PSD triangle；
* NT/Jordan 代数。

非对称锥：

* Exp；
* Power；
* conjugate solve；
* Hessian/double-secant scaling；
* higher-order correction。

`ProductConeRuntime` 只允许看到规范化锥：

```text
Nonnegative / SOC / PSD / Exp / Power
```

以下对象不得进入 runtime：

```text
Nonpositive / RSOC / ZeroCone / Reals
```

## 3. HSD 层

只负责：

* HSD 残差；
* predictor/corrector；
* \(\mu\)、\(\sigma\)；
* neighborhood；
* common step；
* backtracking；
* recovery ladder；
* termination candidate。

它不负责：

* 选择具体 LDL 实现；
* 原始坐标 reconstruction；
* MOI status；
* presolve；
* cone-specific matrix algebra细节。

## 4. KKT 层

建立唯一的：

```julia
NewtonSystem
```

它一次性定义：

* primal affine equation；
* dual affine equation；
* homogeneous gap equation；
* cone complementarity equation；
* \(\tau\kappa\) equation。

所有路线都只能是同一个 `NewtonSystem` 的不同消元方式：

```text
DenseExpandedKKT
SparseExpandedKKT
DenseReducedSchur
SparseReducedSchur
FixedTraceQ3Contribution
```

禁止每个路线重新手写符号和 \(\tau,\kappa\) border。

## 5. LA 层

建立窄接口：

```julia
analyze!
factorize!
solve!
solve_multi!
refine!
invalidate!
factor_diagnostics
provider_capabilities
```

`FactorizationSession` 唯一拥有：

* symbolic ordering；
* numeric factor；
* matrix epoch；
* factor epoch；
* regularization；
* inertia；
* refinement statistics。

每个 HSD iteration：

* predictor；
* corrector；
* centering restoration；
* iterative refinement；

必须共享同一个 factor。

## 6. Certificate 层

输入：

* 只读 HSD candidate；
* canonical program；
* reconstruction chain；
* original model；
* 用户容差。

输出：

* verified optimal；
* verified primal infeasible ray；
* verified dual infeasible ray；
* unverified candidate。

regularization、equilibration、presolve、RSOC 和 Nonpositive 的所有坐标变化都必须逆向恢复后再检查。

---

# 五、当前代码的迁移决策

## 保留并升级

| 当前代码                                | 处理方式                                    |
| ----------------------------------- | --------------------------------------- |
| `modeling/*`                        | 保留，成为 public model frontend             |
| `ir/types.jl`、`ir/storage.jl`       | 保留并收敛为 Program 类型                       |
| `program/transforms*.jl`            | 生产化，替代旧 `CanonicalBlockMap` 权威          |
| `cones/symmetric/*`                 | 保留 A1 数值实现                              |
| `cones/nonsymmetric/*`              | 保留，统一 runtime 接口                        |
| `cones/runtime/*`                   | 合并成 ProductConeRuntime                  |
| `certificates/certificates.jl`      | 保留并拆分                                   |
| `_public_original_certificate`      | 成为唯一 public success authority           |
| `factor_cache/*`                    | 保留协议和 epoch 设计                          |
| `kkt_route.jl`                      | 升级成 `FactorizationSession`              |
| `sparse_la.jl`                      | 提取 sparse symbolic/numeric 机制           |
| `kernels/mixed_precision_kkt.jl`    | 提取精度预测和 refinement 策略                   |
| `kernels/threaded.jl`               | 保留 deterministic LPT 和 thread ownership |
| `kernels/extended_precision_blas/*` | 作为 MultiFloat 高性能内核                     |
| `cold_start.jl`                     | 提取 cone interior repair 和 centering     |
| `step_hot.jl`                       | 保留热路径纪律，不保留旧数学状态机                       |
| `soc_native.jl` 的 `fixq3`           | 抽取为局部 KKT contribution                  |

## 拆分或合并

| 当前代码                                 | 目标                                            |
| ------------------------------------ | --------------------------------------------- |
| `hsd/hsd.jl`                         | `embedding/state/residuals`                   |
| `hsd/product_cone_hsd.jl`            | `rhs/predictor_corrector/linesearch/recovery` |
| `hsd/product_cone_solve.jl`          | `termination/solve`                           |
| `hsd/equality_reduction.jl`          | `program/equalities.jl`                       |
| `kkt.jl`、`schur.jl`、`kkt_backend.jl` | 统一到 `kkt/*` 和 `la/*`                          |
| `pipeline/*`、`midend/*`              | 从 engine planner 改成 KKT/LA/precision planner  |
| `public/optimize.jl`                 | 只负责 compile → plan → HSD → certificate        |

## 最终删除

在新 HSD 通过全部验收后删除：

```text
src/lp_solver.jl
src/lp_sparse.jl
src/solver/interior_point.jl
src/step.jl
src/hsd/nonnegative_hsd.jl
src/soc_native.jl              # 抽取 fixq3 后
src/hsd/nonsymmetric_schur3.jl
src/hsd/nonsymmetric_coupled.jl
```

同时删除或移入 `test/reference/`：

```text
cones/nonsymmetric/full_newton_reference.jl
重复的 cone_algebra 实现
旧 Schur/KKT reference kernels
```

代码行数不是正式验收指标；真正指标是：

* 一个 HSD solve loop；
* 一个 `NewtonSystem`；
* 一个 certificate authority；
* 每种数学操作只有一个 production implementation。

---

# 六、版本与阶段计划

## v0.6：数学语义与 Program 层收敛

### Phase 0 — 冻结新基线　✅ **已完成**（P0+P1.5）

**目标**

将你已经修复的上一轮问题冻结成不可回退基线。

**工作**

* 为当前 commit 打基线 tag；
* quick gate 记录：

  * source SHA；
  * test manifest SHA；
  * Julia version；
  * BLAS/provider；
  * arithmetic；
  * thread count；
* 加入 include 唯一性和生产实现唯一性检查；
* 更新 stale `@test_broken` 和 known-gap manifest。

**文件**

```text
src/SDPX.jl
test/quick_gate.jl
test/kernel_failure_regressions.jl
test/architecture_regressions.jl
```

**验收**

* `<60 s` quick gate 继续绿色；
* full suite 绿色；
* 不改变容差；
* 所有成功状态继续通过原坐标证书。

**周期**

2–3 天。

---

### Phase 1 — Typed transforms 完全接管生产 lowering　🔄 **进行中 ~70%**

**目标**

让 P1 transform 不再只是 additive test infrastructure，而成为唯一生产坐标变换权威。

**文件**

```text
src/program/transforms*
src/ir/canonical.jl
src/ir/layout.jl
src/ir/reconstruction.jl
src/ir/lower_lp.jl
src/ir/lower_soc.jl
src/ir/lower_sdp.jl
src/modeling/compile.jl
src/cones/runtime/product.jl
src/public/optimize.jl
```

**工作**

* 统一 transform hierarchy；
* 区分 coordinate transform 与 program reduction；
* production wiring：

  * Nonpositive；
  * RSOC；
  * PSD svec metric；
* stack 组合 objective constant；
* 支持不同 source/target dimension；
* 取消生产路径中的 `Any` transform metadata；
* runtime 构造只接受 `NormalizedConeLayout`；
* primal、dual、primal ray、dual ray 全部使用同一 chain。

**验收**

对每个 transform 验证：

$$
\langle s,y\rangle
=
\langle Ts,T^{-T}y\rangle,
$$

以及：

* stationarity invariant；
* objective invariant；
* optimal solution round trip；
* primal ray round trip；
* dual ray round trip；
* composition invariant；
* full `Model → compile → solve → original certificate`。

**周期**

7–10 天。

**可并行**

* Program 实现代理；
* 独立 property-test 代理；
* PSD metric 审核代理。

---

### Phase 2 — 建立唯一 `NewtonSystem`　🔄 **进行中 ~40%**（P2 已落地实现，本阶段补 spec/oracle 冻结）

**目标**

先冻结数学方程，再实现新的 factorization route。

**新文件**

```text
src/kkt/system.jl
src/kkt/contribution.jl
src/hsd/residuals.jl
src/hsd/rhs.jl
```

**工作**

定义：

```julia
struct NewtonSystem{T}
    ...
end

struct NewtonRHS{T}
    primal
    dual
    gap
    cone
    scalar
end
```

Cone 层只返回：

* local self-adjoint linearization；
* affine shift；
* corrector shift；
* boundary information。

KKT 路线只消费 `NewtonSystem`，不得自行重新推导符号。

**验收**

* 手工小例验证五组 Newton 方程；
* predictor 与 corrector 共享 affine residual RHS；
* corrector 不改变 feasibility homotopy；
* common step 后：

  $$
  r_{\mathrm{new}}\approx(1-\alpha)r_{\mathrm{old}};
  $$
* 每种 KKT route 的方向都通过同一 unregularized Newton residual verifier；
* 独立 Python/BigFloat oracle 与 Julia production 方程一致。

**周期**

5–7 天。

**回滚边界**

本阶段先以 adapter 连接旧 HSD，不修改迭代控制流。

---

## v0.7：鲁棒 KKT 与统一 HSD

### Phase 3 — Dense expanded quasidefinite KKT

**目标**

建立通用鲁棒路线，停止把 dense bordered normal equations 当作唯一 HSD KKT。

**主要来源**

```text
src/kkt_formulations/dense_augmented.jl
src/factor_cache/routes/dense_augmented_ldlt.jl
src/kkt_backend.jl
src/soc_native.jl::_native_soc_assemble_factor!
```

**新目标文件**

```text
src/kkt/expanded_quasidefinite.jl
src/kkt/regularization.jl
src/kkt/refinement.jl
src/la/factorization_session.jl
src/la/dense.jl
```

**必须实现**

* symmetric indefinite / quasidefinite assembly；
* pivoted \(LDL^\top\)；
* expected inertia；
* signed static regularization；
* dynamic pivot regularization；
* one factor / multiple RHS；
* unregularized iterative refinement；
* structured failure reason。

**恢复顺序**

```text
planned factorization
→ static regularization
→ dynamic pivot regularization
→ refinement
→ alternate KKT route
→ precision escalation
→ typed failure
```

**验收**

* full rank；
* dependent equalities；
* mixed free/equality/PSD；
* near-singular KKT；
* wrong inertia 注入；
* regularized direction 必须通过未正则化方程；
* 每个 iteration 只有一次 numeric factorization；
* fixed-width warm solve 零 Julia allocation。

**周期**

10–15 天。

---

### Phase 4 — Equilibration、presolve、equality policy、KKT cold start

**新文件**

```text
src/program/equilibrate.jl
src/program/presolve.jl
src/program/equalities.jl
src/hsd/initialize.jl
```

**从现有代码迁入**

```text
cold_start.jl::_continuation_psd_repair!
cold_start.jl::_cold_start_identity_mass_shifts
cold_start.jl::_cold_start_centering_shifts
soc_presolve.jl 的 singleton elimination
lp_solver.jl 的 KKT-derived start
solver/interior_point.jl 的 SDP KKT start
```

**Equilibration 策略**

* free-variable columns：独立正比例；
* equality rows：独立正比例；
* Nonnegative：允许逐行 scaling；
* SOC/RSOC：每块统一 scaling；
* PSD：v1 每块统一 scaling；
* Exp/Power：每块统一 scaling；
* PSD diagonal congruence scaling 放到后续版本。

**Equality 策略**

```text
小型 dense + 明显降维        → pivoted RRQR
大型 sparse + fill 可控      → sparse QR
mixed free/equality/PSD      → 默认保留在 expanded KKT
```

全流程最多进行一次数值 rank reduction。

**Cold start**

1. 在 equilibrated 坐标求 primal affine least-residual start；
2. 求 dual affine least-residual start；
3. 最小 strict-interior shift；
4. identity-mass floor；
5. primal/dual cross-centering；
6. PSD continuation repair；
7. Exp/Power validated central initialization；
8. \(\tau=\kappa=1\)。

**验收**

* 初始化严格 interior；
* 初始 residual 显著优于 raw identity start；
* 不再强制建立 dense null basis；
* equality reconstruction 通过原坐标 stationarity；
* scaling round trip 不改变证书。

**周期**

7–12 天。

---

> **Wave F 评估结论（2026-08-28）**：:bordered 保持默认；equilibration 保持
> opt-in（小探针条件数恶化：2.88→4.42、5.34→21.90）。**新增 Phase 5 门项**：
> :expanded 在简单模型（LP/SOCP/Matrix）上 iteration-0 breakdown——统一 HSD
> 迁移必须让 expanded 路由同时覆盖简单模型，不得只在缺口模型上工作。
> 数据与决策记录：test/wave_f_defaults.jl。

### Phase 5 — 迁移并重写唯一 HSD 状态机

**目标文件**

```text
src/hsd/embedding.jl
src/hsd/state.jl
src/hsd/predictor_corrector.jl
src/hsd/neighborhood.jl
src/hsd/linesearch.jl
src/hsd/recovery.jl
src/hsd/termination.jl
src/hsd/solve.jl
```

**工作**

* 将 `HSDState` 中的：

  * dense `Ad`；
  * reduced `Ar`；
  * rank basis；
  * Schur-specific buffers；

  移入 route-specific KKT workspace；
* HSDState 只保留数学状态；
* symmetric 和 nonsymmetric cones 统一消费 `ConeLinearization`；
* 取消以 cone family 决定不同迭代器；
* 保留 common step；
* 将 family-gated backtracking 替换为：

  * useful-progress threshold；
  * neighborhood criterion；
  * current/trial terminal certificate；
  * centering restoration；
* 在任何 direction、scaling、factor 或 line-search failure 前，先验证当前和有限 trial candidate；
* 加入 route fallback 和 precision callback。

**状态分类**

```text
Optimal
PrimalInfeasible
DualInfeasible
MaxIterations
TimeLimit
StalledUnverified
InsufficientPrecision
NumericalFailure
RankAmbiguous
InvalidModel
```

**验收矩阵**

每种规范化数值锥：

```text
Nonnegative
SOC
PSD
Exp
Power
```

至少覆盖：

* interior optimum；
* rank-one/boundary optimum；
* ill-scaled optimum；
* mixed product-cone optimum；
* primal infeasible；
* dual infeasible；
* redundant/dependent rows。

RSOC 和 Nonpositive 在原坐标中单独验证 reconstruction。

**周期**

10–15 天。

**Wave G/H closeout（2026-08-28）**

* ✅ `HSDState` 只拥有 embedding 数学状态、方向、残差、trial、计数与诊断；
  dense/reduced/rank/Schur storage 归 `BorderedHSDWorkspace`，expanded factor
  storage 归 `ExpandedKKTSession`；
* ✅ symmetric/nonsymmetric 保留同一 common-step 接受路径；family-gated
  backtracking 已由统一 64-trial、neighborhood 和 useful-progress 判据替代；
* ✅ direction/scaling/line-search/耗尽出口维持 certificate-first authority；
* ✅ expanded factor/solve/refinement 失败可在**同一 HSD iterate**切换 bordered，
  不重启、不改 frozen equations；
* ✅ `IterLimit`、`TimeLimit`、`InsufficientPrecision`、`NumericalFailure` 等
  typed failure 贯穿 product/public termination；证书仍只在原坐标晋升状态；
* ✅ expanded 默认重评：quick/Phase-0/Wave-D 子集无退化且关闭 tiny SOC/rank-one
  PSD；完整 opt-in 矩阵中 LP+SOC 从 certified optimal（dual 3.29e-11、gap
  1.19e-11）退化为 `IterLimit`，SOC+PSD 从 certified optimal（dual 4.10e-11、
  gap 1.81e-13）退化为 iteration-0 `NumericalFailure`。Expanded 同时关闭
  bordered 的 RSOC failure，并改变已失败 SOC 的退出类型，净结果 287/9，
  bordered 基线 290/6，故 **bordered 保持默认**；
* ✅ Ruiz 已以 `equilibration=:ruiz` 接线但保持 `:off` 默认：mixed probe
  cond(A) 2.0→√2 且 18/18 迭代等价；CFT probe cond(A) 2.0→10.9293，
  `:off` 12 迭代 certified optimal，`:ruiz` 400 次 IterLimit。

**并行限制**

`src/hsd/*` 只能由一个主 HSD 代理写入。其他代理只能提供 cone adapter、测试或 review，不能同时改 HSD 主循环。

---

# 七、v0.8：Sparse、多精度与硬件性能

> **LA 后端政策（2026-08-28，约束性，适用于 Phase 6/7/8 全部）**：MFLA
> （MultiFloatLinearAlgebra）与 BFLA（BigFloatLinearAlgebra）是项目所有者自研库，
> 是**首选线性代数提供者**。任何阶段不得重复实现它们已覆盖的能力（MF 稠密
> LDL/Cholesky/RRQR/TRSM/GEMM、BigFloat 因子化等）。集成通过既有
> `la_backend.jl` 能力层 + `ext/` 适配；缺失的能力以 provider gap 上报，
> 而不是私写内核。现有资产：`ext/SDPXMultiFloatLinearAlgebraExt.jl`（1272 行，
> capability-fact 驱动 + MFWorkspace 复用）、`ext/SDPXBigFloatLinearAlgebraExt.jl`
> （1034 行，因子化 API 带有限性验证）。

## Phase 6 — Sparse KKT 路线

当前 `sparse_la.jl` 和 `factor_cache/routes/sparse_symbolic_numeric.jl` 的 symbolic/numeric 分离应保留，但要接入新的 `NewtonSystem`。

### v1 最快可行路线

先支持：

```text
SparseReducedSchur
DenseExpandedKKT fallback
```

Sparse Schur 必须满足：

* condition estimate 可接受；
* fill 和 memory 预算可接受；
* unregularized Newton residual 通过；
* 失败后可以切换 expanded route；
* 禁止超过内存预算时静默 densify。

### 完整目标路线

进一步增加：

```text
SparseExpandedQuasidefiniteLDLT
```

它是最终面向大规模 bootstrap 的关键路线，但不必阻塞最初的 HSD-only v1。

### 高精度 sparse provider 路线

高精度稀疏分解不进入 MFLA/BFLA 内部，也不在 SDPX 重写。冻结 HSD exact expanded operator 的 `(x,tau)` 块为 skew adjoints，reduced Schur 也含 `c-g`/`c+g`，因此 exact Newton solve 确实非对称；对称 companion 只提供 inertia 证据，不能替代 exact solve。

* Float64 exact nonsymmetric sparse solve：继续使用 Julia `SparseArrays`/UMFPACK；
* BigFloat/MultiFloat exact nonsymmetric sparse solve：评估纯 Julia `PureKLU.jl`；
* signed-regularized symmetric quasi-definite companion inertia：独立评估纯 Julia `QDLDL.jl`；
* PureKLU 与 QDLDL 在 SDPX 中职责互补而非重复：前者产生 physical Newton direction，后者提供 inertia/sign 证据；
* 如果硬性限制只能新增一个包，必须保留 PureKLU，因为 exact equation solve 是必需项；此时 sparse inertia capability 必须标为 unsupported，不得宣称完整 expanded robustness；
* `LDLFactorizations.jl` 仅作为 generic no-pivot benchmark reference，不作为运行时依赖；
* provider adapter 直接调用包，不重新引入 `LinearSolve`/`SciMLBase`；
* BigFloat、Float64x2、Float64x4 必须保持原 scalar type，并通过 unregularized Newton residual 与原坐标证书。

完整决策、实测兼容性和 promotion gates：
`docs/design/HIGH_PRECISION_SPARSE_PROVIDERS.md`。

**文件**

```text
src/sparse_la.jl
src/factor_cache/routes/sparse_symbolic_numeric.jl
src/kkt/reduced_schur.jl
src/kkt/expanded_quasidefinite.jl
src/la/sparse.jl
src/program/route_plan.jl
```

**验收**

* symbolic analysis setup 一次；
* numeric factor 每 epoch 一次；
* pattern 不漂移；
* memory estimate 与实际 peak RSS 接近；
* sparse route 不形成全局 dense `A` 或 dense null basis；
* route fallback 保留同一 HSD iterate；
* deterministic ordering。

**Phase 6 增量 2 closeout（2026-08-28）**

* ✅ `kkt_route=:sparse_schur` 已接入 native product-cone solve loop，仍为
  opt-in；默认保持 `:bordered`；
* ✅ predictor/corrector 共享一次 sparse numeric factor，每个 session 只冻结
  一次结构 CSC pattern，后续 epoch 只更新 `nzval` 与 RHS；
* ✅ Julia stdlib `SparseArrays.lu` 不公开独立 symbolic/numeric API，因此
  `symbolic_reuse_supported=false`；项目只声明真实的 pattern/buffer reuse，
  不伪称复用了 UMFPACK symbolic analysis；
* ✅ factor/refinement/condition gate 失败时，在同一 HSD iterate 执行
  `:sparse_schur → :expanded → :bordered`，不重启、不触碰证书权威；
* ✅ stdlib Float32 sparse LU 会隐式转换 Float64，BigFloat/MultiFloat 无原生
  sparse LU；故 sparse factor 仅对 Float64 开放，其余精度 fail-closed 到显式
  expanded/bordered capability 路线，绝不 downcast；
* ✅ P1 review closeout：sparse 路线的 cone linearization、inverse、augmented
  scratch 全部按真实 cone block 分配；最大单块 dense workspace 为
  `O(max_block²)`，无 `m×m H` 或 global inverse；
* ✅ sparse factor 带 numeric epoch + source-pattern signature authority；任何
  singular/condition/exception/stale 出口清空 factor，RHS reuse 前重验签名；
* ✅ 单一 block-range validator 在 construction 和 NewtonSystem use boundary
  拒绝 gap、overlap、越界、维度不符和 pattern drift；
* ✅ public plan/diagnostics 以 typed `NativeHSDKKTDescriptor` 分别报告 planned
  与 executed route/storage/backend/factorization/provider，fallback 后不得继续
  冒充 sparse execution；`attempted_kkt_routes` / `executed_fallback_chain`
  保留完整尝试序列（direct、sparse→expanded、sparse→expanded→bordered），
  `ExecutionPlan.parameters.factorization_reuse` 取 route descriptor 而非 dense
  mathematical descriptor；
* ✅ LP、SOCP、rank-one PSD、mixed PSD、bounded Nonpositive、dependent/
  ill-scaled equality 的公共解与原坐标证书已进入 `phase6_sparse_integration`，
  且该文件在 quick/full 标准 profile 中各恰好执行一次；
* ⏳ `SparseExpandedQuasidefiniteLDLT`、provider-owned sparse symbolic/numeric
  分离、fill/RSS budget 仍属于 Phase 6 后续完整目标，不由本增量伪装完成。

**周期**

15–25 天。

---

## Phase 7 — MultiFloat 和 BigFloat 深度支持

### 统一 Provider 协议

```julia
struct ProviderCapabilities
    dense_spd::Bool
    dense_indefinite::Bool
    sparse_spd::Bool
    sparse_indefinite::Bool
    multi_rhs::Bool
    iterative_refinement::Bool
    threaded::Bool
    allocation_free_fixed_width::Bool
end
```

Provider：

```text
StandardLAProvider
MultiFloatLAProvider
BigFloatLAProvider
LinearSolveAdapter（临时兼容面，Phase 7 closeout 退役候选）
```

`LinearSolve` 只能适配 factor/solve，不应接管：

* KKT regularization policy；
* inertia policy；
* certificate；
* HSD recovery。

### 依赖精简门（Phase 7 closeout）

依赖退役不能与数值修改混合，必须单独提交并运行 provider/MOI/quick gate：

1. 删除未使用的 test extra `Downloads`；
2. `SHA` 保留为 runtime dependency，但删除 `[extras]`/test target 中的重复声明；
3. MFLA v0.3 / BFLA v0.2 最新适配门通过后，删除不进入生产热路径且历史 A/B 首次分解显著更慢的 `LinearSolve` / `SciMLBase` / `SDPXLinearSolveExt`；
4. MFLA/BFLA 缺失路径已明确 fail closed 后，评估并删除仅作 reference-role 的 `GenericLinearAlgebra` weakdep/extension；
5. 保留 `MultiFloats`、`MultiFloatLinearAlgebra`、`BigFloatLinearAlgebra` 与 macOS 可选的 `AppleAccelerate`；
6. 保留 `JLD2`、`Serialization`、`SHA`，因为长时间集群任务需要 checkpoint、结果导出和 prepared-model 指纹；
7. weakdep 删除的目标是减少安装树、API 和维护表面，不得冒充数值热路径加速。

### MultiFloat 优化顺序

只优化 profile 证明最热的内核：

1. KKT/Schur block assembly；
2. `syrk!`；
3. `gemm!`；
4. `trsm!`；
5. dense Cholesky/LDLT；
6. PSD congruence；
7. residual dot/reduction。

不要为每个精度复制一套 solver。通过 dispatch 专门化 LA kernel。

### BigFloat 规则

* precision bits 在 setup 冻结；
* 不在并行 task 中反复调用全局精度变更；
* 所有 BigFloat destination 自有 storage；
* Julia heap allocation 目标为零或固定上界；
* MPFR-native allocation 单独记录；
* precision escalation 必须从原始 source coefficients 重建。

**文件**

```text
src/kernels/extended_precision_blas/*
src/kernels/mixed_precision_kkt.jl
src/kernels/bigfloat.jl
src/la_backend.jl
ext/SDPXMultiFloatLinearAlgebraExt.jl
ext/SDPXBigFloatLinearAlgebraExt.jl
ext/SDPXLinearSolveExt.jl
```

**验收**

* Float64、Float64x4、BigFloat256 使用同一语义测试；
* factor identity；
* direction residual；
* certificate consistency；
* fixed-width HSD warm iteration 0 Julia bytes；
* BigFloat allocation 不随迭代数线性增长；
* provider 缺失时明确 fail closed；
* 不允许 provider 名义选择与实际执行不一致。

**周期**

10–20 天，可与 sparse route 后半程并行。

---

## Phase 8 — Cache-aware assembly 与线程模型

保留 `src/kernels/threaded.jl` 的正确思想：

* deterministic LPT；
* 每个输出 tile 唯一 owner；
* 无 atomics；
* 无锁；
* 固定 reduction order；
* Julia threads 与 BLAS threads 不嵌套过度订阅。

### 组装策略

* 小 cone：直接 local contribution；
* dense PSD block：panelized `G*A_J` + `syrk!`；
* sparse PSD block：sparse–dense contribution；
* fixed-trace Q3：局部 \(2\times2\) elimination；
* Exp/Power：冻结 3×3 block factor 后批量组装。

### `fixq3` 迁移

抽取：

```text
_fixed_trace_q3_active_variables
_fixed_trace_q3_reduction
fixed_trace_q3_local_elimination
```

放到：

```text
src/kkt/specializations/fixed_trace_q3.jl
```

它只能是 KKT contribution，不能拥有：

* 迭代器；
* cold start；
* termination；
* public result。

**验收**

* 一个 factorization/epoch；
* warm iteration fixed-width 零分配；
* 1/2/4/8 threads 结果可复现；
* 不发生 BLAS/Julia oversubscription；
* assembly 和 factorization 分别计时；
* 性能回归使用多个 fresh process，而不是单次 timing。

**周期**

10–20 天。

---

# 八、v0.9：API、MOI 和代码净化

## Phase 9 — 简化 public API

最终删除：

```julia
engine = :legacy
engine = :native_hsd
algorithm = :lp
algorithm = :socp
algorithm = :sdp
```

因为算法只有 HSD。

可以保留的设置应是：

```julia
Settings(
    kkt_route = :auto,
    provider = :auto,
    precision = :fixed,
    presolve = :auto,
    equilibration = :auto,
    sparse = :auto,
    threads = :auto,
    tolerances = ...,
    limits = ...,
)
```

`algorithm` 可以保留为只读诊断标签，但不再决定正确性路径。

## Low-level API

提供绕开 Modeling/MOI 的直接入口：

```julia
program = ConicProgram(
    A,
    b,
    c,
    ProductConeLayout(...);
    objective_constant=zero(T),
)

prepared = prepare(program, settings)
result = solve!(prepared)
```

v1 可以不支持结构更新，但应明确 ownership：

* `copy_data=true`；
* 或 `owned=true`，调用者不得再修改。

## MOI

v1 只需要可靠的 one-shot `copy_to`：

* Reals；
* Zero；
* Nonnegative；
* Nonpositive；
* SOC；
* RSOC；
* PSD；
* Exp；
* Power。

暂不需要：

* incremental delete；
* coefficient mutation；
* quadratic objective；
* callback；
* persistent warm updates。

**验收**

* 对所有声明支持的 function/set 运行适用的 `MOI.Test`；
* `ListOfVariableAttributesSet` 等 introspection 与实际支持一致；
* MOI primal/dual sign 与 direct API 一致；
* 所有 MOI success status 都有原坐标 certificate；
* unsupported feature 给出明确错误。

**周期**

7–10 天。

---

## Phase 10 — 删除所有 legacy 引擎

删除工作必须是独立 merge，不与数值修改混合。

**删除前门槛**

* 新 HSD 是 public 唯一路线；
* 全 cone/status matrix 绿色；
* quick gate `<60 s`；
* full suite 绿色；
* Float64、Float64x4、BigFloat 核心语义一致；
* sparse 和 dense 至少各有一个生产路线；
* 没有 hidden legacy fallback。

**删除后架构检查**

代码搜索必须只找到：

* 一个 `HSDState`；
* 一个 HSD solve loop；
* 一个 `NewtonSystem`；
* 一个 public success promotion 点；
* 一套 NT/Jordan production kernels；
* 一套 KKT factor session。

### Legacy 依赖同步退役

* 删除 `src/solver/interior_point.jl` 后同步删除仅供其迭代表格使用的 `Printf` runtime dependency；
* 删除的是旧 `@printf` 文本日志，不是结果能力；公开结果继续通过 `Result`、`status(result)`、`value(result, variable)`、`primal_objective(result)`、`certificate(result)` 与 `diagnostics(result)` 查询；
* `Base.show(result)`、结构化 benchmark 输出、MOI/JuMP 状态和证书接口必须在删除后继续通过测试；
* 如果需要实时迭代进度，应由统一 HSD diagnostics/callback 或 Julia logging 接口提供，不得为显示重新保留 legacy solver。

**周期**

5–7 天。

---

# 九、v1.0 的最小可信范围与完整终局范围

## 最小可信 v1.0

| 项目        | v1.0 要求                                                               |
| --------- | --------------------------------------------------------------------- |
| 算法        | 一个 product-cone HSD-IPM                                               |
| 锥         | Nonnegative、SOC、PSD、Exp、Power；Nonpositive/RSOC 通过 typed transform     |
| 自由变量/等式   | Reals + ZeroCone                                                      |
| KKT       | Dense expanded + Sparse reduced Schur                                 |
| 稳定性       | equilibration、regularization、inertia、refinement、terminal verification |
| 精度        | Float64、Float64x4、BigFloat                                            |
| 证书        | 原坐标 optimal / primal infeasible / dual infeasible                     |
| API       | direct Model、Low-level API、one-shot MOI                               |
| 性能        | fixed-width warm iteration 零 Julia allocation                         |
| 旧引擎       | 全部删除                                                                  |
| benchmark | 通用 conic correctness + 独立 application benchmark                       |

## 不阻塞 v1.0 的项目

这些进入 v1.1/v1.2：

* automatic precision escalation；
* sparse expanded indefinite LDL；
* chordal decomposition；
* facial reduction；
* warm starts；
* prepared structural updates；
* Gondzio multi-corrector；
* separate primal/dual step；
* matrix-free KKT；
* GPU；
* distributed assembly；
* full formal proof；
* 多个 sparse provider；
* PSD diagonal congruence equilibration。

## 完整终局

完整愿景还需要：

1. sparse expanded quasidefinite LDL 支持所有核心精度；
2. automatic Float64 → MultiFloat → BigFloat escalation；
3. chordal SDP；
4. advanced presolve/facial reduction；
5. repeated-solve/warm-update；
6. verified MultiFloat accumulation kernels；
7. public reproducible benchmark suite；
8. 面向 bootstrap 的专门 block assembly 和 spectrum/data ingest，但不改变 solver core。

---

# 十、多子代理组织结构

## 1. 固定角色

| 代理                      | 唯一写权限                                    |
| ----------------------- | ---------------------------------------- |
| Lead/Integrator         | `src/SDPX.jl`、共享接口、merge、最终数学决策          |
| Program Agent           | `modeling/`、`ir/`、`program/`             |
| Symmetric Cone Agent    | `cones/symmetric/`、相关 runtime adapter    |
| Nonsymmetric Cone Agent | `cones/nonsymmetric/`、相关 runtime adapter |
| HSD Agent               | `hsd/`，唯一主循环作者                           |
| KKT Agent               | `kkt/`、`factor_cache/`                   |
| LA/Precision Agent      | `la*`、`sparse_la.jl`、`kernels/`、`ext/`   |
| Public/MOI Agent        | `public/`、`moi_wrapper.jl`、frontend      |
| Test/Oracle Agent       | `test/`、`benchmark/`，原则上不改 production    |
| Review Agent            | 只读审查，不提交生产代码                             |

有效并发量建议控制在：

```text
4–6 个实现代理
+ 1 个独立测试代理
+ 1 个只读 reviewer
```

代理数量继续增加，通常只会增加接口冲突和 merge 成本。

## 2. 文件所有权纪律

任何一个 wave 内：

* 两个代理不得同时修改同一个 production 文件；
* `src/SDPX.jl` 只由 Lead 修改；
* frozen canonical/HSD 文档只由 Lead 接受修改；
* 测试代理不得复制 production 算法作为“oracle”；
* HSD Agent 不得自行改 cone 数学；
* Cone Agent 不得决定 solver status；
* LA Agent 不得改变 Newton 方程；
* Public Agent 不得增加隐藏 fallback。

跨层接口变化必须先提交一份小型 RFC：

```text
接口名称
数学语义
ownership
allocation contract
failure contract
需要修改的调用方
测试方法
```

## 3. 每个子代理的任务卡

```text
TASK_ID:
BASE_SHA:
OWNED_FILES:
READ_ONLY_FILES:
GOAL:
NON_GOALS:
FROZEN_INVARIANTS:
EXPECTED_API:
TESTS_TO_ADD:
COMMANDS_TO_RUN:
PERFORMANCE_GATE:
ROLLBACK_BOUNDARY:
DELIVERABLES:
```

每个代理最终必须交付：

* 修改摘要；
* 文件列表；
* 数学公式或接口说明；
* targeted test 结果；
* quick gate 结果；
* allocation 结果；
* 未解决风险；
* commit SHA。

不能只交付“测试通过”。

---

# 十一、推荐的并行 Wave 安排

## Wave A：Program 与数学接口

并行派出：

1. Program Agent：production transform migration；
2. Test Agent：transform composition/ray/property tests；
3. HSD Math Agent：只编写 `NewtonSystem` spec 和 fixture；
4. LA Agent：只设计 `FactorizationSession` capability contract；
5. Review Agent：审查当前 duplicate math 和 deletion map。

合并顺序：

```text
tests/spec
→ transform interface
→ production transform migration
→ architecture gate
```

## Wave B：Dense KKT

并行派出：

1. KKT Agent：expanded assembly；
2. LA Agent：dense LDL + inertia；
3. Refinement Agent：unregularized residual/refinement；
4. Test Agent：manufactured KKT oracle；
5. HSD Agent：仅编写 adapter，不重写主循环。

合并顺序：

```text
NewtonSystem
→ factor session
→ dense expanded assembly
→ regularization
→ refinement
→ HSD adapter
```

## Wave C：Initialization 与统一 HSD

并行派出：

1. Program Agent：equilibration；
2. Presolve Agent：reversible structural presolve；
3. HSD Agent：state machine migration；
4. Cone Agents：initialization/boundary adapters；
5. Test Agent：cone × status matrix。

HSD 主循环仍只能有一个作者。

## Wave D：Sparse 与多精度

并行派出：

1. Sparse Agent；
2. MultiFloat Agent；
3. BigFloat Agent；
4. Threading/Assembly Agent；
5. Provider Test Agent；
6. Cluster Benchmark Agent。

前提是 Phase 3 的 factorization API 已冻结。

## Wave E：API、删除与发布

并行派出：

1. MOI Agent；
2. Low-level API Agent；
3. Documentation Agent；
4. Dead-code Agent；
5. Independent Release Reviewer。

legacy 删除最后合并。

---

# 十二、CI 与验收体系

## 每次 commit

* package load；
* architecture smoke；
* targeted tests；
* `<60 s` quick gate；
* no tolerance change check；
* source SHA 记录。

## 每个 wave

* full test；
* Float64 semantic matrix；
* original-coordinate certificates；
* allocation gate；
* one-factor-per-epoch gate；
* three fresh-process benchmark；
* independent review。

## Nightly/cluster

精度矩阵：

```text
Float64
Float64x2
Float64x4
BigFloat256
BigFloat512
```

线程矩阵：

```text
1 / 2 / 4 / 8 threads
BLAS threads coordinated
```

问题矩阵：

```text
dense / sparse
full rank / rank deficient
interior / boundary
well scaled / ill scaled
single cone / mixed product cone
optimal / primal infeasible / dual infeasible
```

记录：

* setup time；
* assembly time；
* factor time；
* solve/refinement time；
* line-search time；
* iteration count；
* factorization count；
* Julia allocation；
* MPFR allocation proxy；
* peak RSS；
* certificate residual；
* selected route；
* precision；
* source fingerprint。

Bootstrap benchmark 应作为 **application performance gate**，不作为核心数学正确性的唯一依据。

---

# 十三、性能目标的合理分层

## v1.0 必须实现

* Float64/MultiFloat warm HSD iteration 0 Julia bytes；
* 每 iteration 一次 factorization；
* sparse route 不静默 densify；
* HSD hot loop无字符串、闭包和动态 route 分支；
* deterministic block ownership；
* original-coordinate certificate 无额外模型重编译；
* setup、solve、certificate 分开计时。

## v1.1 性能目标

* panelized Schur/KKT assembly；
* fixed-trace Q3；
* sparse/dense PSD block kernel selection；
* provider-specific panel width；
* deterministic LPT threading；
* MultiFloat optimized SYRK/TRSM/LDLT。

## v2.0 竞争目标

“性能统治力”必须由可复现结果定义，而不是写在设计文档中：

* 固定 source SHA；
* 固定硬件；
* 固定线程和 BLAS；
* 相同精度和 certificate 要求；
* 至少三个 fresh processes；
* 同时报告失败和内存超限；
* 不允许为单个 benchmark 写隐藏 route。

---

# 十四、现实工期

在一个开发者负责集成、多个 AI 子代理并行的情况下：

| 目标                                         |        预计日历时间 |
| ------------------------------------------ | ------------: |
| Phase 0–2：Program + NewtonSystem           |         2–3 周 |
| Phase 3–5：robust KKT + unified HSD         |         4–6 周 |
| Phase 6–8：sparse + precision + performance |         4–7 周 |
| Phase 9–10：API + legacy deletion + RC      |         2–3 周 |
| 最小可信 v1.0                                  | **约 10–16 周** |
| 完整成熟目标                                     |  **约 6–9 个月** |

AI 代理可以显著压缩：

* 代码盘点；
* 独立测试；
* provider adapter；
* 文档；
* benchmark harness；
* dead-code removal。

但不能完全并行化：

* `NewtonSystem` 符号冻结；
* HSD 主循环迁移；
* KKT/HSD 集成；
* 数值失败诊断；
* cluster profile 后的路线选择。

这些仍是关键串行路径。

---

# 十五、现在应立即派出的第一批任务

## Agent 1 — Production Transform Migration

**拥有**

```text
src/program/*
src/ir/canonical.jl
src/ir/layout.jl
src/ir/reconstruction.jl
src/ir/lower_*.jl
```

**目标**

* typed transforms 成为生产唯一权威；
* runtime 只见规范化锥；
* optimal/ray/objective 全部经过统一 reconstruction chain。

**禁止**

* 修改 HSD；
* 修改 KKT；
* 修改 public tolerance。

---

## Agent 2 — NewtonSystem Specification

**拥有**

```text
docs/design/NEWTON_SYSTEM.md
test/newton_system_reference.jl
```

**目标**

* 写出五组 Newton 方程；
* predictor/corrector RHS；
* symmetric/nonsymmetric contribution contract；
* hand fixtures；
* Python/BigFloat independent checks。

第一轮只写 spec 和测试，不写 factorization。

---

## Agent 3 — Dense LDL Factorization Session

**拥有**

```text
src/factor_cache/*
src/kkt_route.jl
src/kkt_formulations/dense_augmented.jl
新 src/la/factorization_session.jl
```

**目标**

* one-factor/multi-RHS；
* signed regularization；
* inertia；
* unregularized refinement；
* fail-closed diagnostics。

**禁止**

* 自行推导 HSD 符号；
* 修改 cone scaling。

---

## Agent 4 — Cone Interface Audit

**只读审查**

```text
cones/symmetric/*
cones/nonsymmetric/*
cones/runtime/*
```

**输出**

* 当前每种 cone 已实现的方法表；
* 缺失的统一接口；
* duplicate implementation；
* scratch ownership；
* allocation risk；
* production/reference 划分建议。

暂不大规模改代码。

---

## Agent 5 — Independent QA and Architecture Gates

**拥有**

```text
test/architecture_regressions.jl
test/quick_gate.jl
test/kernel_failure_regressions.jl
新 test/program_transform_composition.jl
```

**目标**

* exact-tree provenance；
* include uniqueness；
* one production transform authority；
* one success promotion authority；
* transform composition；
* cone × status matrix manifest。

测试实现必须独立于生产实现。

---

## Agent 6 — Read-only Integration Reviewer

在前五个代理完成后审查：

* 是否改变 frozen HSD convention；
* 是否存在隐藏 legacy fallback；
* 是否放宽 tolerance；
* 是否出现多个 transform authority；
* 是否出现两个 Newton equation implementation；
* 是否破坏原坐标 certificate；
* 是否增加热路径 allocation。

---

# 最终建议

不要立即同时重写 HSD、sparse KKT、MultiFloat kernel 和 MOI。最快且风险最低的顺序是：

```text
生产 typed transforms
→ 唯一 NewtonSystem
→ dense expanded KKT
→ KKT-derived cold start
→ 统一 HSD 状态机
→ sparse route
→ MultiFloat/BigFloat 性能
→ public/MOI 收敛
→ legacy 一次性删除
```

其中前三步决定数学正确性，第四和第五步决定鲁棒性，第六和第七步决定规模与性能，最后两步决定项目能否真正成为可维护的 v1.0。

这条路线不是简单“继续优化 native HSD”，而是把 SDPX.jl 从一个拥有许多数值技术的实验型代码库，收敛为一个**单一数学内核、多个可替换线性代数实现、原坐标证书权威、能够长期演进的多精度科学求解器**。

[1]: https://arxiv.org/pdf/2505.18791?utm_source=chatgpt.com "Automatic Verification of Floating-Point Accumulation ..."
