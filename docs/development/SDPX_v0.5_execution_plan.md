# SDPX.jl v0.5 Codex 统一执行主计划

> **用途**：把六份只读技术审阅整合为一份可以直接交给 Codex 执行的主任务文件。  
> **目标**：在不重写现有求解器、不弱化数值合同、不引入隐藏 fallback 的前提下，依次完成数值正确性闭环、单一执行计划权威、provider 集成优化、可信 benchmark、低风险维护性整理与 v0.5 发布验证。  
> **性质**：这是执行规范，不是建议清单。Codex 必须按依赖顺序工作，每项先建立失败测试或可复现证据，再修改实现，再运行规定的 gate。  
> **固定审阅基线**：`SDPX.jl main@eca719c62820fd7bf6c3317245af9f78978898f3`。  
> **审阅时 provider 基线**：
> - `MultiFloatLinearAlgebra.jl@2b93685c773db323e9397e448063925dc6cb96fe`
> - `BigFloatLinearAlgebra.jl@b2aed396718d03c60faf017388c3e2fc30934513`

本计划综合以下输入报告，并以它们的共同不变量为准，而不是把各自的任务列表直接串联：

```text
SDPX_main_execution_architecture_review_eca719c6.md
SDPX_main_numerical_correctness_audit_eca719c.md
SDPX_maintenance_review_eca719c6.md
SDPX_MFLA_BFLA_interoperability_audit_2026-08-16.md
SDPX_performance_architecture_review_eca719c.md
SDPX_v0.5_release_readiness_audit.md
```

冲突处理原则：数值正确性高于性能；original-coordinate certification 高于内部 success；显式 plan authority 高于 late binding；当前 SHA 的 end-to-end evidence 高于历史 benchmark；低风险 cleanup 必须等待数学与架构合同稳定。

---

# 0. Codex 角色与工作方式

你是 SDPX.jl 的数值软件执行工程师。你必须同时遵守以下职责边界：

```text
SDPX owns
  mathematical equations
  rank and inconsistency policy
  formulation/provider/fallback authorization
  structured residual and refinement policy
  trial-step acceptance semantics
  reconstruction and original-coordinate certification

MFLA/BFLA own
  public linear-algebra capabilities
  factorization and solve primitives
  factor/workspace lifetime
  public factor metadata and diagnostics
  lower-authoritative kernels
  one correction-solve primitive
```

建议的 agent 分工：

```text
SOL
  architecture decisions, contract review, final merge/release review

Luna
  Newton/KKT mathematics, equality rank, refinement, certification

DSPro / Flash
  regression fixtures, CI, benchmark instrumentation, scripts,
  low-risk cleanup and artifact collection

Escalation
  Flash -> Luna: numerical or semantic uncertainty
  Luna  -> SOL: architecture or cross-family contract uncertainty
  Flash -> SOL: final test/benchmark acceptance package
```

不要把 agent 分工变成额外架构。每个任务仍必须有一个明确 owner、一个独立 commit、一个可复现验收结果。

---

# 1. 执行前硬约束

## 1.1 HEAD 漂移检查

开始前运行：

```bash
set -euo pipefail

export REVIEW_BASE_SHA="eca719c62820fd7bf6c3317245af9f78978898f3"
git fetch --prune origin

git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
```

若 `HEAD` 或 `origin/main` 不等于固定基线：

1. 不要把旧报告中的行号机械套到新代码；
2. 生成 `HEAD_DRIFT_REVIEW.md`；
3. 记录：
   - 新完整 SHA、tree SHA、commit date；
   - `git diff --name-status $REVIEW_BASE_SHA..HEAD`；
   - 本计划涉及文件中哪些发生变化；
   - 每个受影响任务仍成立、已修复、变形或失效；
4. 只在完成 drift review 后继续；
5. 后续所有源码链接和证据改用新固定 SHA。

不得通过 reset、force checkout 或覆盖用户未提交改动来“恢复基线”。

## 1.2 分支与提交

- 从干净基线创建独立工作分支，例如：

```bash
git switch -c codex/v05-hardening
```

- 一个任务一个 commit；大任务允许“测试 commit + 实现 commit”，但不要把多个数学合同揉成一个提交。
- 不允许把 correctness、performance 和 cleanup 混在同一个 commit。
- 每个 commit 必须独立可测试、独立可回滚。
- 未经用户明确授权，不创建 release tag，不发布 GitHub Release，不 force-push，不改写历史。
- MFLA、BFLA 若需修改，必须各自独立 branch、独立 commit、独立 CI；不能先让 SDPX 引用尚未落地的 provider API。

## 1.3 证据状态

每个任务只能报告以下状态之一：

| 状态 | 含义 |
|---|---|
| `PASS` | 所有要求的测试、语义和证据均完成 |
| `FAIL` | 完整环境中出现代码、数值、证书或路径错误 |
| `BLOCKED_<reason>` | 外部环境、私有输入、权限、provider 仓库或集群缺失 |
| `SKIPPED_OPTIONAL_UNAVAILABLE` | 可选 provider 不可用，且该能力不在本次声明范围 |
| `SKIPPED_OUT_OF_SCOPE` | 只有在本文件明确允许缩小发布范围时使用 |
| `MORE_DATA` | 只有 profile-gated 性能任务可使用；不能计为完成 |

`BLOCKED`、`SKIPPED`、`MORE_DATA` 均不能计为 `PASS`。

## 1.4 不可谈判的数值与架构不变量

以下项目在所有提交中必须保持：

1. DenseNormalEquations 与 DenseAugmentedKKT 实现同一 Newton 方程。
2. predictor、corrector 和 structured refinement 复用同一 retained factor；除显式 regularization retry 外，每 outer iteration 不重复 factorization。
3. residual 验证针对未正则化的 structured KKT operator。
4. Cholesky success 不等于 requested-accuracy numerical full rank。
5. equality RRQR 只能由 plan 明确授权；`equality_solver=:normal_equations` 不得隐式转 QR。
6. regularization retry 是同一 formulation 内的数值 retry，不是 route switch。
7. mixed-precision fallback 只能走 plan 中记录的 fallback chain。
8. explicit sparse/provider/formulation 请求不支持时 fail closed，不得静默转 dense/Standard/另一 provider。
9. NativeSOC 失败不得偷偷 PSD lift。
10. lower triangle 是 Schur、Gram、LDLT/Cholesky 和 provider seam 的 authoritative storage；inactive upper 不进入数值决策。
11. BigFloat 必须保留 explicit precision 与 mutable scalar ownership；不能依赖 ambient precision 或共享 MPFR storage。
12. factor 借用 destructive input storage 时，覆盖 buffer 前必须释放旧 handle；residual 不得从已覆盖的 factor buffer读取原 operator。
13. provider 不拥有 SDPX rank、fallback、refinement、termination 或 certification policy。
14. reconstruction 后的 original-coordinate certification 是最终成功状态的权威边界。
15. 不通过放宽 tolerance、减少测试、改变 benchmark 输入或关闭 certification 来“修复”失败。

---

# 2. 本计划的范围决策

## 2.1 v0.5 的执行架构声明

本计划采用完整目标：

```text
Frontend
→ preprocessing / presolve
→ ProblemFeatures / AutoPlanner
→ one top-level ExecutionPlan
→ exact attempt / storage / provider authority
→ Workspace
→ KKT formulation
→ LA provider factor/solve
→ structured refinement
→ original-coordinate reconstruction
→ certification
```

因此：

- `NativeSOCPlan` 可以保留为 typed payload，但不能继续作为与顶层 `ExecutionPlan` 平行的公开生产规划权威；
- LP 的最终 formulation/storage/provider 必须在进入 `LPWorkspace` 前冻结到 plan；
- BigFloat working-precision ladder 必须成为 first-class attempt plan；
- sparse factor provider 与 exact storage microplan 必须可由 plan/diagnostics 一致表达。

如果项目最终决定只发布“SDPProblem 主路线统一”的较窄声明，必须在 README、CHANGELOG 和 release note 中明确缩小范围；不能一边保留平行 planner，一边声明全仓库只有一个执行计划。

## 2.2 `Optimal` 与 `certification=false`

本计划默认采用强语义：

```text
status == Optimal
=> minimal original-coordinate success gate passed
```

`certification=false` 可以关闭详细 certificate payload、昂贵诊断或二次报告，但不应允许一个已知不满足原坐标基本残差/锥约束的 raw `Optimal` 保持为 `Optimal`。

若源码、已发布文档或 downstream compatibility 证明项目故意把 `certification=false` 定义为“完全信任 core raw status”，则不要暗改合同；改为：

- 明确文档写出 conditional semantics；
- `SolveStatus.Optimal` 注释不再声称无条件 verified；
- 保留显式 regression；
- 在发布说明中列为 expert opt-out。

无论选择哪条路径，都禁止通过 tolerance 放宽解决。

## 2.3 BigFloat 512-bit 发布范围

- 若 README/API 表仍把 BigFloat 512-bit 当作 v0.5 正式支持：必须添加 current-candidate 端到端求解与 original-coordinate certificate gate。
- 若无法完成该 gate：把 512-bit 明确降为 experimental/not release-gated，不得继续作为已验证正式承诺。

## 2.4 Provider 修改范围

默认只修改 SDPX。

- MFLA：允许独立增加真实 `JULIA_NUM_THREADS=1` CI leg。
- BFLA：默认不改 production API。
- 只有 SDPX profile 证明 inertia/diagnostics scratch 成为代表性端到端热点时，才允许提出通用、provider-owned、workspace-aware inertia API；必须先在 BFLA 独立实现、测试、version/push/CI，再更新 SDPX compat 与固定 SHA。

---

# 3. 总依赖图与执行顺序

```text
B0  Baseline + immutable evidence snapshot
 |
 +--> N1 true SDP trial residual ----------------------+
 |                                                     |
 +--> N3 FixedTrace rank parity                        |
 +--> N4 fixed-refinement rollback                     |
 +--> N5 quarantine sparse-equality dead branch        |
 +--> N2 runtime equality scaling invariance           |
 +--> D1 decide/implement Optimal semantics             |
 |                                                     v
 +--------------------------------------------------> A0 attempt authority schema
                                                        |
                      +---------------------------------+------------------+
                      |                                 |                  |
                      v                                 v                  v
                 A1 precision ladder               A2 LP final plan    A3 NativeSOC top plan
                      |                                 |                  |
                      +---------------------+-----------+------------------+
                                            |
                                       A4 sparse provider plan
                                            |
                                       A5 WorkspaceStoragePlan
                                            |
                                       A6 Prepared reuse diagnostics

A0 --> O0 trustworthy measurement/counter contract
A2 + O0 --> P1 sparse-LP persistent solve buffer
A3 + O0 --> P2 FixedTrace equality-Gram experiment
A6 + O0 --> P3 Prepared structural reuse experiment
A5 + O0 --> P4 Schur partial-replica architecture experiment
N1 + A2 + A3 --> N7 family-specific LP/NativeSOC trial-residual closure

N/A/O stable --> LA1..LA4 provider metadata optimization
N/A/O stable --> M1..M8 maintainability cleanup
all mandatory tasks --> R0..R8 final release validation
```

推荐执行波次：

1. **Wave 0**：基线与证据；
2. **Wave 1**：数值正确性；
3. **Wave 2**：attempt authority 与 observability；
4. **Wave 3**：LP/NativeSOC/precision/sparse/storage/Prepared 统一；
5. **Wave 4**：provider metadata 优化与低风险性能；
6. **Wave 5**：维护性 cleanup；
7. **Wave 6**：完整 release validation、metadata、remote CI、tag 前 GO/NO-GO。

---

# 4. Wave 0 — 基线与不可变证据

## B0.1 建立 evidence root

```bash
set -euo pipefail

export WORK_ROOT="$(git rev-parse --show-toplevel)"
export EVIDENCE_ROOT="${EVIDENCE_ROOT:-$(dirname "$WORK_ROOT")/sdpx-v05-evidence}"
export VALIDATION_ENV="${VALIDATION_ENV:-${TMPDIR:-/tmp}/sdpx-v05-validation}"
export BENCH_ENV="${BENCH_ENV:-${TMPDIR:-/tmp}/sdpx-v05-benchmark}"
export PROVIDER_ENV="${PROVIDER_ENV:-${TMPDIR:-/tmp}/sdpx-v05-provider}"

mkdir -p "$EVIDENCE_ROOT" "$VALIDATION_ENV" "$BENCH_ENV" "$PROVIDER_ENV"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$EVIDENCE_ROOT/start-utc.txt"

git status --short --branch > "$EVIDENCE_ROOT/git-status-initial.txt"
git show --no-patch --format=fuller HEAD > "$EVIDENCE_ROOT/base-commit.txt"
git rev-parse HEAD > "$EVIDENCE_ROOT/base-head.txt"
git rev-parse 'HEAD^{tree}' > "$EVIDENCE_ROOT/base-tree.txt"
git diff --check > "$EVIDENCE_ROOT/git-diff-check.txt"
```

## B0.2 记录环境

至少记录：

- Julia 1.10 exact version；
- intended latest-stable Julia exact version；
- OS、CPU model、physical/logical cores；
- Julia threads、BLAS vendor/threads；
- Project/Manifest hashes；
- SDPX/MFLA/BFLA SHA；
- provider availability；
- CPU affinity/cpuset/NUMA；
- memory limit/available memory。

## B0.3 基线测试

在首次代码修改前运行：

```bash
SDPX_TEST_PROFILE=quick JULIA_NUM_THREADS=4 \
  julia --startup-file=no --project=. \
  -e 'using Pkg; Pkg.test()' \
  2>&1 | tee "$EVIDENCE_ROOT/baseline-quick.log"
```

若环境允许，再运行当前基线 full 4-thread。任何基线失败必须先分类：已有失败、环境失败或代码 drift；不得把已有失败归因于新修改。

## B0.4 基线输出

创建 `EVIDENCE_ROOT/BASELINE.md`，包含：

- fixed SHA/tree；
- clean/dirty 状态；
- test pass/fail/broken/skip 计数；
- provider pin；
- 未能运行的 gate 及原因；
- 当前已知 release decision：`NO-GO`，直到本计划的 required gates 闭合。

**验收**：`PASS` 只表示基线被可靠记录，不表示代码已发布就绪。

---

# 5. Wave 1 — 数值正确性闭环

## N1 — 用真实 structured residual 闭合 SDP direction → trial acceptance → controller

### 目标

修复当前 cone line search 接受方向后，按理想精确 Newton 公式更新残差的问题。

定义：

\[
\rho_r = r-(S\,dx-B\,dy),\qquad
\rho_p = p-B^Tdx.
\]

接受步长 `tX`, `tY` 后必须使用：

\[
p^+ = (1-t_X)p+t_X\rho_p,
\]

\[
d^+ = (1-t_Y)d-t_Y\rho_r.
\]

### 先写失败测试

必须包含：

\[
p=1,\quad t_X=1,\quad \rho_p=1,
\]

证明旧 carry 公式得到 0，而真实 trial equality residual 为 1。

至少测试：

1. nonzero `rho_p` + full primal step；
2. nonzero `rho_r` + full dual step；
3. structured residual 超过现有 accuracy/refinement 派生 gate 时，不得进入 accepted update；
4. `rho_r=rho_p=0` 时 trajectory 与旧代码在 roundoff 内一致；
5. controller 接收真实 `primal_residual_after` / `dual_residual_after`；
6. line search 继续只负责 cone geometry，不在每个 backtracking trial重算完整 SDP contraction。

### Owned files

```text
src/kkt.jl
src/solver/interior_point.jl
src/adaptive_parameters.jl       # 仅在 after-residual API 需要时
src/step.jl                      # 若 residual helper/step boundary 位于此处
test/kkt_regressions.jl
test/solver_regressions.jl
test/adaptive_parameter_policy.jl
```

### 实现边界

- 复用现有 `_kkt_direction_residual!` 或等价 structured residual；
- operator 必须是未正则化 KKT；
- 不新增 factorization；
- 不改 Schur/KKT assembly；
- 不改 provider、formulation、precision、parameter constants；
- 新 gate 的阈值从现有 arithmetic/requested accuracy/refinement 合同派生，不增加随意 magic tolerance；
- final original-coordinate certification 保留，不能由新 gate 替代。

### 覆盖矩阵

- Float64；
- Float64x4；
- BigFloat 128/256；
- DenseNormalEquations；
- DenseAugmentedKKT（provider 可用时）。

### KEEP / REVERT

KEEP：

- failing fixture 修复；
- exact-direction control trajectory 不变；
- no extra O(n^3) work；
- status/certificate 正确；
- controller history 与真实 trial residual 一致。

REVERT：

- well-conditioned case 迭代明显漂移且无数值解释；
- 新增 factorization；
- route/provider 改变；
- 通过放宽 tolerance 才能通过。

### 建议 commit

```text
test: expose inexact SDP trial-residual carry
fix: validate and record true SDP trial residuals
```

---

## N2 — Runtime equality equation-scaling invariance

### 目标

对 `presolve=false`、direct Workspace/KKT 或其他未经过 normalized equality presolve 的 dense normal-equation runtime route，保证非零 diagonal equality rescaling 不改变 rank semantics 和原坐标解。

反例：

\[
S=I_2,\qquad B(\varepsilon)=\operatorname{diag}(1,\varepsilon),
\]

第二个 equality 与 RHS 同步缩放。

采用 congruence map：

\[
\widehat B=\widetilde B D,\quad
\widehat Q=DQD,\quad
\widehat p=Dp,\quad
 dy=Dd\widehat y,
\]

refinement：

\[
\widehat\rho_p=D\rho_p,\qquad
\delta y=D\delta\widehat y.
\]

### Owned files

```text
src/kkt.jl
src/workspace.jl
test/kkt_regressions.jl
test/mfla_backend.jl
test/bfla_backend.jl
```

### 实现边界

- 建立一个共享 equality scaling helper/map；
- zero column 继续 exact special case；
- 不用 epsilon clipping 把零列变成非零；
- 不改 presolve 中原 arithmetic relation validation；
- multiplier recovery 必须映回 original coordinates；
- 不读取 provider private factor fields；
- normal/augmented 的原方程解保持一致，但不要为了“统一”强迫 augmented 改成不必要的 normal-equation representation。

### 必测

1. `epsilon=1` 与极小非零 `epsilon` rank/solution/certificate 等价；
2. exact duplicate 仍 rank deficient；
3. zero equality + tiny nonzero RHS 仍 inconsistent/fail closed；
4. recovered `dy` 满足 original equations；
5. refinement correction正确映回；
6. Float64 / Float64x4 / BigFloat128/256；
7. Standard/MFLA/BFLA 只通过 public API。

### REVERT

- zero-column semantics 变化；
- multiplier reconstruction 错；
- unit-scaled control 精度退化；
- normalization触发 hidden formulation switch。

### 建议 commit

```text
test: require runtime equality scaling invariance
fix: solve equality Schur systems in normalized coordinates
```

---

## N3 — FixedTraceQ3 equality rank parity

### 目标

FixedTraceQ3 successful Cholesky 不能独立作为 full-rank proof；与 GeneralLorentz 一样，必须调用现有 numerical-rank gate，并仅在 plan 授权时 RRQR。

### Owned files

```text
src/soc_native.jl
test/soc_native_solver.jl
```

### 失败 fixture

合法 FixedTrace Q3 + nearly dependent equalities：

\[
A_{eq}=\begin{bmatrix}1&0\\1&\varepsilon\end{bmatrix}.
\]

选择 `epsilon` 使 factorization success，但 requested-accuracy rank gate 应拒绝。

### 必须保持

- 复用 `_la_factor_has_numerical_rank`；
- 复用 plan-authorized `_factor_equality_qr`；
- 不复制 threshold；
- auto 可按计划 RRQR；forced normal equations fail closed；
- specialization、provider 授权、original-Lorentz certificate 不变。

### 覆盖

Float64、MultiFloat provider route、BigFloat；general/fixed verdict parity。

### 建议 commit

```text
fix: apply numerical-rank gate to FixedTrace equality factors
```

---

## N4 — Fixed-count refinement 的 worsening/nonfinite rollback

### 目标

`refine_policy=:fixed` 固定 correction budget，但不能授权保留被 structured residual 明确证明更坏的方向。

\[
\|\rho_{accepted}\|_\infty\leq\|\rho_{previous}\|_\infty.
\]

### Owned files

```text
src/kkt.jl
test/kkt_regressions.jl
```

### 必测

- 复用 adaptive rollback fixture；
- `refine_policy=:fixed`, `refine_steps=1`；
- worsening → rollback；
- nonfinite → rollback；
- improvement → keep；
- zero steps unchanged；
- DenseNormal / DenseAugmented；
- Float64 / Float64x4 / BigFloat。

### 实现边界

使用现有 snapshot scratch、同一 factor、同一 unregularized structured residual；不改 tolerance/provider，不新增 factorization。

### 建议 commit

```text
fix: roll back worsening fixed refinement corrections
```

---

## N5 — 隔离不可达 sparse-SDP equality double-factor branch

### 目标

当前 sparse Schur public route 要求 `n==0`。删除或 hard-assert equality-bearing internal branch，避免未来误入 double-Cholesky，但不要借机实现 sparse equality KKT。

### Owned files

```text
src/kkt.jl
src/workspace.jl
test/sparse.jl
test/bigfloat_sparse_schur_regressions.jl
```

### 必测

- 直接构造 equality-bearing sparse-Schur factor path 时 deterministic fail closed；
- 不自动转 dense；
- 不切 provider；
- Float64/MultiFloat/BigFloat route invariant；
- failure 不产生 success certificate。

### Stop rule

若任务扩张为 sparse saddle-point、sparse LDLT 或 sparse equality solver，立即停止并另开架构项目。

### 建议 commit

```text
fix: fail closed on equality-bearing sparse Schur factors
```

---

## D1 — 固化 `Optimal` / certification contract

### 默认决策

实现强语义：即使 `certification=false`，也执行 minimal original-coordinate status gate；详细 certificate payload 可关闭。

### Owned files

```text
src/types.jl
src/validation.jl
src/solver/interior_point.jl
src/frontend/high_level_solve.jl
test/result_certificate.jl
test/soc_native_solver.jl
```

### 必测

- forged invalid raw `Optimal` + `certification=false` 被 downgrade；
- diagnostics 可标 detailed certificate unavailable；
- valid Optimal 保持；
- Float64/MultiFloat/BigFloat；
- SDP 与 NativeSOC；
- provider/formulation independent。

若 compatibility 审核证明必须保留 raw semantics，则改为 docs-only contract clarification，并在执行报告中写明未采用强语义的证据。

### 建议 commit

```text
fix: preserve minimal original-coordinate gate when certification is disabled
```

---

## N7 — 把 validated-direction 原则扩展到 LP 与 NativeSOC

> **依赖**：N1、A2、A3 完成后执行，避免在 LP/NativeSOC planner 重构前重复修改。

### 目标

共享“accepted direction must be validated”的原则，不共享错误的数学 kernel：

```text
SDP       -> existing structured Schur residual
LP        -> LP-specific primal/dual KKT residual
NativeSOC -> Lorentz NT/HKM linearized residual
```

### Owned files

```text
src/lp_solver.jl
src/soc_native.jl
src/validation.jl
test/soc_native_solver.jl
LP regression tests
```

### 必测

- regularized LP KKT 的可控 inexact direction；
- General NativeSOC 与 FixedTraceQ3 的可控 inexact direction；
- cone trial 仍 interior，但 residual 不应被记成 exact reduction；
- exact direction trajectory 不变；
- final certificate 保留；
- 不引入 generic dense refinement 黑箱。

### 建议 commits

```text
fix: validate LP directions before accepted updates
fix: validate NativeSOC directions before accepted updates
```

---

# 6. Wave 2 — 统一 execution-attempt authority 与观测合同

## A0 — First-class execution attempt schema

### 目标

建立不改变数值行为的稳定合同：

```text
one execution attempt
→ one immutable plan
→ one exact executed route
→ only explicitly authorized divergence
```

### Owned files

```text
src/types.jl                    # ExecutionPlan, diagnostics/attempt records
src/kkt_backend.jl              # planned/executed assertions
src/solver/interior_point.jl    # attempt result construction
src/performance_trace.jl
test/architecture_regressions.jl
test/executed_diagnostics.jl
test/performance_trace.jl
```

### Required fields

每个 attempt 至少记录：

- attempt id、plan id；
- planned solver family；
- planned formulation/storage/provider/precision/threads；
- executed solver family；
- executed formulation/storage/provider/precision/threads；
- ordered fallback events；
- authorization source；
- cumulative fallback provenance；
- regularization events（与 route switch 区分）；
- final status 与 certificate summary；
- prepared reuse facts。

### 兼容

- 旧 diagnostics 字段保留或通过兼容 property 映射；
- `diagnostics=false` 不应每 iteration 新增动态对象；
- attempt schema 不创建 benchmark-specific fingerprint。

### 必测

1. plan/backend mismatch；
2. unlisted fallback；
3. authorized equality RRQR；
4. authorized mixed fallback；
5. regularization不是 route switch；
6. certification downgrade不是 fallback；
7. planned/executed parity；
8. diagnostics disabled allocation A/B。

### KEEP / REVERT

- solution/iterations/factorization 不变；
- diagnostics disabled stable overhead ≤2%，短 solve allocation 不增加超过2%；
- 旧字段兼容；
- 否则 revert。

### 建议 commits

```text
test: freeze execution-attempt provenance contracts
refactor: add first-class execution attempt records
```

---

## O0 — 可信 timing/counter 与 benchmark 采样合同

> **依赖**：A0。此任务只增加可观测性与 benchmark protocol，不改数值 kernel 或 planner policy。

### 需要补齐的 phase/counter

- cold startup/JIT；
- problem build；
- frontend；
- preprocessing；
- planning；
- Workspace/provider setup；
- residual；
- Schur assembly；
- sparse-SDP factor subphases；
- KKT factor；
- equality panel/Gram/factor；
- predictor/corrector；
- refinement residual 与 correction 分开；
- recovery；
- line search；
- update；
- finalization；
- certification；
- owner task count；
- owner max/mean elapsed；
- effective Julia/BLAS/provider workers；
- allocated bytes、GC seconds、peak RSS、cgroup peak。

### Owned files

```text
benchmark/runner_impl.jl
benchmark/compare_impl.jl
src/performance_trace.jl
src/kkt.jl
src/kkt_backend.jl
src/kernels/threaded.jl
benchmark tests
```

### 采样协议

所有保留的性能结论必须：

1. 每个 thread/arithmetic/provider/candidate 使用独立 Julia process；
2. 冷启动/JIT 单独记录；
3. 一次完整 warmup solve；
4. 至少 3 次，优先 5 次完整 timed solve；
5. 报告 median、minimum、maximum、MAD/relative spread；
6. 记录 Julia/BLAS/provider threads、CPU、NUMA、affinity、Manifest/provider SHA；
7. 每次 solve 都通过相同 status、语义、provider/formulation 与 original-coordinate certificate gate。

### Acceptance

- instrumentation overhead <1%；
- phase closure error <2%；
- `timing=false` 行为与分配基本不变；
- 短 solve allocation 增幅 <2%；
- 任何数值差异立即 revert。

### 建议 commits

```text
test: require phase-closed benchmark records
perf: expose missing KKT and refinement timing counters
bench: add warm repetitions and robust timing summaries
```

---

# 7. Wave 3 — 单一顶层规划权威

## A1 — BigFloat precision ladder 纳入 attempt plan

### 目标

引入 `PrecisionAttemptPlan` 或等价 typed schema，包含：

- requested bits；
- selected bits；
- ordered attempts；
- retry statuses；
- time-budget policy；
- child `ExecutionPlan`；
- attempt status/certificate/provider/formulation/storage summary。

### Owned files

```text
src/solver/interior_point.jl
src/types.jl
test/pipeline.jl
test/prepared_structure.jl
BigFloat/working-precision tests
```

### 必须保持

- bit-selection formula 不变；
- retry status set 不变；
- lower attempt success 不重跑；
- resume 直接使用 requested bits；
- 不跨 precision 复用 factor；
- 不同时保留两套大 Workspace/result；
- 每个返回的 success 都走 original-coordinate certification；
- 不增加第三个中间 precision rung。

### 测试 seam

使用可注入 test seam 强制：

- no ladder；
- lower success；
- retry-eligible failure；
- noneligible failure；
- exhausted time；
- resume；
- two-attempt provenance；
- PreparedSolver partial reuse。

### 覆盖

BigFloat LP、dense SDP、sparse opt-in、block-arrow、BFLA on/off、fixed working precision。

### REVERT

- 同时保留双份大对象；
- fixed policy 多出规划；
- final certificate 变化；
- lower-success 明显退化。

### 建议 commits

```text
test: model BigFloat precision attempts explicitly
refactor: plan and record BigFloat precision ladders
```

---

## A2 — 把 LP final route 冻结到 `ExecutionPlan`

### 目标

在最终 plan 前完成 LP structural finalization：

- row structural presolve 与 row map；
- standard-form detection；
- sparse ingress；
- sparse assembled-pattern probe；
- final equality count；
- provider availability。

Plan 必须准确表达：

```text
:diagonal_reduced_cholesky
:sparse_normal
:positive_definite_cholesky
:dense_lu
```

`solve_lp!` 只 instantiate、assert、execute，不再拥有第二次 route planning。

### Owned files

```text
src/pipeline.jl
src/lp_solver.jl
src/lp_sparse.jl
src/kkt_sparse_backend.jl
src/types.jl
LP diagnostics/tests
```

### 必须保持

- LP scaling 位置；
- parameter resolver 在 scaling 后只解析一次；
- regularization policy；
- reconstruction maps；
- unsupported sparse equality fail closed；
- factor failure不重选 route；
- current actual route 在没有明确修复理由时不改变。

### 禁止

- 重写 LP predictor-corrector；
- 实现 sparse indefinite KKT；
- 改 sparse crossover threshold；
- 改 standard-form math/ordering。

### 必测

- dense equality-free；
- dense equality；
- diagonal standard form；
- Float64 sparse；
- BigFloat sparse；
- explicit sparse equality failure；
- auto sparse rejected to planned dense；
- factor failure same route；
- Prepared LP reuse；
- planned route == executed route。

### 性能 gate

稳定 end-to-end 不得退化 >5%；pattern probe 只能执行一次；setup/allocations 记录。

### 建议 commits

```text
test: require finalized LP routes in ExecutionPlan
refactor: freeze LP backend selection before workspace setup
```

---

## A3 — NativeSOC 纳入顶层 `ExecutionPlan`

### 目标

生产调用链变为：

```text
ConicProblem
→ feature collection / AutoPlanner
→ ExecutionPlan with NativeSOCPlan payload
→ NativeSOCWorkspace
→ NativeSOC core
```

保留 `NativeSOCPlan` 和 `NativeSOCWorkspace` 的数值结构；只消除平行顶层 planner authority。

### Owned files

```text
src/frontend/high_level_solve.jl
src/soc_native.jl
src/types.jl
src/midend/*                      # 仅需要的 feature/route integration
NativeSOC/MOI tests
```

### Workspace 必须 assert

- cone representation；
- FixedTrace specialization；
- formulation；
- provider；
- arithmetic/precision；
- threads；
- equality fallback chain。

### 必须保持

- general Lorentz NT/HKM algebra；
- FixedTrace eligibility；
- cold start；
- factor/RHS reuse；
- normal/augmented route；
- original Lorentz certificate；
- public API。

### 禁止

- 转 PSD lift；
- 删除 FixedTrace；
- 改 scaling；
- 添加 sparse NativeSOC KKT；
- 强行合并 SDP/NativeSOC workspace buffers。

### 必测

GeneralLorentz、FixedTrace auto/off/on、invalid forced FixedTrace、normal、augmented、RRQR、unavailable provider、factor failure no lift、MOI、BigFloat/MultiFloat、planned/executed parity。

### 性能 gate

small SOC frontend 不显著变慢；planning/setup/allocations 单独记录；status/iterations/certificate 不变。

### 建议 commits

```text
test: require top-level plans for NativeSOC execution
refactor: carry NativeSOCPlan as an ExecutionPlan payload
```

---

## A4 — Sparse provider first-class planning

### 目标

顶层 plan 明确记录：

```text
:cholmod
:generic_sparse_cholesky
```

以及 arithmetic、SPD-only、symbolic reuse、ordering、numeric refactorization、no LDLT、factor identity。

### Owned files

```text
src/pipeline.jl
src/la_backend.jl
src/sparse_la.jl
src/kkt.jl
src/types.jl
sparse tests
```

### 必须保持

- sparse formulation 与 equality-free contract；
- frozen CSC/assembly map；
- no dense fallback；
- ordering/fill heuristic不变；
- MFLA/BFLA不接管 sparse route。

### 必测

Float64 CHOLMOD、BigFloat generic、optional MultiFloat、unavailable provider、pattern reuse、no dense fallback、top-level planned/executed provider parity。

### 建议 commit

```text
refactor: make sparse factor providers first-class plan data
```

---

## A5 — 冻结 exact `WorkspaceStoragePlan`

### 目标

在 `ExecutionPlan` 中冻结 exact physical layout：

- memory snapshot；
- assembly mode；
- partial count；
- owner tasks；
- block/Schur bins；
- boundaries；
- lower-only；
- per-block packing；
- mixed storage；
- exact estimated/minimum bytes。

Workspace 只按 plan 分配和验证。容量不足必须 structured fail；不得换 formulation/provider/storage route。

### Owned files

```text
src/types.jl
src/pipeline.jl
src/workspace.jl
src/kernels/threaded.jl
Workspace/extended-precision/block-arrow/sparse tests
```

### 禁止

- 顺便调 crossover 常数；
- benchmark fingerprint；
- expensive full materialization；
- 内存不足时静默改 route。

### 必测

- same plan + same simulated memory => same layout；
- insufficient memory fail closed；
- exact buffer/task counts；
- Prepared reuse；
- BigFloat ownership；
- dense owner；
- sparse coefficient assembly；
- reduced/mixed arrow。

### 性能 gate

stable time 不退化 >5%；planned bytes 与实际峰值的定义清楚；planning scan 不得成为明显热点。

### 建议 commits

```text
test: freeze exact workspace storage decisions
refactor: allocate workspaces from immutable storage plans
```

---

## A6 — PreparedSolver 真实 attempt-level reuse diagnostics

### 目标

每个 attempt 记录：

- preprocess/equality map/plan 是否复用；
- fresh plan；
- fresh Workspace；
- warm start；
- precision mismatch；
- partial reuse 原因。

### Owned files

```text
src/prepared.jl
src/solver/interior_point.jl
src/types.jl
test/prepared_structure.jl
```

### 必须保持

- 不缓存 Workspace/factor/mutable numeric buffers；
- fingerprint 与 RHS relation validation 不放宽；
- FixedTrace value facts 每 solve 重查；
- previous result 只作为显式 warm start；
- cold/prepared result 与 certificate 一致。

### 依赖

A1；最好等待 A2/A3/A5 plan shape 稳定。

### 建议 commit

```text
fix: report actual PreparedSolver reuse per execution attempt
```

---

# 8. Wave 4 — Provider 集成与 source-backed 性能优化

## LA1 — 冻结 LDLT metadata accessor call-count contract

### Owned file

```text
test/dense_augmented_kkt.jl
```

扩展 scripted backend counters：

```text
factor_calls
inertia_calls
blocks_calls
permutation_calls
precision_calls
diagnostics_calls
correction_calls
```

### 必须满足

Accepted candidate：

```text
factor       = 1
inertia      = 1
blocks       = 1
permutation  = 1
precision    = 1
diagnostics  = 0 when diagnostics=false
diagnostics <= 1 when diagnostics=true
```

Retry：每个 candidate 的 inertia 最多一次，只有最终 accepted candidate 保留 metadata。

Terminal rejection：若 diagnostics 开启，必须在 factor buffer 被覆盖前 snapshot，之后不再访问旧 handle。

### 建议 commit

```text
test: count and freeze LDLT metadata extraction
```

---

## LA2 — 在 SDPX provider-neutral wrapper 缓存 validated LDLT metadata

缓存：

- compact blocks；
- final permutation；
- factor precision；
- immutable provider/factor kind identity。

不要缓存：

- solver rank verdict；
- acceptance/fallback；
- requested tolerance；
- certificate state；
- mutable factor-storage derived views。

继续 fail closed 验证：shape、provider identity、precision、compact block grammar、permutation、success state、authoritative triangle。

### Owned files

```text
src/la_backend.jl
src/kkt_formulations/dense_augmented.jl
provider extensions only if wrapper construction requires it
test/dense_augmented_kkt.jl
```

### 建议 commit

```text
perf: cache validated LDLT metadata in SDPX
```

---

## LA3 — Full diagnostics lazy

- `diagnostics=false`：successful factor 不调用 full `factor_diagnostics`；
- `diagnostics=true`：每 candidate 最多一次；
- terminal failure：仅在用户可见诊断确实需要时，在 storage overwrite 前 snapshot；
- inertia acceptance 仍每 candidate 计算一次，不能由 full diagnostics 决定。

### 建议 commit

```text
perf: defer full provider diagnostics to observable cold paths
```

### REVERT

若 cache/lazy diagnostics 没有端到端收益且显著增加复杂度：回滚实现，保留 call-count tests；不要修改 provider 来掩盖。

---

## LA4 — Profile duplicate boundary scans

记录：

```text
provider_factor_seconds
adapter_validation_seconds
metadata_seconds
diagnostics_seconds
solve_seconds
```

代表性 cases：

- BigFloat256/512 moderate augmented KKT；
- Float64x4 augmented KKT；
- accepted candidate；
- two retries；
- terminal failure；
- diagnostics on/off。

只有 representative certified end-to-end solve 显示重复 scan 有实质占比时，才进一步删除被 public provider contract 完全保证的重复验证。

---

## LA5 — MFLA 真实 1-thread CI（独立仓库）

在 MFLA `.github/workflows/ci.yml` 增加 Ubuntu matrix：

```yaml
julia-version: ["1.10", "1.11", "1.12"]
threads: ["1", "4"]
```

并设置/断言：

```yaml
JULIA_NUM_THREADS: ${{ matrix.threads }}
EXPECTED_JULIA_THREADS: ${{ matrix.threads }}
```

不得把 SDPX problem semantics、rank threshold、fallback 或 certificate 放入 MFLA。

建议 commit：

```text
ci: exercise real one-thread and four-thread runtimes
```

---

# 9. Wave 4B — 低风险、证据驱动性能队列

## P1 — Sparse LP persistent solve destination

> **依赖**：A2、O0。

### 目标

`lp_sparse_solve!` 不再为 predictor/corrector 每个 RHS 新分配 solution vector；在 solver-owned sparse LP workspace 中增加一个可复用 destination buffer。

### Owned files

```text
src/lp_sparse.jl
src/lp_solver.jl
sparse LP tests/benchmarks
```

### 不得改变

factor、symbolic structure、RHS formula、provider、one-factor/two-RHS reuse。

### 验收

KEEP 若：

- end-to-end ≥3%，或
- allocation 降低 ≥20%，或 peak RSS 降低 ≥10%，且 wall time regression <1%；
- 所有 representative route regression <2%；
- status/iterations/certificate 不变。

否则回滚实现，保留 allocation regression test（若测试本身通用且低成本）。

建议 commit：

```text
perf: reuse sparse LP solve destination storage
```

---

## P2 — FixedTraceQ3 equality-Gram tuning（先测后改）

> **依赖**：A3、O0。初始状态必须是 `MORE_DATA`。

只有满足以下任一条件才进入 implementation：

```text
equality_gram >= 20% of total solve time
```

或：

```text
equality_gram >= 40% of KKT preparation
```

允许第一轮只调整 worker crossover 或 lower-column/tile partition。

禁止：

- 改 factorization/precision/provider；
- 改 reduction semantics；
- full per-thread Gram replicas；
- 仅凭 isolated SYRK microbenchmark 保留。

测量：Float64、x2/x3/x4、BigFloat256；1/2/5/8/16 threads；J40 全网格，J80 先 memory preflight。

KEEP：J40/J80 任一端到端 >10%，或至少两个代表 case >5%，1-thread regression <2%。

---

## P3 — PreparedSolver frozen structural reuse（实验）

> **依赖**：A6、O0。

允许缓存的只读结构：

- CSC pattern；
- ordering/permutation；
- symbolic factor metadata；
- assembly maps；
- block bins/pair counts/column boundaries。

禁止缓存：

- numeric factor；
- Schur values；
- iterate；
- provider mutable storage；
- unsafe BigFloat scalar arrays。

使用同一结构连续 20 次 objective/RHS solve；分别报告 first solve 与 2–20 amortized。

KEEP：至少两个 sparse family amortized ≥5%，one-shot regression <1%，且 invalidation/fingerprint/precision/provider contract 完整。

---

## P4 — 消除 fixed-width/sparse SDP full Schur partial replicas（独立架构实验）

> **依赖**：A5、O0。只有 profile 证明 task-local `O(p m^2)` partials 显著占用 RSS/time 才启动。

首个 prototype 仅面向 isbits fixed-width arithmetic，不含 BigFloat。

要求：

- exclusive lower-output ownership；
- panel transform once per block；
- inner provider threads = 1；
- no atomics；
- no per-thread full output matrix；
- same status/iterations/certificate。

KEEP：peak RSS 改善 ≥25% 且 time regression <2%，或 end-to-end ≥8%。

REVERT：1-thread regression >3%、NUMA 严重退化、任何数值差异。

---

## P5 — Refinement residual scan 与 BLAS/task crossover（只做 profile-gated 工作）

### Refinement scan

N1 完成后 residual validation 是 correctness invariant，不能为提速直接跳过。只有 residual-evaluation phase ≥5% total，且多数 iteration 没有 useful correction 时，才允许研究狭窄 skip condition。

绝不跳过：mixed precision、tight tolerance、regularized/rank-risk/stalled cases。

### BLAS/task crossover

只有 factor/KKT ≥10% total 或 task/barrier ≥5% total，才建立基于

\[
m^3+m^2e+me^2+e^3
\]

以及 arithmetic/provider 的透明 work model。

禁止 hidden calibration、process-global tuning cache、persistent worker pool、provider/formulation switch。

---

## P6 — 明确拒绝的性能工作

没有新的 current-SHA 端到端证据时，不做：

- MFLA lower-SYRK “race fix”或纯审美 tile rewrite；
- equality panel transpose/copy micro-optimization；
- complementarity scan speculative fusion；
- BigFloat zero/one constructor micro-tuning；
- Task_Low08-specific sparse Cholesky；
- manual short-dot unroll；
- packed-triangular factor API；
- matrix-free PCG for small reduced systems；
- generic BigFloat `Threads.@threads`；
- 只改善一个 microbenchmark 的复杂 kernel。

---

# 10. Wave 5 — 维护性 cleanup（数值/架构稳定后）

## 通用约束

- 不改算法；
- 不改 threshold；
- 不改 fallback；
- 不删 public API；
- 不删 struct field；
- 不重写历史 benchmark report；
- 每项独立 commit；
- cleanup 失败不能通过数值行为变化来“适配”。

## M1 — 把 benchmark-specific source comments 改写为 invariant

建议 commit：

```text
docs: replace benchmark-specific source comments with solver invariants
```

只改 comments/docstrings。具体 Task_Low08、CSDR、J40/J80、某次内存数字留在 benchmark/docs provenance；生产注释改为 dimension、density、memory、ownership、precision、provider、residual invariant。

涉及文件可包括：

```text
src/soc.jl
src/kkt.jl
src/schur.jl
src/types.jl
src/workspace.jl
src/kernels/generic.jl
src/kernels/threaded.jl
src/kernels/mixed_precision_kkt.jl
src/stagnation.jl
src/ingest.jl
src/preprocessing.jl
src/chordal.jl
src/nullspace.jl
src/solve.jl
src/public_api.jl
```

diff 出现 executable token 即 revert。

## M2 — 删除 bound analysis 中恒空的 private fixed-equality filter

建议 commit：

```text
cleanup: remove inert fixed-equality filtering from bound analysis
```

只删除局部 `BitSet`/identity filter。必须保留：

- `FixedEqualityCandidate`；
- `BoundExtractionPlan.fixed_equalities`；
- `ReconstructionMap.fixed_equalities`；
- struct layout 与 reconstruction/Prepared compatibility。

测试：单列 equality 保留、字段仍存在且为空、`keep_equalities==1:n`、dual/prepared/MOI behavior 不变。

## M3 — Deduplicate PerformanceTrace setup projection

建议 commit：

```text
cleanup: deduplicate performance-trace setup projection
```

只让 SDP/Conic setup projection 复用 `_setup_facts_for_record`；不合并语义不同的 iteration projection。snapshot key/order/type/value 与 `Unavailable` semantics。

## M4 — 移动 saturating memory helpers

建议 commit：

```text
refactor: centralize saturating memory accounting
```

把 `saturating_bytes` / `saturating_sum_bytes` 原样移动到 `src/memory_utils.jl`，保持 `SDPX.*` binding、include order、typemax saturation 与所有 caller behavior。

## M5 — Canonical arithmetic tag resolver

先建立 Float32/Float64/BigFloat/x2/x3/x4/其他 width/custom type 的 snapshot，再统一 `_arithmetic_symbol` / `_la_arithmetic_symbol`。

任何 diagnostics/provenance/provider selection/tag 变化即 revert。

## M6 — 共享 high-level requested BigFloat precision scope

helper 必须包住完整：

```text
conversion
resolved options
planning
workspace construction
solve
result construction
```

测试 ambient precision 正常/异常恢复、128/256/512、SDP/NativeSOC、result entry precision、certificate。

## M7 — 只合并 legacy options builder，不删 legacy API

保留 `sdp`、`findFeasible`、`setArithmeticType`、warning text、keyword defaults、thread-local arithmetic、warm-start semantics。

建议 commit：

```text
compat: centralize legacy option construction
```

## M8 — 消除 test include-order coupling

把共享 fixture/assertion 移入显式 helper 文件；`correctness.jl` 与 `sparse.jl` 能独立运行；assertion count 和 full pass/broken count 不下降。

建议 commit：

```text
test: remove historical include-order coupling
```

## M9 — 大文件 physical split（最后、可选）

只有前述任务稳定后才做。每个 commit 只移动定义，不改函数体、method precedence、exports 或 include semantics。优先边界：

```text
src/kkt/{equality_rank,factorization,solve,refinement,diagnostics}.jl
src/schur/{contractions,dense,sparse,arrow,materialize}.jl
src/solver/{initialization,iteration,termination,finalization}.jl
src/lp/{workspace,factorization,iteration,certification}.jl
src/soc_native/{plan,workspace,kkt,iteration,diagnostics}.jl
```

不按 arithmetic 把同一公式拆成难以比较的孤岛。

---

# 11. 每个任务的统一测试 gate

## 11.1 Focused test helper

在 evidence root 建立：

```julia
# run_test_group.jl
root = ENV["SDPX_ROOT"]
for file in ARGS
    println("\n===== ", file, " =====")
    include(joinpath(root, "test", file))
end
```

## 11.2 必跑 correctness groups

### Planner / diagnostics

```text
pipeline.jl
auto_planner.jl
frontend_auto_options.jl
architecture_regressions.jl
executed_diagnostics.jl
performance_trace.jl
v05_core_invariants.jl
```

### LP

```text
lp_regressions.jl
lp_sparse.jl
infeasibility_diagnostics.jl
```

### NativeSOC / FixedTrace

```text
soc_native_algebra.jl
soc_native_solver.jl
fixed_trace_benchmark_regressions.jl
soc_q3_kernel_regressions.jl
moi_native_soc.jl
```

### SDP/KKT

```text
correctness.jl
solver_regressions.jl
kkt_regressions.jl
dense_augmented_kkt.jl
sparse.jl
sparse_sdp_kkt.jl
bigfloat_sparse_schur_regressions.jl
sparse_execution_round6.jl
sparse_schur_round7.jl
result_certificate.jl
```

### Prepared / frontend / MOI

```text
prepared_structure.jl
public_api.jl
options_interface.jl
error_handling.jl
moi.jl
moi_native_soc.jl
```

### Precision/provider

```text
mixed_precision_kkt_regressions.jl
extended_precision_blas.jl
extended_blas_regressions.jl
bigfloat_kernel_regressions.jl
bigfloat_ownership_regressions.jl
mfla_backend.jl
bfla_backend.jl
provider_smoke.jl
multifloat_linear_algebra_integration.jl
```

## 11.3 Arithmetic/formulation/provider matrix

每个涉及数值行为的任务至少评估：

| Dimension | Required values |
|---|---|
| Arithmetic | Float64；Float64x4；BigFloat128/256；x2/x3 或 512 按任务范围 |
| Dense formulation | normal；augmented（可用时） |
| Sparse | CHOLMOD Float64；generic extended；explicit unsupported fail closed |
| SOC | GeneralLorentz；FixedTraceQ3；MOI |
| Provider | core-only；MFLA；BFLA；loaded-incomplete fail closed |
| Threads | 1；4，性能任务另做 1/2/5/8/16 |
| Certification | default enabled；`certification=false` contract fixture |
| Prepared | cold；prepared；objective/RHS replacement；precision mismatch |

## 11.4 每个 commit 的最小 gate

1. `git diff --check`；
2. task-focused tests；
3. quick suite；
4. 若触及数值 kernel/planner/provider/precision/certification：full suite；
5. 若触及 provider：real-provider smoke；
6. 若触及 performance：完整 certified end-to-end A/B；
7. 记录 status、iterations、objective、residual、certificate、planned/executed route。

---

# 12. Wave 6 — v0.5 完整发布验证

## R0 — 创建不可变 validation snapshot

不得在正在编辑的 working tree 上直接生成最终证据。保存 patch/untracked inventory，创建 repository-external clean clone，应用 candidate patch，提交一个本地 validation snapshot，记录 commit/tree，并确保 snapshot clean。

最终 release commit 的 tree 必须与 validated tree 完全相同。

## R1 — Full suite matrix

必须运行：

```bash
run_full_suite () {
  local julia_bin="$1"
  local label="$2"
  local threads="$3"

  SDPX_TEST_PROFILE=full \
  JULIA_NUM_THREADS="$threads" \
  "$julia_bin" \
    --startup-file=no \
    --project="$SDPX_ROOT" \
    -t"$threads" \
    -e 'using Pkg; Pkg.test(; coverage=false)' \
    2>&1 | tee "$EVIDENCE_ROOT/full-${label}-t${threads}.log"
}

run_full_suite "$JULIA_110_BIN"    "julia-1.10" 1
run_full_suite "$JULIA_110_BIN"    "julia-1.10" 4
run_full_suite "$JULIA_LATEST_BIN" "julia-latest" 1
run_full_suite "$JULIA_LATEST_BIN" "julia-latest" 4
```

允许的 broken/skip 必须逐项列出。optional provider 未加载与明确要求多线程的单线程 fixture 可被解释；其他 skip/broken 需要人工 release review。

缺 Julia 1.10 是 `BLOCKED_JULIA_1_10`，release 仍 `NO-GO`。

## R2 — Core-only 与 real-provider environments

### Core-only

不安装 MFLA/BFLA：

- `using SDPX` 成功；
- Float64 LP/SDP 成功；
- Standard/Legacy explicit route 成功；
- explicit `:bfla` / `:multifloat` actionable fail closed。

### Real providers

固定并记录 release candidate provider SHA。审阅基线 pins：

```text
MFLA 2b93685c773db323e9397e448063925dc6cb96fe
BFLA b2aed396718d03c60faf017388c3e2fc30934513
```

在最终执行前重新确认这些 pins 与 SDPX compat/API 一致。使用 `Pkg.develop` local checkout，运行：

- `test/provider_smoke.jl`；
- `test/multifloat_linear_algebra_integration.jl`；
- MFLA x2/x3/x4；
- BFLA 128/256/512（按声明范围）；
- Cholesky/LU/RRQR/LDLT；
- factor lifetime；
- vector/multi-RHS；
- lower-only poison-upper tests；
- NativeSOC normal/augmented；
- original-coordinate certificate。

provider repo unavailable 是 `BLOCKED_PROVIDER_REPO`，不能报告 provider pass。

## R3 — 补齐 release-level executed behavior

必须有真实 end-to-end fixture 证明：

1. complete sparse SDP solve：planned/executed storage/provider、symbolic analysis count、numeric refactor count；
2. general SDP explicit dense-augmented solve：LDLT、inertia、provider、certificate；
3. mixed-precision full solve：实际 active，或明确 static rejection reason；不能只证明 option 被请求；
4. iterative refinement production path；
5. BigFloat512 gate 或正式缩小声明；
6. N1/N7 true trial residual contract；
7. all `Optimal` statuses符合最终 certification contract。

## R4 — Canonical micro / representative / local_full

### Micro

- 所有 generated rows 执行；
- 不允许 unexpected structured skip；
- `semantic_pass=true`；
- shared runner timing 仅记录，不作为 release performance threshold。

### Representative / local_full

运行 canonical runner，并生成 `skipped-inventory.tsv`。

- 每个 non-skipped row 必须 semantic pass；
- 每个 skipped row 必须有结构化原因；
- skip 不计为 pass。

Benchmark result schema 至少包含：

- problem/input fingerprint；
- source commit/tree；
- Project/Manifest SHA；
- provider SHA；
- arithmetic/precision；
- planned/executed route；
- certificate；
- iterations/objective/residual/gap；
- cold/warm samples；
- median/spread；
- allocation/GC/peak RSS；
- threads/affinity/NUMA。

## R5 — Canonical private application holdout

Active application anchor 的 payload/source-model SHA 必须由 registry/loader验证。对 Float64x2 与 Float64x4 分别运行至少 3 个独立 PBS jobs。

每个 job 保留：

- exact SDPX/tree、MFLA SHA、Julia version、Manifest SHA；
- payload SHA、source-model SHA；
- requested/allocated cores；
- Julia/BLAS/provider threads；
- CPU、cpuset/affinity、NUMA；
- memory limit、peak RSS；
- status/objective/residual/gap；
- original-coordinate Lorentz certificate；
- executed `fixed_trace_q3`；
- no PSD lift；
- no unexpected fallback；
- PASSED marker。

PBS launcher 必须 fail closed：

```bash
requested_threads="${SDPX_JULIA_THREADS:-1}"
allocated_threads="${PBS_NP:-1}"

if [ "$requested_threads" -gt "$allocated_threads" ]; then
  echo "requested Julia threads exceed PBS allocation" >&2
  exit 4
fi
```

必须显式请求 memory，并做 preflight/headroom；记录 scheduler accounting。最终报告 x2/x4 的 samples、median、min、max、spread/MAD。

缺私有 payload、PBS 或 MFLA 环境分别报告 `BLOCKED_PRIVATE_PAYLOAD`、`BLOCKED_CLUSTER`、`BLOCKED_MFLA`。只要 application claim 仍在 v0.5 范围，任一 blocker 都保持 release `NO-GO`。

## R6 — CI artifact retention 与 required checks

Release-triggered CI 上传：

- test logs；
- `Pkg.status`；
- Project/Manifest；
- Julia/provider SHAs；
- micro/representative TOML/TSV；
- skipped inventory；
- numerical gates；
- docs build；
- metadata/evidence manifest。

建议 required checks：

```text
Julia 1.10 full t1
Julia 1.10 full t4
latest full t1
latest full t4
macOS quick
Windows quick
provider smoke
docs
quality/Aqua
canonical micro
release metadata check
```

若 Codex 无 GitHub admin 权限，报告 `BLOCKED_GITHUB_ADMIN` 并输出 required-check 名单；不能把分支保护标为完成。

## R7 — Docs 与正式 metadata

只有前述 numerical/CI/application gates 均通过后，才把：

```text
Project.toml        version = "0.5.0"
CITATION.cff        version: 0.5.0
CITATION.cff        date-released: YYYY-MM-DD
CHANGELOG.md        ## [0.5.0] — YYYY-MM-DD
README citation     version 0.5.0
```

同时添加 release evidence manifest：

- release commit/tree；
- CI run IDs；
- Julia versions；
- provider SHAs；
- benchmark payload/source-model SHA；
- application artifacts；
- docs artifact hash；
- release date。

metadata checker 必须拒绝：`-DEV`、`unreleased`、missing date、stale README、missing evidence manifest。

## R8 — Final tree / remote CI / tag

最终步骤：

1. `git diff --check`；
2. working tree clean；
3. final commit tree == validated tree；
4. push candidate branch/commit；
5. 监控所有 required remote jobs；
6. 下载并核对 artifacts；
7. branch protection 生效；
8. 重新确认 provider pins；
9. 生成最终 GO/NO-GO；
10. 只有 GO 后才创建 signed/annotated `v0.5.0` tag 与 GitHub Release。

---

# 13. 性能接受与回滚统一规则

## 13.1 所有性能候选

任何一项出现以下情况立即 REVERT：

- status、certificate、original-coordinate residual 或 objective 语义回归；
- 未解释的 iteration count 变化；
- provider/formulation/storage/precision 暗换；
- 只在 microbenchmark 改善，代表性 end-to-end 无收益；
- 为提速关闭 certification 或缩短 correctness gate。

## 13.2 收益阈值

| 改动类型 | KEEP 门槛 |
|---|---|
| 极简单、通用、低风险 | 3–5% E2E；或显著 allocation/RSS 降低且 time 不退化 |
| 中等复杂度 | 至少两个代表 family ≥5% |
| 高复杂度架构 | ≥8–10% E2E；或 ≥25% peak RSS 且 time regression <2% |
| 单一 benchmark | 默认拒绝，除非它代表正式 release workload 且其他 control 不退化 |
| instrumentation | overhead <1%，phase closure error <2% |

A/B 必须使用相同：input hash、status/certificate、iterations（或解释）、provider/formulation、Julia/BLAS threads、Manifest、hardware/NUMA、warmup/repetitions。

---

# 14. 禁止混入本轮的独立项目

除非另开设计任务，不实现：

- sparse indefinite KKT / general sparse augmented LDLT；
- sparse equality solver；
- HSD Newton embedding；
- chordal decomposition；
- reduced-dual / L-BFGS solver；
- PSD lift removal；
- NativeSOC 数值重写；
- SDPWorkspace 与 NativeSOCWorkspace 全面合并；
- packed-triangular provider factor interface；
- process-wide concurrent solver BLAS-thread ownership重构；
- persistent worker-pool scheduler；
- benchmark-specific production calibration；
- compatibility constructor全面删除。

这些项目不能作为当前 task 的“顺手优化”。

---

# 15. Commit / PR 推荐顺序

## PR 1 — Numerical correctness

```text
1. test: expose inexact SDP trial-residual carry
2. fix: validate and record true SDP trial residuals
3. test/fix: runtime equality scaling invariance
4. fix: apply numerical-rank gate to FixedTrace equality factors
5. fix: roll back worsening fixed refinement corrections
6. fix: fail closed on equality-bearing sparse Schur factors
7. fix/docs: finalize Optimal/certification contract
```

## PR 2 — Attempt authority and observability

```text
1. test: freeze execution-attempt provenance contracts
2. refactor: add first-class execution attempt records
3. perf: expose missing phase counters
4. bench: add warm repetitions and robust summaries
```

## PR 3 — ExecutionPlan authority

推荐拆成多个 PR，避免巨型变更：

```text
A1 BigFloat precision attempts
A2 LP final route
A3 NativeSOC top-level plan
A4 sparse provider plan
A5 WorkspaceStoragePlan
A6 Prepared reuse diagnostics
```

A2 与 A3 可并行开发，但必须在 A0 的公共 schema 上 rebase；不要同时修改同一 diagnostics contract 的不同版本。

## PR 4 — Provider metadata performance

```text
1. test: count and freeze LDLT metadata extraction
2. perf: cache validated LDLT metadata in SDPX
3. perf: defer full provider diagnostics
```

MFLA CI 是独立仓库/独立 PR。BFLA production API 默认无 PR。

## PR 5 — Low-risk performance

优先 sparse LP buffer。其他项必须先有 O0 数据，且每项独立 PR。

## PR 6 — Maintainability

comments、private no-op、projection、memory utility、arithmetic tags、precision scope、legacy builder、test-order coupling均独立 commit；physical split 最后单独 PR。

## PR 7 — Release engineering

CI matrix、artifact retention、PBS provenance、metadata checker可以在代码稳定后合并；正式 `0.5.0` metadata 必须最后提交。

---

# 16. Codex 每轮输出格式

每完成一个 task，输出并写入 evidence 目录：

```markdown
# <TASK-ID> Execution Report

## Baseline
- base SHA:
- candidate SHA:
- changed files:

## Source-backed problem
- exact symbol/call path:
- failing fixture or counter:

## Implementation
- what changed:
- what explicitly did not change:

## Tests
| Command | Result | Pass/Fail/Broken/Skipped |
|---|---|---|

## Numerical parity
- status:
- iterations:
- objective:
- primal/dual/gap:
- certificate:
- planned route:
- executed route:

## Performance / allocation
- protocol:
- samples:
- median/min/MAD:
- allocated bytes / GC / peak RSS:

## Provider and precision
- arithmetic:
- provider SHA:
- factor kind:
- precision:

## Decision
- PASS / FAIL / BLOCKED / MORE_DATA:
- KEEP / REVERT:
- unresolved risks:

## Commit
- hash:
- title:
```

最终总报告必须包含：

1. 所有 task 状态矩阵；
2. 所有 commit 与 tree；
3. 所有 blocked/skipped inventory；
4. current supported arithmetic/provider/formulation matrix；
5. benchmark summary；
6. release GO/NO-GO；
7. 若 NO-GO，剩余最短闭环路径。

---

# 17. 最终 GO / NO-GO 清单

`v0.5.0` 只能在以下全部满足时为 **GO**：

- [ ] N1 true trial residual contract 完成；
- [ ] N2 runtime equality scaling invariance 完成；
- [ ] N3 FixedTrace rank parity 完成；
- [ ] N4 fixed refinement rollback 完成；
- [ ] N5 sparse-equality dead branch fail closed；
- [ ] `Optimal`/certification contract 已明确并测试；
- [ ] A0 first-class attempt provenance；
- [ ] A1 BigFloat precision attempts first-class；
- [ ] A2 LP final route frozen；
- [ ] A3 NativeSOC under top-level ExecutionPlan；
- [ ] A4 sparse provider first-class；
- [ ] A5 exact WorkspaceStoragePlan；
- [ ] A6 Prepared reuse diagnostics真实；
- [ ] N7 LP/NativeSOC validated-direction contract，或正式列为 post-v0.5 且不作相应强声明；
- [ ] provider private fields 未被读取；
- [ ] lower-authoritative/factor lifetime/factor reuse contracts通过；
- [ ] Julia 1.10 full t1/t4；
- [ ] latest full t1/t4；
- [ ] macOS/Windows quick；
- [ ] core-only 与 fixed-SHA provider smoke；
- [ ] complete sparse SDP executed-route fixture；
- [ ] general SDP dense-augmented E2E；
- [ ] mixed-precision full certificate；
- [ ] BigFloat512 gate或正式缩小声明；
- [ ] canonical micro semantic pass；
- [ ] representative/local_full artifacts与skip inventory；
- [ ] canonical application x2/x4，至少3个独立样本 each；
- [ ] median/spread、peak RSS、memory limit、affinity、NUMA；
- [ ] CI artifacts retained；
- [ ] branch protection/required checks；
- [ ] Project/CITATION/CHANGELOG/README 正式 0.5.0 且 dated；
- [ ] final release tree == validated tree；
- [ ] final remote CI 全部通过；
- [ ] 没有 required gate 仅为 BLOCKED/SKIPPED；
- [ ] 只有完成上述项目后才 tag/release。

若任一 required item 未闭合，最终结论必须是：

```text
NO-GO for v0.5.0
```

不得因为“没有发现 P0”而把不完整证据链升级为 GO。

---

# 18. 本计划的最终原则

1. 先修数学合同，再统一规划权威，再谈性能和 cleanup。
2. 同一个未正则化 structured residual 应连接 factorization、refinement、trial acceptance 与 controller diagnostics。
3. 保留已经正确的 predictor/corrector factor reuse，不重做 O(n^3) 主路径。
4. provider 提供 primitive，SDPX 保留 solver policy。
5. 优先消除结构性重复、内存复制和真实 allocation；拒绝单一 microbenchmark 驱动的复杂化。
6. 任何性能结论必须来自 current candidate、完整 certificate、稳定 warm/repeated end-to-end measurements。
7. 发布不是“测试文件存在”，而是 final tree、minimum Julia、providers、application holdout、artifacts、metadata 和 governance 全部闭合。
