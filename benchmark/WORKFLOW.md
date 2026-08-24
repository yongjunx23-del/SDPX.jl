# Benchmark-driven optimization workflow

This is the day-to-day loop for making SDPX faster without making it wrong.
It runs on a laptop (Mac, `benchmark/benchenv` or the main project) for the
common case; the cluster only appears at the final validation step. The
canonical statement of what may gate what lives in `benchmark/README.md`;
the short version: **correctness and semantics gate, timings inform a human
decision** — no merge is blocked by a timer, and no timing claim exists
without a recorded baseline.

## 0. 冻结不变量（一次性，改动前确认）

- 候选与基线解**同一数学问题**：同目标函数、同锥表示、同容差、同重建路径；
- `status` 不算成功——必须 `semantic_pass` + `certificate_valid`（原坐标证书）；
- provider 在规划期定型：候选不得引入隐藏回退（`unexpected_fallback` 必须为 false）；
- 计时纪律：`--samples>=3`、预热不计入、性能声明走 fresh-process；
- 输入指纹与 solver source sha256 必须配对一致（compare 会拒绝不配对的比较）。

## 1. 基线（改动前）

```sh
julia benchmark/optimize.jl baseline              # 默认 HEAD^，ladder 套件
julia benchmark/optimize.jl baseline --base=<sha> --suites=ladder,micro
```

驱动会把 `<base>` 检出到 `work/optimize/baseline-tree` 工作树、复制当前
`Manifest.toml`（锁定同一依赖版本）、实例化并跑同一组套件，写入
`work/optimize/baseline.toml`。

## 2. 候选（改动后）

```sh
# ...改代码...
julia --project=. -e 'using Pkg; Pkg.test()'      # 或依赖 CI quick 档
julia benchmark/optimize.jl candidate
```

## 3. 比对 + 记录 + 裁决

```sh
julia benchmark/optimize.jl loop      # baseline → candidate → compare → record
julia benchmark/optimize.jl compare   # 只重跑比对
```

`compare` 输出逐行 `total_seconds_ratio` 与**全部 phase 的 ratio**
（setup/frontend/preprocess/presolve/core/certification），并按偏离排序。
裁决规则：

- 任何一行 `semantic_pass`/`certificate_valid`/`status` 变化 → **reject**（无论多快）；
- 目标 phase 的 ratio < 0.95 且语义不变 → retain 候选，`record` 入账；
- 1.05 > ratio ≥ 0.95 → neutral，默认不合并（避免噪声驱动的 churn）；
- ratio > 1.05 → 找出变慢的 phase，回退或修复后再来。

## 4. 记录与趋势

`loop` 的 `record` 步骤把候选结果追加进
`benchmark/history/performance-log.csv`（commit/时间戳/树洁净度）。
查看趋势：

```sh
julia benchmark/history_log.jl trend ladder/sdp_750 --last=10
```

dirty tree 的记录行会被标记，不用于任何声明。

## 5. 集群终验（合并前）

本地 ladder 通过后，合并前在大规模行上做一次 fresh-process 终验
（ucs-hpc，`large` 套件 + `fresh_process_runner.jl`，≥3 独立进程）：

```sh
# 集群侧（PBS）
julia --project=benchmark/benchenv benchmark/fresh_process_runner.jl large \
  --problem=<affected-row> --campaign-dir=work/fresh/<name>
julia --project=. benchmark/compare.jl work/fresh/base.toml work/fresh/cand.toml
```

`process_peak_rss_bytes` 只用于同环境回归对比。通过后手动触发
`gh workflow run optimization-loop.yml`；成功的 optimization-loop run 会把
parent-versus-candidate 证据写入 `benchmark/history/`。

## 6. 微基准（定位到 phase 之后）

compare 指认变慢的 phase 后，为该 phase 冻结一个微基准再改：
使用单文件脚本（固定输入、`@timed` 多样本、checksum 防漂移），并把可复核
的 driver 与结果归档到 `docs/evidence/bench/<campaign>/`。
先证明微基准复现总时间变化，再动手优化；改完先过微基准，再回到步骤 2。
