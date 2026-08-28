# Test

`test/runtests.jl` is the sole regression suite. It exercises the public path
from typed modeling through `optimize!` to a terminal original-coordinate
certificate for LP, SOCP, SDP, exponential, and power-cone cases.

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Historical unit, kernel, wave, provider, and internal regression tests were
archived outside the repository before removal. Manual provider and independent
Newton checks remain under `validation/`.
