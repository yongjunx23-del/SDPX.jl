using Aqua
using SDPX
using Test

# Registry-quality gates: no type piracy, no method ambiguities among owned
# methods, no unbound type parameters, every export defined, no stale or
# compat-less dependencies. These are the checks General's automerge and
# every packaging review run first; keeping them in the suite means a
# violation fails here instead of at registration time.
@testset "Aqua quality checks" begin
    # SDPX's dependency graph can take longer than Aqua's 10-second default
    # to finish the clean wrapper precompile on constrained CI/macOS hosts.
    # The status file is written before this timeout starts, so extending the
    # grace period still detects genuinely persistent tasks rather than
    # masking package-load work.
    Aqua.test_all(
        SDPX;
        ambiguities=(recursive=false,),
        persistent_tasks=(tmax=60,),
    )
end
