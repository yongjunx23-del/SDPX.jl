using Aqua
using SDPX
using Test

# Registry-quality gates: no type piracy, no method ambiguities among owned
# methods, no unbound type parameters, every export defined, no stale or
# compat-less dependencies. These are the checks General's automerge and
# every packaging review run first; keeping them in the suite means a
# violation fails here instead of at registration time.
@testset "Aqua quality checks" begin
    Aqua.test_all(SDPX; ambiguities=(recursive=false,))
end
