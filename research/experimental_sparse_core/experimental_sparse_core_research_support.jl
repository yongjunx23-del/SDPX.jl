# TEST-ONLY explicit alias for unadmitted numerical research construction.
# This is never selected by the memory-admitting preparation function and is
# not evidence that any memory budget was certified. Tests run small bounded
# local numerical primitives while production memory admission is unavailable.
using SDPX
const research_prepare_sparse_core_unadmitted =
    SDPX._research_prepare_experimental_sparse_core_state
