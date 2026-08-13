function _heavy(id, family, source, purpose, url, format, note)
    return BenchmarkSpec(
        id,
        replace(id, '/' => ' '),
        family,
        family === :lp ? :linear_program : :conic_program,
        source,
        (:heavy,),
        (:public, :heavy, :register_only),
        purpose,
        nothing,
        Symbol("external_", format),
        (;),
        BenchmarkReference(:unknown, nothing, Inf, Inf, note),
        (local_available=false,),
        ExternalSource(
            string(source), url, last(split(url, '/')), format, nothing, note,
        ),
    )
end

const HEAVY_SPECS = BenchmarkSpec[
    _heavy(
        "heavy/netlib_full", :lp, :netlib_full, :full_netlib_regression,
        "https://netlib.org/lp/data/", :collection,
        "Full Netlib collection; future heavy run only.",
    ),
    _heavy(
        "heavy/sdplib_full", :sdp, :sdplib_full, :full_sdplib_regression,
        "https://plato.asu.edu/ftp/sdplib/", :collection,
        "Full SDPLIB collection; future heavy run only.",
    ),
    _heavy(
        "heavy/cblib_full", :socp, :cblib_full, :full_cblib_continuous,
        "https://cblib.zib.de/download/all/", :collection,
        "Continuous supported-cone CBLIB subset; future heavy run only.",
    ),
    _heavy(
        "heavy/mittelmann", :mixed_conic, :mittelmann,
        :same_hardware_solver_comparison,
        "https://plato.asu.edu/bench.html", :index,
        "Mittelmann source/reference index only; timings are not cross-machine baselines.",
    ),
    _heavy(
        "heavy/large_sparse", :sdp, :sdpx_future, :large_sparse_schur,
        "https://github.com/yongjunx23-del/SDPX.jl", :generated,
        "Large sparse/chordal workloads; no Round 2 execution.",
    ),
    _heavy(
        "heavy/bootstrap", :sdp, :sdpx_future, :bootstrap_workload,
        "https://github.com/yongjunx23-del/SDPX.jl", :generated,
        "Bootstrap application workloads; no Round 2 execution.",
    ),
    _heavy(
        "heavy/precision_sweep", :mixed_conic, :sdpx_future,
        :bigfloat_512_1024_precision_sweep,
        "https://github.com/yongjunx23-del/SDPX.jl", :generated,
        "Full provider/precision sweep is deliberately decoupled from Mac development.",
    ),
]
