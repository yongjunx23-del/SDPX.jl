function _public_spec(
    id, family, problem_type, source, tiers, tags, purpose, external;
    status=:optimal,
    objective=nothing,
    absolute_tolerance=1.0e-7,
    relative_tolerance=1.0e-7,
    note="reference supplied by the authoritative source",
    size=(;),
)
    return BenchmarkSpec(
        id,
        uppercase(last(split(id, '/'))),
        family,
        problem_type,
        source,
        tiers,
        tags,
        purpose,
        nothing,
        Symbol("external_", external.format),
        (;),
        BenchmarkReference(
            status, objective, absolute_tolerance, relative_tolerance, note,
        ),
        size,
        external,
    )
end

const _NETLIB_ROOT = "https://netlib.org/lp/data"
const _NETLIB_NOTE = "Official Netlib LP compressed-MPS collection; retain upstream notices and attribution."

function _netlib(
    name, tiers, purpose, dimensions, sha256;
    tags=(:public, :netlib),
)
    filename = lowercase(name)
    return _public_spec(
        "netlib/$filename", :lp, :linear_program, :netlib,
        tiers, tags, purpose,
        ExternalSource(
            "NETLIB LP",
            "$_NETLIB_ROOT/$filename",
            filename,
            :netlib_compressed_mps,
            sha256,
            _NETLIB_NOTE,
        );
        objective=dimensions.objective,
        note="Netlib readme reference optimum",
        size=(
            rows=dimensions.rows,
            columns=dimensions.columns,
            nonzeros=dimensions.nonzeros,
            compressed_bytes=dimensions.bytes,
        ),
    )
end

const NETLIB_SPECS = BenchmarkSpec[
    _netlib("AFIRO", (:representative, :local_full), :tiny_real_lp,
        (rows=28, columns=32, nonzeros=88, bytes=794,
         objective=-4.6475314286e2),
        "964d9df9579e2d16a025693b64057910ba99f1ec6bafe40ef0098319e40b28ba"),
    _netlib("SC50A", (:representative, :local_full), :small_real_lp,
        (rows=51, columns=48, nonzeros=131, bytes=1615,
         objective=-6.4575077059e1),
        "f6eb22f39b9978bba1ed5de3bba0b63ce70a058ef5e77a6dc7f06901edfd9ef8"),
    _netlib("SC50B", (:representative, :local_full), :small_bound_lp,
        (rows=51, columns=48, nonzeros=119, bytes=1567,
         objective=-7.0e1),
        "830f16f59525f09fadde86144ed9a317bb19129c4c4c9fe05620807ac89f3968"),
    _netlib("KB2", (:representative, :local_full), :bound_handling,
        (rows=44, columns=41, nonzeros=291, bytes=2526,
         objective=-1.7499001299e3),
        "f8410e43eac30a999179519932a7a9ed630d6a7428438710bffb10b01433eee0"),
    _netlib("ADLITTLE", (:representative, :local_full), :small_rectangular_lp,
        (rows=57, columns=97, nonzeros=465, bytes=3690,
         objective=2.2549496316e5),
        "a8d81309f69682bdc02ff7b57841f06c7254fa5cca997c2b3deea794ea8efdfd"),
    _netlib("BLEND", (:representative, :local_full), :mixed_row_structure,
        (rows=75, columns=83, nonzeros=521, bytes=3227,
         objective=-3.0812149846e1),
        "68ae6dc918140675881d6b640d7acac9e5e903ac063476e5767ebfee54f4172d"),
    _netlib("RECIPE", (:representative, :local_full), :bound_and_fixed_variables,
        (rows=92, columns=180, nonzeros=752, bytes=6210,
         objective=-2.66616e2),
        "cc053f4111f619da90b223455192543893a638b2d5e0490aeb8afad778ed9736"),
    _netlib("SC105", (:representative, :local_full), :harwell_lp,
        (rows=106, columns=103, nonzeros=281, bytes=3307,
         objective=-5.2202061212e1),
        "6fa3fa2d5feba702f5698c7639a51453d2ae863a90bdd851c0d30ec8cceaa4ef"),
    _netlib("STOCFOR1", (:representative, :local_full), :stochastic_forestry_structure,
        (rows=118, columns=111, nonzeros=474, bytes=4247,
         objective=-4.1131976219e4),
        "a780fcc386354938e9cabfee9e22e05166dbf3fea7ccb5953ce7afa24a17abd2"),
    _netlib("SCAGR7", (:representative, :local_full), :agricultural_scaling,
        (rows=130, columns=140, nonzeros=553, bytes=4953,
         objective=-2.3313892548e6),
        "d6309f6c3d1489d91e0361c8e654a67207f57cafc683eda43c6b1dc59b22bbdd"),
    _netlib("BOEING2", (:representative, :local_full), :degenerate_bound_lp,
        (rows=167, columns=143, nonzeros=1339, bytes=8761,
         objective=-3.1501872802e2),
        "db1c2233bc17d6fa9a5f8df1faf72ca495f34dc7c42b6f06226576eb9fd03a70";
        tags=(:public, :netlib, :degenerate)),
    _netlib("DEGEN2", (:representative, :local_full), :degeneracy,
        (rows=445, columns=534, nonzeros=4449, bytes=24657,
         objective=-1.435178e3),
        "33c756caf406b3b769bd063ca47ad15940da774f742742e16dee813d2d12ab93";
        tags=(:public, :netlib, :degenerate)),
]

const _NETLIB_LOCAL_FULL = (
    ("SC205",  "b56bb782abaf24f7d7a0812cdb227e4c286ab332b03ddbf2681ba13c6f7de4f8", :larger_sc_control, 6228),
    ("SHARE1B","b3071aa424f9af9385c1d9b156577caa1b59fb7217d8e57aeaeb817293f2736a", :shared_resource_lp, 8259),
    ("BORE3D", "92bda52f562dc51c4f2608089e15b68f1618c7199305084738f18e7082715872", :three_dimensional_structure, 12442),
    ("CAPRI",  "d3c1d48a4e4018ae16935be4b5f59454159411cf017311da9216a9030c9010d8", :mixed_bounds_lp, 15244),
    ("BANDM",  "2d43005a87d997095ad6752523232cfb7c35f7482f60bbcf8052639bb9bbb097", :banded_lp, 18716),
    ("SCFXM1", "57fcc8548f2e386d70c04640bd8f2fe28c1cb49ca1838a91ceebfb6e84642e5f", :network_like_lp, 19455),
    ("GROW7",  "3a9006f5891b11d02323465e3f69a32158194dedd58028d51afcf45258c00a1c", :growth_model, 16850),
    ("PILOT4", "baf0c68257a14cc34fd56e50d4b3701f117339e819175da658d37a2158c98dd3", :pilot_model, 40624),
    ("SCSD1",  "bd0fa1ad04e0ba4a1571808866160ec77f9dbe15ff46bc359447ec747d283569", :staircase_sparsity, 16420),
    ("FIT1D",  "c2e46d3b0ccda546a898a9cfd120e6343b46a0ff9cd7ab85a55285cbed5691f1", :very_rectangular_lp, 52002),
)

for (name, checksum, purpose, bytes) in _NETLIB_LOCAL_FULL
    push!(NETLIB_SPECS, _netlib(
        name, (:local_full,), purpose,
        (rows=missing, columns=missing, nonzeros=missing,
         bytes=bytes, objective=nothing),
        checksum,
    ))
end

const _SDPLIB_ROOT = "https://plato.asu.edu/ftp/sdplib"
const _SDPLIB_NOTE = "Official SDPLIB data directory; retain original SDPLIB attribution and per-instance provenance."

function _sdplib(name, tiers, purpose, m, block_sizes, bytes, checksum; tags=(:public, :sdplib))
    filename = "$name.sdp.gz"
    return _public_spec(
        "sdplib/$name", :sdp, :semidefinite_program, :sdplib,
        tiers, tags, purpose,
        ExternalSource(
            "SDPLIB",
            "$_SDPLIB_ROOT/$filename",
            filename,
            :sdpa_sparse_gzip,
            checksum,
            _SDPLIB_NOTE,
        );
        note="reference objective pending authoritative SDPLIB table import",
        size=(variables=m, blocks=length(block_sizes),
              block_sizes=block_sizes, compressed_bytes=bytes),
    )
end

const SDPLIB_SPECS = BenchmarkSpec[
    _sdplib("control1", (:representative, :local_full), :control_lmi, 21, (10, 5), 1644,
        "fb6f544c26c2d88502308c49558b0e0929d063f1efe5d1f76bf0b6727c81871a"),
    _sdplib("control2", (:representative, :local_full), :larger_control_lmi, 66, (20, 10), 11125,
        "3b0027494eb1447d0fc23aaca4026366c66892b1c9c023af64e3c9b1d7cc3305"),
    _sdplib("truss1", (:representative, :local_full), :structured_truss_sdp, 6, (2,2,2,2,2,2,1), 173,
        "ff992a77a95850a625184b4ad9cb999c88fb03b6619dc24ae63882e244cd972f"),
    _sdplib("truss3", (:representative, :local_full), :multiblock_structural_sdp, 27, (5,5,5,5,5,5,1), 389,
        "7fb1ff1e94e02df467c9dbbd2ec87b036c9f25dd68d6b1efd2a487489c13a099"),
    _sdplib("mcp100", (:representative, :local_full), :max_cut_relaxation, 100, (100,), 1736,
        "17f594b630a03dd82154871b0f542f4c8c4ff5d155488938edfa9a146f7ba887"),
    _sdplib("gpp100", (:representative, :local_full), :graph_partition_relaxation, 101, (100,), 16005,
        "7f02e5a5ba4ce6b7584d17b6c563331a5f035949e75314bb418ba8b3b8246710"),
    _sdplib("theta1", (:representative, :local_full), :lovasz_theta, 104, (50,), 3819,
        "2188cbab3a4779d3101a25ca72d90fbb7747a2dfa30cc35b6ee00805e066b27d"),
    _sdplib("theta2", (:representative, :local_full), :larger_lovasz_theta, 498, (100,), 16402,
        "6023c79ee8876060a46f5d92073a33f64a65d8fa389fa846cc7aa28133866eeb"),
    _sdplib("qap5", (:local_full,), :qap_relaxation, 136, (26,), 2455,
        "39010b45dc47bcdf7af28b3f6f365ca67978e5322d1ff68b59fc9cc8205b40f2"),
    _sdplib("qap6", (:local_full,), :larger_qap_relaxation, 229, (37,), 5898,
        "ff2f41311dd292b41d435e89475f0bc1baf4a6b68f7a1709c6f0e7484262df93"),
    _sdplib("mcp124-1", (:local_full,), :max_cut_family_diversity, 124, (124,), 1537,
        "ac48ff67d2a1aa6075bf167deca9a501d1a038a21069a95a8cb77f1e8f1a1e50"),
    _sdplib("gpp124-1", (:local_full,), :graph_partition_family_diversity, 125, (124,), 22484,
        "b1c4c0f036fb74f0e75e18dcccccb92c4be13725529a493fe9af86aceb20e394"),
]

const _DIMACS_ROOT = "https://plato.asu.edu/ftp/dimacs"
const _DIMACS_NOTE = "Official DIMACS conic benchmark directory; SDPA sparse representation may encode Lorentz blocks by negative block sizes."

function _dimacs(name, tiers, purpose, m, blocks, bytes, checksum; family=:socp, tags=(:public, :dimacs))
    filename = "$name.dat-s.gz"
    return _public_spec(
        "dimacs/$name", family, :mixed_conic_program, :dimacs,
        tiers, tags, purpose,
        ExternalSource(
            "DIMACS conic benchmark",
            "$_DIMACS_ROOT/$filename",
            filename,
            :sdpa_sparse_gzip,
            checksum,
            _DIMACS_NOTE,
        );
        note="DIMACS reference retained when loader/reference table is added",
        size=(variables=m, blocks=blocks, compressed_bytes=bytes),
    )
end

const DIMACS_SPECS = BenchmarkSpec[
    _dimacs("nql30", (:representative, :local_full), :small_real_socp, 6380, 901, 127321,
        "62c86b455ce81ff2f75dcfcf5a1f684c1844abefaddf22fd3bb4fad42ec1dc5c"),
    _dimacs("nql60", (:representative, :local_full), :larger_real_socp, 25360, 3601, 496689,
        "538a33fab1f815441258463e15f24c9f3aa2a1aa67d3c98bb459d66767f0b3ca"),
    _dimacs("qssp30", (:representative, :local_full), :quadratic_socp, 15037, 1892, 201873,
        "eb9b549ee34baf20d20a7482244d1b0421af64249395c30daf58167017f12835"),
    _dimacs("qssp60", (:representative, :local_full), :larger_quadratic_socp, 58867, 7382, 798017,
        "4d51bb71975ab3c6a0c3599c3dadee97a86a28ab853c9b03ff027ff995444ed3"),
    _dimacs("filter48_socp", (:representative, :local_full), :filter_design_socp, 2145, 3, 392819,
        "70f8efa93c05c604a689a0182600bff838bc242978174a85e1106c33c12b6f47"),
    _dimacs("hinf13", (:representative, :local_full), :hinfinity_control, 57, 3, 20975,
        "7687c2bca21a843bf4a67bfe8b06c23d2d46537384117b29aec0c868ca2d93ea";
        family=:mixed_conic, tags=(:public, :dimacs, :mixed)),
    _dimacs("minphase", (:representative, :local_full), :minimum_phase_filter, 48, 1, 6062,
        "28fda50de3f18eee9c1d7e584dc94aad865b7b4e9b8aa145e9b03f91794a5786";
        family=:mixed_conic, tags=(:public, :dimacs, :mixed)),
]

const _CBLIB_ROOT = "https://cblib.zib.de/download/all"
const _CBLIB_NOTE = "CBLIB continuous conic benchmark; redistribution is subject to the CBLIB license, so data stay on-demand."

function _cblib(name, tiers, purpose, bytes, checksum)
    filename = "$name.cbf.gz"
    return _public_spec(
        "cblib/$name", :socp, :continuous_conic_program, :cblib,
        tiers, (:public, :cblib, :continuous), purpose,
        ExternalSource(
            "CBLIB",
            "$_CBLIB_ROOT/$filename",
            filename,
            :cbf_gzip,
            checksum,
            _CBLIB_NOTE,
        );
        note="CBF loader intentionally deferred; metadata/download interface only",
        size=(compressed_bytes=bytes,),
    )
end

const CBLIB_SPECS = BenchmarkSpec[
    _cblib("beam7", (:local_full,), :cblib_continuous_interface, 57395879,
        "d337224aeecb482dd020c9bcd293fa2c446806b30ba2434a7591d137958f61f5"),
    _cblib("beam30", (:heavy,), :larger_cblib_continuous_interface, 28912469,
        "137b5c49fc8d7f00b7934e4798b6a62536273605e3c2af87229d84995ced391e"),
]

const PUBLIC_SPECS = vcat(
    NETLIB_SPECS,
    SDPLIB_SPECS,
    DIMACS_SPECS,
    CBLIB_SPECS,
)
