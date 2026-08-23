# Sources and provenance

This archive intentionally stores **manifests and generators**, not multi-GB
copies of public benchmark binaries.

Public source pages used to construct the manifests:

- Hans Mittelmann benchmark index: https://plato.asu.edu/bench.html
- Sparse/other SDP benchmark (25 Apr 2026): https://plato.asu.edu/ftp/sparse_sdp.html
- SDP data directory: https://plato.asu.edu/ftp/sdp/
- Infeasible SDP benchmark (21 Sep 2025): https://plato.asu.edu/ftp/sdp_inf.html
- Large SOCP benchmark (10 Aug 2026): https://plato.asu.edu/ftp/socp.html
- LPfeas benchmark (10 Aug 2026): https://plato.asu.edu/ftp/lpfeas.html
- LPopt benchmark (1 Jul 2026): https://plato.asu.edu/ftp/lpopt.html
- Large Network-LP benchmark (1 Jul 2026): https://plato.asu.edu/ftp/network.html
- CBLIB: https://cblib.zib.de/
- CBLIB file directory: https://cblib.zib.de/download/all/
- CBF v3 technical reference: https://cblib.zib.de/doc/format3.pdf
- DIMACS Seventh Challenge archive: https://archive.dimacs.rutgers.edu/Challenges/Seventh/Instances/
- SDPLIB catalogue: https://github.com/vsdp/SDPLIB
- SDPpack historical compact format/software: https://cs.nyu.edu/~overton/software/sdppack/
- SDPA sparse format: https://sdpa-python.github.io/docs/formats/sdpa.html
- SDPA-GMP high-precision SDPLIB objective table: https://github.com/nakatamaho/sdpa-gmp
- Netlib LP data: https://netlib.org/lp/data/

CBLIB explicitly permits redistribution of its data subject to its license
conditions; nevertheless, this package leaves the large files at the official
source and downloads them on demand. Other datasets must retain their own
upstream licensing/provenance requirements.

Reference runtime columns are copied as *orientation metadata* from the named
Mittelmann tables. Flags:
- `a`: at least one DIMACS error exceeds 1e-4 in the SDP table.
- `f`: failure; in the SDP table this means at least one DIMACS error exceeds 1e-2.
- `m`: memory limit exceeded.
- `t`: time limit exceeded (LP/SOCP tables as applicable).

Do not compare those seconds directly to SDPX timings on another machine.

The Maros–Mészáros collection is a convex QP/QPS collection, not an SDP
library. It must be labelled separately if a future QP-to-conic campaign adds
it; SDPLIB and DIMACS are the SDP/conic sources used here.
