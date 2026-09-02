# Physics catalog contract template

Every physics catalog README and its machine-readable `catalog.jl` should
answer the following questions. The template is a claim-boundary contract,
not a requirement to promote a build-only artifact to a solved benchmark.

## 1. Physical assumptions and conventions

State the physical system, fields/operators/states, dimensions, masses or
couplings, symmetry sector, normalization, sign conventions, and any
assumptions such as positivity, unitarity, reflection positivity, or a
truncation of the Hilbert/operator space.

## 2. Exact primary-paper equations and version

Name the primary source, authors, title, DOI/arXiv identifier and pinned
version. List the exact equations implemented, and distinguish equations
implemented literally from equations used only as motivation. A paper target
or citation is not an implementation claim.

## 3. Truncation and discretization

Declare every finite grid, basis degree, spin/level cutoff, quadrature,
series/product truncation, lifted rank, and precision choice. Explain what
continuum or infinite-dimensional statement is lost. Do not call a finite
sample a continuum proof.

## 4. Convex variables, objective, and cones

Give the physical decision variables, affine equalities, objective and
orientation, and the exact real conic encoding: LP, Lorentz/rotated Lorentz,
exponential, power, or PSD blocks. For complex quantities document
conjugation/Hermiticity and realification, including all factors of two.

## 5. Strict or facial witness

List an independently constructed feasible strict-interior witness. If Slater
feasibility is unavailable, provide an exact face/ray witness and say so. The
witness must satisfy both the physical equations and the lowered canonical
program; it is not an objective oracle.

## 6. Reference status

Use one of the explicit statuses `:verified`, `:sampled_build_only`, or
`:build_only`. `:optimal` requires an independent objective/status oracle and
an original-coordinate certificate. Build success and deterministic
fingerprints alone never establish an optimum or a published bound.

## 7. Excluded claims

State explicitly which paper result, continuum limit, numerical bound,
resonance interpretation, omitted sector, or solver certificate is *not*
claimed. Keep `paper_equivalent=false` unless a separate reproduction dossier
has passed review.

## 8. Scaling tiers

Provide a table of named tiers with dimensions, variable counts, cone/block
counts, rows, arithmetic/precision, and intended route (build, solve, or
profile). Tiers must change mathematical structure or declared resolution,
not merely repeat identical cones.

## Machine-readable parity

`catalog.jl` should carry the same source/version, fidelity/status,
truncation, cone signature, witness/oracle strength, and excluded-claim fields
(or a deterministic reference to them). Tests should load the catalog twice,
require stable IDs/fingerprints and claim status, and mutate/check semantic
inputs where an implementation-specific validator exists.
