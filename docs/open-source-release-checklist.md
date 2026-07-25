# Open-Source Release Checklist

This is a practical checklist, not legal advice.

## Upstream lineage

The current repository history and license identify the upstream work as
`SDPJSolver.jl`, authored by Li-Yuan Chiang and licensed under MIT. Preserve
that provenance.

- Keep the existing MIT license text and the original
  `Copyright (c) 2022 Li-Yuan Chiang` notice.
- State prominently in the README that SDPX is derived from SDPJSolver.jl.
- Record the upstream URL and the exact base commit used for the rewrite.
- Keep Git history when possible. A GitHub fork makes the relationship
  especially clear; a new repository should still retain history and an
  `upstream` remote.
- Add your own copyright notice for original modifications without deleting
  the upstream notice.
- Do not imply that upstream authors endorse the fork.

Suggested README wording:

```text
SDPX.jl is derived from SDPJSolver.jl by Li-Yuan Chiang. The solver core has
been substantially revised to add typed workspaces, sparse and multithreaded
paths, improved stopping diagnostics, and additional arithmetic backends.
The original project and this derivative are distributed under the MIT
License. See LICENSE and THIRD_PARTY_NOTICES.md.
```

## Copied code versus ideas

Implementing a published algorithm or independently recreating an optimization
idea is different from copying source code. If code, comments, tests, or
documentation are copied or closely adapted from another project:

1. Identify the source file and revision.
2. Check that project's license before copying.
3. Preserve all required copyright, attribution, patent, and NOTICE material.
4. Mark modified files when the source license requires it.
5. Document the borrowed component in `THIRD_PARTY_NOTICES.md`.

Clarabel.jl is Apache-2.0 licensed. If SDPX only adopts architectural ideas
and independently implements them, describe Clarabel as inspiration and cite
it. If SDPX copies or adapts Clarabel source, include the Apache-2.0 license
and comply with its redistribution and NOTICE requirements in addition to the
MIT terms applying to the original SDPX/SDPJSolver code.

MIT has no explicit patent grant. Apache-2.0 does. This distinction matters if
substantial Apache-licensed code is incorporated.

## Julia package identity

`Project.toml` currently has:

```toml
name = "SDPX"
uuid = "e804a123-63fa-4b55-87b9-d771b4a5c602"
```

The UUID came from the upstream package identity. Decide before publishing:

- If SDPX is the official continuation/rename of the same package, keep the
  UUID only after confirming registry and user migration expectations.
- If SDPX is a distinct fork intended to coexist with SDPJSolver, generate a
  new UUID. Julia identifies packages by UUID, so a renamed independent
  package should not impersonate the upstream package identity.

Also check that `SDPX` is not confusingly close to an existing registered
package or trademark. The acronym is compact but not self-explanatory; a more
descriptive package name may pass Julia General review more easily.

## Repository files before public release

Required or strongly recommended:

- `README.md` with scope, mathematical form, installation, examples,
  limitations, benchmark protocol, and upstream attribution;
- `LICENSE` preserving the original MIT notice;
- `THIRD_PARTY_NOTICES.md`;
- `AUTHORS.md` or a clear contributor section;
- `CITATION.cff` and citations for the underlying algorithms;
- `CONTRIBUTING.md`;
- `CODE_OF_CONDUCT.md`;
- `SECURITY.md`;
- `CHANGELOG.md`;
- documentation built with Documenter.jl;
- CI on supported Julia versions and operating systems;
- Codecov or equivalent coverage reporting;
- CompatHelper and TagBot after registration.

Optional SPDX headers can make automated compliance scans easier:

```text
SPDX-License-Identifier: MIT
```

Do not add a new copyright line to every legacy file unless authorship is
clear. Git history plus the top-level notices is often cleaner.

## Release engineering

Before the first public tag:

1. Resolve the package name and UUID decision.
2. Clean the repository of generated benchmark binaries, private paths,
   credentials, and large artifacts.
3. Verify dependency licenses, including optional extensions.
4. Reproduce tests from a clean Julia depot.
5. Run formatting, Aqua.jl, JET.jl where useful, and `MOI.Test` once the
   optimizer wrapper exists.
6. Document supported Julia and arithmetic-type combinations.
7. Publish benchmark scripts and raw results together; never publish only a
   selected best number without the protocol.
8. Tag with semantic versioning. Because the public API and package name have
   changed substantially, do not present the rewrite as a drop-in patch
   release.
9. Register with Registrator only after compatibility bounds, tests, and
   documentation are stable.
10. Configure TagBot and protected branches after the repository URL is final.

## Current repository-specific actions

Before publication, address these concrete points:

- The Git remote still points to the upstream `SDPJSolver.jl` repository.
- The README and `THIRD_PARTY_NOTICES.md` now record the SDPJSolver.jl
  derivation and exact upstream base commit.
- `Project.toml` preserves the upstream author and UUID but uses the new
  package name. Add the authors of the new modifications after obtaining
  their preferred names and contact details.
- The original MIT notice is present and must remain.
- Benchmark and parameter documents are now in English, but the public docs
  still need a complete user guide and cited algorithm description.
- Decide whether serialized benchmark inputs belong in Git, Git LFS, a GitHub
  release, or a Julia artifact.

## References

- GitHub licensing guidance:
  <https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository>
- GitHub fork guidance:
  <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks>
- Julia package creation and registration:
  <https://pkgdocs.julialang.org/v1/creating-packages/>
- Julia project UUID documentation:
  <https://pkgdocs.julialang.org/dev/toml-files/>
