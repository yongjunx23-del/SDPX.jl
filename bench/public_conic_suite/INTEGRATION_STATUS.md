# Integration status in the SDPX v0.4.1 development snapshot

The supplied suite is copied into the repository unchanged so its provenance,
manifests and generators can act as a stable acceptance layer.

Implemented in this snapshot:

- suite present under the intended repository path;
- all-auto frontend policy and SDPB-style precision/tolerance CLI;
- development plan that assigns benchmark families to frontend/midend/backend
  milestones.

Not yet implemented (next development tasks):

- SDPA sparse file loader for the external SDP cases;
- CBF loader preserving native Lorentz semantics;
- MPS/Netlib loader feeding the native sparse LP path;
- one unified external-suite runner and normalized result record;
- external reference solver adapters;
- pinned same-hardware baseline results.

Do not treat the presence of the manifest as evidence that these external cases
have been executed by this snapshot.
