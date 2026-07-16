# Cooperative MMQ compiler/resource audit

`extra/qk/mmq_logical_vocabulary.py` describes the operation and logical axes
(`m`, `n`, `k`, `group`, `activation_block`). `PhysicalMapping` currently
provides only wave size, workgroup size, tile order, and WMMA shape. The new
`extra/qk/mmq_lowering_audit.py` joins those facts with final compiler evidence:

| logical claim | physical evidence required |
|---|---|
| workgroup becomes waves | `workgroup_size / wavefront_size`, exact divisibility |
| staged tile becomes LDS | final `lds_bytes`; no geometry-only inference |
| accumulator stays in VGPRs | final `vgpr` count (spills are separate evidence) |
| cooperative visibility is safe | final uniform workgroup barrier count |
| integer matrix operation is real | final `mfma_sites` count |

Admission is fail-closed for missing evidence, LDS over 64 KiB, invalid counts,
and multi-wave candidates with no barrier. The 64 KiB limit and wave sizes are
policy/capability inputs, not proofs of a successful lowering.

Unresolved requirements remain: a real compiler-side lane/owner map, LDS
layout and lifetime, per-fragment VGPR accounting, barrier placement/order,
edge-predicate lowering, and an exact MFMA shape/operand mapping. In particular,
the existing logical Q4 descriptor says direct/no-sync while the cooperative
probe reports LDS/barriers; this audit does not silently reconcile that mismatch.
Emitter and route-selector files are intentionally untouched.
