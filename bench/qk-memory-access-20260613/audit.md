# QK Memory Access Audit

Evidence gate before any Family C v1 implementation.

## Decision

- status: `family_c_v1_requires_core_integer_vector_load_lowering`
- run Family C v1 now: `False`
- run 32B: `False`
- next required change: Patch core integer vector load/store lowering for aligned uint32 global buffers, then rerun the probe and only then build Family C v1.
- stop rule: Do not broaden packed-load expression rewrites unless generated source shows vector/coalesced integer loads.

## Source Audit

- renderer vector pointer-cast syntax: `True`
- integer global vector-load folding blocked: `True`

- CStyle.render_access can render vector pointer casts when an INDEX has max_numel > 1.
- correct_load_store.split_load_store only enables vector fold lengths for float/half/fp8/image/DSP; integer global buffers fall back to length 1.

## Probe Summary

- normal UOp uint4 load supported: `False`
- raw custom uint4 escape supported: `True`
- Family C v0 vector load evidence: `False`
- Family C v0 packed-load kernel present: `True`

## Roofline Context

| model | tinygrad tok/s | llama tok/s | tinygrad % peak | llama % peak | tinygrad % llama |
|---|---:|---:|---:|---:|---:|
| 8B | 52.07 | 101.20 | 27.27 | 53.00 | 51.46 |
| 14B | 40.55 | 65.80 | 38.03 | 61.70 | 61.63 |
| 32B | 17.23 | 30.80 | 35.47 | 63.40 | 55.94 |

## PMC Prior

| kernel | events | GL2 hit rate | VALU / busy | SQ busy | VALU inst |
|---|---:|---:|---:|---:|---:|
| `q4k_gemv_partial_12288_4096_1` | 3 | 0.1613 | 1.2584 | 16411721 | 20653056 |

## Interpretation

The remaining gap is still best explained as memory-load efficiency, but the
next move is not another descriptor candidate. The normal tinygrad codegen
path does not currently preserve aligned integer vector loads for the Q4_K
`uint32` storage. Raw custom C can force such a load, which proves the
hardware/compiler surface exists, but not that BEAM or semantic search can
emit it. Family C v1 should therefore start with a core lowering capability
patch, guarded by this probe, before adding another candidate family.
