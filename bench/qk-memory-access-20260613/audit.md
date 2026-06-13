# QK Memory Access Audit

Evidence gate for Family C v1 implementation.

## Decision

- status: `family_c_v1_source_supported`
- run Family C v1 now: `True`
- run 32B: `False`
- next required change: Build Family C v1 as a generated memory-access candidate using the verified uint32x4 lowering.
- stop rule: Do not broaden packed-load expression rewrites unless generated source shows vector/coalesced integer loads.

## Source Audit

- renderer vector pointer-cast syntax: `True`
- integer uint32x4 global load/store folding supported: `True`

- CStyle.render_access can render vector pointer casts when an INDEX has max_numel > 1.
- correct_load_store.split_load_store now allows aligned uint32x4 global load/store folding.

## Probe Summary

- normal UOp uint4 load supported: `True`
- raw custom uint4 escape supported: `True`
- probe UOp vector load evidence: `True`
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

The remaining gap is still best explained as memory-load efficiency. The
normal tinygrad codegen path now preserves a requested aligned `uint32x4`
global load/store on AMD, and DEBUG=4 source confirms vector pointer casts.
Family C v1 is therefore unblocked as the next generated memory-access
candidate. Family C v0 remains rejected; it did not request this new load
shape and still emitted scalar `u32` loads.
