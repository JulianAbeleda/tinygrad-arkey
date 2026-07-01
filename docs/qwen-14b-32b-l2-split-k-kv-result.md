# L2 Split-K Decode for Occupancy-Starved KV — Result (14B, real GPU)

Lever: raise occupancy for the row-starved KV projections (attn_k/attn_v 5120->1024,
~26% GPU occupancy) with a GENERIC generated split-K decode kernel — no handwritten
kernel, no model/shape special-case.

## Capability built (generic)

`extra/qk_gemv_g3_codegen_lowering.py :: q4k_g3_lanemap_gemv_splitk_kernel(rows, k, parts)`
adds a second global axis (`gidx1`) over `parts` K-slices to the generated G3 GEMV, so
each output row is computed by `parts` workgroups instead of one. Generic: `parts` must
divide `blocks_per_group = (k//256)//4`; wired in `model.py` behind `DECODE_Q4K_SPLIT_K_KV`
with a generic occupancy heuristic (`out_features <= DECODE_SPLIT_K_MAX_ROWS`, `parts` =
largest divisor of blocks_per_group keeping `out*parts <= DECODE_SPLIT_K_TARGET_WG`). No
model-name or exact-shape check.

## Correctness — PASS

- Microgate `extra/qk_decode_split_k_kv_microgate.py` on the real attn_k weight
  (1024x5120): rel_rmse **2.97e-07**, max_abs 3.34e-06 vs direct G3 -> numerically identical.
- Full-model 14B token-match at ctx512: byte-identical output with the flag on vs off.
- Route-bound: `q4k_g3_lanemap_gemv_splitk_1024_5120_5` fires 40x/step (confirmed in the
  ordered trace); parts=5 -> 5x workgroups (1024 -> 5120).

## Speed — REFUTED (no W==D movement)

Authority W==D (`qk_decode_runtime_overhead.py`, synced, NMEAS=40), flag off vs on:

| ctx | baseline tok/s | split-K tok/s | delta |
|-----|----------------|---------------|-------|
| 128 | 44.40 | 44.40 | 0.0% |
| 512 | 43.00 | 43.00 | 0.0% (baseline rerun 42.80, i.e. within noise) |

The KV-projection occupancy gain is **exactly offset by the external combine reduce**
(`.sum` over `parts`) that split-K introduces — a new `r_*` reduce kernel per split GEMV.
The KV projections are also a small fraction of whole-model decode, so even a role-local
bandwidth win (memory: 24->34.9 GB/s @parts4) does not survive to tok/s.

## BoltBeam verdict

`decode_q4k_split_k_kv` -> **refuted** (axis `split_k_kv_external_combine`), rollback
`DECODE_Q4K_SPLIT_K_KV=0` (default off). **Reopen condition:** fuse the K-part combine
IN-KERNEL (LDS/atomics across the split workgroups) so no external reduce is added — only
then can the occupancy gain reach W==D. Do not retry the external-combine form.

## Disposition

Kept default-OFF and token-correct as the substrate for the reopen (in-kernel combine).
No default path changed; flag-off is byte-identical.
