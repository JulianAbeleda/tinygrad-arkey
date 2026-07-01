# Attention Fused In-Kernel LSE Combine — Result (14B) — mechanism PROVEN, architecture REFUTES

Route (b): author a generated fused-flash decode kernel that removes the attention
combine (~12-24% of decode) by doing the online-softmax LSE merge IN LDS. No handwritten
kernel — pure UOp on the fused_xlane substrate.

## Built + correct

`extra/qk_flash_decode_fused_combine.py :: flash_decode_fused_combine_kernel`: workgroup
per head, `Smax` waves (one per KV split), each wave runs the online softmax over its
split (acc/den/mx d-sharded), then an **in-kernel LDS+barrier LSE merge** across the waves
-> `out[h,:]` directly. Removes the 3 external `flash_gmax/den/combine` kernels and the
`po` global buffer. Generated UOp, native primitives only.

- Microgate vs a numpy flash reference: **rel_rmse 2.6e-04** (f16 K/V) — correct.
- 14B in-model: **token-identical**; the kernel fires (`flash_fused_combine_40_128`).
- **Mechanism works:** the `attention_combine` reduce bucket drops **13.57% -> 1.71%**
  (total reduce 19.0% -> 2.5%). The combine really is removed.

## Speed — REFUTED (measured)

Authority W==D (14B):

| ctx | base tok/s | fused tok/s | delta |
|-----|-----------|-------------|-------|
| 128 | 52.3 | 52.3 | flat (flash off <512, fused branch inactive) |
| 512 | 50.2 | **6.2** | **-88%** |

BoltBeam: `reduce_eliminated` guardrail = **PASS** (bucket genuinely shrank 11.9pp), verdict
= **refute** (protected-context regression -87.7%). The mechanism is credited, the net is not.

## Why (the architectural finding, now measured)

The default `gqa_coop_vec` flash deliberately runs the partial phase over **Hq*S
workgroups** (high occupancy) and combines with 3 small separate kernels. Fusing the
combine forces the S splits into **Hq=40 workgroups** (one per head, S waves each) — and
40 workgroups on 96 CUs, with each wave serially reducing L keys, is ~8x less parallel.
**You cannot both keep the Hq*S split-parallelism AND combine in-kernel** — they are
mutually exclusive, and at Hq=40 the parallelism loss dwarfs the ~12% combine saved. The
combine's *separateness IS the occupancy strategy* for this shape.

## Conclusion

The attention combine is **not profitably removable for 14B/32B**. Every path is now
measured-and-refuted, not guessed:
- reachable transforms declined/refuted (FLASH_L, wholecache route, DECODE_OUTER_B_SPLIT);
- the authored fused kernel proves the mechanism (bucket 13.6%->1.7%) but the architecture
  refutes the removal (8x regression).

The fused kernel is kept default-off (`DECODE_ATTN_FUSED_COMBINE`, token-correct, reusable
substrate). No default path changed. This is the honest end of the attention-combine lever:
the flash split-partial/separate-combine structure is correct, and the 13.6% is the price of
the occupancy that keeps the partial phase fast.
