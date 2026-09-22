# NVIDIA prefill Q4-down debug-main prerequisite - C0 (2026-08-29)

## Verdict: BLOCKED / STOP

The requested diagnostic CUDA main is not launchable through the current
compiler path.  The packet-specific probe is
`extra/llm_research/prefill/nv_q4down_debug_main_prerequisite.py` and uses the
distinct diagnostic identity `nv_q4k_down_debug_main_type12_v1`; it never
changes or launches the production asset.

## Compiler boundary

The compiler-generated `DownAsset.main_program` is a precompiled ELF with the
fixed ABI `(out, record, words)`: one final FP32 output, one compact-Q8 record,
and packed Q4 words.  The compiler API provides no source-level main body,
debug-output pointer, stage selector, or additional output slot.  Decoded Q4
metadata and corrected K32 subtotals are register-local, and the pre-epilogue
accumulator has no store before the final epilogue.

| Requested value | Launchable now? |
|---|---|
| decoded Q4 metadata | no |
| per-K32 corrected subtotal | no |
| pre-epilogue FP32 output | no |
| final output | yes, existing production output |

Adding buffers would require recompiling from a source-level CUDA main and
changing the kernel ABI.  That is outside this diagnostic-only prerequisite
and would no longer be the compiler-owned production main.  No arithmetic,
geometry, routing, cache identity, or defaults were changed.

## Next C0 command

There is no valid next launch command for the requested buffers.  C0 remains
STOP/BLOCKED until a separately supplied source-level CUDA main (with explicit
debug pointers and a distinct compiled identity) is available.  The existing
`nv_q4down_matched_ab.py` may continue to compare only compact-record and
final-output boundaries; it cannot identify the first internal mismatch.
