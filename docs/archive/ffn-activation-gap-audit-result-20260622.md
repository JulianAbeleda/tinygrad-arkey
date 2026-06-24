# FFN Activation Gap Audit — Result (2026-06-22)

## Verdict: **FFN_ACTIVATION_GAP_IS_MAPPING_ARTIFACT**

The diff's "FFN activation" 10–20× gap is **not real as mapped**. The silu activation is **fused into
the gate/up GEMV**; the kernels the diff labelled "ffn_activation" (`E_49152`, `E_1536`) are **pure
buffer copies** — specifically a **full-`max_context` KV-cache rematerialization** that is the single
largest *bounded* 8B decode opportunity (handed to the decision doc). Audit only; default unchanged.

## Evidence

### 1. silu is fused — there is no standalone activation kernel
The gate/up Q4_K GEMV `q4k_gemv_partial_12288_4096_1` contains the exponential
(`src_flags.exp = True`): silu(gate)·up is computed in the GEMV epilogue. No separate silu kernel
exists. tinygrad's standalone activation cost ≈ 0, same as llama folding it. **Real activation
gap @ctx1024 = −0.13 ms** (tinygrad is not slower than llama's `unary_gated`).

### 2. The "activation" kernels are pure copies, not silu
Rendered source of `E_49152_32_3` (1419 µs/tok @ctx1024):
```c
float val0 = data1[alu0+4718592]; ... ; data0[alu0] = val0; data0[alu0+1]=val1; data0[alu0+2]=val2;
```
Pure float→float move — **no exp / sqrt / sin / quant** (`op_hist`: only LOAD/STORE/INDEX/ADD/MUL-for-indexing).
`E_1536_32_3` (142 µs) is the same pattern. Both are data movement, not activation.

### 3. It is a full-`max_context` KV-cache copy (MAXC-bound, redundant)
| ctx | E_49152 µs/tok | flat? |
|---|---|---|
| 512 | 1403 | |
| 1024 | 1419 | **flat across ctx** |
| 4096 | 1417 | ⇒ MAXC-bound, not ctx-bound |

MAXC-shrink test (decisive identification): `max_context=4608 → E_49152` (49152·96 = 4718592 = 4608×1024 = MAXC×kv_dim, 1420 µs);
`max_context=1152 → E_12288` (1152×1024, 375 µs). The copy **scales exactly with `max_context`** (4× MAXC → 3.79× cost).
It copies the entire static KV buffer every decode step regardless of actual context length — O(MAXC) where an
in-place append is O(1).

### 4. It is on the critical path (transfers to wall)
Wall transfer test (ctx≈430, default route, 40-sample median wall token_ms):
| max_context | E_49152 µs | wall token_ms | tok/s |
|---|---|---|---|
| 4608 | 1420 | 14.552 | 68.7 |
| 1152 | 375 | 13.051 | 76.6 |

Shrinking the copy by ~1.05 ms reduces **wall by ~1.5 ms (+8 tok/s)** → the copy is on the serial
critical path and transfers ~1:1. (The extra ~0.45 ms is other small MAXC-scaled buffer ops.)

## Answers to the four questions
| question | answer |
|---|---|
| Real? | **No** — no expensive activation op; silu is fused. |
| Mapped correctly? | **No** — `E_49152`/`E_1536` are buffer copies, not silu. |
| Critical-path? | The *copy* is (transfers +1.5 ms); the *activation* is fused (≈0). |
| Bounded (as activation)? | **No FFN-activation primitive exists to build.** |

## Reclassification & handoff
The ~1.56 ms/token in the "ffn_activation" bucket = **KV-cache rematerialization** (`E_49152`+`E_1536`),
failing layer `tinygrad/llm/model.py:952` (`cache_kv.uop.after(slice.store(...))` materializes the full
MAXC buffer; the in-place `.assign()` is commented out at 956–958, a `@function(precompile=True)` purity
workaround — upstream idiom from refactor #15780). This is the largest bounded opportunity → see
`docs/8b-exhaustion-next-implementation-decision-20260622.md`.

## Artifacts
`extra/qk_ffn_activation_gap_audit.py`, `bench/qk-ffn-activation-gap-audit/latest.json`,
`bench/qk-decode-kernel-probe/latest.json` (sources + fingerprints + timeline).
