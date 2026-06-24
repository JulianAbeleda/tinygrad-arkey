# Primitive 1 — gqa_coop coalesced/vectorized loads: RESULT = SHIPPED (default `gqa_coop_vec`) 2026-06-17

Roadmap Primitive 1. Goal: make gqa_coop load K/V like llama (coalesced fp16) instead of scalar/strided.
Outcome: **far exceeded the +3-5% estimate** — the real issue wasn't load *width*, it was that gqa_coop ran as
**1-thread workgroups**. Shipped as the new default. RX 7900 XTX, Qwen3-8B-Q4_K_M, byte-identical greedy.

## Phase 0 — load-pattern audit (the real finding)

Rendered `flash_partial_coop` (DEBUG=4): `amdgpu_flat_work_group_size(1, 1)` and only `__ockl_get_group_id`
(grid), no `local_id`. The output-dim `d` (W=Hd+1=129) was an `AxisType.GLOBAL` (grid) axis → **each work-item
is its own 1-thread workgroup**: the per-`d` V load `half val0 = *(data2 + (kvh*MAXC+t)*Hd + d)` is scalar fp16
with **no wavefront coalescing and ~1/32 lane utilization** (each `d` is a separate wavefront). This — not load
width — is the bottleneck. (Both gqa_coop and the old hoisted/v2 share this 1-thread-workgroup pathology.)

## The fix — map `d` to LOCAL threads (`gqa_coop_vec`)

`flash_partial_coop_vec_kernel`: identical math, but `d = UOp.range(W, 2, AxisType.LOCAL)` instead of GLOBAL.
Now W=129 `d`-lanes run as workgroup threads (grid = Hkv×S), so adjacent lanes read adjacent `V[...+d]` →
**coalesced fp16 loads + full lane utilization**. No LDS/barriers, no extra kernels. Bit-identical to v1
(self-test max|diff|=0).

## Isolated (advisory)

Full-attention `gqa_coop_vec` vs `gqa_coop`, DEBUG2 tm, err 0: KV 512/1024/2048/4096 = **1.60× / 1.97× / 2.70×
/ 3.28×**. (Unlike the gqa_coop *partial*-only probe — whose isolated 3× was cache-inflated — this full-attn
isolated agreed with in-model, because the attention share at long ctx is large.)

## In-model W==D (authoritative) — all gates massively cleared

| ctx | gqa_coop | **gqa_coop_vec** | speedup | % of llama |
|---|---|---|---|---|
| 512  | 44.8 | **47.7** | +6.5% | 45% → **48%** |
| 1024 | 41.4 | **46.9** | +13.3% | 42% → **48%** |
| 2048 | 36.4 | **45.7** | +25.5% | 38% → **48%** |
| 4096 | 29.5 | **43.9** | **+48.8%** | 32% → **48%** |

Byte-identical greedy at every ctx; W≈D (GPU-bound, real). Gates (≥3%@1024 / ≥5%@4096 / no ctx512 regression):
**all passed by wide margins**.

**Slope: −8%** (47.7→43.9 across ctx512→4096), vs gqa_coop −34% and **llama −7%.** The decode-attention **slope
gap is essentially CLOSED** — tinygrad is now ~**48% of llama FLAT across all contexts** (was 45/42/38/32
decaying). The remaining gap is the **base gap** (~48% flat = GEMV/decode-block structural, per the
GEMV-structural audit which found no bounded target).

## Default decision

Flipped default `FLASH_VARIANT` and `FLASH_DECODE_DEFAULT_VARIANT` → **`gqa_coop_vec`** (strictly dominates
gqa_coop at every ctx, exact). `FLASH_VARIANT={v1,hoisted,gqa_coop,gqa_coop_vec}` overrides remain.

## Next action (roadmap)

- **Primitive 2 (Stream-K) value is now SMALL** — its purpose was to flatten the slope, which gqa_coop_vec
  already did (−8% ≈ llama). Re-assess: likely skip or expect <gate.
- The remaining gap is the **base gap** (GEMV/decode-block), which the structural audit refuted as bounded.
- Higher-EV directions: Primitive 7 (prefill WMMA, different phase) or the 14B/32B matrix.

## Files / commits

`extra/qk_flash_decode.py` (`flash_partial_coop_vec_kernel` + SSOT + wiring + self-test), `tinygrad/llm/model.py`
(default), `test/external/test_qk_flash_decode_policy.py`. Commits: `[codegen] add gqa_coop_vec`, `[nn] default ->
gqa_coop_vec`, `[test]`, `[docs] this`.
