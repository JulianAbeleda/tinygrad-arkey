# NV Q4_K x Q8_1 substrate arithmetic trace (fresh-eyes, llama-pinned)

Date: 2026-08-12
Branch: `nvidia-bringup-20260731` (HEAD `6cf7149b8`, post Q4 FFN-down
load-pattern sweep NO-GO record)
Status: **arithmetic confirmed down. Trace only; no implementation authorized by
this record. The route's blockers are the precision gate and the cooperative
provider wall cost, not the arithmetic.**

## 1. Why this trace exists

The Q4 FFN-down load-pattern sweep closed as NO-GO at `6cf7149b8`: no load-
pattern row on the installed scalar-fp16 body clears llama's floor (best
standalone quad 11.43-11.45 us -> 14.4-19.5 us in-loop estimate vs llama
11.776 us/node). The standing hypothesis is that llama's speed comes from the
DP4A/Q8_1 substrate (quantized int8 activations + int8 dot), so the arithmetic
of the dormant Q4/Q8 MMVQ candidate (`tinygrad/llm/q4k_ffn_down_mmvq.py`) was
re-traced from first principles against the pinned llama source
(llama.cpp `ac4cddeb0`, `/home/ubuntu/env/llama.cpp`) before any apply
decision. Process: trace llama -> arithmetic-validate -> e2e dependency map ->
implement (standing process).

## 2. Method

All comparisons are against the pinned llama source files `ggml/src/ggml-cuda/
vecdotq.cuh` and `ggml/src/ggml-quants.c`. Every claim below was re-derived in
this session from the source text; none are carried from memory. The
independent scalar oracle `q4_q8_ffn_down_row_reference`
(`extra/llm_research/decode/ffn_q8_cooperative_producer.py:229`) is the
ground truth for the packed ABI interpretation, and it passed at `3.4e-6` rel
against the UOp consumer in the existing suite.

## 3. Trace: llama's decode-path formula (OBSERVED)

`vec_dot_q4_K_q8_1` (vecdotq.cuh:864) -> `vec_dot_q4_K_q8_1_impl_vmmq`
(vecdotq.cuh:505), selected for Q4_K by `get_vec_dot_q_cuda`
(mmvq.cu:22). Per thread, per `i` in `QR4_K` (2 iterations on the decode path,
`VDR_Q4_K_Q8_1_MMVQ = 2`, mmvq.cu:50):

- `dot1 = dp4a(v1i, u[2i+1], dp4a(v0i, u[2i+0], 0))` where `v0i/v1i` are
  nibble-extracted q4 words.
- `dot2 = dp4a(0x01010101, u[2i+1], dp4a(0x01010101, u[2i+0], 0))` (exact
  byte sum of the q8 words).
- `sumf_d += d8[i] * (dot1 * sc[i])`; `sumf_m += d8[i] * (dot2 * m[i])`.
- Result: `dm4f.x * sumf_d - dm4f.y * sumf_m` (`d * sumf_d - dmin * sumf_m`).

The vmmq path does NOT use the stored q8 sum `s`; it recomputes the sum with
`dp4a(0x01010101, u)`. Only the tensor-core mmq path uses `ds8f.y`. Our
consumer matches the vmmq/decode path.

## 4. Trace: scale/min decode, byte-for-byte (OBSERVED)

llama `get_scale_min_k4` (ggml-quants.c:822), for scale index `j`:

| case | d (scale) | m (min) |
| --- | --- | --- |
| `j < 4` | `q[j] & 63` | `q[j+4] & 63` |
| `j >= 4` | `(q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)` | `(q[j+4] >> 4) | ((q[j] >> 6) << 4)` |

The `vec_dot_q4_K_q8_1` aux decode (vecdotq.cuh:879-884) is the same function
reached through the `uint16` pair view, and `quantize_row_q4_K_ref`
(ggml-quants.c:1399) packs `scales` exactly so this decode inverts it.

Our consumer (`emit_four_warp_direct`, `q4k_ffn_down_mmvq.py`), with words
`w1 = scales[0..3]`, `w2 = scales[4..7]`, `w3 = scales[8..11]` and
`g4 = group % 4`:

| group | our scale | our minimum |
| --- | --- | --- |
| `< 4` | `(w1 byte g4) & 63` = `scales[g] & 63` | `(w2 byte g4) & 63` = `scales[g+4] & 63` |
| `>= 4` | `(w3 byte g4 & 0xF) | ((w1 byte g4 >> 6) << 4)` = `(scales[g+4] & 0xF) | ((scales[g-4] >> 6) << 4)` | `(w3 byte g4 >> 4) | ((w2 byte g4 >> 6) << 4)` = `(scales[g+4] >> 4) | ((scales[g] >> 6) << 4)` |

Substituting `g` for llama's `j` in the table above, all four cells are
identical. The independent scalar oracle in `q4_q8_ffn_down_row_reference`
uses the same decode and agrees (suite, 3.4e-6 rel). **Byte-for-byte match.**

## 5. Trace: qpack nibble order (OBSERVED)

llama `quantize_row_q4_K_ref` (ggml-quants.c:1450): `q[l] = L[j+l] |
(L[j+l+32] << 4)` for each 32-byte chunk: byte `l` holds weight `l` in the low
nibble and weight `l+32` in the high nibble. `dequantize_row_q4_K`
(ggml-quants.c:1485) reads it back exactly that way.

llama's MMVQ thread walks the qpack quarter-strided: per `iqs`, `v[0]` =
`qs[off..off+3]`, `v[1]` = `qs[off+16..off+19]`, covering weights
`{0-3,16-19,32-35,48-51}` for `iqs/2=0`, etc. Our consumer walks it
block-diagonal: group `g` reads qpack words `4+(g//2)*8..+7` at nibble
offset `(g%2)*4`, covering weights `[32g, 32g+31]` paired with q8 group `g`
(x `[32g, 32g+31]`). Both partition the same `(weight, x)` pairs exactly
once, both keep the 32-value q8_1 group alignment, and both apply the same
per-group `(scale, min)` from section 4. The two orderings are equivalent
arithmetic; ours is the natural one for our provider's ABI.

## 6. Trace: signedness (OBSERVED, no trap)

`ggml_cuda_dp4a` lowers to `dp4a.s32.s32` (verified via nvcc PTX dump), which
reads bytes as signed int8. The q4 nibbles 0..15 extract as bytes `0x00..0x0F`
(after `& 0x0F0F0F0F`), all positive int8, so the nibble side is exact: `dot1`
is the true unsigned-nibble dot, and no `16 * sum(high-set nibbles)` term is
missing. The q8 side is genuinely signed, which is correct (x values are
signed). `dp4a(0x01010101, u)` therefore sums the q8 bytes exactly. The
dequant identity `d*sc*q - dmin*m` is preserved term-for-term in the vmmq
formula. llama's math is self-consistent and ours is term-identical.

## 7. Trace: provider (OBSERVED, byte-exact at pin time)

`emit_q8_provider` reproduces llama's CUDA `quantize_q8_1` bit-for-bit when
checked against the pinned sm_120 cubin (byte-exact verification recorded
08-05): `x/d` division spelling (not reciprocal), `roundf` ties-away (not
UOp.round ties-even), clamp to `[-128,127]`, fp16 `d` and `s` halves packed
low/high in the metadata word. The Q8_1 block ABI (`half d, half s, int8
qs[32]`) matches `block_q8_1`; the decode path consumes `d` only, which is
exactly what llama's vmmq does.

## 8. Verdict: arithmetic is down (OBSERVED)

All 17 arithmetic tests pass at this HEAD (`test/unit/test_q4k_ffn_down_mmvq.py`
+ `test/unit/test_ffn_q8_cooperative_producer.py`, `pytest -x -q`: 17 passed),
including the independent dense q4 x q8 scalar oracle, the provider
byte-exactness, the owned-fp32-boundary topology contract, and the `sum_dp4a`
PTX census (2 vs 12 static dp4a). The measured delta between the candidate
Q4/Q8 route and the fp16 control is Q8 quantization noise, not an arithmetic
bug. No arithmetic correction is available to apply; the arithmetic is not
the gap.

## 9. What the trace does NOT change (INFERRED, fail-closed)

The two hard blockers on this route are non-arithmetic and stand:

1. Precision gate: one-layer rel L2 `0.0020060` vs `0.001` maximum (08-05
   included-cost record). Block-13 localization (08-09) shows block 13 inert
   (`2.72e-4`) but blocks 14-18 carry `0.59-1.26` rel L2 and flip tokens;
   no lease expansion is authorized.
2. Provider wall cost: the cooperative 32-row W1/W3->Q8 producer is WALL
   NO-GO `+151.191835 us` (candidate 229.715160 vs control midpoint
   78.523325 us/replay, 2.93x regression, 08-05 static record). Reopen
   requires a distinct cooperative mapping no slower than the installed
   W1/W3 + fp16 boundary.

The load-pattern sweep record (`6cf7149b8`) already closed the scalar body,
so the remaining untried surface is the substrate composition itself, not
the installed kernel body.

## 10. Apply options (each needs its own gate; no code changed here)

| option | mechanism | state | next action |
| --- | --- | --- | --- |
| A | owned-fp32-boundary MMVQ (`owned_boundary_topology()`, provider consumes fp32 directly, removes the fp16 materialization; 3 control nodes vs 3 candidate nodes, net 0 graph members) | dormant, built, gpu-gate contract drafted (exactly 875 calls) | fresh scope + singleton wall gate |
| B | distinct cooperative producer mapping no slower than installed boundary | closed NO-GO; reopen condition recorded | fresh semantic + settled wall gate only after a mapping is identified |
| C | stop: quant GEMV row exhausted except closed mechanisms; spend remaining campaign mass on the 2.27x FFN-down ratio only through a measured substrate arm | current ledger state | no further action from this record |

Recommendation (INFERRED): option A is the only untried, llama-anchored
mechanism on the Q4 FFN-down row; it does not touch the admitted topology, the
precision lease, or any closed record, and its gate contract is already
drafted in `owned_boundary_topology()`. It is a new mechanism and therefore
requires the standing gate sequence (semantic -> wall) before any promotion.
No GPU time was used in this trace; no `flock` was required.

## 11. References

- `nv-quant-gemv-llama-audit-20260812.md` (per-shape deficit cross-reference;
  FFN-down Q4 2.27x ratio, 481.50 us/token)
- `nv-q4k-ffn-down-mmvq-included-cost-and-one-layer-record-20260805.md`
  (standalone 13.952 vs 22.016 us; included 15.648 us; precision gate fail)
- `nv-p3b-ffn-q8-cooperative-producer-static-record-20260805.md` (WALL NO-GO
  +151.191835 us)
- `nv-q4-block13-precision-localization-record-20260809.md` (block 13 inert;
  14-18 flip tokens)
- `nv-q4k-ffn-down-layer8-rebracket-record-20260809.md` (singleton WALL WIN
  -3.190 us at post-census HEAD; two-layer subset still closed)
- `nv-q4kd-load-pattern-measurement-record-20260812.md` (NO-GO sweep)
- llama.cpp `ac4cddeb0` pinned source: `ggml/src/ggml-cuda/vecdotq.cuh`,
  `ggml/src/ggml-cuda/mmvq.cu`, `ggml/src/ggml-quants.c`

