# llama vs tinygrad Q4_K MMVQ inner-loop audit (2026-06-18)

Source/runtime audit only (no kernels). Proves *what llama does differently* in Q4_K ffn_gate/up to reach ~70%
HBM peak while tinygrad sits at 40-48%. RX 7900 XTX (RDNA3/gfx1100), Qwen3-8B-Q4_K_M, llama `b9592`.

## Measured context

| variant | GB/s | % HBM peak |
|---|---|---|
| tinygrad base fp-dequant | 363-365 | 40-41% |
| tinygrad fp coop (coalesced) | 431 | 48% |
| tinygrad dp4a/q8 Family A | 354 | 39% |
| READRAW (no dequant) | 632 | 70% |
| llama MMVQ | 626 | 70% |

## Phase 0 — llama Q4_K decode source map (RDNA3)

| stage | file | function | role |
|---|---|---|---|
| dp4a intrinsic | `common.cuh:694` | `ggml_cuda_dp4a` → `__builtin_amdgcn_sdot4` | 4×int8 MAC (RDNA3 path) |
| Q4_K vecdot caller | `vecdotq.cuh:864` | `vec_dot_q4_K_q8_1` | loads packed v/u/scales, calls impl |
| Q4_K inner loop | `vecdotq.cuh:505` | `vec_dot_q4_K_q8_1_impl_vmmq` | the dot (`VDR_Q4_K_Q8_1_MMVQ=2`) |
| q8_1 activation quant | `quantize.cu` | `quantize_q8_1` | activations → int8 + per-32 scale, once |
| MMVQ wrapper | `mmvq.cu` | `mul_mat_vec_q` | one kernel/linear, row per thread-group |

## Phase 1 — llama Q4_K inner loop (annotated)

```c
// caller (vec_dot_q4_K_q8_1): NO per-nibble work at load
v[0] = q4[0];  v[1] = q4[4];            // 2 int reads = 8 packed weights' nibbles
u[0..3] = q8[0], q8[4], ...;            // q8 activations read as ints (4 q8/int)
aux[0] = scales[j] & 0x3f3f;           // decode TWO 6-bit scales in ONE packed-16b AND
aux[1] = scales[j+2] & 0x3f3f;         // (high case: &0x0f0f | (&0xc0c0)>>2 -- still packed)
sc = (uint8*)aux; m = sc+2;            // per-group scale(sc) + min(m)

// impl (vec_dot_q4_K_q8_1_impl_vmmq):  QR4_K = 2
for (i=0; i<2; ++i) {
  v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;  // extract 4 nibbles in ONE shift + ONE AND
  v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;
  dot1 = dp4a(v1i, u[2i+1], dp4a(v0i, u[2i], 0));      // 4-wide int dot (2 dp4a)
  dot2 = dp4a(0x01010101, u[2i+1], dp4a(0x01010101, u[2i], 0)); // q8 SUM via dp4a-with-ones
  sumf_d += d8[i] * (dot1 * sc[i]);    // per-GROUP scale apply (not per-weight)
  sumf_m += d8[i] * (dot2 * m[i]);     // min-correction term
}
return dm4f.x*sumf_d - dm4f.y*sumf_m;  // block d/dmin applied ONCE
```

## Phase 2 — tinygrad Q4_K ffn_gate/up inner loop (annotated)

```python
# _q4k_quant(grp,pos):  ONE nibble per call (SCALAR)
qword = words[base + 4 + (grp//2)*8 + pos//4]
q     = (qword >> ((pos%4)*8 + (grp%2)*4)) & 0xf          # 1 shift + 1 AND per WEIGHT
# _q4k_group_params(grp): per-byte scale gymnastics (scale_byte shifts), per group
# _q4k_weight(grp,pos):
w = d * sc * float(q) - dmin * mn                          # int->FP convert + per-WEIGHT fp affine
# _q4k_block_dot: for grp in 8: contrib += w(grp,pos) * float(x[...])   # per-weight fp MAC
```
Each weight: 1 scalar nibble extract + 1 int→fp + ~3 fp (affine) + 1 fp madd. No dp4a; fp accumulator.

## Phase 3 — operation accounting (per 256-weight Q4_K block, approx)

| operation | llama | tinygrad | effect | tinygrad can express? |
|---|---|---|---|---|
| bytes loaded | 144 (block once) | 144 | same | — |
| nibble extraction | **~64 ops** (packed `&0x0F0F0F0F`, 4 nibbles/op) | **~512 ops** (scalar shift+AND, 1 nibble/op) | **~8× more (extra work)** | YES (uint shift+AND on int) |
| scale/min decode | ~6 packed ops (`&0x3f3f`) | per-group byte gymnastics ×8 | more in tinygrad | YES |
| int dp4a | ~128 (64 dot + 64 qsum) | 0 | llama intrinsic | YES (`__builtin_amdgcn_sdot4`) |
| int→fp conversions | 0 (stays int) | **~256** (per weight) | **extra work** | YES (avoidable) |
| fp mul/add | ~32 (per-group) | **~768** (per-weight affine+madd) | **~24× more (extra work)** | YES (per-group instead) |
| scale application | per-GROUP (1/32 weights) | per-WEIGHT | extra work | YES |
| q8 sum (min term) | dp4a(0x01010101) | n/a (fp path) | llama trick | YES |
| accumulators | sumf_d, sumf_m, sumi | 1 fp acc | similar | — |
| work decomposition | row per thread | row per thread (base) / lane4 (coop) | same | — |

## Phase 4 — true gap classification

**A (less mathematical work) + C (packed-integer tricks)** — dominant. **B (ILP)** secondary. NOT F (measurement
is real). NOT hand-scheduled asm.

- **A — less work:** llama keeps everything INT (dp4a) and applies scales **per-group** (1 per 32 weights);
  tinygrad converts every nibble to fp and applies the affine **per-weight** (~256 int→fp + ~768 fp/block vs
  llama's ~32 fp). Evidence: `impl_vmmq` (per-group `sc[i]`/`m[i]`, final `dm4f` once) vs `_q4k_weight`
  (per-weight `d*sc*q - dmin*mn`). Maps directly to the 48→70 gap: tinygrad's ~10× ALU caps effective BW below
  the 632 GB/s read roofline; llama's cheap ALU stays read-bound at 70%.
- **C — packed int tricks:** `(v>>4i)&0x0F0F0F0F` (4 nibbles/op), `dp4a` (4 MAC/op), `dp4a(0x01010101,..)`
  (q8-sum), `scales&0x3f3f` (2 scales/op). tinygrad extracts nibbles/scales scalarly. **All portable** — they
  are plain integer ops + the `sdot4` builtin tinygrad already has (`_vdot4_q4_q8_accum`), no inline-asm-only
  idiom.
- **Why Family A (+0.6%, 39%) failed:** it used dp4a but built the q4 u32 with **4 scalar shift+mask+or per
  group** (≈ the scalar extraction cost) instead of the **`&0x0F0F0F0F` packed extract**, and kept overhead that
  offset the dp4a. It missed the *specific* trick (packed nibble extract in dp4a-aligned order).

## Phase 5 verdict (full plan in `q4k-unpack-ilp-safe-rewrite-plan-20260618.md`)

The gap is **portable** (packed-int extraction + dp4a + per-group scale), NOT asm-only. **One scoped attempt is
earned:** a custom_kernel replicating llama's exact inner loop — read packed weight ints, `(v>>4i)&0x0F0F0F0F`,
dp4a dot + dp4a(0x01010101) qsum, per-group `sc`/`m`, block `dm` once — in the coalesced lane structure. This is
the specific thing every prior dp4a attempt missed. **Risk:** tinygrad codegen may still not schedule the dp4a +
packed ops as tightly as llama's hand-unrolled `v[2]/u[4]` (prior dp4a wins evaporated in-model). Hard isolated
gate, single-attempt budget.

## Files
This doc; `bench/qk-q4k-unpack-ilp/llama_inner_loop_audit.json`. No code/model changes.
