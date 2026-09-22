<!-- Copyright (C) 2026 tinycorp, AGPL-3.0-or-later. -->
# Non-norm copy inventory census - flash decode graph

Date: 2026-08-02

Status: census (analysis + measurement + classification only; no repo code changes).

Branch: `nvidia-bringup-20260731`. GPU: NV sm_120. Model: Qwen3-8B-Q4_K_M, d512 prime token.

Probes: `/tmp/non_norm_copy_census_probe.py` (DEBUG=2 kernel histogram, mirrored from
`/tmp/m3_census_probe.py`), `/tmp/non_norm_uop_probe.py` (execute_promoted_program input
UOp chain capture), `/tmp/non_norm_source_probe.py` (per-arg UOp source walk).

---

## 1. Census table

| # | class | count | median us | class total ms | producer | consumer | intermediate uop chain | classification |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | `E_32_32_4_0a5e` | 36 | 1.60 | 0.058 | `flash_fused_gmax_combine` (fp32, 4096) | `q4k_g3_lanemap_gemv_4096_4096` (attn_output, fp16) | TRANSPOSE + RESHAPE triggers contiguous copy of fp32 combine output; downstream CAST(fp16) fused into GEMV | **B** (layout-copy + downstream cast; the copy is pure layout but the cast is real computation) |
| 2 | GEMV activation CONTIGUOUS chain (Q4K) | ~216 | N/A (fused) | N/A (fused) | norm epilogue `E_32_32_4_f14a` or residual add `E_32_32_4_02a9` / `E_32_32_4_fab` (all fp32) | `q4k_g3_lanemap_gemv_*` (fp16 input) | `CONTIGUOUS(CAST(fp16, RESHAPE, RESHAPE, MEMORY_SEMANTIC(MUL/ADD)))` -- fused into GEMV launch | **B** (CAST fp32->fp16 is real computation; CONTIGUOUS alone would be A but is inseparable from the cast in this graph) |
| 3 | GEMV activation CONTIGUOUS chain (Q6K) | ~36 | N/A (fused) | N/A (fused) | norm epilogue or residual add (fp32) | `q6k_gen_partial/coop_*` (fp16 input) | Same chain as class 2 | **B** (same reasoning) |
| 4 | Flash q input view | 36 | 0.00 | 0.000 | qk_norm output `ADD` (fp32, 1,32,1,128) | `flash_block_tiled_xlane_score_pv_tile` (fp32 input) | `RESHAPE(RESHAPE(MEMORY_SEMANTIC(ADD)))` -- pure view, resolved as no-op by transport | **A** (contiguous view of already-produced buffer; already free) |
| 5 | Flash cache_kv input | 36 | 0.00 | 0.000 | KV cache `STORE` / `ASSIGN` (fp16, 2,1,8,4608,128) | `flash_block_tiled_xlane_score_pv_tile` | `AFTER` -- direct buffer identity, no copy | **A** (already direct buffer access) |
| 6 | `E_16_32_4_2` (k/v fp16 cast) | 36 | 2.24 | 0.081 | `q4k_g3_lanemap_gemv_1024_4096` / `q6k_gen_partial` (fp32 output) | `flash_block_tiled_xlane_score_pv_tile` via KV cache store (fp16) | k/v GEMV output fp32 -> fp16 cast for cache write | **B** (real computation: fp32->fp16 cast; absorbable into k/v GEMV epilogue per design doc section 6) |
| 7 | `E_32_32_4_02a9` (residual add x+attn_out) | 72 | 1.63 | 0.117 | `q4k_g3_lanemap_gemv_4096_4096` (attn_output, fp32) + layer input x (fp32) | next-layer `r_16_256` norm reduce + `q4k_g3_lanemap_gemv_*` | `ADD` -- real fp32 addition | **B** (real computation; absorbable into o-proj GEMV epilogue) |
| 8 | `E_32_32_4_fab` (residual add h+ffn_out + contiguous) | 72 | 1.66 | 0.120 | ffn_down GEMV output + residual h (fp32) | next-layer norm + layer-output contiguous | `ADD` -- real fp32 addition | **B** (real computation; absorbable into down GEMV epilogue) |
| 9 | `E_128_32_3_2ba5` (silu(gate)) | 36 | 2.05 | 0.074 | `q4k_g3_lanemap_gemv_12288_4096` (ffn_gate, fp32) | `E_128_32_3_4a0d` (silu(gate)*up) + ffn_down GEMV | `SILU` -- real computation | **B** (real computation; absorbable into ffn_down GEMV prelude) |
| 10 | `E_128_32_3_4a0d` (silu(gate)*up) | 36 | 1.60 | 0.058 | `q4k_g3_lanemap_gemv_12288_4096` (ffn_up, fp32) + silu(gate) | ffn_down GEMV | `MUL` -- real computation | **B** (real computation; absorbable into ffn_down GEMV prelude) |

**Classification key**: **A** = contiguous view of an already-produced buffer (potentially fixable by an
opt-in input ABI mode in KernelProgram). **B** = lazy producer requiring real computation
(NOT a transport fix; requires emitter changes, epilogue absorption, or dtype contract changes).

**Fused classes** (2, 3): these do not appear as standalone kernels in the DEBUG=2 trace. Their
cost is embedded in the consuming GEMV kernel time. The UOp source probe confirms every GEMV
fp16 activation input carries `CONTIGUOUS(CAST(fp16, RESHAPE, RESHAPE, MEMORY_SEMANTIC(...)))`.
The CAST is the real work; the CONTIGUOUS is layout only. In the M2-on baseline, upstream
buffers are fp32, so the cast is always required. With Path 3 (fp16 norm output) or Path 4
(fp16 GEMV epilogue output), the cast would become a no-op and the CONTIGUOUS would resolve
as a free view -- but that belongs to those paths, not Path 1.

---

## 2. Classification analysis

### 2.1 Category A census (contiguous view of already-produced buffer)

Two classes qualify:

- **Flash q input** (class 4, 36x, 0us): `q.reshape(Hq * Hd)` at
  [flash_decode_attention.py:502](/home/ubuntu/tinygrad-arkey/tinygrad/llm/flash_decode_attention.py:502).
  The uop chain is `RESHAPE(RESHAPE(MEMORY_SEMANTIC(ADD)))` -- pure views on the q buffer. The
  custom-kernel transport resolves this as a no-op because the underlying buffer is contiguous
  and the right dtype. **Already free; no transport work needed.**

- **Flash cache_kv input** (class 5, 36x, 0us): direct `AFTER` buffer access. **Already free.**

**Total category-A tax: 0us, 0 kernels, 0.000ms.**

### 2.2 Category B census (lazy producers requiring real computation)

Every other copy/materialization class involves actual computation:

| mechanism | classes | total us | absorber path |
| --- | --- | ---: | --- |
| fp32->fp16 cast (layout copy often co-triggered) | 1, 2, 3, 6 | ~140us (class 1 only; 2-3 fused, 6 standalone at 81us) | Path 3 fp16 norm output; Path 4 GEMV epilogue fp16 write |
| fp32 elementwise (add, mul, silu) | 7, 8, 9, 10 | ~369us | Path 4 GEMV epilogue absorption (design doc sections 2, 6) |
| Layout-copy (transpose -> reshape contiguous) | 1 | ~58us | disappear if fp16 combine output OR if attn_output GEMV accepted strided input |

The layout-copy portion of class 1 (E_32_32_4_0a5e) is the only pure "contiguous view" sub-tax
in the entire graph. At 57.6us, it is below the 0.1ms launch-floor threshold. Even this class
cannot be fixed by transport alone because fixing it requires either (a) the flash combine
emitter to output fp16 directly, making the downstream cast a no-op, or (b) the attn_output
GEMV to accept a strided fp32 input and do the cast in-kernel -- both emitter changes, not
transport changes.

### 2.3 Why the input-boundary `contiguous()` calls in decode_routes.py are not category A

The `_xv` and `x_vec` expressions at
[decode_routes.py:73,119](/home/ubuntu/tinygrad-arkey/tinygrad/llm/decode_routes.py:73)
are `x[:, 0, :].reshape(binding.K).cast(dtypes.float16).contiguous()`. The UOp probe confirms:

```
Ops.CONTIGUOUS shape=(4096,) dtype=half
  Ops.CAST shape=(4096,) dtype=half        <-- real computation
    Ops.RESHAPE shape=(4096,) dtype=float
      Ops.RESHAPE shape=(1, 4096) dtype=float
        Ops.MEMORY_SEMANTIC shape=(1, 1, 4096) dtype=float
          Ops.MUL shape=(1, 1, 4096) dtype=float   <-- norm epilogue
```

The `CONTIGUOUS` is the outermost op, but it wraps a `CAST` from fp32 to fp16. In the M2-on
baseline, all upstream buffers are fp32 (legacy norm output, residual adds), so the cast is
never a no-op. The scheduler fuses this chain into the GEMV kernel; there is no separate
standalone copy kernel for these calls. Their cost is embedded in GEMV kernel times and
cannot be isolated without an alternative (cast-free) implementation.

If Path 3 produced fp16 norm output, the `CAST` would become a no-op and the `CONTIGUOUS`
would resolve as a free view -- but that is a Path 3/Path 4 win, not a Path 1 (transport) win.

---

## 3. Path 1 go/no-go recommendation

**Recommendation: NO-GO for Path 1 as a standalone transport proposal.**

### 3.1 Evidence

| claim | evidence |
| --- | --- |
| Category-A copies (contiguous views) total 0 | Flash q input and cache_kv input are already resolved as free views by the existing transport (UOp probe confirms `RESHAPE` chain with no `CONTIGUOUS` or `CAST`). |
| All other copy classes involve real computation | Every other class (1-3, 6-10) carries `CAST`, `ADD`, `MUL`, or `SILU` -- none are pure layout copies. |
| The one pure-layout copy is below threshold | `E_32_32_4_0a5e` at 57.6us class total (< 0.1ms) is the only layout-only copy, and fixing it requires a dtype contract change (fp16 combine output), not transport. |
| GEMV input CONTIGUOUS chains are cast-bound | The `_xv`/`x_vec` calls fuse `CONTIGUOUS(CAST(...))` into the GEMV; removing the `CONTIGUOUS` without removing the `CAST` saves nothing. |
| No measured copy tax warrants a new ABI mode | Total standalone copy kernel time attributable to pure contiguous-making is 57.6us (class 1 layout portion); total fused copy tax is inseparable from cast cost and < 0.1ms estimated. |

### 3.2 What evidence would change the recommendation

1. **A measurement showing > 0.2ms of standalone layout-copy kernels** (not cast-triggered, not
   elementwise) in a graph variant where upstream buffers are already the right dtype. The
   closest scenario is Path 3 (fp16 norm output) -- if after Path 3, there are still separate
   `E_*` kernels doing only `CONTIGUOUS(RESHAPE(...))` with no `CAST`, that would reopen Path 1.
2. **A flash/GEMV consumer that reads an already-fp16 buffer but pays a separate copy kernel**
   because the custom-kernel transport cannot resolve a strided view. No such consumer exists
   in the current graph; all fp16 consumers either get a fused copy or a free view.
3. **A new emitter** (beyond the current decode family) whose inputs are always the right dtype
   but whose flat-buffer contract forces a contiguous copy of an otherwise-valid strided
   buffer. This is a forward-looking concern, not a current measurement.

### 3.3 Relationship to other paths

| path | interacts with this census how |
| --- | --- |
| Path 3 (generic in-kernel norm) | Would produce fp16 norm output, collapsing the `CAST` in GEMV input chains. The `CONTIGUOUS` would then resolve as a free view -- a Path 3 win, not a Path 1 win. |
| Path 4 (GEMV/flash epilogue absorption) | Would absorb classes 6-10 into the GEMV/flash emitters. Removes the standalone E_ kernels entirely. No transport change needed. |
| Path 2 (launch overhead) | Orthogonal; launch cost is host-side, this census is device-side copy tax. |

---

## 4. Risks

1. **The census is M2-on-baseline only.** If M3 (fused norm) were reopened, the 144 input-boundary
   copies and 72 output materializations documented in the norm-fusion scope would ADD
   category-B copies to the graph. Those are norm-path artifacts (cast-triggered by fp32
   upstream), not a reason to build Path 1. Path 3 should close before M3 reopens.
2. **Fused copy tax is estimated, not measured.** Classes 2 and 3 are embedded in GEMV kernel
   times. The CONTIGUOUS(CAST) chain's cost cannot be isolated without a reference kernel that
   omits it. The estimate (< 0.1ms total) is based on bandwidth arithmetic: ~24KB per
   activation input (16KB fp32 read + 8KB fp16 write) at ~1.5 TB/s = ~16ns per copy x ~234
   inputs = ~3.7us, which is within timing noise of the GEMV kernels themselves.
3. **The classification is dtype-contingent.** If future model variants use fp16 throughout
   (norms output fp16, residual in fp16), the classification shifts: classes 2, 3, and 6
   become category A (contiguous view only). But at that point the copies would already be
   free views under the existing transport, as the flash q input demonstrates.

---

## 5. Probe artifacts

| artifact | path | role |
| --- | --- | --- |
| Kernel histogram (baseline) | `/tmp/non_norm_copy_census_probe.py` | Reuses `/tmp/m3_census_probe.py` pattern; captures DEBUG=2 per-kernel times for the current M2-on-closed state |
| UOp input census | `/tmp/non_norm_uop_probe.py` | Intercepts `execute_promoted_program`; captures per-arg shapes, dtypes, and base uop ops |
| UOp source walk | `/tmp/non_norm_source_probe.py` | Walks the uop src chain for each program arg; identifies CAST/CONTIGUOUS/RESHAPE chains |
| Kernel histogram output | `/tmp/m3_census_closed.out` | 1021 kernels/token, 6178us total (this run) |
| Full DEBUG=2 trace | `/tmp/debug_decode_probe.log` | Per-kernel trace with timestamps |

No repo files modified. No commits.
