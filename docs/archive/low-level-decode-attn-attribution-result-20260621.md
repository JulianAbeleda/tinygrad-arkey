# Low-Level Decode-Attention Attribution — Result

Date: 2026-06-21

Diagnostic-only (no kernel/model change). The `FRONTIER_LOW_LEVEL_TOOLING_FIRST` first gate: counter/ISA attribution
of why tinygrad's decode-attention softmax/V kernels are ~5× slower than llama's fused `flash_attn_tile`. Artifacts:
`bench/qk-low-level-decode-attn-attribution/`.

## Decision: **`LOW_LEVEL_ATTRIBUTION_FIXABLE_CODEGEN`**

The gap is **fixable codegen quality, not a fundamental limit**: tinygrad's flash kernels emit **scalar, un-tiled,
no-`v_dot2`/no-LDS** code (latency-bound at 201 GFLOPS / 60 GB/s), while llama's tile does the same work with
**LDS-staged K/V + dense `v_dot2_f32_f16`** in one fused kernel. Crucially, every tinygrad kernel runs at **100%
occupancy, ≤13 VGPR, 0 spills** — so it is **not** occupancy, registers, or spills. The named lever: route the
dominant PV/softmax-weighting through tinygrad's **tiled-matmul codegen** (which already makes the q·k matmul fast)
instead of the hand-rolled scalar `flash_partial`.

## Tool availability & reliability

| tool | result |
|---|---|
| `llvm-objdump -d` (ISA, tinygrad libs + llama `.co`) | **WORKS** — full per-kernel ISA + histograms |
| kernel-descriptor parse (VGPR/SGPR/LDS/scratch/spill) | **WORKS** (`llvm-readelf --notes`); 0 spills everywhere |
| `roc-obj-ls`/`-extract` (llama code object) | **WORKS** (hipv4-gfx1100) |
| `rocprofv3 --pmc` on **llama** | **PARTIAL** — `SQ_WAVES`/`SQ_BUSY_CYCLES` real; `SQ_INSTS_VALU/LDS`, `GRBM_GUI_ACTIVE` return **0** (unsupported on gfx1100+rocprofv3-7.2.4) |
| `rocprofv3` on **tinygrad** | **FAILS** — does not hook tinygrad's HCQ/direct-ring dispatch (zero output) |
| `rocprof-compute` | **BROKEN** (astunparse/plotext dependency) |

**Counter reliability:** live VALU/cache/LDS counters are **not obtainable** (rocprof-compute broken; rocprofv3
returns 0 for compute counters and is blind to tinygrad's HCQ). But **ISA + resources + occupancy + ProfileGraphEvent
durations were fully obtainable** and are sufficient to attribute the gap without live counters. (Not tooling-opaque —
the binaries told the story.)

## Per-kernel attribution (ctx1024)

| kernel | VGPR | LDS | spill | occupancy | µs | note |
|---|--:|--:|--:|--:|--:|---|
| **flash_partial_coop_vec** (PV) | 13 | **0** | 0 | 100% | **24.7** | 0 `v_dot2`, scalar fp16 V loads → 201 GFLOPS, latency-bound |
| matmul q·k (`r_8_4…`) | 21 | **64** | 0 | 100% | 13.9 | **tiled GEMM → fast** (≈ llama whole tile) |
| flash_prob | 3 | 0 | 0 | 100% | 7.8 | elementwise exp, memory-bound |
| flash_combine | 6 | 0 | 0 | 100% | 6.5 | LSE merge, ~1-thread-wide wg over Hq |
| flash_max | 9 | 0 | 0 | 100% | 5.8 | reduce |
| flash_den | 3 | 0 | 0 | 100% | 4.6 | reduce, 1-thread-wide |
| flash_gmax | 2 | 0 | 0 | 100% | 3.3 | reduce, 1-thread-wide |
| **llama flash_attn_tile** | 128 | **10752** | 0 | ~75% | **9.16** | one fused kernel; 1024 `v_dot2` + 268 `ds_read` |
| llama combine | 55 | 0 | 0 | — | 3.08 | |

**coop ~70µs (8 programs) vs llama 12.2µs (2).** The dominant tinygrad kernel (`flash_partial`, 35%) is the un-tiled
scalar PV; the LSE reduces run as near-empty 1-thread-wide workgroups.

## llama vs tinygrad ISA / dataflow

| | tinygrad `flash_partial` | llama tile |
|---|--:|--:|
| total insns | 294 | 2119–7383 |
| VALU | 50 | 1384–4976 |
| **`v_dot2_f32_f16`** | **0** | **256–1024** |
| `v_pk_*` (packed fp16) | 0 | 264–1056 |
| **`ds_read`/`ds_write` (LDS)** | **0 / 0** | **112–268 / 17–19** |
| global/buffer loads | 2 (scalar fp16) | 21–42 (LDS-staged) |
| `v_exp` | 0 (hoisted) | 14–53 (in-kernel softmax) |

tinygrad loads V **one fp16 at a time from HBM** and does scalar `v_fmac`; llama stages K/V via `ds_load_b128`
(8 fp16/load) and runs dense `v_dot2_f32_f16` chains for the **whole** QK+softmax+PV in ONE kernel. (Snippets:
`loop_snippet_tinygrad_partial.txt`, `loop_snippet_llama_tile.txt`.)

## Conclusion

- **NOT occupancy / registers / spills** (all tinygrad kernels at 100% occupancy, ≤13 VGPR, 0 spills).
- **NOT a fundamental limit** (llama hits 12.2µs on the same hardware).
- **NOT tooling-opaque** (ISA/resources/occupancy obtained; only live VALU/cache counters were unavailable).
- **IT IS fixable codegen quality + fragmentation**: tinygrad emits scalar, un-tiled, no-`v_dot2`/no-LDS code for the
  PV reduction (the `flash_partial`, 24.7µs), and splits attention into 6 launch+HBM-roundtrip kernels — vs llama's
  one LDS-tiled `v_dot2` fused kernel. The q·k matmul is fast **because** tinygrad's tiled-GEMM codegen applies to
  matmul-shaped ops; `flash_partial` is a hand-rolled reduction so it gets none of that.

**`FIXABLE_CODEGEN_LEVER_FOUND`.**

## Next scoped action (scope only — do NOT build in this task)

Scope a **bounded diagnostic**: a **matmul-PV / tiled-PV decode attention** vs `gqa_coop_vec` — express the PV
weighting (`prob @ V`) as a tinygrad **matmul** so the existing tiled-GEMM codegen (LDS-staged, packed) applies,
replacing the scalar `flash_partial`. First gate: local A/B @ctx512/1024/4096; W==D only if local passes.

**Honest EV / stop condition:** the bounded matmul-PV step recovers ~10µs (`flash_partial` 24.7 → ~14µs) ≈ **1.16×
standalone attention**; attention is ~23% of decode → **~3–4% whole-decode**, likely **below the 5% W==D bar**
(`LOCAL_PASS_WD_FAIL` class, like FLASH_L=64). The **full** llama-class win requires the **deep** LDS-tiled fused-flash
codegen capability (tinygrad's UOp codegen doesn't emit LDS-tiled flash reductions; fusion blocked by the Q8L-2
two-granularity store wall + Path A's per-lane exp redundancy — multi-week). **If** the matmul-PV step is
W==D-marginal **and** the deep codegen capability is not funded → **`REST_DECODE`** with this counter-level evidence.

This lever is **NOT** the closed "coop-qk-preserving redesign" (closed on timing — "combine ~1µs, no delta" — without
ISA attribution). The ISA now shows a **specific** fixable inefficiency in the 24.7µs PV partial — new evidence.

## Acceptance gates

| gate | result |
|---|---|
| G1 ≥1 profiler/trace source attempted+logged | PASS (rocprofv3 on llama + tinygrad ProfileGraphEvent) |
| G2 ISA/disassembly for tinygrad kernels | PASS (all 7 + matmul disassembled) |
| G3 llama tile source/profile comparator | PASS (extracted `.co` + disasm + trace) |
| G4 ctx1024 attribution | PASS |
| G5 ctx512/4096 attempted/justified | resources/ISA are ctx-invariant (per-kernel shape fixed); durations from the prior breakdown; justified |
| G6 counter reliability explicit | PASS (table above) |
| G7 no kernel/model/default change | PASS (`git diff tinygrad/` empty) |
| G8 policy guard passes | PASS |
| G9 tree clean after commit | PASS (commit below) |

## Boundary
Diagnostic only. No `tinygrad/`/model/default/kernel change, no new tile/fusion, no W==D, no closed lane reopened.
The one tinygrad run (per-kernel resource/ISA capture) was offline compile/disasm; perf-state unaffected.
