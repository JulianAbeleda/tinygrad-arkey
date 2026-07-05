# Packed Prefill Generated-Tile Scope

This backlog is derived from a BoltBeam practical roofline report. It ranks tinygrad work by reclaimable
pp512 time against llama's measured Q4 packed-matmul rate, not by broad code ownership.

## Conclusion

The first fix is schedule selection on the existing Q4_K direct-output route. The original trace made the path look
like a bandwidth problem, but the useful interpretation is dequant amortization: when token/row work is not register
tiled, Q4_K unpack/dequant is repeated too often. A correct `LOCAL:0:16,LOCAL:1:16,UPCAST:0:4,UPCAST:1:4` schedule
moves 14B pp512 from `135.7` to `173.6 tok/s`.

The remaining gap is still the packed-prefill matmul substrate, but it is narrower: tinygrad needs a correct grouped
or staged reduction that preserves the 4x4 register tile and amortizes dequant over a larger token tile. Naive
`GROUP` on the current custom UOp body is fast but wrong.

## Generated Schedule Requirements

- Route family: `generated_packed_prefill_tile`
- Default state: `off` via `PREFILL_QK_GENERATED_TILE=1`
- Strict gate: `PREFILL_ROUTE_STRICT=1` must fail on hidden fallback
- Current axes: row GLOBAL, token mostly UPCAST/serial, q4 lane4 REDUCE, kblock REDUCE
- Target axes: row tile GLOBAL/LOCAL, token tile GLOBAL/LOCAL, q4 lane4 LOCAL/cooperative, kblock REDUCE
- First shape: `[512,17408,5120]`

## Ranked Work

| priority | role | quant | shape | current us | current GB/s | target GB/s | reclaim us | launch resources | route id |
|---:|---|---|---|---:|---:|---:|---:|---|---|
| 1 | ffn_gate_up | Q4_K | `[512,17408,5120]` | 1901133.712 | 2.110 | 25.999 | 1746865.600 | global=(32, 272, 1), local=(64, 1, 1), threads=64, vgpr=185, sgpr=16, lds=0, scratch=0 | `prefill_q4_k_generated_tile_ffn_gate_up_512_17408_5120` |
| 2 | ffn_down | Q4_K | `[512,5120,17408]` | 556040.883 | 1.803 | 25.999 | 517473.855 | global=(32, 160, 1), local=(32, 1, 1), threads=32, vgpr=185, sgpr=16, lds=0, scratch=0 | `prefill_q4_k_generated_tile_ffn_down_512_5120_17408` |
| 3 | attn_qo | Q4_K | `[512,5120,5120]` | 464256.614 | 2.541 | 25.999 | 418883.640 | global=(32, 80, 1), local=(64, 1, 1), threads=64, vgpr=185, sgpr=16, lds=0, scratch=0 | `prefill_q4_k_generated_tile_attn_qo_512_5120_5120` |
| 4 | attn_kv | Q4_K | `[512,1024,5120]` | 82479.051 | 2.145 | 25.999 | 75673.105 | global=(32, 16, 1), local=(64, 1, 1), threads=64, vgpr=177, sgpr=16, lds=0, scratch=0 | `prefill_q4_k_generated_tile_attn_kv_512_1024_5120` |
| 5 | ffn_down | Q6_K | `[512,5120,17408]` | 648188.161 | 2.256 |  |  | global=(32, 80, 1), local=(64, 1, 1), threads=64, vgpr=225, sgpr=16, lds=0, scratch=0 | `prefill_q6_k_generated_tile_ffn_down_512_5120_17408` |
| 6 | attn_kv | Q6_K | `[512,1024,5120]` | 38153.247 | 2.254 |  |  | global=(32, 32, 1), local=(32, 1, 1), threads=32, vgpr=209, sgpr=16, lds=0, scratch=0 | `prefill_q6_k_generated_tile_attn_kv_512_1024_5120` |

## Implementation Path

1. Add a `PackedPrefillTileSpec` data object for Q4_K with row tile, token tile, lane tile, k-block policy,
   accumulator dtype, output layout, and strict role/shape guards.
2. Lower that spec through a generated UOp emitter. The first emitter should keep lossless fp32 accumulation and
   direct `[tokens, rows]` output; an external lane-partial probe is acceptable only as a short-lived microgate.
3. Wire `tinygrad/llm/prefill_routes.py` behind `PREFILL_QK_GENERATED_TILE=1`, with tensor-role filters so the
   first target can be only `ffn_gate_up`.
4. Add route-manifest metadata with provenance `machine_authored_generated` once the emitter is spec-driven.
5. Gate ffn_gate_up first, then attn_qo and ffn_down Q4_K. Add Q6_K only after the Q4 topology moves.

## Exhaustion Rule

Close a candidate quickly if the bound hot-row kernel stays in the ~2 GB/s class. Continue only when the generated
tile changes the substrate class, visible as wider workgroups/cooperative lanes and a multi-x per-kernel GB/s move.

## 2026-07-04 Candidate Results

### Promoted Schedule

Fable's audit corrected the framing: the current route was effectively too close to 512 independent GEMVs because
dequant work was not sufficiently amortized across tokens. The safe schedule change is:

```text
LOCAL:0:16, LOCAL:1:16, UPCAST:0:4, UPCAST:1:4
```

This is now the default Q4_K direct-packed prefill schedule. Rollback:

```text
PREFILL_Q4K_DIRECT_SCHEDULE=legacy
```

Clean 14B pp512 timing:

| route | pp512 tok/s | elapsed us | verdict |
|---|---:|---:|---|
| old Q4 direct-packed default | 135.7 | 3772608.7 | baseline |
| Q4 4x4 register-tiled schedule | 173.6 | 2950068.1 | promoted |

The tempting grouped schedule:

```text
LOCAL:0:64, GROUP:0:10, UPCAST:1:4
```

looked much faster (`~214 tok/s` when applied to `ffn_gate_up`, `~275 tok/s` when applied to K=5120 Q4 roles), but it
is numerically invalid on real 14B `blk.0.ffn_gate`: `rel_rmse ~= 1.26`. Do not use `GROUP` on this direct-output Q4
custom UOp until the grouped reduction semantics are fixed.

### Refuted Cooperative-Lane Probes

The first generated-UOp cooperative-lane probes are correct but refuted for speed on 14B `ffn_gate_up`.

| candidate | tile | output | whole pp512 tok/s | ffn_gate_up GB/s | verdict |
|---|---|---|---:|---:|---|
| current direct-packed floor | current | direct `[tokens, rows]` | 135.7 | 2.11 | baseline |
| generated tile | rows=4, tokens=8, lanes=8 | external 8-lane partial reduce | 79.2 | 0.99 | refuted |
| generated direct-warp | rows=1, tokens=4, lanes=8 | in-kernel warp reduce | 98.4 | 1.29 | refuted |
| generated direct-warp | rows=2, tokens=2, lanes=8 | in-kernel warp reduce | 86.7 | 1.05 | refuted |
| generated direct-warp | rows=4, tokens=1, lanes=8 | in-kernel warp reduce | 83.5 | 1.00 | refuted |

Correctness for the best direct-warp mode passed against the existing lossless direct-packed route on real 14B
`blk.0.ffn_gate`: `rel_rmse=1.64e-6`, `max_abs=3.81e-5`.

This exhausts the "simple generated UOp cooperative lane" family for the 14B hot row. The failure mode is clear:
external lane partials add too much lifecycle, while a one-wave in-kernel combine removes that lifecycle but loses too
much row/token tile throughput.

## Next Work

1. Fix or replace grouped reduction for the direct-output Q4 custom UOp so that a grouped K-superblock schedule is
   numerically correct.
2. Re-test the fast-but-wrong `GROUP:0:10` family after the reduction fix; the measured speed suggests the schedule
   shape is valuable if semantics can be made correct.
3. Only after correct grouping plateaus, add the dequant-to-fp16-LDS prologue feeding WMMA or an int8/dot path.

## 2026-07-05 Authority-measured verdict (steps 1-2 CLOSED, step 3 is the real solve)

Re-measured everything on the SANCTIONED authority `extra/qk/prefill_whole_synced.py` (synced TinyJit min-of-K,
chunk@start_pos=0), 14B Qwen3-Q4_K_M, one chunk = pp512. NOTE the earlier `135.7/173.6` figures were an
understated methodology; the authority reads ~2x higher, so compare only within this table.

| config | pp512 tok/s | path | note |
|---|---:|---|---|
| **tile4x4 direct-out (default)** | **365** | packed VALU | the VALU ceiling |
| reduce-out `GROUP:0:2` (all roles) | 244 | packed VALU | correct, regressed |
| reduce-out `GROUP:0:4` (all roles) | 185 | packed VALU | correct, regressed harder |
| reduce-out `GROUP:0:10` (ffn_gate_up only) | 201 | packed VALU | correct, regressed |
| reduce-out `GROUP:0:5` (ffn_gate_up only) | 246 | packed VALU | correct, regressed |
| `PREFILL_CHUNKED=1 GRAPH_GEMM=1` | 361 | full-fp16 dequant + WMMA | break-even, NOT a win |

**Step 1/2 verdict — CLOSED as plateau.** Every numerically-correct grouped schedule (`reduce-out` + `GROUP`)
is *slower* than the tile4x4 direct-out default (365), monotonically worse with more grouping. The generic
`GROUP_REDUCE`/LDS combine cost dominates any occupancy gain. The fast-but-wrong `GROUP:0:10` speed does NOT
survive being made correct. **The VALU path is capped at ~365 tok/s pp512** and no schedule/codegen move on it
(config, group, reduce-out) beats that. This confirms + supersedes the earlier "169.7 vs 173.6" hint with
authority numbers.

**Why 14B is stuck on VALU (the root cause, clarified).** `route_prefill_linear` sends 14B's hot Q4_K linears to
`direct_packed` (VALU) because `_pf16_w` (resident fp16 weight) is None — resident fp16 for 14B is ~31GB > the
~20.5GB budget, so it never materializes and the WMMA graph-GEMM path is unreachable. 8B fits → WMMA → ~4408
tok/s. **The 8B-vs-14B prefill gap is fundamentally the WMMA-vs-VALU path split, gated by the fp16 memory budget**,
not a schedule inefficiency. The VALU matmul tops out at ~1% of fp16 peak regardless of occupancy.

**Step 3 is the ONLY structural lever left, and it must be BUILT.** The naive existing WMMA path
(`PREFILL_CHUNKED=1`) materializes the FULL weight to fp16 per call (`prefill_fp16_weight`,
qk_primitives.py:71) — bandwidth-bound, measured break-even (361). The real solve is a **fused per-tile
dequant→fp16-LDS→WMMA** kernel: weights stay Q4_K in HBM; each K-block tile is dequantized to fp16 in registers
and `ds_store`d to LDS inside the WMMA kernel's coop-load stage, then consumed by the existing `build_gemm_lds2`
WMMA compute. No full fp16 materialization, matrix core engaged. This does not exist yet (grep: no dequant→WMMA
fusion in the tree). Concrete first increment: add a Q4_K dequant prologue to the B-operand coop-load in
`extra/qk/prefill/wmma.py::build_gemm_lds2` (`coop_load`/`coop_store`), correctness-gate vs the dequant reference
on the [512,17408,5120] shape, then measure on the authority. This is hand-asm extending the sanctioned
asm_scheduler/BubbleBeam WMMA stream (NOT a new hand kernel on the generated path) — a multi-session build.

Reproduce: `DEVICE_IN_FUNCTION_BUG=1 ALLOW_DEVICE_USAGE=1 DEV=AMD PREFILL_V2=1 PREFILL_GRAPH_GEMM=0 PYTHONPATH=. \
.venv/bin/python extra/qk/prefill_whole_synced.py --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf -K 5`
(inject schedule via `PREFILL_Q4K_REDUCE_OUT=1` + `PREFILL_Q4K_DIRECT_OPTS` / per-role `PREFILL_DIRECT_<ROLE>_OPTS`).

## 2026-07-05 WMMA-via-codegen: mechanism PROVEN, but blocked by a 14B memory wall

Investigated whether tinygrad's OWN codegen can emit RDNA3 WMMA for the Q4_K GEMM (no hand kernel), to
replace the VALU path. Findings (all measured, DEV=AMD, TC=1 TC_OPT=1):

1. **Stock tinygrad codegen emits RDNA3 WMMA.** A plain `half@half` matmul realizes with
   `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` (`TC(0)` fires). The HIPRenderer exposes 3 WMMA
   tensor_cores. So WMMA is NOT missing — it works.
2. **Why the Q4_K GEMM never tensorizes (pinned to IR).** `postrange.py::_apply_tc_opt` requires the
   reduce body to be a single `MUL(in0,in1)` with `in0.dtype==in1.dtype==tc.dtype_in (half)`. The
   direct-out kernel has **0 `Ops.REDUCE`** (manual accumulator) → bails immediately. The reduce-out
   kernel's body is an **`Ops.ADD` tree of 93 fp32 MULs** (the grp×nib dequant unroll) with **fp32**
   operands → bails at the `mul.op is not Ops.MUL` and dtype gates. So it's a **vocab** problem (how the
   dot is spelled) feeding a **codegen** gate.
3. **A dequant expression DOES tensorize + fuse.** `x_half @ (wq.cast(half)*scale).T` under `TC_OPT=1`:
   correct (rel_rmse 3.4e-4) AND emits WMMA AND tinygrad **auto-fuses the dequant into the WMMA kernel**
   (the kernel reads `signed char*` weights + `half*` scales directly — no materialized fp16 buffer).
   So the codegen/vocab fix needs NO hand kernel.
4. **BUT the 14B memory wall blocks both naive forms (hot shape [512,17408,5120], per-GEMM):**
   - **fused-lazy** dequant→WMMA (fits memory): **3.0 TFLOPS** — the heavy Q4_K dequant runs redundantly
     per weight-fragment load (~M/16×), dominating. Fusion is the WRONG move.
   - **materialized** fp16 then WMMA: **20.9 TFLOPS** (~15× the VALU 1.35) — but forcing this whole-model
     (`PREFILL_ROUTE=chunked GRAPH_GEMM=0`) **OOMs at 23.84 GB**: TinyJit holds the per-layer fp16 weights
     live. (This is exactly why the router gates 14B to VALU — `_pf16_w` is None because fp16 doesn't fit.)
     NOTE: the earlier "chunked=361" numbers never reached WMMA — `route=="auto"` returns the direct_packed
     VALU branch before the chunked branch; must set `PREFILL_ROUTE=chunked` to force it.

**Where that leaves it.** WMMA-via-codegen is real and hand-kernel-free, but the 14B fp16-doesn't-fit
constraint makes materialize→OOM and fuse→dequant-redundant-slow. The remaining constraint-respecting
lever is a **schedule/codegen** one: get tinygrad to stage a per-TILE Q4_K dequant into LDS **once**
(dequant-once-per-tile, not per-M-fragment and not whole-weight) — e.g. drive the fused dequant→WMMA with
LOCAL/UPCAST opts that hoist the dequanted B-tile to LDS, or K/N-chunk the dequant so each fp16 chunk fits
and is reused across all M. This is the open next step; it is codegen+vocab work, not a hand kernel.
Repro scripts: scratchpad `tc_probe.py` (IR gate), `tc_dequant_test.py` (fuse+correct proof),
`q4k_tc_realshape.py` (fused vs materialized TFLOPS).

## 2026-07-05 How llama does it (traced) — the answer is INT8 MMQ, not fp16

Traced llama.cpp (`/home/ubuntu/env/llama.cpp`, HIP build, `llama-bench`) on the SAME 14B on this gfx1100:
**llama pp512 = 1849 t/s** (vs our VALU 365 → llama is ~5×). Read the kernel source (`ggml-cuda/mmq.cuh`):

- **MMQ = integer matmul, no fp16 weight ever exists.** Weights stay **Q4_K**, unpacked into int8 tiles in
  **shared memory**; the activations are **quantized to Q8_1 (int8+scale)** and copied to shared memory
  (`block_q8_1_mmq`, "converted to a data layout that can simply be copied to shared memory"). The matmul
  runs in **int8** — on gfx1100 with `AMD_WMMA_AVAILABLE` it uses the **int8 WMMA** MMA path (`mmq.cuh`
  lines 110/120/283/322), else **dp4a** (`v_dot4_i32_i8`). Q4_K/Q8_1 scales+mins are applied AFTER the
  int32 accumulation. This is exactly why llama sidesteps our fp16 memory wall: everything stays
  quantized; only small int8 tiles hit LDS.

- **tinygrad codegen CANNOT reproduce this today (vocab gap).** `tinygrad/codegen/opt/tc.py::amd_rdna3`
  defines tensor cores for `dtype_in ∈ {half, bfloat16}` ONLY — **no int8 (iu8) tensor core**. So an int8
  matmul will NOT tensorize to `v_wmma_i32_16x16x16_iu8` — the instruction isn't in the vocabulary. RDNA3
  HAS the intrinsic (`__builtin_amdgcn_wmma_i32_16x16x16_iu8`); tinygrad just doesn't expose it.

**Revised plan (matches llama, still codegen+vocab, no hand kernel):**
1. VOCAB: add an int8 RDNA3 tensor core `WMMA_16_16_16_iu8_i32` to `tc.py::amd_rdna3` + the HIP intrinsic
   in the renderer. This is a TensorCore descriptor + swizzle + intrinsic define — not a hand kernel.
2. VOCAB/primitive: express Q4_K prefill as an **int8** matmul — unpack Q4_K nibbles → int8 weights,
   quantize activations → int8 (Q8_1), `reduce_k MUL(int8,int8)` → int32, then apply per-32-group
   scale/min corrections elementwise (llama's math). tinygrad tensorizes the int8 core → int8 WMMA.
3. Fits memory (int8 not fp16, weights stay ~4-bit in HBM), matches llama's approach. Target → ~1849.
   Interim: the fork's existing dp4a route `PREFILL_Q4K_Q8=sdot4|mmq` is the hand-rolled v_dot4 version —
   measure it to see how far dp4a alone gets before investing in the int8-WMMA vocab addition.

**Interim measured (2026-07-05, authority pp512, 14B):** `PREFILL_Q4K_Q8=sdot4` = **17 tok/s** (30.9s! —
21× SLOWER than VALU 365); `PREFILL_Q4K_Q8=mmq` = **OOM** at 23.6 GB. So the fork's existing int8/dp4a
routes are NOT usable — it has the v_dot4 primitive but no tiled-LDS-staged int8 GEMM. Confirms: matching
llama needs a REAL tiled int8 matmul (int8 WMMA through LDS), which is exactly the vocab (int8 TC) +
codegen (tensorize + stage) build above, not the existing scalar-ish dp4a path.

### Scoreboard (14B pp512, authority)
| approach | tok/s | note |
|---|---:|---|
| **llama.cpp (int8 MMQ, tiled int8-WMMA/dp4a, LDS)** | **1849** | the target |
| tinygrad VALU direct-out (current default) | 365 | our ceiling |
| tinygrad fp16-WMMA materialized | (21 TFLOPS/GEMM) | OOM whole-model |
| tinygrad fp16-WMMA fused-lazy | (3 TFLOPS/GEMM) | fits, dequant-redundant |
| fork dp4a `PREFILL_Q4K_Q8=sdot4` | 17 | unoptimized, no tiling |
| fork `PREFILL_Q4K_Q8=mmq` | OOM | — |

The only approach that structurally matches llama = **tiled int8 matmul on int8 WMMA through LDS**. In
tinygrad terms: add iu8 tensor core to the vocab + express the int8 tiled Q4_K matmul so codegen
tensorizes and LDS-stages it. That is the fix; everything else measured is a dead end or memory-walled.

## Primitive Root Cause

The direct-output Q4_K primitive used a manual accumulator/store recurrence:

```python
acc = out[bb, row].set(0.0)
acc = out[bb, row].set(acc.after(blk, lane4)[bb, row] + contrib, end=lane4)
return acc.end(row, bb, blk)
```

That form is fine for ordinary `REDUCE` loops, but it is not semantically compatible with tinygrad's `GROUP` lowering.
`GROUP` turns part of the reduce into a `GROUP_REDUCE` local axis. Late GPU-dim lowering sees that local axis is missing
from the global output index, so it masks the store to one local lane. There is no sum over the grouped lanes. That is
why `GROUP:0:10` looked fast and was badly wrong.

The fixed experimental primitive uses a real `Ops.REDUCE`:

```python
total = contrib.reduce(blk, lane4, arg=Ops.ADD)
out[bb, row].store(total)
```

With `PREFILL_Q4K_REDUCE_OUT=1`, the formerly wrong grouped schedule is numerically correct on real 14B `ffn_gate_up`:
`rel_rmse ~= 1.6e-6`, `max_abs ~= 3.4e-5`.

However, correctness costs enough that it is not the fastest route yet:

| route | clean pp512 tok/s | note |
|---|---:|---|
| Q4 4x4 manual direct-output default | 173.6 | current default |
| Q4 reduce-out + `GROUP:0:10` on K=5120 roles | 169.7 | correct, default-off |

So the primitive correctness bug is fixed behind a flag, but the big remaining issue is now clearer: the correct
generic `GROUP_REDUCE`/LDS combine path is too expensive for this packed-prefill matmul. The next primitive must keep
correct grouped reduction semantics while avoiding the current LDS/group overhead, likely via a Q4-specific staged
combine or dequant-to-fp16 tile feeding the existing matmul path.
