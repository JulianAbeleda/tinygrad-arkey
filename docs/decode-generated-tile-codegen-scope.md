# Make the generated decode tile lower to competitive ISA — codegen scope (2026-06-26)

Goal of this scope: take the GENERATED decode-attention tile (which is correct, occupancy-aware, and
route-clean, but ~99× slower than the owned hand-written kernel) and make tinygrad's AMD codegen lower it
to competitive ISA — so the *machine-search* path produces owned-level decode attention with no
hand-written assembly. This is the north-star "renderer lowering" blocker, now scoped with file:line.

This is NOT "rewrite the renderer." The four-area code map below shows the levers are mostly
kernel-authoring that triggers EXISTING codegen machinery, plus targeted codegen fixes only where that
machinery doesn't fire on the custom-kernel shape.

## Where we are (proven this session)

- The generated tile `flash_fused_xlane_score_pv_tile_whole_cache_kernel` is numerically correct
  (microgate PASS), in-model route-clean + token-matched (route gate PASS), occupancy-matched at S=48
  (4 wg/CU), and W==D-refuted: 6.5 / 3.6 / 0.9 tok/s @ ctx 512/1024/4096 vs baseline 103.5/101.8/94.6.
- Per-kernel isolation timing: the **tile** is 17.2 ms/launch = 100% of attention (676× the gmax+combine).
- ISA diff (`ISA_DIFF_PINNED`): owned tile LDS **8192 B**, `global_load_d16` **22**, `v_dot2` 2,
  `cross_lane` 5; generated tile LDS **256 B**, `global_load_d16` **0** (V cast to fp32 scalar), `cross_lane`
  **20**. Same instruction *count* (557 vs 555). Removing the per-token barrier recovered only ~15%.
- Conclusion: the gap is **codegen strategy** — owned block-tiles tokens (TK=16, 8 KB LDS, 128 threads/
  4 warps, vectorized fp16 loads, fdot2, `__shfl_xor`) and is compiled by hipcc -O3; the generated tile is
  per-token, single-warp, scalar-load, fed through tinygrad codegen which does not vectorize or block it.

## Code map (the four mechanisms; absolute repo /home/ubuntu/tinygrad-arkey)

### A. Custom kernels skip auto-UPCAST, but the load coalescer still runs
- `extra/qk_flash_decode.py:116` `_fki(name)=KernelInfo(name, opts_to_apply=())`. The empty tuple (not
  `None`) makes `apply_opts` (`tinygrad/codegen/opt/postrange.py:353-390`, branch `:357-358`) apply ZERO
  opts and skip the heuristic/BEAM auto-optimizer (`:359-364,385-389`). So no range becomes
  `AxisType.UPCAST` and the expander (`tinygrad/codegen/late/expander.py:147-151`, fires only on
  UPCAST/UNROLL ranges) leaves the kernel's `GLOBAL`/`REDUCE` loops scalar.
- BUT the coalescer runs on every kernel: `tinygrad/codegen/late/devectorizer.py:136-149`
  (`load_store_folding`), `:81-117` (`fold_expanded_index` — groups contiguous pointer offsets into a
  `PTRCAT`), `:153-200` (`split_load_store` — width decision; `lengths=[4,2]` float4, `[8,4,2]` if
  `ALLOW_HALF8`, `:167,175-177`; emits `VCAT` of vec loads, `:200`). It fires only when the access is
  presented as a single vectorized `INDEX` (vec offset) or a `STACK` of contiguous `ptr=True` INDEXes.
- The current tile presents neither: `cache[1,0,kvh,t_safe,d]` (V, `qk_flash_decode.py:899`) and the
  `klds`/`q` loads (`:877,886-887`) are scalar `INDEX→LOAD`, and the `STACK` at `:886-887` is over
  `CAST(LOAD(INDEX))` (register packing for fdot2), not over pointer-INDEXes, so it does not coalesce.

### B. Renderer needs no change for vectorization; comgr/LLVM picks the mnemonic
- AMD default renderers (`tinygrad/runtime/ops_amd.py:1026`): `[HIPRenderer, AMDLLVMRenderer,
  HIPCCRenderer]` — all **string** renderers (Track A: emit C++/LLVM, compiled by comgr/LLVM). There is no
  in-tree AMD `ISARenderer`.
- A vec-dtype LOAD renders as a C vector deref (`tinygrad/renderer/cstyle.py:206-210` `render_access`:
  `*((vectype*)(ptr))` when `max_numel()>1`, else scalar) or `load <N x T>` (`llvmir.py:73`). tinygrad does
  NOT choose `global_load_b32` vs `b128`/`d16` — **comgr/LLVM does**, from the vector cast.
  `supports_float4=True` for AMD (`renderer/__init__.py:57`, `llvmir.py:119`).
- So: if the kernel produces a vec-dtype load, the existing pipeline + compiler emit a wide load. No
  renderer edit required.

### C. No software pipeliner in core; the owned kernel relies on the compiler
- `tinygrad/codegen/late/linearizer.py:7-52` is the only ordering pass — a priority topo-sort (`LOAD=-1`
  early, `STORE=1` late, `:23-34`). Coarse load-hoisting, NO within-class reordering, NO pipelining.
- A real scheduler exists but is **not wired in**: `extra/qk_asm_scheduler.py` (operates on `list[Inst]`
  before `Ops.INS` wrapping). `tinygrad/renderer/amd/schedule.py` is dormant metadata (imported nowhere).
- The owned tile (`extra/qk_owned_flash_decode.hip:225` `owned_flash_tile_gqa_whole`) is **hipcc-compiled
  .co injected as an inert `Ops.PROGRAM`** (`extra/qk_owned_flash_decode_graph_node.py:78-88,161-172`) — it
  bypasses tinygrad codegen; hipcc -O3 does the vectorization + scheduling. **Key implication: a generated
  tile fed through Track A (HIPRenderer → comgr) gets comgr's vectorizer + scheduler too** — so a
  well-structured generated kernel may compile to competitive ISA without a tinygrad pipeliner.

### D. Existing env-gated lowering recipe (clone this if a codegen pass is needed)
- `extra/qk_fdot2_lowering.py`: a `PatternMatcher` (`pm_fdot2`, `:81-83`) that rewrites a UOp idiom to
  `UOp(Ops.CUSTOMI, dtype, (shaped_src_first, ...), arg="__builtin_amdgcn_fdot2({1},{2},{0},false)")`
  (`:49-52`). Hooked in `tinygrad/codegen/__init__.py:112-114` (env-gated `graph_rewrite` after
  `devectorize`) + cache-key at `:255`. Renderer emits any `CUSTOMI` via `str.format` at
  `tinygrad/renderer/cstyle.py:79` — **no renderer change**.
- `extra/qk_warp_reduce_lowering.py` (`pm_warp_reduce`, hooked at `__init__.py:84-88` into the expander
  chain) is the other template (a structural REDUCE→cross-lane rewrite).

## The owned tile = the structural target (what to mirror)

`extra/qk_owned_flash_decode.hip:225` `owned_flash_tile_gqa_whole`:
- One **workgroup = 128 threads / 4 warps** per `(kvh, split)` (vs generated: 32 threads / 1 warp).
- **TK=16** token block cooperatively staged into LDS: `__shared__ __half ksh[TK*Hd], vsh[TK*Hd]` →
  **2·16·128·2 = 8192 B** (the 8 KB in the ISA diff).
- Per token: `__builtin_amdgcn_fdot2` q·k (→ `v_dot2_f32_f16`), `__shfl_xor` warp-reduce, online softmax,
  PV from `vsh`. Vectorized fp16 loads (`global_load_d16` ×22). S=48 splits (`DECODE_ATTN_AMDGCN_S`).

## Plan (phased — cheapest, highest-confidence first)

### Phase 0 — baseline the toolchain (no code change)
Run the four gates (commands in Acceptance) to confirm the current PASS/PINNED/refuted state, and capture
the current `disasm_flash_fused_xlane_score_pv_tile_whole_cache_32_128.txt` as the "before".

### Phase 1 — vectorize the loads via existing machinery (kernel-authoring)
In `flash_fused_xlane_score_pv_tile_whole_cache_kernel` (`extra/qk_flash_decode.py:841-911`), make the
contiguous K/V loads coalesce, by ONE of (try in order; the first that the existing coalescer honors wins):
1. Declare the contiguous element axis as `UOp.range(R, …, AxisType.UPCAST)` instead of `AxisType.REDUCE`
   for the K-stage loop (`:875` `rk`) and an explicit V-load loop, so the expander
   (`expander.py:147-151`) widens the `INDEX` and `split_load_store` coalesces it.
2. If (1) doesn't fire on the custom path, build a `STACK` of `ptr=True` INDEXes at contiguous offsets for
   the V tile load so `fold_expanded_index` (`devectorizer.py:81-117`) coalesces → `PTRCAT` → `VCAT`.
3. Load V as fp16 (`.cast(dtypes.half)` paths / keep half) and set `ALLOW_HALF8=1` to allow b128-of-half
   folds (`devectorizer.py:177`).
Acceptance for Phase 1: ISA-diff gate shows `xlane global_load_d16 > 0` (or `global_load_dwordx4 > 0`) and
the microgate + route gate still PASS. Re-time the isolated tile (the DEBUG=2 isolation method in
`docs/decode-block-tile-codegen-result.md`) to quantify.

### Phase 2 — block-tile + multi-warp (kernel-authoring, mirrors owned)
Restructure the tile to process a **TK-token block** with **4 warps / 128 threads**:
- Add a LOCAL axis for the warp-within-workgroup (so the workgroup is 128 threads, 4 warps × 32 lanes),
  and a block loop staging `TK=16` tokens of K (and V) into an 8 KB LDS tile **once per block** (one
  barrier per block, not per token). Inner loop reads K/V from LDS.
- Keep the d-sharded PV, the fdot2 dot (V_DOT2_LOWERING), the cross-lane reduce (WARP_REDUCE_LOWERING or
  the staged helper), the online-softmax recurrence, S=48, and the raw-`cache_kv` 5D buffer identity.
- This is the largest change; prototype in the microgate
  (`extra/qk_decode_attention_fused_xlane_score_pv_microgate.py`) FIRST and validate numerically (scalar
  fp32 ≤1e-7, fp16 ≤2e-5) across its shapes before porting in-model.
Acceptance for Phase 2: ISA-diff `xlane LDS ≥ ~8192`, `cross_lane/token` ↓ toward owned's 5, loads
vectorized; microgate + route gate PASS.

### Phase 3 — only if Phase 1+2 don't close W==D, attribute the residual
Re-run W==D. If still far from baseline and GPU-bound, the residual is instruction scheduling /
pipelining. Options, in order: (a) confirm comgr is/ isn't pipelining (inspect the disasm for interleaved
loads/compute); (b) if Track A comgr already pipelines a well-structured kernel, done; (c) else evaluate
wiring `extra/qk_asm_scheduler.py` or a Track-B (`Ops.INS`) path — a much larger effort, separate scope.

If at any phase the existing codegen machinery cannot express/lower the structured kernel (e.g. the
expander won't honor a manually-set UPCAST axis on the custom path, or the coalescer won't fold the
custom STACK), THAT is the genuine codegen gap: fix it with an env-gated pass following the v_dot2 recipe
(map §D), and label the finding precisely (it is a real renderer/codegen milestone, not a kernel bug).

## Acceptance harness (run after every change)

```
# numeric correctness of the layout (pass = FUSED_XLANE_SCORE_PV_MICROGATE_PASS)
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_attention_fused_xlane_score_pv_microgate.py
# in-model route clean + token-match (pass = FUSED_XLANE_SCORE_PV_ROUTE_CLEAN__ECONOMICS_NEXT)
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_attention_fused_xlane_score_pv_route_gate.py
# owned-vs-generated ISA diff (pass = ISA_DIFF_PINNED; READ the markers: LDS, global_load_d16, cross_lane)
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_attention_isa_diff_gate.py
# W==D tok/s authority vs baseline 82.4/103.5/101.8/94.6 @ ctx 128/512/1024/4096
DEV=AMD JIT=1 DECODE_ATTN_GENERATED_WHOLECACHE=1 DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE=1 V_DOT2_LOWERING=1 \
  PYTHONPATH=. python3 extra/qk_decode_runtime_overhead.py
# isolated per-kernel us (the working timing method; tile prints in ms when slow)
# see docs/decode-block-tile-codegen-result.md Part A for the snippet
```
Success metric: the generated route's W==D tok/s approaches baseline (within the corpus's opt-in margins),
with microgate + route gate still PASS, and ISA-diff showing vectorized loads + multi-KB LDS.

## Constraints (hard)

- **Default-off.** All changes behind `DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE` (+ any new env flag); the
  shipped default route (owned AMDGCN) and the generated q4k GEMVs must be byte-for-byte unchanged. Any
  new codegen pass must be env-gated AND added to the `to_program` cache key (`__init__.py:255`).
- **Correctness first.** Never regress microgate/route-gate token-match. Cross-lane / LDS / fdot2 ops are
  divergence- and shape-sensitive — follow the staging rules in `extra/amd_warp_reduce.py:1-13` and the
  `CUSTOMI`-carries-`src[0]`-shape rule (`tinygrad/uop/ops.py:306`).
- **Do not** add another attention *layout* (the layout is proven). This is strictly codegen strategy:
  vectorization, block-tiling, scheduling.
- **Do not** hand-edit `tinygrad/runtime/autogen/**` (generated) or commit non-deterministic bench timing.

## Forks / stop conditions

- If Phase 1 alone (vectorized loads) gets W==D to baseline → done; block-tiling may be unnecessary.
- If the existing coalescer/expander cannot be triggered from a custom kernel → genuine codegen gap; build
  the env-gated pass (v_dot2 recipe) — this is the milestone.
- If, after vectorization + block-tiling, the kernel compiles but W==D is still gated by scheduling and
  comgr won't pipeline → record `SEARCH_BLOCKED_BY_CODEGEN__SCHEDULING`; wiring a scheduler is a separate,
  larger scope (Track B / `qk_asm_scheduler.py`).

Codex prompt: `docs/decode-generated-tile-codex-prompt.md`.
