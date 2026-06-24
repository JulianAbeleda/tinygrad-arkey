# Route B B4 — External precompiled AMDGCN kernel as a tinygrad JIT graph node (Scope)

Date: 2026-06-21

Executes **Route B B4 only** (`docs/decode-attention-route-a-route-b-full-execution-scope-20260621.md`). B3 produced the
first OWNED, promotable hand-AMDGCN decode-attention tile (`extra/qk_owned_flash_decode.hip`) that beats `gqa_coop_vec`
**2.35× GPU-busy / 1.70× wall** @ctx1024 (correct rel 2e-7, v_dot2=2, 56 VGPR, 8 KB LDS, 0 spill), launched via the
B1 `NamedAMDProgram` + B2 one-bound-HCQ queue. B3 is `default_eligible=false` because **W==D is blocked**: a raw-HCQ
`.co` launched outside the JIT cannot enter the JIT-traced decode graph that `model.generate` replays.

**B4's single, bounded job:** build the *external-precompiled-kernel-as-JIT-graph-node* capability so the B3 tile+combine
replay **inside** tinygrad's decode JIT graph under a default-off flag, then run W==D. **This is NOT Route-A codegen**
(we do not make tinygrad's renderer emit the tile); it is a runtime/graph-scheduling capability.

## Mechanism (audited — the capability already exists in tinygrad)

The "only sanctioned way a custom kernel can join graph/JIT execution is as a compiler-visible program
(`Tensor.custom_kernel` → `Ops.PROGRAM` → TinyJit capture/replay), NOT a raw HIP launch hidden from the graph"
(`test/external/test_custom_kernel_jit_bridge.py`). The bridge is proven for UOp-bodied custom kernels (capture as
`Ops.PROGRAM`, symbolic `start_pos`, sliced KV read, multi-output — `test/external/test_flash_prefill_custom_kernel_bridge.py`).

B4 uses the **same** bridge but injects a **precompiled BINARY** instead of a UOp/INS body:

- `Tensor.custom_kernel(*bufs, fxn=...)`: `fxn(*placeholders)` returns an `Ops.PROGRAM` UOp; `.call(*bufs)` wraps it as
  an opaque `Ops.CALL(PROGRAM, *bufs)` graph node (`uop/ops.py:1083` `_OPAQUE_CALL_BODIES` includes `PROGRAM`).
- A **fully-formed** `Ops.PROGRAM` with `src=(SINK, DEVICE, LINEAR, SOURCE, BINARY)` + `arg=ProgramInfo(...)` **skips
  codegen**: `do_to_program` keeps an explicit `ProgramInfo` arg (`codegen/__init__.py:226`) and `pm_to_program` matches
  only *incomplete* PROGRAMs, so a 5-src PROGRAM passes through untouched. `get_runtime` then builds the `AMDProgram`
  straight from `ast.src[4].arg` (the BINARY = our `.co` ELF) — `engine/realize.py:114`.
- `HCQGraph` schedules every `Ops.PROGRAM` call as one `enqueue_queue.exec(runtime, ji_args, global_size, local_size)`
  node (`runtime/graph/hcq.py:175`), folded into the one bound decode graph queue — exactly the B2 one-doorbell ideal,
  but now driven by TinyJit, so it survives to **W==D**.

### Kernarg ABI (the make-or-break detail — matches with zero glue)
`CLikeArgsState` lays out **[buffer pointers, 8 B each, in `bufs` order] then [scalar vars, 4 B each, as `ProgramInfo.vars`]**
(`runtime/support/hcq.py:324`). The B3 kernel's hipcc ABI is *pointers-first then scalars* in declaration order, so:
- pass `bufs = [Q, K, V, part, meta]` → pointers land at 0/8/16/24/32 (kernel's exact order);
- the only **per-step** scalar is `start_pos` (KV length grows each token) → it is the single `ProgramInfo.vars` entry,
  written at offset 40 and patched per replay from `var_vals["start_pos"]` (the same symbolic the whole decode JIT binds).
- `CLikeArgsState` writes **only `ProgramInfo.vars`** as scalar args (not arbitrary constants), so **`S` (split count)
  and `scale` (1/√Hd) are baked into the kernel** (compile-time constants); the kernel computes `n_valid = start_pos + 1`
  (T=1 decode). This specializes a single-kernel ELF whose kernarg is exactly `[Q,K,V,part,meta][start_pos]`.

### Multi-kernel ELF → single-kernel ELFs
Vanilla `AMDProgram` selects the **first** `.rodata` kernel descriptor (no name selection — `ops_amd.py:584`). The B3
`.co` carries 3 kernels, so B4 compiles **one single-kernel ELF per kernel** (tile, combine) from the B3 source
(specialized: `S`/`scale` baked, `start_pos` arg). Each ELF then loads unambiguously through the idiomatic
`AMDProgram` path. No vendored code, no repack — the kernel logic is byte-identical B3 dataflow on tinygrad's native
`[Hkv, MAXC, Hd]` layout.

## Deliverables
1. This scope doc (before edits).
2. `extra/qk_owned_flash_decode_graph_node.py`: the graph-node injector — specialized single-kernel ELF compile +
   precompiled-`Ops.PROGRAM` builder + `amdgcn_flash_decode(Q,K,V,start_pos_var,S)` returning the output Tensor via two
   chained `custom_kernel` calls (tile → combine), fixed Qwen3-8B decode shape, native KV layout, **no repack**.
3. A fixed-shape correctness + TinyJit capture/replay proof (output matches numpy GQA softmax; the tile+combine are
   captured as `Ops.PROGRAM` nodes and replay with a *different* `start_pos` than capture).
4. Only after (3) passes: an **env-gated** model route `DECODE_ATTN_AMDGCN_TILE=1` in `flash_decode_attention`/
   `model.py`, shape/device/dtype-guarded, **falling back to `gqa_coop_vec`** on any unsupported shape/device/layout.
5. decode_eval candidate registration (`bench/qk-decode-eval/candidates.json` + `binding_templates.json`),
   `default_eligible` gated on a real W==D pass + owner approval.
6. Local correctness + **W==D** vs the canonical baseline; stamped artifact under `bench/qk-decode-attention-route-b-b4/`;
   result doc + a single verdict.

## Gates
- **correctness:** output `rel_rmse ≤ 1e-3` vs numpy (standalone) AND greedy byte-identical or dNLL ≤ 0.01 in-model.
- **graph integration (the B4 capability gate):** the tile+combine are captured by TinyJit as `Ops.PROGRAM` nodes and
  replay correctly with a different `start_pos` (proves real graph-node injection, not a frozen/eager result).
- **W==D promotion:** ≥ **+5% @ctx1024 OR ≥ +7% @ctx4096** whole-decode vs baseline (68.1/66.4/60.7 tok/s), **no ctx512
  regression**, greedy byte-identical or dNLL ≤ 0.01.
- **policy:** default-off unless owner explicitly promotes; unsupported shape/device/layout falls back to `gqa_coop_vec`.

## Stop conditions / classification
- Graph-node injection requires broad **Route-A codegen** work → stop, `B4_BLOCKED_ROUTE_A_REQUIRED`.
- W==D can only be claimed via **eager (un-jitted) decode** → stop (eager is non-production), `B4_WD_FAIL_INTEGRATION`.
- Needs a **KV repack/transpose** bridge → stop, `B4_BLOCKED_LAYOUT_BRIDGE`.
- `custom_kernel` cannot carry a precompiled BINARY into the graph at all → `B4_BLOCKED_GRAPH_NODE`.
- Local graph-integrated route works but **W==D misses** → classify the miss (graph overhead / sync leakage / Amdahl
  limit / kernel mismatch); `B4_WD_FAIL_INTEGRATION`.
- **No blind split/flag (`S`) tuning before a clean W==D measurement.**

## Boundaries
Default stays OFF. `gqa_coop_vec` is the comparator SSOT. No closed-lane reopen (WMMA decode / MMVQ / FLASH_L promotion
/ fused tail / matmul-PV / warp tile). No vendored llama code. No KV repack/transpose. Minimal, AMD-guarded
`tinygrad/` change only if strictly required (prefer zero); model route is additive and default-off.

## Verdict (one of)
`B4_WD_PASS` · `B4_WD_FAIL_INTEGRATION` · `B4_BLOCKED_GRAPH_NODE` · `B4_BLOCKED_ROUTE_A_REQUIRED` ·
`B4_BLOCKED_LAYOUT_BRIDGE` · `B4_REST`
