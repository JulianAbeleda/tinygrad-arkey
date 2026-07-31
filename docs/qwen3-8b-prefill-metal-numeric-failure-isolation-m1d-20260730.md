# M1d — the C-fragment overcount hypothesis: KILLED

Repo `exp` @ `57c704a32`. Compile-only, no GPU workload run (no `.synchronize()`, no
`run_isolated` dispatch, no `Device["AMD"]`/`Device["METAL"]` instantiation at all). Same
dispatch M1b/M1c qualified: Q4_K, `ffn_gate_up`, shape `(512,12288,4096)`, geometry
`(256,64,32,8,1,1)` (`tm,tn,tk,wm,wn,bc`).

Driver: `scratchpad/m1d_confirm_c_fragment.py`. Reuses M1c's payload construction
(`candidate_payload`, `derive_packed_weight_candidate`, `full_kernel_workload`, the same
`_payload_for_local_row` geometry injection) but stops before any device compile: it builds
the Tensor AST via `.schedule_linear()` and renders it with `to_program(ast, renderer)` where
`renderer` is built directly from `Target.parse(...)` — `MetalRenderer(Target.parse("METAL:METAL:Apple9"))`
and `HIPRenderer(Target.parse("AMD:HIP:gfx1100"))` — exactly the technique proven in
`test/unit/test_warp_shfl_xor_renderer_lowering.py`. This lets AMD be rendered (including its
real cross-compile step, `amd_comgr`) on a Mac with no AMD hardware present, and never opens a
real Metal device either. `sys.settrace` captures the exact local variables at the
`tinygrad/codegen/opt/postrange.py` lines the hypothesis is about, without editing that file.

One harness pitfall found and fixed along the way: `to_program` (which triggers
`Kernel.apply_opts`, where the warmstart-table lookup attaches `candidate_context` onto the
ast) must run *inside* the `warmstart_candidate_state(...)` context manager, the same way
`compile_current_prefill_program` nests `compile_linear(...)` inside it. A first draft called
`to_program` after that block had exited; the global candidate-context table was already gone,
so the precontract path never engaged and a generic dense WMMA rendered instead — that was a
bug in the harness, not evidence. Fixed by building the AST and calling `to_program` in one
`with` block. Also hit and worked around: AMD's real `amd_comgr` cross-compiler crashes with a
Bus error compiling this HIP source on this machine (a native-library instability, unrelated to
the code being diagnosed, and occurring strictly *after* the postrange optimization this task
is about, inside `do_compile`). Stubbed `HIPCompiler.compile` to a no-op only for that step
(`unittest.mock.patch.object`), so the real SOURCE and the real postrange trace are unaffected.

## The unresolved question, answered

**`candidate_pipeline` is `None` on both targets, so the branch containing the hypothesized
code never executes.** Concretely (values are frame-locals captured by `sys.settrace` inside
`tinygrad/codegen/opt/postrange.py`, both renders `"error": None`, i.e. both compiled clean):

| | METAL | AMD |
|---|---|---|
| `tc.dims` | `(8,8,8)` | `(16,16,16)` |
| `tc.elements_per_thread` | `(2,2,2)` | `(16,16,8)` |
| `tc_upcast_axes` lengths (line 456) | `(1,1,1)` | `(4,4,3)` |
| `candidate_geometry is None` (line 421) | `False` | `False` |
| `factors.subtiles_m, subtiles_n` (line 435) | `4, 8` | `2, 4` |
| `candidate_axes is None` (line 456) | `False` | `False` |
| `register_mode` (line 466/473) | `False` | `False` |
| **`candidate_pipeline is None` (line 473)** | **`True`** | **`True`** |
| Line 494–522 (the `len(c_axes) != 3` / `accumulator_total = ...*8` block) reached? | **No** | **No** |
| Branch actually taken (line 523 `elif not register_mode:`) | **Yes** | **Yes** |

`register_mode` is `False` on both because this is the packed-Q4_K path
(`kernel_lds.py:246`: `if self.register_mode: raise ValueError("packed-weight candidate
requires LDS tile storage")` forces `register_mode=False` for any packed candidate that
compiles at all). `candidate_pipeline` is `None` on both because this geometry's `bc` (buffer
count) is `1` — a single LDS buffer, not a double-buffered pipeline — so
`PrecontractCandidateContract.pipeline` (`kernel_lds.py:196-197`,
`getattr(self.context, "pipeline", None)`) has nothing to return. This is a property of the
*geometry* (`bc=1`), not of the *target* — both METAL and AMD take the identical branch for the
identical reason.

Consequently `postrange.py:473`'s `if candidate_pipeline is not None and not register_mode:`
is `False` on both, so its body (474–522 — the code with `len(c_axes) != 3`, the literal `8` in
`accumulator_total = factors.subtiles_m*factors.subtiles_n*8`, and the `tc.dtype_out.vec(tc.elements_per_thread[2])`
WMMA-node construction the hypothesis names) **never runs on either target for this dispatch**.
Execution instead falls to `postrange.py:523`, `elif not register_mode:`, which calls
`build_precontract_lds_stage` (`kernel_lds.py:598`) — a structurally different code path the
hypothesis was not written against.

## What the actually-executed path does with the C fragment

`build_precontract_lds_stage` / `instantiate_precontract_fragments`
(`kernel_lds.py:578-596`) only ever produces **A and B** fragments
(`PrecontractLDSStage.fragment_a`, `.fragment_b` — there is no `fragment_c`). Back in
`postrange.py:536-540`, since `pipeline_tc_uop` is `None` on this branch, the WMMA node is built
directly and generically:

```python
wmma = UOp(Ops.WMMA, dtype=tc.dtype_out.vec(tc.elements_per_thread[2]), src=(
  wmma_srcs[0], wmma_srcs[1], UOp.const(tc.dtype_out.vec(tc.elements_per_thread[2]), 0.0)), arg=wmma_arg, tag=1)
tc_uop = UOp(Ops.UNROLL, tc.dtype_out, (wmma,), arg=tc_upcast_axes[2], tag=1)
```

Both the WMMA node's dtype and its zero-initialized C input are `tc.dtype_out.vec(tc.elements_per_thread[2])`
— read directly off the target's own `tc`, no literal `3` or `8` anywhere in this branch. This
is confirmed against the real rendered kernel source (both kernels carry the true packed-Q4_K
ABI — `unsigned int* data2` / `device uint* data2` for the packed B weights, plus the actual
nibble-unpack dequant arithmetic, e.g. Metal line 143: `((val0>>cast4)&63u)` — these are the
real production kernels, not a stand-in dense GEMM):

- **METAL** (`/tmp/m1d_metal_source.c`): `float buf0[64];` (accumulator array — matches
  `subtiles_m(4) * subtiles_n(8) * elements_per_thread[2](2) = 64`). WMMA wrapper:
  `float2 __WMMA_8_8_8_half_float(half2 a, half2 b, float2 c){ ... simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c); ... }`,
  called with 2-wide reads, e.g. `float2((*(buf0+34)),(*(buf0+35)))`. 128 call sites, 1 wrapper
  definition (`grep -c "__WMMA" == 129`), 1 `simdgroup_multiply_accumulate` occurrence (inside
  the one wrapper body, called 128 times — expected, not a bug signal).
- **AMD** (`/tmp/m1d_amd_source.c`): `float buf0[64];` (matches
  `subtiles_m(2) * subtiles_n(4) * elements_per_thread[2](8) = 64`). WMMA:
  `#define __WMMA_16_16_16_half_float __builtin_amdgcn_wmma_f32_16x16x16_f16_w32`, called with
  8-wide reads, e.g. `make_float8((*(buf0+8)),(*(buf0+9)),...,(*(buf0+15)))`. 16 call sites
  (`grep -c "__WMMA_16_16_16_half_float(" == 16`).

Both accumulator arrays land on the same total size (64) via different, target-correct
factorizations (`4*8*2` vs `2*4*8`), and every WMMA call site on each target reads/writes
exactly its own target's `elements_per_thread[2]`-wide slice of `buf0` — 2-wide throughout on
Metal, 8-wide throughout on AMD. No call site on either target reads or writes a width that
disagrees with its own `tc`. `search __WMMA` / `simdgroup_multiply_accumulate` (never
`simdgroup_matrix`, which does not appear in either source, confirming this run avoided the
prior mis-conclusion) confirms this directly in the emitted text, not by re-deriving it from
the code.

## Verdict: KILLED

The specific mechanism named in the hypothesis — `postrange.py:498`'s
`if len(c_axes) != 3: raise KernelOptError(...)` and `postrange.py:505`'s
`accumulator_total = factors.subtiles_m*factors.subtiles_n*8` forcing AMD's 8-element C
fragment onto Metal's 2-element one — **does not execute at all** for the M1b/M1c/M1d
dispatch, on either target. That code is dead for this configuration because `bc=1` (no
double-buffered pipeline) routes both targets through `postrange.py:523`'s
`elif not register_mode:` branch instead, and that branch's own C-fragment/accumulator
construction (`postrange.py:538-540`) is already generic — it reads `tc.elements_per_thread[2]`
directly, not a hardcoded `8` — and renders self-consistently on both targets, confirmed in the
real emitted source text.

This rules out: any hardcoded-`8`-into-Metal's-`2` C-fragment ABI mismatch, for this dispatch,
via this code path, as the cause of the M1c-documented Metal failure (18.7% write coverage,
non-determinism up to 3904.0 between rounds, a wave-4-sized garbage cluster). It does **not**
explain what does cause it — the actual root cause remains open.

## What I could not establish

- Why the Metal kernel under-writes 82% of the tile and produces non-deterministic garbage in
  exactly wave 4's row range (M1c's open question) is **not answered by this task** and I did
  not attempt to answer it here — M1d was scoped to the one hypothesis, and killing it does not
  by itself point at a replacement.
- While reading `build_precontract_lds_stage`'s cooperative-store row election
  (`kernel_lds.py:cooperative_store_row`/`cooperative_store_row_rotation`), I noticed the store
  side (`instantiate_precontract_producer`) optionally re-elects a lane's target row via
  `cooperative_store_row(...)`, gated on the renderer's declared `lds_bank_dwords`/
  `lds_bank_cycle_lanes` (AMD declares `32`/`8`; Metal leaves both `None`, per
  `tinygrad/renderer/cstyle.py:588-589` vs `:496`, so the rotation is unconditionally skipped on
  Metal — `raw_row` returned unchanged). The fragment/load side
  (`instantiate_precontract_fragments`) computes its row directly
  (`row=(wave*subtiles+subtile)*16+lane%16`) without calling `cooperative_store_row` at all. I
  did not work out whether that asymmetry is provably safe (e.g. because the rotation is
  documented as "still an exact one-writer cover of the tile" regardless of whether reads
  compensate) or whether it is a real store/load addressing mismatch on the target that *does*
  enable it (AMD) — and since Metal never enables the rotation, this specific mechanism cannot
  be the Metal bug either way. Flagging it only as an unexamined area for a possible next step,
  not as a finding.
- I did not re-derive the M1c failure signature (missing-write pattern, wave-4 magnitude,
  non-determinism) against this now-confirmed actually-executed code path
  (`build_precontract_lds_stage`/`instantiate_precontract_fragments`/`instantiate_precontract_producer`)
  to look for a different mechanism there. That would be the natural next step (M1e) but is
  outside this task's scope (confirm-or-kill one named hypothesis).

## Files

- `scratchpad/m1d_confirm_c_fragment.py` — the driver used for this run (compile-only; no
  `Device[...]` instantiation, no `.synchronize()`, no dispatch).
- `/tmp/m1d_trace_result.json` — full postrange trace + summary for both targets (not
  committed; local scratch output, reproducible by re-running the script).
- `/tmp/m1d_metal_source.c`, `/tmp/m1d_amd_source.c` — full rendered kernel source for each
  target from this run (not committed; local scratch output).
