# M4 residual_add rangeify substrate scope - flat SPECIAL reads of non-flat opaque outputs

Date: 2026-08-06
Branch boundary: tinygrad `nvidia-bringup-20260731`, HEAD `6f1abd047`
Status: **implementation scope. S1-S3 DONE on `nvidia-bringup-20260731` (commits `3fa55377c`,
`2794d6772`, plus the S3 codegen fold): the open-resadd flash-decode graph, with the PRODUCTION
residual fold ACTIVE, schedules end to end (1620 kernels) after two 5-line scheduler deltas, and
the folded epi_resadd subgraph now EXECUTES on CPU with the residual read through a flat row
index (host proof, bitwise-equal to the copy-ABI variant). S4 (GPU section-6 gate, lock-held)
is the next step, gated on the S3 codegen finding below landing first; the reopen/promotion
decision of `decode-q4k-epilogue-resadd-route-policy.json` stays pending S4. Everything else
stays closed.**

## 1. Why this scope exists

The blocked record (`m4-resadd-landing-blocked-record-20260806.md`) lists three reopen
conditions; condition (a) is:

> the rangeify substrate that lets a kernel read a flat, SPECIAL-indexed view of a non-flat
> opaque output (new scope, touches `schedule/indexing.py` and `schedule/rangeify.py`)

This scope IS that substrate. Its question is narrower than "does the fold fire": the fold
machinery and the epi_resadd emitter already exist and are measured (probe 2). The missing
piece was the scheduler's ability to rangeify a kernel body that (i) inlines precompiled
flash/combine chains and (ii) reads the non-flat `(1,1,4096)` opaque block-output boundary
through flat `INDEX(AFTER(...), SPECIAL gidx0)` views. Both shapes crashed at schedule time
on the flash-decode graph; this session proved both are 5-line fixes with no model change.

### 1.1 Correction to the blocked record's root-cause attribution

The blocked record attributed the crash to the folded residual read alone
(`RESHAPE(INDEX(AFTER(epi GEMV output, CALL), SPECIAL gidx0), (1,1,4096))`). This session
established the fuller picture:

- The production fold **fires on the real graph** for every block except layer 0:
  `_validated_residual_view` returns `(CONTIGUOUS, "ok")` for the real residual chain
  `RESHAPE(RESHAPE(MS(CONTIGUOUS(GETTUPLE(FUNCTION(FFNBlock._run, precompile=True))))))`
  (verified by instrumentation over the live forward pass, 36 blocks). Layer-0 embedding
  (`CAST(REDUCE(...))`) correctly fails closed ("producer has no buffer/precompiled-output
  identity") and keeps its boundary copy.
- The schedule-time crash had two sources in the epi_resadd body: the fold's own
  `RESHAPE(INDEX(AFTER(...), SPECIAL), (1,1,4096))` nodes (the `INDEX` arm of
  `remove_movement_op_after_rangeify`, already present) and the **inlined flash tile/combine
  chains** `RESHAPE(REDUCE(INDEX(STAGE(1,1,1,4096,128)), CONST 0), (1,1,4096))` (the missing
  `REDUCE` arm). The latter was the actual `ValueError: bad reshape`; the former was masked
  by it. Both are now handled.

## 2. The substrate deltas (5 lines total, already applied and verified)

### D1 - movement-op removal after rangeify also drops REDUCE-scalar views

`tinygrad/schedule/indexing.py:150`, `remove_movement_op_after_rangeify`:

```python
if x in ctx.range_map or x.src[0].op in (Ops.INDEX, Ops.REDUCE): return x.src[0]
```

After rangeify, a movement op over a rangeified scalar (`INDEX` or `REDUCE` whose shape is
`()`) is a pure view: the STAGE/consumer ranges carry the shape. The `INDEX` arm existed;
the `REDUCE` arm was missing, so 3 `RESHAPE(REDUCE(INDEX(STAGE)))` nodes survived into the
debuf pass and crashed `cleanup_dead_axes` with `bad reshape: () -> (1, 1, 4096)`.

### D2 - WAR assign loop skips the edge when the reader already depends on the writer's AFTER

`tinygrad/schedule/rangeify.py:1274`, `_get_kernel_graph` fix_assign:

```python
# The reader already depends on the writer's AFTER (precompiled-output identity): the read
# is ordered after the write, so the WAR edge is redundant and would be a false cycle.
if any(x.op is Ops.AFTER and x.buf_uop is s for x in u.backward_slice): continue
if any(x.op is Ops.AFTER and x.buf_uop is s for x in kernel_assign[u.buf_uop].backward_slice):
  raise RuntimeError(...)
```

The folded graph's consumer kernels carry the producer's `AFTER` as a call argument (the
precompiled-output identity pattern) while also reading the produced buffer raw. The WAR
pass wanted to order the writer after the reader, creating a false cycle. The distinguishing
rule, verified against the upstream guard: when the **reader's own** backward slice contains
`AFTER(s)`, the read is already ordered after the write (skip); when only a **different**
writer of the reader's buffer contains `AFTER(s)` (the `test_crossunder_assign` shape), the
cycle is real (raise).

## 3. Viability evidence (this session, CPU hermetic, no GPU)

Repro: `/tmp/m4_viab_repro.py` (fake NV sm_120 facts, Qwen3-8B-Q4_K_M, flash decode, resadd
record forced open, `decode_enabled` set, schedule-only via `create_linear_with_vars`).

| arm | before deltas | after deltas |
| --- | --- | --- |
| open (fold active) | `ValueError: bad reshape` at schedule time | **`SCHEDULE OK 1620 kernels`** |
| closed (default records) | `SCHEDULE OK 953 kernels` | `SCHEDULE OK 953 kernels` (unchanged) |
| crossunder assign | raises `cycle detected` | still raises (guard intact) |

Unit/adjacent suites on the delta tree (all green): M4 landing + fold probe + kernel
microgate + M5 typed boundary + flash-combine M5 + rangeify multireduce = 65 passed;
adjacent gate suite 43 passed (the 1 failure is the pre-existing
`test_production_llm_does_not_import_or_call_oracle_or_research_execution`); schedule
probes 37 passed. The open census shows the fold active in the scheduled graph
(epi_resadd GEMV residual args are direct `CONTIGUOUS(1,1,4096)` block-output views) plus
the inlined chain kernels (564 anonymous landing kernels, flash tile/combine present).

## 4. Substrate work items

### S1 - Land the deltas

Commit D1+D2 on `nvidia-bringup-20260731` (no push). House rule: scheduler change is a
substrate decision per the blocked record's condition (a); the commit message must cite
this scope and the blocked-record reopen condition.

### S2 - Unit coverage that locks the substrate

- `test/unit/test_rangeify_movement_reduce_view.py` (new): build the exact
  `RESHAPE(REDUCE(INDEX(STAGE(1,1,1,4096,128), CONST 0)), (1,1,4096))` chain plus the
  `INDEX`-arm contrast, run `run_rangeify`/`get_kernel_graph` on a consumer SINK, assert
  both movement arms drop and the debuf pass no longer sees a `()`-to-shaped reshape.
- `test/unit/test_rangeify_war_after_dependency.py` (new): construct the
  precompiled-output identity consumer (kernel reads buffer s raw and carries `AFTER(s)` as
  an arg), assert the WAR loop skips the edge and no cycle is raised; reconstruct the
  crossunder shape (different writer holds `AFTER(s)`), assert the raise is preserved.
- `extra/llm_research/decode/m4_resadd_substrate_schedule_gate.py` (new, CPU hermetic):
  promote the viability repro into the research tree: force the record open, build the
  flash-decode graph, assert `SCHEDULE OK`, assert the fold fires (`_validated_residual_view`
  returns `ok` for a block >= 1 and rejects layer 0), and assert the open census family
  counts (epi_resadd present, legacy attn_qo GEMV count, copy-class baseline). This is the
  regression gate for the substrate, runnable without a GPU.

### S3 - Host-side execution proof (CPU, before any GPU time)

Schedule-time viability is proven; the fold also changes what the HOST must launch: the
folded residual arg is the block-output view, and the kernel's ordering comes from the
`AFTER` dependency argument (which `create_schedule`/`_split_after` already consume). Proof
target: execute the open 1620-kernel schedule on CPU and compare token stream sha + first
token against the closed 953-kernel schedule (bitwise equality, same pins as the gate).
Fallback if full-CPU execution is impractical (custom lane-map kernels may be slow on CPU):
execute a single-block folded epi_resadd subgraph on CPU with the residual as the block-
output AFTER and assert numeric equality against the copy-ABI variant.

**S3 result (DONE, fallback arm).** The single-block host proof is landed and green:
`extra/llm_research/decode/m4_resadd_substrate_host_exec.py` + unit locks
(`test/unit/test_m4_resadd_substrate_host_exec.py`,
`test/unit/test_custom_kernel_shaped_param_fold.py`). Fold fires on the real
`CONTIGUOUS(1,1,N)` block-output chain and fails closed at layer 0; folded vs copy-ABI are
bitwise equal (sha `8991b4eb458eb34e` both arms), and the zero-dot folded kernel reads the
producer buffer directly.

**S3 finding (must land before S4): shared-codegen shaped-PARAM render crash.** Executing the
folded kernel exposed a NEW render blocker that the schedule-only gates could not see:
`UOp.custom_kernel` builds kernel-body placeholders via `UOp.placeholder_like`, which reshapes
when `len(shape) > 1`; the folded residual `CONTIGUOUS(1,1,4096)` arg therefore reaches the
body as `RESHAPE(PARAM, shape-STACK)`. No codegen pass consumes the shape STACK
(`pm_remove_vec_dtypes` scalarizes `weakint.vec(3)` -> `type_verify` "weakint is not
allowed") and no renderer has a RESHAPE rule, so the folded kernel crashed at render (this
would have hit NV at S4). Fix: two rules in `pm_index_is_shrink`
(`tinygrad/codegen/__init__.py`): fold shaped GLOBAL-ptr PARAM views (RESHAPE/EXPAND over the
flat PARAM or a CONST) to the PARAM, and load scalar pointer-typed INDEX value reads in the
explicit `.cast(dtype)` spelling. Image buffers and multi-index (3+ src) INDEX bases are
untouched. Kernel bodies must read shaped args as explicit value reads
(`extra[0][row].cast(dtypes.float32)`, the production epi_resadd spelling); a bare `buf[i]`
in ALU is a pointer and was never renderable on CPU. The flat-placeholder alternative
(`UOp.placeholder((prod(...),), ...)` in `custom_kernel`) was rejected: it broke production
q6k kernels (`partials[row, pos]` multi-dim indexing) and the S2 gate's open arm.

### S4 - GPU section-6 gate re-run (lock-held, after S3)

Re-run `extra/llm_research/decode/m4_resadd_section6_gate.py` per the landing scope
section 3, with the deltas landed:

1. d512/d2048/d4096 wall, open mode, vs the M2-on baseline; positive delta attributable to
   copy-free residual_add.
2. Census assertions, re-derived for the substrate graph: epi_resadd count (expected shape
   needs re-baselining: the open census shows 71 epi calls on this HEAD, i.e. ~2x36-1, with
   a mix of folded `CONTIGUOUS(1,1,4096)` and materialized residual args - resolve which
   blocks/call sites each class is before asserting), `E_32_32_4_86a2` residual-slot copies
   = 0 for blocks 1+ and **exactly 1 remaining for layer 0** (layer-0 fail-closed is
   deliberate, update the landing scope's "count 0" assertion), legacy attn_qo GEMV 0.
3. Pins 3/3 both modes at every depth; token-stream equality between record-open and
   forced-open states after promotion.
4. pg3 legacy hash `27857cb8ca03` for `q4k_g3_lanemap_gemv_4096_4096` unmoved (GPU check).
5. `test/unit/test_m5_typed_boundary.py` (27) green on the same tree.

### S5 - Records and promotion decision

- Update `m4-resadd-landing-blocked-record-20260806.md`: reopen condition (a) is satisfied
  (this scope); correct the root-cause attribution per section 1.1; record the 1620/953
  schedule proof and the delta refs.
- After the S4 gate passes: promote `decode-q4k-epilogue-resadd-route-policy.json` to
  `NV sm_120`, book the recovery per the ledger rules (same-session gate delta, no synthetic
  microgate extrapolation).

## 5. Open questions and risks

- **71 vs 36 epi census (RESOLVED)**: the 71 is the toposort-unique count (precompiled block
  bodies' PARAM-form kernels); the schedule EXECUTES exactly 36 epi_resadd GEMVs (one per
  block), asserted by the S2 gate (`EXPECTED_EPI_RESADD = 36`). S4 census baselines:
  open 1620 kernels / copy_class 150 / epi_resadd 36 / legacy `4096_4096` 36; closed 953
  kernels / copy_class 150 / legacy `4096_4096` 72.
- **CPU execution cost**: the 1620-kernel schedule on CPU may be slow (custom lane-map
  bodies). S3 has a single-block fallback that still exercises the folded AFTER-arg host
  path.
- **Layer-0 copy**: the gate's "copy count 0" assertion must become "0 for blocks 1+, 1 for
  layer 0" (fail-closed by design).
- **d4096 hang**: pre-existing prefill hang (HCQ wait timeout), reproduced on pristine HEAD;
  out of scope, but it will cap which depths the S4 gate can measure.
- **WAR skip generality**: the skip fires whenever the reader's own slice contains
  `AFTER(s)`. The crossunder guard is preserved, but the S2 unit test must also cover the
  reader-carries-AFTER vs writer-carries-AFTER distinction so a future refactor cannot
  silently widen the skip.

## 6. Sequencing

S1 (land) -> S2 (unit + CPU schedule gate) -> S3 (CPU execution/host proof) -> S4 (GPU
section-6 gate) -> S5 (records + promotion decision). **S1, S2, S3 are DONE**; S4 is next.
S2 is required before S3; S3 is required before S4; S4 is required before promotion. Each
step is independently revertible; the deltas are 5 lines and D2 is isolated by the S2
crossunder test.

## 7. HARD STOP

This scope authorizes exactly: S1-S5 as listed. It does NOT authorize: the ffn_down prelude
(`ffn_down_fused`), `fp16_cast`, the combined `decode-q4k-epilogue-fusion-route-policy.json`
record, M3/M5/Path 3 records, w1w3 or kv-store records, any emitter change
(`decode_kernels.py` bodies and names stay), flattening the `(1,1,4096)` h representation,
any GPU probe outside `flock -w 600 /tmp/gpu-bench.lock`, or any promotion without the S4
gate passing.

## 8. References

- `m4-resadd-landing-blocked-record-20260806.md` (reopen condition (a), gate protocol)
- `m4-resadd-landing-scope-20260806.md` (production shape, section-6 gate, recovery rules)
- `m4-residual-boundary-fold-probe-record-20260806.md` (probe 1: chain purity, layer-0
  fail-closed)
- `m5-typed-boundary-p0-implementation-record-20260803.md` (AFTER declared-typed-output ABI)
- `tinygrad/schedule/indexing.py` (D1), `tinygrad/schedule/rangeify.py` (D2)
- Evidence scripts: `/tmp/m4_viab_repro.py`, `/tmp/m4_viab_repro_closed.py`,
  `/tmp/m4_chain_inspect.py`, `/tmp/m4_viab_census2.py`, `/tmp/m4_fold_fire_check2.py`,
  `/tmp/crossunder_test.py`
