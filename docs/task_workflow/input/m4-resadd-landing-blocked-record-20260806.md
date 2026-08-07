# M4 residual_add landing BLOCKED record

Date: 2026-08-06
Status: **BLOCKED. The section-6 gate's OPEN arms crash at schedule time on the flash-decode
graph** (`ValueError: bad reshape: () -> (1, 1, 4096)` in `schedule/rangeify.py`
`cleanup_dead_axes`, second decode token, both d512 and d2048). The production residual fold
is **not realizable in the current runtime**; the per-variant resadd record
(`decode-q4k-epilogue-resadd-route-policy.json`) stays **CLOSED** (`promoted_targets: []`),
and **0 credit** is booked. This is a real landing bug, not environmental: probe-2 passed
because it ran the epi_resadd kernel WITHOUT the fold (boundary copies present); the
production fold was never GPU-exercised until the section-6 gate.

## Protocol

Same-session, lock-held (`flock -w 600 /tmp/gpu-bench.lock`), Qwen3-8B-Q4_K_M, nmeas 20,
reps 3, median tok/s, fused prefill attention disabled. Open mode = the per-variant record
forced open for NV sm_120 (`mrp._DECODE_Q4K_EPILOGUE_RESADD_PROMOTED_TARGETS =
frozenset({("NV","sm_120")})`) with the **production fold ACTIVE** (no admission surgery);
closed mode = default records (fold dormant). Each arm runs as a fresh subprocess. The
runner is `extra/llm_research/decode/m4_resadd_section6_gate.py` (patched this session:
`--arm record` mode, per-leg error tolerance instead of abort, full run continues past
failed legs).

## Gate results

| arm | d512 | d2048 | d4096 |
| --- | --- | --- | --- |
| closed | median **180.694 tok/s** (reps `[6.549, 180.709, 180.694]`), sha `227ad3ce...` 3/3, first `271` 3/3, census 948 kernels/token, epi 0, legacy 72, copy class 1, resadd 72 | median **169.990 tok/s** (reps `[2.591, 169.990, 170.169]`), sha `aca13ac6...` 3/3, first `271` 3/3 | **pre-existing hang** (HCQ wait timeout; see below) |
| open (production fold ACTIVE) | **CRASH at schedule time**, second decode token: `ValueError: bad reshape: () -> (1, 1, 4096)` | **same crash** | not reached (closed arm hang blocks the depth; the decode-graph crash is depth-independent) |
| record (checked-in policy) | not run | not run | not run |

The closed arms are **exact probe-2 control reproductions**: same wall, same census shape,
same pins - no regression from the landing code (dormant wiring). The open arms crash on
the second decode token in BOTH depths, at schedule time (before any kernel executes),
with the identical bad reshape. The section-6 gate therefore FAILS at item 1 before any
wall can be measured, and the census/pin items are unattainable in open mode.

## Root cause (traced to the node)

`create_bufferize_and_index_based_on_ranges` (`tinygrad/schedule/indexing.py:56`, created
at the `x.replace(src=tns)` return, line ~88) produces, for the folded residual read:

```text
RESHAPE(
  INDEX(
    AFTER(
      <epi_resadd GEMV output PARAM>,
      CALL q4k_g3_lanemap_gemv_epi_resadd_4096_4096
    ),
    SPECIAL gidx0
  ),
  (1, 1, 4096)
)
```

The emitter reads the residual slot as `extra[0][row]` with `row = UOp.special(rows,
"gidx0")` (`tinygrad/llm/decode_kernels.py:133`, `:233`). `INDEX(ptr, SPECIAL)` has shape
`()` because SPECIAL contributes no shape; reshaping `()` to the `(1,1,4096)` block-output
buffer shape is illegal (`prod 1 != 4096`), and `rangeify.cleanup_dead_axes` raises
`ValueError: bad reshape: () -> (1, 1, 4096)`.

The M5 typed fold works only because its producer AFTER is already FLAT (`(Hq*Hd,)`). The
runtime cannot express a zero-copy flat (SPECIAL-indexed) kernel read of the non-flat
`(1,1,4096)` opaque block-output boundary. The closed arm avoids it by materializing the
flat boundary copy (`E_32_32_4_86a2`, 72/token) - exactly the copy the fold was meant to
remove.

## Fix candidates tried (all fail identically)

Three substitution strategies for the folded residual input were verified in the real
decode graph; all produce the same schedule-time crash:

| Candidate | Substitution | Result |
| --- | --- | --- |
| raw base | `CONTIGUOUS(GETTUPLE(FUNCTION))` directly | same crash |
| flat view | `Tensor(view).reshape(N)` over the validated view | same crash |
| flat GETTUPLE | `Tensor(GETTUPLE).reshape(N)` (`has_precompiled_output_identity()==True`) | same crash |

The crash also reproduces with the M5 combine-fusion record closed (`/tmp/m4_fold_m5off.py`),
so it is not an M5 interaction. Hermetic repro/evidence: `/tmp/m4_fold_repro.py`,
`/tmp/m4_fold_trace.py` (bad-node dump), `/tmp/m4_fold_scan.py`, `/tmp/m4_fold_fix2_test.py`,
`/tmp/m4_fold_repro.log`, `/tmp/m4_fold_trace.log`, `/tmp/m4_open_d512_debug.log`,
`/tmp/m4_open_d2048.log`.

## Verdict and decision point

The zero-copy M4 fold is **not realizable in the current runtime** without either (a) a
scheduler/rangeify change to support flat SPECIAL reads of non-flat opaque buffers, or (b)
flattening the model's `(1,1,4096)` h representation. The epi_resadd route must **stay
CLOSED**: opening it currently crashes; opening it without the fold is a measured **-0.88%
wall regression** (probe-2, copies present). Book **0 recovery**.

Reopen conditions (any one): the rangeify substrate that lets a kernel read a flat,
SPECIAL-indexed view of a non-flat opaque output (new scope, touches `schedule/indexing.py`
and `schedule/rangeify.py`), or flattening the block-output h boundary to `(N,)` so the
fold re-proves against a flat producer, or abandoning the route.

## d4096 pre-existing hang (not caused by the landing)

d4096 remains blocked by a **pre-existing prefill hang**: kernels slow to 4-5s at large KV,
HCQ `Wait timeout: 30000 ms` (timeline 1-5 signals behind), reproduced on pristine HEAD
(`/tmp/m4_pristine_d4096.log`) and on the landing tree (`/tmp/m4_d4096_hang_probe.log`,
`/tmp/m4_d4096_retry.log`). The campaign previously dodged it with resident-zero-KV gates.
The d4096 closed-arm failure in the gate artifact is this hang, not the fold.

## Files

- Gate runner: `extra/llm_research/decode/m4_resadd_section6_gate.py`
- Gate artifact: `/tmp/m4_resadd_section6_gate_out.json` (closed d512/d2048 only)
- Landing wiring (dormant while record closed): `tinygrad/llm/model_route_plan.py`,
  `tinygrad/llm/qk_primitives.py`, `tinygrad/llm/kernel_program.py`,
  `tinygrad/llm/decode_routes.py`, `tinygrad/llm/model.py`
- Unit tests: `test/unit/test_m4_resadd_landing.py`
- Record: `tinygrad/llm/generated/decode-q4k-epilogue-resadd-route-policy.json`
  (`promoted_targets: []`, stays closed)
- Scope: `m4-resadd-landing-scope-20260806.md` (section 6 gate authority)

## Update 2026-08-06 (rangeify substrate, S1-S3 landed)

Reopen condition (a) is now satisfied by the rangeify substrate scope
(`m4-resadd-rangeify-substrate-scope-20260806.md`), landed on
`nvidia-bringup-20260731`:

- S1: scheduler deltas D1 (`remove_movement_op_after_rangeify` REDUCE arm) and D2
  (`fix_assign` WAR AFTER skip) land the open-resadd flash-decode graph end to end on CPU:
  `SCHEDULE OK 1620 kernels` open vs 953 closed, crossunder raise preserved.
- S2: unit locks (`test_rangeify_movement_reduce_view.py`,
  `test_rangeify_war_after_dependency.py`) plus the hermetic schedule gate
  (`extra/llm_research/decode/m4_resadd_substrate_schedule_gate.py`, `--arm both`):
  open 1620 / epi_resadd 36 / legacy `4096_4096` 36 / copy_class 150; closed 953 / legacy
  72 / copy_class 150. This also resolves the record's 71-vs-36 census question: 71 is the
  toposort-unique count (precompiled PARAM-form bodies), the schedule EXECUTES exactly 36
  epi_resadd GEMVs (one per block).
- S3: the folded epi_resadd subgraph EXECUTES on CPU with the residual read through a flat
  row index, bitwise-equal to the copy-ABI variant (host proof + unit locks). This required
  a new shared-codegen finding: `UOp.placeholder_like` reshapes non-flat custom-kernel args,
  so the folded `CONTIGUOUS(1,1,4096)` residual reached the body as
  `RESHAPE(PARAM, shape-STACK)` and crashed render (no codegen pass consumes the shape
  STACK, no renderer has a RESHAPE rule). `pm_index_is_shrink` now folds shaped GLOBAL-ptr
  PARAM views to the flat PARAM and loads scalar pointer-typed INDEX value reads (explicit
  `.cast()` spelling); flat-placeholder ABI was rejected (breaks q6k multi-dim indexing).
  This codegen fix must land before any S4 GPU gate run.

Status remains **BLOCKED for promotion**: the section-6 gate has not yet re-run on GPU in
open mode with the substrate landed. Promotion of
`decode-q4k-epilogue-resadd-route-policy.json` requires the S4 gate (lock-held, census
assertions re-derived per the substrate scope: epi 36, copy-class 150, layer-0 copy count 1)
to pass; 0 credit remains booked until then.

## Update 2026-08-06 (S4 gate re-run: NEW render-time blocker, record stays CLOSED)

The S4 gate was executed on the GPU with the S1-S3 deltas landed (same-session,
lock-held, Qwen3-8B-Q4_K_M, nmeas 20, reps 3, median tok/s, per-arm fresh subprocesses,
runner `extra/llm_research/decode/m4_resadd_section6_gate.py`). Results:

| arm | d512 | d2048 | d4096 |
| --- | --- | --- | --- |
| closed | **PASS** median 180.982 tok/s (reps `[6.538, 181.019, 180.982]`), sha `227ad3ce...` 3/3, first `271` 3/3, census 948 kernels/token, epi 0, legacy 72, copy class 1, resadd 72 | **pre-existing HCQ hang** (`Wait timeout: 30000 ms!` in `runtime/support/hcq.py`, "NV synchronization failed before finalizing") | **pre-existing HCQ hang** (same) |
| record (checked-in policy) | **PASS** median 180.376 tok/s, sha `227ad3ce...` 3/3, first `271` 3/3, census 948 / epi 0 / legacy 72 / copy 1 / resadd 72 (record == closed behavior, as designed) | **pre-existing HCQ hang** | not run (d4096 capped by the known hang; record already proven == closed at d512) |
| open (production fold ACTIVE) | **FAIL: NEW render-time crash** (below), deterministic (2/2 runs) | **FAIL: same render-time crash** | compiled and ran kernels on GPU, then hit the pre-existing HCQ hang |

**NEW blocker (distinct from the old `bad reshape` schedule crash, which is fixed).** The
open arms now crash at render time in the precompile-kernels walk:

```text
RuntimeError: UOp verification failed at 31 on Ops.SPECIAL dtypes.weakint 1
[(Ops.CONST, dtypes.weakint, 4096)] gidx0
```

raised at `tinygrad/uop/spec.py:69` (`type_verify`), reached from
`tinygrad/codegen/__init__.py:337` (`if SPEC: type_verify(sink, spec_program)`) inside
`_full_rewrite_to_sink`, i.e. AFTER schedule-time succeeds. The failing node is a
weakint-typed `SPECIAL gidx0` whose src is `CONST weakint 4096`. `spec_program` bans it:
the weakint catch-all `(UPat(GroupOp.All, dtypes.weakint), lambda: False)` at
`spec.py:490` matches, while the permissive `SPECIAL` rule at `spec.py:237` lives in
`spec_shared`, which is appended after the catch-all in the concatenated matcher and
never saves weakint `SPECIAL`s. This is exactly the class of finding S3's CPU-only proof
could not see (the CPU render path has no gpudims/`SPECIAL`); it would have hit NV at S4.

The d4096 HCQ hang remains the documented pre-existing environmental hang (reproduces on
closed and record arms too) and caps measurable depths. The section-6 gate therefore FAILS:
the open arm cannot render on NV at any depth, so no wall/census/pins are attainable in
open mode. **No promotion.** `decode-q4k-epilogue-resadd-route-policy.json` stays CLOSED
(`promoted_targets: []`), 0 credit booked. The substrate deltas (D1/D2) are NOT reverted;
they are correct and unit-locked (68 passed incl. `test_m5_typed_boundary` 27/27). A
follow-on scope is required to make weakint `SPECIAL` renderable on NV before S4 can pass.

Evidence: `/tmp/m4_gate_closed_d512.json`, `/tmp/m4_gate_closed_d2048.json`,
`/tmp/m4_gate_closed_d4096.json`, `/tmp/m4_gate_open_d512.json`,
`/tmp/m4_gate_open_d2048.json`, `/tmp/m4_gate_open_d4096.json`,
`/tmp/m4_gate_record_d512.json`, `/tmp/m4_gate_record_d2048.json`, and tracebacks in
`/tmp/m4_gate_open_d512.err`, `/tmp/m4_gate_open_d2048.err`,
`/tmp/m4_gate_open_d4096.err`. Full run record:
`m4-resadd-s4-gate-run-record-20260806.md`.
