# NV Q6 RegionLoad split-bridge microgate decision (2026-09-01)

## Verdict

`BLOCKED_MARKER_ERASURE`

The CUDA split register-bridge implementation is uncommitted and is **not released** to a real Q6 candidate. The focused UOp/source run stops before CStyle rendering, NVRTC, cubin generation, SASS inspection, GPU execution, correctness, or timing.

## Intended contract

The proposed API is `LOAD.load_in_region_bridge(region)`, distinct from ordinary `LOAD.load_in_region(region)`. It carries an opaque `RegionLoadBridge` marker beside the LOAD and outside its INDEX.

Admission is deliberately narrow:

- CUDA renderer only; unsupported renderers fail closed.
- Exactly one constant-true, workgroup-uniform `PostBarrierRegion`.
- The region is anchored on exactly one existing `Ops.BARRIER` occurrence.
- Exactly 18 unmasked scalar32 GLOBAL loads from one immutable, unqualified PARAM owner.
- Each LOAD has exactly one direct scalar32 LOCAL/shared STORE consumer.
- All 18 STOREs target one LOCAL allocation and are the unique direct roots of the same region.
- No const/restrict pointer qualification, named ordinary C load temporaries, volatile C memory objects, new or suppressed synchronization, MEMBAR, atomics, or fallback lowering.
- Intended CUDA source is one volatile inline-PTX block with 18 `ld.global.u32` outputs immediately before the existing C `__syncthreads()`, followed immediately by one block with 18 `st.shared.u32` inputs. The later publication barrier remains ordinary C.
- Default/unmarked and ordinary RegionLoad paths must remain byte-identical.

The intended physical gate was stable repeated sm120 cubins, exactly 18 LDG and 18 STS, `LDG < overwrite BAR < STS`, first-LDG-to-first-STS span at most 160 instructions, one physical operation per logical copy, no extra LDS versus control, and zero stack/local/LDL/STL/MEMBAR/ATOM. That gate was not reached.

## Uncommitted implementation files

- `tinygrad/uop/ops.py`: `RegionLoadBridge`, singleton marker, and explicit `load_in_region_bridge` builder.
- `tinygrad/uop/spec.py`: typed marker and marked-LOAD tensor-spec admission.
- `tinygrad/renderer/__init__.py`: default fail-closed capability.
- `tinygrad/codegen/__init__.py`: unsupported-renderer admission guard.
- `tinygrad/renderer/cstyle.py`: exact group validation and barrier-preserving split-bridge orchestration.
- `tinygrad/renderer/cuda.py`: CUDA-only 18-output/18-input inline-PTX spelling.
- `test/unit/test_cuda_region_load_bridge.py`: API, byte-identity, malformed-shape, source, and intended SASS microgates.

## Focused result

Command:

```sh
timeout 120s python3 -m pytest -q \
  test/unit/test_cuda_region_load_bridge.py \
  test/unit/test_lexical_load_region.py \
  test/unit/test_post_barrier_region.py
```

Final bounded result:

```text
21 passed, 3 failed in 0.95s
```

The existing lexical/post-barrier coverage and the new explicit API and byte-identity controls passed. Frozen source controls remained:

- Unmarked source SHA-256: `19d2dcc6ec58cd214fcc0f09141b453fbb761f2a6acd973b0020ef460f1b6492`.
- Ordinary RegionLoad source SHA-256: `fd6cf6b86eca16d01f493a5f34236d1dfff8c5dc63913cc6abc8b61c5a369b61`.

All three failures share the same marker-erasure blocker. No generated bridge CUDA source was produced.

## Exact UOp failure

The API constructs the intended marked shape:

```text
LOAD(
  INDEX(GLOBAL_PARAM, scalar_index),
  AFTER(
    IF(CONST(True), anchor_barrier, arg=PostBarrierRegion(workgroup_uniform=True)),
    arg=RegionLoadBridge()
  )
)
```

By final program verification, the LOAD has become:

```text
LOAD(
  INDEX(GLOBAL_PARAM, scalar_index),
  IF(CONST(True), anchor_barrier, arg=PostBarrierRegion(workgroup_uniform=True))
)
```

The exact traceback leaf is:

```text
UOp verification failed ... on Ops.LOAD dtypes.uint 2
[(Ops.INDEX, dtypes.uint, None),
 (Ops.IF, dtypes.void, PostBarrierRegion(version=1, workgroup_uniform=True))]
```

The opaque `AFTER(..., arg=RegionLoadBridge())` wrapper has been erased while its IF source is retained. Accepting this naked IF in `spec_program` would weaken the contract and is not an acceptable repair.

## Rewrite boundary

After the bounded repair, the bridge marker is admitted by initial `type_verify(ast, spec_tensor)`. It is absent when `_full_rewrite_to_sink` reaches final `type_verify(sink, spec_program)`. Therefore erasure occurs inside the full rewrite pipeline before linearization, `validate_post_barrier_regions`, CStyle bridge validation, and CUDA rendering.

The individual rewrite rule was not identified within the established one-repair bound. The next narrow investigation target is the existing ordinary-`RegionLoad` marker-preservation path: `RegionLoadBridge` must survive through the same boundary without making naked `IF` a legal LOAD dependency.

## Bounded repair history

The first focused run exposed two pre-render structural issues: missing UOp spec admission for `RegionLoadBridge`, and an invalid test-only value-valued `GROUP` used to retain the large phase body. The single permitted repair added typed marker admission and made the existing BARRIER depend directly on the live accumulator values. The second run reached final program verification and exposed the marker-erasure blocker above. Work stopped at that bound.

## No SASS or GPU claim

- NVRTC was not invoked for the bridge candidate.
- No cubin was generated.
- No SASS opcode, PC, register, spill, or traffic census exists for this implementation.
- No kernel was launched.
- No correctness or timing measurement was run.

## Q6 isolation

This work did not edit or execute the Q6 builder, Q6 harness, kernel geometry, architecture policy, launch bounds, or GPU path. In particular, `extra/llm_research/prefill/nv_q6_oracle_broad_cta.py` and `extra/llm_research/prefill/bench_nv_q6_oracle_true_late_panel1.py` were untouched by this experiment.

