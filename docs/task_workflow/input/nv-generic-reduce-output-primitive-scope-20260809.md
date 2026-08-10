# NV generic cooperative reduction-to-output primitive scope (CPU capability gate)

Date: 2026-08-09
Branch: `nvidia-bringup-20260731`, HEAD `bd61bdcf2` (post landing of the
shared-Q8 promotion, section-6 booking, and all three reopen arms).
Status: **implementation/test scope. Authorizes a CPU-only capability gate
that turns the single-recipe `REDUCE_OUTPUT` body into a shape/recipe-generic
cooperative reduction-to-output primitive, plus the production census that
proves admissions become possible. No GPU time, no policy promotion, no
model wiring change.**

## 1. Why this scope exists

The fusion/dataflow ledger bucket is 662.128 us/token of attribution. The
08-07 capability audit (`nv-substrate-capability-vs-ledger-scope-20260807.md`)
found 658.359 us of it (norms 495.330 + flash 163.029) sits behind ONE
missing construct, C1: a generic cooperative reduction-to-output primitive.
The phase-0 record (`nv-boundary-free-ordinary-uop-phase0-record-20260805.md`)
defines the required primitive precisely:

> "reduce once, broadcast within the output workgroup, then store vector
> output" ... 1. A reduction result may feed an elementwise output in the
> same ordinary SINK program. 2. It maps one decode row to one cooperative
> workgroup ... 3. It is selected structurally (one reduced row feeding an
> elementwise consumer), preserves lazy input indexing, and rejects matmul,
> multi-row/prefill, movement views it cannot index, and arbitrary custom
> programs. 4. It first passes an isolated realized-and-lazy topology/profile
> gate, then a single RMSNorm family real-token A/B at d512.

Since that record, the repo built and shipped a **single-recipe instance** of
this primitive (08-05 campaign, `a1a51c349`): the `REDUCE_OUTPUT` semantic
marker, the `emit_reduce_output_rmsnorm` UOp body, the rangeify lowering, and
the callify invocation-input proof. The isolated microgate
(`nv-reduce-output-rmsnorm-microgate-record-20260805.md`) proved it bitwise
and measured **-1.253 us/replay** vs the ordinary two-program RMSNorm
(55.374 -> 54.121 us). The production census, however, admitted **zero**
decode norms, and the one forced GPU attempt hit `Xid 31 MMU fault` because
the admission stripped dependency-bearing movement and treated a bare
callified PARAM as sufficient ownership evidence.

Separately, the M4 resadd landing built and shipped the ownership contract
that route was missing: `ResidualViewRequest` / `TypedViewRequest` /
`_validated_residual_view` / `_DECLARED_TYPED_OUTPUTS`, proven on the
promoted M4 epilogue route.

This scope closes the loop: generalize the existing primitive and rewire its
admission onto the M4-validated typed-view proof, then prove via a production
census that decode norms become admissible. This is the C1 unblock.

## 2. What is generic today vs what is hardcoded

Already generic / proven:

- `Ops.REDUCE_OUTPUT` semantic marker with a fail-closed ordinary fallback
  source (`tinygrad/uop/__init__.py`, `tinygrad/tensor.py`).
- UOp body machinery: REDUCE axis, staged warp XOR-tree
  (`_warp_reduce_sum_staged`), shared-memory publish + barrier, LOOP-restored
  epilogue (`tinygrad/codegen/late/reduce_output.py`).
- Op-generic warp-reduce dispatch ladder:
  `_LADDER = {Ops.ADD: _warp_reduce_sum_staged, Ops.MAX: warp_reduce_max}`
  (`tinygrad/codegen/late/warp_reduce.py`).
- rangeify STORE selector + fail-closed fallback
  (`tinygrad/schedule/rangeify.py:326-391`).
- callify invocation-input slot proof
  (`tinygrad/callify.py:259-330`).
- M4 typed-view ownership proof (`tinygrad/llm/kernel_program.py:327`,
  `_fold_residual_input_views`).

Hardcoded (the genericity gaps):

1. **Emitter shape**: `emit_reduce_output_rmsnorm` accepts only
   `rows == 1`, `dim % 512 == 0`, fixed 16-warp / 32-lane / per-lane
   association. The production census needs three associations:
   `r_16_256` x73 (attention/FFN/output norms, dim 4096),
   `r_2_8_4_4_16` x36 (q-norm), `r_8_16_8` x36 (k-norm) - 145 reductions.
2. **Recipe vocabulary**: `ReduceOutputSpec` carries `eps`, `affine`, and a
   recipe string with one value (`sumsq_rsqrt_affine`). No MAX-reduce
   recipe, no reduce-op field that composes with the existing `_LADDER`.
3. **Admission proof**: `lower_reduce_output_store` accepts only
   `_identity_buffer_view` / `_owned_precompiled_output_after_view` /
   `_proven_invocation_input_view` (bare buffer or one exact invocation
   PARAM). It does not use the M4 typed-view machinery
   (`_validated_residual_view`, `_DECLARED_TYPED_OUTPUTS`), which is what
   proved safe ownership for the same class of callified model graphs.

## 3. The test (all CPU, hermetic first)

### 3.1 Generalize the spec and emitter

- Extend `ReduceOutputSpec` with the fields the generic body needs:
  reduce op (default `Ops.ADD`, composed with `_LADDER`), warp/lane/per-lane
  association derived from the ordinary reduce shape, and the recipe string.
  Keep every existing field name and default so the closed policy record and
  existing unit tests stay valid.
- Generalize `emit_reduce_output_rmsnorm` (or add a sibling
  `emit_reduce_output(spec, ...)`) so the body is derived from the spec:
  warp count and per-lane elements from the association, reduce op from the
  `_LADDER`, epilogue from the recipe. The 08-05 body for the current shape
  must remain byte-identical (legacy body pin).
- Fail closed for any association/recipe the body builder cannot express
  exactly (same `ValueError -> reject` path as today).

### 3.2 Rewire admission onto the M4 typed-view proof

- `lower_reduce_output_store` may accept the production spelling
  `CONTIGUOUS(RESHAPE(MEMORY_SEMANTIC(REDUCE_OUTPUT)))` (the C6 chain) when
  the input view passes the same ownership contract M4 validates: pure
  offset-0 view over a producer with buffer/precompiled-output identity or a
  declared typed output (`_DECLARED_TYPED_OUTPUTS`). Reuse
  `_validated_residual_view`'s structure or `_proven_invocation_input_view`;
  do not reimplement ownership from scratch.
- Keep the strict existing rejections (PERMUTE/SHRINK/EXPAND, arbitrary
  AFTER, bare unproven PARAM, lazy/movement inputs) with the same trace
  reasons so the census can distinguish admission causes.

### 3.3 Hermetic CPU gate (the test itself)

New `test/unit/test_generic_reduce_output.py` (plus any fixtures it needs):

1. For each census association (r_16_256, r_2_8_4_4_16, r_8_16_8), a
   realized fp16 (1, dim) RMSNorm lowers to ONE ordinary CALL named
   `reduce_output_*`, bitwise equal to the ordinary two-program RMSNorm, with
   the expected reduce-op body (REDUCE range, one barrier, LOOP restore).
2. A MAX-reduce recipe (e.g. affine epilogue over a warp-reduced max) lowers
   to one CALL via `_LADDER[Ops.MAX]`, bitwise equal to the ordinary form.
   This proves the reduce op is generic, not RMSNorm-specific.
3. Lazy input (`x+x`) fails closed with no materialization (existing
   behavior preserved).
4. Movement/offset inputs (PERMUTE, SHRINK, EXPAND, arbitrary AFTER, bare
   unproven PARAM) fail closed with the distinct trace reasons.
5. The 08-05 legacy body for the current shape is byte-identical (same
   program name and UOp body digest as before this change).

All hermetic assertions must run on CPU (`DEV=CPU`) - no GPU, no lock.

### 3.4 Production census (CPU, decode graph)

- Run the existing decode DAG census tooling on the redirect-on authority
  with the generalized admission and record, per norms population:
  `admissions > 0`, `programs removed`, and the changed-node census (which
  reduce roles are now eligible, which still reject and why).
- Artifact JSON to `docs/task_workflow/output/`
  (`nv-generic-reduce-output-census-20260809.json`) with the gate verdict,
  admission counts per association, rejection reasons, and the digest of the
  ordinary authority DAG used.

## 4. Success criteria

1. Hermetic gate green (all of 3.3) on CPU.
2. Census admissions > 0 for the norms population (145 reductions eligible
   or a documented subset with a per-shape reason for every rejection).
3. Existing tripwire green: `test_shared_q8_attention_landing.py`,
   `test_reduce_output_rmsnorm.py`, `test_decode_graph_position_invariance.py`,
   M4/M5 sets, and `python3 sz.py` within budget.
4. Legacy byte pins unmoved (pg3 legacy sha `27857cb8ca03`, 08-05
   REDUCE_OUTPUT body).

## 5. HARD STOP

- No GPU arm, no `/tmp/gpu-bench.lock`, no wall bracket, no logits gate on
  device. The NV render path (Xid 31 class) is re-tested only in a later,
  separately scoped arm under lock after this CPU gate passes.
- No policy promotion: `decode-reduce-output-rmsnorm-route-policy.json` stays
  `promoted_targets: []`; no model wiring change; no default flip.
- No M4/M5/Path-3/M3 record changes; no changes to `decode_routes.py`,
  `qk_primitives.py`, or the shared-Q8 promotion record.
- No `--no-verify`; code changes require test changes in the same commit;
  `[nv]` prefix for code, `[docs]` for records.
- Scratch in `/tmp` only (disk is ~99% full); no large artifacts in the repo.

## 6. Deliverables

1. Code: generalized spec + emitter + admission (this scope's sections 3.1,
   3.2) with unit tests in the same commit.
2. Hermetic gate + census artifact + record
   `docs/task_workflow/input/nv-generic-reduce-output-primitive-record-20260809.md`
   documenting verdict, per-association admission counts, rejection reasons,
   and the exact reason this unblocks the 495.330 us norms row (and 163.029
   us flash row) without claiming recovered wall.
3. Commits on `nvidia-bringup-20260731`, pushed.

