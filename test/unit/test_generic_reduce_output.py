"""Generic cooperative reduction-to-output primitive: CPU hermetic capability gate.

Scope: ``docs/task_workflow/input/nv-generic-reduce-output-primitive-scope-20260809.md``
section 3.3.  Every assertion runs on ``DEV=CPU`` - no GPU, no lock.  The gate
locks that the ``REDUCE_OUTPUT`` body is derived entirely from
``ReduceOutputSpec`` (reduce op composed with the warp/lane ``_LADDER``,
warp/lane/per-lane association from the ordinary reduce shape, recipe-driven
epilogue), that the rangeify admission accepts the production
``CONTIGUOUS(RESHAPE(MEMORY_SEMANTIC(REDUCE_OUTPUT)))`` spelling through the
M4-style typed-view proof, and that every shape/recipe the builder cannot
express exactly fails closed with the existing trace reasons.
"""
import hashlib
import os
import numpy as np

from tinygrad.helpers import DEV

DEV.value = "CPU"

from tinygrad import Tensor, dtypes, nn
from tinygrad.uop.ops import Ops, UOp, ReduceOutputSpec, AxisType

# census associations: (ordinary reduce shape, dim, warps, per_lane)
ASSOCIATIONS = {
  "r_16_256": ((16, 256), 4096, 16, 8),
  "r_2_8_4_4_16": ((2, 8, 4, 4, 16), 4096, 2, 64),
  "r_8_16_8": ((8, 16, 8), 1024, 8, 4),
}

# The 08-10 body pin: the correction from the 08-05 strided shuffle-tree
# ladder to the ordinary serial-contiguous association.  The 08-05 body was
# NOT bitwise-equal to the ordinary r_16_256 reduce for fp32 x + fp16 w
# (1-ulp fp32 scale differences flipping downstream fp16 values at rounding
# boundaries), which is exactly the NV exact-logits gate failure.  The 08-10
# body mirrors the ordinary kernel's serial 256-contiguous per-thread chain
# and serial 16-partial combine, so the fused output is bitwise equal for
# every dtype mix (see test_native_value_matches_ordinary).
LEGACY_BODY_DIGEST = "23264243d010bc91916ec4ed071a42c3e3ee4004d697b1f215d5482c7844afc8"

# Production per-site spellings from the fp32 q/k route and the FFN-down C6
# route: (rows, dim) -> (warps, per_lane) as derived by
# Tensor._semantic_reduce_output_rmsnorm.
PER_SITE_SPELLINGS = {(32, 128): (32, 4), (8, 128): (8, 4), (1, 4096): (16, 8)}

# sha256(repr(body)) pins for the multi-row q/k bodies.  Stage-3 P1 reworked
# the multi-row launch to the per-row-grid geometry (grid = rows, block = 32
# lanes, GLOBAL row range), so these digests now pin that geometry along with
# the ordinary r_8_16_8 / r_2_8_4_4_16 partial-chain association (not a plain
# per-row serial chain).  The bitwise association itself stays pinned by the
# per-row (P, S, t_stride, s_stride) association assertions in
# test/unit/test_reduce_output_rmsnorm.py, not by this repr digest.
MULTI_ROW_BODY_DIGESTS = {
  (32, 128): "5c50bae49fece748112aef9db971bc75652d65d9d440a7c786ca38a50e74d575",
  (8, 128): "50acf004de4594050a749dee665901a8ad620f0a36cbf0c04a259604f2bd09c2",
}


def _rng():
  return np.random.default_rng(20260809)


def _marked(dim, assoc, x, w, eps=1e-6, reduce_op=Ops.ADD, recipe="sumsq_rsqrt_affine"):
  """Build a REDUCE_OUTPUT marker with an explicit warp/lane/per-lane spec."""
  norm = nn.RMSNorm(dim, eps=eps)
  norm.weight = w
  out = norm(x)
  warps, _, per_lane = assoc
  spec = ReduceOutputSpec(1, dim, eps, out.dtype, input_identity_at_marker=True,
                          reduce_op=reduce_op, recipe=recipe, warps=warps, lanes=32, per_lane=per_lane)
  return Tensor(UOp(Ops.REDUCE_OUTPUT, out.dtype, (out.uop, x.uop, w.uop), spec), device=out.device)


def _names(marked) -> list[str]:
  linear, _ = marked.linear_with_vars()
  return [c.src[0].arg.name for c in linear.src]


def _body_structure(spec):
  from tinygrad.codegen.late.reduce_output import emit_reduce_output
  out, x, w = (UOp.placeholder((spec.rows * spec.dim,), dtypes.float16, i) for i in range(3))
  body = emit_reduce_output(spec, dtypes.float16, dtypes.float16)(out, x, w)
  topo = body.toposort()
  assert body.arg.name == f"reduce_output_{'rmsnorm' if spec.recipe == 'sumsq_rsqrt_affine' else 'max'}_1_{spec.dim}"
  assert sum(u.op is Ops.BARRIER for u in topo) == 1
  assert any(u.op is Ops.RANGE and u.arg == (2, AxisType.REDUCE) for u in topo)
  assert any(u.op is Ops.RANGE and u.arg == (2, AxisType.LOOP) for u in topo)


def test_each_census_association_lowers_to_one_call_and_matches_ordinary():
  rng = _rng()
  # CPU buffer pooling reuses the previous iteration's physical output buffer when
  # two marker flows produce identical dim-4096 graphs, so the second read would
  # see stale data.  Pin every iteration's marked tensor (its realized buffer) for
  # the duration of the loop; the fused-body bitwise proof stays on the NV tripwire
  # (test_native_value_matches_ordinary), while CPU honestly executes the fallback.
  pinned: list = []
  for name, (shape, dim, warps, per_lane) in ASSOCIATIONS.items():
    x = Tensor(rng.normal(0, .2, (1, dim)).astype(np.float16)).realize()
    w = Tensor(rng.normal(1, .05, (dim,)).astype(np.float16)).realize()
    norm = nn.RMSNorm(dim, eps=1e-6)
    norm.weight = w
    ref = norm(x).numpy()
    marked = _marked(dim, (warps, 32, per_lane), x, w)
    names = _names(marked)
    assert names == [f"reduce_output_rmsnorm_1_{dim}"], f"{name}: {names}"
    np.testing.assert_array_equal(marked.numpy(), ref)
    pinned.append(marked)
    _body_structure(ReduceOutputSpec(1, dim, 1e-6, dtypes.float16, warps=warps, lanes=32, per_lane=per_lane))


def test_max_recipe_lowers_via_ladder_and_matches_ordinary():
  rng = _rng()
  dim = 4096
  x = Tensor(rng.normal(0, .2, (1, dim)).astype(np.float16)).realize()
  w = Tensor(rng.normal(1, .05, (dim,)).astype(np.float16)).realize()
  eps = 1e-6
  scale = (x.cast(dtypes.float32).abs().max(axis=-1, keepdim=True) + eps).reciprocal()
  ref = ((x.cast(dtypes.float32) * scale).cast(dtypes.float16) * w.cast(dtypes.float16)).cast(dtypes.float16)
  ref = ref.numpy()
  marked = _marked(dim, (16, 32, 8), x, w, reduce_op=Ops.MAX, recipe="max_affine")
  names = _names(marked)
  assert names == ["reduce_output_max_1_4096"], names
  np.testing.assert_array_equal(marked.numpy(), ref)
  spec = ReduceOutputSpec(1, dim, eps, dtypes.float16, reduce_op=Ops.MAX, recipe="max_affine")
  _body_structure(spec)
  # The MAX recipe must route through the ladder's MAX entry, not the ADD one.
  from tinygrad.codegen.late.reduce_output import emit_reduce_output
  out, xx, ww = (UOp.placeholder((dim,), dtypes.float16, i) for i in range(3))
  body = emit_reduce_output(spec, dtypes.float16, dtypes.float16)(out, xx, ww)
  assert any(u.op is Ops.MAX for u in body.toposort())


def test_lazy_input_fails_closed_without_materialization():
  rng = _rng()
  dim = 4096
  x = Tensor(rng.normal(0, .2, (1, dim)).astype(np.float16)).realize()
  w = Tensor(rng.normal(1, .05, (dim,)).astype(np.float16)).realize()
  norm = nn.RMSNorm(dim, eps=1e-6)
  norm.weight = w
  lazy = x + x
  marked = lazy._semantic_reduce_output_rmsnorm(norm(lazy), lazy, w, norm.eps)
  assert marked.uop.op is Ops.REDUCE_OUTPUT
  assert marked.uop.arg.input_identity_at_marker is False
  # The marker is not a concrete buffer at creation, so no buffer identity exists.
  # This must be captured before _names(): linear_with_vars rewrites the marker to
  # its fallback source, which is an ordinary (buffer-backed) graph.
  assert not marked.uop.has_buffer_identity()
  assert "reduce_output" not in " ".join(_names(marked))


def test_movement_and_unproven_inputs_fail_closed_with_distinct_trace_reasons():
  from tinygrad.helpers import Context, SCACHE
  from tinygrad.llm.reduce_output_trace import REDUCE_OUTPUT_TRACE, reset_reduce_output_trace, reduce_output_trace_snapshot
  rng = _rng()
  dim = 4096
  x = Tensor(rng.normal(0, .2, (1, dim)).astype(np.float16)).realize()
  w = Tensor(rng.normal(1, .05, (dim,)).astype(np.float16)).realize()
  norm = nn.RMSNorm(dim, eps=1e-6)
  norm.weight = w
  out = norm(x)
  base = x.uop

  def marker_for(view_uop, *, out_u=out, w_u=w, **flags):
    spec = ReduceOutputSpec(1, dim, 1e-6, out_u.dtype, **flags)
    return Tensor(UOp(Ops.REDUCE_OUTPUT, out_u.dtype, (out_u.uop, view_uop, w_u.uop), spec), device=out_u.device)

  # PERMUTE / SHRINK / EXPAND / arbitrary AFTER: no durable ownership at marker
  # creation, so the marker is not even eligible (marker_not_eligible).
  permuted = marker_for(base.reshape(64, 64).permute((1, 0)).reshape(1, dim))
  shrunk = marker_for(base.shrink(((0, 1), (0, dim))))
  expanded = marker_for(base.reshape(1, 1, dim).expand(1, 1, dim))
  arbitrary_after = marker_for(base.after(UOp(Ops.NOOP)))
  for marked in (permuted, shrunk, expanded, arbitrary_after):
    assert "reduce_output" not in " ".join(_names(marked))

  reset_reduce_output_trace()
  with Context(REDUCE_OUTPUT_TRACE=1, SCACHE=0):
    # Rebuild the whole flow for the trace pass: the first _names() above
    # rewrote the shared out/x/w tensor uops to their fallback sources via
    # _apply_map_to_tensors, so reusing them would skip the selector entirely.
    # SCACHE must be off too: the first pass already populated schedule_cache
    # with this graph's normalized key, and a cache hit skips rangeify (and
    # with it the selector trace) entirely.
    x2 = Tensor(rng.normal(0, .2, (1, dim)).astype(np.float16)).realize()
    w2 = Tensor(rng.normal(1, .05, (dim,)).astype(np.float16)).realize()
    norm2 = nn.RMSNorm(dim, eps=1e-6)
    norm2.weight = w2
    out2 = norm2(x2)
    base2 = x2.uop
    for marked in (marker_for(base2.reshape(64, 64).permute((1, 0)).reshape(1, dim), out_u=out2, w_u=w2),
                   marker_for(base2.shrink(((0, 1), (0, dim))), out_u=out2, w_u=w2),
                   marker_for(base2.reshape(1, 1, dim).expand(1, 1, dim), out_u=out2, w_u=w2),
                   marker_for(base2.after(UOp(Ops.NOOP)), out_u=out2, w_u=w2)):
      _names(marked)
  reasons = reduce_output_trace_snapshot()["selector"]
  assert reasons.get("marker_not_eligible", 0) >= 4

  # Bare unproven PARAM with an owned-contiguous hint: eligible but the durable
  # proofs all fail, so the rejection is the distinct input_proof_missing.
  bare = marker_for(UOp.param(7, dtypes.float16, (1, dim), "CPU"), owned_contiguous_candidate=True)
  assert "reduce_output" not in " ".join(_names(bare))
  reset_reduce_output_trace()
  with Context(REDUCE_OUTPUT_TRACE=1, SCACHE=0):
    x3 = Tensor(rng.normal(0, .2, (1, dim)).astype(np.float16)).realize()
    w3 = Tensor(rng.normal(1, .05, (dim,)).astype(np.float16)).realize()
    norm3 = nn.RMSNorm(dim, eps=1e-6)
    norm3.weight = w3
    out3 = norm3(x3)
    _names(marker_for(UOp.param(7, dtypes.float16, (1, dim), "CPU"), out_u=out3, w_u=w3,
                      owned_contiguous_candidate=True))
  bare_reasons = reduce_output_trace_snapshot()["selector"]
  assert bare_reasons.get("input_proof_missing", 0) == 1
  assert "marker_not_eligible" not in bare_reasons


def test_legacy_body_pin_is_unchanged():
  from tinygrad.codegen.late.reduce_output import emit_reduce_output, emit_reduce_output_rmsnorm
  spec = ReduceOutputSpec(1, 4096, 1e-6, dtypes.float16)
  out, x, w = (UOp.placeholder((4096,), dtypes.float16, i) for i in range(3))
  body = emit_reduce_output(spec, dtypes.float16, dtypes.float16)(out, x, w)
  legacy = emit_reduce_output_rmsnorm(spec, dtypes.float16, dtypes.float16)(out, x, w)
  assert body == legacy
  assert body.arg.name == "reduce_output_rmsnorm_1_4096"
  assert hashlib.sha256(repr(body).encode()).hexdigest() == LEGACY_BODY_DIGEST


def test_semantic_marker_derives_legacy_association_for_dim_4096():
  rng = _rng()
  dim = 4096
  x = Tensor(rng.normal(0, .2, (1, dim)).astype(np.float16)).realize()
  w = Tensor(rng.normal(1, .05, (dim,)).astype(np.float16)).realize()
  norm = nn.RMSNorm(dim, eps=1e-6)
  norm.weight = w
  marked = norm(x)._semantic_reduce_output_rmsnorm(x, norm(x), w, norm.eps)
  assert marked.uop.arg.warps == 16 and marked.uop.arg.lanes == 32 and marked.uop.arg.per_lane == 8
  assert _names(marked) == ["reduce_output_rmsnorm_1_4096"]


def test_c6_chain_store_spelling_lowers_and_keeps_owner():
  from tinygrad.llm.memory_semantics import runtime_scratch
  from tinygrad.schedule.rangeify import pm_reduce_output_store
  from tinygrad.uop.ops import graph_rewrite
  x = UOp.param(0, dtypes.float16, (1, 4096), "CPU")
  w = UOp.param(1, dtypes.float16, (4096,), "CPU")
  out = UOp.new_buffer("CPU", 4096, dtypes.float16).reshape(1, 4096)
  marker = UOp(Ops.REDUCE_OUTPUT, dtypes.float16, (out, x, w),
                ReduceOutputSpec(1, 4096, 1e-6, dtypes.float16, input_identity_at_marker=True))
  chain = runtime_scratch(marker).reshape(4096).contiguous()
  target = UOp.new_buffer("CPU", 4096, dtypes.float16)
  sink = graph_rewrite(UOp.sink(target.store(chain)), pm_reduce_output_store, bottom_up=False, name="test c6")
  call = sink.src[0]
  assert call.op is Ops.CALL
  assert call.src[0].arg.name == "reduce_output_rmsnorm_1_4096"
  assert call.src[0].arg.memory_semantic_slots == ((0, chain.src[0].src[0].arg),)


def test_c6_chain_rejects_movement_and_bare_inputs():
  from tinygrad.llm.memory_semantics import runtime_scratch
  from tinygrad.schedule.rangeify import lower_reduce_output_store, pm_reduce_output_store
  from tinygrad.uop.ops import graph_rewrite
  x = UOp.param(0, dtypes.float16, (1, 4096), "CPU")
  w = UOp.param(1, dtypes.float16, (4096,), "CPU")
  out = UOp.new_buffer("CPU", 4096, dtypes.float16).reshape(1, 4096)
  marker = UOp(Ops.REDUCE_OUTPUT, dtypes.float16, (out, x, w),
                ReduceOutputSpec(1, 4096, 1e-6, dtypes.float16, input_identity_at_marker=True))
  target = UOp.new_buffer("CPU", 4096, dtypes.float16)
  moved = runtime_scratch(marker).reshape(64, 64).permute((1, 0)).reshape(4096).contiguous()
  moved_sink = graph_rewrite(UOp.sink(target.store(moved)), pm_reduce_output_store, bottom_up=False, name="test c6 moved")
  assert moved_sink.src[0].op is not Ops.CALL
  # A bare MS(marker) direct carrier still lowers (the legacy spelling).
  direct = runtime_scratch(marker)
  assert lower_reduce_output_store(target.store(direct), direct) is not None


def test_c6_call_input_fused_body_never_aliases_its_input():
  """The C6 CALL-input fusion must write a FRESH output buffer.

  The carrier is a lazy materialization, so ``arg.buf_uop`` resolves down to
  the marker's input base; reusing it as the fused body's output would write
  the norm in place over the input and corrupt the input's other consumers
  (the decode block residual).  Regression for the NV wall bracket's
  all-NaN exact-logits gate failure under
  ``CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT``.
  """
  from tinygrad.llm.memory_semantics import runtime_scratch
  from tinygrad.schedule.rangeify import pm_reduce_output_store
  from tinygrad.uop.ops import graph_rewrite
  x = UOp.param(0, dtypes.float16, (1, 4096), "CPU")
  w = UOp.param(1, dtypes.float16, (4096,), "CPU")
  out = UOp.new_buffer("CPU", 4096, dtypes.float16).reshape(1, 4096)
  marker = UOp(Ops.REDUCE_OUTPUT, dtypes.float16, (out, x, w),
                ReduceOutputSpec(1, 4096, 1e-6, dtypes.float16, owned_contiguous_candidate=True,
                                 invocation_input_slot=0))
  chain = runtime_scratch(marker).reshape(4096).contiguous()
  # A consumer CALL whose input is the production C6 chain.
  consumer = UOp(Ops.CALL, dtypes.void, (UOp.sink(chain), chain))
  sink = graph_rewrite(UOp.sink(consumer), pm_reduce_output_store, bottom_up=False, name="test c6 call input")
  rewritten = sink.src[0]
  assert rewritten.op is Ops.CALL
  # The consumer's first argument is the fresh output AFTER the fused body.
  replaced = rewritten.src[1]
  assert replaced.op is Ops.AFTER and len(replaced.src) == 2
  out_buf, fused = replaced.src
  assert out_buf.op is Ops.BUFFER and out_buf.numel() == 4096
  assert fused.op in (Ops.FUNCTION, Ops.CALL)
  assert fused.src[0].arg.name == "reduce_output_rmsnorm_1_4096"
  # The fused body's output slot is the fresh buffer; its input slot is the
  # original input param.  They must never alias.
  assert fused.src[1] is out_buf
  assert fused.src[2] is x
  assert out_buf is not x


def test_c6_call_input_coalesces_one_body_per_marker_across_consumers():
  """The production decode graph feeds one marked norm to several consumer
  CALLs (q/k/v projections, FFN gate/up/down).  The per-argument rule emitted
  one fused body plus one output buffer per consumer (the 54-vs-18 census
  multiplicity); the graph-level pass must emit ONE body and ONE fresh output
  buffer per unique marker, with every consumer reading the same AFTER."""
  from tinygrad.llm.memory_semantics import runtime_scratch
  from tinygrad.schedule.rangeify import coalesce_c6_call_inputs
  x = UOp.param(0, dtypes.float16, (1, 4096), "CPU")
  w = UOp.param(1, dtypes.float16, (4096,), "CPU")
  out = UOp.new_buffer("CPU", 4096, dtypes.float16).reshape(1, 4096)
  marker = UOp(Ops.REDUCE_OUTPUT, dtypes.float16, (out, x, w),
                ReduceOutputSpec(1, 4096, 1e-6, dtypes.float16, owned_contiguous_candidate=True,
                                 invocation_input_slot=0))
  chain = runtime_scratch(marker).reshape(4096).contiguous()
  calls = tuple(UOp(Ops.CALL, dtypes.void, (UOp.sink(chain), chain), arg=None) for _ in range(3))
  sink = coalesce_c6_call_inputs(UOp.sink(*calls))
  assert sink is not None
  bodies, buffers = set(), set()
  for consumer in sink.src:
    assert consumer.op is Ops.CALL
    replaced = consumer.src[1]
    assert replaced.op is Ops.AFTER and len(replaced.src) == 2
    out_buf, fused = replaced.src
    assert fused.src[0].arg.name == "reduce_output_rmsnorm_1_4096"
    bodies.add(fused); buffers.add(out_buf)
  assert len(bodies) == 1 and len(buffers) == 1


def test_m4_style_view_proof_admits_declared_typed_output():
  from tinygrad.llm.kernel_program import DeclaredTypedOutput, TypedLayout, _DECLARED_TYPED_OUTPUTS
  from tinygrad.schedule.rangeify import _reduce_output_m4_input_view
  x = UOp.param(0, dtypes.float16, (1, 4096), "CPU")
  after = x.after(UOp(Ops.NOOP))
  _DECLARED_TYPED_OUTPUTS[after] = DeclaredTypedOutput(TypedLayout(dtypes.float16, (4096,)), True)
  try:
    view = after.reshape(1, 4096).contiguous()
    assert _reduce_output_m4_input_view(view) is view
    assert _reduce_output_m4_input_view(after) is None  # bare base: durable proofs own it
  finally:
    _DECLARED_TYPED_OUTPUTS.pop(after, None)


def test_association_derivation_matches_ordinary_shapes():
  from tinygrad.codegen.late.reduce_output import reduce_output_association
  for name, (shape, dim, warps, per_lane) in ASSOCIATIONS.items():
    got = reduce_output_association(shape)
    assert got == (warps, 32, per_lane), f"{name}: {got}"


def test_per_site_admission_spelling_lowers_to_one_call_body_free():
  """P1 site admission: every production reduce_output_rmsnorm shape admits
  through the generic primitive with its production spelling and lowers to
  exactly one fused-body CALL, with no ``*_weight_store``-style program in
  the lowered graph (the body-free 1:1 swap contract, scope section 5 gate
  3).  The ordinary norm renders two programs (reduce + epilogue); the marked
  norm renders exactly one."""
  rng = _rng()
  for (rows, dim), (warps, per_lane) in PER_SITE_SPELLINGS.items():
    x = Tensor(rng.normal(0, .2, (rows, dim)).astype(np.float16)).realize()
    w = Tensor(rng.normal(1, .05, (dim,)).astype(np.float16)).realize()
    norm = nn.RMSNorm(dim, eps=1e-6)
    norm.weight = w
    out = norm(x)
    marked = out._semantic_reduce_output_rmsnorm(x, out, w, norm.eps)
    spec = marked.uop.arg
    assert (spec.warps, spec.per_lane) == (warps, per_lane), f"{(rows, dim)}: {(spec.warps, spec.per_lane)}"
    names = _names(marked)
    assert names == [f"reduce_output_rmsnorm_{rows}_{dim}"], f"{(rows, dim)}: {names}"
    assert not any("weight_store" in name for name in names), f"{(rows, dim)}: {names}"
    # The ordinary arm of the same site renders two programs (reduce + epilogue).
    ordinary = _names(norm(x))
    assert len(ordinary) == 2, f"{(rows, dim)} ordinary: {ordinary}"


def test_per_site_bodies_compile_to_one_cpu_program():
  """The fused bodies must lower through the CPU renderer to exactly one
  PROGRAM with the production name.  Regression for the CPU thread-mode
  gpudims path: the cooperative body has only LOCAL/WARP/LOOP ranges (no
  global shape), and the split LOCAL ranges must serialize as loops instead
  of indexing a one-element core_id list."""
  from tinygrad.codegen import to_program
  from tinygrad.codegen.late.reduce_output import emit_reduce_output
  from tinygrad.helpers import Target
  from tinygrad.renderer.cstyle import ClangRenderer
  ren = ClangRenderer(Target.parse("CPU:CLANG:x86_64,znver2"))
  for (rows, dim), (warps, per_lane) in PER_SITE_SPELLINGS.items():
    spec = ReduceOutputSpec(rows, dim, 1e-6, dtypes.float16, warps=warps, lanes=32, per_lane=per_lane)
    out, x, w = (UOp.placeholder((rows * dim,), dtypes.float16, i) for i in range(3))
    body = emit_reduce_output(spec, dtypes.float16, dtypes.float16)(out, x, w)
    program = to_program(body, ren)
    assert program.op is Ops.PROGRAM, f"{(rows, dim)}: {program.op}"
    assert program.arg.name == f"reduce_output_rmsnorm_{rows}_{dim}", f"{(rows, dim)}: {program.arg.name}"
    assert any(u.op is Ops.SOURCE for u in program.toposort()), f"{(rows, dim)}: no source"


def test_multi_row_recipe_digests_are_pinned():
  """The multi-row q/k bodies pin the per-row-grid geometry (grid = rows,
  block = 32 lanes) with the NV ordinary partial-chain association
  byte-identical (r_8_16_8 / r_2_8_4_4_16 tiling).  A drift in the emitted
  geometry or association breaks these pins exactly like LEGACY_BODY_DIGEST
  breaks for the single-row body."""
  from tinygrad.codegen.late.reduce_output import emit_reduce_output
  for (rows, dim), digest in MULTI_ROW_BODY_DIGESTS.items():
    spec = ReduceOutputSpec(rows, dim, 1e-6, dtypes.float16, warps=rows, lanes=32, per_lane=dim // 32)
    out, x, w = (UOp.placeholder((rows * dim,), dtypes.float16, i) for i in range(3))
    body = emit_reduce_output(spec, dtypes.float16, dtypes.float16)(out, x, w)
    assert body.arg.name == f"reduce_output_rmsnorm_{rows}_{dim}"
    assert hashlib.sha256(repr(body).encode()).hexdigest() == digest, f"{(rows, dim)} body digest moved"


def test_multi_row_geometry_pin_per_row_grid():
  """Stage-3 P1 geometry pin: each multi-row body is one block per row -- the
  row index is a GLOBAL range with extent == spec.rows (grid = rows), the
  block is exactly 32 LOCAL lanes, exactly one barrier, and the shared-memory
  partial slots are P (the block owns its row, so rows*P is gone)."""
  from tinygrad.codegen.late.reduce_output import emit_reduce_output, _NV_MULTI_ROW_ASSOC
  for (rows, dim), (warps, per_lane) in PER_SITE_SPELLINGS.items():
    if rows == 1: continue
    P = _NV_MULTI_ROW_ASSOC[(rows, dim)][0]
    spec = ReduceOutputSpec(rows, dim, 1e-6, dtypes.float16, warps=rows, lanes=32, per_lane=per_lane)
    out, x, w = (UOp.placeholder((rows * dim,), dtypes.float16, i) for i in range(3))
    body = emit_reduce_output(spec, dtypes.float16, dtypes.float16)(out, x, w)
    topo = body.toposort()
    global_rows = [u for u in topo if u.op is Ops.RANGE and u.arg == (0, AxisType.GLOBAL)]
    assert len(global_rows) == 1, f"{(rows, dim)} must have exactly one GLOBAL row range"
    assert global_rows[0].src[0].arg == rows, f"{(rows, dim)} GLOBAL extent must equal rows"
    local_lanes = [u for u in topo if u.op is Ops.RANGE and u.arg == (0, AxisType.LOCAL)]
    assert len(local_lanes) == 1 and local_lanes[0].src[0].arg == 32, f"{(rows, dim)} block must be 32 lanes"
    assert sum(u.op is Ops.BARRIER for u in topo) == 1, f"{(rows, dim)} must have exactly one barrier"
    smem_after = [u for u in topo if u.op is Ops.AFTER and len(u.src) == 2 and u.src[0].op is Ops.DEFINE_LOCAL]
    assert len(smem_after) == 1 and smem_after[0].src[0].shape == (P,), \
      f"{(rows, dim)} shared-memory slots must be P, got {smem_after[0].src[0].shape}"


def test_lazy_weight_marker_is_not_body_free():
  """A marker whose weight has no durable identity must not be admitted
  body-free: the lazy weight materializes as an extra program instead of
  being absorbed.  Production never hits this path because the route helpers
  bind a load-time identity weight (_decode_reduce_output_weight), which is
  what keeps the census free of *_weight_store additions; this test locks the
  fail-closed side so a spelling change cannot silently reintroduce the
  per-body materialization (the 08-09 net +72 lesson)."""
  rng = _rng()
  dim = 4096
  x = Tensor(rng.normal(0, .2, (1, dim)).astype(np.float16)).realize()
  lazy = Tensor(rng.normal(1, .05, (dim,)).astype(np.float32)).realize().cast(dtypes.float16)
  assert not lazy.uop.has_buffer_identity()
  norm = nn.RMSNorm(dim, eps=1e-6)
  norm.weight = lazy
  out = norm(x)
  marked = out._semantic_reduce_output_rmsnorm(x, out, lazy, norm.eps)
  assert marked.uop.op is Ops.REDUCE_OUTPUT
  # The identity-weight production spelling is exactly one fused body.
  w2 = lazy.contiguous().realize()
  norm2 = nn.RMSNorm(dim, eps=1e-6)
  norm2.weight = w2
  out2 = norm2(x)
  identity_marked = out2._semantic_reduce_output_rmsnorm(x, out2, w2, norm2.eps)
  identity_names = _names(identity_marked)
  assert identity_names == ["reduce_output_rmsnorm_1_4096"], identity_names
  assert not any("weight_store" in name for name in identity_names)
  # The lazy-weight arm renders the fused body PLUS the weight materialization
  # (not body-free), so a census diff would show a non-zero materialization.
  lazy_names = _names(marked)
  assert any("reduce_output_rmsnorm" in name for name in lazy_names)
  assert len(lazy_names) > len(identity_names), lazy_names
