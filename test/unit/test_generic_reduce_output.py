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

LEGACY_BODY_DIGEST = "c82e25f5a4c7cb7758dc31fb8dd5bee72ee01bcff1eb08e26c030415b7a89337"


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
