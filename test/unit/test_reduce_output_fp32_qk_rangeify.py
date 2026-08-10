"""Rangeify-side admission for the fp32 q/k reduce-output route (Wave 2, piece C).

Structural CPU-only tests: the rangeify selector admits the fp32 q/k marker
through the production PERMUTE-view spelling (``PERMUTE(RESHAPE(precompiled
AFTER))``), the PERMUTE-carrier pass lowers ONE fused body per marker with the
row-aware name, every consumer reads the same dependency-bearing fused output
buffer, and the fp32 route binds ``norm.weight`` directly (no fp16 load-time
materialization, no owned cast).  Every unexpected view stays fail-closed on
the ordinary graph.  The cooperative body itself cannot compile on the CPU
renderer (pre-existing: ``test_native_value_matches_ordinary``), so these
tests are structural and never execute the fused kernel.
"""

from tinygrad import Tensor, dtypes, nn
from tinygrad.uop.ops import Ops


def _norm(dim, dtype=dtypes.float32):
  n = nn.RMSNorm(dim, eps=1e-6)
  n.weight = Tensor.ones(dim, dtype=dtype)
  return n


def _precompiled_producer():
  from tinygrad.function import function
  @function(precompile=True)
  def producer(v): return v + 1
  return producer


def _rope(value, rows, dim, freqs=None):
  from tinygrad.llm.model import apply_rope
  if freqs is None: freqs = Tensor.empty(1, 1, dim, dtype=dtypes.float32, device="CPU")
  return apply_rope(value, freqs)


def _attention(value):
  from tinygrad.function import function
  @function(precompile=True)
  def attention(v): return v * 2
  return attention(value.contiguous())


def _names(out):
  linear, _ = out.linear_with_vars()
  return [x.src[0].arg.name for x in linear.src]


def _qk_graph(rows, promoted=True, producer=None, pre="q"):
  """Production spelling: precompiled q4k GEMV -> PERMUTE -> marker -> rope -> attention."""
  from tinygrad.llm.model import _decode_reduce_output_rmsnorm
  from tinygrad.llm.memory_semantics import runtime_scratch
  producer = _precompiled_producer() if producer is None else producer
  x = Tensor.empty(1, 1, rows * 128, dtype=dtypes.float32, device="CPU")
  q = producer(x).reshape(1, 1, rows, 128).transpose(1, 2)
  norm = _norm(128)
  if promoted:
    q_m = _decode_reduce_output_rmsnorm(norm, q, True)
  else:
    q_m = norm(q)
  q_s = runtime_scratch(q_m)
  roped = _rope(q_s, rows, 128)
  return _attention(roped)


def test_permute_identity_admits_precompiled_output_through_rangeify():
  """The selector admits the fp32 q/k marker whose input is a pure PERMUTE of
  the precompiled q4k GEMV output (rows 32 and 8), lowering the row-aware
  fused bodies into the schedule."""
  from tinygrad.helpers import Context
  for rows, name in ((32, "reduce_output_rmsnorm_32_128"), (8, "reduce_output_rmsnorm_8_128")):
    out = _qk_graph(rows)
    with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
      assert name in _names(out)


def test_permute_carrier_lowers_one_body_per_marker_with_shared_buffer():
  """One fused body per unique marker, and every ordinary-elementwise consumer
  reads the same dependency-bearing fused output buffer."""
  from tinygrad.helpers import Context
  from tinygrad.uop.ops import UOp
  out = _qk_graph(32)
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
    linear, _ = out.linear_with_vars()
  fused = [c for c in linear.src if "reduce_output_rmsnorm" in c.src[0].arg.name]
  assert [c.src[0].arg.name for c in fused] == ["reduce_output_rmsnorm_32_128"]
  fused_call = fused[0]
  out_buf = fused_call.src[1].buf_uop
  assert out_buf.op in (Ops.BUFFER, Ops.SLICE)
  consumers = [c for c in linear.src if c is not fused_call and
               any(a.buf_uop is out_buf for a in c.src[1:])]
  assert consumers, "the rope elementwise consumers must read the fused output buffer"
  # Two rope elementwise kernels (the x1 and x2 halves) read the fused buffer.
  assert len(consumers) >= 2 and all(c.src[0].arg.name == "test" for c in consumers)


def test_two_markers_lower_to_two_distinct_bodies():
  """A graph carrying both q (rows 32) and k (rows 8) markers lowers exactly one
  body per marker with the right name, and each consumer reads its own fused
  output."""
  from tinygrad.function import function
  from tinygrad.helpers import Context
  from tinygrad.llm.model import _decode_reduce_output_rmsnorm
  from tinygrad.llm.memory_semantics import runtime_scratch
  @function(precompile=True)
  def producer(v): return v + 1
  q = producer(Tensor.empty(1, 1, 32 * 128, dtype=dtypes.float32, device="CPU")).reshape(1, 1, 32, 128).transpose(1, 2)
  k = producer(Tensor.empty(1, 1, 8 * 128, dtype=dtypes.float32, device="CPU")).reshape(1, 1, 8, 128).transpose(1, 2)
  q_m = runtime_scratch(_decode_reduce_output_rmsnorm(_norm(128), q, True))
  k_m = runtime_scratch(_decode_reduce_output_rmsnorm(_norm(128), k, True))
  roped_q = _rope(q_m, 32, 128)
  roped_k = _rope(k_m, 8, 128)
  @function(precompile=True)
  def attention(v): return v * 2
  out_q = attention(roped_q.contiguous())
  out_k = attention(roped_k.contiguous())
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
    linear, _ = out_q.linear_with_vars(out_k)
  names = [c.src[0].arg.name for c in linear.src if "reduce_output_rmsnorm" in c.src[0].arg.name]
  assert sorted(names) == ["reduce_output_rmsnorm_32_128", "reduce_output_rmsnorm_8_128"]
  bodies = [c for c in linear.src if "reduce_output_rmsnorm" in c.src[0].arg.name]
  assert len(bodies) == 2
  # Distinct physical output views (memory planning may pack scratch slices into
  # one parent allocation, so compare the exact output views, not the parent).
  assert bodies[0].src[1] is not bodies[1].src[1]
  assert bodies[0].src[1].buf_uop is bodies[1].src[1].buf_uop  # same parent scratch block
  assert bodies[0].src[1].arg != bodies[1].src[1].arg  # disjoint offsets


def test_fp32_route_binds_fp32_weight_directly():
  """The multi-row fp32 q/k marker binds norm.weight (fp32), never the fp16
  load-time materialization, and owns no cast."""
  from tinygrad.llm.model import _decode_reduce_output_rmsnorm
  n = _norm(128)
  fp16_materialized = Tensor.ones(128, dtype=dtypes.float16).contiguous().realize()
  n._decode_reduce_output_weight = fp16_materialized
  q = _precompiled_producer()(Tensor.empty(1, 1, 32 * 128, dtype=dtypes.float32, device="CPU")).reshape(1, 1, 32, 128).transpose(1, 2)
  marked = _decode_reduce_output_rmsnorm(n, q, True)
  assert marked.uop.op is Ops.REDUCE_OUTPUT
  assert marked.dtype is dtypes.float32
  assert marked.uop.src[0].op is not Ops.CAST
  assert marked.uop.src[2] is n.weight.uop
  assert marked.uop.src[2].dtype is dtypes.float32
  assert marked.uop.src[2] is not fp16_materialized.uop


def test_single_row_c6_route_keeps_materialized_identity_weight():
  """The C6 norms (rows 1) keep the existing materialized fp16 identity weight
  behavior; only the multi-row fp32 route switches to norm.weight."""
  from tinygrad.llm.model import _decode_reduce_output_rmsnorm
  n = _norm(4096)
  fp16_materialized = Tensor.ones(4096, dtype=dtypes.float16).contiguous().realize()
  n._decode_reduce_output_weight = fp16_materialized
  x = Tensor.empty(1, 4096, dtype=dtypes.float32, device="CPU")
  marked = _decode_reduce_output_rmsnorm(n, x, True)
  assert marked.uop.op is Ops.REDUCE_OUTPUT
  assert marked.uop.src[2] is fp16_materialized.uop


def test_identity_view_admits_pure_permute_of_precompiled_output():
  """UOp-level: a pure PERMUTE(RESHAPE(...)) over a precompiled-output AFTER is
  an identity view; the original dependency-bearing view is retained."""
  from tinygrad.schedule.rangeify import _identity_buffer_view
  from tinygrad.uop.ops import CallInfo, UOp
  base = UOp.param(0, dtypes.float32, (4096,), "NV")
  out = UOp.new_buffer("NV", 4096, dtypes.float32)
  body_sink = UOp.sink(out.store(UOp.param(1, dtypes.float32, (4096,), "NV")))
  body = UOp(Ops.LINEAR, src=(body_sink.call(out),))
  call = UOp(Ops.CALL, dtypes.void, (body, base, out), CallInfo(name="gemv", precompile=True, precompiled_output_slots=()))
  after = out.after(call)
  view = after.reshape(1, 1, 32, 128).permute((0, 2, 1, 3))
  assert view.shape == (1, 32, 1, 128)
  assert _identity_buffer_view(view) is view
  # A pure permute over a bare PARAM stays rejected (pinned by
  # test_identity_view_rejects_offsets_movements_and_dependencies).
  p = UOp.param(2, dtypes.float32, (4096,), "NV")
  assert _identity_buffer_view(p.reshape(64, 64).permute((1, 0))) is None
  # SHRINK legs, non-precompiled AFTER, and multi-producer PERMUTE stay rejected.
  assert _identity_buffer_view(after.shrink(((0, 2048),))) is None
  nonpre = UOp(Ops.CALL, dtypes.void, (body, base, out), CallInfo(name="gemv", precompile=False))
  assert _identity_buffer_view(out.after(nonpre).reshape(1, 1, 32, 128).permute((0, 2, 1, 3))) is None


def test_permute_carrier_fails_closed_for_non_identity_marker_input():
  """A marker whose input is a PERMUTE of an unproven value (ADD chain) is not
  admitted, so the ordinary graph survives with no fused body."""
  from tinygrad.function import function
  from tinygrad.helpers import Context
  from tinygrad.llm.model import _decode_reduce_output_rmsnorm
  from tinygrad.llm.memory_semantics import runtime_scratch
  @function(precompile=True)
  def producer(v): return v + 1
  x = Tensor.empty(1, 1, 32 * 128, dtype=dtypes.float32, device="CPU")
  moved = (producer(x) + 1).reshape(1, 1, 32, 128).transpose(1, 2)
  marked = _decode_reduce_output_rmsnorm(_norm(128), moved, True)
  assert marked.uop.arg.input_identity_at_marker is False
  out = _attention(_rope(runtime_scratch(marked), 32, 128))
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
    assert "reduce_output_rmsnorm_32_128" not in _names(out)


def test_permute_carrier_fails_closed_for_cast_and_shrink_chains():
  """CAST and SHRINK legs below the marker input never enter the identity walk,
  so the fp32 route stays on the ordinary graph."""
  from tinygrad.function import function
  from tinygrad.helpers import Context
  from tinygrad.llm.model import _decode_reduce_output_rmsnorm
  from tinygrad.llm.memory_semantics import runtime_scratch
  @function(precompile=True)
  def producer(v): return v + 1
  x = Tensor.empty(1, 1, 32 * 256, dtype=dtypes.float32, device="CPU")
  shrink_moved = producer(x).shrink(((0, 1), (0, 1), (0, 32 * 128))).reshape(1, 1, 32, 128).transpose(1, 2)
  assert _decode_reduce_output_rmsnorm(_norm(128), shrink_moved, True).uop.arg.input_identity_at_marker is False
  y = Tensor.empty(1, 1, 32 * 128, dtype=dtypes.float32, device="CPU")
  cast_moved = producer(y).cast(dtypes.float16).reshape(1, 1, 32, 128).transpose(1, 2)
  assert _decode_reduce_output_rmsnorm(_norm(128), cast_moved, True).uop.arg.input_identity_at_marker is False
  for moved in (shrink_moved, cast_moved):
    marked = _decode_reduce_output_rmsnorm(_norm(128), moved, True)
    out = _attention(_rope(runtime_scratch(marked), 32, 128))
    with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
      assert "reduce_output_rmsnorm_32_128" not in _names(out)


def test_unpromoted_marker_keeps_ordinary_graph():
  """With promotion off the marker never exists and the ordinary q/k path stays
  byte-identical (no fused body)."""
  from tinygrad.helpers import Context
  out = _qk_graph(32, promoted=False)
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
    assert "reduce_output_rmsnorm_32_128" not in _names(out)
