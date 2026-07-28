from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp, graph_rewrite

from tinygrad.codegen.late.fdot2 import line_lower_fdot2, lower_fdot2_add, pm_fdot2


def _half2(name: str) -> UOp:
  seed = (sum(ord(c) for c in name) % 17) + 1
  return UOp(Ops.STACK, dtypes.half.vec(2),
             (UOp.const(dtypes.half, float(seed)), UOp.const(dtypes.half, float(seed + 1))))


def _lane(v: UOp, i: int, *, index_dtype=dtypes.int, lane_dtype=dtypes.half) -> UOp:
  return UOp(Ops.INDEX, lane_dtype, (v, UOp.const(index_dtype, i)))


def _term(a: UOp, b: UOp, i: int, **kwargs) -> UOp:
  return (_lane(a, i, **kwargs) * _lane(b, i, **kwargs)).cast(dtypes.float)


def _fdot_nodes(u: UOp) -> list[UOp]:
  return [x for x in u.toposort() if x.op is Ops.CUSTOMI and "fdot2" in str(x.arg)]


def test_fdot2_pair_and_accumulator_contract():
  a, b = _half2("a"), _half2("b")
  pair = _term(a, b, 0) + _term(a, b, 1)
  assert _fdot_nodes(graph_rewrite(pair, pm_fdot2))

  acc = UOp.const(dtypes.float, 7.0)
  for expr in (acc + pair, pair + acc):
    lowered = lower_fdot2_add(expr)
    assert lowered is not None
    assert lowered.dtype == dtypes.float
    assert lowered.src[0] is acc
    assert lowered.src[1:] == (a, b)
    assert lowered.arg == "__builtin_amdgcn_fdot2({1}, {2}, {0}, false)"


def test_fdot2_rejects_mismatched_sources_and_lanes():
  a, b, c = _half2("a"), _half2("b"), _half2("c")
  assert not _fdot_nodes(graph_rewrite(_term(a, b, 0) + _term(a, c, 1), pm_fdot2))
  assert not _fdot_nodes(graph_rewrite(_term(a, b, 0) + _term(a, b, 0), pm_fdot2))
  assert not _fdot_nodes(graph_rewrite(_term(a, b, 0) + _term(a, b, 2), pm_fdot2))


def test_fdot2_rejects_wrong_index_and_lane_dtype():
  a, b = _half2("a"), _half2("b")
  assert not _fdot_nodes(graph_rewrite(
    _term(a, b, 0, index_dtype=dtypes.float) + _term(a, b, 1), pm_fdot2))
  assert not _fdot_nodes(graph_rewrite(
    _term(a, b, 0, lane_dtype=dtypes.float) + _term(a, b, 1), pm_fdot2))


def test_line_lower_fdot2_replaces_downstream_uses_in_topological_order():
  a, b = _half2("a"), _half2("b")
  pair = _term(a, b, 0) + _term(a, b, 1)
  consumer = pair + UOp.const(dtypes.float, 1.0)
  lowered = line_lower_fdot2(list(consumer.toposort()))
  fdots = [u for u in lowered if u.op is Ops.CUSTOMI and "fdot2" in str(u.arg)]
  assert len(fdots) == 1
  fdot = fdots[0]
  rewritten_consumer = lowered[-1]
  assert rewritten_consumer.op is Ops.ADD
  assert fdot in rewritten_consumer.src
  positions = {u: i for i, u in enumerate(lowered)}
  assert all(positions[s] < positions[rewritten_consumer] for s in rewritten_consumer.src)


def test_fdot2_lowering_declines_non_add_nodes():
  a, b = _half2("a"), _half2("b")
  assert lower_fdot2_add(_term(a, b, 0)) is None


def test_fdot2_core_owns_all_public_hooks():
  import importlib.util
  assert importlib.util.find_spec("tinygrad.codegen.experimental") is None
  assert importlib.util.find_spec("extra.llm_research.fdot2_lowering") is None
