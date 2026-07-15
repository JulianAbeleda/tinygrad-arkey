import json

import pytest

from extra.qk.mmq_logical_vocabulary import (
  Axis, BackendCapability, DType, DotOp, EdgePredicate, LogicalMMQDescriptor,
  MMQCandidate, PhysicalMapping, Q4KDecode, Q8DS4Semantics, Stage, Staging,
  SyncScope, Synchronization,
)


def _descriptor(**kwargs):
  axes = tuple(Axis(name, extent, tile=tile) for name, extent, tile in (
    ("m", 512, 16), ("n", 4096, 16), ("k", 4096, 256),
    ("group", "k/256", None), ("activation_block", "k/32", None),
  ))
  defaults = dict(
    axes=axes,
    edge_predicates=(EdgePredicate("m"), EdgePredicate("n"), EdgePredicate("k")),
  )
  defaults.update(kwargs)
  return LogicalMMQDescriptor(**defaults)


def test_bounded_q4k_q8_descriptor_is_logical_and_json_stable():
  descriptor = _descriptor()
  candidate = MMQCandidate(
    descriptor=descriptor,
    mapping=PhysicalMapping(wave_size=32, workgroup_size=64),
    capability=BackendCapability(
      backend="AMD", device="gfx1100", supported_ops=(DotOp.WMMA_I8_I8_I32,),
      wave_sizes=(32,), max_workgroup_size=256,
    ),
  )
  encoded = candidate.canonical_json()
  payload = json.loads(encoded)
  assert payload["descriptor"]["q4k"]["block_elements"] == 256
  assert payload["descriptor"]["q8"]["block_elements"] == 32
  assert payload["descriptor"]["operation"]["name"] == "wmma_i8_i8_i32"
  assert payload["descriptor"]["abi"] == {"output_layout": "tokens_rows"}
  # Logical vocabulary must not smuggle a physical lane/index schedule.
  assert not {"lidx0", "lane", "lane_zero", "wmma_m", "wmma_n", "wmma_k"} & set(encoded)
  assert candidate.rollback_identity == "direct-packed"
  assert candidate.identity().startswith("mmq-")


@pytest.mark.parametrize("bad", [
  dict(axes=tuple(Axis(name, 1) for name in ("m", "n", "k", "group"))),
  dict(edge_predicates=()),
  dict(ownership={"writeback": "many_owners"}),
])
def test_logical_descriptor_rejects_incomplete_or_nonuniform_contract(bad):
  if "ownership" in bad:
    from extra.qk.mmq_logical_vocabulary import Ownership
    with pytest.raises(ValueError, match="one owner"):
      Ownership(**bad["ownership"])
  else:
    with pytest.raises(ValueError):
      _descriptor(**bad)


def test_edge_predicates_cover_each_tiled_output_axis():
  descriptor = _descriptor(edge_predicates=(EdgePredicate("m"), EdgePredicate("n")))
  assert {predicate.axis for predicate in descriptor.edge_predicates} == {"m", "n"}
  assert all(predicate.required and predicate.predicate == "index < extent"
             for predicate in descriptor.edge_predicates)


def test_nonuniform_synchronization_is_rejected():
  with pytest.raises(ValueError, match="must be uniform"):
    Synchronization(scope=SyncScope.WORKGROUP, uniform=False)
