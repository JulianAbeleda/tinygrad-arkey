from dataclasses import dataclass

import pytest

from tinygrad.codegen.opt.gemm_consumer import (DOT2_CONSUMER, WMMA_CONSUMER,
  consumer_adapter_identity, validate_consumer_wait_coverage)
from tinygrad.codegen.opt.compiler_policies import PipelinePolicy, WaitDependency
from tinygrad.codegen.opt.tc import amd_rdna3
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp


@dataclass(frozen=True)
class _Tile:
  role: str = "A"
  dtype: object = dtypes.half
  carrier_width: int = 2
  tile_shape: tuple[int, int, int] = (1, 1, 2)
  slot_count: int = 1
  slot_addressing: str = "sequential"
  layout: str = "lane_pair"


def test_dot2_adapter_accepts_bounded_logical_tile_and_identity():
  DOT2_CONSUMER.validate_tile(_Tile())
  assert consumer_adapter_identity(DOT2_CONSUMER) == "amd.rdna3.dot2.fp16.v1"


def test_dot2_adapter_rejects_unproven_double_buffer_addressing():
  with pytest.raises(ValueError, match="double-buffer"):
    DOT2_CONSUMER.validate_tile(_Tile(slot_count=2, slot_addressing="static"))


def test_dot2_adapter_rejects_layout_only_shape_match():
  with pytest.raises(ValueError, match="layout"):
    DOT2_CONSUMER.validate_tile(_Tile(layout="rdna3_wmma"))


def test_dot2_adapter_reuses_existing_backend_lowering_marker():
  a = UOp(Ops.STACK, dtypes.half.vec(2), (UOp.const(dtypes.half, 1), UOp.const(dtypes.half, 2)))
  b = UOp(Ops.STACK, dtypes.half.vec(2), (UOp.const(dtypes.half, 3), UOp.const(dtypes.half, 4)))
  lowered = DOT2_CONSUMER.lower(a, b)
  assert lowered.op is Ops.CUSTOMI
  assert lowered.dtype == dtypes.float
  assert "__builtin_amdgcn_fdot2" in lowered.arg


def test_dot2_adapter_rejects_non_pair_carriers():
  a = UOp(Ops.STACK, dtypes.half.vec(4), tuple(UOp.const(dtypes.half, x) for x in range(4)))
  b = UOp(Ops.STACK, dtypes.half.vec(2), (UOp.const(dtypes.half, 3), UOp.const(dtypes.half, 4)))
  with pytest.raises(ValueError, match=r"half.vec\(2\)"):
    DOT2_CONSUMER.lower(a, b)


def test_wmma_adapter_keeps_existing_descriptor_and_node_validators():
  WMMA_CONSUMER.validate_descriptor(amd_rdna3[0])
  assert WMMA_CONSUMER.identity == "amd.rdna3.wmma.fp16.v1"


def test_consumer_wait_coverage_is_joined_to_typed_policy():
  policy = PipelinePolicy.register_resident()
  dep = WaitDependency(policy.wait, "global_load", "dot2", "A", producer_stage=0, consumer_stage=1)
  coverage = validate_consumer_wait_coverage(DOT2_CONSUMER, policy, (dep,), (("A", 0, 1),))
  assert coverage.passed and coverage.covered == (("A", 0, 1),)


def test_consumer_wait_coverage_rejects_missing_edges():
  policy = PipelinePolicy.register_resident()
  with pytest.raises(ValueError, match="lacks complete wait coverage"):
    validate_consumer_wait_coverage(DOT2_CONSUMER, policy, (), (("A", 0, 1),))
