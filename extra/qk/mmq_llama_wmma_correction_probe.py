"""One-wave diagnostic for the llama MMQ WMMA/correction consumer.

The full five-buffer kernel has three separable responsibilities:

* cooperative global -> LDS production,
* signed-int8 WMMA plus Q4/Q8 metadata correction, and
* lane-owned row-major writeback.

This probe deliberately removes the first responsibility.  The host supplies
one already-arranged 16-byte A and B fragment per lane, plus the exact fp16
``dm`` and ``ds`` sidecars consumed by the recurrence.  It therefore provides
a small GPU diagnostic for the consumer path without making any claim about
the cooperative Q4/Q8 producers.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.codegen.opt.kernel_lds import validate_precontract_wmma_abi
from tinygrad.helpers import Target
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp

from extra.qk.kernel_lds import rdna3_wmma_output_coord
from extra.qk.mmq_llama_candidate_plan import llama_mmq_candidate_plan


SCHEMA = "tinygrad.mmq_llama_wmma_correction_probe.v1"
LOCAL_SIZE = (32, 1, 1)
OUTPUT_SHAPE = (16, 16)
FRAGMENT_SHAPE = (32, 16)
DM_SHAPE = (8, 2)
DS_SHAPE = (2,)


@dataclass(frozen=True)
class WMMAConsumerProbeFixture:
  a_fragments: np.ndarray
  b_fragments: np.ndarray
  dm: np.ndarray
  ds: np.ndarray
  reference: np.ndarray

  def __post_init__(self) -> None:
    expected = (
      ("a_fragments", self.a_fragments, np.dtype(np.int8), FRAGMENT_SHAPE),
      ("b_fragments", self.b_fragments, np.dtype(np.int8), FRAGMENT_SHAPE),
      ("dm", self.dm, np.dtype(np.float16), DM_SHAPE),
      ("ds", self.ds, np.dtype(np.float16), DS_SHAPE),
      ("reference", self.reference, np.dtype(np.float32), OUTPUT_SHAPE),
    )
    for name, value, dtype, shape in expected:
      if not isinstance(value, np.ndarray) or value.dtype != dtype or value.shape != shape:
        raise ValueError(f"{name} must be an ndarray with dtype={dtype} shape={shape}")
    if not np.isfinite(self.reference).all(): raise ValueError("probe reference must be finite")


@dataclass(frozen=True)
class WMMAConsumerProbe:
  sink: UOp
  fixture: WMMAConsumerProbeFixture
  program: UOp | None = None

  @property
  def emitted(self) -> bool: return self.program is not None


def _wmma_arg() -> tuple:
  """Build the exact gfx1100 signed-int8 WMMA carrier ABI."""
  tc = llama_mmq_candidate_plan().tensor_core
  # These ids describe the four binary A/B carrier axes and three binary C
  # result axes.  The direct host fragments are already in descriptor order,
  # so no symbolic CONTRACT ranges need to remain in the executable graph.
  a_axes = tuple((2100+i, 2) for i in range(4))
  b_axes = tuple((2110+i, 2) for i in range(4))
  c_axes = tuple((2120+i, 2) for i in range(3))
  return (str(tc), tc.dims, tc.dtype_in, tc.dtype_out, "gfx1100", tc.threads,
          (a_axes, b_axes, c_axes), ())


def _direct_fragment(source: UOp, lane: UOp, role: str, substep: int) -> UOp:
  """Load one host-arranged 16-byte carrier for this hardware lane."""
  base = lane*16
  return UOp(Ops.STACK, dtypes.char.vec(16), tuple(
    source.index(base+i).load().replace(tag=("wmma_consumer_probe_fragment", role, substep, i))
    for i in range(16)))


def _probe_sink() -> UOp:
  output = UOp.param(0, dtypes.float.ptr(16*16))
  a = UOp.param(1, dtypes.char.ptr(32*16))
  b = UOp.param(2, dtypes.char.ptr(32*16))
  dm = UOp.param(3, dtypes.half.ptr(8*2))
  ds = UOp.param(4, dtypes.half.ptr(2))
  lane = UOp.special(32, "lidx0")
  index = lambda value: UOp.const(dtypes.weakint, value)

  arg = _wmma_arg()
  zero = UOp.const(dtypes.int.vec(8), 0).replace(tag=("wmma_consumer_probe_i32_zero",))
  # The two native K16 operations reproduce one source-level K32 scale group.
  # Reuse the same deterministic carrier for each substep; the fixture uses
  # uniform fragments, so this represents a uniform logical K32 tile.
  first = UOp(Ops.WMMA, dtypes.int.vec(8),
    (_direct_fragment(a, lane, "A", 0), _direct_fragment(b, lane, "B", 0), zero), arg,
    tag=("wmma_consumer_probe_wmma", 0))
  second = UOp(Ops.WMMA, dtypes.int.vec(8),
    (_direct_fragment(a, lane, "A", 1), _direct_fragment(b, lane, "B", 1), first), arg,
    tag=("wmma_consumer_probe_wmma", 1))
  validate_precontract_wmma_abi(first, context="WMMA consumer probe K[0:16]")
  validate_precontract_wmma_abi(second, context="WMMA consumer probe K[16:32]")

  ds_scale, ds_bias = ds.index(index(0)).load().cast(dtypes.float), ds.index(index(1)).load().cast(dtypes.float)
  corrected = []
  for element in range(8):
    dm_scale = dm.index(index(element*2)).load().cast(dtypes.float)
    dm_bias = dm.index(index(element*2+1)).load().cast(dtypes.float)
    corrected.append((dm_scale*ds_scale*second.gep(element).cast(dtypes.float) + dm_bias*ds_bias).replace(
      tag=("wmma_consumer_probe_correction", element)))

  prior = None
  for element, value in enumerate(corrected):
    local_b, local_a = rdna3_wmma_output_coord(0, element, tc=llama_mmq_candidate_plan().tensor_core)
    row = lane//16 + local_a
    col = lane%16 + local_b
    pointer = output.index(row*16+col, ptr=True)
    if prior is not None: pointer = pointer.after(prior)
    prior = pointer.store(value).replace(tag=("wmma_consumer_probe_writeback", element, "row_major"))
  assert prior is not None
  closed = prior.end(*prior.ranges)
  return UOp(Ops.SINK, dtypes.void, (closed,),
             KernelInfo(name="mmq_llama_wmma_correction_probe", opts_to_apply=()))


def _reference(a_value: int, b_value: int, dm: np.ndarray, ds: np.ndarray) -> np.ndarray:
  """CPU reference for uniform direct fragments and the exact lane mapping."""
  # Two signed K16 instructions form the source-level K32 dot.
  dot = np.int32(32*a_value*b_value)
  lane_values = np.empty((8,), dtype=np.float32)
  for element in range(8):
    # Match the executable's fp16 storage boundary, then its fp32 arithmetic.
    scale = np.float32(dm[element, 0]) * np.float32(ds[0])
    bias = np.float32(dm[element, 1]) * np.float32(ds[1])
    lane_values[element] = np.float32(scale*np.float32(dot) + bias)

  tc = llama_mmq_candidate_plan().tensor_core
  out = np.empty(OUTPUT_SHAPE, dtype=np.float32)
  owners: set[tuple[int, int]] = set()
  for lane in range(32):
    for element in range(8):
      local_b, local_a = rdna3_wmma_output_coord(0, element, tc=tc)
      row, col = lane//16 + local_a, lane%16 + local_b
      if (row, col) in owners: raise ValueError("duplicate probe writeback owner")
      owners.add((row, col))
      out[row, col] = lane_values[element]
  if owners != {(row, col) for row in range(16) for col in range(16)}:
    raise ValueError("probe writeback does not cover the complete 16x16 tile")
  return out


def build_wmma_consumer_probe(*, a_value: int = 2, b_value: int = -3) -> WMMAConsumerProbe:
  """Build the compile/GPU diagnostic and its deterministic CPU fixture."""
  for name, value in (("a_value", a_value), ("b_value", b_value)):
    if not isinstance(value, int) or isinstance(value, bool) or not -128 <= value <= 127:
      raise ValueError(f"{name} must be a signed int8 value")
  dm = np.asarray(((0.5, 1.0), (1.0, 2.0), (1.5, 3.0), (2.0, 4.0),
                   (-0.5, 5.0), (-1.0, 6.0), (-1.5, 7.0), (-2.0, 8.0)), dtype=np.float16)
  ds = np.asarray((2.0, -0.25), dtype=np.float16)
  fixture = WMMAConsumerProbeFixture(
    np.full(FRAGMENT_SHAPE, a_value, dtype=np.int8),
    np.full(FRAGMENT_SHAPE, b_value, dtype=np.int8),
    dm, ds, _reference(a_value, b_value, dm, ds))
  return WMMAConsumerProbe(_probe_sink(), fixture)


def compile_wmma_consumer_probe(probe: WMMAConsumerProbe,
                                target: str = "AMD:ISA:gfx1100") -> WMMAConsumerProbe:
  if not isinstance(probe, WMMAConsumerProbe): raise TypeError("expected WMMAConsumerProbe")
  return replace(probe, program=to_program(probe.sink, AMDISARenderer(Target.parse(target))))


__all__ = ["DM_SHAPE", "DS_SHAPE", "FRAGMENT_SHAPE", "LOCAL_SIZE", "OUTPUT_SHAPE", "SCHEMA",
  "WMMAConsumerProbe", "WMMAConsumerProbeFixture", "build_wmma_consumer_probe", "compile_wmma_consumer_probe"]
