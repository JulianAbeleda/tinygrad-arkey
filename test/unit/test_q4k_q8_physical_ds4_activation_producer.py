from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops

from extra.qk.layout import Q8_1_MMQ_BLOCK_ELEMS, Q8_1_MMQ_GROUPS_PER_BLOCK
from extra.qk.mmq_q4k_q8_reference import q8_1_mmq_ds4_quantize_reference
from extra.qk.q4k_q8_activation_producer import (
  PHYSICAL_DS4_LAYOUT, PhysicalDS4Q8ActivationSpec, produce_physical_ds4_q8_1)


def test_physical_ds4_descriptor_is_frozen_and_validates_exact_grammar():
  spec = PhysicalDS4Q8ActivationSpec(3, 256)
  spec.validate()
  assert spec.layout == PHYSICAL_DS4_LAYOUT
  assert spec.values_shape == (2, 3, Q8_1_MMQ_BLOCK_ELEMS)
  assert spec.metadata_shape == (2, 3, Q8_1_MMQ_GROUPS_PER_BLOCK)
  assert spec.waves == 24
  with pytest.raises(FrozenInstanceError): spec.m = 4
  for bad in (replace(spec, k=160), replace(spec, wave_size=64),
              replace(spec, sum_semantics="sum_dequant_q8"), replace(spec, value_dtype="uint8")):
    with pytest.raises(ValueError): bad.validate()


def test_physical_indices_have_one_wave_lane_owner_and_match_reference_layout():
  spec = PhysicalDS4Q8ActivationSpec(3, 256)
  source_owned, value_owned, metadata_owned = set(), set(), set()
  for wave in range(spec.waves):
    block, row, group = spec.logical_owner(wave)
    metadata_owned.add(spec.metadata_index(block, row, group))
    for lane in range(spec.group_elems):
      source_owned.add(spec.source_index(block, row, group, lane))
      value_owned.add(spec.value_index(block, row, group, lane))
  assert source_owned == value_owned == set(range(spec.m * spec.k))
  assert metadata_owned == set(range(spec.waves))

  x = np.arange(spec.m * spec.k, dtype=np.float32).reshape(spec.m, spec.k) - 301.25
  ref_values, ref_scales, ref_sums = q8_1_mmq_ds4_quantize_reference(x)
  for wave in range(spec.waves):
    block, row, group = spec.logical_owner(wave)
    mi = spec.metadata_index(block, row, group)
    assert ref_scales.reshape(-1)[mi] == ref_scales[block, row, group]
    assert ref_sums.reshape(-1)[mi] == ref_sums[block, row, group]
    for lane in range(spec.group_elems):
      vi = spec.value_index(block, row, group, lane)
      assert ref_values.reshape(-1)[vi] == ref_values[block, row, group * spec.group_elems + lane]


def _producer_sink(m=2, k=Q8_1_MMQ_BLOCK_ELEMS):
  source = Tensor.empty(m, k, dtype=dtypes.float32, device="CPU")
  out = produce_physical_ds4_q8_1(source)
  assert out.values.shape == (k // Q8_1_MMQ_BLOCK_ELEMS, m, Q8_1_MMQ_BLOCK_ELEMS)
  assert out.scales.shape == out.sums.shape == (k // Q8_1_MMQ_BLOCK_ELEMS, m, Q8_1_MMQ_GROUPS_PER_BLOCK)
  # The three distinct output buffers are AFTER views of the same custom-kernel sink.
  assert out.values.uop.base.src[1] is out.scales.uop.base.src[1] is out.sums.uop.base.src[1]
  sinks = [u for u in out.values.schedule_linear().toposort() if u.op is Ops.SINK]
  assert len(sinks) == 1
  return sinks[0]


def test_physical_producer_has_one_materialization_and_semantic_store_per_output():
  sink = _producer_sink()
  stores = [u for u in sink.toposort() if u.op is Ops.STORE and u.src[0].src[0].op is Ops.PARAM]
  assert len(stores) == 3
  gated = [u for u in stores if len(u.src) == 3]
  assert len(gated) == 2  # one lane owns each scale and original-fp sum; every lane owns one value


def test_physical_producer_static_amd_isa_compile_does_not_open_runtime():
  from tinygrad.codegen import full_rewrite_to_sink
  from tinygrad.helpers import Target
  from tinygrad.renderer.isa.amd import AMDISARenderer
  lowered = full_rewrite_to_sink(_producer_sink(), AMDISARenderer(Target.parse("AMD:ISA:gfx1100")), optimize=True)
  assert any(u.op is Ops.STORE for u in lowered.toposort())
