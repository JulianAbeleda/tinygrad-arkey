import numpy as np
import pytest

from tinygrad.uop.ops import Ops

from extra.qk.mmq_llama_wmma_correction_probe import (
  LOCAL_SIZE, build_wmma_consumer_probe, compile_wmma_consumer_probe)
from extra.qk.prefill.amd_native_program_resources import amd_native_program_resources


def test_probe_fixture_has_exact_finite_lane_mapped_reference():
  probe = build_wmma_consumer_probe()
  assert probe.fixture.reference.shape == (16, 16)
  assert np.isfinite(probe.fixture.reference).all()
  # Two K16 signed WMMA operations over A=2 and B=-3 produce -192.
  assert set(np.unique(probe.fixture.reference).tolist()) == {
    -192.25, -384.5, -576.75, -769.0, 190.75, 382.5, 574.25, 766.0}
  # Each WMMA result element has one owner in every lane.
  values, counts = np.unique(probe.fixture.reference, return_counts=True)
  assert len(values) == 8 and set(counts.tolist()) == {32}


def test_probe_graph_is_direct_fragment_wmma_correction_and_row_major_writeback():
  nodes = list(build_wmma_consumer_probe().sink.toposort())
  wmmas = [x for x in nodes if x.op is Ops.WMMA]
  assert len(wmmas) == 2 and wmmas[1].src[2] is wmmas[0]
  assert all(x.src[0].dtype.count == x.src[1].dtype.count == 16 for x in wmmas)
  assert not [x for x in nodes if x.op in (Ops.DEFINE_LOCAL, Ops.BARRIER)]
  assert len([x for x in nodes if x.tag and x.tag[:1] == ("wmma_consumer_probe_correction",)]) == 8
  stores = [x for x in nodes if x.op is Ops.STORE and x.tag and
            x.tag[:1] == ("wmma_consumer_probe_writeback",)]
  assert len(stores) == 8 and {x.tag[-1] for x in stores} == {"row_major"}
  specials = [x for x in nodes if x.op is Ops.SPECIAL]
  assert len(specials) == 1 and specials[0].arg == "lidx0" and specials[0].src[0].arg == LOCAL_SIZE[0]


def test_probe_rejects_non_int8_fragment_constants():
  with pytest.raises(ValueError, match="signed int8"): build_wmma_consumer_probe(a_value=128)
  with pytest.raises(ValueError, match="signed int8"): build_wmma_consumer_probe(b_value=-129)


def test_probe_compiles_to_real_signed_i8_wmma_and_zero_scratch():
  compiled = compile_wmma_consumer_probe(build_wmma_consumer_probe())
  assert compiled.emitted and compiled.program is not None
  source = next(x.arg for x in compiled.program.src if x.op is Ops.SOURCE)
  assert source.count("v_wmma_i32_16x16x16_iu8") == 2
  assert "global_store_b32" in source
  assert "ds_read" not in source and "ds_write" not in source
  binary = next(x.arg for x in compiled.program.src if x.op is Ops.BINARY)
  assert isinstance(binary, bytes) and binary
  resources = amd_native_program_resources(compiled.program, target="AMD:ISA:gfx1100")
  expected = {"lds_bytes": 0, "scratch_bytes": 0, "vgpr_spills": 0, "sgpr_spills": 0,
              "workgroup_threads": 32, "wavefront_size": 32}
  assert {key:resources[key] for key in expected} == expected
  assert 0 < resources["vgpr"] <= 256 and resources["sgpr"] > 0
