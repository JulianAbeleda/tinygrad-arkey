import struct

import numpy as np
import pytest

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp

from extra.qk.q6k_bounded_mmq import (
  BOUNDED_BLOCKER, CapabilityStatus, Q6KBoundedMMQSpec, Q6KGfx1100Proof, describe_q6k_bounded_mmq,
  emit_q6k_bounded_mmq_kernel, execute_q6k_bounded_mmq, owner_map, owner_map_sha256, q6k_bounded_mmq,
  q6k_bounded_reference, q6k_row_routes,
)
from extra.qk.q6k_mmq_vocabulary import Q6K_BLOCK_BYTES, q6k_weight
from extra.qk.layout import Q6K_HALFWORDS_PER_BLOCK


def _blocks(n, k):
  raw = bytearray(n * ((k + 255)//256) * Q6K_BLOCK_BYTES)
  for off in range(0, len(raw), Q6K_BLOCK_BYTES):
    raw[off:off+192] = np.arange(192, dtype=np.uint8).tobytes()
    raw[off+192:off+208] = np.arange(-8, 8, dtype=np.int8).tobytes()
    raw[off+208:off+210] = struct.pack("<e", 0.125)
  return raw


def test_shape_grid_tails_and_packed_stride_are_fact_derived():
  spec = describe_q6k_bounded_mmq(17, 19, 257)
  assert spec.grid == (2, 2, 1) and spec.k_tiles == 2
  assert spec.tails == {"M": 1, "N": 3, "K": 1}
  assert spec.packed_row_stride_bytes == 2 * 210
  assert spec.packed_weight_bytes == 19 * 2 * 210
  payload = spec.to_json()
  assert "profile" not in str(payload).lower()
  assert payload["grammar"]["qh_offset"] == 128 and payload["grammar"]["zero_point"] == 32


@pytest.mark.parametrize("activation,expected", [("FP16", 16*256*2), ("Q8_1", 16*(256+8*8))])
def test_resource_contract_is_exact_and_tile_bounded(activation, expected):
  small = describe_q6k_bounded_mmq(1, 1, 256, activation=activation).resources()
  huge = describe_q6k_bounded_mmq(511, 151937, 12288, activation=activation).resources()
  assert small == huge
  assert huge.q6_packed_tile_bytes == 16 * 210
  assert huge.q6_decoded_tile_bytes == 16 * 256 * 2
  assert huge.activation_tile_bytes == expected
  assert huge.accumulator_tile_bytes == 16 * 16 * 4
  assert huge.global_workspace_bytes == 0
  assert huge.to_json()["full_weight_decode_bytes"] == 0


def test_candidate_admission_fails_closed_without_final_resource_proof_but_emitter_exists():
  spec = describe_q6k_bounded_mmq(16, 16, 256)
  gate = spec.admission()
  assert gate.status is CapabilityStatus.REQUIRES_FALLBACK and not gate.admitted
  assert gate.fallback_route == "q6k_direct_packed_fp16" and not gate.production_coverage
  assert BOUNDED_BLOCKER in gate.reasons and "one-thread direct owners" in BOUNDED_BLOCKER
  assert emit_q6k_bounded_mmq_kernel(spec).__name__ == "kernel"


def test_scanned_target_facts_are_matched_structurally_without_model_or_vram_tiers():
  gate = describe_q6k_bounded_mmq(16, 16, 256, target="amd_gfx1200", wave_size=32).admission()
  assert gate.status is CapabilityStatus.REQUIRES_FALLBACK
  assert "target backend/architecture/wave is outside the gfx1100 capability" in gate.reasons
  semantic_inputs = set(Q6KBoundedMMQSpec.__dataclass_fields__)
  assert not semantic_inputs.intersection({"model", "model_name", "model_path", "profile", "parameter_count",
                                           "vram_tier", "total_vram", "free_vram"})


def test_cpu_fp16_reference_matches_independent_packed_decode_at_edge_shape():
  spec = describe_q6k_bounded_mmq(3, 2, 257, tile_m=2, tile_n=3)
  packed = _blocks(spec.n, spec.k)
  x = np.linspace(-1, 1, spec.m*spec.k, dtype=np.float32).reshape(spec.m, spec.k)
  got = np.asarray(q6k_bounded_reference(packed, x.tolist(), spec), dtype=np.float32)
  want = np.zeros((spec.m, spec.n), dtype=np.float32)
  for m in range(spec.m):
    for n in range(spec.n):
      for k in range(spec.k):
        bo = n*spec.packed_row_stride_bytes + (k//256)*210
        w = q6k_weight(packed[bo:bo+210], (k%256)//16, k%16)
        want[m, n] = np.float32(want[m, n] + np.float32(w)*x[m, k])
  np.testing.assert_array_equal(got, want)


def test_cpu_q8_reference_applies_per_32_scale_with_k_tail():
  spec = describe_q6k_bounded_mmq(2, 1, 33, activation="Q8_1")
  packed = _blocks(spec.n, spec.k)
  q = np.arange(-16, 17, dtype=np.int8)[None].repeat(2, axis=0)
  scales = [[0.25, 0.5], [0.5, 1.0]]
  got = np.asarray(q6k_bounded_reference(packed, q.tolist(), spec, q8_scales=scales))
  deq = q.astype(np.float32) * np.asarray(scales, dtype=np.float32).repeat(32, axis=1)[:, :33]
  fp = describe_q6k_bounded_mmq(2, 1, 33)
  want = np.asarray(q6k_bounded_reference(packed, deq.tolist(), fp))
  np.testing.assert_allclose(got, want, rtol=0, atol=1e-5)


def test_generated_surface_is_the_bounded_packed_emitter():
  spec = describe_q6k_bounded_mmq(3, 2, 256)
  kernel = emit_q6k_bounded_mmq_kernel(spec, allow_direct_packed_fallback=True)
  out = UOp.placeholder((spec.m, spec.n), dtypes.float32, 0)
  halfs = UOp.placeholder((spec.n * Q6K_HALFWORDS_PER_BLOCK,), dtypes.uint16, 1)
  x = UOp.placeholder((spec.m * spec.k,), dtypes.float16, 2)
  sink = kernel(out, halfs, x)
  assert sink.arg.name == "q6k_bounded_fp16_3_2_256"
  assert sink.arg.opts_to_apply == ()


@pytest.mark.parametrize("activation", ["FP16", "Q8_1"])
def test_custom_kernel_executes_packed_records_fp32_accumulation_and_k_tail(activation):
  spec = Q6KBoundedMMQSpec(3, 2, 257, activation=activation, tile_m=2, tile_n=3)
  raw = _blocks(spec.n, spec.k)
  halfs = Tensor(np.frombuffer(raw, dtype=np.uint16).copy(), device="PYTHON")
  if activation == "FP16":
    host_x = np.linspace(-1, 1, spec.m * spec.k, dtype=np.float16).reshape(spec.m, spec.k)
    got = q6k_bounded_mmq(halfs, Tensor(host_x, device="PYTHON"), spec).numpy()
    want = q6k_bounded_reference(raw, host_x.tolist(), spec)
  else:
    host_x = np.resize(np.arange(-31, 32, dtype=np.int8), (spec.m, spec.k))
    host_scales = np.full((spec.m, (spec.k+31)//32), 0.125, dtype=np.float32)
    got = q6k_bounded_mmq(halfs, Tensor(host_x, device="PYTHON"), spec,
                           q8_scales=Tensor(host_scales, device="PYTHON")).numpy()
    want = q6k_bounded_reference(raw, host_x.tolist(), spec, q8_scales=host_scales.tolist())
  np.testing.assert_allclose(got, np.asarray(want, dtype=np.float32), rtol=2e-5, atol=2e-5)


def _proof(spec, **kw):
  values = dict(candidate_id="q6k", kernel_name="q6k_final", target="amd_gfx1100", wavefront_size=32,
                workgroup_threads=64, vgpr=48, lds_bytes=spec.resources().lds_bytes, scratch_bytes=0,
                vgpr_spills=0, sgpr_spills=0, occupancy=0.5, barrier_sites=1, matrix_core_sites=1,
                owner_map_sha256=owner_map_sha256(spec.tile_m, spec.tile_n, spec.workgroup_size))
  values.update(kw)
  return Q6KGfx1100Proof(**values)


def test_owner_map_is_total_unique_and_hashed_into_final_proof():
  spec = describe_q6k_bounded_mmq(3, 5, 257)
  mapping = owner_map(16, 16)
  assert len(mapping) == 256 and len({(m, n) for m, n, _ in mapping}) == 256
  assert all(owner == (m*16+n) % 64 for m, n, owner in mapping)
  assert spec.admission(_proof(spec)).admitted
  with pytest.raises(ValueError, match="owner/writeback"):
    _proof(spec, owner_map_sha256="wrong").validate(spec)


def test_final_proof_rejects_estimates_spills_and_missing_matrix_core():
  spec = describe_q6k_bounded_mmq(1, 1, 256)
  for bad in (_proof(spec, source="bounded_formula"), _proof(spec, scratch_bytes=4), _proof(spec, matrix_core_sites=0)):
    assert not spec.admission(bad).admitted


def test_executable_route_reports_per_row_direct_packed_fallback_for_q8_outer_k_and_tails():
  spec = describe_q6k_bounded_mmq(3, 5, 513, activation="Q8_1", tile_m=2, tile_n=3)
  packed = _blocks(spec.n, spec.k)
  q = np.resize(np.arange(-31, 32, dtype=np.int8), (spec.m, spec.k))
  scales = np.full((spec.m, (spec.k+31)//32), 0.125, dtype=np.float32)
  got, census = execute_q6k_bounded_mmq(packed, q.tolist(), spec, q8_scales=scales.tolist())
  want = q6k_bounded_reference(packed, q.tolist(), spec, q8_scales=scales.tolist())
  np.testing.assert_array_equal(np.asarray(got, dtype=np.float32), np.asarray(want, dtype=np.float32))
  assert census == q6k_row_routes(spec) and len(census) == spec.n
  assert all(row.route == "q6k_direct_packed_q8_1" and row.reason for row in census)
