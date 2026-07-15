import json
import pytest
from extra.qk.q4k_q8_mmq_prefill_spec import Q4KQ8MMQPrefillSpec, enumerate_q4k_q8_mmq_candidates
from extra.qk.prefill_primitive_spec import PrimitiveABI, LaunchMetadata

def make(**kw):
  fields = dict(workload="prefill", profile="qwen3-14b", role="ffn_gate_up", quant_format="Q4_K", activation_format="Q8_1", weight_layout="q4k_rows", output_layout="tokens_rows", m=32, n=64, k=256, abi=PrimitiveABI(), launch=LaunchMetadata(512, 16))
  fields.update(kw); return Q4KQ8MMQPrefillSpec(**fields)

def test_descriptor_is_json_safe_and_identity_includes_schedule():
  a, b = make(tile_m=16), make(tile_m=32)
  assert a.canonical_identity() != b.canonical_identity()
  assert json.loads(json.dumps(a.canonical_payload(), sort_keys=True)) == a.canonical_payload()
  assert a.to_json()["mmq"]["tile_m"] == 16

def test_candidate_axes_are_data_and_invalid_candidates_fail_closed():
  got = list(enumerate_q4k_q8_mmq_candidates(make(), tile_m=(16, 32), tile_n=(16,)))
  assert len(got) == 2
  with pytest.raises(ValueError, match="alignment"):
    make(k=250).validate()
  with pytest.raises(ValueError, match="LDS"):
    make(lds_bytes=64 * 1024 + 1).validate()
  with pytest.raises(ValueError, match="owner"):
    make(parts=2).validate()

def test_explicit_abi_and_launch_metadata_are_validated():
  with pytest.raises(ValueError, match="ABI"):
    make(abi=PrimitiveABI(("out", "out"), ("float32", "uint8"))).validate()
  with pytest.raises(ValueError, match="wave"):
    make(workgroup_size=96).validate()

def test_default_geometry_matches_cooperative_probe_and_shared_logical_contract():
  spec = make()
  assert spec.workgroup_size == 512
  candidate = spec.logical_candidate()
  assert candidate.mapping.workgroup_size == 512
  assert candidate.descriptor.q4k.block_elements == 256
  assert candidate.descriptor.q8.block_elements == 32
  assert candidate.capability.wave_sizes == (32,)

def test_search_rejects_inert_resource_axes_and_uses_gfx1100_wave32():
  assert make().wave_width == 32
  with pytest.raises(ValueError, match="inert search axes"):
    list(enumerate_q4k_q8_mmq_candidates(make(), staging_strategy=("register", "lds")))
  with pytest.raises(ValueError, match="not lowered"):
    make(accumulator_slots=8).validate()
  with pytest.raises(ValueError, match="unsupported schedule_options"):
    make(schedule_options=(("staging", "lds"),)).validate()

@pytest.mark.parametrize("field", ["activation_layout", "tile_x_layout", "tile_y_layout"])
def test_descriptor_rejects_unknown_layout_vocabulary(field):
  with pytest.raises(ValueError, match="layout"):
    make(**{field: "arbitrary_layout"}).validate()
