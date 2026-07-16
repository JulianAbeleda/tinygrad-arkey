import pytest

from extra.qk.mmq_lowering_audit import admit_lowering, trace_lowering
from extra.qk.q4k_q8_mmq_prefill_spec import Q4KQ8MMQPrefillSpec
from extra.qk.prefill_primitive_spec import LaunchMetadata, PrimitiveABI


def candidate():
  return Q4KQ8MMQPrefillSpec(workload="prefill", profile="test", role="ffn_gate_up",
    quant_format="Q4_K", activation_format="Q8_1", weight_layout="q4k_rows",
    output_layout="tokens_rows", m=32, n=64, k=256, abi=PrimitiveABI(),
    launch=LaunchMetadata(512, 16)).logical_candidate()


def evidence(**overrides):
  value = {"resources": {"vgpr": 96, "lds_bytes": 4096, "wavefront_size": 32},
           "isa": {"barrier_sites": 2, "mfma_sites": 8}}
  for section, changes in overrides.items(): value[section] = {**value[section], **changes}
  return value


def test_trace_is_explicit_about_axis_to_wave_and_resource_join():
  trace = admit_lowering(candidate(), evidence())
  assert trace.axes == ("m", "n", "k", "group", "activation_block")
  assert (trace.waves, trace.wave_size, trace.lds_bytes, trace.vgpr) == (16, 32, 4096, 96)
  assert trace.to_dict()["schema"] == "tinygrad.mmq.lowering_audit.v1"


@pytest.mark.parametrize("section,key", [("resources", "vgpr"), ("resources", "lds_bytes"),
                                          ("isa", "barrier_sites"), ("isa", "mfma_sites")])
def test_missing_physical_evidence_fails_closed(section, key):
  payload = evidence(); del payload[section][key]
  with pytest.raises(ValueError, match="missing"):
    trace_lowering(candidate(), payload)


def test_multi_wave_without_barrier_is_not_admitted():
  with pytest.raises(ValueError, match="barrier"):
    admit_lowering(candidate(), evidence(isa={"barrier_sites": 0}))


def test_mfma_is_required_even_when_resources_are_present():
  with pytest.raises(ValueError, match="mfma"):
    admit_lowering(candidate(), evidence(isa={"mfma_sites": 0}))
