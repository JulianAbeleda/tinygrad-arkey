import pytest

from extra.qk.generated_candidates import GeneratedCandidateRegistry, builtin_registry, select_generated_candidate
from extra.qk.quant_specs import activation_spec, quant_spec
from extra.qk import route_manifest
from extra.qk.runtime_specs import ActivationQuantSpec, GeneratedCandidate, QuantizedTensorSpec, RuntimeOpSpec


def _manifest_authority_gates(route_id):
  return tuple(part.strip() for part in route_manifest.ROUTES[route_id]["authority_gate"].split(" + ") if part.strip())


def _manifest_runtime_roles(route_id):
  aliases = {"attn_k": "attn_kv", "attn_v": "attn_kv", "attention_tile": "attention", "attention_combine": "attention"}
  roles = []
  for role in route_manifest.ROUTES[route_id]["roles"]:
    normalized = aliases.get(role, role)
    if normalized not in roles:
      roles.append(normalized)
  return tuple(roles)


def test_runtime_op_spec_round_trips_and_validates():
  op = RuntimeOpSpec("QuantizedLinear", "prefill", "ffn_gate_up", {"M": 512, "N": 17408, "K": 5120},
                     quant_spec("Q4_K").tensor_spec(), activation_spec("Q8_1").activation_spec(),
                     lowering_strategy="iu8_wmma_grouped_dot", route_id="prefill_q4k_int8_wmma_generated_research")
  row = op.to_json()
  assert RuntimeOpSpec.from_json(row) == op
  with pytest.raises(ValueError, match="family"):
    RuntimeOpSpec("HandKernel", "prefill", "unknown", {}, QuantizedTensorSpec("Q4_K"))


def test_generated_candidate_round_trip_and_provenance():
  cand = GeneratedCandidate("c.q4", "QuantizedLinear", ("Q4_K",), ("Q8_1",), ("prefill",), ("ffn_gate_up",),
                            "iu8_wmma_grouped_dot", "machine_authored_generated",
                            route_id="prefill_q4k_int8_wmma_generated_research")
  assert cand.is_generated_only
  assert GeneratedCandidate.from_json(cand.to_json()) == cand
  banned = GeneratedCandidate("c.bad", "QuantizedLinear", ("Q4_K",), ("Q8_1",), ("prefill",), ("unknown",),
                              "iu8_wmma_grouped_dot", "banned")
  with pytest.raises(ValueError, match="non-generated provenance"):
    GeneratedCandidateRegistry([banned])


def test_builtin_registry_selects_wmma_and_blocks_unknown():
  op = RuntimeOpSpec("QuantizedLinear", "prefill", "ffn_gate_up", {"M": 512, "N": 17408, "K": 5120},
                     quant_spec("Q4_K").tensor_spec(), activation_spec("Q8_1").activation_spec(),
                     lowering_strategy="iu8_wmma_grouped_dot")
  selected = select_generated_candidate(op)
  assert selected.status == "selected"
  assert selected.candidate and selected.candidate.candidate_id == "quant_linear_prefill.q4k_int8_wmma_tensor_substrate"
  blocked = builtin_registry().select(RuntimeOpSpec("ActivationFusion", "prefill", "unknown", {},
                                                    QuantizedTensorSpec("unknown"), ActivationQuantSpec("none")))
  assert blocked.status == "blocked"
  assert blocked.candidate is None


def test_builtin_registry_selects_wmma_tiled_candidate():
  op = RuntimeOpSpec("QuantizedLinear", "prefill", "ffn_down", {"M": 512, "N": 5120, "K": 17408},
                     quant_spec("Q4_K").tensor_spec(), activation_spec("Q8_1").activation_spec(),
                     lowering_strategy="iu8_wmma_tiled_grouped_dot")
  selected = select_generated_candidate(op, preferred=("quant_linear_prefill.q4k_int8_wmma_tiled_substrate",))
  assert selected.status == "selected"
  assert selected.candidate and selected.candidate.route_id == "prefill_q4k_int8_wmma_tiled_research"


def test_builtin_generated_candidates_match_manifest_route_metadata():
  for candidate in builtin_registry().all():
    assert candidate.route_id in route_manifest.ROUTES
    manifest = route_manifest.ROUTES[candidate.route_id]
    assert candidate.provenance == route_manifest.route_provenance(candidate.route_id)
    assert candidate.supported_quant_formats == tuple(manifest["quant"])
    assert candidate.authority_gates == _manifest_authority_gates(candidate.route_id)
    assert set(_manifest_runtime_roles(candidate.route_id)).issubset(set(candidate.roles))


def test_quant_specs_are_data_descriptors():
  q4 = quant_spec("Q4_K")
  assert q4.block_size == 256
  assert q4.group_size == 32
  assert "Q8_1" in q4.supported_activation_formats
  q8 = activation_spec("Q8_1")
  assert q8.block_size == 32
  assert q8.signed is True
