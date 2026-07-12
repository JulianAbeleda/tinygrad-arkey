import pytest

from extra.qk.generated_candidates import GeneratedCandidateRegistry, builtin_registry, select_generated_candidate
from extra.qk.quant_specs import activation_spec, quant_spec
from extra.qk import route_manifest
from extra.qk.runtime_specs import (
  FULL_KERNEL_CANDIDATE_SCHEMA, ActivationQuantSpec, GeneratedCandidate, QuantizedTensorSpec, RuntimeOpSpec,
)


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


def _strict_full_kernel_candidate(**overrides):
  payload = {
    "schema_version": FULL_KERNEL_CANDIDATE_SCHEMA,
    "workload": {"profile": "qwen3_8b_q4k_m_gfx1100", "role": "ffn_gate_up",
      "shape": {"m": 512, "n": 12288, "k": 4096},
      "dtypes": {"a": "fp16", "b": "fp16", "c": "fp16", "accumulator": "fp32"},
      "layout": {"a": "row_major", "b": "transposed_row_major", "c": "row_major"},
      "target": {"backend": "AMD", "arch": "gfx1100", "wave_size": 32}},
    "schedule": {"tile": {"m": 128, "n": 128, "k": 32}, "waves": {"m": 4, "n": 2}, "threads": 256,
      "lane_ownership": "wave_tile_v1",
      "cooperative_load": {"a": {"lane_mapping": "contiguous", "vector_width": 8, "alignment": 16},
                           "b": {"lane_mapping": "contiguous", "vector_width": 8, "alignment": 16}},
      "lds": {"windows": {"a": [0, 16384], "b": [16384, 40960]}, "strides": {"a": 144, "b": 144},
              "padding": 16, "banks": 32, "store_vector_width": 8, "load_vector_width": 8},
      "pipeline": {"buffer_count": 2, "stage_count": 2,
                   "epoch_graph": [{"epoch": "body", "reads": ["slot_previous"], "writes": ["slot_next"]}]},
      "wmma": {"instruction_family": "wmma_f32_16x16x16_f16", "fragment_layout": "gfx11_wave32_v1",
               "accumulator_ownership": "wave_tile_v1"},
      "dependency_policy": {"waitcnt": {"vm": 0, "lgkm": 0}, "barriers": ["after_lds_store"]},
      "residency": {"preload": ["a", "b"], "resident": ["accumulator"], "reuse": {"a": 2, "b": 4}},
      "epilogue": {"lane_mapping": "wmma_accumulator_v1", "vector_width": 8},
      "numerical_mode": "ieee_fp16_acc_fp32"},
    "static_constraints": {"max_lds_bytes": 65536, "max_vgpr_per_thread": 256, "allow_spill": False},
    "applicability": {"exact_shape": True, "profiles": ["qwen3_8b_q4k_m_gfx1100"],
                      "roles": ["ffn_gate_up"], "targets": ["AMD:gfx1100:wave32"]}}
  values = dict(candidate_id="fp16.ffn_gate_up.m512n12288k4096.gfx1100", op_family="DenseLinear",
                supported_quant_formats=("fp16",), supported_activation_formats=("fp16",), phases=("prefill",),
                roles=("ffn_gate_up",), lowering_strategy="tinygrad_scheduler",
                provenance="machine_authored_generated", full_kernel_candidate=payload)
  values.update(overrides)
  return GeneratedCandidate(**values)


def _strict_op(**overrides):
  values = dict(family="DenseLinear", phase="prefill", role="ffn_gate_up",
                shape={"M": 512, "N": 12288, "K": 4096}, weight=QuantizedTensorSpec("fp16"),
                activation=ActivationQuantSpec("fp16"), lowering_strategy="tinygrad_scheduler",
                codegen_features=("wmma_f32_16x16x16_f16",), profile="qwen3_8b_q4k_m_gfx1100",
                target={"backend": "AMD", "arch": "gfx1100", "wave_size": 32})
  values.update(overrides)
  return RuntimeOpSpec(**values)


def test_strict_full_kernel_candidate_identity_round_trip_and_tamper_rejection():
  candidate = _strict_full_kernel_candidate()
  row = candidate.to_json()
  assert row["full_kernel_candidate"]["schema_version"] == FULL_KERNEL_CANDIDATE_SCHEMA
  assert row["canonical_identity"] == "56ab6f662cb52bca958b92cdbffac790784a29047c9a1f4dc1e0e9a8b6d6da3d"
  assert GeneratedCandidate.from_json(row) == candidate
  row["full_kernel_candidate"]["workload"]["shape"]["m"] = 1024
  with pytest.raises(ValueError, match="canonical_identity"):
    GeneratedCandidate.from_json(row)


@pytest.mark.parametrize("change", (
  {"shape": {"M": 1024, "N": 12288, "K": 4096}},
  {"target": {"backend": "AMD", "arch": "gfx1200", "wave_size": 32}},
  {"codegen_features": ()},
))
def test_strict_full_kernel_selection_fails_closed_on_binding_mismatch(change):
  candidate = _strict_full_kernel_candidate()
  registry = GeneratedCandidateRegistry([candidate])
  assert registry.select(_strict_op(), require_full_kernel=True,
                         required_canonical_identity=candidate.canonical_identity).status == "selected"
  selection = registry.select(_strict_op(**change), require_full_kernel=True,
                              required_canonical_identity=candidate.canonical_identity)
  assert selection.status == "blocked"
  assert selection.candidate is None


def test_strict_full_kernel_contract_rejects_incomplete_candidate_and_legacy_route():
  payload = _strict_full_kernel_candidate().full_kernel_candidate
  assert payload is not None
  payload["applicability"]["exact_shape"] = False
  with pytest.raises(ValueError, match="exact_shape"):
    _strict_full_kernel_candidate(full_kernel_candidate=payload)
  legacy = GeneratedCandidate("legacy", "QuantizedLinear", ("Q4_K",), ("Q8_1",), ("prefill",), ("ffn_gate_up",),
                              "iu8_wmma_grouped_dot", "machine_authored_generated")
  selection = GeneratedCandidateRegistry([legacy]).select(_strict_op(), require_full_kernel=True,
                                                          required_canonical_identity="0" * 64)
  assert selection.status == "blocked"


def test_strict_full_kernel_selection_requires_exact_canonical_identity():
  candidate = _strict_full_kernel_candidate()
  registry = GeneratedCandidateRegistry([candidate])
  assert registry.select(_strict_op(), require_full_kernel=True).status == "blocked"
  assert registry.select(_strict_op(), require_full_kernel=True,
                         required_canonical_identity=candidate.canonical_identity.upper()).status == "blocked"
  assert registry.select(_strict_op(), require_full_kernel=True,
                         required_canonical_identity="0" * 64).status == "blocked"
