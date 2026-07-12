import pytest
import json

from extra.qk.generated_candidates import GeneratedCandidateRegistry, builtin_registry, select_generated_candidate
from extra.qk.quant_specs import activation_spec, quant_spec
from extra.qk import route_manifest
from extra.qk import pure_search_guard
from extra.qk.runtime_specs import (
  ANCHOR_SINGLE_BUFFER_CANDIDATE_HASH, FULL_KERNEL_CANDIDATE_SCHEMA, ActivationQuantSpec, GeneratedCandidate,
  QuantizedTensorSpec, RuntimeOpSpec,
  bind_full_kernel_candidate,
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


def _single_buffer_anchor_candidate():
  payload = _strict_full_kernel_candidate().full_kernel_candidate
  assert payload is not None
  payload["schedule"]["pipeline"] = {"buffer_count": 1, "stage_count": 1,
    "epoch_graph": [{"epoch": "body", "slot": 0, "produce": ["a", "b"], "wait": ["global", "lds"],
                     "barrier": "before_fragment_load", "consume": ["a", "b"]}]}
  payload["schedule"].update(lane_ownership="rdna3_wmma_f32_16x16x16_f16_lds2_static")
  payload["schedule"]["cooperative_load"] = {
    "a": {"lane_mapping": "cooperative_row_stride_64_b128", "vector_width": 8, "alignment": 16},
    "b": {"lane_mapping": "cooperative_row_stride_64_b128", "vector_width": 8, "alignment": 16}}
  payload["schedule"]["lds"].update(windows={"a": [0, 10240], "b": [10240, 20480]},
                                     strides={"a": 80, "b": 80})
  payload["schedule"]["wmma"].update(fragment_layout="rdna3_wmma_f32_16x16x16_f16_lds2_static",
    accumulator_ownership="wmma_accum_wm_x_wn_8_vgprs")
  payload["schedule"]["dependency_policy"] = {
    "waitcnt": {"vm": 0, "lgkm": 0}, "barriers": ["before_fragment_load", "after_wmma_before_slot_reuse"]}
  payload["schedule"]["residency"] = {"preload": ["a", "b"], "resident": ["accumulator"], "reuse": {"a": 4, "b": 2}}
  payload["schedule"]["epilogue"] = {"lane_mapping": "wmma_accumulator_scalar_b16", "vector_width": 1}
  return _strict_full_kernel_candidate(full_kernel_candidate=payload)


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
  context = candidate.kernel_candidate_context()
  assert (context.schema_version, context.canonical_identity) == (FULL_KERNEL_CANDIDATE_SCHEMA, row["canonical_identity"])
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


def test_bind_full_kernel_candidate_to_exact_single_buffer_anchor():
  candidate = _single_buffer_anchor_candidate()
  payload = candidate.full_kernel_candidate
  assert payload is not None
  assert candidate.canonical_identity == ANCHOR_SINGLE_BUFFER_CANDIDATE_HASH
  context = bind_full_kernel_candidate(payload, candidate.canonical_identity,
    profile="qwen3_8b_q4k_m_gfx1100", role="ffn_gate_up", shape=(512, 12288, 4096),
    target={"backend": "AMD", "arch": "gfx1100", "wave_size": 32}, tile=(128, 128, 32), waves=(4, 2),
    threads=256, buffer_count=1, stage_count=1, lds_windows={"a": [0, 10240], "b": [10240, 20480]},
    lds_strides={"a": 80, "b": 80}, lds_padding=16, lds_bytes=20480)
  assert context.canonical_identity == candidate.canonical_identity
  assert context.geometry.tile == (128, 128, 32)
  assert context.geometry.waves == (4, 2)
  assert context.geometry.threads == 256 and context.geometry.wave_size == 32
  assert [(w.role, w.base, w.end, w.stride_bytes) for w in context.geometry.lds_windows] == [
    ("A", 0, 10240, 80), ("B", 10240, 20480, 80)]


@pytest.mark.parametrize(("field", "value", "error"), (
  ("canonical_identity", "0" * 64, "canonical identity"),
  ("shape", (1024, 12288, 4096), "shape"),
  ("target", {"backend": "AMD", "arch": "gfx1200", "wave_size": 32}, "target"),
  ("buffer_count", 2, "pipeline.buffer_count"),
  ("stage_count", 2, "pipeline.stage_count"),
  ("lds_windows", {"a": [0, 16384], "b": [16384, 40960]}, "lds.windows"),
  ("lds_strides", {"a": 144, "b": 144}, "lds.strides"),
))
def test_bind_full_kernel_candidate_fails_closed(field, value, error):
  candidate = _single_buffer_anchor_candidate()
  payload = candidate.full_kernel_candidate
  assert payload is not None
  kwargs = dict(canonical_identity=candidate.canonical_identity, profile="qwen3_8b_q4k_m_gfx1100", role="ffn_gate_up",
                shape=(512, 12288, 4096), target={"backend": "AMD", "arch": "gfx1100", "wave_size": 32},
                tile=(128, 128, 32), waves=(4, 2), threads=256, buffer_count=1, stage_count=1,
                lds_windows={"a": [0, 10240], "b": [10240, 20480]}, lds_strides={"a": 80, "b": 80},
                lds_padding=16, lds_bytes=20480)
  kwargs[field] = value
  with pytest.raises(ValueError, match=error): bind_full_kernel_candidate(payload, **kwargs)


@pytest.mark.parametrize(("mutation", "error"), (
  (lambda p: p["schedule"]["pipeline"].update(stage_count=2), "pipeline.stage_count"),
  (lambda p: p["schedule"]["lds"].update(windows={"a": [0, 16384], "b": [16384, 40960]}), "lds.windows"),
  (lambda p: p["schedule"]["lds"].update(strides={"a": 144, "b": 144}), "lds.strides"),
  (lambda p: p["schedule"]["cooperative_load"]["a"].update(vector_width=4), "cooperative_load.a"),
  (lambda p: p["schedule"]["wmma"].update(fragment_layout="gfx11"), "wmma.fragment_layout"),
  (lambda p: p["static_constraints"].update(max_lds_bytes=16384), "static_constraints.max_lds_bytes"),
))
def test_bind_rejects_self_consistent_hash_with_false_emitted_descriptor(mutation, error):
  candidate = _single_buffer_anchor_candidate()
  payload = candidate.full_kernel_candidate
  assert payload is not None
  mutation(payload)
  candidate = _strict_full_kernel_candidate(full_kernel_candidate=payload)
  with pytest.raises(ValueError, match=error):
    bind_full_kernel_candidate(payload, candidate.canonical_identity, profile="qwen3_8b_q4k_m_gfx1100", role="ffn_gate_up",
      shape=(512, 12288, 4096), target={"backend": "AMD", "arch": "gfx1100", "wave_size": 32},
      tile=(128, 128, 32), waves=(4, 2), threads=256, buffer_count=1, stage_count=1,
      lds_windows={"a": [0, 10240], "b": [10240, 20480]}, lds_strides={"a": 80, "b": 80},
      lds_padding=16, lds_bytes=20480)


def test_exact_anchor_candidate_selects_truthful_pure_research_route():
  candidate = _single_buffer_anchor_candidate()
  env = {"PREFILL_GRAPH_GEMM": "1", "PREFILL_WMMA_LDS_PRIMITIVE": "1",
         "BOLTBEAM_FULL_KERNEL_CANDIDATE_JSON": json.dumps(candidate.full_kernel_candidate),
         "BOLTBEAM_FULL_KERNEL_CANDIDATE_HASH": candidate.canonical_identity}
  row = {x["family"]: x for x in pure_search_guard.effective_routes(env)}["prefill_gemm"]
  assert row["effective_route"] == "prefill_wmma_lds_single_buffer_candidate_generated"
  assert row["provenance"] == "tinygrad_scheduler_generated"
  assert row["strict_pure"] is True and row["pure"] is True and row["rolled_back_to_oracle"] is False


def test_candidate_route_selector_fails_closed_and_default_is_unchanged():
  default = {x["family"]: x for x in pure_search_guard.effective_routes({})}["prefill_gemm"]
  assert default["effective_route"] == "prefill_v2_scheduler_matmul_default"
  candidate = _single_buffer_anchor_candidate()
  base = {"PREFILL_GRAPH_GEMM": "1", "PREFILL_WMMA_LDS_PRIMITIVE": "1",
          "BOLTBEAM_FULL_KERNEL_CANDIDATE_JSON": json.dumps(candidate.full_kernel_candidate)}
  with pytest.raises(ValueError, match="provided together"): pure_search_guard.effective_routes(base)
  with pytest.raises(ValueError, match="canonical identity"):
    pure_search_guard.effective_routes({**base, "BOLTBEAM_FULL_KERNEL_CANDIDATE_HASH": "0" * 64})
