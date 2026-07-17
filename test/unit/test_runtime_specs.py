import pytest
import json, pickle
import hashlib

from extra.qk.generated_candidates import GeneratedCandidateRegistry, builtin_registry, select_generated_candidate
from extra.qk.quant_specs import activation_spec, quant_spec
from extra.qk import route_manifest
from extra.qk import pure_search_guard
from extra.qk import prefill_graph_gemm_route
from extra.qk.runtime_specs import (
  ANCHOR_SINGLE_BUFFER_CANDIDATE_HASH, FULL_KERNEL_CANDIDATE_SCHEMA, PACKED_SCALAR_DECODER_VERSION, ActivationQuantSpec, GeneratedCandidate,
  CandidateAdmissionFacts, QuantizedTensorSpec, RuntimeOpSpec, FullKernelCandidateSet, FullKernelCandidateSetEntry,
  GFX1100_Q4K_Q8_FIVE_BUFFER_CAPABILITY, GFX1100_TWO_BUFFER_STAGE1_CAPABILITY, Q4KQ8FiveBufferEmitterPlan,
  admit_full_kernel_candidate, admit_full_kernel_candidate_set, capability_transport, derive_packed_weight_candidate,
  derive_q4k_q8_1_five_buffer_candidate,
  bind_full_kernel_candidate, full_kernel_candidate_set_from_legacy, full_kernel_candidate_capability,
  full_kernel_workload, q4k_q8_1_five_buffer_abi_plan, rebind_full_kernel_workload,
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
                     lowering_strategy="iu8_wmma_grouped_dot", target={"backend":"AMD", "arch":"gfx1100", "wave_size":32},
                     admission=CandidateAdmissionFacts(scheduler_owned=True))
  assert select_generated_candidate(op).status == "blocked"
  selected = select_generated_candidate(op, allow_research=True)
  assert selected.status == "selected"
  assert selected.candidate and selected.candidate.candidate_id == "quant_linear_prefill.q4k_int8_wmma_tensor_substrate"
  blocked = builtin_registry().select(RuntimeOpSpec("ActivationFusion", "prefill", "unknown", {},
                                                    QuantizedTensorSpec("unknown"), ActivationQuantSpec("none")))
  assert blocked.status == "blocked"
  assert blocked.candidate is None


def test_builtin_registry_selects_wmma_tiled_candidate():
  op = RuntimeOpSpec("QuantizedLinear", "prefill", "ffn_down", {"M": 512, "N": 5120, "K": 17408},
                     quant_spec("Q4_K").tensor_spec(), activation_spec("Q8_1").activation_spec(),
                     lowering_strategy="iu8_wmma_tiled_grouped_dot", target={"backend":"AMD", "arch":"gfx1100", "wave_size":32},
                     admission=CandidateAdmissionFacts(scheduler_owned=True))
  selected = select_generated_candidate(op, preferred=("quant_linear_prefill.q4k_int8_wmma_tiled_substrate",), allow_research=True)
  assert selected.status == "selected"
  assert selected.candidate and selected.candidate.route_id == "prefill_q4k_int8_wmma_tiled_research"


def test_primitive_selection_uses_target_shape_and_typed_admission_facts():
  base = dict(family="QuantizedLinear", phase="prefill", role="ffn_down", shape={"M":512, "N":5120, "K":17408},
              weight=quant_spec("Q6_K").tensor_spec(), activation=activation_spec("fp16").activation_spec(),
              target={"backend":"AMD", "arch":"gfx1100", "wave_size":32})
  fused = select_generated_candidate(RuntimeOpSpec(**base, lowering_strategy="fused_dequant_wmma",
    admission=CandidateAdmissionFacts(scheduler_owned=True, fused_wmma_admitted=True)))
  assert fused.status == "blocked", "a deferred emitter must never be selected from caller-supplied admission facts"
  once = select_generated_candidate(RuntimeOpSpec(**base, lowering_strategy="dequant_once_matmul",
    admission=CandidateAdmissionFacts(memory_budget_bytes=2_000_000, dequant_buffer_bytes=1_000_000,
                                      scheduler_owned=True, dequant_once_admitted=True)), allow_research=True)
  assert once.candidate and once.candidate.candidate_id == "quant_linear_prefill.q6k_dequant_once"
  assert select_generated_candidate(RuntimeOpSpec(**base, lowering_strategy="dequant_once_matmul",
    admission=CandidateAdmissionFacts(memory_budget_bytes=999_999, dequant_buffer_bytes=1_000_000,
                                      scheduler_owned=True, dequant_once_admitted=True)), allow_research=True).status == "blocked"


def test_shape_constraints_are_alternatives_not_conjunctions():
  candidate = GeneratedCandidate("c.shapes", "QuantizedLinear", ("Q6_K",), ("fp16",), ("prefill",), ("ffn_down",),
    "dequant_once_matmul", "machine_authored_generated",
    shape_constraints=({"M": 256, "N": 1024, "K": 4096}, {"M": 512, "N": 5120, "K": 17408}))
  op = RuntimeOpSpec("QuantizedLinear", "prefill", "ffn_down", {"M": 512, "N": 5120, "K": 17408},
                     quant_spec("Q6_K").tensor_spec(), activation_spec("fp16").activation_spec(),
                     lowering_strategy="dequant_once_matmul")
  assert candidate.supports(op)


def test_direct_packed_is_explicit_lower_priority_rollback_only():
  op = RuntimeOpSpec("QuantizedLinear", "prefill", "ffn_down", {"M":512, "N":5120, "K":17408},
                     quant_spec("Q6_K").tensor_spec(), activation_spec("fp16").activation_spec(),
                     lowering_strategy="packed_dequant_dot")
  assert select_generated_candidate(op).status == "blocked"
  selected = select_generated_candidate(op, allow_rollback=True)
  assert selected.candidate and selected.candidate.candidate_class == "rollback" and selected.candidate.priority == 0


def test_builtin_generated_candidates_match_manifest_route_metadata():
  for candidate in builtin_registry().all():
    assert candidate.route_id in route_manifest.ROUTES
    manifest = route_manifest.ROUTES[candidate.route_id]
    assert candidate.provenance == route_manifest.route_provenance(candidate.route_id)
    assert candidate.supported_quant_formats == tuple(manifest["quant"])
    assert candidate.authority_gates == _manifest_authority_gates(candidate.route_id)
    if candidate.is_full_kernel_candidate:
      assert set(candidate.roles).issubset(set(_manifest_runtime_roles(candidate.route_id)))
    else:
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
  assert row["canonical_identity"] == "f6f7b6cc27a670010416805d6dba95bdd35c08549b95073ffdc4374303c58cb0"
  context = candidate.kernel_candidate_context()
  assert (context.schema_version, context.canonical_identity) == (FULL_KERNEL_CANDIDATE_SCHEMA, row["canonical_identity"])
  assert GeneratedCandidate.from_json(row) == candidate
  row["full_kernel_candidate"]["workload"]["shape"]["m"] = 1024
  with pytest.raises(ValueError, match="canonical_identity"):
    GeneratedCandidate.from_json(row)


def _operand_sources_b(kind="dense", quant_format="Q4_K"):
  a = {"kind":"dense", "logical_dtype":"fp16", "storage_dtype":"fp16", "abi_slot":1}
  if kind == "dense": b = {"kind":"dense", "logical_dtype":"fp16", "storage_dtype":"fp16", "abi_slot":2}
  else:
    b = {"kind":"packed_scalar_decoder", "logical_dtype":"fp16",
         "storage_dtype":"uint32" if quant_format == "Q4_K" else "uint16", "abi_slot":2,
         "quant_format":quant_format, "rows":12288, "k":4096, "block_elems":256,
         "block_bytes":144 if quant_format == "Q4_K" else 210,
         "decoder_version":PACKED_SCALAR_DECODER_VERSION}
  return {"a":a, "b":b}


@pytest.mark.parametrize("kind,quant_format", (("dense", "Q4_K"), ("packed_scalar_decoder", "Q4_K"),
                                                ("packed_scalar_decoder", "Q6_K")))
def test_full_kernel_operand_sources_roundtrip_and_typed_context(kind, quant_format):
  payload = _single_buffer_anchor_candidate().full_kernel_candidate
  assert payload is not None
  payload["operand_sources"] = _operand_sources_b(kind, quant_format)
  candidate = _strict_full_kernel_candidate(full_kernel_candidate=payload)
  assert GeneratedCandidate.from_json(candidate.to_json()) == candidate
  admission = admit_full_kernel_candidate(payload, candidate.canonical_identity,
    profile="qwen3_8b_q4k_m_gfx1100", role="ffn_gate_up", shape=(512,12288,4096),
    target={"backend":"AMD","arch":"gfx1100","wave_size":32})
  if kind == "dense": assert admission.context.packed_weight is None
  else:
    assert admission.context.packed_weight.quant_format == quant_format
    assert (admission.context.packed_weight.rows, admission.context.packed_weight.k) == (12288,4096)


@pytest.mark.parametrize("quant_format,storage_dtype,block_bytes", (("Q4_K", "uint32", 144), ("Q6_K", "uint16", 210)))
def test_derive_packed_weight_candidate_is_canonical_and_geometry_owned(quant_format, storage_dtype, block_bytes):
  payload = _single_buffer_anchor_candidate().full_kernel_candidate
  assert payload is not None and "operand_sources" not in payload
  entry = derive_packed_weight_candidate(payload, quant_format)
  assert "operand_sources" not in payload
  b = entry.payload["operand_sources"]["b"]
  assert (b["quant_format"], b["storage_dtype"], b["rows"], b["k"], b["block_bytes"]) == \
         (quant_format, storage_dtype, 12288, 4096, block_bytes)
  assert FullKernelCandidateSetEntry(entry.canonical_identity, entry.to_json()["payload"]) == entry

def test_q4k_q8_1_five_buffer_candidate_cpu_roundtrip_and_context():
  payload = _single_buffer_anchor_candidate().full_kernel_candidate
  assert payload is not None
  entry = derive_q4k_q8_1_five_buffer_candidate(payload)
  abi = entry.payload["kernel_abi"]
  assert entry.to_json()["payload"]["kernel_abi"] == q4k_q8_1_five_buffer_abi_plan()
  assert [(x["abi_slot"], x["storage_dtype"]) for x in abi["buffers"].values()] == \
         [(0,"float32"),(1,"uint32"),(2,"int8"),(3,"float32"),(4,"float32")]
  assert abi["buffers"]["q8_ds4_values"]["signed"] is True
  assert {name:row["logical_axes"] for name,row in abi["buffers"].items()} == {
    "output":("m","n"),"q4_packed_words":("n","q4_blocks","q4_words"),
    "q8_ds4_values":("ds4_blocks","m","ds4_block_elems"),
    "q8_scales":("ds4_blocks","m","q8_groups_per_ds4_block"),
    "q8_weighted_sums":("ds4_blocks","m","q8_groups_per_ds4_block")}
  assert {name:row["access"] for name,row in abi["buffers"].items()} == {
    "output":"logical","q4_packed_words":"flat","q8_ds4_values":"flat","q8_scales":"flat","q8_weighted_sums":"flat"}
  assert FullKernelCandidateSetEntry(entry.canonical_identity, entry.to_json()["payload"]) == entry
  admission = admit_full_kernel_candidate(entry.payload, entry.canonical_identity,
    profile="qwen3_8b_q4k_m_gfx1100", role="ffn_gate_up", shape=(512,12288,4096),
    target={"backend":"AMD","arch":"gfx1100","wave_size":32})
  assert admission.capability is GFX1100_Q4K_Q8_FIVE_BUFFER_CAPABILITY
  assert capability_transport(admission.capability) == "direct_global"
  assert admission.active_lds_bytes == 0 and admission.geometry is None
  assert admission.plan == admission.pipeline_plan == admission.context.pipeline == Q4KQ8FiveBufferEmitterPlan()
  assert admission.context.geometry is None
  assert admission.context.packed_weight is None and admission.context.packed_operand_b is None
  assert admission.operand_plan == abi
  assert entry.payload["workload"]["dtypes"] == {"a":"Q8_1","b":"Q4_K","c":"fp32","accumulator":"int32_fp32"}
  assert entry.payload["workload"]["layout"] == {"a":"physical_ds4","b":"q4_k_packed_words","c":"tokens_rows"}
  assert "fp16" not in json.dumps(entry.payload)
  schedule = entry.payload["schedule"]
  assert (schedule["tile"], schedule["waves"], schedule["threads"]) == \
         ({"m":16,"n":16,"k":256},{"m":1,"n":1},32)
  assert schedule["variant"] == "q4k_q8_1_physical_ds4_direct_v1"
  assert schedule["transport"] == "direct_global" and schedule["lds_bytes"] == 0
  assert schedule["pipeline"] == {"buffer_count":0,"stage_count":0}
  assert schedule["tail_policy"] == "aligned_only_no_tails"
  assert schedule["compile_environment"] == {"REGALLOC_ADDR_REMAT":1}
  assert schedule["operands"]["q8_ds4_values"] == {"source":"global","alignment":16,"signed":True}
  assert schedule["wmma"]["instruction_family"] == "wmma_i32_16x16x16_iu8"
  assert schedule["wmma"]["instruction_family"] == admission.capability.instruction_family == admission.plan.instruction_family
  assert schedule["wmma"]["fragment_layout"] == admission.capability.fragment_layout
  assert schedule["threads"] == admission.plan.threads == admission.capability.wave_size
  assert entry.payload["static_constraints"]["max_lds_bytes"] == 0
  with pytest.raises(TypeError, match="immutable"): admission.operand_plan["quant_format"] = "Q6_K"
  with pytest.raises(ValueError, match="typed direct-global capability"):
    admit_full_kernel_candidate(entry.payload, entry.canonical_identity, profile="unused", role="ffn_gate_up",
      shape=(512,12288,4096), target={"backend":"AMD","arch":"gfx1100","wave_size":32},
      capability=GFX1100_TWO_BUFFER_STAGE1_CAPABILITY)

def test_q4k_q8_1_five_buffer_direct_schedule_is_canonical_and_identity_bound():
  base = _single_buffer_anchor_candidate().full_kernel_candidate
  entry = derive_q4k_q8_1_five_buffer_candidate(base)
  inherited = json.loads(json.dumps(base)); inherited["schedule"]["threads"] = 128
  assert derive_q4k_q8_1_five_buffer_candidate(inherited) == entry
  mutated = entry.to_json()["payload"]
  mutated["workload"]["shape"]["m"] *= 2
  rebound = derive_q4k_q8_1_five_buffer_candidate(mutated)
  assert rebound.canonical_identity != entry.canonical_identity
  with pytest.raises(ValueError, match="identity_mismatch"):
    admit_full_kernel_candidate(rebound.payload, entry.canonical_identity, profile="unused", role="ffn_gate_up",
      shape=(1024,12288,4096), target={"backend":"AMD","arch":"gfx1100","wave_size":32})
  schedule_drift = entry.to_json()["payload"]
  schedule_drift["schedule"]["threads"] = 64
  with pytest.raises(ValueError, match="five-buffer schedule"):
    _strict_full_kernel_candidate(full_kernel_candidate=schedule_drift)

@pytest.mark.parametrize("mutation", (
  lambda a: a["buffers"]["output"].update(abi_slot=1),
  lambda a: a["buffers"]["q4_packed_words"].update(storage_dtype="uint16"),
  lambda a: a["buffers"]["q8_ds4_values"].update(signed=False),
  lambda a: a["buffers"]["q8_scales"].update(abi_slot=4),
  lambda a: a["buffers"]["q8_weighted_sums"].update(storage_dtype="float16"),
  lambda a: a.update(quant_format="Q6_K"),
  lambda a: a.update(activation_layout="dense"),
  lambda a: a["block_geometry"].update(q8_ds4_block_elems=32),
  lambda a: a.update(output_layout="packed"),
  lambda a: a.update(emitter_family="q6k_packed"),
))
def test_q4k_q8_1_five_buffer_candidate_rejects_tamper_and_q6(mutation):
  entry = derive_q4k_q8_1_five_buffer_candidate(_single_buffer_anchor_candidate().full_kernel_candidate)
  payload = entry.to_json()["payload"]
  mutation(payload["kernel_abi"])
  with pytest.raises(ValueError, match="kernel_abi"):
    _strict_full_kernel_candidate(full_kernel_candidate=payload)

def test_q4k_q8_1_five_buffer_rejects_dense_or_packed_b_ambiguity_and_preserves_legacy():
  payload = _single_buffer_anchor_candidate().full_kernel_candidate
  entry = derive_q4k_q8_1_five_buffer_candidate(payload)
  ambiguous = entry.to_json()["payload"]
  ambiguous["operand_sources"] = _operand_sources_b("dense")
  with pytest.raises(ValueError, match="cannot combine"):
    _strict_full_kernel_candidate(full_kernel_candidate=ambiguous)
  assert GeneratedCandidate.from_json(_single_buffer_anchor_candidate().to_json()) == _single_buffer_anchor_candidate()


def test_schedule_template_rebinds_to_typed_workload_without_model_logic():
  payload = _single_buffer_anchor_candidate().full_kernel_candidate
  assert payload is not None
  rebound = rebind_full_kernel_workload(payload, profile="qwen3_14b_q4k_m_gfx1100", role="ffn_gate_up",
                                        shape=(512,17408,5120))
  workload = full_kernel_workload(rebound.payload)
  assert (workload.profile, workload.role, workload.shape) == \
         ("qwen3_14b_q4k_m_gfx1100", "ffn_gate_up", (512,17408,5120))
  assert rebound.payload["applicability"] == {"exact_shape":True, "profiles":(workload.profile,),
    "roles":(workload.role,), "targets":(workload.target_id,)}
  assert "operand_sources" not in rebound.payload
  assert full_kernel_candidate_capability(rebound.payload).capability_id.endswith("single_buffer.v1")


@pytest.mark.parametrize("mutation,error", (
  (lambda s: s["a"].update(abi_slot=2), "operand_sources.a"),
  (lambda s: s["b"].update(logical_dtype="uint32"), "logical_dtype"),
  (lambda s: s["b"].update(storage_dtype="uint16"), "storage_dtype"),
  (lambda s: s["b"].update(rows=12287), "rows/k"),
  (lambda s: s["b"].update(k=3840), "rows/k"),
  (lambda s: s["b"].update(block_bytes=210), "block_bytes"),
  (lambda s: s["b"].update(decoder_version="v2"), "decoder_version"),
  (lambda s: s["b"].update(abi_slot=3), "ABI slot 2"),
  (lambda s: s["b"].update(extra=True), "unknown fields"),
  (lambda s: s["b"].update(kind="mystery"), "kind"),
))
def test_full_kernel_packed_operand_sources_reject_mismatch_and_unknown_keys(mutation, error):
  payload = _single_buffer_anchor_candidate().full_kernel_candidate
  assert payload is not None
  payload["operand_sources"] = _operand_sources_b("packed_scalar_decoder", "Q4_K")
  mutation(payload["operand_sources"])
  with pytest.raises(ValueError, match=error): _strict_full_kernel_candidate(full_kernel_candidate=payload)


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


def test_two_buffer_stage1_requires_separate_capability_and_typed_plan():
  payload = _single_buffer_anchor_candidate().full_kernel_candidate
  assert payload is not None
  payload["schedule"]["pipeline"]["buffer_count"] = 2
  candidate = _strict_full_kernel_candidate(full_kernel_candidate=payload)
  kwargs = dict(profile="qwen3_8b_q4k_m_gfx1100", role="ffn_gate_up", shape=(512,12288,4096),
                target={"backend":"AMD","arch":"gfx1100","wave_size":32})
  with pytest.raises(ValueError, match="capability_pipeline"):
    admit_full_kernel_candidate(payload, candidate.canonical_identity, **kwargs)
  admission = admit_full_kernel_candidate(payload, candidate.canonical_identity,
    capability=GFX1100_TWO_BUFFER_STAGE1_CAPABILITY, **kwargs)
  assert admission.capability.capability_id == "amd.gfx1100.prefill.wmma_lds.two_buffer_stage1.v1"
  assert admission.active_lds_bytes == 40960
  assert admission.pipeline_plan.slot_window(0) == (0,20480)
  assert admission.pipeline_plan.slot_window(1) == (20480,40960)
  assert admission.context.pipeline == admission.pipeline_plan


def test_register_candidate_admission_uses_zero_lds_typed_plan():
  payload = json.loads(json.dumps(_single_buffer_anchor_candidate().full_kernel_candidate))
  payload["workload"]["role"] = "attn_qo"
  payload["workload"]["shape"] = {"m": 512, "n": 4096, "k": 4096}
  payload["applicability"]["roles"] = ["attn_qo"]
  payload["schedule"]["pipeline"].update(buffer_count=1, stage_count=2)
  payload["schedule"]["residency"]["resident"] = ["accumulator", "stage_ab_register"]
  payload["schedule"]["wmma"]["fragment_layout"] = "rdna3_wmma_f32_16x16x16_f16_register_static"
  for operand in ("a", "b"): payload["schedule"]["cooperative_load"][operand]["lane_mapping"] = "wave_contiguous_b128"
  candidate = _strict_full_kernel_candidate(full_kernel_candidate=payload)
  admission = admit_full_kernel_candidate(payload, candidate.canonical_identity,
    profile="qwen3_8b_q4k_m_gfx1100", role="attn_qo", shape=(512,4096,4096),
    target={"backend":"AMD","arch":"gfx1100","wave_size":32})
  assert admission.active_lds_bytes == 0
  assert admission.pipeline_plan.storage.kind == "global_register_resident"
  assert admission.pipeline_plan.stages == 2
  assert admission.pipeline_plan.pipeline_policy.storage_kind == "global_register_resident"
  assert admission.pipeline_plan.wait_coverage.passed
  assert admission.context.pipeline == admission.pipeline_plan


def test_register_candidate_route_selects_typed_install_without_lds(monkeypatch):
  payload = json.loads(json.dumps(_single_buffer_anchor_candidate().full_kernel_candidate))
  payload["workload"]["role"] = "attn_qo"
  payload["workload"]["shape"] = {"m": 512, "n": 4096, "k": 4096}
  payload["applicability"]["roles"] = ["attn_qo"]
  payload["schedule"]["pipeline"].update(buffer_count=1, stage_count=2)
  payload["schedule"]["residency"]["resident"] = ["accumulator", "stage_ab_register"]
  payload["schedule"]["wmma"]["fragment_layout"] = "rdna3_wmma_f32_16x16x16_f16_register_static"
  for operand in ("a", "b"): payload["schedule"]["cooperative_load"][operand]["lane_mapping"] = "wave_contiguous_b128"
  candidate = _strict_full_kernel_candidate(full_kernel_candidate=payload)
  candidate_config = {"BOLTBEAM_FULL_KERNEL_CANDIDATE_JSON": json.dumps(payload),
                      "BOLTBEAM_FULL_KERNEL_CANDIDATE_HASH": candidate.canonical_identity}
  seen = {}

  def capture(_x, _w, _out_f, _in_f, admission, _compile_artifact):
    seen["storage"] = admission.context.pipeline.storage.kind
    seen["lds"] = admission.active_lds_bytes
    return "register_install"

  monkeypatch.setattr(prefill_graph_gemm_route, "_install_candidate_matmul", capture)
  class Fake:
    ndim = 3
    shape = (1, 512, 4096)
  class FakeWeight:
    shape = (4096, 4096)
  class Lin: pass
  lin = Lin(); lin._pf16_w = FakeWeight(); lin.bias = None; lin._prefill_graph_role = "attn_qo"
  registry = prefill_graph_gemm_route._candidate_registry_from_env(candidate_config)
  admission = registry.admissions[0]
  workload = admission.normalized_payload["workload"]
  set_identity = route_manifest.canonical_candidate_set_identity(registry.candidate_set.to_json())
  inventory_identity = "inventory:sha256:" + "a"*64
  lin._prefill_graph_gemm_binding = {"candidate_registry": registry, "inventory_identity": inventory_identity,
    "candidate_set_identity": set_identity, "scanned_target_facts": {"target": workload["target"]},
    "selected_policy": {"role": workload["role"], "shape": workload["shape"], "target": workload["target"],
      "inventory_identity": inventory_identity, "candidate_set_identity": set_identity,
      "candidate_identity": admission.canonical_identity}}
  assert prefill_graph_gemm_route.route_pf16_graph_gemm(lin, Fake()) == "register_install"
  assert seen == {"storage": "global_register_resident", "lds": 0}


@pytest.mark.parametrize(("field", "value", "error"), (
  ("canonical_identity", "0" * 64, "identity_mismatch"),
  ("shape", (1024, 12288, 4096), "shape"),
  ("target", {"backend": "AMD", "arch": "gfx1200", "wave_size": 32}, "target"),
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
  (lambda p: p["schedule"]["pipeline"].update(stage_count=2), "capability_pipeline"),
  (lambda p: p["schedule"]["lds"].update(windows={"a": [0, 16384], "b": [16384, 40960]}), "capability_geometry"),
  (lambda p: p["schedule"]["lds"].update(strides={"a": 144, "b": 144}), "capability_geometry"),
  (lambda p: p["schedule"]["cooperative_load"]["a"].update(vector_width=4), "capability_vector"),
  (lambda p: p["schedule"]["wmma"].update(fragment_layout="gfx11"), "capability_tc"),
  (lambda p: p["static_constraints"].update(max_lds_bytes=16384), "capability_lds"),
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
  env = {"BOLTBEAM_FULL_KERNEL_CANDIDATE_JSON": json.dumps(candidate.full_kernel_candidate),
         "BOLTBEAM_FULL_KERNEL_CANDIDATE_HASH": candidate.canonical_identity}
  row = {x["family"]: x for x in pure_search_guard.effective_routes(env)}["prefill_gemm"]
  assert row["effective_route"] == "prefill_wmma_lds_dbuf_generated"
  assert row["provenance"] == "tinygrad_scheduler_generated"
  assert row["strict_pure"] is True and row["pure"] is True and row["rolled_back_to_oracle"] is False


def test_dynamic_second_supported_candidate_admits_and_joins_route_identity():
  payload = _single_buffer_anchor_candidate().full_kernel_candidate
  assert payload is not None
  payload["schedule"]["tile"] = {"m": 64, "n": 64, "k": 32}
  payload["schedule"]["waves"] = {"m": 2, "n": 2}
  payload["schedule"]["threads"] = 128
  payload["schedule"]["lds"]["windows"] = {"a": [0, 5120], "b": [5120, 10240]}
  payload["schedule"]["lds"]["strides"] = {"a": 80, "b": 80}
  candidate = _strict_full_kernel_candidate(full_kernel_candidate=payload)
  admission = admit_full_kernel_candidate(payload, candidate.canonical_identity, profile="qwen3_8b_q4k_m_gfx1100",
    role="ffn_gate_up", shape=(512,12288,4096), target={"backend":"AMD","arch":"gfx1100","wave_size":32})
  assert admission.geometry.tile == (64,64,32) and admission.active_lds_bytes == 10240
  assert (admission.plan.subtiles_m, admission.plan.subtiles_n, admission.plan.k_substeps) == (2,2,2)
  assert admission.canonical_identity != ANCHOR_SINGLE_BUFFER_CANDIDATE_HASH
  env = {"BOLTBEAM_FULL_KERNEL_CANDIDATE_JSON":json.dumps(payload),
    "BOLTBEAM_FULL_KERNEL_CANDIDATE_HASH":candidate.canonical_identity}
  row = {x["family"]:x for x in pure_search_guard.effective_routes(env)}["prefill_gemm"]
  assert row["candidate_identity"] == candidate.canonical_identity


def test_candidate_route_selector_fails_closed_and_absent_artifact_is_scheduler_fallback():
  default = {x["family"]: x for x in pure_search_guard.effective_routes({})}["prefill_gemm"]
  assert default["effective_route"] == "prefill_v2_scheduler_matmul_default"
  assert "candidate_set_identities" not in default
  candidate = _single_buffer_anchor_candidate()
  base = {"BOLTBEAM_FULL_KERNEL_CANDIDATE_JSON": json.dumps(candidate.full_kernel_candidate)}
  with pytest.raises(ValueError, match="provided together"): pure_search_guard.effective_routes(base)
  with pytest.raises(ValueError, match="identity_mismatch"):
    pure_search_guard.effective_routes({**base, "BOLTBEAM_FULL_KERNEL_CANDIDATE_HASH": "0" * 64})

def _buffer2_set_entry(role,shape):
  payload=json.loads(json.dumps(_single_buffer_anchor_candidate().full_kernel_candidate))
  payload["workload"]["role"]=role; payload["workload"]["shape"]=dict(zip(("m","n","k"),shape))
  payload["applicability"]["roles"]=[role]; payload["schedule"]["pipeline"]["buffer_count"]=2
  identity=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode("ascii")).hexdigest()
  return FullKernelCandidateSetEntry(identity,payload)

def test_four_role_8b_candidate_set_admits_and_exactly_indexes():
  entries=tuple(_buffer2_set_entry(role,shape) for role,shape in (
    ("ffn_gate_up",(512,12288,4096)),("ffn_down",(512,4096,12288)),
    ("attn_qo",(512,4096,4096)),("attn_kv",(512,1024,4096))))
  registry=admit_full_kernel_candidate_set(FullKernelCandidateSet(entries))
  target={"backend":"AMD","arch":"gfx1100","wave_size":32}
  assert len(registry.exact_index) == 4
  for entry in entries:
    role,m,n,k,*_=entry.exact_key
    admission=registry.get(role,(m,n,k),target)
    assert admission is not None and admission.canonical_identity == entry.canonical_identity
    assert admission.active_lds_bytes == 40960
  assert registry.get("attn_kv",(512,2048,4096),target) is None
  assert registry.legacy_get("renamed-profile", "ffn_gate_up", (512,12288,4096), target) is not None
  with pytest.raises(TypeError,match="immutable"): entries[0].payload["workload"]["role"]="other"

def test_frozen_candidate_payload_round_trips_across_guarded_spawn():
  entry = _buffer2_set_entry("attn_qo", (512,4096,4096))
  restored = pickle.loads(pickle.dumps(entry.payload))
  assert restored == entry.payload and restored is not entry.payload
  with pytest.raises(TypeError, match="immutable"): restored["workload"]["role"] = "other"

def test_profile_rename_preserves_semantic_identity_and_duplicate_admission_fails_closed():
  entry=_buffer2_set_entry("attn_qo",(512,4096,4096))
  renamed=entry.to_json()["payload"]
  renamed["workload"]["profile"]="same-content-new-evidence-name"
  renamed["applicability"]["profiles"]=["same-content-new-evidence-name"]
  legacy_identity=hashlib.sha256(json.dumps(renamed,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode("ascii")).hexdigest()
  renamed_entry=FullKernelCandidateSetEntry(legacy_identity,renamed)
  assert renamed_entry.canonical_identity == entry.canonical_identity
  assert renamed_entry.exact_key == entry.exact_key
  assert renamed_entry.legacy_exact_key != entry.legacy_exact_key
  with pytest.raises(ValueError,match="duplicate_exact_key"):
    admit_full_kernel_candidate_set(FullKernelCandidateSet((entry,renamed_entry)))

def test_candidate_set_rejects_duplicate_exact_key_identity_mismatch_and_weak_collision():
  entry=_buffer2_set_entry("attn_qo",(512,4096,4096))
  with pytest.raises(ValueError,match="duplicate_exact_key"):
    admit_full_kernel_candidate_set(FullKernelCandidateSet((entry,entry)))
  with pytest.raises(ValueError,match="identity_mismatch"):
    FullKernelCandidateSetEntry("0"*64,entry.payload)
  swapped=_buffer2_set_entry("ffn_down",(4096,512,4096))
  with pytest.raises(ValueError,match="warmstart_key_collision"):
    admit_full_kernel_candidate_set(FullKernelCandidateSet((entry,swapped)))

def test_one_entry_legacy_candidate_set_adapter_preserves_identity():
  entry=_buffer2_set_entry("ffn_gate_up",(512,12288,4096))
  candidate_set=full_kernel_candidate_set_from_legacy(entry.payload,entry.canonical_identity)
  assert candidate_set.entries == (entry,)
  assert FullKernelCandidateSet.from_json(candidate_set.to_json()) == candidate_set

def test_candidate_set_json_path_and_legacy_environment_loaders(tmp_path):
  entry=_buffer2_set_entry("attn_kv",(512,1024,4096)); candidate_set=FullKernelCandidateSet((entry,))
  text=json.dumps(candidate_set.to_json()); path=tmp_path/"candidates.json"; path.write_text(text)
  for env in ({"BOLTBEAM_FULL_KERNEL_CANDIDATE_SET_JSON":text},
              {"BOLTBEAM_FULL_KERNEL_CANDIDATE_SET_PATH":str(path)},
              {"BOLTBEAM_FULL_KERNEL_CANDIDATE_JSON":json.dumps(entry.payload),
               "BOLTBEAM_FULL_KERNEL_CANDIDATE_HASH":entry.canonical_identity}):
    registry=prefill_graph_gemm_route._candidate_registry_from_env(env)
    assert registry.admissions[0].canonical_identity == entry.canonical_identity
  with pytest.raises(ValueError,match="mutually exclusive"):
    prefill_graph_gemm_route._candidate_registry_from_env({"BOLTBEAM_FULL_KERNEL_CANDIDATE_SET_JSON":text,
      "BOLTBEAM_FULL_KERNEL_CANDIDATE_SET_PATH":str(path)})

def test_pure_guard_accepts_exact_non_gate_legacy_candidate():
  entry=_buffer2_set_entry("attn_qo",(512,4096,4096))
  env={"BOLTBEAM_FULL_KERNEL_CANDIDATE_JSON":json.dumps(entry.payload),
       "BOLTBEAM_FULL_KERNEL_CANDIDATE_HASH":entry.canonical_identity}
  assert pure_search_guard._prefill_candidate_selected(env)
