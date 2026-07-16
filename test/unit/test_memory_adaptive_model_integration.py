import inspect
from types import SimpleNamespace

import pytest

import tinygrad.llm.model as model_module
from tinygrad.llm.device_facts import DeviceCapabilities, DeviceFacts, ProbeRecord
from extra.qk.memory_adaptive_allocation_observer import make_memory_facts
from extra.qk.memory_adaptive_runtime_collector import install_model_adapters
from tinygrad.llm.model import (Transformer, TransformerConfig, _graph_gemm_binding, _memory_adaptive_measurement_authority,
  derive_selected_gguf_prefill_inventory, select_memory_adaptive_runtime_policy)


def device_facts(free=24_000_000_000):
  probe = ProbeRecord("injected", "2026-07-15T00:00:00+00:00")
  return DeviceFacts("AMD", "AMD", "gfx1100", 24_000_000_000, free, DeviceCapabilities(wave_size=32), probe, probe)

@pytest.fixture(autouse=True)
def _active_production_adapters(): install_model_adapters()


def metadata():
  kv = {"general.architecture": "qwen3", "qwen3.embedding_length": 8, "qwen3.feed_forward_length": 16,
        "qwen3.attention.head_count": 2, "qwen3.attention.head_count_kv": 1, "qwen3.attention.key_length": 4}
  meta = {"tensor_infos": [("blk.0.ffn_gate.weight", (8, 16), 12, 0),
                            ("blk.0.ffn_down.weight", (16, 8), 14, 0)]}
  return kv, meta

def metadata_with_fixed_lm_head():
  kv, meta = metadata()
  return kv, {"tensor_infos": [*meta["tensor_infos"], ("output.weight", (8, 32), 1, 0)]}

def metadata_with_quantized_lm_head():
  kv, meta = metadata()
  return kv, {"tensor_infos": [*meta["tensor_infos"], ("output.weight", (8, 32), 14, 0)]}

def select_with_collector(kv, meta, collector, facts=None):
  facts = device_facts() if facts is None else facts
  inventory = derive_selected_gguf_prefill_inventory(kv, meta)
  with _memory_adaptive_measurement_authority(device_facts=facts, inventory=inventory,
      workload={"prefill_ubatch": 512}, collector=collector):
    return select_memory_adaptive_runtime_policy(kv=kv, meta=meta, device_facts=facts)


def exact_memory_facts():
  return {"resident_copies": 1, "candidate_workspace_bytes": 256, "batch_size": 1, "kv_element_bytes": 2,
          "runtime_persistent_bytes": 0, "peak_prefill_activation_bytes": 128,
          "peak_prefill_output_bytes": 64, "peak_prefill_scratch_bytes": 0,
          "provenance": "complete measured unit evidence"}


def exact_memory_bundle(candidate_id="measured-overlay"):
  facts = {k:v for k,v in exact_memory_facts().items() if k != "provenance"}
  provenance = {key:{"source":"unit measured allocator", "detail":key} for key in facts}
  return make_memory_facts(candidate_id, facts, provenance)


def test_inventory_is_derived_from_selected_gguf_content_not_name_or_profile():
  kv, meta = metadata()
  a = derive_selected_gguf_prefill_inventory(kv, meta)
  b = derive_selected_gguf_prefill_inventory({**kv, "general.name": "14B profile"}, meta)
  assert a == b
  assert {row["quant_format"] for row in a["rows"]} == {"Q4_K", "Q6_K"}

def test_inventory_includes_fixed_lm_head_with_truthful_geometry_and_route():
  kv, meta = metadata_with_fixed_lm_head()
  inventory = derive_selected_gguf_prefill_inventory(kv, meta)
  lm_head = next(row for row in inventory["rows"] if row["role"] == "lm_head")
  assert lm_head["candidate_controlled"] is False
  assert lm_head["fixed_route_id"] == "fixed-ggml-linear"
  assert lm_head["shape"] == {"m": 1, "n": 32, "k": 8}
  policy = select_memory_adaptive_runtime_policy(kv=kv, meta=meta, device_facts=device_facts())
  assert policy["routes"][lm_head["invocation_id"]] == lm_head["fixed_route_id"]

def test_quantized_lm_head_remains_fixed_final_token_work_not_pp512_candidate_work():
  kv, meta = metadata_with_quantized_lm_head()
  lm_head = next(row for row in derive_selected_gguf_prefill_inventory(kv, meta)["rows"] if row["role"] == "lm_head")
  assert lm_head["quant_format"] == "Q6_K" and lm_head["candidate_controlled"] is False
  assert lm_head["fixed_route_id"] == "fixed-ggml-linear"
  assert lm_head["shape"] == {"m":1, "n":32, "k":8}

def test_inventory_materializes_tied_embedding_as_fixed_runtime_lm_head():
  kv, _ = metadata()
  inventory = derive_selected_gguf_prefill_inventory(kv, {"tensor_infos": [("token_embd.weight", (8, 32), 14, 0)]})
  assert len(inventory["rows"]) == 1
  row = inventory["rows"][0]
  assert row["tensor_identity"] == "output.weight" and row["source_tensor_identity"] == "token_embd.weight"
  assert row["role"] == "lm_head" and row["candidate_controlled"] is False
  assert row["shape"] == {"m": 1, "n": 32, "k": 8}

def test_collector_cannot_change_fixed_inventory_route():
  kv, meta = metadata_with_fixed_lm_head()
  def collector(request):
    return {"decision": "SELECTED", "validation": "measured", "validated_request": request,
            "policy": {"strategy": "DIRECT_PACKED_FALLBACK", "candidate_id": "bad-fixed-route",
                       "routes": {row["invocation_id"]: "candidate-route" for row in request["inventory"]["rows"]}}}
  with pytest.raises(ValueError, match="changed a fixed"):
    select_with_collector(kv, meta, collector)


def test_no_collector_truthfully_binds_direct_packed_baseline():
  kv, meta = metadata()
  policy = select_memory_adaptive_runtime_policy(kv=kv, meta=meta, device_facts=device_facts())
  assert policy["strategy"] == "DIRECT_PACKED_FALLBACK"
  assert policy["measured"] is False
  assert set(policy["routes"]) == {x["invocation_id"] for x in derive_selected_gguf_prefill_inventory(kv, meta)["rows"]}
  with pytest.raises(TypeError): policy["strategy"] = "FULL_RESIDENT_OVERLAY"

def test_selected_model_source_invokes_internal_authority_then_exact_runtime_collector(monkeypatch):
  kv, meta = metadata()
  source = {"completed": "controller-envelope"}
  monkeypatch.setattr(model_module, "resolve_memory_adaptive_policy", lambda selected: source if selected == "/chosen.gguf" else None)
  import extra.qk.memory_adaptive_runtime_collector as runtime_collector
  def collect(request, observed):
    assert observed is source
    return {"decision": "SELECTED", "validation": "exact_cache", "validated_request": request,
            "policy": {"strategy": "DIRECT_PACKED_FALLBACK", "candidate_id": "cached-direct",
                       "routes": {row["invocation_id"]: "cached-direct" for row in request["inventory"]["rows"]}}}
  monkeypatch.setattr(runtime_collector, "collect_runtime_policy", collect)
  from tinygrad.llm.memory_adaptive_authority import register_memory_adaptive_adapters
  register_memory_adaptive_adapters(policy_adapter=collect)
  policy = select_memory_adaptive_runtime_policy(kv=kv, meta=meta, device_facts=device_facts(),
                                                  selected_model_source="/chosen.gguf")
  assert policy["candidate_id"] == "cached-direct"

def test_production_model_load_exposes_no_hardware_or_policy_authority_kwargs():
  parameters = inspect.signature(Transformer.from_gguf).parameters
  assert not {"policy_collector", "route", "device", "device_facts", "vram", "reserve", "reserve_policy"} & parameters.keys()


def test_injected_exact_policy_must_match_scanned_facts_and_inventory():
  kv, meta = metadata()
  def collector(request):
    ids = [x["invocation_id"] for x in request["inventory"]["rows"]]
    bundle = exact_memory_bundle()
    return {"decision": "SELECTED", "validation": "exact_cache", "validated_request": request,
            "policy": {"strategy": "FULL_RESIDENT_OVERLAY", "candidate_id": "measured-overlay",
                       "routes": {x: "measured-overlay" for x in ids}, "memory_facts": bundle["facts"],
                       "memory_fact_evidence": bundle}}
  assert select_with_collector(kv, meta, collector)["strategy"] == "FULL_RESIDENT_OVERLAY"
  def stale(request):
    result = collector(request); result["validated_request"] = {**request, "device_facts": {}}
    return result
  with pytest.raises(ValueError, match="exactly match"):
    select_with_collector(kv, meta, stale)


def test_accelerated_policy_cannot_omit_or_partially_supply_exact_memory_evidence():
  kv, meta = metadata()
  def collector(request, memory_facts=None):
    ids = [x["invocation_id"] for x in request["inventory"]["rows"]]
    policy = {"strategy": "FULL_RESIDENT_OVERLAY", "candidate_id": "measured-overlay",
              "routes": {x: "measured-overlay" for x in ids}}
    if memory_facts is not None: policy["memory_facts"] = memory_facts
    return {"decision": "SELECTED", "validation": "measured", "validated_request": request, "policy": policy}
  with pytest.raises(ValueError, match="complete measured memory_facts"):
    select_with_collector(kv, meta, collector)
  with pytest.raises(ValueError, match="not bound to complete measured evidence"):
    select_with_collector(kv, meta, lambda request: collector(request, {"resident_copies": 1}))


def test_accelerated_measurement_trial_is_context_scoped_and_not_normal_load_authority():
  kv, meta = metadata()
  def collector(request):
    ids = [x["invocation_id"] for x in request["inventory"]["rows"]]
    return {"decision": "SELECTED", "validation": "measurement_trial", "validated_request": request,
            "policy": {"strategy": "FULL_RESIDENT_OVERLAY", "candidate_id": "trial-overlay",
                       "routes": {x: "trial-overlay" for x in ids}}}
  assert select_memory_adaptive_runtime_policy(kv=kv, meta=meta, device_facts=device_facts())["strategy"] == "DIRECT_PACKED_FALLBACK"
  policy = select_with_collector(kv, meta, collector)
  assert policy["strategy"] == "FULL_RESIDENT_OVERLAY" and "memory_facts" not in policy


def test_selected_strategy_is_overlay_realization_authority():
  model = object.__new__(Transformer)
  model.config = SimpleNamespace(prefill_policy={"strategy": "DIRECT_PACKED_FALLBACK"})
  model._prefill_v2_covered = lambda: (_ for _ in ()).throw(AssertionError("must not inspect overlay tensors"))
  assert model.realize_prefill_v2_weights() == 0


def test_complete_policy_can_retain_per_invocation_mixed_route_ids():
  kv, meta = metadata()
  def collector(request):
    ids = [x["invocation_id"] for x in request["inventory"]["rows"]]
    return {"decision": "SELECTED", "validation": "measured", "validated_request": request,
            "policy": {"strategy": "DIRECT_PACKED_FALLBACK", "candidate_id": "complete-policy",
                       "routes": {ids[0]: "q4-direct", ids[1]: "q6-direct"}}}
  policy = select_with_collector(kv, meta, collector)
  assert set(policy["routes"].values()) == {"q4-direct", "q6-direct"}


def test_relevant_linears_receive_the_single_scanned_device_facts_object():
  facts = device_facts()
  weight = SimpleNamespace(shape=(16, 8))
  tr = object.__new__(Transformer)
  tr.config = SimpleNamespace(prefill_policy=None, prefill_device_facts=facts)
  tr._prefill_graph_gemm_registry = None
  tr.blk = [SimpleNamespace(ffn_gate=SimpleNamespace(weight=weight))]
  tr.output = None
  linear = next(tr._prefill_v2_covered())[0]
  assert linear._prefill_device_facts is facts

def test_transformer_owns_selected_graph_registry_used_for_linear_bindings(monkeypatch):
  registry = object()
  monkeypatch.setattr(model_module, "_graph_gemm_registry", lambda policy: registry)
  config = TransformerConfig(num_blocks=0, dim=8, hidden_dim=16, n_heads=2, n_kv_heads=1, norm_eps=1e-5,
    vocab_size=32, head_dim=4, rope_theta=10000, rope_dim=4, v_head_dim=4, max_context=32,
    prefill_policy={"graph_gemm":{}})
  tr = Transformer(config)
  assert tr._prefill_graph_gemm_registry is registry


def test_graph_binding_requires_one_exact_selected_policy_row():
  facts = device_facts()
  target = {"backend": "AMD", "arch": "gfx1100", "wave_size": 32}
  row = {"role": "ffn_gate_up", "shape": {"m": 512, "n": 16, "k": 8}, "target": target,
         "inventory_identity": "inventory-id", "candidate_set_identity": "set-id", "candidate_identity": "candidate-id"}
  policy = {"inventory_identity": "inventory-id", "graph_gemm": {"candidate_set_identity": "set-id", "policy_rows": [row]}}
  registry = object()
  binding = _graph_gemm_binding(policy, registry, "ffn_gate_up", (512, 16, 8), facts)
  assert binding is not None and binding["selected_policy"] is row and binding["candidate_registry"] is registry
  assert _graph_gemm_binding(policy, registry, "ffn_gate_up", (512, 16, 9), facts) is None
  assert _graph_gemm_binding({**policy, "inventory_identity": "stale"}, registry, "ffn_gate_up", (512, 16, 8), facts) is None
