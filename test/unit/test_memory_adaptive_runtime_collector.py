import copy, json

from extra.qk.memory_adaptive_policy import SCHEMA, make_cache_record, select_policy
from extra.qk.memory_adaptive_runtime_collector import collect_runtime_policy, make_file_policy_collector, make_policy_collector
from extra.qk.memory_adaptive_allocation_observer import EXACT_MEMORY_KEYS, make_memory_facts


def fixture():
  inventory = {"schema": "tinygrad.model_runtime_prefill_inventory.v1", "inventory_identity": "content-id",
               "rows": [{"invocation_id": "a", "tensor_identity": "blk.0.a.weight"},
                        {"invocation_id": "b", "tensor_identity": "blk.0.b.weight"}]}
  device = {"device": "AMD:0", "architecture": "gfx", "free_vram_bytes": 123}
  request = {"schema": "tinygrad.model_memory_adaptive_request.v1", "inventory": inventory,
             "device_facts": device, "workload": {"prefill_ubatch": 32}}
  candidates = [
    {"candidate_id": "base:M32", "policy_candidate_id": "base", "strategy": "DIRECT_PACKED_FALLBACK",
     "routes": {"a": "q4-direct", "b": "q6-direct"},
     "workload_choice": {"candidate_id": "base", "full_m": 32, "feasible": True}},
    {"candidate_id": "fast:M32", "policy_candidate_id": "fast", "strategy": "FULL_RESIDENT_OVERLAY",
     "routes": {"a": "q4-overlay", "b": "q6-overlay"},
     "workload_choice": {"candidate_id": "fast", "full_m": 32, "feasible": True}},
  ]
  facts = {key: (1 if key in ("resident_copies", "batch_size", "kv_element_bytes") else 0) for key in EXACT_MEMORY_KEYS}
  provenance = {key: {"source": "runtime collector fixture", "detail": f"measured {key}"} for key in EXACT_MEMORY_KEYS}
  candidates[1]["memory_fact_evidence"] = make_memory_facts("fast:M32", facts, provenance)
  candidates[1]["memory_facts"] = facts
  proof = {"correctness": {"status": "PASS"}, "resource": {"status": "PASS"},
           "gpu_health": {"status": "PASS"}, "route_census": {"status": "PASS", "complete": True},
           "end_to_end_timing": {"scope": "end_to_end", "metric": "tok_s", "samples": [100, 100, 100]}}
  revision = {"compiler": "c1", "runtime": "r1"}
  result = select_policy(gpu_facts=device, model_facts={"facts": {"content_hash": "hash"}, "inventory": inventory},
    workload={"prompt_tokens": 32}, candidates=candidates, compiler_runtime_revision=revision,
    evidence={"base:M32": proof, "fast:M32": {**proof, "end_to_end_timing": {"scope": "end_to_end", "metric": "tok_s", "samples": [120]*3}}},
    baseline_candidate_id="base:M32")
  return request, result, make_cache_record(result), revision


def test_exact_cache_returns_selected_canonical_policy_and_preserves_routes():
  request, result, cache, revision = fixture()
  selected = collect_runtime_policy(request, cache, compiler_runtime_revision=revision, search_revision=SCHEMA)
  assert selected == {"decision": "SELECTED", "validation": "exact_cache", "validated_request": request,
                      "policy": next(x for x in result["canonical_inputs"]["candidates"] if x["candidate_id"] == result["selected_candidate_id"])}
  assert selected["policy"]["routes"] == {"a": "q4-overlay", "b": "q6-overlay"}


def test_completed_controller_output_is_accepted_but_partial_or_interrupted_is_not():
  request, result, cache, _ = fixture()
  controller = {"decision": "SELECTED", "selected_candidate_id": result["selected_candidate_id"],
                "interrupted": False, "from_cache": False, "policy": result, "cache_record": cache}
  assert collect_runtime_policy(request, controller)["validation"] == "measured"
  for mutation in ({"interrupted": True}, {"cache_record": None}, {"selected_candidate_id": "other"}):
    assert collect_runtime_policy(request, {**controller, **mutation}) is None


def test_every_runtime_and_search_identity_mismatch_fails_closed():
  request, _, cache, revision = fixture()
  mutations = []
  for section, key, value in (("inventory", "inventory_identity", "stale"), ("device_facts", "free_vram_bytes", 122),
                              ("workload", "prefill_ubatch", 16)):
    changed = copy.deepcopy(request); changed[section][key] = value; mutations.append(changed)
  assert all(collect_runtime_policy(x, cache) is None for x in mutations)
  assert collect_runtime_policy(request, cache, compiler_runtime_revision={**revision, "runtime": "r2"}) is None
  assert collect_runtime_policy(request, cache, search_revision="stale") is None
  assert collect_runtime_policy(request, cache, candidate_set_identity="sha256:stale") is None
  forged = copy.deepcopy(cache); forged["result"]["canonical_inputs"]["candidates"][0]["routes"]["a"] = "changed"
  assert collect_runtime_policy(request, forged) is None


def test_malformed_selected_policy_and_factories_fail_closed(tmp_path):
  request, _, cache, _ = fixture()
  assert make_policy_collector(cache)(request)["decision"] == "SELECTED"
  path = tmp_path / "policy.json"; path.write_text(json.dumps(cache))
  assert make_file_policy_collector(path)(request)["decision"] == "SELECTED"
  path.write_text("{")
  assert make_file_policy_collector(path)(request) is None
  for mutate in (lambda x: x["result"].update(decision="REFUSE"),
                 lambda x: x["result"].pop("accepted_candidates"),
                 lambda x: x["result"]["canonical_inputs"]["candidates"][1]["routes"].pop("b")):
    bad = copy.deepcopy(cache); mutate(bad)
    assert collect_runtime_policy(request, bad) is None


def test_accelerated_cache_cannot_omit_or_forge_memory_evidence():
  request, _, cache, _ = fixture()
  for mutate in (lambda x: x["result"]["canonical_inputs"]["candidates"][1].pop("memory_facts"),
                 lambda x: x["result"]["canonical_inputs"]["candidates"][1]["memory_facts"].update(batch_size=2),
                 lambda x: x["result"]["canonical_inputs"]["candidates"][1]["memory_fact_evidence"]["facts"].update(batch_size=2)):
    bad = copy.deepcopy(cache); mutate(bad)
    assert collect_runtime_policy(request, bad) is None
