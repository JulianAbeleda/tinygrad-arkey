import copy
import hashlib
import json

import pytest

from extra.qk.route_manifest import (canonical_candidate_set_identity, canonical_capability_identity,
  canonical_inventory_identity, canonical_policy_rows, canonical_route_id, lookup_policy_row,
  automatic_promoted_prefill_graph_policy, promoted_prefill_candidate_policy, route,
  immutable_route_registry)


TARGET = {"backend": "AMD", "arch": "gfx1100", "wave_size": 32}
CAPABILITY = {"target": TARGET, "phase": "prefill", "quant_formats": ["Q4_K"], "dtypes": {
  "input": "fp16", "output": "fp16", "accumulator": "fp32"}, "max_lds_bytes": 65536,
  "packed_abis": ["ggml_k_blocks:Q4_K:256:144"]}


def inventory(profile="renamed-model.gguf"):
  return {"schema": "test.inventory.v1", "profile": profile, "model_path": profile, "rows": [{
    "phase": "prefill", "role": "attn_qo", "quant_format": "Q4_K",
    "shape": {"m": 512, "n": 4096, "k": 4096}, "target": TARGET,
    "tensor_identities": ["blk.0.attn_q.weight"], "call_count": 1,
    "layout": {"packed": "ggml_k_blocks", "block_elems": 256, "block_bytes": 144},
  }]}


def candidate_set(profile="old_8b_benchmark_label"):
  return {"schema": "boltbeam.full_kernel_candidate_set.v1", "entries": [{
    "canonical_identity": "candidate-attn-qo", "payload": {
      "workload": {"profile": profile, "phase": "prefill", "role": "attn_qo", "quant_format": "Q4_K",
        "shape": {"m": 512, "n": 4096, "k": 4096}, "target": TARGET},
      "applicability": {"profiles": [profile], "exact_shape": True}, "schedule": {"tile": [128, 128, 32]},
    }}]}

def fallback(workload, route_id="direct_packed", evidence=None):
  evidence = evidence or {"schema":"qk.direct_packed_qualification.v1", "status":"qualified", "binary":"q6-baseline"}
  semantic = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
  evidence_identity = "fallback_evidence:sha256:" + hashlib.sha256(semantic(evidence)).hexdigest()
  content = {"workload":workload, "route_id":route_id, "evidence_identity":evidence_identity}
  return {**content, "evidence":evidence,
          "fallback_identity":"fallback:sha256:" + hashlib.sha256(semantic(content)).hexdigest()}


def test_model_and_profile_renames_are_provenance_only():
  inv_a, inv_b = inventory("qwen-size-label.gguf"), inventory("completely-renamed.gguf")
  set_a, set_b = candidate_set("known-profile"), candidate_set("unknown-profile")
  assert canonical_inventory_identity(inv_a) == canonical_inventory_identity(inv_b)
  assert canonical_candidate_set_identity(set_a) == canonical_candidate_set_identity(set_b)
  assert canonical_policy_rows(inv_a, CAPABILITY, set_a) == canonical_policy_rows(inv_b, CAPABILITY, set_b)


def test_profiles_are_provenance_only_and_legacy_artifact_is_readable():
  policy = promoted_prefill_candidate_policy()
  assert policy["candidate_profiles"] == policy["provenance_profiles"]
  assert policy["candidate_set_identity"].startswith("candidate_set:sha256:")
  assert policy["semantic_policy_rows"] == ()

def test_automatic_promoted_policy_binds_normalized_exact_runtime_inventory():
  shapes = {"attn_qo":(4096, 4096), "attn_kv":(1024, 4096),
            "ffn_down":(4096, 12288), "ffn_gate_up":(12288, 4096)}
  rows = []
  for i, (role, (n, k)) in enumerate(shapes.items()):
    rows.append({"invocation_id":f"inv-{i}", "tensor_identity":f"blk.0.{role}.weight", "role":role,
                 "quant_format":"Q4_K", "candidate_controlled":True, "shape":{"m":512, "n":n, "k":k}})
  rows.append({"invocation_id":"lm", "tensor_identity":"output.weight", "role":"lm_head", "quant_format":"Q6_K",
               "candidate_controlled":False, "fixed_route_id":"fixed-ggml-linear", "shape":{"m":1, "n":151936, "k":4096}})
  inv = {"schema":"runtime", "inventory_identity":"selected-content", "rows":rows}
  facts = {"backend":"AMD", "architecture":"gfx1100", "capabilities":{"wave_size":32}}
  policy = automatic_promoted_prefill_graph_policy(inv, facts)
  assert policy is not None and policy["strategy"] == "FULL_RESIDENT_OVERLAY"
  assert policy["routes"]["lm"] == "fixed-ggml-linear"
  assert set(policy["routes"].values()) == {"prefill_wmma_lds_dbuf_generated", "fixed-ggml-linear"}
  assert len(policy["graph_gemm"]["policy_rows"]) == 4
  normalized = policy["graph_gemm"]["candidate_set"]
  assert policy["graph_gemm"]["candidate_set_identity"] == canonical_candidate_set_identity(normalized)

def test_automatic_promoted_policy_rejects_wrong_shape_or_target_without_profile_logic():
  base = {"schema":"runtime", "inventory_identity":"selected-content", "rows":[
    {"invocation_id":"q", "tensor_identity":"blk.0.attn_q.weight", "role":"attn_qo", "quant_format":"Q4_K",
     "candidate_controlled":True, "shape":{"m":512, "n":4096, "k":4096}}]}
  assert automatic_promoted_prefill_graph_policy(base,
    {"backend":"AMD", "architecture":"gfx1200", "capabilities":{"wave_size":32}}) is None
  assert automatic_promoted_prefill_graph_policy(base,
    {"backend":"AMD", "architecture":"gfx1100", "capabilities":{"wave_size":32}}) is None


def test_route_alias_resolution_is_exact_and_preserves_legacy_reads():
  registry = immutable_route_registry()
  legacy = "decode_flash_live_split_g4_8b_kvboth"
  canonical = "decode_flash_live_split_g4_kvboth"
  assert canonical_route_id(legacy, registry) == canonical
  assert route(legacy, registry) is route(canonical, registry)
  for malformed in (legacy.upper(), legacy[:-1], f"prefix-{legacy}"):
    with pytest.raises(KeyError): canonical_route_id(malformed)


def test_exact_structural_lookup_and_route_id_alias():
  inv, cap, candidates = inventory(), CAPABILITY, candidate_set()
  rows = canonical_policy_rows(inv, cap, candidates)
  found = lookup_policy_row(rows, phase="prefill", role="attn_qo", quant="Q4_K",
    shape={"m": 512, "n": 4096, "k": 4096}, target=TARGET,
    capability_identity=canonical_capability_identity(cap),
    inventory_identity=canonical_inventory_identity(inv),
    candidate_set_identity=canonical_candidate_set_identity(candidates))
  assert found is not None
  assert found["candidate_identity"] == "candidate-attn-qo"
  assert found["selected_route"] in found["route_aliases"] == ["prefill_wmma_lds_dbuf_generated"]


def test_mixed_policy_covers_exact_inventory_with_identity_qualified_fallback():
  inv, candidates = inventory(), candidate_set()
  q6 = {"phase":"prefill", "role":"attn_kv", "quant_format":"Q6_K",
        "shape":{"m":512, "n":1024, "k":5120}, "target":TARGET}
  inv["rows"].append({**q6, "tensor_identities":["blk.0.attn_k.weight"], "call_count":1,
                      "layout":{"packed":"ggml_k_blocks", "block_elems":256, "block_bytes":210}})
  cap = {**CAPABILITY, "quant_formats":["Q4_K", "Q6_K"]}
  candidates["fallbacks"] = [fallback(q6)]
  rows = canonical_policy_rows(inv, cap, candidates)
  assert [row["binding_kind"] for row in rows] == ["candidate", "fallback"]
  assert rows[1]["selected_route"] == "direct_packed"
  assert rows[1]["fallback_identity"].startswith("fallback:sha256:")
  assert "candidate_identity" not in rows[1]


def test_exact_six_row_mixed_policy_has_one_binding_per_discovered_row():
  specs = (("ffn_gate_up", "Q4_K", 17408, 5120), ("attn_qo", "Q4_K", 5120, 5120),
           ("ffn_down", "Q4_K", 5120, 17408), ("attn_kv", "Q4_K", 1024, 5120),
           ("ffn_down", "Q6_K", 5120, 17408), ("attn_kv", "Q6_K", 1024, 5120))
  rows, entries, fallbacks = [], [], []
  for index, (role, quant, n, k) in enumerate(specs):
    workload = {"phase":"prefill", "role":role, "quant_format":quant,
                "shape":{"m":512, "n":n, "k":k}, "target":TARGET}
    rows.append({**workload, "tensor_identities":[f"tensor.{index}"], "call_count":1,
                 "layout":{"packed":"ggml_k_blocks", "block_elems":256,
                           "block_bytes":144 if quant == "Q4_K" else 210}})
    if quant == "Q4_K": entries.append({"canonical_identity":f"candidate-{index}", "payload":{"workload":workload}})
    else: fallbacks.append(fallback(workload))
  inv = {"schema":"test.inventory.v1", "rows":rows}
  candidates = {"schema":"boltbeam.full_kernel_candidate_set.v1", "entries":entries, "fallbacks":fallbacks}
  cap = {**CAPABILITY, "quant_formats":["Q4_K", "Q6_K"]}
  policy = canonical_policy_rows(inv, cap, candidates)
  assert len(policy) == 6
  assert [row["binding_kind"] for row in policy].count("candidate") == 4
  assert [row["binding_kind"] for row in policy].count("fallback") == 2
  assert all(row["selected_route"] == "direct_packed" for row in policy if row["quant"] == "Q6_K")


@pytest.mark.parametrize(("mutation", "match"), [
  (lambda row: row.pop("evidence"), "requires route and evidence"),
  (lambda row: row["evidence"].__setitem__("status", "measured"), "evidence is not qualified"),
  (lambda row: row.__setitem__("evidence_identity", "fallback_evidence:sha256:stale"), "evidence identity mismatch"),
  (lambda row: row.__setitem__("fallback_identity", "fallback:sha256:stale"), "fallback identity mismatch"),
])
def test_declared_fallback_rejects_missing_evidence_and_identity_drift(mutation, match):
  q6 = {"phase":"prefill", "role":"attn_kv", "quant_format":"Q6_K",
        "shape":{"m":512, "n":1024, "k":5120}, "target":TARGET}
  row = fallback(q6)
  mutation(row)
  with pytest.raises(ValueError, match=match):
    canonical_candidate_set_identity({"schema":"boltbeam.full_kernel_candidate_set.v1", "entries":[], "fallbacks":[row]})


def test_candidate_and_fallback_cannot_both_bind_one_inventory_row():
  candidates = candidate_set()
  workload = copy.deepcopy(candidates["entries"][0]["payload"]["workload"])
  workload.pop("profile")
  candidates["fallbacks"] = [fallback(workload)]
  with pytest.raises(ValueError, match="ambiguous candidate/fallback binding"):
    canonical_policy_rows(inventory(), CAPABILITY, candidates)


@pytest.mark.parametrize(("field", "value"), [
  ("phase", "decode"), ("role", "attn_kv"), ("quant", "Q6_K"),
  ("shape", {"m": 256, "n": 4096, "k": 4096}),
  ("target", {"backend": "AMD", "arch": "gfx1101", "wave_size": 32}),
  ("capability_identity", "capability:sha256:wrong"), ("inventory_identity", "inventory:sha256:wrong"),
  ("candidate_set_identity", "candidate_set:sha256:wrong"),
])
def test_lookup_rejects_every_semantic_mismatch(field, value):
  inv, candidates = inventory(), candidate_set()
  rows = canonical_policy_rows(inv, CAPABILITY, candidates)
  query = {"phase": "prefill", "role": "attn_qo", "quant": "Q4_K",
    "shape": {"m": 512, "n": 4096, "k": 4096}, "target": TARGET,
    "capability_identity": canonical_capability_identity(CAPABILITY),
    "inventory_identity": canonical_inventory_identity(inv),
    "candidate_set_identity": canonical_candidate_set_identity(candidates)}
  query[field] = value
  assert lookup_policy_row(rows, **query) is None


def test_candidate_binding_rejects_structural_mismatch():
  candidates = candidate_set()
  candidates = copy.deepcopy(candidates)
  candidates["entries"][0]["payload"]["workload"]["shape"]["n"] = 1024
  with pytest.raises(ValueError, match="does not cover exact inventory row"):
    canonical_policy_rows(inventory(), CAPABILITY, candidates)


def test_recorded_inventory_identity_mismatch_is_rejected():
  inv = inventory()
  inv["inventory_identity"] = "inventory:sha256:stale"
  with pytest.raises(ValueError, match="inventory identity mismatch"):
    canonical_inventory_identity(inv)


def test_scanned_capability_target_must_match_inventory_and_candidate_target():
  scanned = copy.deepcopy(CAPABILITY)
  scanned["target"]["arch"] = "gfx1101"
  with pytest.raises(ValueError, match="inventory target does not match scanned capability target"):
    canonical_policy_rows(inventory(), scanned, candidate_set())


def test_scanned_resource_facts_are_material_but_vram_labels_are_not():
  a, b = copy.deepcopy(CAPABILITY), copy.deepcopy(CAPABILITY)
  a["device_memory"] = {"total_bytes": 24_000_000_000, "free_bytes": 20_000_000_000, "size_label": "24GB"}
  b["device_memory"] = {"total_bytes": 24_000_000_000, "free_bytes": 19_000_000_000, "size_label": "large-tier"}
  assert canonical_capability_identity(a) != canonical_capability_identity(b)
  b["device_memory"]["free_bytes"] = a["device_memory"]["free_bytes"]
  assert canonical_capability_identity(a) == canonical_capability_identity(b)
