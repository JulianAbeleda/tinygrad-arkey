import json
from pathlib import Path

import pytest

from tinygrad.llm.model_facts import model_facts_from_gguf_metadata
from extra.qk.prefill.workload_inventory import (CANDIDATE_INVENTORY_SCHEMA, MeasuredRow,
  build_workload_inventory, generate_candidate_inventory)
from extra.qk.runtime_specs import FullKernelCandidateSet, admit_full_kernel_candidate_set


def _facts():
  kv = {"general.architecture":"qwen3", "qwen3.embedding_length":5120, "qwen3.feed_forward_length":17408,
        "qwen3.attention.head_count":40, "qwen3.attention.head_count_kv":8, "qwen3.attention.key_length":128}
  tensors = []
  for block, quant in enumerate((14, 12)):
    tensors += [(f"blk.{block}.ffn_gate.weight", (5120, 17408), 12, 0),
                (f"blk.{block}.ffn_up.weight", (5120, 17408), 12, 0),
                (f"blk.{block}.ffn_down.weight", (17408, 5120), quant, 0),
                (f"blk.{block}.attn_q.weight", (5120, 5120), 12, 0),
                (f"blk.{block}.attn_output.weight", (5120, 5120), 12, 0),
                (f"blk.{block}.attn_k.weight", (5120, 1024), 12, 0),
                (f"blk.{block}.attn_v.weight", (5120, 1024), quant, 0)]
  return model_facts_from_gguf_metadata(kv, {"tensor_infos":tensors})


ROWS = tuple(MeasuredRow(role, quant, 512, n, k) for role, quant, n, k in (
  ("ffn_gate_up", "Q4_K", 17408, 5120), ("attn_qo", "Q4_K", 5120, 5120),
  ("ffn_down", "Q4_K", 5120, 17408), ("attn_kv", "Q4_K", 1024, 5120),
  ("ffn_down", "Q6_K", 5120, 17408), ("attn_kv", "Q6_K", 1024, 5120)))


def _templates():
  artifact = json.loads(Path("bench/prefill-pure-full-kernel/multirole-buffer2-candidate-set-v1/candidate-set.json").read_text())
  return {x["payload"]["workload"]["role"]:x["payload"] for x in artifact["entries"]}


def test_exact_mixed_inventory_and_canonical_candidate_sets_are_admitted():
  inventory = build_workload_inventory(_facts(), ROWS, profile="fixture")
  assert [(x["role"], x["quant_format"], x["call_count"]) for x in inventory["rows"]] == [
    ("ffn_gate_up", "Q4_K", 4), ("attn_qo", "Q4_K", 4), ("ffn_down", "Q4_K", 1),
    ("attn_kv", "Q4_K", 3), ("ffn_down", "Q6_K", 1), ("attn_kv", "Q6_K", 1)]
  assert inventory["rows"][0]["logical_flop"] == 4 * 2 * 512 * 17408 * 5120
  artifact = generate_candidate_inventory(inventory, _templates())
  assert artifact["schema"] == CANDIDATE_INVENTORY_SCHEMA
  assert {q:len(s["entries"]) for q,s in artifact["candidate_sets"].items()} == {"Q4_K":4, "Q6_K":2}
  assert len({x["canonical_identity"] for x in artifact["bindings"]}) == 6
  assert artifact["inventory_identity"] == inventory["inventory_identity"]
  assert all(x["inventory_key"]["inventory_identity"] == inventory["inventory_identity"] for x in artifact["bindings"])


def test_inventory_identity_ignores_optional_provenance_and_model_rename():
  plain = build_workload_inventory(_facts(), ROWS)
  named = build_workload_inventory(_facts(), ROWS, profile="benchmark-label", model_path="renamed/model.gguf")
  assert plain["inventory_identity"] == named["inventory_identity"]
  assert named["provenance"] == {"profile":"benchmark-label", "model_path":"renamed/model.gguf"}
  assert generate_candidate_inventory(plain, _templates())["bindings"] == \
         generate_candidate_inventory(named, _templates())["bindings"]


@pytest.mark.parametrize("mutate", [
  lambda row: row["tensor_identities"].__setitem__(0, "renamed.tensor"),
  lambda row: row.__setitem__("quant_format", "Q6_K"),
  lambda row: row["shape"].__setitem__("m", 256),
  lambda row: row.__setitem__("call_count", row["call_count"] + 1),
  lambda row: row.__setitem__("source_bytes", row["source_bytes"] + 1),
  lambda row: row["layout"].__setitem__("block_bytes", row["layout"]["block_bytes"] + 1),
])
def test_exact_tensor_inventory_mutation_invalidates_identity(mutate):
  inventory = build_workload_inventory(_facts(), ROWS)
  mutate(inventory["rows"][0])
  with pytest.raises(ValueError, match="inventory identity mismatch"):
    generate_candidate_inventory(inventory, _templates())


def test_legacy_profile_only_inventory_is_read_with_content_identity():
  inventory = build_workload_inventory(_facts(), ROWS, profile="legacy-label")
  legacy = {"schema":inventory["schema"], "profile":"legacy-label", "rows":inventory["rows"]}
  artifact = generate_candidate_inventory(legacy, _templates())
  assert artifact["inventory_identity"] == inventory["inventory_identity"]
  assert all(x["payload"]["workload"]["profile"] == inventory["inventory_identity"]
             for candidate_set in artifact["candidate_sets"].values() for x in candidate_set["entries"])


@pytest.mark.parametrize("mutation,match", [
  (lambda rows: rows + rows[:1], "duplicate measured"),
  (lambda rows: rows[:-1] + (MeasuredRow("missing", "Q6_K", 512, 1024, 5120),), "unknown packed"),
  (lambda rows: rows[:-1] + (MeasuredRow("attn_kv", "Q5_K", 512, 1024, 5120),), "unsupported packed"),
])
def test_inventory_fails_closed_on_duplicate_unknown_and_unsupported(mutation, match):
  with pytest.raises(ValueError, match=match): build_workload_inventory(_facts(), mutation(ROWS), profile="fixture")


def test_candidate_generation_rejects_duplicate_exact_key_and_tensor_shape_mismatch():
  inventory = build_workload_inventory(_facts(), ROWS, profile="fixture")
  duplicate = {**inventory, "rows":inventory["rows"] + inventory["rows"][:1]}
  duplicate.pop("inventory_identity")  # legacy input still reaches structural duplicate admission
  with pytest.raises(ValueError, match="duplicate exact inventory key"): generate_candidate_inventory(duplicate, _templates())
  mismatch = json.loads(json.dumps(inventory))
  mismatch.pop("inventory_identity")
  mismatch["rows"][0]["shape"]["n"] = 5120
  # No role-specific branch repairs inconsistent tensor evidence; rebinding eventually fails admission.
  with pytest.raises(ValueError, match="candidate/tensor shape mismatch"): generate_candidate_inventory(mismatch, _templates())


def test_runtime_warmstart_collision_fails_closed_within_quant_partition():
  inventory = build_workload_inventory(_facts(), ROWS, profile="fixture")
  collision = json.loads(json.dumps(inventory))
  collision.pop("inventory_identity")
  collision["rows"][1]["shape"] = dict(collision["rows"][0]["shape"])
  collision["rows"][1]["source_bytes"] = collision["rows"][0]["source_bytes"]
  collision["rows"][1]["logical_flop"] = collision["rows"][0]["logical_flop"]
  with pytest.raises(ValueError, match="warmstart_key_collision"):
    generate_candidate_inventory(collision, _templates())


def test_committed_14b_inventory_contains_six_admitted_identity_bound_rows():
  path = Path("bench/prefill-pure-full-kernel/qwen3-14b-mixed-quant-candidate-inventory-v1.json")
  artifact = json.loads(path.read_text())
  assert artifact["schema"] == CANDIDATE_INVENTORY_SCHEMA
  assert len(artifact["inventory"]["rows"]) == len(artifact["bindings"]) == 6
  assert {q:len(row["entries"]) for q,row in artifact["candidate_sets"].items()} == {"Q4_K":4, "Q6_K":2}
  identities = {binding["canonical_identity"] for binding in artifact["bindings"]}
  for row in artifact["candidate_sets"].values():
    candidate_set = FullKernelCandidateSet.from_json(row)
    assert {entry.canonical_identity for entry in candidate_set.entries} <= identities
    admit_full_kernel_candidate_set(candidate_set)
