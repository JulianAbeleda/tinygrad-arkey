import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tinygrad.llm import prefill_graph_gemm
from tinygrad.llm.prefill_candidate_runtime import (ARTIFACT, canonical_candidate_set_identity,
  decode_prefill_graph_candidate_set, promoted_candidate_registry, promoted_candidate_set,
  automatic_promoted_prefill_graph_policy as runtime_automatic_promoted_prefill_graph_policy)
from tinygrad.llm.promoted_prefill_policy import automatic_promoted_prefill_graph_policy


ROOT = Path(__file__).parents[2]
FACTS = {"backend":"AMD", "architecture":"gfx1100", "capabilities":{"wave_size":32}}
EXPECTED = {
  "attn_kv":((512, 1024, 4096), "51b0562291285f98693f5320a5dce21673a32813c507377d0436afa53fe3b006"),
  "attn_qo":((512, 4096, 4096), "7508432bc2ab86532eb07bea71fb4f518e82dc259252a704f60131b2aa608d24"),
  "ffn_down":((512, 4096, 12288), "fe0e765afd86cdda318f1950ad59b4374d95e862e0f1112d0e576d5c32231d9d"),
  "ffn_gate_up":((512, 12288, 4096), "8b6e3b2a9b25f7ad35e2e252d74129d96958b8367653024ad73e81fcac2aebb9"),
}


def _inventory(candidate_set):
  rows = []
  for index,entry in enumerate(candidate_set["entries"]):
    workload = entry["payload"]["workload"]
    rows.append({"invocation_id":f"candidate-{index}", "candidate_controlled":True,
                 "role":workload["role"], "shape":workload["shape"]})
  rows.append({"invocation_id":"lm-head", "candidate_controlled":False, "fixed_route_id":"fixed-ggml-linear"})
  return {"inventory_identity":"inventory:sha256:" + "a" * 64, "rows":rows}


def test_compact_artifact_expands_to_the_frozen_promoted_candidate_set():
  expanded = promoted_candidate_set().to_json()
  assert expanded["schema"] == "boltbeam.full_kernel_candidate_set.v1"
  assert {entry["payload"]["workload"]["role"]:
          (tuple(entry["payload"]["workload"]["shape"][axis] for axis in ("m", "n", "k")), entry["canonical_identity"])
          for entry in expanded["entries"]} == EXPECTED
  assert canonical_candidate_set_identity(expanded) == \
         "candidate_set:sha256:2783d3ebb084e465d733cc161aa31485ac3f8ec45ff5c3aa4c1790795e852847"
  assert ARTIFACT.stat().st_size < 6000


def test_typed_admission_preserves_frozen_geometry_pipeline_and_identity():
  current = promoted_candidate_registry()
  assert len(current.admissions) == 4
  for got in current.admissions:
    workload = got.normalized_payload["workload"]
    assert (tuple(workload["shape"][axis] for axis in ("m", "n", "k")), got.canonical_identity) == EXPECTED[workload["role"]]
    assert (got.geometry.tile, got.geometry.waves, got.geometry.threads, got.geometry.wave_size) == \
           ((128, 128, 32), (4, 2), 256, 32)
    assert tuple((x.role, x.base, x.end, x.stride_bytes) for x in got.geometry.lds_windows) == \
           (("A", 0, 10240, 80), ("B", 10240, 20480, 80))
    assert (got.pipeline_plan.buffer_count, got.pipeline_plan.slot_bytes, got.pipeline_plan.stage_count,
            got.active_lds_bytes) == (2, 20480, 1, 40960)


def test_automatic_policy_preserves_exact_promoted_authority_shape():
  candidate_set = promoted_candidate_set().to_json()
  inventory = _inventory(candidate_set)
  policy = automatic_promoted_prefill_graph_policy(inventory, FACTS)
  assert runtime_automatic_promoted_prefill_graph_policy(inventory, FACTS) == policy
  assert policy["strategy"] == "FULL_RESIDENT_OVERLAY"
  assert policy["candidate_id"] == "prefill_wmma_lds_dbuf_generated"
  assert policy["routes"] == {**{f"candidate-{index}":"prefill_wmma_lds_dbuf_generated" for index in range(4)},
                              "lm-head":"fixed-ggml-linear"}
  assert policy["graph_gemm"]["candidate_set"] == candidate_set
  assert {(row["role"], tuple(row["shape"][axis] for axis in ("m", "n", "k")), row["candidate_identity"])
          for row in policy["graph_gemm"]["policy_rows"]} == \
         {(role, shape, identity) for role,(shape,identity) in EXPECTED.items()}


@pytest.mark.parametrize("mutation", ("identity", "partial", "foreign_target"))
def test_decoder_and_policy_fail_closed(mutation):
  candidate_set = promoted_candidate_set().to_json()
  if mutation == "identity":
    candidate_set["entries"][0]["canonical_identity"] = "0" * 64
    with pytest.raises(ValueError, match="exact promoted"): decode_prefill_graph_candidate_set(candidate_set)
  elif mutation == "partial":
    inventory = _inventory(candidate_set)
    inventory["rows"].pop(0)
    assert automatic_promoted_prefill_graph_policy(inventory, FACTS) is None
  else:
    facts = copy.deepcopy(FACTS); facts["architecture"] = "gfx1200"
    assert automatic_promoted_prefill_graph_policy(_inventory(candidate_set), facts) is None


def test_promoted_registry_is_sufficient_for_graph_gemm_exact_attachment(monkeypatch):
  registry = decode_prefill_graph_candidate_set(promoted_candidate_set().to_json())
  policy = automatic_promoted_prefill_graph_policy(_inventory(registry.candidate_set.to_json()), FACTS)
  row = next(row for row in policy["graph_gemm"]["policy_rows"] if row["role"] == "attn_qo")
  binding = {"candidate_registry":registry, "inventory_identity":policy["inventory_identity"],
    "candidate_set_identity":policy["graph_gemm"]["candidate_set_identity"],
    "scanned_target_facts":{"target":row["target"]}, "selected_policy":row}
  admission = registry.get("attn_qo", (512, 4096, 4096), row["target"])
  monkeypatch.setattr(prefill_graph_gemm, "_install_candidate_matmul",
                      lambda x,w,n,k,selected,artifact:selected.canonical_identity)
  tensor = lambda shape: SimpleNamespace(shape=shape, ndim=len(shape))
  linear = SimpleNamespace(_prefill_graph_role="attn_qo", bias=None, _prefill_graph_gemm_binding=binding)
  assert prefill_graph_gemm.route_pf16_graph_gemm(linear, tensor((1, 512, 4096)), tensor((4096, 4096))) == \
         admission.canonical_identity


def test_new_production_policy_runtime_has_no_research_imports():
  for relative in ("tinygrad/llm/prefill_candidate_runtime.py", "tinygrad/llm/promoted_prefill_policy.py"):
    tree = ast.parse((ROOT / relative).read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(name == "extra" or name.startswith("extra.llm_research") for name in imports), relative
