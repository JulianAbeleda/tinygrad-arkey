import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tinygrad.llm import prefill_graph_gemm
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.llm.prefill_candidate_runtime import (ARTIFACT, NV_ARTIFACT, canonical_candidate_set_identity,
  decode_prefill_graph_candidate_set, promoted_candidate_registry, promoted_candidate_set,
  automatic_promoted_prefill_graph_policy)


ROOT = Path(__file__).parents[2]
FACTS = {"backend":"AMD", "architecture":"gfx1100", "capabilities":{"wave_size":32}}
NV_FACTS = {"backend":"NV", "architecture":"sm_120", "capabilities":{"wave_size":32}}
_TC_ONLY = (Opt(OptOps.TC, 0, (-1, 2, 1)),)
_NV_MEASURED = _TC_ONLY
EXPECTED = {
  "attn_kv":((512, 1024, 4096), "51b0562291285f98693f5320a5dce21673a32813c507377d0436afa53fe3b006"),
  "attn_qo":((512, 4096, 4096), "7508432bc2ab86532eb07bea71fb4f518e82dc259252a704f60131b2aa608d24"),
  "ffn_down":((512, 4096, 12288), "fe0e765afd86cdda318f1950ad59b4374d95e862e0f1112d0e576d5c32231d9d"),
  "ffn_gate_up":((512, 12288, 4096), "8b6e3b2a9b25f7ad35e2e252d74129d96958b8367653024ad73e81fcac2aebb9"),
}


def test_candidate_warmstart_opts_are_target_declared_with_safe_default():
  # Candidate contexts own the complete output tile.  NV sm_120 therefore uses
  # the correctness-qualified TC-only schedule; generic output UPCASTs may not
  # be layered on top of the candidate geometry.
  assert prefill_graph_gemm._candidate_warmstart_opts("NV", "sm_120", 32) == _NV_MEASURED
  assert prefill_graph_gemm._candidate_warmstart_opts("AMD", "gfx1100", 32) == _TC_ONLY
  assert prefill_graph_gemm._candidate_warmstart_opts("NV", "sm_90", 32) == _TC_ONLY
  assert prefill_graph_gemm._candidate_warmstart_opts("METAL", "m1", 32) == _TC_ONLY
NV_EXPECTED = {
  "attn_kv":((512, 1024, 4096), "81b2583b95e4fcddb614036cfd9ab0abcbd8a245774be7c36dcc143e3bbdb945"),
  "attn_qo":((512, 4096, 4096), "c45a763ae5c9670c8487face4b4e20a015b239ee60a2ad1520cde4ede6ef36c2"),
  "ffn_down":((512, 4096, 12288), "03896c56299ec804cdeeb477becc2a574b33e76d5e2cabc7c4dc86678b7b1e62"),
  "ffn_gate_up":((512, 12288, 4096), "fcc738029cd7357fe9574e421f2d0f8874aad12a8fdaa5f798e255c14013d558"),
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
  expanded = promoted_candidate_set("AMD", "gfx1100", 32).to_json()
  assert expanded["schema"] == "boltbeam.full_kernel_candidate_set.v1"
  assert {entry["payload"]["workload"]["role"]:
          (tuple(entry["payload"]["workload"]["shape"][axis] for axis in ("m", "n", "k")), entry["canonical_identity"])
          for entry in expanded["entries"]} == EXPECTED
  assert canonical_candidate_set_identity(expanded) == \
         "candidate_set:sha256:2783d3ebb084e465d733cc161aa31485ac3f8ec45ff5c3aa4c1790795e852847"
  assert ARTIFACT.stat().st_size < 6000


def test_nv_compact_artifact_expands_to_the_frozen_sm120_promoted_candidate_set():
  expanded = promoted_candidate_set("NV", "sm_120", 32).to_json()
  assert expanded["schema"] == "boltbeam.full_kernel_candidate_set.v1"
  assert {entry["payload"]["workload"]["role"]:
          (tuple(entry["payload"]["workload"]["shape"][axis] for axis in ("m", "n", "k")), entry["canonical_identity"])
          for entry in expanded["entries"]} == NV_EXPECTED
  assert canonical_candidate_set_identity(expanded) == \
         "candidate_set:sha256:1b8ea95d50bb55962474721cf013a6c3a704038916856353c65281112a166c7f"
  assert NV_ARTIFACT.stat().st_size < 6000
  for entry in expanded["entries"]:
    assert entry["payload"]["workload"]["target"] == {"backend":"NV", "arch":"sm_120", "wave_size":32}


def test_typed_admission_preserves_frozen_geometry_pipeline_and_identity():
  current = promoted_candidate_registry("AMD", "gfx1100", 32)
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


def test_nv_typed_admission_preserves_frozen_geometry_pipeline_and_identity():
  current = promoted_candidate_registry("NV", "sm_120", 32)
  assert len(current.admissions) == 4
  for got in current.admissions:
    workload = got.normalized_payload["workload"]
    assert (tuple(workload["shape"][axis] for axis in ("m", "n", "k")), got.canonical_identity) == NV_EXPECTED[workload["role"]]
    assert (got.geometry.tile, got.geometry.waves, got.geometry.threads, got.geometry.wave_size) == \
           ((128, 128, 32), (4, 2), 256, 32)
    assert tuple((x.role, x.base, x.end, x.stride_bytes) for x in got.geometry.lds_windows) == \
           (("A", 0, 10240, 80), ("B", 10240, 20480, 80))
    assert (got.pipeline_plan.buffer_count, got.pipeline_plan.slot_bytes, got.pipeline_plan.stage_count,
            got.active_lds_bytes) == (2, 20480, 1, 40960)


def test_automatic_policy_preserves_exact_promoted_authority_shape():
  candidate_set = promoted_candidate_set("AMD", "gfx1100", 32).to_json()
  inventory = _inventory(candidate_set)
  policy = automatic_promoted_prefill_graph_policy(inventory, FACTS)
  assert policy["strategy"] == "FULL_RESIDENT_OVERLAY"
  assert policy["candidate_id"] == "prefill_wmma_lds_dbuf_generated"
  assert policy["routes"] == {**{f"candidate-{index}":"prefill_wmma_lds_dbuf_generated" for index in range(4)},
                              "lm-head":"fixed-ggml-linear"}
  assert policy["graph_gemm"]["candidate_set"] == candidate_set
  assert {(row["role"], tuple(row["shape"][axis] for axis in ("m", "n", "k")), row["candidate_identity"])
          for row in policy["graph_gemm"]["policy_rows"]} == \
         {(role, shape, identity) for role,(shape,identity) in EXPECTED.items()}


def test_nv_automatic_policy_admits_the_four_sm120_roles_exactly():
  candidate_set = promoted_candidate_set("NV", "sm_120", 32).to_json()
  inventory = _inventory(candidate_set)
  policy = automatic_promoted_prefill_graph_policy(inventory, NV_FACTS)
  assert policy["strategy"] == "FULL_RESIDENT_OVERLAY"
  assert policy["candidate_id"] == "prefill_wmma_lds_dbuf_generated"
  assert policy["graph_gemm"]["candidate_set"] == candidate_set
  assert policy["graph_gemm"]["candidate_set_identity"] == \
         "candidate_set:sha256:1b8ea95d50bb55962474721cf013a6c3a704038916856353c65281112a166c7f"
  assert {(row["role"], tuple(row["shape"][axis] for axis in ("m", "n", "k")), row["candidate_identity"])
          for row in policy["graph_gemm"]["policy_rows"]} == \
         {(role, shape, identity) for role,(shape,identity) in NV_EXPECTED.items()}
  assert all(row["target"] == {"backend":"NV", "arch":"sm_120", "wave_size":32}
             for row in policy["graph_gemm"]["policy_rows"])


@pytest.mark.parametrize("mutation", ("identity", "partial", "foreign_target"))
def test_decoder_and_policy_fail_closed(mutation):
  candidate_set = promoted_candidate_set("AMD", "gfx1100", 32).to_json()
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


def test_cross_target_decode_fails_closed():
  nv_set = promoted_candidate_set("NV", "sm_120", 32).to_json()
  nv_set["entries"][0]["payload"]["workload"]["target"]["arch"] = "sm_130"
  with pytest.raises(ValueError, match="compact target is unsupported"): decode_prefill_graph_candidate_set(nv_set)


def test_promoted_registry_is_sufficient_for_graph_gemm_exact_attachment(monkeypatch):
  registry = decode_prefill_graph_candidate_set(promoted_candidate_set("AMD", "gfx1100", 32).to_json())
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
  for relative in ("tinygrad/llm/prefill_candidate_runtime.py",):
    tree = ast.parse((ROOT / relative).read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(name == "extra" or name.startswith("extra.llm_research") for name in imports), relative
