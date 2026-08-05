"""Hermetic CPU-only tests for the L2 constant-DAG llama-kernel oracle config.

Covers synthetic config generation (swap table, preserved edges, per-group
node lists), kernel class detection on the route_kernel_census vocabulary,
source-DAG hash determinism, validation pass/fail on tampered configs, and a
skipif-missing integration check that loads the real capture manifest and the
real quantized-oracle ABI report and asserts >100 MMV swap nodes with a stable
hash across two builds.  No GPU is touched.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from extra.llm_research.decode.l2_constant_dag_oracle import (
  EXPECTED_ARG_COUNT, FIXTURE_ORACLE, FUSION_ARGS_SIZE, ORACLE_SCHEMA,
  SCHEMA, SRC1_ROW_PADDING, abi_arg_template, build_config,
  build_synthetic_capture, canonical_physical_edges, classify, dag_hash,
  swap_entry, validate_config,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
CAPTURE = REPO / "docs/task_workflow/output/nv-decode-overlap-b3-2-aligned-capture-manifest-20260804.json"
ORACLE = REPO / "scratchpad/llama_cuda_quantized_oracle_dump/llama_cuda_quantized_oracle_v1.json"


@pytest.fixture(scope="module")
def synthetic_capture() -> dict:
  return build_synthetic_capture()


@pytest.fixture(scope="module")
def synthetic_config(synthetic_capture) -> dict:
  return build_config(
    synthetic_capture,
    FIXTURE_ORACLE,
    source_capture="synthetic-fixture-12n-2g-4mmv",
    oracle_report="embedded-llama-quantized-oracle-abi",
  )


def test_synthetic_schema_counts_and_groups(synthetic_config):
  assert synthetic_config["schema"] == SCHEMA
  assert synthetic_config["node_count"] == 12
  assert synthetic_config["edge_count"] == 14
  assert synthetic_config["group_count"] == 2
  assert synthetic_config["predicted_swap_count"] == 4
  assert set(synthetic_config["per_group"].keys()) == {"0", "1"}
  for group in synthetic_config["per_group"].values():
    assert len(group) == 6
    for node in group:
      assert {"id", "name", "class", "group"} <= set(node.keys())
      assert node["group"] in (0, 1)
      assert node["identity_sha256"]
  edges = synthetic_config["preserved_physical_edges"]
  assert len(edges) == 14
  assert {"from", "to", "kind", "crosses_group"} <= set(edges[0].keys())
  assert sum(1 for edge in edges if edge["crosses_group"]) == 2


def test_synthetic_swap_table_entries(synthetic_config):
  table = synthetic_config["mmv_swap_table"]
  assert len(table) == 4
  variants = sorted(entry["variant"] for entry in table.values())
  assert variants == ["Q4_K", "Q4_K", "Q4_K", "Q6_K"]
  for entry in table.values():
    assert entry["block_layout"]["size_bytes"] == (144 if entry["variant"] == "Q4_K" else 210)
    abi = entry["abi_arg_template"]
    assert abi["arg_count"] == EXPECTED_ARG_COUNT
    assert len(abi["argument_order"]) == EXPECTED_ARG_COUNT
    assert abi["argument_order"][0]["name"] == "vx_ptr"
    assert abi["fusion_args"]["size_bytes"] == FUSION_ARGS_SIZE
    assert abi["src1"]["row_padding"] == SRC1_ROW_PADDING
    assert abi["src1"]["quant"] == "block_q8_1"
    assert "mul_mat_vec_q" in abi["canonical_entry"]


def test_synthetic_swap_variants_match_class(synthetic_config):
  nodes_by_id = {node["id"]: node for group in synthetic_config["per_group"].values() for node in group}
  expected = {"q4k_gemv": "Q4_K", "q6k_gemv": "Q6_K"}
  for swap_key, entry in synthetic_config["mmv_swap_table"].items():
    node = nodes_by_id[int(swap_key)]
    assert entry["variant"] == expected[node["class"]]
    assert entry["node_id"] == int(swap_key)


@pytest.mark.parametrize("name,expected", [
  ("q4k_g3_lanemap_gemv_4096_4096", "q4k_gemv"),
  ("q4k_g3_lanemap_gemv_12288_4096", "q4k_gemv"),
  ("q6k_gen_coop_4096_12288", "q6k_gemv"),
  ("q6k_gen_partial_1024_4096_4", "q6k_gemv"),
  ("flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128", "flash_decode_attention"),
  ("E_32_32_4_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", "elementwise_fusion"),
  ("r_8_16_8_synthetic", "elementwise_fusion"),
  ("E_1187_32_4_scatter", "scatter"),
  ("q6k_gen_coop_151936_4096", "vocab_head"),
  ("q6k_vocab_scalar_reduce_151936_4096", "vocab_head"),
  ("decode_kv_rope_store_1024", "kv_store"),
  ("some_unknown_kernel_1", "other"),
])
def test_classify_vocabulary(name, expected):
  assert classify(name) == expected


def test_dag_hash_deterministic(synthetic_capture):
  edges = synthetic_capture["arms"]["physical"]["edges"]
  first = dag_hash(edges)
  assert first == dag_hash(list(reversed(edges)))
  assert first == dag_hash(edges[:7] + edges[7:])
  canonical = canonical_physical_edges(edges)
  assert canonical == sorted(canonical)
  config = build_config(synthetic_capture, FIXTURE_ORACLE)
  assert config["source_dag_hash"] == first


def test_validate_passes(synthetic_config):
  result = validate_config(synthetic_config)
  assert result["valid"] is True
  assert result["issues"] == []


def test_validate_fails_when_swap_node_removed(synthetic_config):
  broken = json.loads(json.dumps(synthetic_config))
  node_id = next(iter(broken["mmv_swap_table"]))
  del broken["mmv_swap_table"][node_id]
  result = validate_config(broken)
  assert result["valid"] is False
  assert any("predicted_swap_count" in issue for issue in result["issues"])


def test_validate_fails_when_edge_added(synthetic_config):
  broken = json.loads(json.dumps(synthetic_config))
  broken["preserved_physical_edges"].append({"from": 0, "to": 11, "kind": "RAW", "crosses_group": True})
  result = validate_config(broken)
  assert result["valid"] is False
  assert any("recomputed source_dag_hash" in issue for issue in result["issues"])


def test_validate_fails_when_group_intact_broken(synthetic_config):
  broken = json.loads(json.dumps(synthetic_config))
  node = broken["per_group"]["0"][0]
  node["group"] = 1
  result = validate_config(broken)
  assert result["valid"] is False
  assert any("does not match group key" in issue for issue in result["issues"])


def test_validate_fails_on_variant_mismatch(synthetic_config):
  broken = json.loads(json.dumps(synthetic_config))
  swap_key = next(iter(broken["mmv_swap_table"]))
  entry = broken["mmv_swap_table"][swap_key]
  entry["variant"] = "Q6_K" if entry["variant"] == "Q4_K" else "Q4_K"
  result = validate_config(broken)
  assert result["valid"] is False
  assert any("does not match class" in issue for issue in result["issues"])


def test_swap_entry_block_layouts():
  q4 = swap_entry(1, "Q4_K", FIXTURE_ORACLE)
  q6 = swap_entry(2, "Q6_K", FIXTURE_ORACLE)
  assert q4["block_layout"]["name"] == "block_q4_k"
  assert q4["block_layout"]["size_bytes"] == 144
  assert q6["block_layout"]["name"] == "block_q6_k"
  assert q6["block_layout"]["size_bytes"] == 210
  assert "ggml_type12" in q4["abi_arg_template"]["canonical_entry"]
  assert "ggml_type14" in q6["abi_arg_template"]["canonical_entry"]
  assert abi_arg_template(FIXTURE_ORACLE, "Q4_K")["arg_count"] == EXPECTED_ARG_COUNT
  assert FIXTURE_ORACLE["schema"] == ORACLE_SCHEMA


def test_cli_synthetic_roundtrip(tmp_path):
  out = tmp_path / "l2-synthetic.json"
  env = dict(os.environ)
  env["PYTHONPATH"] = str(REPO)
  result = subprocess.run(
    [sys.executable, "-m", "extra.llm_research.decode.l2_constant_dag_oracle", "--synthetic", "--out", str(out)],
    cwd=str(REPO), env=env, capture_output=True, text=True, check=False,
  )
  assert result.returncode == 0, result.stderr
  config = json.loads(out.read_text())
  assert config["predicted_swap_count"] == 4
  validation = subprocess.run(
    [sys.executable, "-m", "extra.llm_research.decode.l2_constant_dag_oracle", "--validate", str(out)],
    cwd=str(REPO), env=env, capture_output=True, text=True, check=False,
  )
  assert validation.returncode == 0, validation.stdout + validation.stderr
  assert json.loads(validation.stdout)["valid"] is True


@pytest.mark.skipif(not (CAPTURE.is_file() and ORACLE.is_file()), reason="real capture manifest or oracle report missing")
def test_real_capture_swap_count_and_stable_hash():
  oracle = json.loads(ORACLE.read_text())
  first = build_config(
    json.loads(CAPTURE.read_text()), oracle,
    source_capture=str(CAPTURE), oracle_report=str(ORACLE),
  )
  second = build_config(
    json.loads(CAPTURE.read_text()), oracle,
    source_capture=str(CAPTURE), oracle_report=str(ORACLE),
  )
  assert first["predicted_swap_count"] > 100
  assert first["predicted_swap_count"] == second["predicted_swap_count"]
  assert first["source_dag_hash"] == second["source_dag_hash"]
  assert first["node_count"] == 1021
  assert first["group_count"] == 6
  assert first["edge_count"] == 355772
  assert first["class_counts"]["q4k_gemv"] == 216
  assert first["class_counts"]["q6k_gemv"] == 36
  result = validate_config(first)
  assert result["valid"] is True
  assert result["issues"] == []
