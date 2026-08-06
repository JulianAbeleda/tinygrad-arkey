"""Hermetic tests for the CPU-only NV fusion/dataflow population ledger."""
import json, pathlib, pytest

from extra.llm_research.decode.nv_fusion_population_ledger import (
  POP_FLASH, POP_NORMS, POP_OTHER, POP_QUANT, POP_RESIDUAL, POP_ROPE_KV, POP_VOCAB,
  analyze, classify, load,
)


def h16(prefix: str) -> str:
  """A synthetic 64-hex hash suffix whose first 16 hex match `prefix`."""
  return "_" + prefix + "0" * (64 - len(prefix))


NORM_EPI = "E_32_32_4" + h16("f14a5cc0d0ed4c90")
ATTN_CAST = "E_32_32_4" + h16("0a5eb0ac56c097a0")
ATTN_RES = "E_32_32_4" + h16("02a9738c0547f555")
FFN_RES = "E_32_32_4" + h16("81c96a8e654e707f")
Q_NORM_EPI = "E_4_2_8_16_4" + h16("607f37a022a3f8d7")
ROPE_Q = "E_16_32_4_2" + h16("f97dcdcb1846b482")
TOKEN_FB = "E" + h16("c9699af0d7ec2ef0")
VOCAB_SAMPLER = "E_1187_32_4" + h16("76d37a73a646ec3a")


def _dag(nodes: list[tuple[int, str, float]], edges: list[tuple[int, int]] | None = None,
         groups: list[int] | None = None) -> dict:
  groups = groups if groups is not None else [0] * len(nodes)
  return {"nodes": [{"id": i, "name": name, "duration_us": dur, "group_id": gid}
                    for i, (_, name, dur) in enumerate(nodes) for gid in [groups[i]]],
          "edges": [{"from": a, "to": b} for a, b in (edges or [])]}


def test_classify_exact_identity_rules():
  assert classify("flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128") == (POP_FLASH, "flash_score", True)
  assert classify("flash_fused_gmax_combine_32_128") == (POP_FLASH, "flash_combine", True)
  assert classify("q4k_g3_lanemap_gemv_w1w3fused_12288_4096") == (POP_QUANT, "quant_core", True)
  assert classify("q6k_gen_coop_151936_4096_inkernel") == (POP_QUANT, "quant_core", True)
  assert classify("r_16_256" + h16("ed256c4ae79e0e20")) == (POP_NORMS, "rmsnorm_reduce", True)
  assert classify("r_2_8_4_4_16" + h16("88919c0cbc4a4f2d")) == (POP_NORMS, "q_norm_reduce", True)
  assert classify("r_8_16_8" + h16("8be251892c3b04a8")) == (POP_NORMS, "k_norm_reduce", True)
  assert classify(Q_NORM_EPI) == (POP_NORMS, "q_norm_epilogue", True)
  assert classify(NORM_EPI) == (POP_NORMS, "rmsnorm_epilogue", True)
  assert classify(ATTN_CAST) == (POP_RESIDUAL, "attention_cast", True)
  assert classify(ATTN_RES) == (POP_RESIDUAL, "attention_residual_add_or_ffn_down_cast", True)
  assert classify("E_128_32_3" + h16("580aa0aa9999c3d8")) == (POP_RESIDUAL, "ffn_activation_cast", True)
  assert classify(ROPE_Q) == (POP_ROPE_KV, "rope_q", True)
  assert classify("E_8_8_16_2" + h16("90f6c6e71d996ecf")) == (POP_ROPE_KV, "kv_store_k_rope_cast", True)
  assert classify("r_8_8_16_2_4" + h16("da68eeaa95084da8")) == (POP_ROPE_KV, "kv_store_k_rope_cast_with_q6_partial_reduce", True)
  assert classify(TOKEN_FB) == (POP_VOCAB, "token_feedback", True)
  assert classify(VOCAB_SAMPLER) == (POP_VOCAB, "vocab_sampler", True)
  assert classify("r_32_32_4_32_4" + h16("b7c5eeb3ced63845")) == (POP_VOCAB, "vocab_sampler", True)
  assert classify("r_16_8" + h16("b4bb03fadd22815e")) == (POP_VOCAB, "vocab_sampler", True)
  assert classify("quantize_q8_1") == ("llama_q8_pack", "llama_q8_pack", True)


def test_classify_unknown_hash_flagged_not_silent():
  unknown = "E_32_32_4" + h16("deadbeefdeadbeef")
  population, role, exact = classify(unknown)
  assert population == POP_RESIDUAL and role == "elementwise_ambiguous" and not exact
  assert classify("totally_unknown_program") == (POP_OTHER, "unclassified", False)


def test_classify_short_stems_are_exact_equality():
  # "E" and "E_2" must not swallow longer elementwise stems.
  assert classify("E_2" + h16("c8a3207cf1b8954e")) == (POP_VOCAB, "token_feedback", True)
  assert classify("E_2_8_16_4_4" + h16("1ca44358cb95f4d7")) == (POP_NORMS, "q_norm_epilogue", True)
  assert classify("E_2_8_16_4" + h16("cf5f7bca5f52b0d7")) == (POP_NORMS, "k_norm_epilogue", True)
  assert classify("E") == (POP_VOCAB, "token_feedback", True)


def test_analyze_per_population_arithmetic():
  # One full attention block: q/k norms, rope, flash, quant O, residual chain.
  nodes = [
    (0, "r_2_8_4_4_16" + h16("88919c0cbc4a4f2d"), 2.0),
    (1, Q_NORM_EPI, 3.0),
    (2, ROPE_Q, 4.0),
    (3, "flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128", 10.0),
    (4, "flash_fused_gmax_combine_32_128", 5.0),
    (5, "q4k_g3_lanemap_gemv_4096_4096", 20.0),
    (6, ATTN_RES, 1.5),
  ]
  edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
  result = analyze(_dag(nodes, edges))
  assert result["status"] == "PASS"
  assert result["capture"]["total_duration_us"] == 45.5
  norms = result["populations"][POP_NORMS]
  assert norms["node_count"] == 2 and norms["total_us"] == 5.0 and norms["mean_us"] == 2.5
  assert norms["max_us"] == 3.0 and norms["min_us"] == 2.0
  assert result["populations"][POP_FLASH]["node_count"] == 2
  assert result["populations"][POP_FLASH]["total_us"] == 15.0
  assert result["populations"][POP_QUANT]["total_us"] == 20.0
  assert result["populations"][POP_ROPE_KV]["node_count"] == 1
  residual = result["populations"][POP_RESIDUAL]
  assert residual["node_count"] == 1 and residual["total_us"] == 1.5
  # Per-node rows reconcile exactly.
  assert sum(row["duration_us"] for row in result["per_node"]) == 45.5


def test_fusion_candidate_detection_parent_is_quant_or_flash():
  nodes = [
    (0, "q4k_g3_lanemap_gemv_4096_4096", 20.0),
    (1, ATTN_RES, 1.5),                       # child of quant
    (2, "flash_fused_gmax_combine_32_128", 5.0),
    (3, ATTN_CAST, 1.2),                      # child of flash
    (4, NORM_EPI, 2.0),                       # child of nothing anchor
  ]
  edges = [(0, 1), (2, 3), (4, 3)]
  result = analyze(_dag(nodes, edges))
  candidate_ids = {c["node"] for c in result["fusion_candidates"]}
  assert candidate_ids == {1, 3}
  assert result["populations"][POP_RESIDUAL]["fusion_candidate_count"] == 2
  assert result["populations"][POP_RESIDUAL]["fusion_candidate_us"] == pytest.approx(2.7)
  assert result["populations"][POP_NORMS]["fusion_candidate_count"] == 0


def test_boundary_free_eligibility_flags():
  nodes = [
    (0, "r_16_256" + h16("ed256c4ae79e0e20"), 2.0),
    (1, NORM_EPI, 3.0),                       # norms: reduction + epilogue -> blocked
    (2, "q4k_g3_lanemap_gemv_4096_4096", 20.0),
    (3, ATTN_RES, 1.5),                       # pure ordinary epilogue -> eligible
    (4, "flash_fused_gmax_combine_32_128", 5.0),
  ]
  result = analyze(_dag(nodes, [(0, 1), (2, 3), (3, 4)]))
  assert result["populations"][POP_NORMS]["boundary_free_eligible"] is False
  assert result["populations"][POP_NORMS]["reduction_count"] == 1
  assert result["populations"][POP_RESIDUAL]["boundary_free_eligible"] is True
  assert result["populations"][POP_RESIDUAL]["epilogue_count"] == 1
  assert result["populations"][POP_FLASH]["boundary_free_eligible"] is False
  assert result["populations"][POP_FLASH]["custom_kernel_count"] == 1
  assert result["populations"][POP_QUANT]["boundary_free_eligible"] is False


def test_fail_closed_on_malformed_input(tmp_path: pathlib.Path):
  good = _dag([(0, NORM_EPI, 1.0)], [])
  path = tmp_path / "dag.json"
  path.write_text(json.dumps(good))
  assert load(path)["nodes"][0]["name"] == NORM_EPI

  def expect_failure(payload: dict):
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError): load(path)

  expect_failure({"edges": []})                                        # nodes missing
  expect_failure({"nodes": [], "edges": []})                           # empty nodes
  expect_failure(_dag([(0, NORM_EPI, 1.0)], []) | {"edges": None})     # edges missing
  bad_ids = _dag([(0, NORM_EPI, 1.0)], [])
  bad_ids["nodes"][0]["id"] = 1
  expect_failure(bad_ids)                                              # non-dense ids
  zero = _dag([(0, NORM_EPI, 0.0)], [])
  expect_failure(zero)                                                 # non-positive duration
  noname = _dag([(0, "", 1.0)], [])
  expect_failure(noname)                                               # empty name
  nogroup = _dag([(0, NORM_EPI, 1.0)], [])
  del nogroup["nodes"][0]["group_id"]
  expect_failure(nogroup)                                              # missing group_id
  out_of_range = _dag([(0, NORM_EPI, 1.0)], [(0, 3)])
  expect_failure(out_of_range)                                         # edge out of range


def test_unknown_names_yield_partial_status_not_silent():
  nodes = [(0, NORM_EPI, 1.0), (1, "some_future_program", 2.0)]
  result = analyze(_dag(nodes, [(0, 1)]))
  assert result["status"] == "PARTIAL"
  assert result["classifier"]["unclassified_node_count"] == 1
  assert result["classifier"]["exact_node_count"] == 1
  assert result["populations"][POP_OTHER]["total_us"] == 2.0
