"""Hermetic tests for the tinygrad CUPTI node ledger.

Covers the tinygrad kernel-name classifier and the multi-graph aggregation:
the decode token is six CUDA graphs, and analyze() concatenates the matching
replay of every graph into one timeline before computing the llama-ledger
metrics.  All tests are CPU-only synthetic sqlite fixtures.
"""
import sqlite3

import pytest

from extra.llm_research.decode.nv_tinygrad_node_ledger import analyze, tg_classify


def test_tg_classify_vocabulary():
  assert tg_classify("q4k_g3_lanemap_gemv_4096_4096") == "gemv"
  assert tg_classify("q6k_gen_coop_151936_4096") == "gemv"
  assert tg_classify("q6k_gen_partial_1024_4096_4") == "gemv"
  assert tg_classify("flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128") == "flash_score"
  assert tg_classify("flash_fused_gmax_combine_32_128") == "flash_combine"
  assert tg_classify("r_16_256_ed256c4ae79e0e20f4fdab97c161cae8055c723ab1fe435fd6bb57b2df6424cc") == "norms"
  assert tg_classify("E_16_32_4_2_f97dcdcb1846b48225dea340aec269b51112d42e1510960ebb941d28524bf1c2") == "rope_kv"
  assert tg_classify("E_128_32_3_2ba53b0e7c103a17221f1338cefdf7a455bdce4577bf982f53da9d0a4efe2961") == "residual_cast"
  assert tg_classify("E_1187_32_4_76d37a73a646ec3a4b5e32ab783e0ace496f5855bbba7492c8c7fe87bbff5f1f") == "vocab_aux"
  # The two prefixes the population classifier does not own.
  assert tg_classify("q6k_vocab_scalar_reduce_151936_4096") == "vocab_aux"
  assert tg_classify("r_32_32_4_2_8_cfa9da8eaf323595d8304be2253d71d6e6ba721142f56a31957010b4ce2a766b") == "reduce_output"
  assert tg_classify("totally_unknown_program") == "other"


def _make_trace(path, *, overlap: bool = False, replay_count_mismatch: bool = False):
  con = sqlite3.connect(path)
  con.execute("create table StringIds (id integer primary key, value text)")
  con.executemany("insert into StringIds values (?,?)", [
    (1, "q4k_g3_lanemap_gemv_4096_4096"),
    (2, "r_16_256_ed256c4ae79e0e20f4fdab97c161cae8055c723ab1fe435fd6bb57b2df6424cc"),
    (3, "E_1187_32_4_76d37a73a646ec3a4b5e32ab783e0ace496f5855bbba7492c8c7fe87bbff5f1f"),
  ])
  con.execute("create table CUPTI_ACTIVITY_KIND_KERNEL (start integer, end integer, streamId integer, "
              "shortName integer, demangledName integer, graphNodeId integer, graphId integer, registersPerThread integer, "
              "gridX integer, gridY integer, gridZ integer, blockX integer, blockY integer, blockZ integer, "
              "staticSharedMemory integer, dynamicSharedMemory integer, localMemoryPerThread integer)")
  # graphId 1: a 10 us GEMV then a 4 us norm reduce (overlap variant hides the
  # norm's first 2 us behind the GEMV). graphId 2: a 3 us vocab aux kernel.
  graph1_replays = 4 if not replay_count_mismatch else 4
  graph2_replays = 4 if not replay_count_mismatch else 5
  def _row(base, short, node, gid, dur_ns):
    con.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (base, base + dur_ns, 1, short, short, node, gid, 48, 1, 1, 1, 32, 1, 1, 0, 0, 0))
  for replay in range(graph1_replays):
    base = replay * 100_000
    _row(base, 1, 10, 1, 10_000)
    norm_start = base + 8_000 if overlap else base + 10_000
    _row(norm_start, 2, 11, 1, 4_000)
  for replay in range(graph2_replays):
    base = replay * 100_000 + 14_000
    _row(base, 3, 20, 2, 3_000)
  con.commit()
  con.close()


def test_analyze_multi_graph_serial(tmp_path):
  path = tmp_path / "trace.sqlite"
  _make_trace(path)
  out = analyze(str(path), (1, 2), warmup=1)
  assert out["replay_counts"] == {1: 4, 2: 4}
  assert out["steady_replays"] == 3
  m = out["median"]
  assert m["nodes"] == 3
  assert m["node_sum_us"] == 17.0
  assert m["span_us"] == 17.0
  assert m["kernel_union_us"] == 17.0
  assert m["overlap_mass_us"] == 0.0
  assert out["classes"]["gemv"]["node_sum_us"] == 10.0
  assert out["classes"]["norms"]["node_sum_us"] == 4.0
  assert out["classes"]["vocab_aux"]["node_sum_us"] == 3.0
  # Fully serial: every non-anchor class is exposed, none hidden.
  assert out["non_anchor_aggregate"]["exposed_vs_gemv_us"] == 7.0
  assert out["non_anchor_aggregate"]["hidden_behind_gemv_us"] == 0.0


def test_analyze_multi_graph_overlap(tmp_path):
  path = tmp_path / "trace.sqlite"
  _make_trace(path, overlap=True)
  out = analyze(str(path), (1, 2), warmup=1)
  m = out["median"]
  assert m["node_sum_us"] == 17.0
  assert m["kernel_union_us"] == 15.0
  assert m["overlap_mass_us"] == 2.0
  assert out["classes"]["norms"]["hidden_behind_gemv_us"] == 2.0
  assert out["classes"]["norms"]["exposed_vs_gemv_us"] == 2.0


def test_analyze_requires_matching_replay_counts(tmp_path):
  path = tmp_path / "trace.sqlite"
  _make_trace(path, replay_count_mismatch=True)
  with pytest.raises(ValueError, match="share a replay count"):
    analyze(str(path), (1, 2), warmup=1)
