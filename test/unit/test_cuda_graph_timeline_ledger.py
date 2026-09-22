import sqlite3

from extra.llm_research.decode.cuda_graph_timeline_ledger import (
  analyze, classify, intersection_ns, interval_ns, split_replays,
)


def _trace(path):
  con = sqlite3.connect(path)
  con.execute("create table StringIds (id integer primary key, value text)")
  con.executemany("insert into StringIds values (?,?)", [(1, "mul_mat_vec_q"), (2, "quantize_q8_1"),
                                                          (3, "mul_mat_vec_q<Q4_K, fusion=false>"),
                                                          (4, "quantize_q8_1<float>")])
  con.execute("create table CUPTI_ACTIVITY_KIND_KERNEL (start integer, end integer, streamId integer, "
              "shortName integer, demangledName integer, graphNodeId integer, graphId integer, registersPerThread integer, "
              "gridX integer, gridY integer, gridZ integer, blockX integer, blockY integer, blockZ integer, "
              "staticSharedMemory integer, dynamicSharedMemory integer, localMemoryPerThread integer)")
  for replay in range(4):
    base = replay * 100_000
    # 10 us MMQ, 4 us quant; 2 us of quant overlaps MMQ. Span/union = 12 us.
    con.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (base, base+10_000, 1, 1, 3, 10, 7, 48, 4, 1, 1, 32, 4, 1, 0, 0, 0))
    con.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (base+8_000, base+12_000, 2, 2, 4, 11, 7, 20, 1, 1, 1, 32, 1, 1, 0, 0, 0))
  con.commit(); con.close()


def test_interval_math():
  assert interval_ns([(0, 10), (5, 12), (20, 25)]) == 17
  assert intersection_ns([(0, 10), (20, 30)], [(5, 22)]) == 7


def test_classification():
  assert classify("mul_mat_vec_q") == "mmq"
  assert classify("flash_attn_combine_results") == "flash_combine"
  assert classify("unrecognized") == "other"


def test_split_replays_rejects_partial_tail():
  rows = [{"graphNodeId": x} for x in (1, 2, 1, 2, 1)]
  assert [len(x) for x in split_replays(rows)] == [2, 2]


def test_split_replays_rejects_overlapping_launch_ranges():
  rows = [{"graphNodeId": 1, "start": 0, "end": 10},
          {"graphNodeId": 2, "start": 1, "end": 20},
          {"graphNodeId": 1, "start": 15, "end": 25},
          {"graphNodeId": 2, "start": 16, "end": 26}]
  import pytest
  with pytest.raises(ValueError, match="ranges overlap"): split_replays(rows)


def test_analyze_timeline(tmp_path):
  path = tmp_path / "trace.sqlite"
  _trace(path)
  out = analyze(str(path), 7, warmup=1)
  assert out["steady_replays"] == 3
  assert out["replay_split"]["discarded_fragments"] == 0
  assert out["median"]["node_sum_us"] == 14.0
  assert out["median"]["span_us"] == 12.0
  assert out["median"]["kernel_union_us"] == 12.0
  assert out["median"]["overlap_mass_us"] == 2.0
  assert out["non_anchor_aggregate"]["union_us"] == 4.0
  assert out["non_anchor_aggregate"]["hidden_behind_mmq_us"] == 2.0
  assert out["non_anchor_aggregate"]["exposed_vs_mmq_us"] == 2.0
  assert out["classes"]["quantize_q8_1"]["hidden_behind_mmq_us"] == 2.0
  assert out["classes"]["quantize_q8_1"]["exposed_vs_mmq_us"] == 2.0
  assert out["shape_census"][0]["count_per_replay"] == 1
  assert {x["variant"] for x in out["shape_census"]} == {
    "mul_mat_vec_q<Q4_K, fusion=false>", "quantize_q8_1<float>"}
  assert out["inter_replay_gap"]["excluded_count"] == 0
