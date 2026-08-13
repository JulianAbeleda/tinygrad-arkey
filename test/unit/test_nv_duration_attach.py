"""Hermetic CPU-only tests for the NV DEBUG=2 prime-trace duration attach.

Covers the trace parser, positional per-group alignment, fail-closed
signature-mismatch behavior (groups UNKNOWN, durations never zeroed), the
last-token window selection, and the HEAD duration-bearing DAG emission with
its canonical schema digest.
"""
import hashlib
import json

import pytest

from extra.llm_research.decode.nv_duration_attach import (
  DAG_SCHEMA, NVDurationAttachError, attach_durations, build_duration_dag,
  parse_trace, run_synthetic, select_token_window,
)
from extra.llm_research.decode.route_b3_dag_attribution import (
  build_attribution_fixture, compute_attribution_report,
)


def _fixture_capture() -> dict:
  calls, manifest = build_attribution_fixture()
  return compute_attribution_report(calls, calls, manifest)


def _fixture_rows() -> list[dict]:
  calls, _ = build_attribution_fixture()
  return [{"idx": i + 1, "name": c.name, "duration_us": c.duration_us}
          for i, c in enumerate(calls)]


def _trace_line(idx: int, name: str, duration_us: float) -> str:
  return ("*** NV %8d %-60s arg  2 mem   5.03 GB tm %10.2fus/  2983.14ms "
          "(      0 GFLOPS    2|2      GB/s) \n" % (idx, name, duration_us))


def test_parse_trace_rows_and_units(tmp_path):
  log = tmp_path / "trace.log"
  log.write_text("".join([
    "scheduled 2 kernels in 1.15 ms | cache hit\n",
    "*** NV         1 copy    4.68 GB,      NV <- DISK:/h            arg  2 mem   5.03 GB tm   2983.14ms/  2983.14ms (      0 GFLOPS    2|2      GB/s) \n",
    _trace_line(2, "q4k_g3_lanemap_gemv_4096_4096", 15.8),
    _trace_line(3, "flash_fused_gmax_combine_f16_32_128", 3.35),
  ]))
  rows = parse_trace(str(log))
  assert [r["idx"] for r in rows] == [1, 2, 3]
  assert rows[0]["name"] == "copy"
  assert abs(rows[0]["duration_us"] - 2983140.0) < 1e-9
  assert abs(rows[1]["duration_us"] - 15.8) < 1e-9
  assert abs(rows[2]["duration_us"] - 3.35) < 1e-9


def test_parse_trace_fails_closed_on_malformed_row(tmp_path):
  log = tmp_path / "bad.log"
  log.write_text("*** NV not-a-row tm 1.0us/ 2.0ms\n")
  with pytest.raises(NVDurationAttachError):
    parse_trace(str(log))
  empty = tmp_path / "empty.log"
  empty.write_text("no rows here\n")
  with pytest.raises(NVDurationAttachError):
    parse_trace(str(empty))


def test_attach_aligned_positional_per_group():
  capture = _fixture_capture()
  rows = _fixture_rows()
  attach = attach_durations(capture, rows)
  assert attach["aligned_groups"] == attach["total_groups"] == 2
  assert attach["matched_calls"] == 8
  assert attach["window"]["rows"] == 8
  calls, _ = build_attribution_fixture()
  for c in calls:
    assert abs(attach["duration_by_call"][c.index] - c.duration_us) < 1e-9


def test_attach_fails_closed_on_signature_mismatch():
  capture = _fixture_capture()
  rows = _fixture_rows()
  # Corrupt position 1 of group 0 only; every other position matches.
  rows[1]["name"] = "SOME_OTHER_KERNEL"
  attach = attach_durations(capture, rows)
  g0 = next(g for g in attach["groups"] if g["group_id"] == 0)
  g1 = next(g for g in attach["groups"] if g["group_id"] == 1)
  assert g0["aligned"] is False
  assert g0["mismatched_positions"] == 1
  assert g0["examples"][0]["dag_name"] == "a_r1"
  assert g0["examples"][0]["trace_name"] == "SOME_OTHER_KERNEL"
  assert g1["aligned"] is True
  # Group 0 gets no durations; group 1 is attached. Nothing is zeroed.
  assert attach["matched_calls"] == 5
  assert all(0 <= c.index <= 2 for c in build_attribution_fixture()[0]
             if c.index not in attach["duration_by_call"])
  assert all(v > 0 for v in attach["duration_by_call"].values())


def test_attach_no_trace_token_fails_closed():
  capture = _fixture_capture()
  rows = _fixture_rows()
  for r in rows:
    r["name"] = "UNRELATED_KERNEL"
  attach = attach_durations(capture, rows)
  assert attach["matched_calls"] == 0
  assert attach["aligned_groups"] == 0
  assert attach["duration_by_call"] == {}


def test_select_token_window_takes_last_match():
  rows = [{"idx": i, "name": name, "duration_us": float(i)}
          for i, name in enumerate(["prefill", "a", "b", "a", "b"])]
  assert select_token_window(rows, ["a", "b"]) == (3, 5)
  # No exact match: the window with the most matches (2/3 at rows 1..4) wins.
  assert select_token_window(rows, ["a", "b", "c"]) == (1, 4)
  assert select_token_window(rows, ["prefill", "a"]) == (0, 2)


def test_select_token_window_prefers_max_matches():
  rows = [{"idx": i, "name": name, "duration_us": float(i)}
          for i, name in enumerate(["x", "a", "b", "c", "a", "b"])]
  # Exact match on ["a","b","c"] is unique at rows 1..4.
  assert select_token_window(rows, ["a", "b", "c"]) == (1, 4)
  # Ties resolve to the LAST window; a partial match still beats nothing.
  assert select_token_window(rows, ["a", "b"]) == (4, 6)
  assert select_token_window(rows, ["a", "z"]) == (4, 6)  # 1/2 beats 0
  assert select_token_window(rows, ["z", "z"]) is None  # nothing matches


def test_emit_duration_dag_records_schema_digest_and_post_split_names(tmp_path):
  capture = _fixture_capture()
  rows = _fixture_rows()
  attach = attach_durations(capture, rows)
  dag = build_duration_dag(capture, attach, capture_path="capture.json",
                           capture_sha256="c" * 64, trace_path="trace.log",
                           trace_sha256="t" * 64, commit="deadbeef")
  assert dag["schema"] == DAG_SCHEMA
  assert dag["node_count"] == 8
  assert dag["duration_attach"]["aligned"] is True
  assert dag["unknown_node_count"] == 0
  assert dag["cross_group_edge_count"] == sum(1 for e in capture["arms"]["physical"]["edges"]
                                              if e["crosses_group"])
  body = {k: v for k, v in dag.items() if k != "schema_digest"}
  assert dag["schema_digest"] == hashlib.sha256(
    json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
  # Node names are the capture's post-split names, and durations match.
  by_id = {n["id"]: n for n in capture["arms"]["physical"]["nodes"]}
  for n in dag["nodes"]:
    assert n["name"] == by_id[n["id"]]["name"]
    assert n["duration_us"] == pytest.approx(attach["duration_by_call"][n["id"]], abs=1e-6)
  # All fixture names classify as kernel names (no pre-split placeholder).
  for n in dag["nodes"]:
    assert n["name"]


def test_emit_duration_dag_fail_closed_marks_unknown_never_zeroes(tmp_path):
  capture = _fixture_capture()
  rows = _fixture_rows()
  rows[5]["name"] = "BAD_KERNEL"  # group 1 position 2
  attach = attach_durations(capture, rows)
  dag = build_duration_dag(capture, attach, commit="x")
  assert dag["duration_attach"]["aligned"] is False
  assert dag["duration_attach"]["aligned_groups"] == 1
  assert dag["unknown_node_count"] == 5
  unknown = [n for n in dag["nodes"] if n["duration_us"] is None]
  assert len(unknown) == 5
  assert all(3 <= n["id"] <= 7 for n in unknown)
  assert all(n["duration_us"] is not None and n["duration_us"] > 0
             for n in dag["nodes"] if n["id"] < 3)


def test_run_synthetic_emits_scan_consumable_dag():
  dag = run_synthetic()
  assert dag["node_count"] == 8
  assert all(n["duration_us"] is not None and n["duration_us"] > 0 for n in dag["nodes"])
  ids = [n["id"] for n in dag["nodes"]]
  assert ids == list(range(len(ids)))
