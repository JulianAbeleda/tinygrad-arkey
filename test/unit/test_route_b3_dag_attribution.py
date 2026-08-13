"""Hermetic CPU-only tests for Route B3.1 aligned logical/physical DAG tooling.

Covers the required cases of the B3 exhaustive execution scope section 8.6
plus the RecordingDepsTracker parity control from the risk register.
"""
import hashlib
import json

import pytest

from extra.llm_research.decode.full_token_dag_capture import RecordingDepsTracker, attach_summary, restrict_dag
from extra.llm_research.decode.route_b3_dag_attribution import (
  B3AttributionError, PLANNER_ALIAS, SEMANTIC, UNKNOWN, CallAccess, CallRecord,
  attach_compiled_descriptors,
  build_attribution_fixture, build_duration_weighted_fixture, build_edges,
  build_partial_overlap_fixture, check_alignment, collect_manifest,
  collect_nv_compiled_descriptors, compute_attribution_report, emit_report,
  to_dag, validate_report,
)


def _pair_edges(edges):
  return {(e["from"], e["to"], e["kind"]) for e in edges}


def test_logical_independent_chains_physically_chained_by_reuse():
  """Case 1: two logical independent chains become physically chained."""
  calls, manifest = build_attribution_fixture()
  logical = build_edges(calls)
  physical = build_edges(calls, manifest)
  # Logical arm: no cross-chain edges at all.
  assert not any((e["from"] < 3 <= e["to"]) or (e["to"] < 3 <= e["from"]) for e in logical)
  assert _pair_edges(logical) == {(0, 1, "RAW"), (1, 2, "RAW"), (3, 4, "RAW"), (4, 5, "RAW"), (5, 6, "RAW"), (6, 7, "RAW")}
  # Physical arm: planner reuse adds the cross-chain edges.
  assert _pair_edges(physical) >= {(0, 3, "WAW"), (1, 3, "WAR"), (0, 4, "RAW"), (1, 4, "WAW"), (1, 5, "RAW"), (2, 4, "WAR"), (2, 5, "WAW"), (2, 6, "RAW")}
  report = compute_attribution_report(calls, calls, manifest)
  aliases = [e for e in report["attributed_edges"] if e["source"] == PLANNER_ALIAS]
  assert {(e["from"], e["to"]) for e in aliases} == {(0, 3), (1, 3), (0, 4), (1, 4), (1, 5), (2, 4), (2, 5), (2, 6)}
  assert report["summary"]["planner_delta_cp_us"] == 28.0
  assert report["summary"]["planner_delta_cp_pct"] == 80.0


def test_semantic_raw_chain_unchanged_by_planning():
  """Case 2: semantic RAW chains are preserved and labeled SEMANTIC."""
  calls, manifest = build_attribution_fixture()
  report = compute_attribution_report(calls, calls, manifest)
  semantic = {(e["from"], e["to"]) for e in report["attributed_edges"] if e["source"] == SEMANTIC}
  assert semantic == {(0, 1), (1, 2), (3, 4), (4, 5), (5, 6), (6, 7)}
  for e in report["attributed_edges"]:
    if e["source"] == SEMANTIC:
      assert e["logical_kind"] == e["kind"]


def test_planner_added_war_and_waw_classification():
  """Case 3: planner-added WAR and WAW are classified PLANNER_ALIAS with ranges."""
  calls, manifest = build_attribution_fixture()
  report = compute_attribution_report(calls, calls, manifest)
  by_pair = {(e["from"], e["to"]): e for e in report["attributed_edges"]}
  war = by_pair[(2, 4)]
  assert war["kind"] == "WAR" and war["source"] == PLANNER_ALIAS
  assert war["arena"] == "arena:0" and war["range"] == [4096, 4608]
  assert "X" in war["logical_buffer_ids"] and "C" in war["logical_buffer_ids"]
  waw = by_pair[(0, 3)]
  assert waw["kind"] == "WAW" and waw["source"] == PLANNER_ALIAS
  assert waw["range"] == [0, 4096] and set(waw["logical_buffer_ids"]) == {"A", "B"}


def test_partially_overlapping_byte_ranges():
  """Case 4: partially overlapping ranges produce an edge with the true overlap."""
  calls, manifest = build_partial_overlap_fixture()
  report = compute_attribution_report(calls, calls, manifest)
  by_pair = {(e["from"], e["to"]): e for e in report["attributed_edges"]}
  # P at [0,2048), Q at [1024,3072): WAW 0->1 with overlap [1024,2048).
  edge = by_pair[(0, 1)]
  assert edge["kind"] == "WAW" and edge["source"] == PLANNER_ALIAS
  assert edge["range"] == [1024, 2048]
  assert set(edge["logical_buffer_ids"]) == {"P", "Q"}


def test_adjacent_non_overlapping_ranges_have_no_edge():
  """Case 5: adjacent non-overlapping physical ranges create no dependency."""
  calls, manifest = build_partial_overlap_fixture()
  edges = build_edges(calls, manifest)
  # R at [4096,5120) and S at [5120,6144): no overlap, no edge between 3 and 4.
  assert not any(e["from"] == 3 and e["to"] == 4 for e in edges)
  # The semantic RAW 1->2 (Q read after Q write) is preserved.
  assert (1, 2, "RAW") in _pair_edges(edges)


def test_stable_signature_is_pointer_independent():
  """Case 6: stable identities depend on sizes/dtypes/order, never pointers."""
  a = CallRecord(0, "k", [CallAccess("A", 0, 512, True)], identity={"outputs": [{"dtype": "float32", "size": 512}]})
  b = CallRecord(0, "k", [CallAccess("different_buf", 0, 512, True)], identity={"outputs": [{"dtype": "float32", "size": 512}]})
  da, db = to_dag([a], []), to_dag([b], [])
  # Same shape/dtype/order -> same identity hash regardless of buffer pointer/name.
  assert da["nodes"][0]["metadata"]["identity_sha256"] == db["nodes"][0]["metadata"]["identity_sha256"]
  c = CallRecord(0, "k", [CallAccess("A", 0, 256, True)], identity={"outputs": [{"dtype": "float32", "size": 256}]})
  assert to_dag([c], [])["nodes"][0]["metadata"]["identity_sha256"] != da["nodes"][0]["metadata"]["identity_sha256"]


def test_call_order_signature_mismatch_fails_closed():
  """Case 7: ordered signature mismatch must fail closed as ALIGNMENT_CONFOUNDED."""
  calls, _ = build_attribution_fixture()
  altered = [CallRecord(c.index, c.name, c.accesses, c.duration_us, c.group,
                        {**c.identity, "program_identity": "different_kernel"}) for c in calls]
  result = check_alignment(calls, altered)
  assert result["aligned"] is False
  assert "ALIGNMENT_CONFOUNDED" in result["reason"]
  with pytest.raises(B3AttributionError):
    compute_attribution_report(calls, altered, {})


def test_unknown_dependency_propagation():
  """Case 8: UNKNOWN nodes propagate to incident edges; never assumed independent."""
  calls, manifest = build_attribution_fixture()
  calls[3].unknown = True
  report = compute_attribution_report(calls, calls, manifest)
  unknown = [e for e in report["attributed_edges"] if e["source"] == UNKNOWN]
  assert all(e["from"] == 3 or e["to"] == 3 for e in unknown)
  assert report["summary"]["unknown_dep_node_count"] == 1
  # A fully edge-less DAG is all-UNKNOWN, not all-independent.
  assert attach_summary(to_dag(calls, []))["summary"]["independence_assumption"] is True


def test_cross_group_edge_preservation():
  """Case 9: cross-group edges are preserved and marked in both arms."""
  calls, manifest = build_attribution_fixture()
  report = compute_attribution_report(calls, calls, manifest)
  cross = [e for e in report["attributed_edges"] if e["crosses_group"]]
  assert {(e["from"], e["to"]) for e in cross} == {(0, 3), (1, 3), (0, 4), (1, 4), (1, 5), (2, 4), (2, 5), (2, 6)}
  # Restriction to a group keeps only intra-group edges.
  g0 = restrict_dag(report["arms"]["physical"], 0)
  assert {(e["from"], e["to"]) for e in g0["edges"]} == {(0, 1), (1, 2)}
  g1 = restrict_dag(report["arms"]["physical"], 1)
  assert {(e["from"], e["to"]) for e in g1["edges"]} == {(3, 4), (4, 5), (5, 6), (6, 7)}


def test_duration_weighted_critical_path_not_small_node_swamp():
  """Case 10: many small nodes do not outweigh one expensive branch."""
  calls = build_duration_weighted_fixture()
  edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
  dag = to_dag(calls, [{"from": f, "to": t, "kind": "RAW", "overlaps": []} for f, t in edges])
  metrics = attach_summary(dag)["summary"]
  # Five 1us nodes (5us chain) vs one 50us node: CP is 50, not 55.
  assert metrics["critical_path_us"] == 50.0
  assert metrics["serialized_us"] == 55.0


def test_planner_collector_absent_present_byte_identical():
  """Case 11: installing a manifest collector changes no default planning output."""
  import re
  from tinygrad.dtype import dtypes
  from tinygrad.tensor import Tensor
  from tinygrad.schedule import memory as tmem
  from tinygrad.uop.ops import UOp

  a = Tensor.empty(64, 64, dtype=dtypes.float32, device="CPU")
  linear = (a @ a.T).schedule_linear()
  held: set[UOp] = set()

  collector = __import__("extra.llm_research.decode.route_b3_dag_attribution", fromlist=["PlannerManifestCollector"]).PlannerManifestCollector()
  token = tmem._memory_manifest_collectors.set((collector,))
  try:
    with_collector = tmem.memory_plan_rewrite(linear, held)
  finally:
    tmem._memory_manifest_collectors.reset(token)
  without_collector = tmem.memory_plan_rewrite(linear, held)
  # UOps do not define value equality and Ops.UNIQUE renders a global counter
  # that advances between runs. Normalize those counters: everything else must
  # be byte-identical (arena sizes, offsets, substitutions).
  normalize = lambda text: re.sub(r"UOp\(Ops.UNIQUE, dtypes.void, arg=\d+", "UOp(Ops.UNIQUE, dtypes.void, arg=#", text)
  assert normalize(str(with_collector)) == normalize(str(without_collector))
  assert len(with_collector.src) == len(without_collector.src)
  assert len(collector.manifest) > 0
  entry = next(iter(collector.manifest.values()))
  assert entry["placement"] == "observed" and entry["arena"] is not None and entry["offset"] is not None
  # Placement evidence is deterministic: a second collector sees the same manifest.
  collector2 = __import__("extra.llm_research.decode.route_b3_dag_attribution", fromlist=["PlannerManifestCollector"]).PlannerManifestCollector()
  token2 = tmem._memory_manifest_collectors.set((collector2,))
  try:
    tmem.memory_plan_rewrite(linear, held)
  finally:
    tmem._memory_manifest_collectors.reset(token2)
  assert collector.manifest == collector2.manifest


def test_schema_validation_and_deterministic_emission():
  """Case 12: JSON schema validation and deterministic output/hash."""
  calls, manifest = build_attribution_fixture()
  report = compute_attribution_report(calls, calls, manifest)
  validate_report(report)
  text1 = emit_report(report)
  text2 = emit_report(compute_attribution_report(calls, calls, manifest))
  assert text1 == text2
  parsed = json.loads(text1)
  validate_report(parsed)
  bad = json.loads(text1)
  bad["attributed_edges"][0]["source"] = "PLANET"
  with pytest.raises(B3AttributionError):
    validate_report(bad)
  bad2 = json.loads(text1)
  bad2["alignment"] = {"aligned": False}
  with pytest.raises(B3AttributionError):
    validate_report(bad2)


class _StubBuf:
  def __init__(self, base, offset, nbytes):
    self.base, self.offset, self.nbytes = base, offset, nbytes


def test_edges_parity_with_recording_deps_tracker():
  """Risk-control: our range-aware builder matches canonical DepsTracker semantics."""
  calls, _ = build_attribution_fixture()
  tracker = RecordingDepsTracker()
  for call in calls:
    bufs, write_idx = [], []
    for i, acc in enumerate(call.accesses):
      bufs.append(_StubBuf(acc.buf, acc.offset, acc.nbytes))
      if acc.write:
        write_idx.append(i)
    tracker.access_resources(bufs, write_idx, call.index)
  canonical = {(f, t, k) for f, t, k in tracker.edges}
  ours = _pair_edges(build_edges(calls))
  assert ours == canonical


def _valid_descriptor(index: int) -> dict:
  return {
    "binary_sha256": "%064x" % (index + 1),
    "grid": [1, 2, 3],
    "block": [4, 5, 6],
    "registers_per_thread": 32,
    "static_smem_bytes": 1024,
    "dynamic_smem_bytes": 0,
    "local_mem_bytes": 512,
  }


def test_attach_compiled_descriptors_exact_set_contract():
  """Compiled descriptors are keyed by occurrence index: missing/extra/partial
  rows raise; names never substitute for occurrence identity."""
  calls = [CallRecord(i, "k%d" % i, []) for i in range(4)]
  rows = attach_compiled_descriptors(calls, {i: _valid_descriptor(i) for i in range(4)})
  assert [r["index"] for r in rows] == [0, 1, 2, 3]
  assert all(set(r["descriptor"]) == {"binary_sha256", "grid", "block", "registers_per_thread",
                                      "static_smem_bytes", "dynamic_smem_bytes", "local_mem_bytes"}
             for r in rows)
  # Missing occurrence.
  with pytest.raises(B3AttributionError):
    attach_compiled_descriptors(calls, {i: _valid_descriptor(i) for i in range(3)})
  # Extra occurrence.
  with pytest.raises(B3AttributionError):
    attach_compiled_descriptors(calls, {i: _valid_descriptor(i) for i in range(4)} | {9: _valid_descriptor(9)})


def test_attach_compiled_descriptors_partial_and_malformed_rows_raise():
  calls = [CallRecord(i, "k%d" % i, []) for i in range(2)]
  descriptors = {i: _valid_descriptor(i) for i in range(2)}
  # Partial row (missing a required field).
  bad = {i: dict(d) for i, d in descriptors.items()}
  del bad[0]["local_mem_bytes"]
  with pytest.raises(B3AttributionError):
    attach_compiled_descriptors(calls, bad)
  # Extra field in a row.
  bad = {i: dict(d) for i, d in descriptors.items()}
  bad[0]["extra"] = 1
  with pytest.raises(B3AttributionError):
    attach_compiled_descriptors(calls, bad)
  # Non-64-hex sha.
  bad = {i: dict(d) for i, d in descriptors.items()}
  bad[0]["binary_sha256"] = "not-a-sha"
  with pytest.raises(B3AttributionError):
    attach_compiled_descriptors(calls, bad)
  # Negative scalar.
  bad = {i: dict(d) for i, d in descriptors.items()}
  bad[0]["registers_per_thread"] = -1
  with pytest.raises(B3AttributionError):
    attach_compiled_descriptors(calls, bad)
  # Non-3 grid/block.
  bad = {i: dict(d) for i, d in descriptors.items()}
  bad[0]["grid"] = [1, 2]
  with pytest.raises(B3AttributionError):
    attach_compiled_descriptors(calls, bad)
  bad = {i: dict(d) for i, d in descriptors.items()}
  bad[0]["block"] = [1, 2, 3, 4]
  with pytest.raises(B3AttributionError):
    attach_compiled_descriptors(calls, bad)
  # A name collision must not satisfy two occurrences: index 0 and 1 share a
  # name, so only the full per-index set passes.
  same_name = [CallRecord(0, "shared", []), CallRecord(1, "shared", [])]
  rows = attach_compiled_descriptors(same_name, descriptors)
  assert [r["descriptor"] for r in rows] == [descriptors[0], descriptors[1]]


def test_collect_nv_compiled_descriptors_reads_nvprogram_objects():
  """The emitter sources exactly one compiled descriptor per occurrence from
  the NV compiled kernel objects (launch geometry + regs/smem/lmem + binary)."""
  from types import SimpleNamespace
  from tinygrad.engine.realize import runtime_cache
  from tinygrad.uop.ops import Ops

  class _PrgInfo:
    name = "q4k_g3_lanemap_gemv_4096_4096"
    global_size = (512, 1, 1)
    local_size = (32, 1, 1)
    def launch_dims(self, var_vals):
      return self.global_size, self.local_size

  def fake_call(name, key, lib):
    ast = SimpleNamespace(op=Ops.PROGRAM, key=key, arg=_PrgInfo())
    ast.arg.name = name
    return SimpleNamespace(src=(ast,))

  linear = SimpleNamespace(src=(
    fake_call("q4k_g3_lanemap_gemv_4096_4096", b"k0", b"binary-0"),
    fake_call("flash_fused_gmax_combine_f16_32_128", b"k1", b"binary-1"),
  ))
  runtime_cache[(b"k0", "NV")] = SimpleNamespace(lib=b"binary-0", regs_usage=40,
                                                 shmem_usage=2048, lcmem_usage=128)
  runtime_cache[(b"k1", "NV")] = SimpleNamespace(lib=b"binary-1", regs_usage=32,
                                                 shmem_usage=4096, lcmem_usage=0)
  try:
    descriptors = collect_nv_compiled_descriptors(linear)
  finally:
    runtime_cache.pop((b"k0", "NV"), None)
    runtime_cache.pop((b"k1", "NV"), None)
  assert set(descriptors) == {0, 1}
  d0 = descriptors[0]
  assert d0["grid"] == [512, 1, 1] and d0["block"] == [32, 1, 1]
  assert d0["registers_per_thread"] == 40
  assert d0["static_smem_bytes"] == 2048
  assert d0["dynamic_smem_bytes"] == 0
  assert d0["local_mem_bytes"] == 128
  assert d0["binary_sha256"] == hashlib.sha256(b"binary-0").hexdigest()
  assert len(d0["binary_sha256"]) == 64
  # Distinct occurrences keep distinct descriptors even when names collide.
  assert descriptors[1]["binary_sha256"] == hashlib.sha256(b"binary-1").hexdigest()


def test_collect_nv_compiled_descriptors_fails_closed_without_kernel():
  from types import SimpleNamespace
  from tinygrad.uop.ops import Ops

  ast = SimpleNamespace(op=Ops.PROGRAM, key=b"missing", arg=SimpleNamespace(name="orphan"))
  linear = SimpleNamespace(src=(SimpleNamespace(src=(ast,)),))
  with pytest.raises(B3AttributionError):
    collect_nv_compiled_descriptors(linear)
  # Non-PROGRAM calls (no compiled kernel) also fail closed, never skipped.
  ast2 = SimpleNamespace(op=Ops.COPY, key=b"copy", arg=None)
  linear2 = SimpleNamespace(src=(SimpleNamespace(src=(ast2,)),))
  with pytest.raises(B3AttributionError):
    collect_nv_compiled_descriptors(linear2)
