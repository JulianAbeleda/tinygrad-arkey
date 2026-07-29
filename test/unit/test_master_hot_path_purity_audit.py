import hashlib, json, pathlib
from extra.audit.master_hot_path_purity_audit import audit_catalog, audit_current_baseline

ROOT = pathlib.Path(__file__).resolve().parents[2]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def fixture(tmp_path, *, default=True, artifact_text=b"# @generated\nPLAN = {}\n", mutate=None):
  for directory in ("plans", "artifacts", "provenance", "evidence"): (tmp_path / directory).mkdir()
  plan, artifact = tmp_path / "plans/plan.json", tmp_path / "artifacts/route.py"
  plan.write_text('{"op":"fixture"}\n'); artifact.write_bytes(artifact_text)
  for name in ("correctness.json", "performance.json"): (tmp_path / "evidence" / name).write_text("{}\n")
  record = {"schema_version": 1, "route_id": "fixture", "workload_role": "fixture", "target_backend": "TEST", "target_architecture": "test",
    "shape_quant_guards": {"n": 8}, "search_space_id": "fixture-space", "search_space_sha256": "a"*64, "search_request_digest": "b"*64,
    "search_system": "fixture-search", "search_revision": "r1", "search_run_id": "run-1", "search_timestamp": "2026-01-01T00:00:00Z",
    "objective": "latency", "budget": 1, "candidate_count": 1, "selected_candidate_rank": 1, "selected_objective_values": {"latency": 1},
    "selected_plan": json.loads(plan.read_text()), "selected_plan_sha256": sha(plan), "exporter_name": "fixture-exporter", "exporter_revision": "r1",
    "generated_path": "artifacts/route.py", "generated_sha256": sha(artifact), "declared_reusable_primitives": [], "forbidden_fallback_kernel_identities": [],
    "correctness_evidence": [{"path": "evidence/correctness.json", "sha256": sha(tmp_path / "evidence/correctness.json")}],
    "performance_evidence": [{"path": "evidence/performance.json", "sha256": sha(tmp_path / "evidence/performance.json")}],
    "runtime_trace_schema": "fixture-trace/v1", "expected_runtime_identity": "fixture", "recovery_commit": "abc123", "manual_post_edit": False}
  provenance = tmp_path / "provenance/route.json"; provenance.write_text(json.dumps(record))
  entry = {"route_id": "fixture", "verdict": "SEARCH_GENERATED_REPRODUCIBLE", "default": default, "target_guard": {"backend": "TEST", "architecture": "test"}, "shape_guard": {"n": 8},
    "plan_path": "plans/plan.json", "plan_sha256": sha(plan), "artifact_path": "artifacts/route.py", "artifact_sha256": sha(artifact),
    "provenance_path": "provenance/route.json", "provenance_sha256": sha(provenance), "manual_post_edit": False}
  catalog = {"schema": "tinygrad.llm.generated-artifacts/v1", "artifacts": [entry]}
  if mutate: mutate(catalog, record, artifact)
  # Mutations to provenance represent a rewritten record, not accidental drift.
  provenance.write_text(json.dumps(record)); entry["provenance_sha256"] = sha(provenance)
  path = tmp_path / "catalog.json"; path.write_text(json.dumps(catalog)); return path

def test_real_m1_catalog_passes_and_reports_trace_binding(tmp_path):
  report = audit_catalog(tmp_path, fixture(tmp_path))
  assert report["verdict"] == "MASTER_HOT_PATH_PURITY_PASS"
  assert report["routes"][0]["runtime_trace"]["route_id"] == "fixture"
  assert report["routes"][0]["closure_edges"] == []

def test_empty_or_nondefault_catalog_cannot_vacuously_pass(tmp_path):
  path = tmp_path / "catalog.json"; path.write_text(json.dumps({"schema": "tinygrad.llm.generated-artifacts/v1", "artifacts": []}))
  assert "no_optimized_generated_routes" in audit_catalog(tmp_path, path)["errors"]
  other = tmp_path / "other"; other.mkdir()
  assert "no_optimized_generated_routes" in audit_catalog(other, fixture(other, default=False))["errors"]

def test_drift_and_renamed_hand_builder_fail(tmp_path):
  path = fixture(tmp_path, mutate=lambda _c, _p, artifact: artifact.write_text("# @generated\ndef innocuous_dataclass_name():\n  return UOp.range(8, 0).store(0)\n"))
  report = audit_catalog(tmp_path, path)
  assert report["verdict"] == "MASTER_HOT_PATH_PURITY_FAIL"
  assert any("generated_runtime_validation" in error for error in report["errors"])
  # Rebuild a hash-valid artifact with a renamed builder: naming cannot hide the executable signal.
  fresh = tmp_path / "fresh"; fresh.mkdir()
  path = fixture(fresh, artifact_text=b"# @generated\ndef innocent_descriptor(): return UOp.range(8, 0).store(0)\n")
  report = audit_catalog(fresh, path)
  assert report["routes"][0]["classification"] == "UNPROVEN" or report["verdict"] == "MASTER_HOT_PATH_PURITY_FAIL"

def test_current_baseline_is_explicit_and_fails_with_edges():
  report = audit_current_baseline(ROOT)
  assert report["route_count"] == 9 and report["verdict"] == "MASTER_HOT_PATH_PURITY_FAIL"
  assert all("closure_edges" in route for route in report["routes"])
