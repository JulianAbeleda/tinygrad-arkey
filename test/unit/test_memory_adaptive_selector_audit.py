import json
from pathlib import Path

from extra.qk.memory_adaptive_selector_audit import audit_repository, main, stable_json


def write(tmp_path: Path, name: str, source: str) -> str:
  path = tmp_path/name
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(source)
  return name


def rules(report): return {(x["rule_id"], x["line"], x["gating"]) for x in report["findings"]}


def test_identity_names_profiles_and_size_labels_are_semantic_findings(tmp_path):
  path = write(tmp_path, "runtime.py", '''
def choose(model_name, profile_id, candidate):
  if model_name == "Qwen-14B": return candidate
  return candidate if profile_id else None
''')
  report = audit_repository(tmp_path, [path])
  assert report["status"] == "FAIL"
  assert ("FORBIDDEN_IDENTITY_SELECTOR", 3, True) in rules(report)
  assert ("FORBIDDEN_PARAMETER_SIZE_LABEL", 3, True) in rules(report)


def test_fixture_labels_are_visible_but_do_not_fail_gate(tmp_path):
  path = write(tmp_path, "test/benchmark_fixture.py", 'def pick(profile_id):\n  return 1 if profile_id == "qwen-14B" else 0\n')
  report = audit_repository(tmp_path, [path])
  assert report["status"] == "PASS" and report["findings"]
  assert all(not x["gating"] and x["classification"] == "evidence_fixture" for x in report["findings"])


def test_exact_candidate_constraint_is_allowed_but_unmatched_target_is_forbidden(tmp_path):
  path = write(tmp_path, "runtime.py", '''
good = CandidateSpec("c", strategy, ("r",), target_requirements={"architecture": "gfx1100", "wave_size": 32})
bad = {"backend": "AMD", "architecture": "gfx1100"}
if device.architecture == "gfx1100": choose_route()
''')
  report = audit_repository(tmp_path, [path])
  allowed = [x for x in report["findings"] if x["rule_id"] == "ALLOWED_EXACT_CANDIDATE_CONSTRAINT"]
  forbidden = [x for x in report["findings"] if x["rule_id"] == "UNMATCHED_TARGET_FACT" and x["gating"]]
  assert allowed and all(x["classification"] == "candidate_capability_constraint" for x in allowed)
  assert len(forbidden) == 3 and report["status"] == "FAIL"


def test_fixed_and_duplicated_memory_thresholds_are_deterministic(tmp_path):
  a = write(tmp_path, "a.py", "def fit(free_vram):\n  return free_vram >= 24 * 1024\n")
  b = write(tmp_path, "b.py", "def fit(memory_budget):\n  return memory_budget >= 24 * 1024\n")
  first = audit_repository(tmp_path, [b, a])
  second = audit_repository(tmp_path, [a, b])
  assert stable_json(first) == stable_json(second)
  assert first["counts"]["FIXED_MEMORY_TIER"] == 2
  assert first["counts"]["DUPLICATED_MEMORY_THRESHOLD"] == 2


def test_cli_is_machine_readable_and_fails_live_selector_gate(tmp_path, capsys):
  path = write(tmp_path, "runtime.py", 'def route(model_path):\n  return "x" if model_path.endswith("8B.gguf") else "y"\n')
  assert main(["--root", str(tmp_path), path]) == 1
  payload = json.loads(capsys.readouterr().out)
  assert payload["schema"].endswith(".v1") and payload["status"] == "FAIL"


def test_comments_and_informational_strings_are_not_selectors(tmp_path):
  path = write(tmp_path, "runtime.py", '# 14B evidence on gfx1100\nMESSAGE = "8B benchmark profile"\ndef f(size): return size * 2\n')
  report = audit_repository(tmp_path, [path])
  assert report["status"] == "PASS" and report["findings"] == []


def test_identity_passed_to_admission_surface_is_forbidden(tmp_path):
  path = write(tmp_path, "runtime.py", "def f(model_profile):\n  return admit_q6k_wmma(model_profile=model_profile)\n")
  report = audit_repository(tmp_path, [path])
  assert report["status"] == "FAIL"
  assert any(x["rule_id"] == "FORBIDDEN_IDENTITY_SELECTOR" and "admit_q6k_wmma" in x["message"] for x in report["findings"])


def test_separately_declared_exact_requirements_matched_to_scanned_facts_are_allowed(tmp_path):
  path = write(tmp_path, "runtime.py", '''
REQUIRED = {"backend": "AMD", "architecture": "gfx1100"}
def eligible(target_capabilities):
  return requirements_met(REQUIRED, target_capabilities)
''')
  report = audit_repository(tmp_path, [path])
  assert report["status"] == "PASS"
  assert sum(x["rule_id"] == "ALLOWED_EXACT_CANDIDATE_CONSTRAINT" for x in report["findings"]) == 2


def test_route_manifest_only_exempts_specific_provenance_fields(tmp_path):
  path = write(tmp_path, "extra/qk/route_manifest.py", '''
ROUTES = {"prefill_14b_runtime": {"profile_id": "qwen_14b_evidence", "note": "14B measurement"}}
def select(route_id): return ROUTES[route_id]
''')
  report = audit_repository(tmp_path, [path])
  live = [x for x in report["findings"] if x["gating"]]
  metadata = [x for x in report["findings"] if x["classification"] == "evidence_metadata"]
  assert report["status"] == "FAIL" and live and metadata


def test_route_manifest_compatibility_aliases_remain_reported_but_non_gating(tmp_path):
  path = write(tmp_path, "extra/qk/route_manifest.py", '''
ALIASES = ({"canonical_route_id": "decode_flash_live_split_g4_kvboth",
            "compatibility_aliases": ("decode_flash_live_split_g4_8b_kvboth",)},)
''')
  report = audit_repository(tmp_path, [path])
  assert report["status"] == "PASS" and report["findings"]
  assert all(not row["gating"] and row["classification"] == "evidence_metadata" for row in report["findings"])


def test_environment_route_and_admission_controls_are_gating(tmp_path):
  path = write(tmp_path, "runtime.py", '''
import os
from tinygrad import getenv
PREFILL_GRAPH_GEMM = bool(getenv("PREFILL_GRAPH_GEMM", 0))
def route(device_facts, policy):
  if os.environ.get("TC_ATTN", "0") and PREFILL_GRAPH_GEMM:
    return "graph-gemm"
  return policy
''')
  report = audit_repository(tmp_path, [path])
  ids = {x["rule_id"] for x in report["findings"] if x["gating"]}
  assert {"FORBIDDEN_ENVIRONMENT_CONTROL", "FORBIDDEN_MODULE_GLOBAL_CONTROL"} <= ids
  assert report["status"] == "FAIL"


def test_diagnostic_environment_text_and_fixture_controls_are_non_gating(tmp_path):
  path = write(tmp_path, "tests/evidence_fixture.py", '''
import os
PREFILL_TC_ATTN = os.environ.get("PREFILL_TC_ATTN", "0")
def report():
  print("PREFILL_TC_ATTN", PREFILL_TC_ATTN)
''')
  report = audit_repository(tmp_path, [path])
  assert report["status"] == "PASS"
  assert report["findings"] and all(not x["gating"] for x in report["findings"])
