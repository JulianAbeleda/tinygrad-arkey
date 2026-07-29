"""Opt-in validator/selector for immutable LLM generated artifacts.

It deliberately does not execute an artifact or wire itself into production
dispatch.  Its only fallback verdict is generic tinygrad lowering.
"""
from __future__ import annotations

import argparse, hashlib, json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

VERDICTS = frozenset({"SEARCH_GENERATED_REPRODUCIBLE", "SEARCH_GENERATED_ATTESTED"})
GENERIC_VERDICT = "TINYGRAD_GENERIC_GENERATED"
_PROVENANCE_FIELDS = frozenset({"schema_version", "route_id", "workload_role", "target_backend", "target_architecture", "shape_quant_guards", "search_space_id", "search_space_sha256", "search_request_digest", "search_system", "search_revision", "search_run_id", "search_timestamp", "objective", "budget", "candidate_count", "selected_candidate_rank", "selected_objective_values", "selected_plan", "selected_plan_sha256", "exporter_name", "exporter_revision", "generated_path", "generated_sha256", "declared_reusable_primitives", "forbidden_fallback_kernel_identities", "correctness_evidence", "performance_evidence", "runtime_trace_schema", "expected_runtime_identity", "recovery_commit", "manual_post_edit"})

class GeneratedArtifactError(ValueError): pass

def _sha256_bytes(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def _canonical_hash(value: Any) -> str: return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode())
def _digest(value: Any) -> bool:
  return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value) and len(set(value)) > 1
def _text(value: Any) -> bool: return isinstance(value, str) and bool(value.strip()) and value.strip().lower() not in {"none", "todo", "placeholder"}

@dataclass(frozen=True)
class ArtifactSelection:
  artifact_path: str | None
  trace: Mapping[str, Any]
  @property
  def uses_generated_artifact(self) -> bool: return self.artifact_path is not None

@dataclass(frozen=True)
class GeneratedPlanDispatch:
  """Result of the core generated-plan seam.

  The seam deliberately accepts an executor callback instead of importing an
  artifact or any route module.  A production route can provide a generated
  plan executor later; until then every request follows its normal fallback.
  """
  used_generated_plan: bool
  trace: Mapping[str, Any]

def _catalog_and_root(catalog_path: str | Path | None) -> tuple[Mapping[str, Any], Path]:
  path = Path(catalog_path) if catalog_path else Path(__file__).with_name("generated") / "catalog.json"
  try: catalog = json.loads(path.read_text())
  except (OSError, json.JSONDecodeError) as exc: raise GeneratedArtifactError(f"cannot read generated catalog: {exc}") from exc
  if catalog.get("schema") != "tinygrad.llm.generated-artifacts/v1" or not isinstance(catalog.get("artifacts"), list):
    raise GeneratedArtifactError("unsupported generated-artifact catalog schema")
  return catalog, path.parent.resolve()

def _generated_file(root: Path, relative: Any) -> Path:
  if not isinstance(relative, str) or not relative: raise GeneratedArtifactError("generated path must be a nonempty relative path")
  candidate = (root / relative).resolve()
  if root not in candidate.parents: raise GeneratedArtifactError(f"generated path escapes catalog root: {relative}")
  if not candidate.is_file(): raise GeneratedArtifactError(f"generated file missing: {relative}")
  return candidate

def _verified_file(root: Path, relative: Any, digest: Any, label: str) -> tuple[Path, bytes]:
  path = _generated_file(root, relative)
  data = path.read_bytes()
  if not isinstance(digest, str) or _sha256_bytes(data) != digest: raise GeneratedArtifactError(f"{label} hash drift: {relative}")
  return path, data

def _verify_evidence(root: Path, records: Any, kind: str) -> None:
  if not isinstance(records, list) or not records: raise GeneratedArtifactError(f"{kind} evidence is required")
  for evidence in records:
    if not isinstance(evidence, Mapping): raise GeneratedArtifactError(f"{kind} evidence must be an object")
    _verified_file(root, evidence.get("path"), evidence.get("sha256"), f"{kind} evidence")

def _verify_entry(entry: Mapping[str, Any], root: Path) -> None:
  required = {"route_id", "verdict", "default", "target_guard", "shape_guard", "plan_path", "plan_sha256", "artifact_path", "artifact_sha256", "provenance_path", "provenance_sha256", "manual_post_edit"}
  if missing := sorted(required - entry.keys()): raise GeneratedArtifactError(f"catalog entry missing fields: {missing}")
  if entry["verdict"] not in VERDICTS: raise GeneratedArtifactError("catalog verdict is not a generated verdict")
  if entry["manual_post_edit"] is not False: raise GeneratedArtifactError("catalog manual_post_edit must be false")
  _, plan_bytes = _verified_file(root, entry["plan_path"], entry["plan_sha256"], "plan")
  artifact_path, artifact_bytes = _verified_file(root, entry["artifact_path"], entry["artifact_sha256"], "artifact")
  if artifact_path.suffix == ".py" and not artifact_bytes.startswith(b"# @generated"):
    raise GeneratedArtifactError("generated Python artifact lacks @generated marker")
  _, provenance_bytes = _verified_file(root, entry["provenance_path"], entry["provenance_sha256"], "provenance")
  try: plan, provenance = json.loads(plan_bytes), json.loads(provenance_bytes)
  except json.JSONDecodeError as exc: raise GeneratedArtifactError("plan and provenance must be JSON") from exc
  if not isinstance(provenance, Mapping) or _PROVENANCE_FIELDS - provenance.keys(): raise GeneratedArtifactError("provenance missing required contract fields")
  if provenance["manual_post_edit"] is not False: raise GeneratedArtifactError("provenance manual_post_edit must be false")
  for field in ("route_id", "workload_role", "target_backend", "target_architecture", "search_space_id", "search_system",
                "search_revision", "search_run_id", "search_timestamp", "objective", "exporter_name", "exporter_revision",
                "generated_path", "runtime_trace_schema", "expected_runtime_identity", "recovery_commit"):
    if not _text(provenance[field]): raise GeneratedArtifactError(f"invalid provenance field: {field}")
  for field in ("search_space_sha256", "search_request_digest", "selected_plan_sha256", "generated_sha256"):
    if not _digest(provenance[field]): raise GeneratedArtifactError(f"invalid provenance digest: {field}")
  try: datetime.fromisoformat(provenance["search_timestamp"].replace("Z", "+00:00"))
  except ValueError as exc: raise GeneratedArtifactError("invalid provenance search_timestamp") from exc
  for field in ("candidate_count", "selected_candidate_rank"):
    if type(provenance[field]) is not int or provenance[field] <= 0: raise GeneratedArtifactError(f"invalid provenance field: {field}")
  if provenance["selected_candidate_rank"] > provenance["candidate_count"]: raise GeneratedArtifactError("selected candidate rank exceeds population")
  if not isinstance(provenance["budget"], (int, Mapping)) or not provenance["budget"]:
    raise GeneratedArtifactError("invalid provenance field: budget")
  if not isinstance(provenance["selected_objective_values"], Mapping) or not provenance["selected_objective_values"]:
    raise GeneratedArtifactError("selected objective values are required")
  for field in ("declared_reusable_primitives", "forbidden_fallback_kernel_identities"):
    if not isinstance(provenance[field], list) or any(not _text(value) for value in provenance[field]):
      raise GeneratedArtifactError(f"invalid provenance field: {field}")
  if not isinstance(entry["target_guard"], Mapping) or not isinstance(entry["shape_guard"], Mapping):
    raise GeneratedArtifactError("target and shape guards must be objects")
  if set(("backend", "architecture")) - entry["target_guard"].keys():
    raise GeneratedArtifactError("target guard must pin backend and architecture")
  if provenance["target_backend"] != entry["target_guard"]["backend"] or provenance["target_architecture"] != entry["target_guard"]["architecture"] or provenance["shape_quant_guards"] != entry["shape_guard"]:
    raise GeneratedArtifactError("provenance guard binding mismatch")
  if provenance["route_id"] != entry["route_id"] or provenance["generated_path"] != entry["artifact_path"]: raise GeneratedArtifactError("provenance binding mismatch")
  if provenance["expected_runtime_identity"] != entry["route_id"]: raise GeneratedArtifactError("runtime identity binding mismatch")
  if provenance["generated_sha256"] != entry["artifact_sha256"] or provenance["selected_plan_sha256"] != entry["plan_sha256"]: raise GeneratedArtifactError("provenance digest mismatch")
  if provenance["selected_plan"] != plan: raise GeneratedArtifactError("provenance selected plan drift")
  _verify_evidence(root, provenance["correctness_evidence"], "correctness")
  _verify_evidence(root, provenance["performance_evidence"], "performance")

def validate_catalog(catalog_path: str | Path | None = None) -> None:
  """Fail closed if any catalog record or its provenance chain is invalid."""
  catalog, root = _catalog_and_root(catalog_path)
  seen = set()
  for entry in catalog["artifacts"]:
    if not isinstance(entry, Mapping): raise GeneratedArtifactError("catalog entry must be an object")
    key = (entry.get("route_id"), _canonical_hash(entry.get("target_guard")), _canonical_hash(entry.get("shape_guard")))
    if key in seen: raise GeneratedArtifactError("ambiguous duplicate generated artifact")
    seen.add(key); _verify_entry(entry, root)

def load_generated_artifact(route_id: str, target: Mapping[str, Any], shape: Mapping[str, Any], *, catalog_path: str | Path | None = None) -> ArtifactSelection:
  """Return a verified generated artifact or an explicit generic fallback trace."""
  catalog, root = _catalog_and_root(catalog_path)
  matching_route = [e for e in catalog["artifacts"] if isinstance(e, Mapping) and e.get("route_id") == route_id]
  for entry in matching_route: _verify_entry(entry, root)
  matched = [e for e in matching_route if e["target_guard"] == dict(target) and e["shape_guard"] == dict(shape)]
  if len(matched) > 1: raise GeneratedArtifactError("ambiguous generated artifact selection")
  if matched:
    entry = matched[0]
    return ArtifactSelection(entry["artifact_path"], {"route_id": route_id, "verdict": entry["verdict"], "plan_digest": entry["plan_sha256"], "artifact_digest": entry["artifact_sha256"], "target": dict(target), "shape": dict(shape), "fallback_reason": None})
  reason = "guard_mismatch" if matching_route else "unknown_route"
  return ArtifactSelection(None, {"route_id": route_id, "verdict": GENERIC_VERDICT, "plan_digest": None, "artifact_digest": None, "target": dict(target), "shape": dict(shape), "fallback_reason": reason})

def dispatch_verified_generated_plan(route_id: str, target: Mapping[str, Any], shape: Mapping[str, Any], fallback, *,
                                    execute_plan=None, catalog_path: str | Path | None = None) -> tuple[Any, GeneratedPlanDispatch]:
  """Select a verified generated artifact or invoke the caller's ordinary fallback.

  No artifact is imported or evaluated here. ``execute_plan`` is opt-in and is
  called only after catalog/provenance verification and an exact guard match.
  A missing executor (the current production state) is an explicit fallback,
  so adding generated artifacts alone cannot change runtime behavior.
  """
  if not callable(fallback): raise GeneratedArtifactError("ordinary fallback must be callable")
  if execute_plan is not None and not callable(execute_plan): raise GeneratedArtifactError("generated plan executor must be callable")
  selection = load_generated_artifact(route_id, target, shape, catalog_path=catalog_path)
  if not selection.uses_generated_artifact:
    return fallback(), GeneratedPlanDispatch(False, selection.trace)
  if execute_plan is None:
    trace = dict(selection.trace); trace["fallback_reason"] = "no_generated_plan_executor"
    trace["verdict"] = GENERIC_VERDICT
    return fallback(), GeneratedPlanDispatch(False, trace)
  # The callback receives only an already hash-verified, catalog-relative path
  # plus immutable selection metadata; it owns no fallback authority.
  return execute_plan(selection.artifact_path, selection.trace), GeneratedPlanDispatch(True, selection.trace)

def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="validate LLM generated artifact catalog")
  parser.add_argument("--check", action="store_true", help="validate catalog and all referenced evidence")
  parser.add_argument("--catalog")
  args = parser.parse_args(argv)
  if args.check: validate_catalog(args.catalog); return 0
  parser.error("--check is required")

if __name__ == "__main__": main()
