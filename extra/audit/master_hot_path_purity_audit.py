#!/usr/bin/env python3
"""Strict final-master hot-path provenance audit.

Final mode consumes the production generated-artifact catalog and calls its
runtime validator first.  ``--current-baseline`` is deliberately separate: it
walks the old runtime dispatch roots to preserve an honest migration-debt view.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, pathlib, re
from enum import Enum
from typing import Any, Mapping

from tinygrad.llm import generated_runtime

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA = "tinygrad.master-hot-path-purity-audit.v1"
DIGEST = re.compile(r"^[0-9a-f]{64}$")
EXTRA = re.compile(r"extra\.llm_research")
HAND = re.compile(r"(?:custom_kernel|Ops\.CUSTOM(?:I)?\b|\.load\(|\.store\(|UOp(?:\.|\()|WMMA|PACKED_WMMA_GEOM)")

class Classification(str, Enum):
  SEARCH_GENERATED_REPRODUCIBLE = "SEARCH_GENERATED_REPRODUCIBLE"
  SEARCH_GENERATED_ATTESTED = "SEARCH_GENERATED_ATTESTED"
  TINYGRAD_GENERIC_GENERATED = "TINYGRAD_GENERIC_GENERATED"
  HAND_AUTHORED_CUSTOM = "HAND_AUTHORED_CUSTOM"
  UNPROVEN = "UNPROVEN"

# Actual production dispatch modules, rather than an emitter named by a
# manifest. Literal dynamic imports in route_ops are resolved by _imports.
CURRENT_DEFAULTS = {
  "decode_q4k_g3_generated": ("tinygrad/llm/decode_routes.py", ["extra/llm_research/gemv_g3_codegen_lowering.py"]),
  "decode_q6k_coop_generated": ("tinygrad/llm/decode_routes.py", ["extra/llm_research/q6k_route_spec.py"]),
  "decode_flash_live_split_g4_kvboth": ("tinygrad/llm/decode_routes.py", ["extra/llm_research/flash_kernels.py"]),
  "decode_flash_live_split_g5_kvboth": ("tinygrad/llm/decode_routes.py", ["extra/llm_research/flash_kernels.py"]),
  "prefill_flash_attention_generated": ("tinygrad/llm/fused_attention.py", ["tinygrad/schedule/wmma/kernels.py"]),
  "prefill_wmma_lds_dbuf_generated": ("tinygrad/llm/prefill_routes.py", []),
  "prefill_q4k_direct_tile4x4_default": ("tinygrad/llm/prefill_routes.py", ["extra/llm_research/prefill/q4k_prefill_route_spec.py"]),
  "prefill_q6k_direct_generated": ("tinygrad/llm/prefill_routes.py", ["extra/llm_research/prefill/q6k_prefill_route_spec.py"]),
  "packed_wmma_prefill_generated": ("tinygrad/llm/prefill_routes.py", ["extra/llm_research/prefill/packed_wmma_prefill_candidates.py"]),
}

def _sha(path: pathlib.Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _rel(root: pathlib.Path, name: str) -> pathlib.Path | None:
  try: path = (root / name).resolve(); path.relative_to(root.resolve()); return path
  except (TypeError, ValueError): return None

def _imports(root: pathlib.Path, path: pathlib.Path) -> set[pathlib.Path]:
  try: tree = ast.parse(path.read_text())
  except (OSError, SyntaxError, UnicodeDecodeError): return set()
  modules: set[str] = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Import): modules.update(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module:
      modules.add(node.module)
      # ``from tinygrad.llm import route_ops`` imports the named child module
      # even though the AST stores only ``tinygrad.llm`` in ``node.module``.
      # Resolve both possibilities and let the filesystem decide which exists.
      modules.update(f"{node.module}.{alias.name}" for alias in node.names if alias.name != "*")
    # route_ops uses importlib.import_module("extra.llm_research...."); this
    # catches only literal module names, never guesses dynamic code.
    elif isinstance(node, ast.Constant) and isinstance(node.value, str) and re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", node.value): modules.add(node.value)
  found = set()
  for module in modules:
    base = root.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
      if candidate.is_file(): found.add(candidate.resolve()); break
  return found

def source_closure(root: pathlib.Path, entrypoints: list[str]) -> tuple[list[str], list[dict[str, str]], list[str]]:
  pending = [_rel(root, name) for name in entrypoints]
  missing = [name for name, path in zip(entrypoints, pending) if path is None or not path.is_file()]
  seen: set[pathlib.Path] = set(); edges: set[tuple[str, str]] = set()
  while pending:
    path = pending.pop()
    if path is None or path in seen or not path.is_file(): continue
    seen.add(path)
    for child in _imports(root, path):
      edges.add((str(path.relative_to(root)), str(child.relative_to(root))))
      if child not in seen: pending.append(child)
  return (sorted(str(path.relative_to(root)) for path in seen),
          [{"from": a, "to": b} for a, b in sorted(edges)], sorted(missing))

def _provenance_errors(root: pathlib.Path, entry: Mapping[str, Any], catalog_root: pathlib.Path) -> list[str]:
  errors = []
  prov_name = entry.get("provenance_path"); prov = catalog_root / prov_name if isinstance(prov_name, str) else None
  try: record = json.loads(prov.read_text()) if prov and prov.is_file() else None
  except json.JSONDecodeError: record = None
  if not isinstance(record, Mapping): return ["invalid provenance record"]
  string_fields = ("route_id", "search_space_id", "search_system", "search_revision", "search_run_id", "search_timestamp", "objective", "exporter_name", "exporter_revision", "generated_path", "runtime_trace_schema", "expected_runtime_identity", "recovery_commit")
  digest_fields = ("search_space_sha256", "search_request_digest", "selected_plan_sha256", "generated_sha256")
  for field in string_fields:
    if not isinstance(record.get(field), str) or not record[field].strip() or record[field].strip().lower() in {"none", "placeholder", "todo"}: errors.append("invalid provenance " + field)
  for field in digest_fields:
    if not isinstance(record.get(field), str) or not DIGEST.fullmatch(record[field]) or set(record[field]) == {"0"}: errors.append("invalid provenance digest " + field)
  for field in ("budget", "candidate_count", "selected_candidate_rank"):
    if not isinstance(record.get(field), int) or isinstance(record[field], bool) or record[field] <= 0: errors.append("invalid provenance " + field)
  if not isinstance(record.get("selected_objective_values"), Mapping) or not record["selected_objective_values"]: errors.append("invalid provenance selected_objective_values")
  if record.get("manual_post_edit") is not False: errors.append("manual_post_edit must be false")
  # Runtime trace must bind exactly the route and the two immutable digests.
  expected = record.get("expected_runtime_identity")
  if expected != entry.get("route_id"): errors.append("runtime trace route identity mismatch")
  if record.get("generated_sha256") != entry.get("artifact_sha256") or record.get("selected_plan_sha256") != entry.get("plan_sha256"):
    errors.append("runtime trace digest binding mismatch")
  return errors

def _catalog_path(root: pathlib.Path, catalog_path: str | pathlib.Path | None) -> pathlib.Path:
  return pathlib.Path(catalog_path).resolve() if catalog_path else root / "tinygrad/llm/generated/catalog.json"

def audit_catalog(root: pathlib.Path = ROOT, catalog_path: str | pathlib.Path | None = None) -> dict[str, Any]:
  """Audit M1's exact catalog schema; no fixture-only schema is accepted."""
  root, path = pathlib.Path(root).resolve(), _catalog_path(pathlib.Path(root).resolve(), catalog_path)
  errors: list[str] = []
  try:
    generated_runtime.validate_catalog(path)
  except generated_runtime.GeneratedArtifactError as exc:
    errors.append("generated_runtime_validation: " + str(exc))
  try: catalog = json.loads(path.read_text())
  except (OSError, json.JSONDecodeError): catalog = {}
  if catalog.get("schema") != "tinygrad.llm.generated-artifacts/v1" or not isinstance(catalog.get("artifacts"), list):
    errors.append("unsupported_generated_artifact_catalog")
  entries = [entry for entry in catalog.get("artifacts", []) if isinstance(entry, Mapping) and entry.get("default") is True]
  optimized = [entry for entry in entries if entry.get("verdict") in generated_runtime.VERDICTS]
  if not optimized: errors.append("no_optimized_generated_routes")
  rows = []
  catalog_root = path.parent
  for entry in sorted(optimized, key=lambda item: str(item.get("route_id", ""))):
    artifact = entry.get("artifact_path")
    artifact_path = (catalog_root / artifact).resolve() if isinstance(artifact, str) else None
    try: artifact_entry = str(artifact_path.relative_to(root)) if artifact_path is not None else None
    except ValueError: artifact_entry = None
    closure, edges, missing = source_closure(root, [artifact_entry] if artifact_entry is not None else [])
    if artifact_entry is None: missing.append(str(artifact))
    row_errors = list(missing) + _provenance_errors(root, entry, catalog_root)
    if any(name.startswith("extra/llm_research/") or EXTRA.search((root / name).read_text(errors="ignore")) for name in closure):
      row_errors.append("production closure depends on extra/llm_research")
    if any(HAND.search((root / name).read_text(errors="ignore")) for name in closure):
      row_errors.append("route-local hand-authored custom-kernel signal")
    classification = entry.get("verdict") if not row_errors else Classification.UNPROVEN.value
    rows.append({"route_id": entry.get("route_id"), "optimized": True, "classification": classification,
                 "closure": closure, "closure_edges": edges, "errors": sorted(set(row_errors)),
                 "runtime_trace": {"route_id": entry.get("route_id"), "plan_digest": entry.get("plan_sha256"), "artifact_digest": entry.get("artifact_sha256")}})
  failures = errors + [row["route_id"] for row in rows if row["classification"] not in generated_runtime.VERDICTS]
  return {"schema": SCHEMA, "mode": "final-master", "deterministic": True,
          "verdict": "MASTER_HOT_PATH_PURITY_PASS" if not failures else "MASTER_HOT_PATH_PURITY_FAIL",
          "errors": sorted(set(errors)), "route_count": len(rows), "failing_optimized_routes": failures, "routes": rows}

def audit_current_baseline(root: pathlib.Path = ROOT) -> dict[str, Any]:
  root = pathlib.Path(root).resolve(); rows = []
  for route_id, (entrypoint, owners) in sorted(CURRENT_DEFAULTS.items()):
    closure, edges, missing = source_closure(root, [entrypoint])
    extra = any(name.startswith("extra/llm_research/") or EXTRA.search((root / name).read_text(errors="ignore")) for name in closure)
    owner_text = "\n".join((_rel(root, p).read_text(errors="ignore") for p in owners if _rel(root, p) and _rel(root, p).is_file() and p in closure))
    hand = bool(HAND.search(owner_text)); errors = list(missing)
    if extra: errors.append("production closure depends on extra/llm_research")
    if not owners: errors.append("missing generated search/provenance chain")
    rows.append({"route_id": route_id, "optimized": True, "classification": (Classification.HAND_AUTHORED_CUSTOM if hand else Classification.UNPROVEN).value,
                 "closure": closure, "closure_edges": edges, "errors": sorted(set(errors)), "hand_authored_signals": hand})
  return {"schema": SCHEMA, "mode": "current-baseline", "deterministic": True, "verdict": "MASTER_HOT_PATH_PURITY_FAIL",
          "route_count": len(rows), "failing_optimized_routes": [row["route_id"] for row in rows], "routes": rows}

def main() -> None:
  parser = argparse.ArgumentParser(); parser.add_argument("--catalog"); parser.add_argument("--current-baseline", action="store_true"); parser.add_argument("--output")
  args = parser.parse_args(); report = audit_current_baseline(ROOT) if args.current_baseline else audit_catalog(ROOT, args.catalog)
  text = json.dumps(report, sort_keys=True, indent=2) + "\n"
  if args.output: pathlib.Path(args.output).write_text(text)
  print(text, end=""); raise SystemExit(0 if report["verdict"].endswith("PASS") else 1)

if __name__ == "__main__": main()
