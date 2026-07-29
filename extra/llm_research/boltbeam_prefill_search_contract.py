"""Fail-closed M2a contract for the BoltBeam prefill full-kernel search.

This is intentionally a request/checker, not a replacement search engine.  It
keeps the public request and the historical evidence bindings immutable until
BoltBeam can emit an actual ranked search run.
"""
from __future__ import annotations

import argparse, hashlib, json
import re
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "tinygrad.llm.boltbeam-prefill-search-contract.v1"
ROUTE_ID = "prefill_wmma_lds_dbuf_generated"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_REQUIRED_COUPLED_DIMENSIONS = frozenset(("tile_waves_threads", "lds_windows_strides_padding",
  "cooperative_loads", "pipeline_buffer_stage", "dependency_wait_barriers", "residency", "epilogue"))


def _root() -> Path: return Path(__file__).resolve().parents[2]
def _digest(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _read(path: Path) -> Mapping[str, Any]: return json.loads(path.read_text())
def _is_sha256(value: Any) -> bool: return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def check_contract(root: Path | None = None) -> Mapping[str, Any]:
  root = _root() if root is None else Path(root).resolve()
  base = root / "tinygrad/llm/generated"
  request_path = base / "requests/prefill_wmma_lds_dbuf_generated.json"
  blocked_path = base / "provenance/prefill_wmma_lds_dbuf_generated.blocked.json"
  request, blocked = _read(request_path), _read(blocked_path)
  if request.get("schema") != SCHEMA or blocked.get("schema") != SCHEMA:
    raise ValueError("unexpected M2a search contract schema")
  if request.get("route_id") != ROUTE_ID or blocked.get("route_id") != ROUTE_ID:
    raise ValueError("M2a route binding drift")
  if request.get("candidate_space_status") != "MISSING":
    raise ValueError("M2a request must declare the candidate space MISSING")
  if "candidate_dimensions" in request:
    raise ValueError("M2a request must not present singleton candidate dimensions as a search space")
  if set(request.get("required_coupled_dimensions", ())) != _REQUIRED_COUPLED_DIMENSIONS:
    raise ValueError("M2a request must name every undefined coupled search dimension")
  revision = request.get("search_system", {}).get("revision")
  if not isinstance(revision, str) or _GIT_REVISION.fullmatch(revision) is None:
    raise ValueError("M2a search revision must be a lowercase full git revision")
  if blocked.get("status") != "BLOCKED" or blocked.get("default") is not False:
    raise ValueError("blocked M2a contract must remain default:false")
  if not _is_sha256(blocked.get("search_request_sha256")):
    raise ValueError("M2a request digest must be lowercase SHA-256")
  if blocked.get("search_request_sha256") != _digest(request_path):
    raise ValueError("M2a request digest drift")
  if (root / str(blocked.get("missing_worker_path", ""))).is_file():
    raise ValueError("M2a block record is stale: Tinygrad admission worker now exists")
  catalog = _read(base / "catalog.json")
  for entry in catalog.get("artifacts", []):
    if entry.get("route_id") == ROUTE_ID:
      raise ValueError("blocked M2a route must not have a generated catalog artifact")
  for record in blocked.get("historical_evidence", []):
    if not _is_sha256(record.get("sha256")):
      raise ValueError(f"historical evidence digest is not SHA-256: {record.get('path')}")
    path = root / record["path"]
    if not path.is_file() or _digest(path) != record["sha256"]:
      raise ValueError(f"historical evidence hash drift: {record.get('path')}")
  required = {"missing_search_command", "missing_required_inputs", "missing_worker_path", "historical_selected_plan_input"}
  if required - blocked.keys(): raise ValueError("blocked M2a contract omits exact recovery requirements")
  historical = blocked["historical_selected_plan_input"]
  if not isinstance(historical, Mapping) or not _is_sha256(historical.get("sha256")):
    raise ValueError("blocked M2a contract must bind its historical selected-plan input by SHA-256")
  if blocked.get("search_system", {}).get("revision") != revision:
    raise ValueError("blocked M2a search revision drift")
  if blocked_path.read_text() != json.dumps(blocked, indent=2, sort_keys=True) + "\n":
    raise ValueError("blocked M2a record is not deterministically serialized")
  return blocked


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="check the blocked BoltBeam M2a search contract")
  parser.add_argument("--root", type=Path, default=None)
  args = parser.parse_args(argv)
  check_contract(args.root)
  return 0


if __name__ == "__main__": main()
