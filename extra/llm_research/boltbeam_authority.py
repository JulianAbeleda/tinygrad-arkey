"""Read-only bridge to the authoritative BoltBeam route ledger."""
import json
import os
from pathlib import Path

DEFAULT_LEDGER = Path(__file__).resolve().parents[3] / "BoltBeam" / "boltbeam" / "data" / "nv_sm120_kernel_authority.json"

def load_promoted_routes(path: str | os.PathLike | None = None) -> dict[str, dict]:
  ledger_path = Path(path or os.environ.get("BOLTBEAM_NV_AUTHORITY_LEDGER", DEFAULT_LEDGER))
  data = json.loads(ledger_path.read_text())
  if data.get("schema") != "boltbeam.kernel_authority_ledger.v1": raise ValueError("unsupported BoltBeam authority ledger")
  routes = data.get("routes")
  if not isinstance(routes, list) or not routes: raise ValueError("authority ledger has no routes")
  result = {}
  for route in routes:
    route_id = route.get("route_id")
    if not isinstance(route_id, str) or route_id in result: raise ValueError("authority ledger route_id is invalid or duplicated")
    result[route_id] = route
  return result

__all__ = ["DEFAULT_LEDGER", "load_promoted_routes"]
