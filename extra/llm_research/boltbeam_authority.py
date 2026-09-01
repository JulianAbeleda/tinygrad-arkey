"""Read-only bridge to the authoritative BoltBeam route ledger."""
import json
import os
import hashlib
import json
from pathlib import Path

DEFAULT_LEDGER = Path(__file__).resolve().parents[2] / "tinygrad" / "llm" / "generated" / "boltbeam-nv-sm120-kernel-authority.json"
BOLTBEAM_SOURCE_LEDGER = Path(__file__).resolve().parents[3] / "BoltBeam" / "boltbeam" / "data" / "nv_sm120_kernel_authority.json"

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

def ticket_for_authority(route_id: str, component: str, candidate_hash: str,
                         target_identity: str, provider_revision: str = "semantic-lowering-v1"):
  from extra.llm_research.boltbeam_runtime_ticket import BoltbeamKernelTicket
  route = load_promoted_routes().get(route_id)
  if route is None: raise ValueError(f"route is not present in BoltBeam authority ledger: {route_id}")
  if component not in route.get("components", ()): raise ValueError(f"component {component} is not authorized for {route_id}")
  if route.get("state") == "blocked": raise ValueError(f"route is blocked in BoltBeam authority ledger: {route_id}")
  route_hash = hashlib.sha256((route_id + "\0" + component).encode()).hexdigest()
  return BoltbeamKernelTicket(candidate_hash, route_hash, component, target_identity, provider_revision)

def tickets_for_candidate(candidate: dict, authorities: tuple[tuple[str, str], ...], target_identity: str = "nvidia_sm120"):
  from extra.llm_research.boltbeam_runtime_ticket import BoltbeamKernelTicketBundle
  if not isinstance(candidate, dict) or not isinstance(candidate.get("family"), str) or not candidate["family"]:
    raise ValueError("route-bound candidate requires a family")
  envelope = {"schema":"boltbeam.route_bound_candidate.v1","target":target_identity,"family":candidate["family"],
              "parameters":{key:value for key,value in candidate.items() if key != "family"},
              "authorities":[{"route_id":route,"component":component} for route,component in authorities]}
  encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
  candidate_hash = hashlib.sha256(encoded).hexdigest()
  return BoltbeamKernelTicketBundle(tuple(ticket_for_authority(route, component, candidate_hash, target_identity)
                                          for route, component in authorities))

def lower_authorized_candidate(candidate: dict, authorities: tuple[tuple[str, str], ...], target_identity: str = "nvidia_sm120"):
  """Return the provider-selected emitter and its inseparable authority ticket bundle."""
  tickets = tickets_for_candidate(candidate, authorities, target_identity)
  envelope = {"schema":"boltbeam.route_bound_candidate.v1","target":target_identity,"family":candidate["family"],
              "parameters":{key:value for key,value in candidate.items() if key != "family"},
              "authorities":[{"route_id":route,"component":component} for route,component in authorities]}
  from extra.llm_research.boltbeam_kernel_provider import generate_route_bound_candidate
  generated = generate_route_bound_candidate(envelope, tickets.tickets[0].candidate_hash)
  return generated.artifact, tickets

__all__ = ["BOLTBEAM_SOURCE_LEDGER", "DEFAULT_LEDGER", "load_promoted_routes", "lower_authorized_candidate",
           "ticket_for_authority", "tickets_for_candidate"]
