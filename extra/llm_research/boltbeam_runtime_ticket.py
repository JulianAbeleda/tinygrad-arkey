from dataclasses import dataclass
import re
import os

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

@dataclass(frozen=True)
class BoltbeamKernelTicket:
  candidate_hash: str
  route_hash: str
  component: str
  target_identity: str
  provider_revision: str
  def __post_init__(self):
    for name in ("candidate_hash", "route_hash"):
      if not isinstance(getattr(self, name), str) or not _SHA256.fullmatch(getattr(self, name)):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    for name in ("component", "target_identity", "provider_revision"):
      if not isinstance(getattr(self, name), str) or not getattr(self, name):
        raise ValueError(f"{name} must be non-empty text")
  def to_dict(self):
    return {"candidate_hash": self.candidate_hash, "route_hash": self.route_hash, "component": self.component,
            "target_identity": self.target_identity, "provider_revision": self.provider_revision}

@dataclass(frozen=True)
class BoltbeamKernelTicketBundle:
  tickets: tuple[BoltbeamKernelTicket, ...]
  def __post_init__(self):
    if not self.tickets or not all(isinstance(ticket, BoltbeamKernelTicket) for ticket in self.tickets):
      raise ValueError("BoltBeam ticket bundle must contain tickets")
    if len({(ticket.route_hash, ticket.component) for ticket in self.tickets}) != len(self.tickets):
      raise ValueError("BoltBeam ticket bundle contains duplicate authority")

def require_promoted_ticket(ticket, route_id: str, program_id: str):
  if not int(os.environ.get("TINYGRAD_BOLTBEAM_ENFORCE", "1")): return
  if not isinstance(ticket, (BoltbeamKernelTicket, BoltbeamKernelTicketBundle)):
    raise ValueError(f"{route_id}/{program_id} requires a promoted BoltBeam ticket")
