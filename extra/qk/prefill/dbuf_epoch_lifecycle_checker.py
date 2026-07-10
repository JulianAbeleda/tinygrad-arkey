#!/usr/bin/env python3
"""Standalone DBUF epoch lifecycle checker.

The checker is intentionally decoupled from tinygrad lowering. It validates a
small event trace:

  produce(role, epoch, slot)
  barrier()
  consume(role, epoch, slot)

and proves the properties that matter for a rotating LDS/DBUF pipeline:

  * every consumer has exactly one prior producer,
  * producer and consumer agree on role/epoch/slot/window,
  * a barrier separates producer and consumer,
  * a slot is not overwritten before the prior epoch in that slot is consumed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import pathlib
from typing import Any

PROOF_LAYERS: tuple[dict[str, str], ...] = (
  {"id": "P1", "name": "epoch_lifecycle", "status": "done",
   "proof": "producer/consumer/barrier/overwrite correctness"},
  {"id": "P2", "name": "byte_window", "status": "pending",
   "proof": "producer and consumer agree on exact LDS byte interval"},
  {"id": "P3", "name": "value_key", "status": "pending",
   "proof": "global tile loaded by producer equals tile consumed by WMMA"},
  {"id": "P4", "name": "layout", "status": "pending",
   "proof": "A/B row or transposed layout matches WMMA operand contract"},
  {"id": "P5", "name": "wait_sync", "status": "pending",
   "proof": "VMEM waits, LGKM waits, and barriers are present in the right phase"},
  {"id": "P6", "name": "lifetime_pressure", "status": "pending",
   "proof": "address and fragment live ranges are bounded by the lifecycle"},
  {"id": "P7", "name": "lowered_stream", "status": "pending",
   "proof": "generated graph or stream exports actual stores/loads/waits/WMMA into this schema"},
)

EXPORTERS: tuple[dict[str, str], ...] = (
  {"id": "E1", "name": "hybrid_primitive_exporter", "status": "done",
   "source": "hybrid-s9-s10-role-trace.json"},
  {"id": "E2", "name": "s10_lds_spec_exporter", "status": "pending",
   "source": "WMMALDSSpec / slot identity proof"},
  {"id": "E3", "name": "hand_lifecycle_exporter", "status": "pending",
   "source": "kernel_lifecycle_trace.py / wmma.py lifecycle template"},
  {"id": "E4", "name": "generated_postrange_exporter", "status": "pending",
   "source": "pre-lowering Ops.STAGE / owner metadata"},
  {"id": "E5", "name": "lowered_stream_exporter", "status": "pending",
   "source": "generated AMD ISA or UOp stream"},
)


def s10_readiness_roadmap() -> dict[str, Any]:
  return {
    "schema": "dbuf-epoch-lifecycle-s10-roadmap.v1",
    "complete_for_s10": False,
    "current_proof_coverage": "epoch/slot/barrier only",
    "proof_layers": [dict(x) for x in PROOF_LAYERS],
    "exporters": [dict(x) for x in EXPORTERS],
    "reopen_generated_dbuf_when": "P1-P7 pass on both hand/hybrid trace and generated candidate trace with equivalent coverage",
  }


@dataclass(frozen=True)
class DBUFEvent:
  op: str
  role: str = ""
  epoch: int | str | None = None
  slot: int | str | None = None
  window: str = "default"
  step: int = 0

  @classmethod
  def from_json(cls, data: dict[str, Any]) -> "DBUFEvent":
    return cls(op=str(data["op"]), role=str(data.get("role", "")), epoch=data.get("epoch"),
               slot=data.get("slot"), window=str(data.get("window", "default")), step=int(data.get("step", 0)))

  def to_json(self) -> dict[str, Any]:
    out: dict[str, Any] = {"op": self.op, "step": self.step}
    if self.role: out["role"] = self.role
    if self.epoch is not None: out["epoch"] = self.epoch
    if self.slot is not None: out["slot"] = self.slot
    if self.window != "default": out["window"] = self.window
    return out

  def key(self) -> tuple[str, Any, Any, str]:
    return (self.role, self.epoch, self.slot, self.window)

  def slot_key(self) -> tuple[str, Any, str]:
    return (self.role, self.slot, self.window)


def _event_error(i: int, event: DBUFEvent, message: str) -> dict[str, Any]:
  return {"event_index": i, "event": event.to_json(), "error": message}


def check_events(events: list[DBUFEvent]) -> dict[str, Any]:
  errors: list[dict[str, Any]] = []
  producers_by_key: dict[tuple[str, Any, Any, str], dict[str, Any]] = {}
  live_by_slot: dict[tuple[str, Any, str], tuple[str, Any, Any, str]] = {}
  consumed: set[tuple[str, Any, Any, str]] = set()
  barrier_id = 0
  producer_count = 0
  consumer_count = 0

  for i, event in enumerate(events):
    if event.op == "barrier":
      barrier_id += 1
      continue
    if event.op not in ("produce", "consume"):
      errors.append(_event_error(i, event, f"unknown op {event.op!r}"))
      continue
    if not event.role or event.epoch is None or event.slot is None:
      errors.append(_event_error(i, event, "produce/consume requires role, epoch, and slot"))
      continue

    key = event.key()
    slot_key = event.slot_key()
    if event.op == "produce":
      producer_count += 1
      if key in producers_by_key:
        errors.append(_event_error(i, event, "duplicate producer for same role/epoch/slot/window"))
      previous_live = live_by_slot.get(slot_key)
      if previous_live is not None and previous_live not in consumed:
        errors.append(_event_error(i, event, f"slot overwrite before consume: previous={previous_live!r}"))
      producers_by_key[key] = {"event_index": i, "barrier_id": barrier_id, "event": event.to_json()}
      live_by_slot[slot_key] = key
      continue

    consumer_count += 1
    producer = producers_by_key.get(key)
    if producer is None:
      errors.append(_event_error(i, event, "consumer has no prior matching producer"))
      continue
    if producer["event_index"] >= i:
      errors.append(_event_error(i, event, "matching producer does not happen before consumer"))
    if int(producer["barrier_id"]) >= barrier_id:
      errors.append(_event_error(i, event, "no barrier separates producer and consumer"))
    if key in consumed:
      errors.append(_event_error(i, event, "same producer consumed more than once"))
    consumed.add(key)
    if live_by_slot.get(slot_key) == key:
      live_by_slot.pop(slot_key, None)

  unconsumed = [key for key in producers_by_key if key not in consumed]
  for key in unconsumed:
    errors.append({"event_index": producers_by_key[key]["event_index"], "event": producers_by_key[key]["event"],
                   "error": "producer was never consumed"})

  return {
    "schema": "dbuf-epoch-lifecycle-check.v1",
    "ok": not errors,
    "producer_count": producer_count,
    "consumer_count": consumer_count,
    "barrier_count": barrier_id,
    "unconsumed_count": len(unconsumed),
    "errors": errors,
  }


def canonical_dbuf_events(*, roles: tuple[str, ...] = ("A", "B"), k_tiles: int = 4, nbuf: int = 2) -> list[DBUFEvent]:
  if k_tiles < 1: raise ValueError("k_tiles must be >= 1")
  if nbuf < 2: raise ValueError("nbuf must be >= 2")
  events: list[DBUFEvent] = []
  step = 0
  for role in roles:
    events.append(DBUFEvent("produce", role=role, epoch=0, slot=0, step=step)); step += 1
  events.append(DBUFEvent("barrier", step=step)); step += 1
  for epoch in range(k_tiles):
    slot = epoch % nbuf
    for role in roles:
      events.append(DBUFEvent("consume", role=role, epoch=epoch, slot=slot, step=step)); step += 1
    next_epoch = epoch + 1
    if next_epoch < k_tiles:
      next_slot = next_epoch % nbuf
      for role in roles:
        events.append(DBUFEvent("produce", role=role, epoch=next_epoch, slot=next_slot, step=step)); step += 1
      events.append(DBUFEvent("barrier", step=step)); step += 1
  return events


def events_from_epoch_primitive(primitive: dict[str, Any], *, roles: tuple[str, ...] = ("A", "B"),
                                k_tiles: int = 4) -> list[DBUFEvent]:
  """Export a DBUFEpochPrimitive-style contract into checker events.

  This is a metadata exporter, not a lowering pass. The primitive supplies the
  ring size; the exported trace models the prologue/body/tail ownership implied
  by the primitive.
  """
  nbuf = int(primitive.get("nbuf", 2))
  slot_expr = str(primitive.get("slot_expr", "epoch % 2")).replace(" ", "")
  if slot_expr != "epoch%2" and nbuf == 2:
    raise ValueError(f"unsupported slot_expr for DBUF exporter: {primitive.get('slot_expr')!r}")
  return canonical_dbuf_events(roles=roles, k_tiles=k_tiles, nbuf=nbuf)


def events_from_s10_role_trace(trace: dict[str, Any], *, role: str = "ffn_gate_up", k_tiles: int = 4,
                               roles: tuple[str, ...] = ("A", "B")) -> list[DBUFEvent]:
  rows = trace.get("rows", [])
  if not isinstance(rows, list): raise ValueError("S10 role trace has no rows list")
  matches = [row for row in rows if isinstance(row, dict) and row.get("role") == role]
  if len(matches) != 1: raise ValueError(f"expected exactly one role row for {role!r}, found {len(matches)}")
  primitive = matches[0].get("hand_coded_epoch_primitive")
  if not isinstance(primitive, dict): raise ValueError(f"role {role!r} has no hand_coded_epoch_primitive")
  return events_from_epoch_primitive(primitive, roles=roles, k_tiles=k_tiles)


def load_events(path: pathlib.Path) -> list[DBUFEvent]:
  data = json.loads(path.read_text())
  if isinstance(data, dict): data = data.get("events", [])
  if not isinstance(data, list): raise ValueError("expected a JSON list or an object with an events list")
  return [DBUFEvent.from_json(x) for x in data]


def main(argv: list[str] | None = None) -> dict[str, Any]:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--input", type=pathlib.Path, help="JSON list of events or object with events")
  ap.add_argument("--s10-role-trace", type=pathlib.Path, help="export events from a S10 hybrid role trace artifact")
  ap.add_argument("--role", default="ffn_gate_up", help="role to export from --s10-role-trace")
  ap.add_argument("--canonical", action="store_true", help="check the built-in canonical DBUF plan")
  ap.add_argument("--k-tiles", type=int, default=4)
  ap.add_argument("--roles", default="A,B", help="comma-separated roles for --canonical")
  ap.add_argument("--roadmap", action="store_true", help="print S10 proof/exporter roadmap")
  ap.add_argument("--json", action="store_true")
  args = ap.parse_args(argv)

  if args.roadmap:
    report = s10_readiness_roadmap()
    if args.json: print(json.dumps(report, indent=2))
    else: print("S10_READY" if report["complete_for_s10"] else "S10_INCOMPLETE")
    return report

  roles = tuple(x.strip() for x in args.roles.split(",") if x.strip())
  if args.input:
    events = load_events(args.input)
    source = {"kind": "input", "path": str(args.input)}
  elif args.s10_role_trace:
    trace = json.loads(args.s10_role_trace.read_text())
    events = events_from_s10_role_trace(trace, role=args.role, k_tiles=args.k_tiles, roles=roles)
    source = {"kind": "s10_role_trace", "path": str(args.s10_role_trace), "role": args.role}
  else:
    events = canonical_dbuf_events(roles=roles, k_tiles=args.k_tiles)
    source = {"kind": "canonical"}
  report = check_events(events)
  report["source"] = source
  report["events"] = [event.to_json() for event in events] if args.canonical or args.s10_role_trace else []
  if args.json: print(json.dumps(report, indent=2))
  else: print("PASS" if report["ok"] else "FAIL")
  return report


if __name__ == "__main__":
  main()
