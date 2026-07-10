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
  {"id": "P2", "name": "byte_window", "status": "done_for_s10_lds_spec",
   "proof": "producer and consumer agree on exact LDS byte interval"},
  {"id": "P3", "name": "value_key", "status": "pending",
   "proof": "global tile loaded by producer equals tile consumed by WMMA"},
  {"id": "P4", "name": "layout", "status": "done_for_s10_lds_spec_static",
   "proof": "A/B row or transposed layout matches static WMMA operand contract when exporters provide layout_key"},
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
  {"id": "E2", "name": "s10_lds_spec_exporter", "status": "done",
   "source": "WMMALDSSpec / slot identity proof"},
  {"id": "E3", "name": "hand_lifecycle_exporter", "status": "done_for_lds2_template",
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
    "current_proof_coverage": "epoch/slot/barrier plus optional LDS byte-window equality when exporters provide windows",
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
  lds_window: dict[str, Any] | None = None
  layout_key: dict[str, Any] | None = None
  kind: str = ""
  count: int | None = None
  phase: str = ""
  step: int = 0

  @classmethod
  def from_json(cls, data: dict[str, Any]) -> "DBUFEvent":
    return cls(op=str(data["op"]), role=str(data.get("role", "")), epoch=data.get("epoch"),
               slot=data.get("slot"), window=str(data.get("window", "default")),
               lds_window=data.get("lds_window"), layout_key=data.get("layout_key"),
               kind=str(data.get("kind", "")), count=data.get("count"),
               phase=str(data.get("phase", "")),
               step=int(data.get("step", 0)))

  def to_json(self) -> dict[str, Any]:
    out: dict[str, Any] = {"op": self.op, "step": self.step}
    if self.role: out["role"] = self.role
    if self.epoch is not None: out["epoch"] = self.epoch
    if self.slot is not None: out["slot"] = self.slot
    if self.window != "default": out["window"] = self.window
    if self.lds_window is not None: out["lds_window"] = dict(self.lds_window)
    if self.layout_key is not None: out["layout_key"] = dict(self.layout_key)
    if self.kind: out["kind"] = self.kind
    if self.count is not None: out["count"] = self.count
    if self.phase: out["phase"] = self.phase
    return out

  def key(self) -> tuple[str, Any, Any, str]:
    return (self.role, self.epoch, self.slot, self.window)

  def slot_key(self) -> tuple[str, Any, str]:
    return (self.role, self.slot, self.window)


def _event_error(i: int, event: DBUFEvent, message: str) -> dict[str, Any]:
  return {"event_index": i, "event": event.to_json(), "error": message}


def _window_tuple(window: dict[str, Any] | None) -> tuple[Any, Any, Any] | None:
  if window is None: return None
  return (window.get("base"), window.get("bytes"), window.get("stride"))


def _layout_tuple(layout: dict[str, Any] | None) -> tuple[Any, ...] | None:
  if layout is None: return None
  return (
    layout.get("role"), layout.get("operand"), layout.get("lds_layout"), layout.get("wmma_contract"),
    tuple(layout.get("fragment_shape", ())), layout.get("lane_map_id"), layout.get("lane_count"),
    layout.get("lane_replication"), layout.get("per_lane_elements"), layout.get("vector_bytes"),
    layout.get("lds_row_stride_bytes"),
  )


def _layout_errors(event: DBUFEvent) -> list[str]:
  layout = event.layout_key
  if layout is None: return []
  errors: list[str] = []
  role = layout.get("role")
  operand = layout.get("operand")
  lds_layout = layout.get("lds_layout")
  if role != event.role:
    errors.append(f"layout_key role does not match event role: layout={role!r} event={event.role!r}")
  expected_operand = {"A": "src0", "B": "src1"}.get(str(role))
  if expected_operand is None:
    errors.append(f"unknown layout_key role {role!r}")
  elif operand != expected_operand:
    errors.append(f"layout_key operand mismatch for role {role!r}: expected {expected_operand!r} got {operand!r}")
  expected_layout = {"A": "global_row_major_fp16_to_lds", "B": "global_row_major_bt_fp16_to_lds"}.get(str(role))
  if expected_layout is not None and lds_layout != expected_layout:
    errors.append(f"layout_key LDS layout mismatch for role {role!r}: expected {expected_layout!r} got {lds_layout!r}")
  if not layout.get("wmma_contract"):
    errors.append("layout_key requires wmma_contract")
  if not layout.get("lane_map_id"):
    errors.append("layout_key requires lane_map_id")
  return errors


def _wait_satisfies(event: DBUFEvent, kind: str) -> bool:
  return event.op == "wait" and event.count == 0 and event.kind in (kind, "full")


def check_events(events: list[DBUFEvent], *, require_p5: bool = False) -> dict[str, Any]:
  errors: list[dict[str, Any]] = []
  producers_by_key: dict[tuple[str, Any, Any, str], dict[str, Any]] = {}
  live_by_slot: dict[tuple[str, Any, str], tuple[str, Any, Any, str]] = {}
  consumed: set[tuple[str, Any, Any, str]] = set()
  barrier_id = 0
  producer_count = 0
  consumer_count = 0
  wait_count = 0
  vm_wait_ready = False
  lgkm_store_drained = False
  lgkm_frag_ready = False
  produced_since_barrier = False

  for i, event in enumerate(events):
    if event.op == "wait":
      wait_count += 1
      if event.kind not in ("vm", "lgkm", "full"):
        errors.append(_event_error(i, event, f"invalid wait kind {event.kind!r}"))
        continue
      if not isinstance(event.count, int) or event.count < 0 or event.count > 63:
        errors.append(_event_error(i, event, "wait count must be an integer in 0..63"))
        continue
      if _wait_satisfies(event, "vm"):
        vm_wait_ready = True
      if _wait_satisfies(event, "lgkm"):
        lgkm_store_drained = True
        lgkm_frag_ready = True
      continue
    if event.op == "barrier":
      if require_p5 and produced_since_barrier and not lgkm_store_drained:
        errors.append(_event_error(i, event, "P5 requires LGKM wait after LDS stores before barrier"))
      barrier_id += 1
      produced_since_barrier = False
      lgkm_store_drained = False
      lgkm_frag_ready = False
      vm_wait_ready = False
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
      if require_p5 and not vm_wait_ready:
        errors.append(_event_error(i, event, "P5 requires VM wait before LDS produce"))
      if key in producers_by_key:
        errors.append(_event_error(i, event, "duplicate producer for same role/epoch/slot/window"))
      if event.lds_window is not None and (event.lds_window.get("base") is None or event.lds_window.get("bytes") is None):
        errors.append(_event_error(i, event, "lds_window requires base and bytes"))
      for message in _layout_errors(event):
        errors.append(_event_error(i, event, message))
      previous_live = live_by_slot.get(slot_key)
      if previous_live is not None and previous_live not in consumed:
        errors.append(_event_error(i, event, f"slot overwrite before consume: previous={previous_live!r}"))
      producers_by_key[key] = {"event_index": i, "barrier_id": barrier_id, "event": event.to_json(),
                               "lds_window": event.lds_window, "layout_key": event.layout_key}
      live_by_slot[slot_key] = key
      produced_since_barrier = True
      lgkm_store_drained = False
      continue

    consumer_count += 1
    if require_p5 and not lgkm_frag_ready:
      errors.append(_event_error(i, event, "P5 requires LGKM wait before LDS consume/WMMA phase"))
    producer = producers_by_key.get(key)
    if producer is None:
      errors.append(_event_error(i, event, "consumer has no prior matching producer"))
      continue
    if producer["event_index"] >= i:
      errors.append(_event_error(i, event, "matching producer does not happen before consumer"))
    if int(producer["barrier_id"]) >= barrier_id:
      errors.append(_event_error(i, event, "no barrier separates producer and consumer"))
    producer_window = _window_tuple(producer.get("lds_window"))
    consumer_window = _window_tuple(event.lds_window)
    if producer_window is not None and consumer_window is not None and producer_window != consumer_window:
      errors.append(_event_error(i, event, f"consumer LDS window does not match producer: producer={producer_window!r} consumer={consumer_window!r}"))
    if event.lds_window is not None and (event.lds_window.get("base") is None or event.lds_window.get("bytes") is None):
      errors.append(_event_error(i, event, "lds_window requires base and bytes"))
    for message in _layout_errors(event):
      errors.append(_event_error(i, event, message))
    producer_layout = _layout_tuple(producer.get("layout_key"))
    consumer_layout = _layout_tuple(event.layout_key)
    if producer_layout is not None and consumer_layout is not None and producer_layout != consumer_layout:
      errors.append(_event_error(i, event, f"consumer layout_key does not match producer: producer={producer_layout!r} consumer={consumer_layout!r}"))
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
    "wait_count": wait_count,
    "p5_wait_sync": "checked" if require_p5 else ("events_present_not_required" if wait_count else "not_proven"),
    "unconsumed_count": len(unconsumed),
    "errors": errors,
  }


def canonical_dbuf_events_with_waits(*, roles: tuple[str, ...] = ("A", "B"), k_tiles: int = 4,
                                     nbuf: int = 2) -> list[DBUFEvent]:
  if k_tiles < 1: raise ValueError("k_tiles must be >= 1")
  if nbuf < 2: raise ValueError("nbuf must be >= 2")
  events: list[DBUFEvent] = []
  step = 0

  def append_produce_phase(epoch: int, slot: int, phase: str) -> None:
    nonlocal step
    events.append(DBUFEvent("wait", kind="vm", count=0, phase=f"{phase}_after_coop_load", step=step)); step += 1
    for role in roles:
      events.append(DBUFEvent("produce", role=role, epoch=epoch, slot=slot, phase=phase, step=step)); step += 1
    events.append(DBUFEvent("wait", kind="lgkm", count=0, phase=f"{phase}_after_coop_store", step=step)); step += 1
    events.append(DBUFEvent("barrier", phase=phase, step=step)); step += 1
    events.append(DBUFEvent("wait", kind="lgkm", count=0, phase=f"{phase}_after_frag_load", step=step)); step += 1

  append_produce_phase(0, 0, "prologue")
  for epoch in range(k_tiles):
    slot = epoch % nbuf
    for role in roles:
      events.append(DBUFEvent("consume", role=role, epoch=epoch, slot=slot, phase="body", step=step)); step += 1
    next_epoch = epoch + 1
    if next_epoch < k_tiles:
      append_produce_phase(next_epoch, next_epoch % nbuf, "body")
  return events


def _hand_lds2_window(layout: Any, role: str, slot: int) -> dict[str, Any]:
  base = slot * layout.BUFSZ
  if role == "A":
    return {"base": base, "bytes": layout.LDS_A, "stride": layout.SA}
  if role == "B":
    return {"base": base + layout.LDS_A, "bytes": layout.BUFSZ - layout.LDS_A, "stride": layout.SB}
  raise ValueError(f"unsupported hand LDS2 role {role!r}")


def events_from_hand_lds2_lifecycle(*, roles: tuple[str, ...] = ("A", "B"), k_tiles: int = 4,
                                    bm: int = 128, bn: int = 128, bk: int = 32, pad: int = 16,
                                    dbuf: int = 1) -> list[DBUFEvent]:
  """Export the hand LDS2 lifecycle template into proof-checker events.

  This intentionally exports the logical hand oracle, not final instruction
  rows. P7 owns lowered-stream proof. The function still reads the existing
  hand template/layout/wait policy so the checker contract stays tied to the
  current S9/S10 backend atom.
  """
  if not dbuf: raise ValueError("hand LDS2 DBUF lifecycle exporter requires dbuf=1")
  from extra.qk.prefill.wmma import default_lds2_lifecycle_template, default_lds2_memory_layout, default_lds2_wait_policy
  template = default_lds2_lifecycle_template(dbuf).validate(dbuf)
  layout = default_lds2_memory_layout(bm, bn, bk, pad, dbuf).validate()
  wait = default_lds2_wait_policy().validate()
  if not any(step.op == "compute" and step.slot == 0 for step in template.body):
    raise ValueError("hand LDS2 template does not expose compute(slot 0) in body")
  if not any(step.op == "compute" and step.slot == 1 for step in template.body):
    raise ValueError("hand LDS2 template does not expose compute(slot 1) in body")

  events: list[DBUFEvent] = []
  step = 0

  def append_produce_phase(epoch: int, slot: int, phase: str) -> None:
    nonlocal step
    events.append(DBUFEvent("wait", kind="vm", count=wait.vm_after_coop_load, phase=f"{phase}_wait_coop_load", step=step)); step += 1
    for role in roles:
      events.append(DBUFEvent("produce", role=role, epoch=epoch, slot=slot, window=f"{role}:slot{slot}",
                              lds_window=_hand_lds2_window(layout, role, slot), phase=phase, step=step)); step += 1
    events.append(DBUFEvent("wait", kind="lgkm", count=wait.lgkm_after_coop_store, phase=f"{phase}_wait_coop_store", step=step)); step += 1
    events.append(DBUFEvent("barrier", phase=phase, step=step)); step += 1
    events.append(DBUFEvent("wait", kind="lgkm", count=wait.lgkm_after_frag_load, phase=f"{phase}_wait_frag_load", step=step)); step += 1

  append_produce_phase(0, 0, "prologue")
  for epoch in range(k_tiles):
    slot = epoch % 2
    for role in roles:
      events.append(DBUFEvent("consume", role=role, epoch=epoch, slot=slot, window=f"{role}:slot{slot}",
                              lds_window=_hand_lds2_window(layout, role, slot), phase="body_compute", step=step)); step += 1
    next_epoch = epoch + 1
    if next_epoch < k_tiles:
      append_produce_phase(next_epoch, next_epoch % 2, "body")
  return events


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
  ap.add_argument("--hand-lds2", action="store_true", help="export events from the hand LDS2 lifecycle template")
  ap.add_argument("--role", default="ffn_gate_up", help="role to export from --s10-role-trace")
  ap.add_argument("--canonical", action="store_true", help="check the built-in canonical DBUF plan")
  ap.add_argument("--k-tiles", type=int, default=4)
  ap.add_argument("--roles", default="A,B", help="comma-separated roles for --canonical")
  ap.add_argument("--require-p5", action="store_true", help="require explicit VM/LGKM wait events for P5")
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
  elif args.hand_lds2:
    events = events_from_hand_lds2_lifecycle(roles=roles, k_tiles=args.k_tiles)
    source = {"kind": "hand_lds2_lifecycle_template"}
  else:
    events = canonical_dbuf_events(roles=roles, k_tiles=args.k_tiles)
    source = {"kind": "canonical"}
  report = check_events(events, require_p5=args.require_p5)
  report["source"] = source
  report["events"] = [event.to_json() for event in events] if args.canonical or args.s10_role_trace or args.hand_lds2 else []
  if args.json: print(json.dumps(report, indent=2))
  else: print("PASS" if report["ok"] else "FAIL")
  return report


if __name__ == "__main__":
  main()
