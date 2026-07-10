#!/usr/bin/env python3
"""Export S10 WMMALDSSpec slot identity into DBUF checker-style metadata.

This module is intentionally a metadata adapter. It does not import or call any
lowering path, and it does not prove DBUF cadence. Its job is to carry the P2
byte-window proof from WMMALDSSpec into event-like rows that lifecycle checkers
or reports can consume.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import pathlib
from typing import Any

from extra.qk.wmma_lds_spec import WMMALDSSpec, wmma_lds_slot_identity_proof


SCHEMA = "dbuf-s10-lds-spec-export.v1"
EVENT_SCHEMA = "dbuf-s10-lds-spec-event.v1"


@dataclass(frozen=True)
class S10LDSByteWindow:
  role: str
  slot: int
  base: int
  end: int
  bytes: int
  rows: int
  row_stride_bytes: int
  vector_bytes: int
  total_vectors: int

  @property
  def window_id(self) -> str:
    return f"{self.role}:slot{self.slot}:{self.base}-{self.end}"

  @classmethod
  def from_proof_window(cls, data: dict[str, Any]) -> "S10LDSByteWindow":
    role = str(data["operand"])
    slot = int(data["buffer"])
    base = int(data["base"])
    end = int(data["end"])
    byte_count = int(data.get("bytes", end - base))
    if end - base != byte_count:
      raise ValueError(f"window {role}{slot} has inconsistent byte count: base={base} end={end} bytes={byte_count}")
    return cls(role=role, slot=slot, base=base, end=end, bytes=byte_count, rows=int(data["rows"]),
               row_stride_bytes=int(data["row_stride_bytes"]), vector_bytes=int(data.get("vector_bytes", 16)),
               total_vectors=int(data["total_vectors"]))

  def to_json(self) -> dict[str, Any]:
    return {
      "role": self.role, "slot": self.slot, "window": self.window_id, "base": self.base, "end": self.end,
      "bytes": self.bytes, "rows": self.rows, "row_stride_bytes": self.row_stride_bytes,
      "vector_bytes": self.vector_bytes, "total_vectors": self.total_vectors,
    }


def _proof_windows(proof: dict[str, Any]) -> dict[tuple[str, int], S10LDSByteWindow]:
  windows = [S10LDSByteWindow.from_proof_window(w) for w in proof.get("windows", [])]
  out: dict[tuple[str, int], S10LDSByteWindow] = {}
  for window in windows:
    key = (window.role, window.slot)
    if key in out: raise ValueError(f"duplicate LDS byte window for role={window.role!r} slot={window.slot}")
    out[key] = window
  return out


def _event(op: str, *, step: int, window: S10LDSByteWindow | None = None,
           epoch: int | None = None, phase: str = "") -> dict[str, Any]:
  row: dict[str, Any] = {"schema": EVENT_SCHEMA, "op": op, "step": step}
  if phase: row["phase"] = phase
  if epoch is not None: row["epoch"] = epoch
  if window is not None:
    row.update({
      "role": window.role,
      "slot": window.slot,
      "window": window.window_id,
      "byte_window": window.to_json(),
    })
  return row


def s10_lds_spec_dbuf_events(spec: WMMALDSSpec, *, active_buffers: int = 2, k_tiles: int | None = None,
                             roles: tuple[str, ...] = ("A", "B")) -> list[dict[str, Any]]:
  """Return DBUF checker-style events with exact P2 byte-window identity.

  The output deliberately follows the checker row vocabulary (`produce`,
  `consume`, `barrier`, `role`, `epoch`, `slot`, `window`, `step`) while keeping
  the richer byte interval in `byte_window`. `k_tiles` defaults to the static
  number of K-blocks in the spec.
  """
  if not isinstance(spec, WMMALDSSpec):
    raise TypeError(f"s10_lds_spec_dbuf_events expected WMMALDSSpec, got {type(spec).__name__}")
  if active_buffers < 1: raise ValueError("active_buffers must be >= 1")
  if k_tiles is None: k_tiles = spec.k // spec.tile_k
  if k_tiles < 1: raise ValueError("k_tiles must be >= 1")
  proof = wmma_lds_slot_identity_proof(spec, active_buffers=active_buffers)
  if not proof["ok"]:
    raise ValueError("cannot export S10 LDS DBUF events from failed slot identity proof: " + "; ".join(proof["errors"]))
  windows = _proof_windows(proof)
  missing = [(role, slot) for slot in range(active_buffers) for role in roles if (role, slot) not in windows]
  if missing: raise ValueError(f"slot identity proof did not provide all requested role/slot windows: {missing!r}")

  events: list[dict[str, Any]] = []
  step = 0
  for role in roles:
    events.append(_event("produce", step=step, epoch=0, window=windows[(role, 0)], phase="prologue"))
    step += 1
  events.append(_event("barrier", step=step, phase="prologue_to_body"))
  step += 1

  for epoch in range(k_tiles):
    slot = epoch % active_buffers
    for role in roles:
      events.append(_event("consume", step=step, epoch=epoch, window=windows[(role, slot)], phase="body"))
      step += 1
    next_epoch = epoch + 1
    if next_epoch < k_tiles:
      next_slot = next_epoch % active_buffers
      for role in roles:
        events.append(_event("produce", step=step, epoch=next_epoch, window=windows[(role, next_slot)], phase="body"))
        step += 1
      events.append(_event("barrier", step=step, phase="body"))
      step += 1
  return events


def export_s10_lds_spec(spec: WMMALDSSpec, *, active_buffers: int = 2, k_tiles: int | None = None,
                        roles: tuple[str, ...] = ("A", "B")) -> dict[str, Any]:
  proof = wmma_lds_slot_identity_proof(spec, active_buffers=active_buffers)
  if not proof["ok"]:
    return {
      "schema": SCHEMA, "ok": False, "source": "extra.qk.wmma_lds_spec.WMMALDSSpec",
      "proof_schema": proof["schema"], "active_buffers": active_buffers, "errors": list(proof["errors"]),
      "events": [],
    }
  events = s10_lds_spec_dbuf_events(spec, active_buffers=active_buffers, k_tiles=k_tiles, roles=roles)
  producer_count = sum(1 for event in events if event["op"] == "produce")
  consumer_count = sum(1 for event in events if event["op"] == "consume")
  return {
    "schema": SCHEMA,
    "ok": True,
    "source": "extra.qk.wmma_lds_spec.WMMALDSSpec",
    "proof_schema": proof["schema"],
    "proof_coverage": {
      "P1_epoch_lifecycle_shape": "event-like metadata only",
      "P2_byte_window": "done",
      "P3_value_key": "not_proven",
      "P5_wait_sync": "not_proven",
      "dbuf_cadence": "not_proven" if not proof["dbuf_cadence_proven"] else "proven",
    },
    "active_buffers": active_buffers,
    "k_tiles": spec.k // spec.tile_k if k_tiles is None else k_tiles,
    "roles": list(roles),
    "lds_buffer_bytes": proof["lds_buffer_bytes"],
    "active_lds_bytes": proof["active_lds_bytes"],
    "windows": [S10LDSByteWindow.from_proof_window(w).to_json() for w in proof["windows"]],
    "event_counts": {"produce": producer_count, "consume": consumer_count,
                     "barrier": sum(1 for event in events if event["op"] == "barrier")},
    "events": events,
    "errors": [],
  }


def checker_compatible_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Strip S10-only fields, preserving the DBUF checker event vocabulary."""
  out: list[dict[str, Any]] = []
  for event in events:
    row = {key: event[key] for key in ("op", "step", "role", "epoch", "slot", "window") if key in event}
    if "byte_window" in event:
      bw = event["byte_window"]
      row["lds_window"] = {"base": bw["base"], "bytes": bw["bytes"], "stride": bw["row_stride_bytes"]}
    out.append(row)
  return out


def main(argv: list[str] | None = None) -> dict[str, Any]:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--spec", type=pathlib.Path, required=True, help="JSON WMMALDSSpec payload")
  ap.add_argument("--active-buffers", type=int, default=2)
  ap.add_argument("--k-tiles", type=int)
  ap.add_argument("--roles", default="A,B")
  ap.add_argument("--checker-compatible", action="store_true", help="print only checker-compatible event rows")
  ap.add_argument("--json", action="store_true")
  args = ap.parse_args(argv)

  spec = WMMALDSSpec.from_json(json.loads(args.spec.read_text()))
  roles = tuple(x.strip() for x in args.roles.split(",") if x.strip())
  report = export_s10_lds_spec(spec, active_buffers=args.active_buffers, k_tiles=args.k_tiles, roles=roles)
  printable: Any = checker_compatible_events(report["events"]) if args.checker_compatible else report
  if args.json or args.checker_compatible: print(json.dumps(printable, indent=2))
  else: print("PASS" if report["ok"] else "FAIL")
  return report


if __name__ == "__main__":
  main()
