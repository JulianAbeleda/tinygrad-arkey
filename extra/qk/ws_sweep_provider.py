"""Typed, hardware-neutral consumer for BoltBeam working-set sweep requests.

The actual measurement implementation is deliberately external.  A caller must
inject a runner which executes every point in a fresh, spawn-created process and
returns measurement and telemetry evidence.  This module has no default GPU
adapter and never falls back to in-process execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


REQUEST_SCHEMA = "boltbeam.ws_sweep_request.v1"
SAMPLES_SCHEMA = "boltbeam.ws_sweep_samples.v1"
PROVIDER_ID = "tinygrad.external_isolated_ws_sweep.v1"
_MODES = {"bandwidth", "pointer_chase"}
_TRAFFIC = {"read", "write", "mixed"}
_TEMPERATURES = {"warm", "cold"}


class SweepRequestError(ValueError):
  """A malformed or unsafe request; no runner call has occurred."""


@dataclass(frozen=True)
class SweepPoint:
  bytes: int
  mode: str
  traffic: str
  temperatures: tuple[str, ...]
  repetitions: int
  warmups: int
  kind: str
  reuse: int | None = None


@dataclass(frozen=True)
class IsolationAttestation:
  process_isolated: bool
  start_method: str


@dataclass(frozen=True)
class PointOutcome:
  status: str
  samples: Mapping[str, Sequence[int | float]] | None = None
  system_snapshot_id: str | None = None
  compiler: Mapping[str, Any] | None = None
  clocks: Mapping[str, Any] | None = None
  temperature: Mapping[str, Any] | None = None
  health_preflight: Mapping[str, Any] | None = None
  health_postflight: Mapping[str, Any] | None = None
  unsupported: Mapping[str, Any] | None = None
  error: str | None = None


class SpawnIsolatedRunner(Protocol):
  @property
  def isolation(self) -> IsolationAttestation: ...
  def run_point(self, point: SweepPoint, *, timeout_seconds: float) -> PointOutcome: ...


def _integer(value: Any, name: str, *, minimum: int) -> int:
  if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
    raise SweepRequestError(f"{name} must be an integer >= {minimum}")
  return value


def _point(row: Any, name: str, default_warmups: int) -> SweepPoint:
  if not isinstance(row, Mapping): raise SweepRequestError(f"{name} must be an object")
  size = _integer(row.get("bytes"), f"{name}.bytes", minimum=4096)
  mode = row.get("mode", "bandwidth" if row.get("kind") in (None, "copy", "lds_resident") else "pointer_chase")
  if mode not in _MODES: raise SweepRequestError(f"{name}.mode is unsupported: {mode!r}")
  traffic = row.get("traffic", "mixed" if row.get("kind") == "lds_resident" else "read")
  if traffic not in _TRAFFIC: raise SweepRequestError(f"{name}.traffic is unsupported: {traffic!r}")
  temperatures = row.get("temperatures")
  if not isinstance(temperatures, list) or not temperatures or any(v not in _TEMPERATURES for v in temperatures):
    raise SweepRequestError(f"{name}.temperatures must be a non-empty warm/cold list")
  if len(set(temperatures)) != len(temperatures): raise SweepRequestError(f"{name}.temperatures must be unique")
  repetitions = _integer(row.get("repeats"), f"{name}.repeats", minimum=2)
  warmups = _integer(row.get("warmups", default_warmups), f"{name}.warmups", minimum=0)
  kind = row.get("kind")
  if kind not in {"copy", "pointer_chase", "lds_resident"}: raise SweepRequestError(f"{name}.kind is unsupported: {kind!r}")
  reuse = _integer(row["reuse"], f"{name}.reuse", minimum=1) if "reuse" in row else None
  return SweepPoint(size, mode, traffic, tuple(temperatures), repetitions, warmups, kind, reuse)


def _unsupported(scope: str, code: str, reason: str, *, point: SweepPoint | None = None) -> dict[str, Any]:
  row: dict[str, Any] = {"status": "unsupported", "scope": scope, "code": code, "reason": reason}
  if point is not None: row.update(bytes=point.bytes, mode=point.mode, traffic=point.traffic)
  return row


def _validate_outcome(outcome: PointOutcome, point: SweepPoint) -> dict[str, list[float]]:
  if not isinstance(outcome, PointOutcome): raise TypeError("runner must return PointOutcome")
  if outcome.status != "passed": return {}
  if not outcome.system_snapshot_id: raise SweepRequestError("runner omitted system_snapshot_id")
  for name, evidence in (("compiler", outcome.compiler), ("clocks", outcome.clocks), ("temperature", outcome.temperature),
                         ("health_preflight", outcome.health_preflight), ("health_postflight", outcome.health_postflight)):
    if not isinstance(evidence, Mapping) or not evidence: raise SweepRequestError(f"runner omitted {name} evidence")
  expected = ({"cold_gbs" if t == "cold" else "sustained_gbs" for t in point.temperatures}
              if point.mode == "bandwidth" else {"cold_latency_ns" if t == "cold" else "warm_latency_ns" for t in point.temperatures})
  samples = outcome.samples
  if not isinstance(samples, Mapping): raise SweepRequestError("passed runner outcome omitted samples")
  normalized: dict[str, list[float]] = {}
  for key in expected:
    values = samples.get(key)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != point.repetitions:
      raise SweepRequestError(f"runner {key} must contain exactly {point.repetitions} samples")
    normalized[key] = [float(v) for v in values]
    if any(v < 0 for v in normalized[key]): raise SweepRequestError(f"runner {key} samples must be non-negative")
  return normalized


def _health_failure(outcome: PointOutcome, point: SweepPoint) -> dict[str, Any] | None:
  for phase, evidence in (("preflight", outcome.health_preflight), ("postflight", outcome.health_postflight)):
    if isinstance(evidence, Mapping) and evidence.get("status") not in {"healthy", "passed", "pass", "ok"}:
      return _unsupported("point", f"health_{phase}_failed", str(evidence.get("reason") or evidence.get("status")), point=point)
  return None


def consume_ws_sweep_request(request: Mapping[str, Any], runner: SpawnIsolatedRunner | None, *,
                             timeout_seconds: float = 30.0) -> dict[str, Any]:
  """Execute a validated request exclusively through an injected spawn runner."""
  if not isinstance(request, Mapping): raise SweepRequestError("request must be an object")
  if request.get("schema") != REQUEST_SCHEMA: raise SweepRequestError(f"schema must be {REQUEST_SCHEMA}")
  target_id = request.get("target_id")
  if not isinstance(target_id, str) or not target_id: raise SweepRequestError("target_id must be non-empty")
  execution = request.get("execution")
  if not isinstance(execution, Mapping): raise SweepRequestError("execution requirements are required")
  required = ("isolated", "health_preflight", "health_postflight", "record_system_clocks_compiler")
  if any(execution.get(v) is not True for v in required):
    raise SweepRequestError("isolated execution, pre/post health, and clock/compiler evidence are mandatory")
  warmups = _integer(execution.get("warmups", 1), "execution.warmups", minimum=0)
  if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
    raise SweepRequestError("timeout_seconds must be positive")
  raw_points = request.get("points")
  if not isinstance(raw_points, list) or not raw_points: raise SweepRequestError("points must be a non-empty list")
  points = [_point(row, f"points[{i}]", warmups) for i, row in enumerate(raw_points)]
  if len({(p.bytes, p.mode, p.traffic) for p in points}) != len(points): raise SweepRequestError("points must be unique")
  lds = _point(request.get("lds_probe"), "lds_probe", warmups) if request.get("lds_probe") is not None else None
  if runner is None:
    return {"schema": SAMPLES_SCHEMA, "target_id": target_id, "provider": PROVIDER_ID, "status": "unsupported",
            "points": [], "unsupported_outcomes": [_unsupported("provider", "adapter_unregistered", "no isolated sweep runner was registered")],
            "units": {"bytes": "bytes", "bandwidth": "GB/s", "latency": "ns", "temperature": "runner-declared", "clock": "runner-declared"}}
  isolation = runner.isolation
  if not isinstance(isolation, IsolationAttestation) or not isolation.process_isolated or isolation.start_method != "spawn":
    raise SweepRequestError("runner must attest fresh-process spawn isolation")

  rows, unsupported_rows, evidence, snapshot_id = [], [], [], None
  def run(point: SweepPoint) -> dict[str, Any] | None:
    nonlocal snapshot_id
    outcome = runner.run_point(point, timeout_seconds=float(timeout_seconds))
    if outcome.status != "passed":
      typed = dict(outcome.unsupported or _unsupported("point", "timeout" if outcome.status == "timeout" else "runner_failure",
                                                      outcome.error or outcome.status, point=point))
      typed.setdefault("status", "unsupported"); typed.setdefault("scope", "point")
      unsupported_rows.append(typed); return None
    health_failure = _health_failure(outcome, point)
    if health_failure is not None:
      unsupported_rows.append(health_failure); return None
    samples = _validate_outcome(outcome, point)
    if snapshot_id is not None and outcome.system_snapshot_id != snapshot_id: raise SweepRequestError("runner system_snapshot_id changed during sweep")
    snapshot_id = outcome.system_snapshot_id
    evidence.append({"bytes": point.bytes, "mode": point.mode, "traffic": point.traffic, "compiler": dict(outcome.compiler or {}),
                     "clocks": dict(outcome.clocks or {}), "temperature": dict(outcome.temperature or {}),
                     "health_preflight": dict(outcome.health_preflight or {}), "health_postflight": dict(outcome.health_postflight or {})})
    return {"bytes": point.bytes, "kind": point.kind, "mode": point.mode, "traffic": point.traffic,
            "temperatures": list(point.temperatures), "repeats": point.repetitions, "warmups": point.warmups, **samples}
  for point in points:
    row = run(point)
    if row is not None: rows.append(row)
  lds_row = run(lds) if lds is not None else None
  return {"schema": SAMPLES_SCHEMA, "target_id": target_id, "system_snapshot_id": snapshot_id, "provider": PROVIDER_ID,
          "identity": {"target_id": target_id, "system_snapshot_id": snapshot_id, "provider": PROVIDER_ID},
          "status": "passed" if not unsupported_rows else ("partial" if rows else "unsupported"), "points": rows,
          "lds_probe": lds_row, "unsupported_outcomes": unsupported_rows, "measurement_evidence": evidence,
          "isolation": {"process_isolated": True, "start_method": "spawn"},
          "units": {"bytes": "bytes", "bandwidth": "GB/s", "latency": "ns", "temperature": "runner-declared", "clock": "runner-declared"}}


__all__ = ["IsolationAttestation", "PointOutcome", "PROVIDER_ID", "SpawnIsolatedRunner", "SweepPoint",
           "SweepRequestError", "consume_ws_sweep_request"]
