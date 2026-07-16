"""Independent allocator/VRAM checkpoint evidence for whole-model runs.

Physical allocation ownership is proved by ``PhysicalMemoryLedger``.  This
observer has the smaller, orthogonal job of corroborating the sampled process
peak, device visibility, and return to the pre-run counter baseline.
"""
from __future__ import annotations

import contextlib, hashlib, json, re, subprocess, threading, time
from typing import Any, Callable, Iterable, Mapping

SCHEMA = "tinygrad.memory_adaptive_checkpoint_observer.v1"
MEMORY_FACT_SCHEMA = "tinygrad.measured_policy_memory_facts.v1"
EXACT_MEMORY_KEYS = ("resident_copies", "candidate_workspace_bytes", "batch_size", "kv_element_bytes",
                     "runtime_persistent_bytes", "peak_prefill_activation_bytes", "peak_prefill_output_bytes",
                     "peak_prefill_scratch_bytes")


def _nonnegative(value: Any) -> int | None:
  return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def validate_memory_facts(value: Any, *, candidate_id: str | None = None) -> dict[str, Any] | None:
  """Validate the self-authenticating fact bundle accepted by search/cache seams.

  This establishes integrity, not merely shape: the digest binds every value,
  its per-field provenance, and the candidate which was measured.
  """
  if not isinstance(value, Mapping) or value.get("schema") != MEMORY_FACT_SCHEMA: return None
  bound = value.get("candidate_id")
  facts, provenance = value.get("facts"), value.get("provenance")
  if not isinstance(bound, str) or not bound or (candidate_id is not None and bound != candidate_id): return None
  if not isinstance(facts, Mapping) or set(facts) != set(EXACT_MEMORY_KEYS): return None
  if any(_nonnegative(facts.get(key)) is None for key in EXACT_MEMORY_KEYS): return None
  if facts["resident_copies"] == 0 or facts["batch_size"] == 0 or facts["kv_element_bytes"] == 0: return None
  if not isinstance(provenance, Mapping) or set(provenance) != set(EXACT_MEMORY_KEYS): return None
  for key in EXACT_MEMORY_KEYS:
    row = provenance[key]
    if (not isinstance(row, Mapping) or not isinstance(row.get("source"), str) or not row["source"] or
        not isinstance(row.get("detail"), str) or not row["detail"]): return None
  payload = {"schema": MEMORY_FACT_SCHEMA, "candidate_id": bound,
             "facts": {key: facts[key] for key in EXACT_MEMORY_KEYS},
             "provenance": {key: dict(provenance[key]) for key in EXACT_MEMORY_KEYS}}
  digest = "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
  if value.get("evidence_digest") != digest: return None
  return {**payload, "evidence_digest": digest}


def make_memory_facts(candidate_id: str, facts: Mapping[str, Any], provenance: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
  payload = {"schema": MEMORY_FACT_SCHEMA, "candidate_id": candidate_id, "facts": dict(facts),
             "provenance": {str(k): dict(v) for k, v in provenance.items()}}
  canonical = {"schema": MEMORY_FACT_SCHEMA, "candidate_id": candidate_id,
               "facts": {key: payload["facts"].get(key) for key in EXACT_MEMORY_KEYS},
               "provenance": {key: payload["provenance"].get(key) for key in EXACT_MEMORY_KEYS}}
  payload["evidence_digest"] = "sha256:" + hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
  validated = validate_memory_facts(payload, candidate_id=candidate_id)
  if validated is None: raise ValueError("memory facts must be complete, nonnegative, and provenance-backed")
  return validated


def derive_memory_facts(candidate_id: str, structure: Mapping[str, Any], allocation_evidence: Mapping[str, Any]) -> dict[str, Any]:
  """Join selected-runtime structure to explicitly attributed allocator rows.

  The three scalar geometry facts cannot be inferred from byte deltas and must
  be supplied with provenance by the loaded runtime. Every byte class is taken
  solely from matching allocation rows; absent/unknown attribution fails.
  """
  if allocation_evidence.get("schema") != "tinygrad.reconciled_measured_allocation.v1" or allocation_evidence.get("complete") is not True:
    raise ValueError("allocation evidence is incomplete")
  rows = allocation_evidence.get("allocations")
  if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
    raise ValueError("allocation evidence has no explicit allocation rows")
  structural = ("resident_copies", "batch_size", "kv_element_bytes")
  facts, provenance = {}, {}
  structural_provenance = structure.get("provenance")
  if not isinstance(structural_provenance, Mapping): raise ValueError("runtime structure has no provenance")
  for key in structural:
    facts[key] = structure.get(key)
    provenance[key] = structural_provenance.get(key)
  allocation_kinds = {
    "candidate_workspace_bytes": "candidate_workspace", "runtime_persistent_bytes": "runtime_persistent",
    "peak_prefill_activation_bytes": "prefill_activation", "peak_prefill_output_bytes": "prefill_output",
    "peak_prefill_scratch_bytes": "prefill_scratch",
  }
  for key, kind in allocation_kinds.items():
    matched = [row for row in rows if row.get("kind") == kind and
               (kind != "candidate_workspace" or row.get("candidate_id") == candidate_id)]
    if not matched or any(_nonnegative(row.get("bytes")) is None or not isinstance(row.get("identity"), str) or not row["identity"] for row in matched):
      raise ValueError(f"missing explicit allocation measurement for {key}")
    # These rows describe disjoint explicitly tagged allocations in the named
    # lifetime class. Peak classes are emitted as one measured peak row.
    if key.startswith("peak_") and len(matched) != 1: raise ValueError(f"ambiguous peak allocation measurement for {key}")
    facts[key] = sum(row["bytes"] for row in matched)
    provenance[key] = {"source": SCHEMA, "detail": "explicit allocation identities: " + ",".join(row["identity"] for row in matched)}
  return make_memory_facts(candidate_id, facts, provenance)


def rocm_smi_vram_probe() -> Mapping[str, Mapping[str, int]]:
  """Return per-card total/used VRAM. Missing or ambiguous fields fail loudly."""
  proc = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True, timeout=10, check=True)
  cards: dict[str, dict[str, int]] = {}
  # Accept both common `GPU[0] ... Total Memory (B):` and `card0 ... VRAM Total Memory` forms.
  for line in proc.stdout.splitlines():
    card = re.search(r"(?:GPU\[|card)(\d+)\]?", line, re.I)
    metric = re.search(r"(?:VRAM\s+)?(Total|Used)\s+Memory\s*\(B\)\s*:\s*(\d+)", line, re.I)
    if card and metric: cards.setdefault(f"card{card.group(1)}", {})[metric.group(1).lower()+"_bytes"] = int(metric.group(2))
  complete = {card: row for card, row in cards.items() if set(row) == {"total_bytes", "used_bytes"}}
  if not complete: raise RuntimeError("rocm-smi returned no complete per-device VRAM records")
  return complete


class AllocationObserver:
  """Checkpoint and peak observer with explicit, scoped allocation attribution."""
  def __init__(self, devices: Iterable[str] | None = None, *, probe: Callable[[], Mapping[str, Mapping[str, int]]] | None = rocm_smi_vram_probe,
               counter_source: Callable[[], Mapping[str, int]] | None = None, poll_interval: float = 0.01,
               clock_ns: Callable[[], int] = time.monotonic_ns, planned_peak_bytes: int | None = None):
    if poll_interval <= 0: raise ValueError("poll_interval must be positive")
    if planned_peak_bytes is not None and _nonnegative(planned_peak_bytes) is None: raise ValueError("planned peak must be nonnegative")
    if counter_source is None:
      from tinygrad.helpers import GlobalCounters
      counter_source = lambda: dict(GlobalCounters.mem_used_per_device)
    self.devices, self.probe, self.counter_source = None if devices is None else tuple(devices), probe, counter_source
    self.poll_interval, self.clock_ns = poll_interval, clock_ns
    self.planned_peak_bytes = planned_peak_bytes
    self._samples: list[dict[str, Any]] = []
    self._blockers: list[str] = []
    self._peak: dict[str, int] = {}
    self._stop: threading.Event | None = None
    self._thread: threading.Thread | None = None

  def _counters(self) -> dict[str, int]:
    try: raw = self.counter_source()
    except Exception as exc:
      self._blockers.append(f"tinygrad counter unavailable: {type(exc).__name__}: {exc}"); return {}
    selected = raw if self.devices is None else {device: raw.get(device, 0) for device in self.devices}
    out = {str(k): v for k, v in selected.items() if _nonnegative(v) is not None}
    if len(out) != len(selected): self._blockers.append("tinygrad counter contains invalid bytes")
    return out

  def checkpoint(self, phase: str) -> dict[str, Any]:
    if not isinstance(phase, str) or not phase: raise ValueError("phase must be non-empty")
    counters = self._counters()
    device: Mapping[str, Mapping[str, int]] | None = None
    probe_error = None
    if self.probe is not None:
      try:
        raw = self.probe()
        device = {str(k): dict(v) for k, v in raw.items()}
        if not device or any(_nonnegative(row.get("total_bytes")) is None or _nonnegative(row.get("used_bytes")) is None for row in device.values()):
          raise ValueError("probe returned incomplete device bytes")
      except Exception as exc:
        probe_error = f"{type(exc).__name__}: {exc}"
        self._blockers.append(f"device probe unavailable at {phase}: {probe_error}")
    row = {"phase": phase, "timestamp_ns": self.clock_ns(), "tinygrad_bytes": counters,
           "tinygrad_total_bytes": sum(counters.values()), "device_vram": device, "probe_error": probe_error}
    self._samples.append(row)
    for key, value in counters.items(): self._peak[key] = max(self._peak.get(key, 0), value)
    return row

  def start(self) -> "AllocationObserver":
    if self._thread is not None: raise RuntimeError("observer already started")
    self.checkpoint("pre_load")
    self._stop = threading.Event()
    def poll():
      while self._stop is not None and not self._stop.wait(self.poll_interval): self.checkpoint("poll")
    self._thread = threading.Thread(target=poll, name="allocation-observer", daemon=True); self._thread.start()
    return self

  def post_load(self) -> dict[str, Any]: return self.checkpoint("post_load")

  def stop(self) -> dict[str, Any]:
    if self._thread is not None and self._stop is not None:
      self._stop.set(); self._thread.join(); self._thread = None
    self.checkpoint("post_run_cleanup")
    return self.evidence()

  @contextlib.contextmanager
  def phase(self, name: str):
    self.checkpoint(f"{name}_begin")
    try: yield self
    finally: self.checkpoint(f"{name}_end")

  def evidence(self) -> dict[str, Any]:
    start = self._samples[0]["tinygrad_total_bytes"] if self._samples else None
    peak = max((x["tinygrad_total_bytes"] for x in self._samples), default=None)
    measured_growth = None if start is None or peak is None else max(0, peak-start)
    cleanup = None
    if len(self._samples) >= 2: cleanup = self._samples[-1]["tinygrad_total_bytes"]-self._samples[0]["tinygrad_total_bytes"]
    if cleanup is not None and cleanup > 0:
      blocker = f"post-run cleanup retained {cleanup} tinygrad bytes above pre-load"
      if blocker not in self._blockers: self._blockers.append(blocker)
    if peak is not None and self.planned_peak_bytes is not None and peak > self.planned_peak_bytes:
      blocker = f"measured peak {peak} exceeds planned peak {self.planned_peak_bytes}"
      if blocker not in self._blockers: self._blockers.append(blocker)
    return {"schema": SCHEMA, "complete": not self._blockers, "blockers": list(dict.fromkeys(self._blockers)),
      "peak_bytes": peak, "planned_peak_bytes": self.planned_peak_bytes, "peak_growth_bytes": measured_growth,
      "per_device_peak_bytes": dict(self._peak),
      "checkpoints": list(self._samples), "post_run_retained_bytes": cleanup,
      "authority": "sampled tinygrad counters and device VRAM checkpoints; no allocation ownership attribution"}

  def __enter__(self): return self.start()
  def __exit__(self, *_): self.stop()


__all__ = ["AllocationObserver", "EXACT_MEMORY_KEYS", "MEMORY_FACT_SCHEMA",
           "SCHEMA", "derive_memory_facts", "make_memory_facts", "validate_memory_facts", "rocm_smi_vram_probe"]
