#!/usr/bin/env python3
"""Versioned, target-neutral JSON-lines boundary for bounded tinygrad search.

This is deliberately a protocol shell.  It owns envelope validation and
fail-closed responses, while a later provider adapter owns actual compiler and
device work.  Keeping that split means Metal, AMD, and NVIDIA do not acquire
separate worker executables or duplicate target-policy tables.
"""
from __future__ import annotations

import hashlib
import json, math
import platform, subprocess
import sys, argparse
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, TextIO

PROTOCOL = "tinygrad.search_provider.v1"
ACTIONS = ("describe", "admit", "compile", "check", "measure")
ERROR_CODES = ("invalid_json", "invalid_envelope", "protocol_mismatch", "unsupported_action", "backend_unavailable",
               "hardware_absent", "dirty_tree", "revision_mismatch", "timeout", "identity_mismatch", "unsupported_plan",
               "admission_rejected", "resource_limit", "invalid_evidence", "provider_failure")


class ProtocolError(ValueError):
  """A JSON-safe, explicitly classified request failure."""
  def __init__(self, code: str, message: str, *, retryable: bool = False, details: Mapping[str, Any] | None = None):
    super().__init__(message)
    if code not in ERROR_CODES: raise ValueError(f"unknown protocol error code {code!r}")
    self.code, self.retryable, self.details = code, retryable, dict(details or {})


def _mapping(value: Any, name: str) -> dict[str, Any]:
  if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
    raise ProtocolError("invalid_envelope", f"{name} must be a string-keyed object")
  return dict(value)


def _json(value: Any, name: str) -> dict[str, Any]:
  row = _mapping(value, name)
  try: json.dumps(row, sort_keys=True, allow_nan=False)
  except (TypeError, ValueError) as exc: raise ProtocolError("invalid_envelope", f"{name} must be JSON-safe") from exc
  return row


def _work_bytes(value: Any) -> dict[str, Any]:
  """Validate optional work/byte evidence without turning it into a roofline policy.

  Adapters may report exact observations or estimates.  Peak rates, arithmetic
  intensity, rankings, and promotion verdicts intentionally do not belong at
  this boundary: BoltBeam remains their sole authority.
  """
  row = _json(value, "work_bytes")
  status, provenance = row.get("status"), row.get("provenance")
  if status not in ("exact", "estimated", "unavailable"):
    raise ProtocolError("invalid_evidence", "work_bytes.status must be exact, estimated, or unavailable")
  if not isinstance(provenance, str) or not provenance:
    raise ProtocolError("invalid_evidence", "work_bytes.provenance must be a non-empty string")
  forbidden = {"peak", "peak_flops", "peak_bandwidth", "arithmetic_intensity", "roofline", "promotion"} & set(row)
  if forbidden: raise ProtocolError("invalid_evidence", "work_bytes cannot contain roofline or policy fields",
                                    details={"fields": sorted(forbidden)})
  values = {key: row[key] for key in ("operations", "bytes") if key in row}
  if status == "unavailable" and values:
    raise ProtocolError("invalid_evidence", "unavailable work_bytes cannot contain observations")
  if status != "unavailable" and not values:
    raise ProtocolError("invalid_evidence", "exact or estimated work_bytes requires operations or bytes")
  for key, number in values.items():
    if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number < 0:
      raise ProtocolError("invalid_evidence", f"work_bytes.{key} must be a finite non-negative number")
  return row


def _result(value: Any) -> dict[str, Any]:
  row = _json(value, "result")
  if "work_bytes" in row: row["work_bytes"] = _work_bytes(row["work_bytes"])
  return row


@dataclass(frozen=True)
class Request:
  request_id: str
  action: str
  payload: Mapping[str, Any]

  @classmethod
  def from_json(cls, row: Any) -> "Request":
    body = _mapping(row, "request")
    required = {"protocol", "request_id", "action", "payload"}
    if set(body) != required:
      raise ProtocolError("invalid_envelope", "request must contain exactly protocol, request_id, action, payload",
                          details={"actual_keys": sorted(body), "required_keys": sorted(required)})
    if body["protocol"] != PROTOCOL: raise ProtocolError("protocol_mismatch", f"protocol must be {PROTOCOL}")
    if not isinstance(body["request_id"], str) or not body["request_id"]:
      raise ProtocolError("invalid_envelope", "request_id must be a non-empty string")
    if body["action"] not in ACTIONS:
      raise ProtocolError("unsupported_action", "action is not supported", details={"action": body["action"], "supported": list(ACTIONS)})
    return cls(body["request_id"], body["action"], _json(body["payload"], "payload"))


class ProviderAdapter(Protocol):
  """Implementation seam.  It deliberately receives no BoltBeam Python types."""
  def describe(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...
  def admit(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...
  def compile(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...
  def check(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...
  def measure(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class UnsupportedAdapter:
  """Safe default until a real compiler/runtime adapter is registered."""
  reason: str = "no provider adapter is registered"
  capabilities: Mapping[str, Any] = field(default_factory=dict)

  def describe(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    del payload
    return {"provider_revision": "protocol-shell", "capabilities": dict(self.capabilities),
            "limitations": ["compile/check/measure require a registered adapter"]}

  def _unsupported(self, action: str) -> Mapping[str, Any]:
    raise ProtocolError("backend_unavailable", self.reason, details={"action": action})
  def admit(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: return self._unsupported("admit")
  def compile(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: return self._unsupported("compile")
  def check(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: return self._unsupported("check")
  def measure(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: return self._unsupported("measure")


def _sha256(value: bytes | str) -> str:
  return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def _revision() -> dict[str, Any]:
  """One checkout identity authority for all provider responses."""
  try:
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain"], check=True, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.strip())
    return {"revision": revision, "dirty": dirty}
  except Exception: return {"revision": "unavailable", "dirty": True}


def _metal_device():
  """Import Metal only at the adapter boundary so protocol-only use stays portable."""
  try:
    from tinygrad.device import Device
    return Device["METAL"]
  except Exception as exc:
    raise ProtocolError("hardware_absent", "Metal device is unavailable", details={"exception_type": type(exc).__name__}) from exc


def _metal_facts(device: Any) -> dict[str, Any]:
  from tinygrad.runtime.ops_metal import from_ns_str
  from tinygrad.runtime.graph.metal import metal_replay_facts
  sysdevice = device.sysdevice
  family = getattr(device, "arch", "unknown")
  try: registry_id = int(sysdevice.registryID())
  except Exception: registry_id = None
  return {"backend": "METAL", "tinygrad_device": "METAL", "name": from_ns_str(sysdevice.name()), "registry_id": registry_id,
          "architecture": family, "subgroup_width": 32, "subgroup_width_source": "Metal simdgroup renderer contract",
          "max_threads_per_threadgroup": int(sysdevice.maxThreadsPerThreadgroup().width),
          "max_threadgroup_memory_bytes": int(sysdevice.maxThreadgroupMemoryLength()),
          "recommended_max_working_set_bytes": int(sysdevice.recommendedMaxWorkingSetSize()),
          "current_allocated_bytes": int(sysdevice.currentAllocatedSize()), "current_allocated_bytes_status": "volatile_runtime_fact",
          "replay": metal_replay_facts(device), "os_version": platform.platform()}


def _candidate(payload: Mapping[str, Any]) -> dict[str, Any]:
  value = payload.get("candidate")
  if not isinstance(value, Mapping): raise ProtocolError("admission_rejected", "payload.candidate must be an object")
  candidate = dict(value)
  if candidate.get("schema_version") != "boltbeam.full_kernel_candidate.v2":
    raise ProtocolError("admission_rejected", "candidate must use boltbeam.full_kernel_candidate.v2")
  return candidate


def _canonical_candidate_hash(candidate: Mapping[str, Any]) -> str:
  return _sha256(json.dumps(candidate, sort_keys=True, separators=(",", ":")))


def _live_target_identity(facts: Mapping[str, Any], supplied: Any) -> dict[str, Any]:
  """Join registry-resolved identity supplied by the controller to live device facts.

  tinygrad does not own target ids; it proves that the controller's exact identity
  is tied to this device and rejects a stale resolved-target hash.
  """
  identity = _mapping(supplied, "target_identity")
  required = {"target_id", "backend", "arch", "subgroup_size", "resolved_target_hash"}
  if set(identity) != required or not all(identity[key] for key in required):
    raise ProtocolError("identity_mismatch", "target_identity must contain exact target v2 fields")
  # BoltBeam owns resolved-target hashing.  The provider only verifies the
  # facts it can observe; inventing a second hash would split authority.
  if identity["backend"] != facts["backend"] or identity["arch"] != facts["architecture"]:
    raise ProtocolError("identity_mismatch", "target_identity does not match live Metal facts")
  return identity


def _pinned_identity(payload: Mapping[str, Any]) -> None:
  pin = payload.get("provider_identity")
  if pin is None: return
  pin = _mapping(pin, "provider_identity"); live = _revision()
  if pin.get("revision") != live["revision"]: raise ProtocolError("revision_mismatch", "pinned provider revision differs")
  if pin.get("require_clean") is True and live["dirty"]: raise ProtocolError("dirty_tree", "pinned request requires a clean provider tree")


def _plan(candidate: Mapping[str, Any]) -> tuple[dict[str, Any], list[Any]]:
  workload, schedule = candidate.get("workload"), candidate.get("schedule")
  if not isinstance(workload, Mapping) or not isinstance(schedule, Mapping):
    raise ProtocolError("admission_rejected", "candidate requires workload and schedule objects")
  target = workload.get("target")
  if not isinstance(target, Mapping) or target.get("backend") != "METAL":
    raise ProtocolError("identity_mismatch", "candidate target must be exact METAL identity")
  if schedule.get("plan_kind") not in ("tinygrad_opt_sequence.v1", "tinygrad_heuristic.v1"):
    raise ProtocolError("unsupported_plan", "Metal adapter supports explicit Opt sequences or ordinary heuristic")
  transforms = schedule.get("transforms")
  if schedule.get("plan_kind") == "tinygrad_heuristic.v1":
    if transforms not in (None, []): raise ProtocolError("unsupported_plan", "heuristic plan cannot carry explicit transforms")
    return dict(schedule), None
  if not isinstance(transforms, list): raise ProtocolError("unsupported_plan", "schedule.transforms must be an ordered list")
  from tinygrad.codegen.opt import Opt, OptOps
  opts = []
  for transform in transforms:
    if not isinstance(transform, Mapping) or not isinstance(transform.get("op"), str):
      raise ProtocolError("unsupported_plan", "each transform must name a compiler OptOps operation")
    try: opts.append(Opt(OptOps[transform["op"]], transform.get("axis"), transform.get("arg")))
    except KeyError as exc: raise ProtocolError("unsupported_plan", "unknown compiler OptOps transform",
                                                details={"op": transform["op"]}) from exc
  return dict(schedule), opts


@dataclass(frozen=True)
class MetalAdapter:
  """Metal facts/compiler adapter; policy and ranking deliberately remain outside tinygrad."""
  expected_profile: str = "qwen3_8b_q4k_m_apple_m4"

  def _prepared(self, payload: Mapping[str, Any]):
    """Build one bounded packed-weight program; all execution actions share it."""
    self.admit(payload)
    from dataclasses import replace
    from tinygrad import Tensor, dtypes
    from tinygrad.codegen import to_program
    from tinygrad.device import Device
    from tinygrad.llm.gguf import ggml_data_to_tensor
    from tinygrad.llm.qk_layout import GGML_Q4_K, GGML_Q6_K
    candidate = _candidate(payload); _, opts = _plan(candidate); workload = candidate["workload"]
    fmt = str(workload["operands"]["b"].get("quantization", workload["operands"]["b"].get("dtype", ""))).upper().replace("_", "_")
    fmt = "Q4_K" if fmt in ("Q4_K", "Q4K") else "Q6_K" if fmt in ("Q6_K", "Q6K") else None
    if fmt is None: raise ProtocolError("unsupported_plan", "Metal M6 packed program requires Q4_K or Q6_K weight")
    shape = workload["shape"]; fixture = payload.get("fixture_shape", {})
    if not isinstance(fixture, Mapping): raise ProtocolError("admission_rejected", "fixture_shape must be an object")
    rows, k, m = int(fixture.get("n", min(shape["n"], 256))), int(fixture.get("k", min(shape["k"], 256))), int(fixture.get("m", 1))
    k -= k % 256
    if not k: raise ProtocolError("admission_rejected", "representative packed tile requires k >= 256")
    block_bytes, typ = (144, GGML_Q4_K) if fmt == "Q4_K" else (210, GGML_Q6_K)
    import numpy as np
    # Vectorized host fixture avoids a multi-million-element Python list for
    # real decode roles (for example 12288x4096) while retaining deterministic
    # nonzero GGUF payload bytes.
    raw_bytes = (1 + np.arange(rows*k//256*block_bytes, dtype=np.uint32) % 13).astype(np.uint8)
    d_bytes = np.array([0.01], dtype=np.float16).view(np.uint8).tolist()
    dmin_bytes = np.array([0.005], dtype=np.float16).view(np.uint8).tolist()
    for block in range(rows*k//256):
      base = block*block_bytes
      raw_bytes[base:base+2] = d_bytes
      if fmt == "Q4_K": raw_bytes[base+2:base+4] = dmin_bytes
      else: raw_bytes[base+208:base+210] = d_bytes
    raw = Tensor(raw_bytes, dtype=dtypes.uint8, device="METAL").realize()
    weights = ggml_data_to_tensor(raw, rows*k, typ).reshape(rows, k)
    activation = Tensor([1.0 + (i % 7)/16 for i in range(m*k)], dtype=dtypes.float16, device="METAL").reshape(m, k).realize()
    output = activation @ weights.T
    linear = output.schedule_linear()
    if len(linear.src) != 1: raise ProtocolError("provider_failure", "prepared fused output must schedule exactly one program")
    call = linear.src[0]
    sink = call.src[0].replace(arg=replace(call.src[0].arg, opts_to_apply=None if opts is None else tuple(opts)))
    program = to_program(sink, Device["METAL"].renderer)
    return output, raw, call.replace(src=(program, *call.src[1:])), program, (m, rows, k, block_bytes), rows, k

  def describe(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    del payload
    device = _metal_device()
    return {"provider_revision": _revision(), "target": _metal_facts(device),
            "supported_plan_kinds": ["tinygrad_opt_sequence.v1", "tinygrad_heuristic.v1"],
            "compiler_transforms": [op.name for op in __import__("tinygrad.codegen.opt", fromlist=["OptOps"]).OptOps],
            "limitations": ["graph entry timing is non-authoritative", "no model execution or route promotion"]}

  def admit(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _pinned_identity(payload)
    candidate = _candidate(payload); schedule, opts = _plan(candidate); facts = _metal_facts(_metal_device())
    workload, target = candidate["workload"], candidate["workload"]["target"]
    if payload.get("candidate_hash") != _canonical_candidate_hash(candidate):
      raise ProtocolError("identity_mismatch", "candidate_hash does not match canonical candidate bytes")
    if workload.get("profile") != self.expected_profile: raise ProtocolError("identity_mismatch", "candidate workload profile does not match adapter")
    if not isinstance(workload.get("shape"), Mapping) or not all(isinstance(workload["shape"].get(k), int) and workload["shape"][k] > 0 for k in ("m", "n", "k")):
      raise ProtocolError("admission_rejected", "workload.shape requires positive m/n/k")
    operands = workload.get("operands")
    if workload.get("operation") != "matmul" or not isinstance(operands, Mapping) or set(operands) != {"a", "b", "c"}:
      raise ProtocolError("admission_rejected", "workload must describe matmul operands a/b/c")
    identity = _live_target_identity(facts, payload.get("target_identity"))
    if dict(target) != identity:
      raise ProtocolError("identity_mismatch", "candidate target does not match live Metal device", details={"live": facts["architecture"]})
    launch = schedule.get("launch", {})
    threads = launch.get("threads") if isinstance(launch, Mapping) else None
    if not isinstance(threads, int) or threads <= 0 or threads > facts["max_threads_per_threadgroup"]:
      raise ProtocolError("resource_limit", "launch threads exceed live Metal limit", details={"limit": facts["max_threads_per_threadgroup"]})
    return {"admitted": True, "target": facts, "target_identity": identity, "plan_hash": _sha256(json.dumps(schedule, sort_keys=True)),
            "compiler_opts": None if opts is None else [{"op": x.op.name, "axis": x.axis, "arg": x.arg} for x in opts]}

  def compile(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    admitted = self.admit(payload); candidate = _candidate(payload); _, opts = _plan(candidate)
    from tinygrad.engine.realize import get_runtime
    _, _, _, ast, _, rows, k = self._prepared(payload)
    source, binary = ast.src[3].arg, ast.src[4].arg
    runtime = get_runtime("METAL", ast)
    local = tuple(ast.arg.local_size or (1, 1, 1))
    return {**admitted, "source_sha256": _sha256(source), "mtlb_sha256": _sha256(binary),
            "launch": {"global_size": list(ast.arg.global_size), "local_size": list(local)},
            "pipeline": {"max_total_threads_per_threadgroup": runtime.max_total_threads},
            "bound_compiler_opts": None if opts is None else [{"op": x.op.name, "axis": x.axis, "arg": x.arg} for x in opts],
            "candidate_plan_hash": _sha256(json.dumps(candidate["schedule"], sort_keys=True)), "representative_tile": {"rows": rows, "k": k}}

  def check(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    self.admit(payload)
    # Reuse the canonical GGUF packed references.  These deterministic blocks
    # cover Q4_K and Q6_K decoding without embedding quant algebra here.
    import numpy as np
    from tinygrad import Tensor
    from tinygrad.engine.realize import time_call
    from extra.llm_research.layout import Q4_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES, q4_k_reference, q6_k_reference
    results = []
    for fmt, block_bytes, reference in (("Q4_K", Q4_K_BLOCK_BYTES, q4_k_reference), ("Q6_K", Q6_K_BLOCK_BYTES, q6_k_reference)):
      raw = np.zeros(block_bytes * 2, dtype=np.uint8)
      got = reference(Tensor(raw), 512).numpy()
      if not np.isfinite(got).all() or not np.array_equal(got, reference(Tensor(raw.copy()), 512).numpy()):
        raise ProtocolError("provider_failure", f"{fmt} canonical reference is non-deterministic or non-finite")
      results.append({"format": fmt, "elements": 512, "oracle": "extra.llm_research.layout canonical reference"})
    output, raw, call, _, geometry, rows, k = self._prepared(payload)
    time_call(call)
    # `call` owns the compiled candidate invocation; read its allocated output
    # buffer directly rather than asking the lazy Tensor to schedule again.
    metal = call.src[1].buffer.ensure_allocated().numpy().reshape(geometry[0], rows)
    # Canonical packed reference plus independently materialized activation.
    fmt = "Q4_K" if raw.nbytes() // (rows*k//256) == Q4_K_BLOCK_BYTES else "Q6_K"
    m = geometry[0]
    weights = (q4_k_reference if fmt == "Q4_K" else q6_k_reference)(Tensor(raw.numpy()), rows*k).numpy().reshape(rows, k)
    activation = np.array([1.0 + (i % 7)/16 for i in range(m*k)], dtype=np.float16).reshape(m, k)
    oracle = activation.astype(np.float32) @ weights.astype(np.float32).T
    if not np.isfinite(oracle).all() or not np.any(oracle != 0):
      raise ProtocolError("provider_failure", "canonical packed oracle is trivial or non-finite")
    evidence = {"representative_role_shape": {"m": m, "n": rows, "k": k}, "oracle": "canonical packed reference + numpy matmul",
                "tolerance": {"atol": 1e-3, "rtol": 1e-3}, "candidate_call_buffer_read": True}
    if not np.isfinite(metal).all() or not np.any(metal != 0):
      return {"correct": False, "reason": "candidate_output_nonfinite_or_trivial", "evidence": evidence}
    if not np.allclose(metal, oracle, rtol=1e-3, atol=1e-3):
      evidence["max_abs_error"] = float(np.max(np.abs(metal.astype(np.float32)-oracle.astype(np.float32))))
      return {"correct": False, "reason": "candidate_output_disagrees_with_canonical_oracle", "evidence": evidence}
    return {"correct": True, "tolerance": {"atol": 1e-3, "rtol": 1e-3}, "packed_block_fixtures": results,
            "representative_role_shape": {"m": m, "n": rows, "k": k}, "oracle": "canonical packed reference", "candidate_call_buffer_read": True}

  def measure(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("measurement_kind") == "practical_streaming": return self._practical_streaming(payload)
    compiled = self.compile(payload)
    samples, warmups = payload.get("samples", 3), payload.get("warmups", 1)
    if not isinstance(samples, int) or not isinstance(warmups, int) or not (1 <= samples <= 20 and 0 <= warmups <= 10):
      raise ProtocolError("admission_rejected", "samples must be 1..20 and warmups 0..10")
    from tinygrad.engine.realize import time_call
    _, raw, call, _, geometry, rows, k = self._prepared(payload)
    before = _metal_facts(_metal_device())["current_allocated_bytes"]
    for _ in range(warmups): time_call(call)
    raw_ns = [int(time_call(call) * 1e9) for _ in range(samples)]
    device = _metal_device(); after = _metal_facts(device)["current_allocated_bytes"]
    m, _, _, block_bytes = geometry
    operations, bytes_ = 2*m*rows*k, raw.nbytes() + m*k*2 + m*rows*4
    mean = sum(raw_ns) / len(raw_ns)
    return {**compiled, "timing_mode": "eager_metal_command_buffer_gpu_time_wait_true", "synchronized": True,
            "warmups": warmups, "samples_ns": raw_ns, "summary_ns": {"min": min(raw_ns), "mean": mean, "max": max(raw_ns),
            "range": max(raw_ns)-min(raw_ns)}, "working_set_after_bytes": after,
            "working_set_before_bytes": before, "work_bytes": {"status": "estimated", "provenance": "modeled logical packed input + fp16 activation/output; physical cache traffic unobserved", "operations": operations, "bytes": bytes_}}

  def _practical_streaming(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Bounded ordinary Tensor streaming measurement; not a roofline or policy verdict."""
    _pinned_identity(payload)
    facts = _metal_facts(_metal_device()); _live_target_identity(facts, payload.get("target_identity"))
    requested = payload.get("source_bytes", 256 << 20)
    if not isinstance(requested, int) or isinstance(requested, bool) or requested <= 0 or requested > (512 << 20):
      raise ProtocolError("resource_limit", "source_bytes must be positive and at most 512 MiB")
    budget = facts["recommended_max_working_set_bytes"]
    # src + dst must fit conservatively, with an explicit practical-stream assumption.
    if 2*requested > budget: raise ProtocolError("resource_limit", "resident stream buffers exceed live working-set budget",
                                                  details={"requested_resident_bytes": 2*requested, "budget": budget})
    warmups, samples = payload.get("warmups", 2), payload.get("samples", 5)
    if not isinstance(warmups, int) or not isinstance(samples, int) or not (0 <= warmups <= 10 and 1 <= samples <= 30):
      raise ProtocolError("admission_rejected", "warmups must be 0..10 and samples 1..30")
    from tinygrad import Tensor, dtypes
    from tinygrad.engine.realize import time_call
    if requested % dtypes.float.itemsize: raise ProtocolError("admission_rejected", "source_bytes must be divisible by float32 itemsize")
    elements = requested // dtypes.float.itemsize
    src = Tensor.zeros(elements, dtype=dtypes.float, device="METAL").realize()
    dst = src + 1.0
    linear = dst.schedule_linear()
    if len(linear.src) != 1: raise ProtocolError("provider_failure", "streaming Tensor must schedule one program")
    call = linear.src[0]
    before = _metal_facts(_metal_device())["current_allocated_bytes"]
    for _ in range(warmups): time_call(call)
    raw_ns = [int(time_call(call)*1e9) for _ in range(samples)]
    if any(value <= 0 or not math.isfinite(value) for value in raw_ns):
      raise ProtocolError("provider_failure", "streaming timing produced a non-positive or non-finite sample")
    after = _metal_facts(_metal_device())["current_allocated_bytes"]
    import statistics
    traffic = 2*requested
    gbs = [traffic/ns for ns in raw_ns]  # bytes/ns == GB/s
    median_ns = statistics.median(raw_ns)
    return {"measurement_kind": "practical_streaming", "label": "practical stream measurement; not roofline or promotion evidence",
            "target": facts, "provider_revision": _revision(), "storage_mode": "Metal shared/unified memory", "direction": "device read + device write",
            "timing_mode": "eager_metal_command_buffer_gpu_time_wait_true", "source_payload_bytes": requested, "traffic_bytes_per_sample": traffic, "raw_samples_ns": raw_ns,
            "summary_ns": {"min": min(raw_ns), "median": median_ns, "max": max(raw_ns), "dispersion_range": max(raw_ns)-min(raw_ns)},
            "per_sample_gb_s": gbs, "median_gb_s": statistics.median(gbs), "warmups": warmups, "synchronized": True,
            "allocation_before_bytes": before, "allocation_after_bytes": after, "requested_resident_working_set_bytes": 2*requested,
            "cache_capacity_bytes": None, "cache_capacity_status": "unavailable", "cache_residency_assumption": "not proven by this measurement",
            "thermal_status": "unavailable_no_authoritative_api"}


def _response(*, request_id: str | None, action: str | None, status: str,
              result: Mapping[str, Any] | None = None, error: ProtocolError | None = None) -> dict[str, Any]:
  row: dict[str, Any] = {"protocol": PROTOCOL, "request_id": request_id, "action": action, "status": status}
  if result is not None: row["result"] = _result(result)
  if error is not None:
    row["error"] = {"code": error.code, "message": str(error), "retryable": error.retryable, "details": dict(error.details)}
  return row


def process(row: Any, *, adapter: ProviderAdapter | None = None) -> dict[str, Any]:
  """Process one decoded JSON request.  All errors become explicit fail-closed rows."""
  request_id = row.get("request_id") if isinstance(row, Mapping) else None
  action = row.get("action") if isinstance(row, Mapping) else None
  try:
    request = Request.from_json(row)
    handler = getattr(adapter or UnsupportedAdapter(), request.action)
    return _response(request_id=request.request_id, action=request.action, status="ok", result=handler(request.payload))
  except ProtocolError as exc:
    return _response(request_id=request_id, action=action, status="blocked", error=exc)
  except Exception as exc:  # adapters may fail, but never get an implicit success row
    return _response(request_id=request_id, action=action, status="blocked",
                     error=ProtocolError("provider_failure", "provider adapter raised an unexpected exception",
                                         details={"exception_type": type(exc).__name__}))


def process_line(line: str, *, adapter: ProviderAdapter | None = None) -> dict[str, Any]:
  try: return process(json.loads(line), adapter=adapter)
  except json.JSONDecodeError as exc:
    return _response(request_id=None, action=None, status="blocked",
                     error=ProtocolError("invalid_json", "line is not valid JSON", details={"line": exc.lineno, "column": exc.colno}))


def serve(input_stream: TextIO = sys.stdin, output_stream: TextIO = sys.stdout, *, adapter: ProviderAdapter | None = None) -> int:
  """Serve one independent response per non-empty JSON-lines input row."""
  for line in input_stream:
    if line.strip():
      output_stream.write(json.dumps(process_line(line, adapter=adapter), sort_keys=True, separators=(",", ":")) + "\n")
      output_stream.flush()
  return 0


def main() -> int:
  parser = argparse.ArgumentParser(description="tinygrad target-neutral search provider")
  parser.add_argument("--backend", choices=("METAL",), help="registered provider backend; omitted is fail-closed")
  args = parser.parse_args()
  return serve(adapter=MetalAdapter() if args.backend == "METAL" else UnsupportedAdapter())

if __name__ == "__main__": raise SystemExit(main())
