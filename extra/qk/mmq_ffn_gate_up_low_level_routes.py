"""Production-faithful low-level routes for matched ``ffn_gate_up`` C8.

This module composes, but does not run, the two exact v2 timing routes.  The
candidate delegates fixed-VA staging and dispatch to the reusable frozen
low-level session.  The direct route invokes tinygrad's production
``_run_direct_packed_baseline`` and reuses the frozen executable attestor.

There is no import-time Device access.  The child-local host-I/O census patches
only one already-instantiated allocator and restores its prior ``_copyout``
binding exactly.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from typing import Any

import numpy as np

from extra.qk.direct_packed_executable_attestor import (
  FrozenDirectPackedExecutableAttestor,
  load_frozen_direct_packed_evidence_by_queue,
)
from extra.qk.mmq_ffn_gate_up_c8_runtime import (
  OUTER_WALL_WRAPPER, OUTPUT_REALIZATION_SEMANTICS,
  FfnGateUpCandidateInputs, FfnGateUpNoReadbackOutputRealizer,
  FfnGateUpRouteCallback,
)
from extra.qk.mmq_ffn_gate_up_matched_timing_contract import (
  CANDIDATE_ROUTE, DIRECT_ROUTE, K_LAUNCHES, QUEUE_MODES,
  validate_ffn_gate_up_matched_complete_role_timing_contract,
)
from extra.qk.mmq_ffn_gate_up_outer_wall_runner import (
  RouteInvocation, build_ffn_gate_up_post_sync_execution_attestation,
)


HOST_IO_AUTHORITY = "amd_allocator_copyout_interposition_v1"
DIRECT_OBSERVATION_AUTHORITY = "frozen_direct_packed_executable_attestor_v1"
CANDIDATE_OBSERVATION_AUTHORITY = "frozen_staged_low_level_session_v1"
_HEX = frozenset("0123456789abcdef")


def _canonical(value: Any) -> bytes:
  return json.dumps(
    value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(value: Any) -> str:
  return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _content_identity(value: Any, label: str) -> str:
  if not isinstance(value, str) or not value.startswith("sha256:") or \
     len(value) != 71 or any(char not in _HEX for char in value[7:]):
    raise ValueError(f"{label} must be a sha256 content identity")
  return value


def _queue_mode(value: Any) -> str:
  if value not in QUEUE_MODES:
    raise ValueError(f"queue_mode must be one of {QUEUE_MODES!r}")
  return value


def _validate_contract(
    contract: Mapping[str, Any], authorities: Mapping[str, Any],
    ) -> dict[str, Any]:
  if not isinstance(authorities, Mapping):
    raise ValueError("contract validation authorities must be a mapping")
  return validate_ffn_gate_up_matched_complete_role_timing_contract(
    contract, **dict(authorities))


def _copyout_nbytes(destination: Any) -> int:
  size = getattr(destination, "nbytes", None)
  if not isinstance(size, int):
    try: size = len(destination)
    except TypeError as exc:
      raise TypeError("allocator copyout destination has no byte length") from exc
  if size < 0:
    raise ValueError("allocator copyout destination byte length is negative")
  return size


def _production_static_q4_builder(source: np.ndarray) -> Any:
  """Upload validated epoch-major Q4 once, before any timed invocation."""
  from tinygrad import Tensor, dtypes
  return Tensor(source, dtype=dtypes.uint32, device="AMD")


def _validate_production_resident_fp16(value: Any) -> Any:
  if tuple(getattr(value, "shape", ())) != (1, 512, 5120) or \
     str(getattr(value, "dtype", None)) != "dtypes.half" or \
     getattr(value, "device", None) != "AMD" or \
     not callable(getattr(value, "realize", None)) or \
     not callable(getattr(value, "cast", None)):
    raise TypeError(
      "production candidate requires the shared AMD Tensor resident FP16 "
      "activation shaped [1,512,5120]")
  return value


class AmdAllocatorHostIoCensus:
  """Child-local cumulative census around one allocator's ``_copyout`` seam."""

  def __init__(
      self, allocator: Any, *, provider_identity: str | None = None,
      ) -> None:
    if allocator is None or not callable(getattr(allocator, "_copyout", None)):
      raise TypeError("AMD host-I/O census requires an allocator with _copyout")
    self.allocator = allocator
    self.provider_identity = provider_identity or _identity({
      "schema": "tinygrad.mmq_q4k_q8_1.amd_host_io_provider.v1",
      "authority": HOST_IO_AUTHORITY,
      "allocator_class":
        f"{type(allocator).__module__}.{type(allocator).__qualname__}",
      "device": getattr(getattr(allocator, "dev", None), "device", "AMD"),
      "pid": os.getpid(),
    })
    if not isinstance(self.provider_identity, str) or not self.provider_identity:
      raise ValueError("host-I/O census provider identity must be non-empty")
    self.readback_count = self.copyout_count = self.copyout_bytes = 0
    self._installed = False
    self._had_instance_binding = False
    self._prior_instance_binding: Any = None
    self._original_copyout: Callable[..., Any] | None = None

  @classmethod
  def for_live_amd(
      cls, *, provider_identity: str | None = None,
      ) -> "AmdAllocatorHostIoCensus":
    """Lazily select the instantiated AMD allocator inside a queue child."""
    from tinygrad.device import Device
    return cls(
      Device["AMD"].allocator, provider_identity=provider_identity)

  def install(self) -> "AmdAllocatorHostIoCensus":
    if self._installed:
      raise RuntimeError("AMD host-I/O census is already installed")
    namespace = getattr(self.allocator, "__dict__", {})
    self._had_instance_binding = "_copyout" in namespace
    self._prior_instance_binding = namespace.get("_copyout")
    self._original_copyout = self.allocator._copyout

    def counted_copyout(destination: Any, source: Any, *args: Any, **kwargs: Any):
      assert self._original_copyout is not None
      self.readback_count += 1
      self.copyout_count += 1
      self.copyout_bytes += _copyout_nbytes(destination)
      return self._original_copyout(destination, source, *args, **kwargs)

    try:
      setattr(self.allocator, "_copyout", counted_copyout)
    except BaseException:
      self._original_copyout = None
      raise
    self._installed = True
    return self

  def close(self) -> None:
    if not self._installed:
      return
    try:
      if self._had_instance_binding:
        setattr(self.allocator, "_copyout", self._prior_instance_binding)
      else:
        delattr(self.allocator, "_copyout")
    finally:
      self._installed = False
      self._original_copyout = None

  def __enter__(self) -> "AmdAllocatorHostIoCensus":
    return self.install()

  def __exit__(self, *_exc: Any) -> None:
    self.close()

  def snapshot(self) -> dict[str, Any]:
    if not self._installed:
      raise RuntimeError("AMD host-I/O census must be installed before sampling")
    return {
      "authority": HOST_IO_AUTHORITY,
      "provider_identity": self.provider_identity,
      "readback_count": self.readback_count,
      "copyout_count": self.copyout_count,
      "copyout_bytes": self.copyout_bytes,
    }


@dataclass
class _CandidateRouteState:
  session: Any
  invocation_type: type
  attestation_type: type
  queue_mode: str
  family_identity: str
  input_identity: str
  executable_identity: str
  program_key: str
  binary_sha256: str
  pending: Any | None = None

  def invoke(self) -> RouteInvocation:
    if self.pending is not None:
      raise RuntimeError("prior candidate invocation has not been attested")
    invocation = self.session.invoke(prefix_epochs=K_LAUNCHES)
    if not isinstance(invocation, self.invocation_type):
      raise TypeError(
        "candidate low-level session returned no typed invocation or a legacy receipt")
    output = getattr(invocation, "output", None)
    trace = getattr(invocation, "candidate_phase_trace", None)
    if output is None or not isinstance(trace, Mapping):
      raise TypeError("candidate low-level invocation omitted output/phase trace")
    self.pending = invocation
    return RouteInvocation(output, trace)

  @staticmethod
  def realize_output(_output: Any) -> None:
    # The reusable low-level session dispatches and synchronizes every epoch.
    return None

  def attest_post_sync(self, output: Any, queue_mode: str) -> Mapping[str, Any]:
    pending, self.pending = self.pending, None
    if pending is None:
      raise RuntimeError("no candidate invocation awaits post-sync attestation")
    if queue_mode != self.queue_mode or output is not getattr(pending, "output", None):
      raise ValueError("candidate post-sync output or queue differs")
    observed = self.session.attest_post_sync(pending, queue_mode)
    if not isinstance(observed, self.attestation_type):
      raise TypeError("candidate low-level session returned no typed attestation")
    checks = {
      "status": getattr(observed, "status", None) == "PASS",
      "queue_mode": getattr(observed, "queue_mode", None) == queue_mode,
      "family_identity":
        getattr(observed, "family_identity", None) == self.family_identity,
      "candidate_executable_identity":
        getattr(observed, "candidate_executable_identity", None) ==
          self.executable_identity,
      "input_identity":
        getattr(observed, "input_identity", None) == self.input_identity,
      "program_key":
        getattr(observed, "program_key", None) == self.program_key,
      "binary_sha256":
        getattr(observed, "binary_sha256", None) == self.binary_sha256,
      "launch_count": getattr(observed, "launch_count", None) == K_LAUNCHES,
    }
    if not all(checks.values()):
      raise ValueError(
        "candidate low-level observation differs: "
        f"{sorted(key for key, passed in checks.items() if not passed)!r}")
    evidence_identity = getattr(observed, "observation_identity", None)
    _content_identity(
      evidence_identity, "candidate low-level attestation evidence identity")
    return build_ffn_gate_up_post_sync_execution_attestation(
      observation_authority=
        f"{CANDIDATE_OBSERVATION_AUTHORITY}:{evidence_identity}",
      queue_mode=queue_mode, route_id=CANDIDATE_ROUTE,
      executable_identity=self.executable_identity,
      input_identity=self.input_identity)


def make_ffn_gate_up_candidate_route(
    *, queue_mode: str, matched_timing_contract: Mapping[str, Any],
    contract_validation_kwargs: Mapping[str, Any], family: Any,
    frozen_bundle: Any, staged_family_manifest: Any,
    runtime_canary_by_queue: Mapping[str, Any],
    candidate_inputs: FfnGateUpCandidateInputs,
    low_level_dependencies: Any | None = None,
    low_level_authority: Any | None = None,
    session_class: Any | None = None, invocation_type: type | None = None,
    attestation_type: type | None = None,
    static_q4_builder: Callable[[np.ndarray], Any] | None = None,
    ) -> FfnGateUpRouteCallback:
  """Prepare one persistent fixed-VA candidate session and expose v2 callbacks."""
  queue_mode = _queue_mode(queue_mode)
  contract = _validate_contract(
    matched_timing_contract, contract_validation_kwargs)
  if not isinstance(candidate_inputs, FfnGateUpCandidateInputs):
    raise TypeError("candidate route requires FfnGateUpCandidateInputs")
  input_identity = _content_identity(
    candidate_inputs.input_identity, "candidate input identity")
  if input_identity != contract["common_inputs"]["identity"]:
    raise ValueError("candidate inputs differ from matched contract")
  executable_identity = _content_identity(
    contract["candidate"]["candidate_executable_identity"],
    "candidate executable identity")
  if not isinstance(runtime_canary_by_queue, Mapping) or \
     queue_mode not in runtime_canary_by_queue:
    raise ValueError("candidate route requires the exact queue runtime canary")
  canary = runtime_canary_by_queue[queue_mode]
  if not isinstance(canary, Mapping) or canary.get("status") != "PASS" or \
     canary.get("queue_mode") != queue_mode or \
     canary.get("family_identity") != getattr(family, "family_identity", None):
    raise ValueError("candidate runtime canary differs")
  production_session = session_class is None
  if production_session:
    _validate_production_resident_fp16(
      candidate_inputs.resident_fp16_activation)
    from extra.qk.mmq_frozen_staged_low_level_session import \
      FrozenStagedLowLevelAttestation, FrozenStagedLowLevelInvocation, \
      FrozenStagedLowLevelSession
    session_class = FrozenStagedLowLevelSession
    invocation_type = FrozenStagedLowLevelInvocation
    attestation_type = FrozenStagedLowLevelAttestation
  if not isinstance(invocation_type, type) or \
     not isinstance(attestation_type, type):
    raise TypeError(
      "candidate route requires exact invocation and attestation types")
  if low_level_authority is None:
    from extra.qk.mmq_frozen_staged_low_level_session import \
      FrozenStagedProgramAuthority
    low_level_authority = FrozenStagedProgramAuthority.from_binding(
      family.binding, family_identity=family.family_identity,
      candidate_executable_identity=executable_identity,
      input_identity=input_identity)
  if low_level_dependencies is None:
    from extra.qk.mmq_frozen_staged_low_level_session import \
      production_frozen_staged_low_level_dependencies
    low_level_dependencies = \
      production_frozen_staged_low_level_dependencies(low_level_authority)
  q4_epoch_major = candidate_inputs.q4_epoch_major
  if production_session or static_q4_builder is not None:
    if not isinstance(q4_epoch_major, np.ndarray) or \
       q4_epoch_major.dtype != np.uint32 or \
       not q4_epoch_major.flags.c_contiguous:
      raise TypeError(
        "candidate epoch-major Q4 source must be contiguous uint32 NumPy")
    builder = _production_static_q4_builder \
      if static_q4_builder is None else static_q4_builder
    if not callable(builder):
      raise TypeError("candidate static Q4 builder must be callable")
    q4_epoch_major = builder(q4_epoch_major)
    if q4_epoch_major is None or isinstance(q4_epoch_major, Mapping):
      raise TypeError("candidate static Q4 builder returned no resident object")
  prepare = getattr(session_class, "prepare", None)
  if not callable(prepare):
    raise TypeError("candidate low-level session class requires prepare")
  session = prepare(
    binding=family.binding, authority=low_level_authority,
    common_resident_fp16=candidate_inputs.resident_fp16_activation,
    q4_epoch_major=q4_epoch_major,
    dependencies=low_level_dependencies)
  if session is None or isinstance(session, Mapping):
    raise TypeError("candidate low-level prepare returned no typed session")
  state = _CandidateRouteState(
    session, invocation_type, attestation_type, queue_mode,
    family.family_identity, input_identity, executable_identity,
    family.binding.program_key, family.binding.binary_sha256)
  return FfnGateUpRouteCallback(
    route_id=CANDIDATE_ROUTE, queue_mode=queue_mode,
    input_identity=input_identity, executable_identity=executable_identity,
    invoke=state.invoke,
    realize_output=FfnGateUpNoReadbackOutputRealizer(
      callback=state.realize_output,
      semantics=OUTPUT_REALIZATION_SEMANTICS, readback_performed=False),
    attest_post_sync=state.attest_post_sync,
    outer_wall_wrapper=OUTER_WALL_WRAPPER, emits_timing_receipt=False).validate(
      route_id=CANDIDATE_ROUTE, queue_mode=queue_mode,
      input_identity=input_identity,
      executable_identity=executable_identity)


@dataclass
class _DirectRouteState:
  queue_mode: str
  input_identity: str
  executable_identity: str
  direct_objects: Any
  resident_packed_q4: Any
  executor: Callable[[Any, Any, Any], Any]
  attestor: Any

  def invoke(self) -> RouteInvocation:
    output = self.executor(
      self.direct_objects.linear, self.direct_objects.activation,
      self.direct_objects.route_spec)
    if output is None or isinstance(output, Mapping):
      raise TypeError(
        "production direct_packed returned no output or a legacy receipt")
    return RouteInvocation(output)

  def attest_post_sync(self, output: Any, queue_mode: str) -> Mapping[str, Any]:
    if queue_mode != self.queue_mode:
      raise ValueError("direct post-sync queue differs")
    evidence = self.attestor.attest_post_sync(output, queue_mode)
    if not isinstance(evidence, Mapping):
      raise TypeError("direct executable attestor returned no evidence")
    checks = {
      "queue_mode": evidence.get("queue_mode") == queue_mode,
      "executable_identity":
        evidence.get("executable_identity") == self.executable_identity,
      "input_identity": evidence.get("input_identity") == self.input_identity,
      "status": evidence.get("status") == "PASS",
    }
    if not all(checks.values()):
      raise ValueError(
        "direct frozen observation differs: "
        f"{sorted(key for key, passed in checks.items() if not passed)!r}")
    evidence_identity = _content_identity(
      evidence.get("evidence_identity"),
      "direct frozen attestation evidence identity")
    return build_ffn_gate_up_post_sync_execution_attestation(
      observation_authority=
        f"{DIRECT_OBSERVATION_AUTHORITY}:{evidence_identity}",
      queue_mode=queue_mode, route_id=DIRECT_ROUTE,
      executable_identity=self.executable_identity,
      input_identity=self.input_identity)


def make_ffn_gate_up_direct_route(
    *, queue_mode: str, matched_timing_contract: Mapping[str, Any],
    contract_validation_kwargs: Mapping[str, Any],
    direct_objects: Any,
    qualification_paths_by_queue: Mapping[str, Any],
    bindings_by_queue: Mapping[str, Any],
    executor: Callable[[Any, Any, Any], Any] | None = None,
    evidence_loader: Callable[..., Any] =
      load_frozen_direct_packed_evidence_by_queue,
    attestor_factory: Callable[..., Any] =
      FrozenDirectPackedExecutableAttestor,
    attestor_primitives: Any | None = None,
    ) -> FfnGateUpRouteCallback:
  """Compose production direct-packed invocation with exact frozen attestation."""
  queue_mode = _queue_mode(queue_mode)
  contract = _validate_contract(
    matched_timing_contract, contract_validation_kwargs)
  if executor is None:
    from tinygrad.llm.prefill_routes import _run_direct_packed_baseline
    executor = _run_direct_packed_baseline
  if not callable(executor) or not callable(evidence_loader) or \
     not callable(attestor_factory):
    raise TypeError("direct route executor/loader/attestor must be callable")
  for field in ("linear", "activation", "route_spec"):
    if getattr(direct_objects, field, None) is None:
      raise ValueError(f"direct route objects omitted {field}")
  packed_weight = getattr(direct_objects.linear, "prefill_packed_weight", None)
  if not callable(packed_weight):
    raise TypeError("direct route linear omits production packed Q4 accessor")
  resident_packed_q4 = packed_weight()
  if resident_packed_q4 is None or isinstance(resident_packed_q4, Mapping) or \
     packed_weight() is not resident_packed_q4:
    raise ValueError("direct route packed Q4 object is absent or unstable")
  realize_packed_q4 = getattr(resident_packed_q4, "realize", None)
  if not callable(realize_packed_q4):
    raise TypeError("direct route packed Q4 object cannot be realized")
  if realize_packed_q4() is not resident_packed_q4:
    raise ValueError("direct route packed Q4 realization changed object identity")
  expected = evidence_loader(
    qualification_paths_by_queue, bindings_by_queue)
  if not isinstance(expected, Mapping) or set(expected) != set(QUEUE_MODES):
    raise ValueError("direct frozen evidence must contain both queue modes")
  kwargs = {
    "expected_evidence_by_queue": expected,
    "bindings_by_queue": bindings_by_queue,
  }
  if attestor_primitives is not None:
    kwargs["primitives"] = attestor_primitives
  attestor = attestor_factory(**kwargs)
  if not callable(getattr(attestor, "realize_output", None)) or \
     not callable(getattr(attestor, "attest_post_sync", None)):
    raise TypeError("direct route requires FrozenDirectPackedExecutableAttestor")
  queue_evidence = expected[queue_mode]
  if not isinstance(queue_evidence, Mapping):
    raise ValueError("direct queue evidence must be a mapping")
  input_identity = _content_identity(
    queue_evidence.get("input_identity"), "direct input identity")
  executable_identity = _content_identity(
    queue_evidence.get("executable_identity"), "direct executable identity")
  expected_contract_executable = contract["direct_packed"][
    "queue_qualifications"][queue_mode]["executable_identity"]
  if input_identity != contract["common_inputs"]["identity"] or \
     executable_identity != expected_contract_executable:
    raise ValueError("direct frozen evidence differs from matched contract")
  state = _DirectRouteState(
    queue_mode, input_identity, executable_identity,
    direct_objects, resident_packed_q4, executor, attestor)
  return FfnGateUpRouteCallback(
    route_id=DIRECT_ROUTE, queue_mode=queue_mode,
    input_identity=input_identity, executable_identity=executable_identity,
    invoke=state.invoke,
    realize_output=FfnGateUpNoReadbackOutputRealizer(
      callback=attestor.realize_output,
      semantics=OUTPUT_REALIZATION_SEMANTICS, readback_performed=False),
    attest_post_sync=state.attest_post_sync,
    outer_wall_wrapper=OUTER_WALL_WRAPPER, emits_timing_receipt=False).validate(
      route_id=DIRECT_ROUTE, queue_mode=queue_mode,
      input_identity=input_identity,
      executable_identity=executable_identity)


__all__ = [
  "AmdAllocatorHostIoCensus", "CANDIDATE_OBSERVATION_AUTHORITY",
  "DIRECT_OBSERVATION_AUTHORITY", "HOST_IO_AUTHORITY",
  "make_ffn_gate_up_candidate_route", "make_ffn_gate_up_direct_route",
]
