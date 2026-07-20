"""Dependency-injected fixed-VA session for one frozen staged MMQ PROGRAM.

This module owns the sequencing and evidence contract, not a second launcher.
Production dependencies are expected to use tinygrad's existing frozen
binding, AMD runtime, same-device SDMA transfer, and runtime dispatch helpers.
All device-facing operations are injected so the complete state machine can
be tested without importing or opening a GPU device.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
from dataclasses import dataclass
import hashlib
import json
from typing import Any


CANDIDATE_TRACE_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.ffn_gate_up_candidate_phase_trace.v1"
PENDING_OBSERVATION_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.frozen_staged_low_level_pending.v1"
ATTESTATION_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.frozen_staged_low_level_attestation.v1"
DIAGNOSTIC_PENDING_OBSERVATION_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.frozen_staged_low_level_diagnostic_pending.v1"
DIAGNOSTIC_RECEIPT_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.frozen_staged_low_level_diagnostic_receipt.v1"
INVOCATION_FAILURE_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.frozen_staged_low_level_failure.v1"
INVOCATION_FAILURE_ATTR = "frozen_staged_low_level_failure"
RUNTIME_DISPATCH_FAILURE_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.runtime_dispatch_failure.v2"
RUNTIME_DISPATCH_FAILURE_ATTR = "mmq_runtime_dispatch_failure"
PM4_PRE_SUBMIT_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.pm4_pre_submit_snapshot.v2"
PM4_PRE_SUBMIT_CAPTURE_POINT = \
  "AMDComputeQueue._submit_after_complete_command_construction_" \
  "before_ring_copy_and_doorbell"
PM4_NO_DOORBELL_RECEIPT_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.pm4_no_doorbell_receipt.v1"
PM4_SUBMIT_POLICIES = ("execute", "snapshot_only")
PM4_NO_DOORBELL_CHECK_KEYS = frozenset({
  "private_stop_raised_at_target_submit",
  "private_stop_caught_by_owner",
  "pre_submit_snapshot_passed",
  "native_submit_not_called",
  "timeline_advanced_exactly_once",
  "prof_exec_counter_advanced_exactly_once",
  "timeline_rollback_restored",
  "prof_exec_counter_rollback_restored",
  "timeline_signal_unchanged",
  "error_state_unchanged",
  "submit_hook_restored",
  "fill_kernargs_hook_restored",
})
PM4_PRE_SUBMIT_CHECK_KEYS = frozenset({
  "queue_device_matches_submit_device",
  "runtime_device_matches_submit_device",
  "args_state_program_matches_runtime",
  "exact_five_argument_buffers",
  "exact_five_kernarg_qwords",
  "five_qwords_match_constructed_buffers",
  "pm4_command_words_concrete",
  "pm4_command_stream_nonempty",
  "pm4_packet_stream_decoded",
  "pm4_kernarg_user_data_found_once",
  "pm4_kernarg_uses_user_data_0",
  "pm4_kernarg_user_data_matches_kernarg_va",
  "pm4_dispatch_direct_found_once",
  "pm4_dispatch_groups_match_requested",
  "pm4_workgroup_size_found_once",
  "pm4_workgroup_size_matches_requested",
  "pm4_program_entry_found_once",
  "pm4_program_entry_matches_runtime",
})
QUEUE_MODES = ("PM4", "AQL")
ABI_NAMES = (
  "output", "q4", "q8_values", "q8_scales", "q8_original_sums")
EXACT_ROLE = "ffn_gate_up"
EXACT_FULL_SHAPE = (512, 17408, 5120)
EXACT_PROGRAM_SHAPE = (512, 17408, 256)
EXACT_GLOBAL_SIZE = (136, 4, 1)
EXACT_LOCAL_SIZE = (256, 1, 1)
DIAGNOSTIC_GLOBAL_SIZE_ALLOWLIST = frozenset({
  (1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 4, 1), (8, 4, 1),
  (32, 4, 1), (40, 4, 1), (41, 4, 1), (136, 1, 1),
  # Boundary-search / deconfound rows (count vs column-index); kept in sync
  # with FFN_REDUCED_GLOBAL_SIZE_ALLOWLIST and the reduced-grid runner.
  (16, 1, 1), (32, 1, 1), (64, 1, 1), (16, 4, 1),
})
EXACT_FUNCTION = "mmq_llama_five_buffer_full_grid_accumulate"
EXACT_COMPILE_TARGET = "AMD:ISA:gfx1100"
EXACT_ABI_ELEMENTS = (8912896, 626688, 131072, 4096, 4096)
EXACT_ABI_NBYTES = (35651584, 2506752, 131072, 16384, 16384)
EXACT_ABI_DTYPES = (
  "dtypes.float.ptr(8912896)", "dtypes.uint.ptr(626688)",
  "dtypes.char.ptr(131072)", "dtypes.float.ptr(4096)",
  "dtypes.float.ptr(4096)")
EXACT_Q8_SOURCE_DTYPES = ("dtypes.char", "dtypes.float", "dtypes.float")
EXACT_Q8_SOURCE_SHAPES = (
  (40, 512, 128), (40, 512, 4), (40, 512, 4))
_EPOCH_PHASES = (
  "gather", "q4_transfer", "q8_values_transfer",
  "q8_scales_transfer", "q8_sums_transfer", "staging_sync",
  "dispatch", "dispatch_sync",
)
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


def _digest(value: Any, label: str) -> str:
  if not isinstance(value, str) or len(value) != 64 or \
     any(char not in _HEX for char in value):
    raise ValueError(f"{label} must be a lowercase SHA-256 digest")
  return value


def _nonempty(value: Any, label: str) -> str:
  if not isinstance(value, str) or not value:
    raise ValueError(f"{label} must be a non-empty string")
  return value


def _positive(value: Any, label: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
    raise ValueError(f"{label} must be a positive integer")
  return value


def _tuple3(value: Any, label: str) -> tuple[int, int, int]:
  if not isinstance(value, (tuple, list)) or len(value) != 3 or \
     any(not isinstance(item, int) or isinstance(item, bool) or item <= 0
         for item in value):
    raise ValueError(f"{label} must contain three positive integers")
  return tuple(value)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
  if not isinstance(value, Mapping):
    raise ValueError(f"{label} must be a mapping")
  return value


@dataclass(frozen=True)
class FrozenStagedAbiSlot:
  slot: int
  name: str
  dtype: str
  elements: int
  nbytes: int
  direction: str

  def validate(self, expected_slot: int) -> "FrozenStagedAbiSlot":
    if self.slot != expected_slot or self.name != ABI_NAMES[expected_slot]:
      raise ValueError("frozen staged ABI slot order/name differs")
    if self.dtype != EXACT_ABI_DTYPES[expected_slot]:
      raise ValueError("frozen staged ABI dtype differs")
    elements = _positive(self.elements, f"ABI slot {expected_slot} elements")
    nbytes = _positive(self.nbytes, f"ABI slot {expected_slot} nbytes")
    if elements != EXACT_ABI_ELEMENTS[expected_slot] or \
       nbytes != EXACT_ABI_NBYTES[expected_slot]:
      raise ValueError("frozen staged ABI extent differs")
    expected_direction = "inout" if expected_slot == 0 else "in"
    if self.direction != expected_direction:
      raise ValueError("frozen staged ABI effects differ")
    return self


@dataclass(frozen=True)
class FrozenStagedProgramAuthority:
  family_identity: str
  candidate_executable_identity: str
  input_identity: str
  program_key: str
  binary_sha256: str
  source_sha256: str
  serialized_program_sha256: str
  function_name: str
  compile_target: str
  role: str
  full_shape: tuple[int, int, int]
  program_shape: tuple[int, int, int]
  dispatch_count: int
  global_size: tuple[int, int, int]
  local_size: tuple[int, int, int]
  globals: tuple[int, ...]
  abi: tuple[FrozenStagedAbiSlot, ...]
  requires_recompile: bool

  @classmethod
  def from_binding(
      cls, binding: Any, *, family_identity: str,
      candidate_executable_identity: str, input_identity: str,
      ) -> "FrozenStagedProgramAuthority":
    """Derive immutable PROGRAM facts from an existing frozen exact binding."""
    role = getattr(binding, "role_spec", None)
    artifact = getattr(binding, "artifact", None)
    manifest = getattr(artifact, "manifest", None)
    program = getattr(artifact, "program", None)
    if role is None or not isinstance(manifest, Mapping) or program is None:
      raise TypeError(
        "authority construction requires a FrozenExactRoleBinding-compatible object")
    if not hasattr(program, "arg") or \
       tuple(program.arg.vals({})) != ():
      raise ValueError(
        "frozen staged PROGRAM must have a pointer-only ABI with no scalar values")
    arg = program.arg
    actual_program_checks = {
      "program_key":
        callable(getattr(getattr(program, "key", None), "hex", None)) and
        program.key.hex() == getattr(binding, "program_key", None),
      "function": getattr(arg, "function_name", None) == EXACT_FUNCTION,
      "global_size": tuple(getattr(arg, "global_size", ())) == EXACT_GLOBAL_SIZE,
      "local_size": tuple(getattr(arg, "local_size", ()) or ()) ==
        EXACT_LOCAL_SIZE,
      "globals": tuple(getattr(arg, "globals", ())) == tuple(range(5)),
      "outs": tuple(getattr(arg, "outs", ())) == (0,),
      "ins": tuple(getattr(arg, "ins", ())) == tuple(range(5)),
    }
    if not all(actual_program_checks.values()):
      raise ValueError(
        "actual frozen PROGRAM facts differ: "
        f"{sorted(key for key, passed in actual_program_checks.items()
                  if not passed)!r}")
    source_payloads = [
      node.arg for node in getattr(program, "src", ())
      if getattr(getattr(node, "op", None), "name", None) == "SOURCE"]
    binary_payloads = [
      node.arg for node in getattr(program, "src", ())
      if getattr(getattr(node, "op", None), "name", None) == "BINARY"]
    device_payloads = [
      node.arg for node in getattr(program, "src", ())
      if getattr(getattr(node, "op", None), "name", None) == "DEVICE"]
    artifact_source = getattr(artifact, "source", None)
    artifact_binary = getattr(artifact, "binary", None)
    if not isinstance(artifact_source, str) or \
       not isinstance(artifact_binary, bytes) or \
       source_payloads != [artifact_source] or \
       binary_payloads != [artifact_binary] or \
       hashlib.sha256(artifact_source.encode()).hexdigest() != \
         getattr(binding, "source_sha256", None) or \
       hashlib.sha256(artifact_binary).hexdigest() != \
         getattr(binding, "binary_sha256", None) or \
       device_payloads != ["AMD"]:
      raise ValueError("actual frozen PROGRAM source/binary payload differs")
    program_manifest = _mapping(
      manifest.get("program"), "frozen artifact program manifest")
    artifacts = _mapping(
      manifest.get("artifacts"), "frozen artifact artifacts manifest")
    consumer = _mapping(
      manifest.get("consumer"), "frozen artifact consumer manifest")
    if consumer.get("requires_recompile") is not False:
      raise ValueError("frozen artifact requires recompilation")
    elements = tuple(getattr(getattr(role, "program", None), "abi_elements", ()))
    raw_abi = program_manifest.get("abi")
    if not isinstance(raw_abi, list) or len(raw_abi) != 5 or \
       len(elements) != 5:
      raise ValueError("frozen binding lacks the exact five-buffer ABI")
    slots = []
    for slot, (row, count) in enumerate(zip(raw_abi, elements)):
      row = _mapping(row, f"frozen ABI slot {slot}")
      if set(row) != {"slot", "name", "dtype", "elements"} or \
         row.get("slot") != slot or row.get("name") != ABI_NAMES[slot] or \
         row.get("dtype") != EXACT_ABI_DTYPES[slot] or \
         row.get("elements") != count:
        raise ValueError(f"frozen manifest ABI slot {slot} differs")
      dtype = row["dtype"]
      # The manifest pointer spelling is ``dtypes.<name>.ptr(<elements>)``.
      # The staged ABI has fixed storage widths: f32/u32/i8/f32/f32.
      itemsize = (4, 4, 1, 4, 4)[slot]
      slots.append(FrozenStagedAbiSlot(
        slot=slot, name=row.get("name"), dtype=dtype,
        elements=int(count), nbytes=int(count) * itemsize,
        direction="inout" if slot == 0 else "in"))
    authority = cls(
      family_identity=family_identity,
      candidate_executable_identity=candidate_executable_identity,
      input_identity=input_identity,
      program_key=getattr(binding, "program_key", None),
      binary_sha256=getattr(binding, "binary_sha256", None),
      source_sha256=getattr(binding, "source_sha256", None),
      serialized_program_sha256=artifacts.get("serialized_program_sha256"),
      function_name=program_manifest.get("function"),
      compile_target=program_manifest.get("compile_target"),
      role=getattr(role, "role", None),
      full_shape=tuple(getattr(role, "shape", ())),
      program_shape=tuple(getattr(getattr(role, "program", None), "shape", ())),
      dispatch_count=getattr(role, "epochs", None),
      global_size=tuple(program_manifest.get("global_size", ())),
      local_size=tuple(program_manifest.get("local_size", ())),
      globals=tuple(program_manifest.get("globals", ())),
      abi=tuple(slots),
      requires_recompile=consumer["requires_recompile"])
    return authority.validate()

  def validate(self) -> "FrozenStagedProgramAuthority":
    _content_identity(self.family_identity, "family identity")
    _content_identity(
      self.candidate_executable_identity, "candidate executable identity")
    _content_identity(self.input_identity, "input identity")
    _digest(self.program_key, "program key")
    _digest(self.binary_sha256, "binary SHA-256")
    _digest(self.source_sha256, "source SHA-256")
    _digest(
      self.serialized_program_sha256, "serialized PROGRAM SHA-256")
    if self.function_name != EXACT_FUNCTION:
      raise ValueError("frozen staged PROGRAM function differs")
    if self.compile_target != EXACT_COMPILE_TARGET:
      raise ValueError("frozen staged compile target differs")
    if self.role != EXACT_ROLE:
      raise ValueError("frozen staged role differs")
    full = _tuple3(self.full_shape, "full role shape")
    compact = _tuple3(self.program_shape, "compact PROGRAM shape")
    if full != EXACT_FULL_SHAPE or compact != EXACT_PROGRAM_SHAPE:
      raise ValueError("frozen staged exact role/program shape differs")
    if compact[:2] != full[:2] or compact[2] != 256 or \
       full[2] != compact[2] * self.dispatch_count:
      raise ValueError("frozen PROGRAM/full-role K recurrence differs")
    if _positive(self.dispatch_count, "dispatch count") != 20:
      raise ValueError("frozen staged production authority requires 20 dispatches")
    if _tuple3(self.global_size, "global size") != EXACT_GLOBAL_SIZE or \
       _tuple3(self.local_size, "local size") != EXACT_LOCAL_SIZE:
      raise ValueError("frozen staged launch geometry differs")
    if self.globals != tuple(range(5)):
      raise ValueError("frozen staged PROGRAM globals differ from five-buffer ABI")
    if len(self.abi) != 5:
      raise ValueError("frozen staged authority must contain five ABI slots")
    for slot, row in enumerate(self.abi): row.validate(slot)
    if self.requires_recompile is not False:
      raise ValueError("frozen staged authority must not require recompilation")
    return self

  def binding_observation(self) -> dict[str, Any]:
    """Canonical facts an injected binding observer must reproduce."""
    return {
      "program_key": self.program_key,
      "binary_sha256": self.binary_sha256,
      "source_sha256": self.source_sha256,
      "serialized_program_sha256": self.serialized_program_sha256,
      "function_name": self.function_name,
      "compile_target": self.compile_target,
      "global_size": list(self.global_size),
      "local_size": list(self.local_size),
      "globals": list(self.globals),
      "requires_recompile": self.requires_recompile,
    }


@dataclass(frozen=True)
class FrozenStagedLowLevelDependencies:
  observe_binding: Callable[[Any], Mapping[str, Any]]
  create_runtime: Callable[[Any], Any]
  observe_runtime: Callable[[Any], Mapping[str, Any]]
  allocate: Callable[[FrozenStagedAbiSlot], Any]
  realize_many: Callable[[tuple[Any, ...]], None]
  observe_buffer: Callable[[Any], Mapping[str, Any]]
  zero_output: Callable[[Any], None]
  produce_q8: Callable[[Any, FrozenStagedProgramAuthority], tuple[Any, Any, Any]]
  epoch_view: Callable[[Any, FrozenStagedAbiSlot, int], Any]
  transfer: Callable[[Any, Any, int], None]
  synchronize: Callable[[], None]
  dispatch: Callable[
    [Any, tuple[Any, ...], FrozenStagedProgramAuthority, int],
    Mapping[str, Any]]
  clock_ns: Callable[[], int]
  diagnostic_global_size: tuple[int, int, int] | None = None
  diagnostic_dispatch_receipt_sink: list[Mapping[str, Any]] | None = None

  def validate(self) -> "FrozenStagedLowLevelDependencies":
    for name in (
        "observe_binding", "create_runtime", "observe_runtime", "allocate",
        "realize_many", "observe_buffer", "zero_output", "produce_q8",
        "epoch_view", "transfer", "synchronize", "dispatch", "clock_ns"):
      if not callable(getattr(self, name)):
        raise TypeError(f"low-level dependency {name} must be callable")
    if (self.diagnostic_global_size is None) != \
       (self.diagnostic_dispatch_receipt_sink is None):
      raise ValueError(
        "diagnostic grid and diagnostic receipt sink must be provided together")
    if self.diagnostic_global_size is not None:
      grid = _tuple3(
        self.diagnostic_global_size, "diagnostic global size")
      if grid not in DIAGNOSTIC_GLOBAL_SIZE_ALLOWLIST:
        raise ValueError("diagnostic global size is not allowlisted")
      if not isinstance(self.diagnostic_dispatch_receipt_sink, list):
        raise TypeError("diagnostic dispatch receipt sink must be a list")
    return self


@dataclass(frozen=True)
class FrozenStagedLowLevelInvocation:
  output: Any
  candidate_phase_trace: Mapping[str, Any]
  pending_observation: Mapping[str, Any]


@dataclass(frozen=True)
class FrozenStagedLowLevelAttestation:
  schema: str
  status: str
  queue_mode: str
  family_identity: str
  candidate_executable_identity: str
  input_identity: str
  program_key: str
  binary_sha256: str
  runtime_class: str
  runtime_name: str
  runtime_device: str
  runtime_object_identity: int
  runtime_device_identity_exact: bool
  runtime_cache_binding_exact: bool
  library_va: int
  library_nbytes: int
  entry_va: int
  fixed_five_vas: tuple[int, ...]
  launch_count: int
  observation_identity: str


@dataclass(frozen=True)
class FrozenStagedLowLevelDiagnosticReceipt:
  schema: str
  status: str
  queue_mode: str
  promotion_evidence_eligible: bool
  target_dispatch_submitted: bool
  native_submit_entered_count: int
  native_submit_returned_count: int
  target_submit_entered: bool
  target_submit_returned: bool
  post_dispatch_sync_completed: bool
  family_identity: str
  candidate_executable_identity: str
  input_identity: str
  program_key: str
  binary_sha256: str
  runtime_class: str
  runtime_name: str
  runtime_device: str
  runtime_object_identity: int
  runtime_device_identity_exact: bool
  runtime_cache_binding_exact: bool
  library_va: int
  library_nbytes: int
  entry_va: int
  fixed_five_vas: tuple[int, ...]
  frozen_global_size: tuple[int, int, int]
  effective_global_size: tuple[int, int, int]
  local_size: tuple[int, int, int]
  pre_submit: Mapping[str, Any]
  launch_count: int
  observation_identity: str


@dataclass
class _PendingState:
  invocation: FrozenStagedLowLevelInvocation
  q8_sources: tuple[Any, Any, Any]
  runtime_observation: Mapping[str, Any]
  launch_observations: tuple[Mapping[str, Any], ...]
  source_observations: tuple[Mapping[str, int], ...]
  diagnostic: bool
  effective_global_size: tuple[int, int, int]


def _buffer_observation(
    dependencies: FrozenStagedLowLevelDependencies, value: Any, *,
    label: str, expected_nbytes: int, expected_dtype: str | None = None,
    expected_shape: tuple[int, ...] | None = None,
    ) -> dict[str, int]:
  row = _mapping(dependencies.observe_buffer(value), label)
  if row.get("device") != "AMD":
    raise ValueError(f"{label} is not resident on AMD")
  va = _positive(row.get("va"), f"{label} VA")
  nbytes = _positive(row.get("nbytes"), f"{label} nbytes")
  if nbytes != expected_nbytes:
    raise ValueError(f"{label} byte extent differs")
  if va + nbytes > 1 << 64:
    raise ValueError(f"{label} address range exceeds uint64")
  if expected_dtype is not None and row.get("dtype") != expected_dtype:
    raise TypeError(f"{label} dtype differs")
  if expected_shape is not None and \
     tuple(row.get("shape", ())) != expected_shape:
    raise ValueError(f"{label} shape differs")
  return {"va": va, "nbytes": nbytes}


def _require_disjoint_buffer_ranges(
    rows: list[Mapping[str, int]], label: str,
    ) -> None:
  ranges = [
    (row["va"], row["va"] + row["nbytes"]) for row in rows]
  if any(
      not (end <= other_start or start >= other_end)
      for index, (start, end) in enumerate(ranges)
      for other_start, other_end in ranges[index+1:]):
    raise ValueError(f"{label} buffer extents overlap")


def _failed_dispatch_observation(
    exc: BaseException, *, epoch: int,
    authority: FrozenStagedProgramAuthority,
    effective_global_size: tuple[int, int, int],
    fixed_five_vas: tuple[int, ...],
    runtime_observation: Mapping[str, Any],
    ) -> dict[str, Any] | None:
  raw = getattr(exc, RUNTIME_DISPATCH_FAILURE_ATTR, None)
  if not isinstance(raw, Mapping):
    return None
  row = dict(raw)
  launch_value = row.get("launch")
  launch = dict(launch_value) if isinstance(launch_value, Mapping) else {}
  arguments = launch.get("arguments")
  argument_vas = [
    item.get("va") for item in arguments
  ] if isinstance(arguments, list) and all(
    isinstance(item, Mapping) for item in arguments) else None
  kernarg_value = launch.get("kernarg")
  kernarg = dict(kernarg_value) if isinstance(kernarg_value, Mapping) else {}
  pre_submit_value = row.get("pre_submit")
  pre_submit = dict(pre_submit_value) \
    if isinstance(pre_submit_value, Mapping) else {}
  pre_submit_arguments = pre_submit.get("argument_buffers")
  pre_submit_argument_vas = [
    item.get("va") for item in pre_submit_arguments
  ] if isinstance(pre_submit_arguments, list) and all(
    isinstance(item, Mapping) for item in pre_submit_arguments) else None
  pre_submit_checks = pre_submit.get("checks")
  pre_submit_checks_exact = isinstance(pre_submit_checks, Mapping) and \
    set(pre_submit_checks) == PM4_PRE_SUBMIT_CHECK_KEYS and all(
      value is True for value in pre_submit_checks.values())
  pm4_user_data_value = pre_submit.get("pm4_kernarg_user_data")
  pm4_user_data = dict(pm4_user_data_value) \
    if isinstance(pm4_user_data_value, Mapping) else {}
  pm4_dispatch = pre_submit.get("pm4_dispatch_direct")
  pm4_workgroup = pre_submit.get("pm4_workgroup_size")
  pm4_program = pre_submit.get("pm4_program_entry")
  submit_value = row.get("submit_evidence")
  submit_evidence = dict(submit_value) \
    if isinstance(submit_value, Mapping) else {}
  submit_keys = {
    "native_submit_entered_count", "native_submit_returned_count",
    "target_submit_entered", "target_submit_returned",
    "target_dispatch_submitted",
  }
  submit_counts_exact = all(
    isinstance(submit_evidence.get(key), int) and
    not isinstance(submit_evidence.get(key), bool) and
    submit_evidence[key] in (0, 1)
    for key in ("native_submit_entered_count",
                "native_submit_returned_count"))
  submit_state_exact = (
    set(submit_evidence) == submit_keys and submit_counts_exact and
    type(submit_evidence.get("target_submit_entered")) is bool and
    type(submit_evidence.get("target_submit_returned")) is bool and
    (submit_evidence.get("target_dispatch_submitted") is None or
     type(submit_evidence.get("target_dispatch_submitted")) is bool) and
    submit_evidence.get("target_submit_entered") is
      (submit_evidence.get("native_submit_entered_count") == 1) and
    submit_evidence.get("target_submit_returned") is
      (submit_evidence.get("native_submit_returned_count") == 1) and
    (not submit_evidence.get("target_submit_returned") or
     submit_evidence.get("target_submit_entered")))
  if submit_state_exact:
    expected_submitted = (
      True if submit_evidence["target_submit_returned"] else
      None if submit_evidence["target_submit_entered"] else False)
    submit_state_exact = \
      submit_evidence["target_dispatch_submitted"] is expected_submitted
  checks = {
    "schema_exact": row.get("schema") == RUNTIME_DISPATCH_FAILURE_SCHEMA,
    "failure_boundary_exact":
      row.get("failure_boundary") ==
      "runtime_call_raised_after_kernarg_capture_before_return",
    "wait_true": row.get("wait") is True,
    "epoch_exact": launch.get("epoch") == epoch,
    "global_size_exact":
      launch.get("global_size") == list(effective_global_size),
    "local_size_exact":
      launch.get("local_size") == list(authority.local_size),
    "argument_vas_exact": argument_vas == list(fixed_five_vas),
    "pre_submit_schema_exact":
      pre_submit.get("schema") == PM4_PRE_SUBMIT_SCHEMA,
    "pre_submit_capture_point_exact":
      pre_submit.get("capture_point") == PM4_PRE_SUBMIT_CAPTURE_POINT,
    "pre_submit_internal_checks_exact": pre_submit_checks_exact,
    "pre_submit_all_checks_pass":
      pre_submit.get("all_checks_pass") is True,
    "pre_submit_argument_vas_exact":
      pre_submit_argument_vas == list(fixed_five_vas),
    "pre_submit_kernarg_qwords_exact":
      pre_submit.get("kernarg_qwords") == list(fixed_five_vas),
    "pre_submit_pm4_kernarg_user_data_exact":
      pm4_user_data.get("register_index") == 0 and
      pm4_user_data.get("pointer") == pre_submit.get("kernarg_va") and
      pm4_user_data.get("pointer") == (
        pm4_user_data.get("low_dword", -1) |
        (pm4_user_data.get("high_dword", -1) << 32)),
    "pre_submit_runtime_object_identity_exact":
      pre_submit.get("runtime_object_identity") ==
      runtime_observation["runtime_object_identity"],
    "pre_submit_runtime_class_exact":
      pre_submit.get("runtime_class") == runtime_observation["runtime_class"],
    "pre_submit_runtime_name_exact":
      pre_submit.get("runtime_name") == runtime_observation["runtime_name"],
    "pre_submit_runtime_device_exact":
      pre_submit.get("runtime_device") == runtime_observation["runtime_device"],
    "pre_submit_dispatch_grid_exact":
      isinstance(pm4_dispatch, Mapping) and
      pm4_dispatch.get("group_counts") == list(effective_global_size),
    "pre_submit_workgroup_size_exact":
      isinstance(pm4_workgroup, Mapping) and
      pm4_workgroup.get("size") == list(authority.local_size),
    "pre_submit_program_entry_exact":
      isinstance(pm4_program, Mapping) and
      pm4_program.get("entry_va") == runtime_observation["entry_va"],
    "submit_evidence_exact": submit_state_exact,
  }
  post_failure_checks = {
    "kernarg_pointer_words_exact":
      kernarg.get("pointer_words") == list(fixed_five_vas),
    "kernarg_pointer_words_match_bound":
      kernarg.get("pointer_words_match_bound") is True,
  }
  observation = {
    "schema": RUNTIME_DISPATCH_FAILURE_SCHEMA,
    "failure_boundary": row.get("failure_boundary"),
    "wait": row.get("wait"), "epoch": epoch,
    "authoritative_qword_snapshot": "pre_submit",
    "pre_submit": pre_submit,
    "submit_evidence": submit_evidence,
    "launch": launch, "checks": checks,
    "post_failure_checks": post_failure_checks,
    "all_authority_checks_pass": all(checks.values()),
  }
  if effective_global_size != authority.global_size:
    observation.update({
      "frozen_global_size": list(authority.global_size),
      "effective_global_size": list(effective_global_size),
    })
  return observation


def _validate_pm4_no_doorbell_receipt(
    value: Any, *, runtime: Any,
    runtime_observation: Mapping[str, Any],
    fixed_five_vas: tuple[int, ...],
    expected_schema: str = PM4_NO_DOORBELL_RECEIPT_SCHEMA,
    ) -> dict[str, Any]:
  receipt = dict(_mapping(value, "PM4 no-doorbell receipt"))
  pre_submit = dict(_mapping(
    receipt.get("pre_submit"), "PM4 no-doorbell pre-submit snapshot"))
  pre_submit_arguments = pre_submit.get("argument_buffers")
  argument_vas = [
    item.get("va") for item in pre_submit_arguments
  ] if isinstance(pre_submit_arguments, list) and all(
    isinstance(item, Mapping) for item in pre_submit_arguments) else None
  argument_layout_exact = isinstance(pre_submit_arguments, list) and \
    len(pre_submit_arguments) == len(EXACT_ABI_NBYTES) and all(
      item.get("slot") == slot and item.get("va") == fixed_five_vas[slot] and
      item.get("size") == EXACT_ABI_NBYTES[slot]
      for slot, item in enumerate(pre_submit_arguments)
      if isinstance(item, Mapping))
  pre_submit_checks = pre_submit.get("checks")
  receipt_checks = receipt.get("checks")
  pm4_user_data = pre_submit.get("pm4_kernarg_user_data")
  native_submit_call_count = receipt.get("native_submit_call_count")
  timeline_before = receipt.get("timeline_value_before")
  timeline_after_unwind = receipt.get("timeline_value_after_runtime_unwind")
  timeline_after_rollback = receipt.get("timeline_value_after_rollback")
  prof_before = receipt.get("prof_exec_counter_before")
  prof_after_unwind = receipt.get("prof_exec_counter_after_runtime_unwind")
  prof_after_rollback = receipt.get("prof_exec_counter_after_rollback")
  timeline_signal_before = receipt.get("timeline_signal_value_before")
  timeline_signal_after = receipt.get("timeline_signal_value_after")
  exact_int = lambda item: isinstance(item, int) and not isinstance(item, bool)
  checks = {
    "schema_exact": receipt.get("schema") == expected_schema,
    "status_exact": receipt.get("status") == "CAPTURED_NO_SUBMIT",
    "submit_policy_exact":
      receipt.get("submit_policy") == "snapshot_only",
    "target_dispatch_not_submitted":
      receipt.get("target_dispatch_submitted") is False,
    "native_submit_never_called":
      exact_int(native_submit_call_count) and native_submit_call_count == 0,
    "ring_copy_not_performed": receipt.get("ring_copy_performed") is False,
    "doorbell_not_rung": receipt.get("doorbell_rung") is False,
    "timeline_rollback_applied":
      receipt.get("timeline_rollback_applied") is True,
    "terminal_child_required":
      receipt.get("terminal_child_required") is True,
    "promotion_evidence_ineligible":
      receipt.get("promotion_evidence_eligible") is False,
    "receipt_all_checks_pass": receipt.get("all_checks_pass") is True,
    "receipt_internal_checks_exact":
      isinstance(receipt_checks, Mapping) and
      set(receipt_checks) == PM4_NO_DOORBELL_CHECK_KEYS and
      all(item is True for item in receipt_checks.values()),
    "timeline_advanced_exactly_once":
      exact_int(timeline_before) and exact_int(timeline_after_unwind) and
      timeline_after_unwind == timeline_before + 1,
    "timeline_rollback_exact":
      exact_int(timeline_after_rollback) and
      timeline_after_rollback == timeline_before,
    "prof_exec_counter_advanced_exactly_once":
      exact_int(prof_before) and exact_int(prof_after_unwind) and
      prof_after_unwind == prof_before + 1,
    "prof_exec_counter_rollback_exact":
      exact_int(prof_after_rollback) and prof_after_rollback == prof_before,
    "timeline_signal_unchanged":
      exact_int(timeline_signal_before) and exact_int(timeline_signal_after) and
      timeline_signal_after == timeline_signal_before,
    "pre_submit_schema_exact":
      pre_submit.get("schema") == PM4_PRE_SUBMIT_SCHEMA,
    "pre_submit_capture_point_exact":
      pre_submit.get("capture_point") == PM4_PRE_SUBMIT_CAPTURE_POINT,
    "pre_submit_internal_checks_exact":
      isinstance(pre_submit_checks, Mapping) and
      set(pre_submit_checks) == PM4_PRE_SUBMIT_CHECK_KEYS and
      all(item is True for item in pre_submit_checks.values()),
    "pre_submit_all_checks_pass":
      pre_submit.get("all_checks_pass") is True,
    "pre_submit_argument_vas_exact":
      argument_vas == list(fixed_five_vas),
    "pre_submit_argument_layout_exact": argument_layout_exact,
    "pre_submit_kernarg_qwords_exact":
      pre_submit.get("kernarg_qwords") == list(fixed_five_vas),
    "pre_submit_runtime_object_identity_exact":
      pre_submit.get("runtime_object_identity") == id(runtime) ==
      runtime_observation["runtime_object_identity"],
    "pre_submit_runtime_class_exact":
      pre_submit.get("runtime_class") == runtime_observation["runtime_class"],
    "pre_submit_runtime_name_exact":
      pre_submit.get("runtime_name") == runtime_observation["runtime_name"],
    "pre_submit_runtime_device_exact":
      pre_submit.get("runtime_device") == runtime_observation["runtime_device"],
    "pre_submit_pm4_kernarg_user_data_exact":
      isinstance(pm4_user_data, Mapping) and
      exact_int(pm4_user_data.get("low_dword")) and
      exact_int(pm4_user_data.get("high_dword")) and
      exact_int(pm4_user_data.get("register_index")) and
      pm4_user_data.get("register_index") == 0 and
      pm4_user_data.get("pointer") == pre_submit.get("kernarg_va") and
      pm4_user_data.get("pointer") == (
        pm4_user_data.get("low_dword", -1) |
        (pm4_user_data.get("high_dword", -1) << 32)),
  }
  if not all(checks.values()):
    raise ValueError(
      "PM4 no-doorbell receipt differs from authority: "
      f"{sorted(key for key, passed in checks.items() if not passed)!r}")
  return copy.deepcopy(receipt)


def _validate_diagnostic_pm4_pre_submit(
    value: Any, *, runtime: Any,
    runtime_observation: Mapping[str, Any],
    fixed_five_vas: tuple[int, ...],
    effective_global_size: tuple[int, int, int],
    local_size: tuple[int, int, int],
    ) -> dict[str, Any]:
  snapshot = dict(_mapping(value, "diagnostic PM4 pre-submit snapshot"))
  argument_buffers = snapshot.get("argument_buffers")
  argument_vas = [
    item.get("va") for item in argument_buffers
  ] if isinstance(argument_buffers, list) and all(
    isinstance(item, Mapping) for item in argument_buffers) else None
  checks = snapshot.get("checks")
  user_data = snapshot.get("pm4_kernarg_user_data")
  dispatch_direct = snapshot.get("pm4_dispatch_direct")
  workgroup_size = snapshot.get("pm4_workgroup_size")
  program_entry = snapshot.get("pm4_program_entry")
  exact_int = lambda item: isinstance(item, int) and not isinstance(item, bool)
  authority_checks = {
    "schema_exact": snapshot.get("schema") == PM4_PRE_SUBMIT_SCHEMA,
    "capture_point_exact":
      snapshot.get("capture_point") == PM4_PRE_SUBMIT_CAPTURE_POINT,
    "internal_checks_exact":
      isinstance(checks, Mapping) and
      set(checks) == PM4_PRE_SUBMIT_CHECK_KEYS and
      all(item is True for item in checks.values()),
    "all_checks_pass": snapshot.get("all_checks_pass") is True,
    "argument_vas_exact": argument_vas == list(fixed_five_vas),
    "argument_layout_exact":
      isinstance(argument_buffers, list) and
      len(argument_buffers) == len(EXACT_ABI_NBYTES) and all(
        isinstance(item, Mapping) and item.get("slot") == slot and
        item.get("va") == fixed_five_vas[slot] and
        item.get("size") == EXACT_ABI_NBYTES[slot]
        for slot, item in enumerate(argument_buffers)),
    "kernarg_qwords_exact":
      snapshot.get("kernarg_qwords") == list(fixed_five_vas),
    "kernarg_layout_exact":
      exact_int(snapshot.get("kernarg_va")) and
      snapshot.get("kernarg_va") > 0 and
      snapshot.get("kernarg_nbytes") == 40,
    "runtime_object_identity_exact":
      snapshot.get("runtime_object_identity") == id(runtime) ==
      runtime_observation["runtime_object_identity"],
    "runtime_class_exact":
      snapshot.get("runtime_class") == runtime_observation["runtime_class"],
    "runtime_name_exact":
      snapshot.get("runtime_name") == runtime_observation["runtime_name"],
    "runtime_device_exact":
      snapshot.get("runtime_device") == runtime_observation["runtime_device"],
    "kernarg_user_data_exact":
      isinstance(user_data, Mapping) and
      user_data.get("register_index") == 0 and
      exact_int(user_data.get("low_dword")) and
      exact_int(user_data.get("high_dword")) and
      user_data.get("pointer") == snapshot.get("kernarg_va") and
      user_data.get("pointer") == (
        user_data.get("low_dword", -1) |
        (user_data.get("high_dword", -1) << 32)),
    "pm4_stream_identity_concrete":
      exact_int(snapshot.get("pm4_dword_count")) and
      snapshot.get("pm4_dword_count") > 0 and
      isinstance(snapshot.get("pm4_sha256"), str) and
      len(snapshot.get("pm4_sha256")) == 64 and
      all(char in _HEX for char in snapshot.get("pm4_sha256")),
    "dispatch_direct_exact":
      isinstance(dispatch_direct, Mapping) and
      dispatch_direct.get("group_counts") == list(effective_global_size),
    "workgroup_size_exact":
      isinstance(workgroup_size, Mapping) and
      workgroup_size.get("size") == list(local_size),
    "program_entry_exact":
      isinstance(program_entry, Mapping) and
      program_entry.get("entry_va") == runtime_observation["entry_va"],
  }
  if not all(authority_checks.values()):
    raise ValueError(
      "diagnostic PM4 pre-submit differs from authority: "
      f"{sorted(key for key, passed in authority_checks.items()
                if not passed)!r}")
  return copy.deepcopy(snapshot)


def _dispatch_production_runtime(
    runtime: Any, buffers: tuple[Any, ...],
    program_authority: FrozenStagedProgramAuthority, epoch: int, *,
    runtime_observation: Mapping[str, Any],
    dispatch_with_runtime_evidence: Callable[..., None],
    pm4_submit_policy: str,
    pm4_no_doorbell_receipt_sink: list[Mapping[str, Any]] | None,
    fixed_five_vas: tuple[int, ...],
    effective_global_size: tuple[int, int, int] | None = None,
    receipt_schema: str = PM4_NO_DOORBELL_RECEIPT_SCHEMA,
    ) -> Mapping[str, Any]:
  effective_grid = program_authority.global_size \
    if effective_global_size is None else _tuple3(
      effective_global_size, "effective global size")
  if effective_global_size is not None and \
     effective_grid not in DIAGNOSTIC_GLOBAL_SIZE_ALLOWLIST:
    raise ValueError("effective diagnostic global size is not allowlisted")
  evidence: dict[str, Any] = {"launch_count": 0, "launches": []}
  dispatch_kwargs: dict[str, Any] = {
    "global_size": effective_grid,
    "local_size": program_authority.local_size,
    "vals": (),
    "runtime_evidence": evidence,
    "context": {"epoch": epoch},
    "wait": True,
    "pm4_submit_policy": pm4_submit_policy,
  }
  if effective_global_size is not None:
    dispatch_kwargs["retain_pm4_pre_submit"] = True
  dispatch_with_runtime_evidence(
    runtime, buffers, program_authority.globals,
    # ``from_binding`` rejects any scalar values. This compact target has
    # exactly five pointer globals and therefore an empty scalar vals tuple.
    **dispatch_kwargs)
  launches = evidence.get("launches")
  if not isinstance(launches, list) or not launches:
    raise ValueError("runtime dispatch produced no launch evidence")
  launch_count = evidence.get("launch_count")
  if pm4_submit_policy == "snapshot_only" and (
      not isinstance(launch_count, int) or isinstance(launch_count, bool) or
      launch_count != 1 or len(launches) != 1):
    raise ValueError(
      "snapshot-only dispatch requires exactly one launch receipt")
  launch = dict(_mapping(launches[-1], "captured target launch"))
  kernarg = _mapping(launch.get("kernarg"), "captured target kernarg")
  row: dict[str, Any] = {
    "epoch": epoch, "program_key": program_authority.program_key,
    "binary_sha256": program_authority.binary_sha256,
    "global_size": list(effective_grid),
    "local_size": list(program_authority.local_size),
    "argument_vas": [
      item["va"] for item in launch["arguments"]],
    "kernarg_pointer_words": kernarg.get("pointer_words"),
    "kernarg_pointer_words_match_bound":
      kernarg.get("pointer_words_match_bound"),
  }
  if effective_global_size is not None:
    row["frozen_global_size"] = list(program_authority.global_size)
    row["effective_global_size"] = list(effective_grid)
    if pm4_submit_policy != "execute":
      raise ValueError("diagnostic dispatch requires execute PM4")
    row["pre_submit"] = _validate_diagnostic_pm4_pre_submit(
      launch.get("pre_submit"), runtime=runtime,
      runtime_observation=runtime_observation,
      fixed_five_vas=fixed_five_vas,
      effective_global_size=effective_grid,
      local_size=program_authority.local_size)
    submit_evidence = dict(_mapping(
      launch.get("submit_evidence"), "diagnostic PM4 submit evidence"))
    expected_submit_evidence = {
      "native_submit_entered_count": 1,
      "native_submit_returned_count": 1,
      "target_submit_entered": True,
      "target_submit_returned": True,
      "target_dispatch_submitted": True,
    }
    if submit_evidence != expected_submit_evidence:
      raise ValueError(
        "diagnostic PM4 submit evidence differs from one returned target")
    row["submit_evidence"] = submit_evidence
  if pm4_submit_policy == "snapshot_only":
    if row["argument_vas"] != list(fixed_five_vas) or \
       row["kernarg_pointer_words"] != list(fixed_five_vas) or \
       row["kernarg_pointer_words_match_bound"] is not True:
      raise ValueError(
        "snapshot-only launch pointers differ from fixed five-buffer authority")
    if "pm4_no_doorbell_receipt" not in launch:
      raise ValueError(
        "snapshot-only dispatch produced no PM4 no-doorbell receipt")
    receipt = _validate_pm4_no_doorbell_receipt(
      launch["pm4_no_doorbell_receipt"], runtime=runtime,
      runtime_observation=runtime_observation,
      fixed_five_vas=fixed_five_vas,
      expected_schema=receipt_schema)
    row["pm4_no_doorbell_receipt"] = copy.deepcopy(receipt)
    assert pm4_no_doorbell_receipt_sink is not None
    pm4_no_doorbell_receipt_sink.append(copy.deepcopy(receipt))
  elif "pm4_no_doorbell_receipt" in launch:
    raise ValueError("execute dispatch unexpectedly produced no-doorbell receipt")
  return row


def _runtime_observation(value: Any, authority: FrozenStagedProgramAuthority) -> dict[str, Any]:
  row = _mapping(value, "runtime observation")
  required = {
    "queue_mode", "runtime_class", "runtime_name", "runtime_device",
    "program_key", "binary_sha256", "runtime_object_identity",
    "runtime_device_identity_exact", "runtime_cache_binding_exact",
    "library_va", "library_nbytes", "entry_va",
  }
  if set(row) != required:
    raise ValueError("runtime observation fields differ")
  if row["queue_mode"] not in QUEUE_MODES or \
     row["program_key"] != authority.program_key or \
     row["binary_sha256"] != authority.binary_sha256:
    raise ValueError("runtime queue/PROGRAM/binary identity differs")
  runtime_class = _nonempty(row["runtime_class"], "runtime class")
  runtime_name = _nonempty(row["runtime_name"], "runtime name")
  runtime_device = _nonempty(row["runtime_device"], "runtime device")
  if runtime_name != authority.function_name or runtime_device != "AMD" or \
     row["runtime_device_identity_exact"] is not True:
    raise ValueError("runtime name/device identity differs")
  runtime_object_identity = _positive(
    row["runtime_object_identity"], "runtime object identity")
  if row["runtime_cache_binding_exact"] is not True:
    raise ValueError("runtime cache does not own the exact runtime object")
  library_va = _positive(row["library_va"], "runtime library VA")
  library_nbytes = _positive(row["library_nbytes"], "runtime library nbytes")
  entry_va = _positive(row["entry_va"], "runtime entry VA")
  if not library_va <= entry_va < library_va + library_nbytes:
    raise ValueError("runtime entry is outside its uploaded code object")
  return {
    "queue_mode": row["queue_mode"], "runtime_class": runtime_class,
    "runtime_name": runtime_name, "runtime_device": runtime_device,
    "runtime_object_identity": runtime_object_identity,
    "runtime_device_identity_exact": True,
    "runtime_cache_binding_exact": True,
    "program_key": row["program_key"],
    "binary_sha256": row["binary_sha256"],
    "library_va": library_va, "library_nbytes": library_nbytes,
    "entry_va": entry_va,
  }


class FrozenStagedLowLevelSession:
  """One queue-selected runtime with persistent fixed-VA candidate storage."""

  def __init__(
      self, *, binding: Any, authority: FrozenStagedProgramAuthority,
      common_resident_fp16: Any, q4_epoch_major: Any,
      dependencies: FrozenStagedLowLevelDependencies, runtime: Any,
      output: Any, stages: tuple[Any, Any, Any, Any],
      fixed_five_vas: tuple[int, ...],
      resident_fp16_observation: Mapping[str, int],
      q4_observation: Mapping[str, int],
      runtime_observation: Mapping[str, Any],
      ) -> None:
    self.binding, self.authority = binding, authority
    self.common_resident_fp16 = common_resident_fp16
    self.q4_epoch_major = q4_epoch_major
    self.dependencies, self.runtime = dependencies, runtime
    self.output, self.stages = output, stages
    self.fixed_five_vas = fixed_five_vas
    self.resident_fp16_observation = dict(resident_fp16_observation)
    self.q4_observation = dict(q4_observation)
    self.prepared_runtime_observation = dict(runtime_observation)
    self._pending: _PendingState | None = None
    self._failed = False
    self._failed_q8_sources: tuple[Any, Any, Any] | None = None
    self._invocation_ordinal = 0

  @classmethod
  def prepare(
      cls, *, binding: Any, authority: FrozenStagedProgramAuthority,
      common_resident_fp16: Any, q4_epoch_major: Any,
      dependencies: FrozenStagedLowLevelDependencies,
      ) -> "FrozenStagedLowLevelSession":
    authority.validate()
    dependencies.validate()
    observed_binding = dict(_mapping(
      dependencies.observe_binding(binding), "binding observation"))
    if observed_binding != authority.binding_observation():
      raise ValueError("loaded frozen binding differs from PROGRAM authority")

    output = dependencies.allocate(authority.abi[0])
    stages = tuple(dependencies.allocate(slot) for slot in authority.abi[1:])
    dependencies.realize_many((
      common_resident_fp16, q4_epoch_major, output, *stages))
    dependencies.synchronize()
    activation_bytes = authority.full_shape[0] * authority.full_shape[2] * 2
    resident_row = _buffer_observation(
      dependencies, common_resident_fp16, label="resident FP16 activation",
      expected_nbytes=activation_bytes)
    q4_row = _buffer_observation(
      dependencies, q4_epoch_major, label="epoch-major Q4 source",
      expected_nbytes=authority.abi[1].nbytes * authority.dispatch_count)
    fixed_rows = [
      _buffer_observation(
        dependencies, value, label=f"fixed ABI {slot.name}",
        expected_nbytes=slot.nbytes)
      for slot, value in zip(authority.abi, (output, *stages))
    ]
    fixed_vas = tuple(row["va"] for row in fixed_rows)
    runtime = dependencies.create_runtime(binding)
    runtime_row = _runtime_observation(
      dependencies.observe_runtime(runtime), authority)
    code_row = {
      "va": runtime_row["library_va"],
      "nbytes": runtime_row["library_nbytes"]}
    _require_disjoint_buffer_ranges(
      [resident_row, q4_row, *fixed_rows, code_row],
      "prepared source/persistent/code")
    return cls(
      binding=binding, authority=authority,
      common_resident_fp16=common_resident_fp16,
      q4_epoch_major=q4_epoch_major, dependencies=dependencies,
      runtime=runtime, output=output, stages=stages,
      fixed_five_vas=fixed_vas,
      resident_fp16_observation=resident_row, q4_observation=q4_row,
      runtime_observation=runtime_row)

  @property
  def has_pending_invocation(self) -> bool:
    return self._pending is not None

  def _phase(
      self, cursor: int, operation: Callable[[], Any],
      ) -> tuple[dict[str, int], Any, int]:
    value = operation()
    ended = self.dependencies.clock_ns()
    if not isinstance(ended, int) or isinstance(ended, bool) or ended <= cursor:
      raise ValueError("low-level phase clock must advance monotonically")
    return {"start_ns": cursor, "end_ns": ended}, value, ended

  def _assert_fixed_vas(self) -> list[dict[str, int]]:
    observations = [
      _buffer_observation(
        self.dependencies, value, label=f"fixed ABI {slot.name}",
        expected_nbytes=slot.nbytes)
      for slot, value in zip(self.authority.abi, (self.output, *self.stages))]
    _require_disjoint_buffer_ranges(
      observations, "persistent five-buffer launch")
    observed = tuple(row["va"] for row in observations)
    if observed != self.fixed_five_vas:
      raise ValueError("persistent five-buffer VAs drifted")
    return observations

  def _assert_all_ranges(
      self, q8_sources: tuple[Any, Any, Any],
      ) -> tuple[dict[str, int], ...]:
    resident = _buffer_observation(
      self.dependencies, self.common_resident_fp16,
      label="resident FP16 activation",
      expected_nbytes=self.resident_fp16_observation["nbytes"])
    q4 = _buffer_observation(
      self.dependencies, self.q4_epoch_major,
      label="epoch-major Q4 source",
      expected_nbytes=self.q4_observation["nbytes"])
    if resident != self.resident_fp16_observation or \
       q4 != self.q4_observation:
      raise ValueError("persistent source VA/extent drifted")
    fixed = self._assert_fixed_vas()
    q8 = tuple(
      _buffer_observation(
        self.dependencies, source, label=f"full {slot.name} source",
        expected_nbytes=slot.nbytes * self.authority.dispatch_count,
        expected_dtype=EXACT_Q8_SOURCE_DTYPES[index],
        expected_shape=EXACT_Q8_SOURCE_SHAPES[index])
      for index, (slot, source) in enumerate(
        zip(self.authority.abi[2:], q8_sources)))
    code = {
      "va": self.prepared_runtime_observation["library_va"],
      "nbytes": self.prepared_runtime_observation["library_nbytes"]}
    _require_disjoint_buffer_ranges(
      [resident, q4, *fixed, *q8, code],
      "complete source/persistent/code")
    return (resident, q4, *q8)

  def invoke(self, prefix_epochs: int) -> FrozenStagedLowLevelInvocation:
    if self.dependencies.diagnostic_global_size is not None:
      raise RuntimeError(
        "ordinary invocation rejects diagnostic-configured dependencies")
    return self._invoke(prefix_epochs, diagnostic_global_size=None)

  def invoke_diagnostic_one_epoch(
      self, grid: tuple[int, int, int],
      ) -> FrozenStagedLowLevelInvocation:
    effective_grid = _tuple3(grid, "diagnostic global size")
    if effective_grid not in DIAGNOSTIC_GLOBAL_SIZE_ALLOWLIST:
      raise ValueError("diagnostic global size is not allowlisted")
    if self.dependencies.diagnostic_global_size is None:
      raise RuntimeError(
        "diagnostic invocation requires diagnostic-configured dependencies")
    if effective_grid != self.dependencies.diagnostic_global_size:
      raise ValueError(
        "diagnostic invocation grid differs from configured dependency grid")
    if self.prepared_runtime_observation["queue_mode"] != "PM4":
      raise ValueError("diagnostic invocation requires PM4 queue mode")
    return self._invoke(1, diagnostic_global_size=effective_grid)

  def _invoke(
      self, prefix_epochs: int, *,
      diagnostic_global_size: tuple[int, int, int] | None,
      ) -> FrozenStagedLowLevelInvocation:
    allowed = (1, 3, self.authority.dispatch_count)
    if diagnostic_global_size is None and (
        not isinstance(prefix_epochs, int) or isinstance(prefix_epochs, bool) or
        prefix_epochs not in allowed):
      raise ValueError(f"prefix_epochs must be one of {allowed!r}")
    if diagnostic_global_size is not None and prefix_epochs != 1:
      raise ValueError("diagnostic invocation requires exactly one epoch")
    if self._failed:
      raise RuntimeError("frozen staged low-level session is failed closed")
    if self._pending is not None:
      raise RuntimeError("prior invocation has not been attested post-sync")
    cursor = self.dependencies.clock_ns()
    if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < 0:
      raise ValueError("low-level invocation clock must be non-negative")
    q8_sources: tuple[Any, Any, Any] | None = None
    failure_phase, failure_epoch = "activation_producer", None
    try:
      def produce() -> tuple[Any, Any, Any]:
        nonlocal q8_sources
        produced = self.dependencies.produce_q8(
          self.common_resident_fp16, self.authority)
        if not isinstance(produced, tuple) or len(produced) != 3:
          raise TypeError("physical Q8 producer must return three tensors")
        q8_sources = produced
        self.dependencies.realize_many(produced)
        self.dependencies.synchronize()
        return produced

      activation_phase, q8_sources, cursor = self._phase(cursor, produce)

      failure_phase = "route_setup"
      def setup() -> None:
        current_runtime = _runtime_observation(
          self.dependencies.observe_runtime(self.runtime), self.authority)
        if current_runtime != self.prepared_runtime_observation:
          raise ValueError("prepared runtime/code range drifted")
        self._assert_all_ranges(q8_sources)

      route_setup_phase, _, cursor = self._phase(cursor, setup)

      failure_phase = "output_initialization"
      def initialize_output() -> None:
        self.dependencies.zero_output(self.output)
        self.dependencies.synchronize()

      output_phase, _, cursor = self._phase(cursor, initialize_output)
      epoch_rows, launch_rows = [], []
      sources = (self.q4_epoch_major, *q8_sources)
      source_observations = (
        self.q4_observation,
        *self._assert_all_ranges(q8_sources)[2:])
      for epoch in range(prefix_epochs):
        failure_epoch = epoch
        epoch_row: dict[str, Any] = {"ordinal": epoch}

        failure_phase = "epoch_gather"
        def gather() -> tuple[Any, Any, Any, Any]:
          views = tuple(
            self.dependencies.epoch_view(source, slot, epoch)
            for source, slot in zip(sources, self.authority.abi[1:]))
          for slot, view, parent in zip(
              self.authority.abi[1:], views, source_observations):
            observed_view = _buffer_observation(
              self.dependencies, view,
              label=f"epoch {epoch} {slot.name} source view",
              expected_nbytes=slot.nbytes)
            if observed_view["va"] != parent["va"] + epoch * slot.nbytes:
              raise ValueError(
                f"epoch {epoch} {slot.name} source view VA differs")
          return views

        epoch_row["gather"], views, cursor = self._phase(cursor, gather)
        for phase_name, source, destination, slot in zip(
            _EPOCH_PHASES[1:5], views, self.stages, self.authority.abi[1:]):
          failure_phase = f"epoch_{phase_name}"
          def transfer(
              source: Any = source, destination: Any = destination,
              nbytes: int = slot.nbytes,
              ) -> None:
            self.dependencies.transfer(destination, source, nbytes)
          epoch_row[phase_name], _, cursor = self._phase(cursor, transfer)
        failure_phase = "epoch_staging_sync"
        epoch_row["staging_sync"], _, cursor = self._phase(
          cursor, self.dependencies.synchronize)

        failure_phase = "epoch_dispatch"
        def dispatch() -> Mapping[str, Any]:
          return self.dependencies.dispatch(
            self.runtime, (self.output, *self.stages),
            self.authority, epoch)

        epoch_row["dispatch"], launch, cursor = self._phase(cursor, dispatch)
        launch_rows.append(dict(_mapping(
          launch, f"epoch {epoch} dispatch observation")))
        failure_phase = "epoch_dispatch_sync"
        epoch_row["dispatch_sync"], _, cursor = self._phase(
          cursor, self.dependencies.synchronize)
        epoch_rows.append(epoch_row)

      pending_payload = {
        "schema": (
          PENDING_OBSERVATION_SCHEMA if diagnostic_global_size is None else
          DIAGNOSTIC_PENDING_OBSERVATION_SCHEMA),
        "invocation_ordinal": self._invocation_ordinal,
        "prefix_epochs": prefix_epochs,
        "queue_mode": self.prepared_runtime_observation["queue_mode"],
        "family_identity": self.authority.family_identity,
        "candidate_executable_identity":
          self.authority.candidate_executable_identity,
        "input_identity": self.authority.input_identity,
        "program_key": self.authority.program_key,
        "binary_sha256": self.authority.binary_sha256,
        "fixed_five_vas": list(self.fixed_five_vas),
        "launch_count": len(launch_rows),
      }
      if diagnostic_global_size is not None:
        pending_payload.update({
          "promotion_evidence_eligible": False,
          "frozen_global_size": list(self.authority.global_size),
          "effective_global_size": list(diagnostic_global_size),
          "local_size": list(self.authority.local_size),
        })
      pending_observation = {
        **pending_payload, "observation_identity": _identity(pending_payload)}
      trace = {
        "schema": CANDIDATE_TRACE_SCHEMA,
        "activation_producer": activation_phase,
        "route_setup": route_setup_phase,
        "output_initialization": output_phase,
        "epochs": epoch_rows,
      }
      invocation = FrozenStagedLowLevelInvocation(
        self.output, trace, pending_observation)
      self._pending = _PendingState(
        invocation=invocation, q8_sources=q8_sources,
        runtime_observation=dict(self.prepared_runtime_observation),
        launch_observations=tuple(launch_rows),
        source_observations=tuple(source_observations),
        diagnostic=diagnostic_global_size is not None,
        effective_global_size=(
          self.authority.global_size if diagnostic_global_size is None else
          diagnostic_global_size))
      self._invocation_ordinal += 1
      return invocation
    except BaseException as exc:
      self._failed = True
      self._pending = None
      dispatch_failure = None if failure_epoch is None else \
        _failed_dispatch_observation(
          exc, epoch=failure_epoch, authority=self.authority,
          effective_global_size=(
            self.authority.global_size if diagnostic_global_size is None else
            diagnostic_global_size),
          fixed_five_vas=self.fixed_five_vas,
          runtime_observation=self.prepared_runtime_observation)
      failure_payload = {
        "schema": INVOCATION_FAILURE_SCHEMA,
        "phase": failure_phase,
        "subphase": (
          dispatch_failure["failure_boundary"]
          if dispatch_failure is not None else None),
        "epoch": failure_epoch,
        "queue_mode": self.prepared_runtime_observation["queue_mode"],
        "family_identity": self.authority.family_identity,
        "candidate_executable_identity":
          self.authority.candidate_executable_identity,
        "input_identity": self.authority.input_identity,
        "program_key": self.authority.program_key,
        "binary_sha256": self.authority.binary_sha256,
        "runtime_observation": dict(self.prepared_runtime_observation),
        "fixed_five_vas": list(self.fixed_five_vas),
        "dispatch_failure": dispatch_failure,
      }
      if diagnostic_global_size is not None:
        failure_payload.update({
          "diagnostic": True,
          "promotion_evidence_eligible": False,
          "frozen_global_size": list(self.authority.global_size),
          "effective_global_size": list(diagnostic_global_size),
        })
      try:
        setattr(exc, INVOCATION_FAILURE_ATTR, failure_payload)
      except BaseException:
        # Preserve the original failure even if its exception object cannot
        # carry diagnostic attributes.
        pass
      if q8_sources is not None:
        try:
          # Drain any submitted producer, SDMA, or dispatch work before
          # releasing its source allocations on the exceptional path.
          self.dependencies.synchronize()
        except BaseException:
          # If the queue cannot be drained, retain the sources for the failed
          # session's entire lifetime rather than risk use-after-free.
          self._failed_q8_sources = q8_sources
        else:
          q8_sources = None
      raise

  def attest_post_sync(
      self, invocation: FrozenStagedLowLevelInvocation, queue_mode: str,
      ) -> FrozenStagedLowLevelAttestation:
    if self._failed:
      raise RuntimeError("frozen staged low-level session is failed closed")
    pending = self._pending
    if pending is None or invocation is not pending.invocation:
      raise ValueError("post-sync attestation invocation identity differs")
    if pending.diagnostic:
      raise ValueError(
        "ordinary post-sync attestation rejects diagnostic invocation")
    try:
      if queue_mode not in QUEUE_MODES or \
         queue_mode != pending.runtime_observation["queue_mode"]:
        raise ValueError("post-sync attestation queue mode differs")
      current_runtime = _runtime_observation(
        self.dependencies.observe_runtime(self.runtime), self.authority)
      if current_runtime != pending.runtime_observation:
        raise ValueError("post-sync runtime/code range drifted")
      current_sources = (
        self.q4_observation,
        *self._assert_all_ranges(pending.q8_sources)[2:])
      if tuple(current_sources) != pending.source_observations:
        raise ValueError("post-sync source VA/extent drifted")
      prefix_epochs = invocation.pending_observation["prefix_epochs"]
      if len(pending.launch_observations) != prefix_epochs:
        raise ValueError("post-sync launch count differs from prefix")
      base_expected_launch_keys = {
        "epoch", "program_key", "binary_sha256", "global_size",
        "local_size", "argument_vas", "kernarg_pointer_words",
        "kernarg_pointer_words_match_bound",
      }
      for epoch, row in enumerate(pending.launch_observations):
        row_keys = set(row)
        snapshot_only = "pm4_no_doorbell_receipt" in row
        expected_launch_keys = base_expected_launch_keys | (
          {"pm4_no_doorbell_receipt"} if snapshot_only else set())
        if row_keys != expected_launch_keys or \
           row["epoch"] != epoch or \
           row["program_key"] != self.authority.program_key or \
           row["binary_sha256"] != self.authority.binary_sha256 or \
           row["global_size"] != list(self.authority.global_size) or \
           row["local_size"] != list(self.authority.local_size) or \
           row["argument_vas"] != list(self.fixed_five_vas) or \
           row["kernarg_pointer_words"] != list(self.fixed_five_vas) or \
           row["kernarg_pointer_words_match_bound"] is not True:
          raise ValueError(
            f"post-sync launch observation {epoch} differs from authority")
        if snapshot_only:
          _validate_pm4_no_doorbell_receipt(
            row["pm4_no_doorbell_receipt"], runtime=self.runtime,
            runtime_observation=current_runtime,
            fixed_five_vas=self.fixed_five_vas)
      payload = {
        "schema": ATTESTATION_SCHEMA, "status": "PASS",
        "queue_mode": queue_mode,
        "family_identity": self.authority.family_identity,
        "candidate_executable_identity":
          self.authority.candidate_executable_identity,
        "input_identity": self.authority.input_identity,
        "program_key": self.authority.program_key,
        "binary_sha256": self.authority.binary_sha256,
        "runtime_class": current_runtime["runtime_class"],
        "runtime_name": current_runtime["runtime_name"],
        "runtime_device": current_runtime["runtime_device"],
        "runtime_object_identity":
          current_runtime["runtime_object_identity"],
        "runtime_device_identity_exact":
          current_runtime["runtime_device_identity_exact"],
        "runtime_cache_binding_exact":
          current_runtime["runtime_cache_binding_exact"],
        "library_va": current_runtime["library_va"],
        "library_nbytes": current_runtime["library_nbytes"],
        "entry_va": current_runtime["entry_va"],
        "fixed_five_vas": list(self.fixed_five_vas),
        "launch_count": len(pending.launch_observations),
      }
      attestation = FrozenStagedLowLevelAttestation(
        **{**payload, "fixed_five_vas": self.fixed_five_vas},
        observation_identity=_identity(payload))
    except BaseException:
      self._failed = True
      raise
    finally:
      # Successful or failed attestation is terminal for the pending
      # invocation. Q8 sources remain strongly held through this point only.
      self._pending = None
    return attestation

  def complete_diagnostic_post_sync(
      self, invocation: FrozenStagedLowLevelInvocation, queue_mode: str,
      ) -> FrozenStagedLowLevelDiagnosticReceipt:
    if self._failed:
      raise RuntimeError("frozen staged low-level session is failed closed")
    pending = self._pending
    if pending is None or invocation is not pending.invocation:
      raise ValueError("diagnostic post-sync invocation identity differs")
    if not pending.diagnostic:
      raise ValueError(
        "diagnostic post-sync completion rejects ordinary invocation")
    try:
      if queue_mode != "PM4" or \
         queue_mode != pending.runtime_observation["queue_mode"]:
        raise ValueError("diagnostic post-sync requires PM4 queue mode")
      current_runtime = _runtime_observation(
        self.dependencies.observe_runtime(self.runtime), self.authority)
      if current_runtime != pending.runtime_observation:
        raise ValueError("diagnostic post-sync runtime/code range drifted")
      current_sources = (
        self.q4_observation,
        *self._assert_all_ranges(pending.q8_sources)[2:])
      if tuple(current_sources) != pending.source_observations:
        raise ValueError("diagnostic post-sync source VA/extent drifted")
      if len(pending.launch_observations) != 1:
        raise ValueError("diagnostic launch count differs from one epoch")
      row = pending.launch_observations[0]
      expected_keys = {
        "epoch", "program_key", "binary_sha256", "global_size",
        "frozen_global_size", "effective_global_size", "local_size",
        "argument_vas", "kernarg_pointer_words", "submit_evidence",
        "kernarg_pointer_words_match_bound", "pre_submit",
      }
      pre_submit = _validate_diagnostic_pm4_pre_submit(
        row.get("pre_submit"), runtime=self.runtime,
        runtime_observation=current_runtime,
        fixed_five_vas=self.fixed_five_vas,
        effective_global_size=pending.effective_global_size,
        local_size=self.authority.local_size)
      checks = {
        "launch_fields_exact": set(row) == expected_keys,
        "epoch_zero": row.get("epoch") == 0,
        "program_key_exact":
          row.get("program_key") == self.authority.program_key,
        "binary_exact":
          row.get("binary_sha256") == self.authority.binary_sha256,
        "frozen_grid_exact":
          row.get("frozen_global_size") == list(self.authority.global_size),
        "effective_grid_exact":
          row.get("global_size") == list(pending.effective_global_size) and
          row.get("effective_global_size") ==
          list(pending.effective_global_size),
        "local_size_exact":
          row.get("local_size") == list(self.authority.local_size),
        "argument_vas_exact":
          row.get("argument_vas") == list(self.fixed_five_vas),
        "kernarg_qwords_exact":
          row.get("kernarg_pointer_words") == list(self.fixed_five_vas),
        "kernarg_qwords_match_bound":
          row.get("kernarg_pointer_words_match_bound") is True,
        "no_no_doorbell_receipt":
          "pm4_no_doorbell_receipt" not in row,
        "submit_evidence_exact": row.get("submit_evidence") == {
          "native_submit_entered_count": 1,
          "native_submit_returned_count": 1,
          "target_submit_entered": True,
          "target_submit_returned": True,
          "target_dispatch_submitted": True,
        },
      }
      if not all(checks.values()):
        raise ValueError(
          "diagnostic post-sync launch differs from authority: "
          f"{sorted(key for key, passed in checks.items() if not passed)!r}")
      payload = {
        "schema": DIAGNOSTIC_RECEIPT_SCHEMA, "status": "PASS",
        "queue_mode": queue_mode,
        "promotion_evidence_eligible": False,
        "target_dispatch_submitted": True,
        "native_submit_entered_count": 1,
        "native_submit_returned_count": 1,
        "target_submit_entered": True,
        "target_submit_returned": True,
        "post_dispatch_sync_completed": True,
        "family_identity": self.authority.family_identity,
        "candidate_executable_identity":
          self.authority.candidate_executable_identity,
        "input_identity": self.authority.input_identity,
        "program_key": self.authority.program_key,
        "binary_sha256": self.authority.binary_sha256,
        "runtime_class": current_runtime["runtime_class"],
        "runtime_name": current_runtime["runtime_name"],
        "runtime_device": current_runtime["runtime_device"],
        "runtime_object_identity":
          current_runtime["runtime_object_identity"],
        "runtime_device_identity_exact":
          current_runtime["runtime_device_identity_exact"],
        "runtime_cache_binding_exact":
          current_runtime["runtime_cache_binding_exact"],
        "library_va": current_runtime["library_va"],
        "library_nbytes": current_runtime["library_nbytes"],
        "entry_va": current_runtime["entry_va"],
        "fixed_five_vas": list(self.fixed_five_vas),
        "frozen_global_size": list(self.authority.global_size),
        "effective_global_size": list(pending.effective_global_size),
        "local_size": list(self.authority.local_size),
        "pre_submit": pre_submit,
        "launch_count": 1,
      }
      payload_with_identity = {
        **payload, "observation_identity": _identity(payload)}
      receipt = FrozenStagedLowLevelDiagnosticReceipt(
        **{
          **payload_with_identity,
          "fixed_five_vas": self.fixed_five_vas,
          "frozen_global_size": self.authority.global_size,
          "effective_global_size": pending.effective_global_size,
          "local_size": self.authority.local_size,
        })
      sink = self.dependencies.diagnostic_dispatch_receipt_sink
      if not isinstance(sink, list):
        raise RuntimeError("diagnostic receipt sink is unavailable")
      sink.append(copy.deepcopy(payload_with_identity))
    except BaseException:
      self._failed = True
      raise
    finally:
      self._pending = None
    return receipt


def production_frozen_staged_low_level_dependencies(
    authority: FrozenStagedProgramAuthority,
    *, pm4_submit_policy: str = "execute",
    pm4_no_doorbell_receipt_sink: list[Mapping[str, Any]] | None = None,
    diagnostic_global_size: tuple[int, int, int] | None = None,
    diagnostic_dispatch_receipt_sink: list[Mapping[str, Any]] | None = None,
    ) -> FrozenStagedLowLevelDependencies:
  """Build the existing tinygrad AMD mechanisms after queue selection.

  All imports that can instantiate or consult a device are intentionally
  inside this function.  Queue children call it only after fixing ``AMD_AQL``.
  The returned callbacks use the frozen binding's existing PROGRAM, tinygrad's
  runtime cache, the AMD allocator's same-device transfer queue, and the
  harness's native dispatch-evidence wrapper.
  """
  authority.validate()
  if pm4_submit_policy not in PM4_SUBMIT_POLICIES:
    raise ValueError(
      f"pm4_submit_policy must be one of {PM4_SUBMIT_POLICIES!r}")
  if pm4_submit_policy == "snapshot_only":
    if not isinstance(pm4_no_doorbell_receipt_sink, list):
      raise TypeError(
        "snapshot-only PM4 dispatch requires a list receipt sink")
  elif pm4_no_doorbell_receipt_sink is not None:
    raise ValueError("execute PM4 dispatch does not accept a receipt sink")
  if (diagnostic_global_size is None) != \
     (diagnostic_dispatch_receipt_sink is None):
    raise ValueError(
      "diagnostic grid and diagnostic receipt sink must be provided together")
  if diagnostic_global_size is not None:
    diagnostic_global_size = _tuple3(
      diagnostic_global_size, "diagnostic global size")
    if diagnostic_global_size not in DIAGNOSTIC_GLOBAL_SIZE_ALLOWLIST:
      raise ValueError("diagnostic global size is not allowlisted")
    if not isinstance(diagnostic_dispatch_receipt_sink, list):
      raise TypeError("diagnostic dispatch receipt sink must be a list")
    if pm4_submit_policy != "execute" or \
       pm4_no_doorbell_receipt_sink is not None:
      raise ValueError(
        "diagnostic dispatch requires execute PM4 without no-doorbell sink")
  import time
  from tinygrad import Tensor, dtypes
  from tinygrad.device import Device
  from tinygrad.engine.realize import get_runtime, runtime_cache
  from extra.qk.mmq_llama_five_buffer_gpu_harness import (
    PM4_NO_DOORBELL_RECEIPT_SCHEMA as HARNESS_RECEIPT_SCHEMA,
    _dispatch_with_runtime_evidence, _runtime_identity_evidence,
  )
  from extra.qk.q4k_q8_activation_producer import (
    PhysicalDS4Q8ActivationSpec, produce_physical_ds4_q8_1_tensor,
  )
  if HARNESS_RECEIPT_SCHEMA != PM4_NO_DOORBELL_RECEIPT_SCHEMA:
    raise RuntimeError("harness and low-level no-doorbell schemas differ")

  dtype_by_slot = (
    dtypes.float32, dtypes.uint32, dtypes.int8, dtypes.float32,
    dtypes.float32)
  device = Device["AMD"]
  runtime_key: dict[str, Any] = {}

  def observe_binding(binding: Any) -> Mapping[str, Any]:
    artifact = getattr(binding, "artifact", None)
    manifest = getattr(artifact, "manifest", None)
    program = getattr(artifact, "program", None)
    if not isinstance(manifest, Mapping) or program is None:
      raise TypeError("production session requires a frozen exact binding")
    row = _mapping(manifest.get("program"), "frozen program manifest")
    artifacts = _mapping(manifest.get("artifacts"), "frozen artifacts manifest")
    consumer = _mapping(manifest.get("consumer"), "frozen consumer manifest")
    return {
      "program_key": getattr(binding, "program_key", None),
      "binary_sha256": getattr(binding, "binary_sha256", None),
      "source_sha256": getattr(binding, "source_sha256", None),
      "serialized_program_sha256":
        artifacts.get("serialized_program_sha256"),
      "function_name": row.get("function"),
      "compile_target": row.get("compile_target"),
      "global_size": list(row.get("global_size", ())),
      "local_size": list(row.get("local_size", ())),
      "globals": list(row.get("globals", ())),
      "requires_recompile": consumer.get("requires_recompile"),
    }

  def create_runtime(binding: Any) -> Any:
    program = binding.artifact.program
    runtime = get_runtime("AMD", program)
    runtime_key["key"] = (program.key, "AMD")
    runtime_key["runtime"] = runtime
    return runtime

  def observe_runtime(runtime: Any) -> Mapping[str, Any]:
    runtime_lib = getattr(runtime, "lib", None)
    if not isinstance(runtime_lib, (bytes, bytearray, memoryview)):
      raise TypeError("AMD runtime does not retain exact code-object bytes")
    observed_binary_sha256 = hashlib.sha256(bytes(runtime_lib)).hexdigest()
    row = _runtime_identity_evidence(
      device, runtime, observed_binary_sha256)
    return {
      "queue_mode": row["queue_mode"],
      "runtime_class": row["runtime_class"],
      "runtime_name": getattr(runtime, "name", None),
      "runtime_device": getattr(getattr(runtime, "dev", None), "device", None),
      "runtime_object_identity": id(runtime),
      "runtime_device_identity_exact": getattr(runtime, "dev", None) is device,
      "runtime_cache_binding_exact":
        runtime_key.get("runtime") is runtime and
        runtime_cache.get(runtime_key.get("key")) is runtime,
      "program_key": authority.program_key,
      "binary_sha256": row["binary_sha256"],
      "library_va": row["lib_va"],
      "library_nbytes": row["lib_nbytes"],
      "entry_va": row["entry_va"],
    }

  def allocate(slot: FrozenStagedAbiSlot) -> Any:
    return Tensor.empty(
      slot.elements, dtype=dtype_by_slot[slot.slot], device="AMD")

  def realize_many(values: tuple[Any, ...]) -> None:
    if not values:
      raise ValueError("production realization requires at least one value")
    values[0].realize(*values[1:])

  def as_buffer(value: Any) -> Any:
    uop = getattr(value, "uop", None)
    return uop.buffer if uop is not None else value

  def observe_buffer(value: Any) -> Mapping[str, Any]:
    buffer = as_buffer(value)
    if getattr(buffer, "device", None) != "AMD":
      raise ValueError("low-level buffer is not resident on AMD")
    handle = buffer.get_buf("AMD")
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", getattr(buffer, "dtype", None))
    return {
      "va": int(handle.va_addr), "nbytes": int(buffer.nbytes),
      "device": buffer.device, "dtype": str(dtype),
      "shape": list(shape) if shape is not None else None}

  def zero_output(output: Any) -> None:
    buffer = as_buffer(output)
    buffer.copyin(memoryview(bytearray(buffer.nbytes)))

  def produce_q8(
      common_resident_fp16: Any, program_authority: FrozenStagedProgramAuthority,
      ) -> tuple[Any, Any, Any]:
    m, _, k = program_authority.full_shape
    if tuple(common_resident_fp16.shape) != (1, m, k):
      raise ValueError("resident FP16 activation shape differs")
    if common_resident_fp16.dtype != dtypes.float16:
      raise TypeError("common resident activation must be FP16")
    source = common_resident_fp16[0].cast(dtypes.float32).contiguous()
    tile = produce_physical_ds4_q8_1_tensor(
      source, PhysicalDS4Q8ActivationSpec(m, k))
    return tile.values, tile.scales, tile.sums

  def epoch_view(
      source: Any, slot: FrozenStagedAbiSlot, epoch: int,
      ) -> Any:
    return as_buffer(source).view(
      slot.elements, dtype_by_slot[slot.slot], epoch * slot.nbytes)

  def transfer(destination: Any, source: Any, nbytes: int) -> None:
    destination_buffer = as_buffer(destination)
    source_buffer = as_buffer(source)
    if getattr(destination_buffer, "device", None) != "AMD" or \
       getattr(source_buffer, "device", None) != "AMD":
      raise ValueError("fixed-VA transfer requires AMD source and destination")
    destination_handle = destination_buffer.get_buf("AMD")
    source_handle = source_buffer.get_buf("AMD")
    allocator = device.allocator
    if device.hw_copy_queue_t is None or not hasattr(allocator, "_transfer"):
      raise RuntimeError("fixed-VA staging requires AMD same-device SDMA")
    allocator._transfer(
      destination_handle, source_handle, nbytes,
      src_dev=device, dest_dev=device)

  def synchronize() -> None:
    device.synchronize()

  def dispatch(
      runtime: Any, values: tuple[Any, ...],
      program_authority: FrozenStagedProgramAuthority, epoch: int,
      ) -> Mapping[str, Any]:
    buffers = tuple(as_buffer(value) for value in values)
    return _dispatch_production_runtime(
      runtime, buffers, program_authority, epoch,
      runtime_observation=observe_runtime(runtime),
      dispatch_with_runtime_evidence=_dispatch_with_runtime_evidence,
      pm4_submit_policy=pm4_submit_policy,
      pm4_no_doorbell_receipt_sink=pm4_no_doorbell_receipt_sink,
      fixed_five_vas=tuple(
        int(buffer.get_buf("AMD").va_addr) for buffer in buffers),
      effective_global_size=diagnostic_global_size,
      receipt_schema=HARNESS_RECEIPT_SCHEMA)

  return FrozenStagedLowLevelDependencies(
    observe_binding=observe_binding, create_runtime=create_runtime,
    observe_runtime=observe_runtime, allocate=allocate,
    realize_many=realize_many, observe_buffer=observe_buffer,
    zero_output=zero_output, produce_q8=produce_q8,
    epoch_view=epoch_view, transfer=transfer, synchronize=synchronize,
    dispatch=dispatch, clock_ns=time.perf_counter_ns,
    diagnostic_global_size=diagnostic_global_size,
    diagnostic_dispatch_receipt_sink=
      diagnostic_dispatch_receipt_sink).validate()


__all__ = [
  "ABI_NAMES", "ATTESTATION_SCHEMA", "CANDIDATE_TRACE_SCHEMA",
  "DIAGNOSTIC_GLOBAL_SIZE_ALLOWLIST",
  "DIAGNOSTIC_PENDING_OBSERVATION_SCHEMA", "DIAGNOSTIC_RECEIPT_SCHEMA",
  "FrozenStagedAbiSlot", "FrozenStagedLowLevelAttestation",
  "FrozenStagedLowLevelDiagnosticReceipt",
  "FrozenStagedLowLevelDependencies", "FrozenStagedLowLevelInvocation",
  "FrozenStagedLowLevelSession", "FrozenStagedProgramAuthority",
  "INVOCATION_FAILURE_ATTR", "INVOCATION_FAILURE_SCHEMA",
  "PENDING_OBSERVATION_SCHEMA", "PM4_NO_DOORBELL_CHECK_KEYS",
  "PM4_NO_DOORBELL_RECEIPT_SCHEMA", "PM4_SUBMIT_POLICIES", "QUEUE_MODES",
  "production_frozen_staged_low_level_dependencies",
]
