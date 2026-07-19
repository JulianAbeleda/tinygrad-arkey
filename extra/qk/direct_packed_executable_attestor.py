"""Exact executable attestation for the production direct-packed fallback.

The production fallback returns a lazy Tensor.  This module realizes that
Tensor through tinygrad's ordinary linearization/compiler/runtime path while
retaining an exact, pointer-independent manifest of the compiled LINEAR.  A
post-synchronization attestation binds that observed executable to the frozen
``tinygrad.direct_packed.complete_role_fallback.v1`` evidence consumed by C8.

No Device is imported or initialized at module import time.  The primitive
bundle is injectable so all state transitions and failure modes are CPU
testable without an AMD device.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence


QUEUE_MODES = ("PM4", "AQL")
ARTIFACT_SCHEMA = "tinygrad.direct_packed.compiled_linear_executable.v1"
EVIDENCE_SCHEMA = "tinygrad.direct_packed.complete_role_fallback.v1"
OBSERVATION_SCHEMA = "tinygrad.direct_packed.compiled_linear_observation.v1"
QUALIFICATION_SCHEMA = "tinygrad.direct_packed.executable_qualification.v1"
_HEX = frozenset("0123456789abcdef")


def _canonical(value: Any) -> bytes:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(value: Mapping[str, Any]) -> str:
  return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _nonempty(value: Any, label: str) -> str:
  if not isinstance(value, str) or not value:
    raise ValueError(f"{label} must be a non-empty string")
  return value


def _digest(value: Any, label: str) -> str:
  value = _nonempty(value, label)
  if not value.startswith("sha256:") or len(value) != 71 or \
     any(char not in _HEX for char in value[7:]):
    raise ValueError(f"{label} must be a sha256 content identity")
  return value


def _lower_hex_digest(value: Any, label: str) -> str:
  if not isinstance(value, str) or len(value) != 64 or any(char not in _HEX for char in value):
    raise ValueError(f"{label} must be a lowercase hexadecimal SHA-256 digest")
  return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
  if set(value) != expected:
    raise ValueError(
      f"{label} fields differ: expected {sorted(expected)!r}, got {sorted(value)!r}")


def _queue_from_environment() -> str:
  value = os.environ.get("AMD_AQL")
  if value not in ("0", "1"):
    raise ValueError("direct_packed attestation child requires explicit AMD_AQL=0 or AMD_AQL=1")
  return "AQL" if value == "1" else "PM4"


@dataclass(frozen=True)
class DirectPackedAttestationBindings:
  """Immutable non-executable identities joined to one queue observation."""

  queue_mode: str
  workload_identity: str
  input_identity: str
  device_identity: str
  software_identity: str
  comparator_identity: str
  clock_identity: str
  required_program_prefix: str

  def validate(self) -> "DirectPackedAttestationBindings":
    if self.queue_mode not in QUEUE_MODES:
      raise ValueError(f"queue_mode must be one of {QUEUE_MODES!r}")
    _nonempty(self.workload_identity, "workload_identity")
    _digest(self.input_identity, "input_identity")
    _nonempty(self.device_identity, "device_identity")
    _nonempty(self.software_identity, "software_identity")
    _digest(self.comparator_identity, "comparator_identity")
    _nonempty(self.clock_identity, "clock_identity")
    _nonempty(self.required_program_prefix, "required_program_prefix")
    return self


@dataclass(frozen=True)
class DirectPackedAttestationPrimitives:
  """Injectable wrappers around the tinygrad execution primitives."""

  linearize: Callable[[Any], tuple[Any, Mapping[str, int]]]
  compile_linear: Callable[[Any], Any]
  execute_compiled: Callable[[Any, Mapping[str, int]], None]
  runtime_lookup: Callable[[bytes, str], Any | None]
  profile_events: Callable[[], Sequence[Any]]

  def validate(self) -> "DirectPackedAttestationPrimitives":
    if not all(callable(value) for value in (
        self.linearize, self.compile_linear, self.execute_compiled,
        self.runtime_lookup, self.profile_events)):
      raise TypeError("direct_packed attestation primitives must be callable")
    return self


def production_attestation_primitives() -> DirectPackedAttestationPrimitives:
  """Return lazy imports of tinygrad's real compile and execution surfaces."""
  from tinygrad.device import Compiled
  from tinygrad.engine.realize import compile_linear, run_linear, runtime_cache

  def linearize(output: Any) -> tuple[Any, Mapping[str, int]]:
    method = getattr(output, "linear_with_vars", None)
    if not callable(method):
      raise TypeError("direct_packed fallback output has no linear_with_vars()")
    return method()

  return DirectPackedAttestationPrimitives(
    linearize=linearize,
    compile_linear=compile_linear,
    # jit=True means "the LINEAR is already compiled"; it does not replace the
    # execution with a TinyJit graph.
    execute_compiled=lambda linear, var_vals: run_linear(
      linear, dict(var_vals), jit=True),
    runtime_lookup=lambda key, device: runtime_cache.get((key, device)),
    profile_events=lambda: Compiled.profile_events,
  ).validate()


def _json_scalar(value: Any, label: str) -> str | int | float | bool | None:
  if value is None or isinstance(value, (str, bool, int)): return value
  if isinstance(value, float) and math.isfinite(value): return value
  raise ValueError(f"{label} is not a finite JSON scalar")


def _concrete_shape(value: Any, label: str) -> list[int]:
  shape = getattr(value, "shape", None)
  if not isinstance(shape, tuple) or any(
      not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape):
    raise ValueError(f"{label} must have one concrete non-negative shape")
  return list(shape)


def _device(value: Any, label: str) -> str:
  device = getattr(value, "device", None)
  if not isinstance(device, str) or not device:
    raise ValueError(f"{label} must have one concrete device")
  return device


def _argument_contract(value: Any, label: str) -> dict[str, Any]:
  op = getattr(getattr(value, "op", None), "name", None)
  dtype = str(getattr(value, "dtype", ""))
  if not isinstance(op, str) or not op or not dtype:
    raise ValueError(f"{label} lacks an exact operation or dtype")
  return {
    "op": op, "device": _device(value, label),
    "dtype": dtype, "shape": _concrete_shape(value, label),
  }


def _positive_dims(value: Any, label: str) -> list[int]:
  if not isinstance(value, tuple) or not value or any(
      not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in value):
    raise ValueError(f"{label} must be a concrete positive tuple")
  return list(value)


@dataclass(frozen=True)
class _RuntimeExpectation:
  key: bytes
  device: str
  function_name: str
  binary_sha256: str


def _compiled_manifest(
    compiled: Any, var_vals: Mapping[str, int], *,
    queue_mode: str, required_program_prefix: str,
    ) -> tuple[dict[str, Any], tuple[_RuntimeExpectation, ...], tuple[bytes, ...]]:
  from tinygrad.engine.realize import get_call_arg_uops
  from tinygrad.uop.ops import Ops

  if getattr(compiled, "op", None) is not Ops.LINEAR:
    raise ValueError("direct_packed compiler did not return one LINEAR")
  if not isinstance(var_vals, Mapping) or any(
      not isinstance(key, str) or not isinstance(value, int) or isinstance(value, bool)
      for key, value in var_vals.items()):
    raise ValueError("direct_packed LINEAR variables must be concrete integer bindings")

  calls, programs, expectations, binaries = [], [], [], []
  direct_ordinals = []
  allowed = {Ops.PROGRAM, Ops.COPY, Ops.SLICE}
  for ordinal, call in enumerate(compiled.src):
    if getattr(call, "op", None) is not Ops.CALL or not getattr(call, "src", ()):
      raise ValueError("direct_packed LINEAR contains a non-CALL row")
    ast = call.src[0]
    if ast.op not in allowed:
      raise ValueError(f"direct_packed LINEAR contains unsupported executable op {ast.op}")
    args = tuple(get_call_arg_uops(call))
    arg_rows = [
      _argument_contract(value, f"calls[{ordinal}].arguments[{index}]")
      for index, value in enumerate(args)
    ]
    row: dict[str, Any] = {
      "ordinal": ordinal, "operation": ast.op.name, "arguments": arg_rows,
    }
    if ast.op is Ops.PROGRAM:
      if len(ast.src) < 5:
        raise ValueError("compiled PROGRAM omits target/source/binary payloads")
      name = getattr(ast.arg, "function_name", None)
      program_name = getattr(ast.arg, "name", None)
      target = ast.src[1].arg
      source, binary = ast.src[3].arg, ast.src[4].arg
      if not isinstance(name, str) or not name or not isinstance(program_name, str) or not program_name or \
         not isinstance(target, str) or not target or not isinstance(source, str) or \
         not isinstance(binary, bytes) or not binary:
        raise ValueError("compiled PROGRAM identity payload is incomplete")
      if not args:
        raise ValueError("compiled PROGRAM has no concrete execution device")
      execution_device = _device(args[0], f"calls[{ordinal}].arguments[0]")
      global_size, local_size = ast.arg.launch_dims(dict(var_vals))
      values = [
        _json_scalar(value, f"calls[{ordinal}].values[{index}]")
        for index, value in enumerate(ast.arg.vals(dict(var_vals)))
      ]
      binary_sha256 = hashlib.sha256(binary).hexdigest()
      program = {
        "call_ordinal": ordinal, "program_key": ast.key.hex(),
        "program_name": program_name, "function_name": name,
        "target": target, "execution_device": execution_device,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "binary_sha256": binary_sha256, "binary_nbytes": len(binary),
        "global_size": _positive_dims(global_size, f"calls[{ordinal}].global_size"),
        "local_size": None if local_size is None else
          _positive_dims(local_size, f"calls[{ordinal}].local_size"),
        "globals": list(ast.arg.globals), "outs": list(ast.arg.outs),
        "ins": list(ast.arg.ins), "values": values,
      }
      if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
             for field in ("globals", "outs", "ins") for value in program[field]):
        raise ValueError("compiled PROGRAM ABI indices must be non-negative integers")
      program_index = len(programs)
      row["program_index"] = program_index
      programs.append(program)
      binaries.append(binary)
      expectations.append(_RuntimeExpectation(
        ast.key, execution_device, name, binary_sha256))
      if name.startswith(required_program_prefix) or program_name.startswith(required_program_prefix):
        direct_ordinals.append(ordinal)
    elif ast.op is Ops.COPY:
      if len(arg_rows) != 2 or arg_rows[0]["shape"] != arg_rows[1]["shape"] or \
         arg_rows[0]["dtype"] != arg_rows[1]["dtype"]:
        raise ValueError("compiled COPY does not preserve concrete shape and dtype")
      itemsize = getattr(getattr(args[0], "dtype", None), "itemsize", None)
      if not isinstance(itemsize, int) or itemsize <= 0:
        raise ValueError("compiled COPY dtype has no concrete itemsize")
      elements = math.prod(arg_rows[0]["shape"])
      row["nbytes"] = elements * itemsize
    else:
      offset = getattr(ast.src[1], "arg", None) if len(ast.src) > 1 else None
      if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("compiled SLICE has no concrete non-negative offset")
      row["offset_elements"] = offset
    calls.append(row)

  if not programs:
    raise ValueError("direct_packed LINEAR contains no compiled PROGRAM")
  if not direct_ordinals:
    raise ValueError(
      "direct_packed LINEAR contains no PROGRAM matching the required production prefix")

  program_metadata = [{
    key: value for key, value in program.items() if key not in ("binary_sha256", "binary_nbytes")
  } for program in programs]
  digest = hashlib.sha256(b"tinygrad.direct_packed.ordered_code_objects.v1\0")
  for metadata, binary in zip(program_metadata, binaries):
    encoded = _canonical(metadata)
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
    digest.update(len(binary).to_bytes(8, "little"))
    digest.update(binary)
  aggregate_binary = digest.hexdigest()
  executable_payload = {
    "artifact_schema": ARTIFACT_SCHEMA, "queue_mode": queue_mode,
    "variables": dict(sorted(var_vals.items())), "calls": calls,
    "programs": programs, "direct_packed_program_ordinals": direct_ordinals,
    "aggregate_binary_sha256": aggregate_binary,
  }
  manifest = {
    **executable_payload,
    "executable_identity": _identity(executable_payload),
  }
  return manifest, tuple(expectations), tuple(binaries)


def _profile_code_object_crosscheck(
    events: Sequence[Any], expectations: Sequence[_RuntimeExpectation],
    ) -> int:
  """Validate any PROFILE code objects emitted during this exact realization.

  A cached runtime emits no new ProfileProgramEvent, so zero matching events is
  honest and the runtime-cache join remains authoritative.  If PROFILE exposes
  a code object, however, it must byte-match the compiled PROGRAM.
  """
  expected = {
    (row.device, row.function_name): row.binary_sha256 for row in expectations
  }
  matched = 0
  for event in events:
    key = (getattr(event, "device", None), getattr(event, "name", None))
    if key not in expected: continue
    lib = getattr(event, "lib", None)
    if not isinstance(lib, bytes) or hashlib.sha256(lib).hexdigest() != expected[key]:
      raise ValueError("PROFILE code object differs from the compiled direct_packed PROGRAM")
    matched += 1
  return matched


class DirectPackedLinearExecutionCapture:
  """Single-invocation state machine around one exact compiled LINEAR."""

  def __init__(
      self, *, bindings_by_queue: Mapping[str, DirectPackedAttestationBindings],
      primitives: DirectPackedAttestationPrimitives | None = None,
      ) -> None:
    if not isinstance(bindings_by_queue, Mapping) or set(bindings_by_queue) != set(QUEUE_MODES):
      raise ValueError(f"bindings_by_queue must contain exactly {QUEUE_MODES!r}")
    self.bindings = {
      queue: bindings_by_queue[queue].validate() for queue in QUEUE_MODES
    }
    if any(self.bindings[queue].queue_mode != queue for queue in QUEUE_MODES):
      raise ValueError("direct_packed attestation binding queue keys differ")
    self.primitives = (
      production_attestation_primitives() if primitives is None else primitives.validate())
    self._pending: dict[str, Any] | None = None

  def realize_output(self, output: Any) -> None:
    if self._pending is not None:
      raise RuntimeError("prior direct_packed execution has not been attested post-sync")
    queue_mode = _queue_from_environment()
    bindings = self.bindings[queue_mode]
    linear, raw_vars = self.primitives.linearize(output)
    var_vals = dict(raw_vars)
    compiled = self.primitives.compile_linear(linear)
    profile_start = len(self.primitives.profile_events())
    self._pending = {
      "output_id": id(output), "queue_mode": queue_mode,
      "compiled": compiled, "var_vals": var_vals,
      "required_program_prefix": bindings.required_program_prefix,
      "profile_start": profile_start,
    }
    try:
      self.primitives.execute_compiled(compiled, var_vals)
    except BaseException:
      self._pending = None
      raise

  def observation_post_sync(self, output: Any, queue_mode: str) -> dict[str, Any]:
    pending, self._pending = self._pending, None
    if not isinstance(pending, dict):
      raise RuntimeError("no direct_packed execution awaits post-sync attestation")
    if id(output) != pending["output_id"] or queue_mode != pending["queue_mode"]:
      raise ValueError("post-sync direct_packed output or queue differs from the executed LINEAR")
    manifest, expectations, _ = _compiled_manifest(
      pending["compiled"], pending["var_vals"], queue_mode=queue_mode,
      required_program_prefix=pending["required_program_prefix"])
    runtime_rows = []
    for expectation in expectations:
      runtime = self.primitives.runtime_lookup(expectation.key, expectation.device)
      if runtime is None:
        raise ValueError("executed direct_packed PROGRAM is absent from exact runtime cache key")
      runtime_name = getattr(runtime, "name", None)
      if runtime_name != expectation.function_name:
        raise ValueError("direct_packed runtime name differs from compiled PROGRAM")
      runtime_rows.append({
        "device": expectation.device,
        "function_name": expectation.function_name,
        "program_key": expectation.key.hex(),
        "binary_sha256": expectation.binary_sha256,
      })
    events = tuple(self.primitives.profile_events())
    profile_count = _profile_code_object_crosscheck(
      events[pending["profile_start"]:], expectations)
    payload = {
      "schema": OBSERVATION_SCHEMA, "status": "PASS",
      "queue_mode": queue_mode, "manifest": manifest,
      "runtime_programs": runtime_rows,
      "runtime_cache_join_verified": True,
      "profile_code_objects_verified": profile_count,
      "post_sync_attestation": True,
    }
    return {**payload, "observation_identity": _identity(payload)}


def build_direct_packed_fallback_evidence(
    observation: Mapping[str, Any], bindings: DirectPackedAttestationBindings,
    ) -> dict[str, Any]:
  """Bind one observed executable to the non-executable C8 identities."""
  bindings.validate()
  if not isinstance(observation, Mapping):
    raise ValueError("direct_packed executable observation must be a mapping")
  _exact_keys(observation, {
    "schema", "status", "queue_mode", "manifest", "runtime_programs",
    "runtime_cache_join_verified", "profile_code_objects_verified",
    "post_sync_attestation", "observation_identity",
  }, "direct_packed executable observation")
  payload = {key: value for key, value in observation.items() if key != "observation_identity"}
  if observation["observation_identity"] != _identity(payload) or \
     observation["schema"] != OBSERVATION_SCHEMA or observation["status"] != "PASS" or \
     observation["queue_mode"] != bindings.queue_mode or \
     observation["runtime_cache_join_verified"] is not True or \
     observation["post_sync_attestation"] is not True:
    raise ValueError("direct_packed executable observation is not an exact passing post-sync capture")
  manifest = observation["manifest"]
  if not isinstance(manifest, Mapping):
    raise ValueError("direct_packed executable manifest must be a mapping")
  _exact_keys(manifest, {
    "artifact_schema", "queue_mode", "variables", "calls", "programs",
    "direct_packed_program_ordinals", "aggregate_binary_sha256",
    "executable_identity",
  }, "direct_packed executable manifest")
  manifest_payload = {
    key: value for key, value in manifest.items() if key != "executable_identity"
  }
  if manifest.get("artifact_schema") != ARTIFACT_SCHEMA or \
     manifest.get("queue_mode") != bindings.queue_mode or \
     manifest.get("executable_identity") != _identity(manifest_payload):
    raise ValueError("direct_packed executable manifest identity is incomplete")
  variables, calls, programs = (
    manifest["variables"], manifest["calls"], manifest["programs"])
  if not isinstance(variables, Mapping) or any(
      not isinstance(key, str) or not isinstance(value, int) or isinstance(value, bool)
      for key, value in variables.items()):
    raise ValueError("direct_packed executable manifest variables are malformed")
  if not isinstance(calls, list) or not isinstance(programs, list) or not programs:
    raise ValueError("direct_packed executable manifest call/program rows are incomplete")
  program_keys = {
    "call_ordinal", "program_key", "program_name", "function_name", "target",
    "execution_device", "source_sha256", "binary_sha256", "binary_nbytes",
    "global_size", "local_size", "globals", "outs", "ins", "values",
  }
  for index, program in enumerate(programs):
    if not isinstance(program, Mapping):
      raise ValueError(f"direct_packed programs[{index}] must be a mapping")
    _exact_keys(program, program_keys, f"direct_packed programs[{index}]")
    if not isinstance(program["call_ordinal"], int) or \
       program["call_ordinal"] < 0 or program["call_ordinal"] >= len(calls):
      raise ValueError(f"direct_packed programs[{index}] call ordinal is invalid")
    _lower_hex_digest(program["program_key"], f"programs[{index}].program_key")
    _lower_hex_digest(program["source_sha256"], f"programs[{index}].source_sha256")
    _lower_hex_digest(program["binary_sha256"], f"programs[{index}].binary_sha256")
    if not isinstance(program["binary_nbytes"], int) or program["binary_nbytes"] <= 0:
      raise ValueError(f"direct_packed programs[{index}] binary size is invalid")
    for field in ("program_name", "function_name", "target", "execution_device"):
      _nonempty(program[field], f"programs[{index}].{field}")
  direct_ordinals = manifest["direct_packed_program_ordinals"]
  expected_direct = [
    program["call_ordinal"] for program in programs
    if program["program_name"].startswith(bindings.required_program_prefix) or
       program["function_name"].startswith(bindings.required_program_prefix)
  ]
  if direct_ordinals != expected_direct or not expected_direct:
    raise ValueError("direct_packed executable manifest production PROGRAM set differs")
  runtime_rows = observation["runtime_programs"]
  expected_runtime = [{
    "device": program["execution_device"],
    "function_name": program["function_name"],
    "program_key": program["program_key"],
    "binary_sha256": program["binary_sha256"],
  } for program in programs]
  if runtime_rows != expected_runtime:
    raise ValueError("direct_packed runtime rows differ from the exact compiled PROGRAM order")
  profile_count = observation["profile_code_objects_verified"]
  if not isinstance(profile_count, int) or isinstance(profile_count, bool) or profile_count < 0:
    raise ValueError("direct_packed PROFILE verification count must be non-negative")
  binary_sha256 = _lower_hex_digest(
    manifest.get("aggregate_binary_sha256"), "aggregate_binary_sha256")
  executable_identity = _digest(
    manifest.get("executable_identity"), "executable_identity")
  artifact_payload = {
    "artifact_schema": ARTIFACT_SCHEMA, "binary_sha256": binary_sha256,
    "executable_identity": executable_identity,
    "comparator_identity": bindings.comparator_identity,
    "queue_mode": bindings.queue_mode,
    "workload_identity": bindings.workload_identity,
  }
  evidence = {
    "schema": EVIDENCE_SCHEMA, "status": "PASS",
    "route_id": "direct_packed", "queue_mode": bindings.queue_mode,
    "artifact_schema": ARTIFACT_SCHEMA,
    "artifact_identity": _identity(artifact_payload),
    "binary_sha256": binary_sha256,
    "executable_identity": executable_identity,
    "comparator_identity": bindings.comparator_identity,
    "workload_identity": bindings.workload_identity,
    "input_identity": bindings.input_identity,
    "device_identity": bindings.device_identity,
    "software_identity": bindings.software_identity,
    "clock_identity": bindings.clock_identity,
  }
  return {**evidence, "evidence_identity": _identity(evidence)}


def validate_direct_packed_fallback_evidence(
    value: Mapping[str, Any], bindings: DirectPackedAttestationBindings,
    ) -> dict[str, Any]:
  """Validate the exact fallback schema without importing the C8 builder."""
  bindings.validate()
  if not isinstance(value, Mapping):
    raise ValueError("frozen direct_packed fallback evidence must be a mapping")
  expected = {
    "schema", "status", "route_id", "queue_mode", "artifact_schema",
    "artifact_identity", "binary_sha256", "executable_identity",
    "comparator_identity", "workload_identity", "input_identity",
    "device_identity", "software_identity", "clock_identity",
    "evidence_identity",
  }
  _exact_keys(value, expected, "frozen direct_packed fallback evidence")
  payload = {key: item for key, item in value.items() if key != "evidence_identity"}
  if value["evidence_identity"] != _identity(payload):
    raise ValueError("frozen direct_packed fallback evidence content identity differs")
  checks = {
    "schema": value["schema"] == EVIDENCE_SCHEMA,
    "status": value["status"] == "PASS",
    "route_id": value["route_id"] == "direct_packed",
    "queue_mode": value["queue_mode"] == bindings.queue_mode,
    "artifact_schema": value["artifact_schema"] == ARTIFACT_SCHEMA,
    "comparator_identity": value["comparator_identity"] == bindings.comparator_identity,
    "workload_identity": value["workload_identity"] == bindings.workload_identity,
    "input_identity": value["input_identity"] == bindings.input_identity,
    "device_identity": value["device_identity"] == bindings.device_identity,
    "software_identity": value["software_identity"] == bindings.software_identity,
    "clock_identity": value["clock_identity"] == bindings.clock_identity,
  }
  if not all(checks.values()):
    raise ValueError(
      f"frozen direct_packed fallback binding differs: "
      f"{sorted(key for key, passed in checks.items() if not passed)!r}")
  binary = _lower_hex_digest(value["binary_sha256"], "binary_sha256")
  executable = _digest(value["executable_identity"], "executable_identity")
  _digest(value["comparator_identity"], "comparator_identity")
  artifact_payload = {
    "artifact_schema": ARTIFACT_SCHEMA, "binary_sha256": binary,
    "executable_identity": executable,
    "comparator_identity": value["comparator_identity"],
    "queue_mode": bindings.queue_mode,
    "workload_identity": bindings.workload_identity,
  }
  if value["artifact_identity"] != _identity(artifact_payload):
    raise ValueError("frozen direct_packed artifact identity differs from exact executable content")
  return dict(value)


def build_direct_packed_qualification_artifact(
    observation: Mapping[str, Any], bindings: DirectPackedAttestationBindings,
    ) -> dict[str, Any]:
  """Build the untimed bootstrap artifact consumed by later timed sessions."""
  evidence = build_direct_packed_fallback_evidence(observation, bindings)
  payload = {
    "schema": QUALIFICATION_SCHEMA, "status": "PASS",
    "queue_mode": bindings.queue_mode,
    "observation": dict(observation), "fallback_evidence": evidence,
    "qualification_only": True, "timing_samples_collected": False,
    "production_dispatch_changed": False,
  }
  return {**payload, "qualification_identity": _identity(payload)}


def validate_direct_packed_qualification_artifact(
    value: Mapping[str, Any], bindings: DirectPackedAttestationBindings,
    ) -> dict[str, Any]:
  bindings.validate()
  if not isinstance(value, Mapping):
    raise ValueError("direct_packed qualification artifact must be a mapping")
  _exact_keys(value, {
    "schema", "status", "queue_mode", "observation", "fallback_evidence",
    "qualification_only", "timing_samples_collected",
    "production_dispatch_changed", "qualification_identity",
  }, "direct_packed qualification artifact")
  payload = {key: item for key, item in value.items() if key != "qualification_identity"}
  if value["qualification_identity"] != _identity(payload) or \
     value["schema"] != QUALIFICATION_SCHEMA or value["status"] != "PASS" or \
     value["queue_mode"] != bindings.queue_mode or \
     value["qualification_only"] is not True or \
     value["timing_samples_collected"] is not False or \
     value["production_dispatch_changed"] is not False:
    raise ValueError("direct_packed qualification artifact identity or scope differs")
  rebuilt = build_direct_packed_fallback_evidence(value["observation"], bindings)
  evidence = validate_direct_packed_fallback_evidence(
    value["fallback_evidence"], bindings)
  if rebuilt != evidence:
    raise ValueError("direct_packed qualification observation differs from frozen evidence")
  return dict(value)


def persist_direct_packed_qualification(
    path: str | Path, artifact: Mapping[str, Any],
    bindings: DirectPackedAttestationBindings,
    ) -> Path:
  """Atomically publish one immutable qualification artifact.

  ``os.link`` supplies no-replace publication in the destination directory:
  an existing frozen artifact is never silently replaced.
  """
  validated = validate_direct_packed_qualification_artifact(artifact, bindings)
  output = Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary = tempfile.mkstemp(
    prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
  temporary_path = Path(temporary)
  try:
    with os.fdopen(fd, "w") as handle:
      json.dump(validated, handle, sort_keys=True, indent=2, allow_nan=False)
      handle.write("\n")
      handle.flush()
      os.fsync(handle.fileno())
    try: os.link(temporary_path, output)
    except FileExistsError as exc:
      raise FileExistsError(
        f"refusing to replace frozen direct_packed qualification {output}") from exc
    try:
      directory_fd = os.open(output.parent, os.O_RDONLY)
      try: os.fsync(directory_fd)
      finally: os.close(directory_fd)
    except OSError:
      # The content file is already fsynced and atomically linked.  Some
      # filesystems do not permit directory fsync.
      pass
  finally:
    try: temporary_path.unlink()
    except FileNotFoundError: pass
  return output


def load_direct_packed_qualification(
    path: str | Path, bindings: DirectPackedAttestationBindings,
    ) -> dict[str, Any]:
  value = json.loads(Path(path).read_text())
  if not isinstance(value, Mapping):
    raise ValueError("direct_packed qualification file must contain one JSON object")
  return validate_direct_packed_qualification_artifact(value, bindings)


def load_frozen_direct_packed_evidence_by_queue(
    paths_by_queue: Mapping[str, str | Path],
    bindings_by_queue: Mapping[str, DirectPackedAttestationBindings],
    ) -> dict[str, dict[str, Any]]:
  """Load both immutable queue qualifications before timing begins."""
  if not isinstance(paths_by_queue, Mapping) or set(paths_by_queue) != set(QUEUE_MODES):
    raise ValueError(f"qualification paths must contain exactly {QUEUE_MODES!r}")
  if not isinstance(bindings_by_queue, Mapping) or set(bindings_by_queue) != set(QUEUE_MODES):
    raise ValueError(f"bindings_by_queue must contain exactly {QUEUE_MODES!r}")
  resolved = {queue: Path(paths_by_queue[queue]).resolve() for queue in QUEUE_MODES}
  if resolved["PM4"] == resolved["AQL"]:
    raise ValueError("PM4 and AQL qualifications must be distinct frozen files")
  artifacts = {
    queue: load_direct_packed_qualification(
      resolved[queue], bindings_by_queue[queue])
    for queue in QUEUE_MODES
  }
  return {
    queue: dict(artifacts[queue]["fallback_evidence"]) for queue in QUEUE_MODES
  }


def qualify_and_freeze_production_direct_packed(
    *, linear: Any, input_tensor: Any, route_spec: Any, queue_mode: str,
    bindings_by_queue: Mapping[str, DirectPackedAttestationBindings],
    output: str | Path,
    synchronize: Callable[[], None] | None = None,
    executor: Callable[[Any, Any, Any], Any] | None = None,
    primitives: DirectPackedAttestationPrimitives | None = None,
    ) -> dict[str, Any]:
  """Run one exact untimed production fallback and freeze its evidence.

  This bootstrap deliberately has no clock or timing result.  It must complete
  before a C8 timing runner loads the PM4/AQL qualification pair.
  """
  if queue_mode not in QUEUE_MODES:
    raise ValueError(f"queue_mode must be one of {QUEUE_MODES!r}")
  if _queue_from_environment() != queue_mode:
    raise ValueError("qualification child AMD_AQL mode differs from queue_mode")
  if Path(output).exists():
    raise FileExistsError(
      f"refusing to replace frozen direct_packed qualification {Path(output)}")
  if executor is None:
    from tinygrad.llm.prefill_routes import _run_direct_packed_baseline
    executor = _run_direct_packed_baseline
  if synchronize is None:
    from tinygrad.device import Device
    synchronize = Device["AMD"].synchronize
  if not callable(executor) or not callable(synchronize):
    raise TypeError("qualification executor and synchronize must be callable")
  capture = DirectPackedLinearExecutionCapture(
    bindings_by_queue=bindings_by_queue, primitives=primitives)
  synchronize()
  lazy_output = executor(linear, input_tensor, route_spec)
  if lazy_output is None:
    raise RuntimeError("production direct_packed qualification returned no output")
  capture.realize_output(lazy_output)
  synchronize()
  observation = capture.observation_post_sync(lazy_output, queue_mode)
  artifact = build_direct_packed_qualification_artifact(
    observation, bindings_by_queue[queue_mode])
  persist_direct_packed_qualification(
    output, artifact, bindings_by_queue[queue_mode])
  return artifact


class FrozenDirectPackedExecutableAttestor:
  """Compare every synchronized execution with pre-existing frozen evidence."""

  def __init__(
      self, *, expected_evidence_by_queue: Mapping[str, Mapping[str, Any]],
      bindings_by_queue: Mapping[str, DirectPackedAttestationBindings],
      primitives: DirectPackedAttestationPrimitives | None = None,
      ) -> None:
    if not isinstance(expected_evidence_by_queue, Mapping) or \
       set(expected_evidence_by_queue) != set(QUEUE_MODES):
      raise ValueError(f"expected_evidence_by_queue must contain exactly {QUEUE_MODES!r}")
    self.capture = DirectPackedLinearExecutionCapture(
      bindings_by_queue=bindings_by_queue, primitives=primitives)
    self.expected = {
      queue: validate_direct_packed_fallback_evidence(
        expected_evidence_by_queue[queue], self.capture.bindings[queue])
      for queue in QUEUE_MODES
    }
    self.last_observation: dict[str, Any] | None = None

  def realize_output(self, output: Any) -> None:
    self.capture.realize_output(output)

  def attest_post_sync(self, output: Any, queue_mode: str) -> Mapping[str, Any]:
    observation = self.capture.observation_post_sync(output, queue_mode)
    self.last_observation = observation
    observed = build_direct_packed_fallback_evidence(
      observation, self.capture.bindings[queue_mode])
    if observed != self.expected[queue_mode]:
      raise ValueError(
        f"{queue_mode} observed direct_packed executable differs from frozen fallback evidence")
    return observed


def make_production_direct_packed_attested_runner(
    *, linear: Any, input_tensor: Any, route_spec: Any,
    qualification_paths_by_queue: Mapping[str, str | Path],
    bindings_by_queue: Mapping[str, DirectPackedAttestationBindings],
    synchronize: Callable[[], None] | None = None,
    executor: Callable[[Any, Any, Any], Any] | None = None,
    clock_ns: Callable[[], int] | None = None,
    primitives: DirectPackedAttestationPrimitives | None = None,
    ):
  """Compose the attestor with C8's production direct-packed timing seam."""
  from extra.qk.mmq_frozen_staged_c8_timing import \
    make_direct_packed_fallback_runner
  expected_evidence_by_queue = load_frozen_direct_packed_evidence_by_queue(
    qualification_paths_by_queue, bindings_by_queue)
  attestor = FrozenDirectPackedExecutableAttestor(
    expected_evidence_by_queue=expected_evidence_by_queue,
    bindings_by_queue=bindings_by_queue, primitives=primitives)
  kwargs: dict[str, Any] = {
    "linear": linear, "input_tensor": input_tensor, "route_spec": route_spec,
    "fallback_evidence_by_queue": expected_evidence_by_queue,
    "realize_output": attestor.realize_output,
    "execution_attestor": attestor.attest_post_sync,
  }
  if synchronize is not None: kwargs["synchronize"] = synchronize
  if executor is not None: kwargs["executor"] = executor
  if clock_ns is not None: kwargs["clock_ns"] = clock_ns
  return make_direct_packed_fallback_runner(**kwargs), attestor


__all__ = [
  "ARTIFACT_SCHEMA", "EVIDENCE_SCHEMA", "OBSERVATION_SCHEMA",
  "QUALIFICATION_SCHEMA",
  "DirectPackedAttestationBindings", "DirectPackedAttestationPrimitives",
  "DirectPackedLinearExecutionCapture", "FrozenDirectPackedExecutableAttestor",
  "build_direct_packed_fallback_evidence",
  "build_direct_packed_qualification_artifact",
  "load_direct_packed_qualification",
  "load_frozen_direct_packed_evidence_by_queue",
  "make_production_direct_packed_attested_runner",
  "persist_direct_packed_qualification",
  "production_attestation_primitives",
  "qualify_and_freeze_production_direct_packed",
  "validate_direct_packed_fallback_evidence",
  "validate_direct_packed_qualification_artifact",
]
