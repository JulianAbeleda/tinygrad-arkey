"""Stage-specific guarded correctness for exact ``ffn_gate_up``.

The safety unit is one fresh child and one operation.  In particular, PM4
prefix-1 can run with only its exact persisted PM4 C4 canary.  Prefix-3 and
full-role candidate runs require the preceding persisted candidate artifact;
direct correctness and every mixed-route transition use distinct children.
The exhaustive view is a CPU-only composition over already guarded artifacts.

This module reuses tinygrad's AMD runtime, frozen staged session, production
direct-packed route, executable capture, reference implementation, health
probe, fault collector, and process isolation.  It introduces no launcher.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np


SCHEMA = "tinygrad.mmq_q4k_q8_1.ffn_gate_up_guarded_correctness_stage.v2"
CANDIDATE_SCHEMA = f"{SCHEMA}.candidate_prefix"
DIRECT_SCHEMA = f"{SCHEMA}.direct_full_role"
TRANSITION_SCHEMA = f"{SCHEMA}.transition"
PM4_NO_DOORBELL_SCHEMA = f"{SCHEMA}.pm4_no_doorbell"
FFN_REDUCED_GRID_SCHEMA = f"{SCHEMA}.ffn_reduced_grid"
ENVELOPE_SCHEMA = f"{SCHEMA}.envelope"
COMPOSITION_SCHEMA = f"{SCHEMA}.composition"
JOINT_C7_SCHEMA = f"{SCHEMA}.joint_c7"
PRODUCER_SCHEMA = f"{SCHEMA}.q8_producer"
LOW_LEVEL_ATTESTATION_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.frozen_staged_low_level_attestation.v1"
LOW_LEVEL_INVOCATION_FAILURE_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.frozen_staged_low_level_failure.v1"
LOW_LEVEL_INVOCATION_FAILURE_ATTR = "frozen_staged_low_level_failure"
PM4_NO_DOORBELL_RECEIPT_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.pm4_no_doorbell_receipt.v1"
LOW_LEVEL_DIAGNOSTIC_RECEIPT_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.frozen_staged_low_level_diagnostic_receipt.v1"
PM4_PRE_SUBMIT_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.pm4_pre_submit_snapshot.v2"
PM4_PRE_SUBMIT_CAPTURE_POINT = \
  "AMDComputeQueue._submit_after_complete_command_construction_" \
  "before_ring_copy_and_doorbell"
PM4_PRE_SUBMIT_CHECK_KEYS = {
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
}
PM4_ARGUMENT_NBYTES = (35651584, 2506752, 131072, 16384, 16384)
PM4_RUNTIME_CLASS = "tinygrad.runtime.ops_amd.AMDProgram"
PM4_RUNTIME_NAME = "mmq_llama_five_buffer_full_grid_accumulate"
QUEUE_MODES = ("PM4", "AQL")
PREFIXES = (1, 3, 20)
FFN_REDUCED_GLOBAL_SIZE_ALLOWLIST = (
  (1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 4, 1), (8, 4, 1),
  (32, 4, 1), (40, 4, 1), (41, 4, 1), (136, 1, 1),
)
FFN_FROZEN_GLOBAL_SIZE = (136, 4, 1)
FFN_FROZEN_LOCAL_SIZE = (256, 1, 1)
OUTPUT_SHAPE = (512, 17408)
OUTPUT_ELEMENTS = OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1]
TRANSITION_SEQUENCES = {
  "candidate_candidate": (("candidate", 20), ("candidate", 20)),
  "direct_direct": (("direct_packed", 20), ("direct_packed", 20)),
  "direct_candidate_prefix1": (
    ("direct_packed", 20), ("candidate", 1)),
  "direct_candidate_full_role": (
    ("direct_packed", 20), ("candidate", 20)),
  "candidate_direct_candidate": (
    ("candidate", 20), ("direct_packed", 20), ("candidate", 20)),
}
_HEX = frozenset("0123456789abcdef")
CANDIDATE_EXECUTABLE_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.staged_candidate_executable.v1"
_COMPARISON_KEYS = {
  "status", "rtol", "atol", "got_shape", "reference_shape", "got_size",
  "reference_size", "mismatch_count", "first_mismatch_index",
  "first_mismatch_got", "first_mismatch_reference", "joint_finite",
  "max_abs_error", "mean_abs_error", "nan_got", "nan_reference",
  "inf_got", "inf_reference",
}
_LOW_LEVEL_ATTESTATION_KEYS = {
  "schema", "status", "queue_mode", "family_identity",
  "candidate_executable_identity", "input_identity", "program_key",
  "binary_sha256", "runtime_class", "runtime_name", "runtime_device",
  "runtime_object_identity", "runtime_device_identity_exact",
  "runtime_cache_binding_exact", "library_va", "library_nbytes", "entry_va",
  "fixed_five_vas", "launch_count", "observation_identity",
}


def _canonical(value: Any) -> bytes:
  return json.dumps(
    value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(value: Any) -> str:
  return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
  if not isinstance(value, Mapping):
    raise ValueError(f"{label} must be a mapping")
  return value


def _content_identity(value: Any, label: str) -> str:
  if not isinstance(value, str) or not value.startswith("sha256:") or \
     len(value) != 71 or any(char not in _HEX for char in value[7:]):
    raise ValueError(f"{label} must be a sha256 content identity")
  return value


def _hex_digest(value: Any, label: str) -> str:
  if not isinstance(value, str) or len(value) != 64 or \
     any(char not in _HEX for char in value):
    raise ValueError(f"{label} must be a lowercase SHA-256 digest")
  return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
  if set(value) != expected:
    raise ValueError(f"{label} fields differ")


def _queue(value: Any) -> str:
  if value not in QUEUE_MODES:
    raise ValueError(f"queue_mode must be one of {QUEUE_MODES!r}")
  return value


def _prefix(value: Any) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or \
     value not in PREFIXES:
    raise ValueError(f"prefix_epochs must be one of {PREFIXES!r}")
  return value


def _ffn_reduced_global_size(value: Any) -> tuple[int, int, int]:
  if not isinstance(value, (tuple, list)) or len(value) != 3 or any(
      not isinstance(item, int) or isinstance(item, bool) for item in value):
    raise ValueError(
      "FFN reduced diagnostic global size requires three integers")
  grid = tuple(value)
  if grid not in FFN_REDUCED_GLOBAL_SIZE_ALLOWLIST:
    raise ValueError(
      "FFN reduced diagnostic global size must be one of "
      f"{FFN_REDUCED_GLOBAL_SIZE_ALLOWLIST!r}")
  return grid


def _positive_seconds(value: Any) -> float:
  if not isinstance(value, (int, float)) or isinstance(value, bool) or \
     not math.isfinite(value) or value <= 0:
    raise ValueError("timeout_seconds must be finite and positive")
  return float(value)


def _array_sha256(value: Any) -> str:
  return hashlib.sha256(
    np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def _identity_valid(value: Any) -> bool:
  try:
    row = _mapping(value, "content-addressed evidence")
    return row.get("evidence_identity") == _identity({
      key: item for key, item in row.items() if key != "evidence_identity"})
  except BaseException:
    return False


def ffn_gate_up_candidate_executable_identity(family: Any) -> str:
  binding = getattr(family, "binding", None)
  if getattr(getattr(binding, "role_spec", None), "role", None) != \
       "ffn_gate_up":
    raise ValueError("candidate executable identity requires ffn_gate_up family")
  return _candidate_executable_identity_from_parts(
    _content_identity(
      getattr(family, "family_identity", None), "family identity"),
    getattr(binding, "program_key", None),
    getattr(binding, "binary_sha256", None))


def _candidate_executable_identity_from_parts(
    family_identity: Any, program_key: Any, binary_sha256: Any,
    ) -> str:
  return _identity({
    "schema": CANDIDATE_EXECUTABLE_SCHEMA,
    "family_identity": _content_identity(family_identity, "family identity"),
    "program_key": _hex_digest(program_key, "program key"),
    "binary_sha256": _hex_digest(binary_sha256, "binary SHA-256"),
  })


def _validate_numeric_comparison(value: Any, label: str) -> dict[str, Any]:
  row = dict(_mapping(value, label))
  _exact_keys(row, _COMPARISON_KEYS, label)
  for field in ("rtol", "atol"):
    number = row[field]
    if not isinstance(number, (int, float)) or isinstance(number, bool) or \
       not math.isfinite(number) or number < 0:
      raise ValueError(f"{label} {field} differs")
  for field in ("got_shape", "reference_shape"):
    shape = row[field]
    if not isinstance(shape, list) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in shape):
      raise ValueError(f"{label} {field} differs")
  for size_field, shape_field in (
      ("got_size", "got_shape"), ("reference_size", "reference_shape")):
    if row[size_field] != math.prod(row[shape_field]):
      raise ValueError(f"{label} {size_field} differs")
  for field in (
      "nan_got", "nan_reference", "inf_got", "inf_reference",
      "joint_finite"):
    if not isinstance(row[field], int) or isinstance(row[field], bool) or \
       row[field] < 0:
      raise ValueError(f"{label} {field} differs")
  if row["nan_got"] + row["inf_got"] > row["got_size"] or \
     row["nan_reference"] + row["inf_reference"] > \
       row["reference_size"] or \
     row["joint_finite"] > min(row["got_size"], row["reference_size"]):
    raise ValueError(f"{label} finite/non-finite counts differ")
  for field in ("max_abs_error", "mean_abs_error"):
    number = row[field]
    if number is not None and (
        not isinstance(number, (int, float)) or isinstance(number, bool) or
        not math.isfinite(number) or number < 0):
      raise ValueError(f"{label} {field} differs")
  mismatch = row["mismatch_count"]
  if mismatch is not None and (
      not isinstance(mismatch, int) or isinstance(mismatch, bool) or
      mismatch < 0):
    raise ValueError(f"{label} mismatch count differs")
  passed = row["status"] == "pass"
  if row["status"] not in ("pass", "mismatch") or \
     passed != (mismatch == 0) or \
     passed and (
       row["got_shape"] != row["reference_shape"] or
       row["first_mismatch_index"] is not None or
       row["first_mismatch_got"] is not None or
       row["first_mismatch_reference"] is not None or
       row["max_abs_error"] is None or row["mean_abs_error"] is None):
    raise ValueError(f"{label} status/mismatch facts differ")
  if mismatch is None and row["got_shape"] == row["reference_shape"]:
    raise ValueError(f"{label} shape mismatch facts differ")
  return row


def _validate_full_comparison(value: Any, label: str) -> dict[str, Any]:
  row = _validate_numeric_comparison(value, label)
  checks = {
    "status": row["status"] == "pass",
    "mismatch_count": row["mismatch_count"] == 0,
    "got_shape": row["got_shape"] == list(OUTPUT_SHAPE),
    "reference_shape": row["reference_shape"] == list(OUTPUT_SHAPE),
    "got_size": row["got_size"] == OUTPUT_ELEMENTS,
    "reference_size": row["reference_size"] == OUTPUT_ELEMENTS,
    "nan_got": row["nan_got"] == 0,
    "nan_reference": row["nan_reference"] == 0,
    "inf_got": row["inf_got"] == 0,
    "inf_reference": row["inf_reference"] == 0,
    "joint_finite": row["joint_finite"] == OUTPUT_ELEMENTS,
    "rtol": row["rtol"] == 3e-3,
    "atol": row["atol"] == 3e-3,
  }
  if not all(checks.values()):
    raise ValueError(
      f"{label} failed exact full-output checks: "
      f"{sorted(key for key, passed in checks.items() if not passed)!r}")
  return row


def ffn_gate_up_consumer_prefix_reference(
    fixture: Any, prefix_epochs: int, *,
    q8_values: np.ndarray | None = None,
    q8_scales: np.ndarray | None = None,
    q8_sums: np.ndarray | None = None,
    ) -> np.ndarray:
  """Frozen candidate oracle from the invocation's captured producer bytes.

  Fixture Q8 arrays are accepted only for CPU oracle construction/tests.
  Production passes all three captured arrays explicitly.
  """
  prefix_epochs = _prefix(prefix_epochs)
  role = getattr(fixture, "role_spec", None)
  if getattr(role, "role", None) != "ffn_gate_up" or \
     tuple(getattr(role, "shape", ())) != (512, 17408, 5120) or \
     getattr(role, "epochs", None) != 20:
    raise ValueError("candidate reference requires exact ffn_gate_up fixture")
  from extra.qk.mmq_q4k_q8_reference import (
    Q81MMQDS4Activation, Q81MMQDS4ActivationSpec, Q4KQ81MMQTileSpec,
    Q8_1_MMQ_DS4_LAYOUT, q4k_q8_1_mmq_ds4_tile_reference,
  )
  blocks = np.asarray(fixture.words).view(np.uint8).reshape(
    role.n, role.epochs, 144)
  records, k = prefix_epochs * 2, prefix_epochs * 256
  supplied = (q8_values, q8_scales, q8_sums)
  if any(value is None for value in supplied) and \
     not all(value is None for value in supplied):
    raise ValueError("candidate reference requires all or no captured Q8 arrays")
  values, scales, sums = (
    (fixture.q8_values, fixture.q8_scales, fixture.q8_sums)
    if all(value is None for value in supplied) else supplied)
  values, scales, sums = (
    np.ascontiguousarray(values), np.ascontiguousarray(scales),
    np.ascontiguousarray(sums))
  if values.shape != fixture.q8_values.shape or \
     scales.shape != fixture.q8_scales.shape or \
     sums.shape != fixture.q8_sums.shape or values.dtype != np.int8 or \
     scales.dtype != np.float32 or sums.dtype != np.float32:
    raise ValueError("candidate reference captured Q8 shape/dtype differs")
  operands = Q81MMQDS4Activation(
    np.ascontiguousarray(values[:records]),
    np.ascontiguousarray(
      scales[:records].astype(np.float16).astype(np.float32)),
    np.ascontiguousarray(
      sums[:records].astype(np.float16).astype(np.float32)),
    Q81MMQDS4ActivationSpec(m=role.m, k=k, m_tile=role.m))
  spec = Q4KQ81MMQTileSpec(
    role="ffn_gate_up_guarded_candidate", m=role.m, n=role.n, k=k,
    m_tile=role.m, n_tile=role.n,
    activation_layout=Q8_1_MMQ_DS4_LAYOUT)
  q4 = np.ascontiguousarray(
    blocks[:, :prefix_epochs, :]).reshape(-1)
  return q4k_q8_1_mmq_ds4_tile_reference(q4, operands, spec)


def ffn_gate_up_direct_dense_reference(fixture: Any) -> np.ndarray:
  """Independent dense FP16-activation/Q4_K dequantization oracle.

  This intentionally does not use Q8 arrays or the staged half2 oracle.
  """
  role = getattr(fixture, "role_spec", None)
  if getattr(role, "role", None) != "ffn_gate_up" or \
     tuple(getattr(role, "shape", ())) != (512, 17408, 5120):
    raise ValueError("direct reference requires exact ffn_gate_up fixture")
  from tinygrad import Tensor
  from extra.qk.layout import q4_k_reference
  raw = np.ascontiguousarray(np.asarray(fixture.words).view(np.uint8))
  dense_q4 = q4_k_reference(
    Tensor(raw.copy()), role.n * role.k).reshape(
      role.n, role.k).numpy().astype(np.float16).astype(np.float32)
  activation = np.ascontiguousarray(
    fixture.resident_fp16_activation.reshape(
      role.m, role.k)).astype(np.float32)
  return np.ascontiguousarray(activation @ dense_q4.T, dtype=np.float32)


@dataclass(frozen=True)
class CandidatePrefixRuntime:
  queue_mode: str
  prefix_epochs: int
  family_identity: str
  fixture_identity: str
  workload_identity: str
  input_identity: str
  logical_q4_identity: str
  resident_fp16_activation_identity: str
  candidate_executable_identity: str
  program_key: str
  binary_sha256: str
  c4_canary_identity: str
  session: Any
  producer_attest: Callable[[int], Mapping[str, Any]]
  synchronize: Callable[[], None]
  readback: Callable[[Any], Any]
  reference: Callable[[int], Any]
  comparator: Callable[[Any, Any], Mapping[str, Any]]

  def validate(self, queue_mode: str, prefix_epochs: int) -> "CandidatePrefixRuntime":
    if self.queue_mode != _queue(queue_mode) or \
       self.prefix_epochs != _prefix(prefix_epochs):
      raise ValueError("candidate runtime stage differs")
    for value, label in (
        (self.family_identity, "family identity"),
        (self.fixture_identity, "fixture identity"),
        (self.workload_identity, "workload identity"),
        (self.input_identity, "input identity"),
        (self.logical_q4_identity, "logical Q4 identity"),
        (self.resident_fp16_activation_identity,
         "resident FP16 activation identity"),
        (self.candidate_executable_identity, "candidate executable identity"),
        (self.c4_canary_identity, "C4 canary identity")):
      _content_identity(value, label)
    for value, label in (
        (self.program_key, "program key"),
        (self.binary_sha256, "binary SHA-256")):
      _hex_digest(value, label)
    if self.candidate_executable_identity != \
         _candidate_executable_identity_from_parts(
           self.family_identity, self.program_key, self.binary_sha256):
      raise ValueError("candidate executable identity differs")
    if not callable(getattr(self.session, "invoke", None)) or \
       not callable(getattr(self.session, "attest_post_sync", None)):
      raise TypeError("candidate runtime requires frozen low-level session")
    for callback in (
        self.producer_attest, self.synchronize, self.readback,
        self.reference, self.comparator):
      if not callable(callback):
        raise TypeError("candidate runtime callback differs")
    return self


@dataclass(frozen=True)
class FrozenCorrectnessEvidenceRef:
  path: str
  file_sha256: str
  envelope_evidence_identity: str

  @property
  def evidence_identity(self) -> str:
    """Compatibility spelling; the bound authority is always the envelope."""
    return self.envelope_evidence_identity

  def validate(self) -> "FrozenCorrectnessEvidenceRef":
    if not isinstance(self.path, str) or not Path(self.path).is_absolute():
      raise ValueError("frozen correctness evidence path must be absolute")
    if not isinstance(self.file_sha256, str) or \
       len(self.file_sha256) != 64 or \
       any(char not in _HEX for char in self.file_sha256):
      raise ValueError("frozen correctness file SHA-256 differs")
    _content_identity(
      self.envelope_evidence_identity,
      "frozen correctness envelope evidence identity")
    return self


def freeze_correctness_evidence(
    path: str | Path, value: Mapping[str, Any],
    ) -> FrozenCorrectnessEvidenceRef:
  """Validate and publish one immutable full guarded PASS envelope."""
  validated = validate_guarded_envelope(value)
  output = Path(path).resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  encoded = (
    json.dumps(validated, indent=2, sort_keys=True, allow_nan=False) +
    "\n").encode()
  fd, temporary = tempfile.mkstemp(
    prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
  temporary_path = Path(temporary)
  try:
    with os.fdopen(fd, "wb") as handle:
      handle.write(encoded)
      handle.flush()
      os.fsync(handle.fileno())
    try: os.link(temporary_path, output)
    except FileExistsError as exc:
      raise FileExistsError(
        f"refusing to replace frozen correctness evidence {output}") from exc
  finally:
    try: temporary_path.unlink()
    except FileNotFoundError: pass
  return FrozenCorrectnessEvidenceRef(
    str(output), hashlib.sha256(encoded).hexdigest(),
    validated["evidence_identity"]).validate()


def load_frozen_correctness_evidence(
    reference: FrozenCorrectnessEvidenceRef,
    ) -> dict[str, Any]:
  if not isinstance(reference, FrozenCorrectnessEvidenceRef):
    raise TypeError("prior correctness evidence must be a frozen reference")
  reference.validate()
  encoded = Path(reference.path).read_bytes()
  if hashlib.sha256(encoded).hexdigest() != reference.file_sha256:
    raise ValueError("frozen correctness evidence file content differs")
  value = json.loads(encoded)
  validated = validate_guarded_envelope(value)
  if validated["evidence_identity"] != reference.envelope_evidence_identity:
    raise ValueError("frozen correctness envelope identity differs")
  return validated


def _load_frozen_stage(
    reference: FrozenCorrectnessEvidenceRef, *, operation_schema: str,
    queue_mode: str | None = None,
    ) -> dict[str, Any]:
  envelope = load_frozen_correctness_evidence(reference)
  if envelope["operation_schema"] != operation_schema or \
     queue_mode is not None and envelope["queue_mode"] != queue_mode:
    raise ValueError("frozen correctness envelope operation differs")
  return dict(envelope["result"])


@dataclass(frozen=True)
class CandidatePrefixRequest:
  config: Mapping[str, Any]
  queue_mode: str
  prefix_epochs: int
  prior_evidence: FrozenCorrectnessEvidenceRef | None
  cross_queue_admission: FrozenCorrectnessEvidenceRef | None
  runtime_builder: Callable[..., CandidatePrefixRuntime]


@dataclass(frozen=True)
class PM4NoDoorbellRequest:
  config: Mapping[str, Any]
  runtime_builder: Callable[..., CandidatePrefixRuntime]


@dataclass(frozen=True)
class FFNReducedGridRequest:
  config: Mapping[str, Any]
  diagnostic_global_size: tuple[int, int, int]
  runtime_builder: Callable[..., CandidatePrefixRuntime]


def _pm4_no_doorbell_request_identity(
    request: PM4NoDoorbellRequest,
    ) -> str:
  return _identity({
    "schema": f"{SCHEMA}.pm4_no_doorbell_request",
    "queue_mode": "PM4",
    "prefix_epochs": 1,
    "submit_policy": "snapshot_only",
    "config_identity": _identity(dict(request.config)),
  })


def _ffn_reduced_grid_request_identity(
    request: FFNReducedGridRequest,
    ) -> str:
  return _identity({
    "schema": f"{SCHEMA}.ffn_reduced_grid_request",
    "queue_mode": "PM4",
    "prefix_epochs": 1,
    "diagnostic_global_size":
      list(_ffn_reduced_global_size(request.diagnostic_global_size)),
    "config_identity": _identity(dict(request.config)),
  })


def _candidate_request_identity(request: CandidatePrefixRequest) -> str:
  return _identity({
    "schema": f"{SCHEMA}.candidate_request",
    "queue_mode": request.queue_mode,
    "prefix_epochs": request.prefix_epochs,
    "config_identity": _identity(dict(request.config)),
    "prior_evidence_identity":
      None if request.prior_evidence is None else
      getattr(request.prior_evidence, "evidence_identity", None),
    "cross_queue_admission_identity":
      None if request.cross_queue_admission is None else
      getattr(request.cross_queue_admission, "evidence_identity", None),
  })


def _validate_pm4_pre_submit_snapshot(value: Any) -> dict[str, Any]:
  row = dict(_mapping(value, "PM4 no-doorbell pre-submit snapshot"))
  _exact_keys(row, {
    "schema", "capture_point", "runtime_object_identity", "runtime_class",
    "runtime_name", "runtime_device", "kernarg_va", "kernarg_nbytes",
    "kernarg_qwords", "argument_buffers", "pm4_kernarg_user_data",
    "pm4_dispatch_direct", "pm4_workgroup_size", "pm4_program_entry",
    "pm4_dword_count", "pm4_sha256", "checks", "all_checks_pass",
  }, "PM4 no-doorbell pre-submit snapshot")
  checks = dict(_mapping(
    row.get("checks"), "PM4 no-doorbell pre-submit checks"))
  _exact_keys(
    checks, set(PM4_PRE_SUBMIT_CHECK_KEYS),
    "PM4 no-doorbell pre-submit checks")
  exact_int = lambda item: \
    isinstance(item, int) and not isinstance(item, bool)
  identity_checks = {
    "schema": row.get("schema") == PM4_PRE_SUBMIT_SCHEMA,
    "capture_point":
      row.get("capture_point") == PM4_PRE_SUBMIT_CAPTURE_POINT,
    "runtime_object_identity":
      exact_int(row.get("runtime_object_identity")) and
      row["runtime_object_identity"] > 0,
    "runtime_class": row.get("runtime_class") == PM4_RUNTIME_CLASS,
    "runtime_name": row.get("runtime_name") == PM4_RUNTIME_NAME,
    "runtime_device": row.get("runtime_device") == "AMD",
    "kernarg_va":
      exact_int(row.get("kernarg_va")) and
      0 < row["kernarg_va"] < 1 << 64,
    "kernarg_nbytes": row.get("kernarg_nbytes") == 40,
    "pm4_dword_count":
      exact_int(row.get("pm4_dword_count")) and
      row["pm4_dword_count"] > 0,
    "pm4_sha256": isinstance(row.get("pm4_sha256"), str) and
      len(row["pm4_sha256"]) == 64 and
      all(char in _HEX for char in row["pm4_sha256"]),
    "checks": all(item is True for item in checks.values()),
    "all_checks_pass": row.get("all_checks_pass") is True,
  }
  if not all(identity_checks.values()):
    raise ValueError(
      "PM4 no-doorbell pre-submit identity differs: "
      f"{sorted(key for key, passed in identity_checks.items() if not passed)!r}")

  arguments = row.get("argument_buffers")
  if not isinstance(arguments, list) or len(arguments) != 5:
    raise ValueError(
      "PM4 no-doorbell pre-submit requires five argument buffers")
  argument_vas: list[int] = []
  for slot, (argument, expected_nbytes) in enumerate(
      zip(arguments, PM4_ARGUMENT_NBYTES)):
    argument = dict(_mapping(
      argument, f"PM4 no-doorbell argument {slot}"))
    _exact_keys(
      argument, {"slot", "va", "size"},
      f"PM4 no-doorbell argument {slot}")
    va = argument.get("va")
    if argument.get("slot") != slot or argument.get("size") != \
       expected_nbytes or not exact_int(va) or not 0 < va < 1 << 64:
      raise ValueError(
        f"PM4 no-doorbell argument {slot} differs from ABI authority")
    argument_vas.append(va)

  qwords = row.get("kernarg_qwords")
  if not isinstance(qwords, list) or len(qwords) != 5 or \
     any(not exact_int(item) or not 0 < item < 1 << 64 for item in qwords) or \
     qwords != argument_vas:
    raise ValueError(
      "PM4 no-doorbell kernarg qwords differ from argument VAs")

  user_data = dict(_mapping(
    row.get("pm4_kernarg_user_data"),
    "PM4 no-doorbell kernarg USER_DATA"))
  _exact_keys(user_data, {
    "packet_dword_offset", "register_index", "low_dword", "high_dword",
    "pointer",
  }, "PM4 no-doorbell kernarg USER_DATA")
  packet_offset = user_data.get("packet_dword_offset")
  low, high, pointer = (
    user_data.get("low_dword"), user_data.get("high_dword"),
    user_data.get("pointer"))
  if not exact_int(packet_offset) or \
     not 0 <= packet_offset < row["pm4_dword_count"] or \
     user_data.get("register_index") != 0 or \
     any(not exact_int(item) or not 0 <= item <= 0xffffffff
         for item in (low, high)) or \
     not exact_int(pointer) or pointer != low | (high << 32) or \
     pointer != row["kernarg_va"]:
    raise ValueError(
      "PM4 no-doorbell USER_DATA_0 differs from kernarg authority")
  dispatch = dict(_mapping(
    row.get("pm4_dispatch_direct"), "PM4 DISPATCH_DIRECT"))
  _exact_keys(dispatch, {
    "packet_dword_offset", "group_counts", "dispatch_initiator"},
    "PM4 DISPATCH_DIRECT")
  workgroup = dict(_mapping(
    row.get("pm4_workgroup_size"), "PM4 workgroup size"))
  _exact_keys(workgroup, {
    "packet_dword_offset", "register_index", "size"},
    "PM4 workgroup size")
  program = dict(_mapping(
    row.get("pm4_program_entry"), "PM4 program entry"))
  _exact_keys(program, {
    "packet_dword_offset", "register_index", "low_dword", "high_dword",
    "entry_va"}, "PM4 program entry")
  if any(
      not exact_int(item) or not 0 <= item < row["pm4_dword_count"]
      for item in (
        dispatch.get("packet_dword_offset"),
        workgroup.get("packet_dword_offset"),
        program.get("packet_dword_offset"))) or \
     not isinstance(dispatch.get("group_counts"), list) or \
     len(dispatch["group_counts"]) != 3 or \
     any(not exact_int(item) or item <= 0
         for item in dispatch["group_counts"]) or \
     not exact_int(dispatch.get("dispatch_initiator")) or \
     workgroup.get("register_index") != 0 or \
     not isinstance(workgroup.get("size"), list) or \
     len(workgroup["size"]) != 3 or \
     any(not exact_int(item) or item <= 0 for item in workgroup["size"]) or \
     program.get("register_index") != 0 or \
     any(not exact_int(program.get(key)) or
         not 0 <= program[key] <= 0xffffffff
         for key in ("low_dword", "high_dword")) or \
     not exact_int(program.get("entry_va")) or \
     program["entry_va"] != (
       program["low_dword"] | (program["high_dword"] << 32)) << 8:
    raise ValueError("PM4 decoded command authority differs")
  return row


def _validate_pm4_no_doorbell_receipt(value: Any) -> dict[str, Any]:
  row = dict(_mapping(value, "PM4 no-doorbell receipt"))
  _exact_keys(row, {
    "schema", "status", "submit_policy", "pre_submit",
    "target_dispatch_submitted", "native_submit_call_count",
    "ring_copy_performed", "doorbell_rung", "timeline_value_before",
    "timeline_value_after_runtime_unwind", "timeline_value_after_rollback",
    "timeline_rollback_applied", "prof_exec_counter_before",
    "prof_exec_counter_after_runtime_unwind",
    "prof_exec_counter_after_rollback", "timeline_signal_value_before",
    "timeline_signal_value_after", "terminal_child_required",
    "promotion_evidence_eligible", "checks", "all_checks_pass",
  }, "PM4 no-doorbell receipt")
  checks = _mapping(row.get("checks"), "PM4 no-doorbell receipt checks")
  _exact_keys(dict(checks), {
    "private_stop_raised_at_target_submit", "private_stop_caught_by_owner",
    "pre_submit_snapshot_passed", "native_submit_not_called",
    "timeline_advanced_exactly_once",
    "prof_exec_counter_advanced_exactly_once",
    "timeline_rollback_restored", "prof_exec_counter_rollback_restored",
    "timeline_signal_unchanged", "error_state_unchanged",
    "submit_hook_restored", "fill_kernargs_hook_restored",
  }, "PM4 no-doorbell receipt checks")
  state = {
    "schema": row.get("schema") == PM4_NO_DOORBELL_RECEIPT_SCHEMA,
    "status": row.get("status") == "CAPTURED_NO_SUBMIT",
    "policy": row.get("submit_policy") == "snapshot_only",
    "target_not_submitted": row.get("target_dispatch_submitted") is False,
    "native_submit_count": row.get("native_submit_call_count") == 0,
    "ring_copy": row.get("ring_copy_performed") is False,
    "doorbell": row.get("doorbell_rung") is False,
    "rollback": row.get("timeline_rollback_applied") is True,
    "terminal": row.get("terminal_child_required") is True,
    "promotion": row.get("promotion_evidence_eligible") is False,
    "checks": all(item is True for item in checks.values()),
    "all_checks": row.get("all_checks_pass") is True,
  }
  integer_fields = (
    "timeline_value_before", "timeline_value_after_runtime_unwind",
    "timeline_value_after_rollback", "prof_exec_counter_before",
    "prof_exec_counter_after_runtime_unwind",
    "prof_exec_counter_after_rollback", "timeline_signal_value_before",
    "timeline_signal_value_after",
  )
  if not all(state.values()) or any(
      not isinstance(row.get(field), int) or isinstance(row.get(field), bool)
      for field in integer_fields):
    raise ValueError(
      "PM4 no-doorbell receipt failed exact checks: "
      f"{sorted(key for key, passed in state.items() if not passed)!r}")
  if row["timeline_value_after_runtime_unwind"] != \
       row["timeline_value_before"] + 1 or \
     row["timeline_value_after_rollback"] != row["timeline_value_before"] or \
     row["prof_exec_counter_after_runtime_unwind"] != \
       row["prof_exec_counter_before"] + 1 or \
     row["prof_exec_counter_after_rollback"] != \
       row["prof_exec_counter_before"] or \
     row["timeline_signal_value_after"] != \
       row["timeline_signal_value_before"]:
    raise ValueError("PM4 no-doorbell receipt counter facts differ")
  _validate_pm4_pre_submit_snapshot(row.get("pre_submit"))
  return row


def validate_pm4_no_doorbell_evidence(value: Any) -> dict[str, Any]:
  row = dict(_mapping(value, "PM4 no-doorbell evidence"))
  _exact_keys(row, {
    "schema", "status", "exact_blocker", "queue_mode", "prefix_epochs",
    "submit_policy", "no_retry", "retry_count", "no_fallback",
    "compile_performed", "requires_recompile",
    "promotion_evidence_eligible", "config_identity", "request_identity",
    "environment", "family_identity", "fixture_identity",
    "workload_identity", "input_identity", "logical_q4_identity",
    "resident_fp16_activation_identity", "candidate_executable_identity",
    "program_key", "binary_sha256", "c4_canary_identity",
    "invocation_count", "receipt_count", "receipt",
    "target_dispatch_submitted", "native_submit_call_count",
    "timeline_rollback_applied", "terminal_child_required",
    "readback_performed", "numeric_validation_performed",
    "attestation_performed", "producer_attestation_performed",
    "attempt", "evidence_identity",
  }, "PM4 no-doorbell evidence")
  if not _identity_valid(row) or row.get("schema") != \
       PM4_NO_DOORBELL_SCHEMA or row.get("status") != "PASS" or \
     row.get("exact_blocker") is not None:
    raise ValueError("PM4 no-doorbell evidence identity/state differs")
  expected = {
    "queue_mode": "PM4", "prefix_epochs": 1,
    "submit_policy": "snapshot_only", "no_retry": True, "retry_count": 0,
    "no_fallback": True, "compile_performed": False,
    "requires_recompile": False, "promotion_evidence_eligible": False,
    "environment": {"DEV": "AMD", "AMD_AQL": "0", "PROFILE": "0"},
    "invocation_count": 1, "receipt_count": 1,
    "target_dispatch_submitted": False, "native_submit_call_count": 0,
    "timeline_rollback_applied": True, "terminal_child_required": True,
    "readback_performed": False, "numeric_validation_performed": False,
    "attestation_performed": False,
    "producer_attestation_performed": False,
  }
  if any(row.get(key) != item for key, item in expected.items()):
    raise ValueError("PM4 no-doorbell evidence diagnostic state differs")
  for field in (
      "config_identity", "request_identity", "family_identity",
      "fixture_identity", "workload_identity", "input_identity",
      "logical_q4_identity", "resident_fp16_activation_identity",
      "candidate_executable_identity", "c4_canary_identity"):
    _content_identity(row.get(field), f"PM4 no-doorbell {field}")
  for field in ("program_key", "binary_sha256"):
    _hex_digest(row.get(field), f"PM4 no-doorbell {field}")
  if row["candidate_executable_identity"] != \
       _candidate_executable_identity_from_parts(
         row["family_identity"], row["program_key"], row["binary_sha256"]):
    raise ValueError("PM4 no-doorbell executable derivation differs")
  receipt = _validate_pm4_no_doorbell_receipt(row.get("receipt"))
  if any(row.get(field) != receipt[field] for field in (
      "target_dispatch_submitted", "native_submit_call_count",
      "timeline_rollback_applied", "terminal_child_required")):
    raise ValueError("PM4 no-doorbell nested receipt binding differs")
  attempt = _mapping(row.get("attempt"), "PM4 no-doorbell attempt")
  _exact_keys(dict(attempt), {
    "phase", "invocation_count", "receipt_count", "receipt",
  }, "PM4 no-doorbell attempt")
  if attempt != {
      "phase": "complete", "invocation_count": 1, "receipt_count": 1,
      "receipt": receipt}:
    raise ValueError("PM4 no-doorbell nested attempt differs")
  return row


_LOW_LEVEL_DIAGNOSTIC_RECEIPT_KEYS = {
  "schema", "status", "queue_mode", "promotion_evidence_eligible",
  "target_dispatch_submitted", "target_submit_entered",
  "target_submit_returned", "native_submit_entered_count",
  "native_submit_returned_count", "post_dispatch_sync_completed",
  "family_identity",
  "candidate_executable_identity", "input_identity", "program_key",
  "binary_sha256", "runtime_class", "runtime_name", "runtime_device",
  "runtime_object_identity", "runtime_device_identity_exact",
  "runtime_cache_binding_exact", "library_va", "library_nbytes", "entry_va",
  "fixed_five_vas", "frozen_global_size", "effective_global_size",
  "local_size", "pre_submit", "launch_count", "observation_identity",
}


def _validate_ffn_reduced_receipt(
    value: Any, *, runtime: CandidatePrefixRuntime | None = None,
    diagnostic_global_size: tuple[int, int, int] | None = None,
    ) -> dict[str, Any]:
  if isinstance(value, Mapping):
    row = dict(value)
  else:
    row = {
      key: (
        list(getattr(value, key, ()))
        if key in (
          "fixed_five_vas", "frozen_global_size", "effective_global_size",
          "local_size")
        else getattr(value, key, None))
      for key in _LOW_LEVEL_DIAGNOSTIC_RECEIPT_KEYS}
  _exact_keys(
    row, set(_LOW_LEVEL_DIAGNOSTIC_RECEIPT_KEYS),
    "FFN reduced-grid low-level receipt")
  exact_int = lambda item: \
    isinstance(item, int) and not isinstance(item, bool)
  expected_grid = (
    _ffn_reduced_global_size(diagnostic_global_size)
    if diagnostic_global_size is not None else
    _ffn_reduced_global_size(row.get("effective_global_size")))
  checks = {
    "schema":
      row.get("schema") == LOW_LEVEL_DIAGNOSTIC_RECEIPT_SCHEMA,
    "status": row.get("status") == "PASS",
    "queue_mode": row.get("queue_mode") == "PM4",
    "promotion_evidence_eligible":
      row.get("promotion_evidence_eligible") is False,
    "target_dispatch_submitted":
      row.get("target_dispatch_submitted") is True,
    "target_submit_entered": row.get("target_submit_entered") is True,
    "target_submit_returned": row.get("target_submit_returned") is True,
    "native_submit_entered_count":
      row.get("native_submit_entered_count") == 1,
    "native_submit_returned_count":
      row.get("native_submit_returned_count") == 1,
    "post_dispatch_sync_completed":
      row.get("post_dispatch_sync_completed") is True,
    "runtime_class": row.get("runtime_class") == PM4_RUNTIME_CLASS,
    "runtime_name": row.get("runtime_name") == PM4_RUNTIME_NAME,
    "runtime_device": row.get("runtime_device") == "AMD",
    "runtime_object_identity":
      exact_int(row.get("runtime_object_identity")) and
      row["runtime_object_identity"] > 0,
    "runtime_device_identity_exact":
      row.get("runtime_device_identity_exact") is True,
    "runtime_cache_binding_exact":
      row.get("runtime_cache_binding_exact") is True,
    "library_va":
      exact_int(row.get("library_va")) and row["library_va"] > 0,
    "library_nbytes":
      exact_int(row.get("library_nbytes")) and row["library_nbytes"] > 0,
    "entry_va":
      exact_int(row.get("entry_va")) and
      row.get("library_va", 0) <= row["entry_va"] <
        row.get("library_va", 0) + row.get("library_nbytes", 0),
    "fixed_five_vas":
      isinstance(row.get("fixed_five_vas"), list) and
      len(row["fixed_five_vas"]) == 5 and
      all(exact_int(item) and 0 < item < 1 << 64
          for item in row["fixed_five_vas"]),
    "frozen_global_size":
      row.get("frozen_global_size") == list(FFN_FROZEN_GLOBAL_SIZE),
    "effective_global_size":
      row.get("effective_global_size") == list(expected_grid),
    "local_size":
      row.get("local_size") == list(FFN_FROZEN_LOCAL_SIZE),
    "launch_count": row.get("launch_count") == 1,
  }
  for field in (
      "family_identity", "candidate_executable_identity", "input_identity"):
    try: _content_identity(row.get(field), f"FFN reduced receipt {field}")
    except ValueError: checks[field] = False
    else: checks[field] = True
  for field in ("program_key", "binary_sha256"):
    try: _hex_digest(row.get(field), f"FFN reduced receipt {field}")
    except ValueError: checks[field] = False
    else: checks[field] = True
  payload = {
    key: item for key, item in row.items()
    if key != "observation_identity"}
  checks["observation_identity"] = \
    row.get("observation_identity") == _identity(payload)
  try:
    pre_submit = _validate_pm4_pre_submit_snapshot(row.get("pre_submit"))
  except ValueError:
    checks["pre_submit"] = False
  else:
    checks["pre_submit"] = True
    checks["pre_submit_five_va_binding"] = \
      pre_submit["kernarg_qwords"] == row.get("fixed_five_vas")
    checks["pre_submit_runtime_binding"] = all((
      pre_submit["runtime_class"] == row.get("runtime_class"),
      pre_submit["runtime_name"] == row.get("runtime_name"),
      pre_submit["runtime_device"] == row.get("runtime_device"),
      pre_submit["runtime_object_identity"] ==
        row.get("runtime_object_identity"),
    ))
    checks["pre_submit_grid_binding"] = \
      pre_submit["pm4_dispatch_direct"]["group_counts"] == \
      row.get("effective_global_size")
    checks["pre_submit_local_binding"] = \
      pre_submit["pm4_workgroup_size"]["size"] == row.get("local_size")
    checks["pre_submit_entry_binding"] = \
      pre_submit["pm4_program_entry"]["entry_va"] == row.get("entry_va")
  if runtime is not None:
    checks.update({
      "runtime_queue_binding": runtime.queue_mode == "PM4",
      "runtime_family_binding":
        row.get("family_identity") == runtime.family_identity,
      "runtime_executable_binding":
        row.get("candidate_executable_identity") ==
          runtime.candidate_executable_identity,
      "runtime_input_binding":
        row.get("input_identity") == runtime.input_identity,
      "runtime_program_binding":
        row.get("program_key") == runtime.program_key,
      "runtime_binary_binding":
        row.get("binary_sha256") == runtime.binary_sha256,
    })
  if not all(checks.values()):
    raise ValueError(
      "FFN reduced-grid low-level receipt failed exact checks: "
      f"{sorted(key for key, passed in checks.items() if not passed)!r}")
  return row


def _validate_ffn_reduced_comparison(
    value: Any, *, diagnostic_global_size: tuple[int, int, int],
    ) -> dict[str, Any]:
  row = _validate_numeric_comparison(
    value, "FFN reduced-grid touched comparison")
  gx, gy, _ = _ffn_reduced_global_size(diagnostic_global_size)
  touched_shape = [gy * 128, gx * 128]
  touched_elements = math.prod(touched_shape)
  checks = {
    "status": row["status"] == "pass",
    "rtol": row["rtol"] == 3e-3,
    "atol": row["atol"] == 3e-3,
    "got_shape": row["got_shape"] == touched_shape,
    "reference_shape": row["reference_shape"] == touched_shape,
    "got_size": row["got_size"] == touched_elements,
    "reference_size": row["reference_size"] == touched_elements,
    "mismatch_count": row["mismatch_count"] == 0,
    "nan_got": row["nan_got"] == 0,
    "nan_reference": row["nan_reference"] == 0,
    "inf_got": row["inf_got"] == 0,
    "inf_reference": row["inf_reference"] == 0,
    "joint_finite": row["joint_finite"] == touched_elements,
  }
  if not all(checks.values()):
    raise ValueError(
      "FFN reduced-grid touched comparison failed exact checks: "
      f"{sorted(key for key, passed in checks.items() if not passed)!r}")
  return row


def validate_ffn_reduced_grid_evidence(
    value: Any, *,
    diagnostic_global_size: tuple[int, int, int] | None = None,
    ) -> dict[str, Any]:
  row = dict(_mapping(value, "FFN reduced-grid evidence"))
  _exact_keys(row, {
    "schema", "status", "exact_blocker", "queue_mode", "prefix_epochs",
    "diagnostic_global_size", "frozen_global_size", "local_size",
    "touched_shape", "touched_element_count", "untouched_element_count",
    "untouched_nonzero_count", "first_untouched_nonzero_index",
    "first_untouched_nonzero_value", "no_retry", "retry_count",
    "no_fallback", "compile_performed", "requires_recompile",
    "promotion_evidence_eligible", "ordinary_attestation_performed",
    "full_grid_validation_performed", "full_prefix1_reference_constructed",
    "target_dispatch_submitted", "target_submit_entered",
    "target_submit_returned", "invocation_count", "receipt_count",
    "readback_performed", "numeric_validation_performed",
    "producer_attestation_performed", "config_identity", "request_identity",
    "environment", "family_identity", "fixture_identity",
    "workload_identity", "input_identity", "logical_q4_identity",
    "resident_fp16_activation_identity", "candidate_executable_identity",
    "program_key", "binary_sha256", "c4_canary_identity", "q8_producer",
    "consumer_reference_q8_sha256", "diagnostic_receipt",
    "touched_comparison", "output_sha256", "reference_sha256", "attempt",
    "evidence_identity",
  }, "FFN reduced-grid evidence")
  if not _identity_valid(row) or row.get("schema") != \
       FFN_REDUCED_GRID_SCHEMA or row.get("status") != "PASS" or \
     row.get("exact_blocker") is not None:
    raise ValueError("FFN reduced-grid evidence identity/state differs")
  grid = _ffn_reduced_global_size(row.get("diagnostic_global_size"))
  if diagnostic_global_size is not None and \
     grid != _ffn_reduced_global_size(diagnostic_global_size):
    raise ValueError("FFN reduced-grid evidence request grid differs")
  gx, gy, _ = grid
  touched_shape = [gy * 128, gx * 128]
  touched_elements = math.prod(touched_shape)
  expected = {
    "queue_mode": "PM4", "prefix_epochs": 1,
    "frozen_global_size": list(FFN_FROZEN_GLOBAL_SIZE),
    "local_size": list(FFN_FROZEN_LOCAL_SIZE),
    "touched_shape": touched_shape,
    "touched_element_count": touched_elements,
    "untouched_element_count": OUTPUT_ELEMENTS - touched_elements,
    "untouched_nonzero_count": 0,
    "first_untouched_nonzero_index": None,
    "first_untouched_nonzero_value": None,
    "no_retry": True, "retry_count": 0, "no_fallback": True,
    "compile_performed": False, "requires_recompile": False,
    "promotion_evidence_eligible": False,
    "ordinary_attestation_performed": False,
    "full_grid_validation_performed": False,
    "full_prefix1_reference_constructed": True,
    "target_dispatch_submitted": True,
    "target_submit_entered": True, "target_submit_returned": True,
    "invocation_count": 1, "receipt_count": 1,
    "readback_performed": True, "numeric_validation_performed": True,
    "producer_attestation_performed": True,
    "environment": {"DEV": "AMD", "AMD_AQL": "0", "PROFILE": "0"},
  }
  if any(row.get(key) != item for key, item in expected.items()):
    raise ValueError("FFN reduced-grid evidence diagnostic state differs")
  for field in (
      "config_identity", "request_identity", "family_identity",
      "fixture_identity", "workload_identity", "input_identity",
      "logical_q4_identity", "resident_fp16_activation_identity",
      "candidate_executable_identity", "c4_canary_identity"):
    _content_identity(row.get(field), f"FFN reduced-grid {field}")
  for field in ("program_key", "binary_sha256"):
    _hex_digest(row.get(field), f"FFN reduced-grid {field}")
  if row["candidate_executable_identity"] != \
       _candidate_executable_identity_from_parts(
         row["family_identity"], row["program_key"], row["binary_sha256"]):
    raise ValueError("FFN reduced-grid executable derivation differs")
  producer = _candidate_producer_evidence(
    row.get("q8_producer"), {
      "queue_mode": "PM4", "prefix_epochs": 1,
      "family_identity": row["family_identity"],
      "input_identity": row["input_identity"],
    })
  if row.get("consumer_reference_q8_sha256") != \
       producer["consumer_reference_q8_sha256"]:
    raise ValueError("FFN reduced-grid Q8 reference binding differs")
  receipt = _validate_ffn_reduced_receipt(
    row.get("diagnostic_receipt"), diagnostic_global_size=grid)
  receipt_bindings = {
    "family_identity": row["family_identity"],
    "candidate_executable_identity": row["candidate_executable_identity"],
    "input_identity": row["input_identity"], "program_key": row["program_key"],
    "binary_sha256": row["binary_sha256"],
  }
  if any(receipt.get(key) != item for key, item in receipt_bindings.items()):
    raise ValueError("FFN reduced-grid receipt/executable binding differs")
  comparison = _validate_ffn_reduced_comparison(
    row.get("touched_comparison"), diagnostic_global_size=grid)
  for field in ("output_sha256", "reference_sha256"):
    _hex_digest(row.get(field), f"FFN reduced-grid {field}")
  attempt = dict(_mapping(row.get("attempt"), "FFN reduced-grid attempt"))
  _exact_keys(attempt, {
    "phase", "invocation_count", "receipt_count", "diagnostic_receipt",
    "q8_producer", "touched_comparison", "output_sha256",
    "reference_sha256", "touched_shape", "untouched_element_count",
    "untouched_nonzero_count", "first_untouched_nonzero_index",
    "first_untouched_nonzero_value",
  }, "FFN reduced-grid attempt")
  expected_attempt = {
    "phase": "complete", "invocation_count": 1, "receipt_count": 1,
    "diagnostic_receipt": receipt, "q8_producer": producer,
    "touched_comparison": comparison,
    "output_sha256": row["output_sha256"],
    "reference_sha256": row["reference_sha256"],
    "touched_shape": touched_shape,
    "untouched_element_count": OUTPUT_ELEMENTS - touched_elements,
    "untouched_nonzero_count": 0,
    "first_untouched_nonzero_index": None,
    "first_untouched_nonzero_value": None,
  }
  if attempt != expected_attempt:
    raise ValueError("FFN reduced-grid nested attempt differs")
  return row


def validate_candidate_prefix_evidence(
    value: Any, *, queue_mode: str | None = None,
    prefix_epochs: int | None = None,
    ) -> dict[str, Any]:
  row = dict(_mapping(value, "candidate prefix evidence"))
  _exact_keys(row, {
    "schema", "queue_mode", "prefix_epochs", "status", "exact_blocker",
    "no_retry", "retry_count", "no_fallback", "compile_performed",
    "requires_recompile", "promotion_evidence_eligible", "config_identity",
    "request_identity", "gate", "family_identity", "fixture_identity",
    "workload_identity", "input_identity", "logical_q4_identity",
    "resident_fp16_activation_identity", "candidate_executable_identity",
    "program_key", "binary_sha256", "c4_canary_identity",
    "predecessor_evidence_identity", "q8_producer", "attestation",
    "consumer_reference_q8_sha256", "comparison", "output_sha256",
    "reference_sha256", "post_sync_before_readback", "readback_performed",
    "attempt", "evidence_identity",
  }, "candidate prefix evidence")
  if not _identity_valid(row) or row.get("schema") != CANDIDATE_SCHEMA or \
     row.get("status") != "PASS" or \
     row.get("exact_blocker") is not None or row.get("retry_count") != 0 or \
     row.get("promotion_evidence_eligible") is not False or \
     row.get("no_retry") is not True or row.get("no_fallback") is not True:
    raise ValueError("candidate prefix evidence identity/state differs")
  _queue(row.get("queue_mode"))
  _prefix(row.get("prefix_epochs"))
  if queue_mode is not None and row["queue_mode"] != queue_mode:
    raise ValueError("candidate prefix evidence queue differs")
  if prefix_epochs is not None and row["prefix_epochs"] != prefix_epochs:
    raise ValueError("candidate prefix evidence stage differs")
  if row.get("gate") != ("C6" if row["prefix_epochs"] == 20 else "C5") or \
     row.get("compile_performed") is not False or \
     row.get("requires_recompile") is not False:
    raise ValueError("candidate gate/compile state differs")
  predecessor = row.get("predecessor_evidence_identity")
  if row["prefix_epochs"] == 1 and row["queue_mode"] == "PM4":
    if predecessor is not None:
      raise ValueError("first PM4 prefix-1 predecessor differs")
  else:
    _content_identity(predecessor, "candidate predecessor identity")
  for field in (
      "family_identity", "fixture_identity", "workload_identity",
      "input_identity", "logical_q4_identity",
      "resident_fp16_activation_identity", "candidate_executable_identity",
      "c4_canary_identity", "request_identity", "config_identity"):
    _content_identity(row.get(field), f"candidate {field}")
  _validate_full_comparison(row.get("comparison"), "candidate comparison")
  producer = _candidate_producer_evidence(row.get("q8_producer"), row)
  if row.get("consumer_reference_q8_sha256") != \
       producer["consumer_reference_q8_sha256"]:
    raise ValueError("candidate consumer reference Q8 binding differs")
  hashes = _mapping(
    row.get("consumer_reference_q8_sha256"),
    "candidate consumer reference hashes")
  _exact_keys(hashes, {"values", "scales", "sums"},
              "candidate consumer reference hashes")
  for field, digest in hashes.items():
    _hex_digest(digest, f"candidate consumer {field} hash")
  attestation = dict(_mapping(
    row.get("attestation"), "candidate attestation"))
  _exact_keys(attestation, _LOW_LEVEL_ATTESTATION_KEYS,
              "candidate attestation")
  expected_attestation = {
    "schema": LOW_LEVEL_ATTESTATION_SCHEMA,
    "status": "PASS", "queue_mode": row["queue_mode"],
    "family_identity": row["family_identity"],
    "candidate_executable_identity":
      row["candidate_executable_identity"],
    "input_identity": row["input_identity"],
    "program_key": row.get("program_key"),
    "binary_sha256": row.get("binary_sha256"),
    "launch_count": row["prefix_epochs"],
  }
  if any(attestation.get(key) != value
         for key, value in expected_attestation.items()):
    raise ValueError("candidate nested attestation differs")
  attestation_payload = {
    key: value for key, value in attestation.items()
    if key != "observation_identity"}
  if attestation["observation_identity"] != _identity(attestation_payload):
    raise ValueError("candidate attestation observation identity differs")
  for field in ("runtime_class", "runtime_name"):
    if not isinstance(attestation[field], str) or not attestation[field]:
      raise ValueError(f"candidate attestation {field} differs")
  if attestation["runtime_device"] != "AMD" or \
     attestation["runtime_device_identity_exact"] is not True or \
     attestation["runtime_cache_binding_exact"] is not True:
    raise ValueError("candidate attestation runtime binding differs")
  for field in (
      "runtime_object_identity", "library_va", "library_nbytes", "entry_va"):
    if not isinstance(attestation[field], int) or \
       isinstance(attestation[field], bool) or attestation[field] <= 0:
      raise ValueError(f"candidate attestation {field} differs")
  if not attestation["library_va"] <= attestation["entry_va"] < \
       attestation["library_va"] + attestation["library_nbytes"]:
    raise ValueError("candidate attestation entry range differs")
  vas = attestation["fixed_five_vas"]
  if not isinstance(vas, list) or len(vas) != 5 or len(set(vas)) != 5 or \
     any(not isinstance(va, int) or isinstance(va, bool) or va <= 0
         for va in vas):
    raise ValueError("candidate attestation fixed VA set differs")
  if row["candidate_executable_identity"] != \
       _candidate_executable_identity_from_parts(
         row["family_identity"], row["program_key"], row["binary_sha256"]):
    raise ValueError("candidate executable derivation differs")
  for field in ("output_sha256", "reference_sha256", "program_key",
                "binary_sha256"):
    _hex_digest(row.get(field), f"candidate {field}")
  if row.get("post_sync_before_readback") is not True or \
     row.get("readback_performed") is not True:
    raise ValueError("candidate readback ordering differs")
  attempt = _mapping(row.get("attempt"), "candidate attempt")
  _exact_keys(attempt, {
    "phase", "family_identity", "fixture_identity", "workload_identity",
    "input_identity", "logical_q4_identity",
    "resident_fp16_activation_identity", "candidate_executable_identity",
    "c4_canary_identity", "attestation", "q8_producer", "comparison",
    "output_sha256", "reference_sha256",
  }, "candidate attempt")
  if attempt.get("phase") != "complete" or \
     attempt.get("attestation") != attestation or \
     attempt.get("q8_producer") != row["q8_producer"] or \
     attempt.get("comparison") != row["comparison"] or \
     attempt.get("output_sha256") != row["output_sha256"] or \
     attempt.get("reference_sha256") != row["reference_sha256"] or \
     any(attempt.get(field) != row[field] for field in (
       "family_identity", "fixture_identity", "workload_identity",
       "input_identity", "logical_q4_identity",
       "resident_fp16_activation_identity",
       "candidate_executable_identity", "c4_canary_identity")):
    raise ValueError("candidate nested attempt differs")
  return row


def _candidate_producer_evidence(
    value: Any, authority: Mapping[str, Any],
    ) -> dict[str, Any]:
  row = dict(_mapping(value, "Q8 producer evidence"))
  _exact_keys(row, {
    "schema", "status", "queue_mode", "prefix_epochs", "family_identity",
    "input_identity", "metadata_storage_dtype",
    "captured_for_consumer_reference", "consumer_reference_q8_sha256",
    "fixture_diagnostic", "promotion_evidence_eligible",
    "evidence_identity",
  }, "Q8 producer evidence")
  checks = {
    "identity": _identity_valid(row),
    "schema": row.get("schema") == PRODUCER_SCHEMA,
    "status": row.get("status") == "PASS",
    "queue": row.get("queue_mode") == authority.get("queue_mode"),
    "family": row.get("family_identity") == authority.get("family_identity"),
    "input": row.get("input_identity") == authority.get("input_identity"),
    "prefix": row.get("prefix_epochs") == authority.get("prefix_epochs"),
    "dtype": row.get("metadata_storage_dtype") == "float32",
    "captured": row.get("captured_for_consumer_reference") is True,
    "promotion": row.get("promotion_evidence_eligible") is False,
  }
  if not all(checks.values()):
    raise ValueError(
      "Q8 producer evidence failed checks: "
      f"{sorted(key for key, passed in checks.items() if not passed)!r}")
  diagnostics = _mapping(
    row.get("fixture_diagnostic"), "Q8 fixture diagnostic")
  if set(diagnostics) != {
      "q8_values_exact", "q8_scales_numeric_match",
      "q8_sums_numeric_match", "observed_sha256", "fixture_sha256",
      "scales_comparison", "sums_comparison"}:
    raise ValueError("Q8 fixture diagnostic fields differ")
  hashes = _mapping(
    row.get("consumer_reference_q8_sha256"),
    "consumer reference Q8 hashes")
  if set(hashes) != {"values", "scales", "sums"}:
    raise ValueError("consumer reference Q8 hash fields differ")
  for label, digest in hashes.items():
    _hex_digest(digest, f"consumer reference Q8 {label} hash")
  observed = _mapping(diagnostics["observed_sha256"],
                      "Q8 observed hashes")
  fixture = _mapping(diagnostics["fixture_sha256"],
                     "Q8 fixture hashes")
  for label, group in (("observed", observed), ("fixture", fixture)):
    _exact_keys(group, {"values", "scales", "sums"}, f"Q8 {label} hashes")
    for name, digest in group.items():
      _hex_digest(digest, f"Q8 {label} {name} hash")
  if dict(observed) != dict(hashes):
    raise ValueError("Q8 observed hashes differ from consumer hashes")
  scales = _validate_numeric_comparison(
    diagnostics["scales_comparison"], "Q8 scale comparison")
  sums = _validate_numeric_comparison(
    diagnostics["sums_comparison"], "Q8 sum comparison")
  if diagnostics["q8_values_exact"] is not \
       (observed["values"] == fixture["values"]) or \
     diagnostics["q8_scales_numeric_match"] is not \
       (scales["status"] == "pass") or \
     diagnostics["q8_sums_numeric_match"] is not \
       (sums["status"] == "pass"):
    raise ValueError("Q8 fixture diagnostic booleans differ")
  return row


def _admit_candidate_predecessor(request: CandidatePrefixRequest) -> dict[str, Any] | None:
  queue_mode, prefix_epochs = _queue(request.queue_mode), \
    _prefix(request.prefix_epochs)
  if prefix_epochs == 1:
    if request.prior_evidence is not None:
      raise ValueError("prefix-1 must not depend on later candidate evidence")
    if queue_mode == "PM4":
      if request.cross_queue_admission is not None:
        raise ValueError("first PM4 prefix-1 has no cross-queue prerequisite")
      return None
    # AQL is never the first target dispatch: require persisted complete PM4.
    pm4 = validate_candidate_prefix_evidence(
      _load_frozen_stage(
        request.cross_queue_admission, operation_schema=CANDIDATE_SCHEMA,
        queue_mode="PM4"),
      queue_mode="PM4", prefix_epochs=20)
    return pm4
  expected = 1 if prefix_epochs == 3 else 3
  prior = validate_candidate_prefix_evidence(
    _load_frozen_stage(
      request.prior_evidence, operation_schema=CANDIDATE_SCHEMA,
      queue_mode=queue_mode),
    queue_mode=queue_mode, prefix_epochs=expected)
  if queue_mode == "AQL":
    validate_candidate_prefix_evidence(
      _load_frozen_stage(
        request.cross_queue_admission, operation_schema=CANDIDATE_SCHEMA,
        queue_mode="PM4"),
      queue_mode="PM4", prefix_epochs=20)
  elif request.cross_queue_admission is not None:
    raise ValueError("PM4 candidate escalation has no cross-queue prerequisite")
  return prior


def _candidate_attestation(
    runtime: CandidatePrefixRuntime, invocation: Any,
    ) -> dict[str, Any]:
  observed = runtime.session.attest_post_sync(invocation, runtime.queue_mode)
  fields = {
    key: (
      list(getattr(observed, key, ())) if key == "fixed_five_vas" else
      getattr(observed, key, None))
    for key in _LOW_LEVEL_ATTESTATION_KEYS}
  expected = {
    "schema": LOW_LEVEL_ATTESTATION_SCHEMA,
    "status": "PASS", "queue_mode": runtime.queue_mode,
    "family_identity": runtime.family_identity,
    "candidate_executable_identity":
      runtime.candidate_executable_identity,
    "input_identity": runtime.input_identity,
    "program_key": runtime.program_key,
    "binary_sha256": runtime.binary_sha256,
    "launch_count": runtime.prefix_epochs,
  }
  if any(fields.get(key) != value for key, value in expected.items()):
    raise ValueError("candidate post-sync attestation differs")
  payload = {
    key: value for key, value in fields.items()
    if key != "observation_identity"}
  if fields["observation_identity"] != _identity(payload):
    raise ValueError("candidate observation identity differs")
  return fields


def _candidate_producer_attestation(
    runtime: CandidatePrefixRuntime,
    ) -> dict[str, Any]:
  row = dict(_mapping(
    runtime.producer_attest(runtime.prefix_epochs), "Q8 producer evidence"))
  return _candidate_producer_evidence(row, {
    "queue_mode": runtime.queue_mode,
    "prefix_epochs": runtime.prefix_epochs,
    "family_identity": runtime.family_identity,
    "input_identity": runtime.input_identity,
  })


def run_candidate_prefix_child(request: CandidatePrefixRequest) -> dict[str, Any]:
  """One candidate prefix in one queue-selected fresh child."""
  queue_mode, prefix_epochs = _queue(request.queue_mode), \
    _prefix(request.prefix_epochs)
  os.environ.update({
    "AMD_AQL": "1" if queue_mode == "AQL" else "0", "DEV": "AMD"})
  base = {
    "schema": CANDIDATE_SCHEMA, "queue_mode": queue_mode,
    "prefix_epochs": prefix_epochs, "status": "BLOCKED",
    "exact_blocker": None, "no_retry": True, "retry_count": 0,
    "no_fallback": True, "compile_performed": False,
    "requires_recompile": False, "promotion_evidence_eligible": False,
    "config_identity": _identity(dict(request.config)),
    "request_identity": _candidate_request_identity(request),
  }
  attempt: dict[str, Any] = {"phase": "admission"}
  try:
    predecessor = _admit_candidate_predecessor(request)
    attempt["phase"] = "runtime_construction"
    runtime = request.runtime_builder(
      dict(_mapping(request.config, "candidate config")),
      queue_mode=queue_mode, prefix_epochs=prefix_epochs)
    if not isinstance(runtime, CandidatePrefixRuntime):
      raise TypeError("candidate builder returned no typed runtime")
    runtime.validate(queue_mode, prefix_epochs)
    if predecessor is not None and any(
        getattr(runtime, field) != predecessor[field] for field in (
          "family_identity", "fixture_identity", "workload_identity",
          "input_identity", "logical_q4_identity",
          "resident_fp16_activation_identity",
          "candidate_executable_identity", "program_key", "binary_sha256")):
      raise ValueError("candidate runtime differs from persisted predecessor")
    attempt.update({
      "phase": "invocation", "family_identity": runtime.family_identity,
      "fixture_identity": runtime.fixture_identity,
      "workload_identity": runtime.workload_identity,
      "input_identity": runtime.input_identity,
      "logical_q4_identity": runtime.logical_q4_identity,
      "resident_fp16_activation_identity":
        runtime.resident_fp16_activation_identity,
      "candidate_executable_identity":
        runtime.candidate_executable_identity,
      "c4_canary_identity": runtime.c4_canary_identity,
    })
    invocation = runtime.session.invoke(prefix_epochs=prefix_epochs)
    output = getattr(invocation, "output", None)
    if output is None:
      raise TypeError("candidate invocation omitted output")
    runtime.synchronize()
    attempt["phase"] = "post_sync_attestation"
    attestation = _candidate_attestation(runtime, invocation)
    attempt["attestation"] = attestation
    attempt["phase"] = "producer_readback"
    producer = _candidate_producer_attestation(runtime)
    attempt["q8_producer"] = producer
    attempt["phase"] = "output_readback_and_comparison"
    got, reference = runtime.readback(output), runtime.reference(prefix_epochs)
    attempt.update({
      "comparison": dict(_mapping(
        runtime.comparator(got, reference), "raw candidate comparison")),
      "output_sha256": _array_sha256(got),
      "reference_sha256": _array_sha256(reference)})
    comparison = _validate_full_comparison(
      attempt["comparison"], "candidate comparison")
    attempt["phase"] = "complete"
    payload = {
      **base, "status": "PASS", "exact_blocker": None,
      "gate": "C6" if prefix_epochs == 20 else "C5",
      "family_identity": runtime.family_identity,
      "fixture_identity": runtime.fixture_identity,
      "workload_identity": runtime.workload_identity,
      "input_identity": runtime.input_identity,
      "logical_q4_identity": runtime.logical_q4_identity,
      "resident_fp16_activation_identity":
        runtime.resident_fp16_activation_identity,
      "candidate_executable_identity":
        runtime.candidate_executable_identity,
      "program_key": runtime.program_key,
      "binary_sha256": runtime.binary_sha256,
      "c4_canary_identity": runtime.c4_canary_identity,
      "predecessor_evidence_identity":
        None if predecessor is None else (
          request.cross_queue_admission.envelope_evidence_identity
          if prefix_epochs == 1 else
          request.prior_evidence.envelope_evidence_identity),
      "q8_producer": producer, "attestation": attestation,
      "consumer_reference_q8_sha256":
        producer["consumer_reference_q8_sha256"],
      "comparison": comparison, "output_sha256": _array_sha256(got),
      "reference_sha256": _array_sha256(reference),
      "post_sync_before_readback": True, "readback_performed": True,
      "attempt": attempt,
    }
  except BaseException as exc:
    low_level_failure = getattr(
      exc, LOW_LEVEL_INVOCATION_FAILURE_ATTR, None)
    if isinstance(low_level_failure, Mapping):
      low_level_failure = dict(low_level_failure)
      attempt["invocation_failure"] = low_level_failure
      if low_level_failure.get("schema") == \
           LOW_LEVEL_INVOCATION_FAILURE_SCHEMA:
        attempt["invocation_subphase"] = low_level_failure.get("subphase")
    payload = {
      **base, "exact_blocker":
        f"{queue_mode} prefix-{prefix_epochs} failed closed: "
        f"{type(exc).__name__}: {exc}",
      "exception": type(exc).__name__,
      "failed_attempt": attempt,
    }
  return {**payload, "evidence_identity": _identity(payload)}


def run_pm4_no_doorbell_child(
    request: PM4NoDoorbellRequest,
    ) -> dict[str, Any]:
  """Capture exactly one target PM4 command without submitting it."""
  if not isinstance(request, PM4NoDoorbellRequest):
    raise TypeError("PM4 no-doorbell child requires a typed request")
  environment = {"DEV": "AMD", "AMD_AQL": "0", "PROFILE": "0"}
  os.environ.update(environment)
  base = {
    "schema": PM4_NO_DOORBELL_SCHEMA, "status": "BLOCKED",
    "exact_blocker": None, "queue_mode": "PM4", "prefix_epochs": 1,
    "submit_policy": "snapshot_only", "no_retry": True, "retry_count": 0,
    "no_fallback": True, "compile_performed": False,
    "requires_recompile": False, "promotion_evidence_eligible": False,
    "config_identity": _identity(dict(request.config)),
    "request_identity": _pm4_no_doorbell_request_identity(request),
    "environment": environment,
  }
  receipts: list[Mapping[str, Any]] = []
  attempt: dict[str, Any] = {
    "phase": "runtime_construction", "invocation_count": 0,
    "receipt_count": 0, "receipt": None}
  try:
    runtime = request.runtime_builder(
      dict(_mapping(request.config, "PM4 no-doorbell config")),
      queue_mode="PM4", prefix_epochs=1,
      pm4_submit_policy="snapshot_only",
      pm4_no_doorbell_receipt_sink=receipts)
    if not isinstance(runtime, CandidatePrefixRuntime):
      raise TypeError("PM4 no-doorbell builder returned no typed runtime")
    runtime.validate("PM4", 1)
    attempt["phase"] = "invocation"
    runtime.session.invoke(prefix_epochs=1)
    attempt["invocation_count"] = 1
    attempt["receipt_count"] = len(receipts)
    if len(receipts) != 1:
      raise ValueError(
        "PM4 no-doorbell invocation requires exactly one harness receipt")
    receipt = _validate_pm4_no_doorbell_receipt(receipts[0])
    attempt.update({
      "phase": "complete", "receipt_count": 1, "receipt": receipt})
    payload = {
      **base, "status": "PASS",
      "family_identity": runtime.family_identity,
      "fixture_identity": runtime.fixture_identity,
      "workload_identity": runtime.workload_identity,
      "input_identity": runtime.input_identity,
      "logical_q4_identity": runtime.logical_q4_identity,
      "resident_fp16_activation_identity":
        runtime.resident_fp16_activation_identity,
      "candidate_executable_identity":
        runtime.candidate_executable_identity,
      "program_key": runtime.program_key,
      "binary_sha256": runtime.binary_sha256,
      "c4_canary_identity": runtime.c4_canary_identity,
      "invocation_count": 1, "receipt_count": 1, "receipt": receipt,
      "target_dispatch_submitted": False, "native_submit_call_count": 0,
      "timeline_rollback_applied": True, "terminal_child_required": True,
      "readback_performed": False, "numeric_validation_performed": False,
      "attestation_performed": False,
      "producer_attestation_performed": False, "attempt": attempt,
    }
  except BaseException as exc:
    attempt["receipt_count"] = len(receipts)
    invocation_failure = getattr(
      exc, LOW_LEVEL_INVOCATION_FAILURE_ATTR, None)
    if isinstance(invocation_failure, Mapping):
      # Preserve the exact pre-submit snapshot even when a diagnostic
      # authority check rejects the command before producing a receipt.
      attempt["invocation_failure"] = dict(invocation_failure)
    payload = {
      **base, "exact_blocker":
        f"PM4 no-doorbell diagnostic failed closed: "
        f"{type(exc).__name__}: {exc}",
      "exception": type(exc).__name__, "failed_attempt": attempt,
    }
  return {**payload, "evidence_identity": _identity(payload)}


def _ffn_reduced_output_facts(
    got: Any, reference: Any, *,
    diagnostic_global_size: tuple[int, int, int],
    comparator: Callable[[Any, Any], Mapping[str, Any]],
    ) -> dict[str, Any]:
  grid = _ffn_reduced_global_size(diagnostic_global_size)
  gx, gy, _ = grid
  rows, cols = gy * 128, gx * 128
  got_array = np.ascontiguousarray(np.asarray(got))
  reference_array = np.ascontiguousarray(np.asarray(reference))
  if got_array.shape != OUTPUT_SHAPE or reference_array.shape != OUTPUT_SHAPE:
    raise ValueError("FFN reduced-grid output/reference shape differs")
  touched = dict(_mapping(
    comparator(
      np.ascontiguousarray(got_array[:rows, :cols]),
      np.ascontiguousarray(reference_array[:rows, :cols])),
    "raw FFN reduced-grid comparison"))
  untouched_mask = np.ones(OUTPUT_SHAPE, dtype=np.bool_)
  untouched_mask[:rows, :cols] = False
  untouched_nonzero = untouched_mask & (got_array != 0)
  untouched_nonzero_count = int(np.count_nonzero(untouched_nonzero))
  first_index = None
  if untouched_nonzero_count:
    first_index = [
      int(item) for item in np.unravel_index(
        int(np.argmax(untouched_nonzero)), OUTPUT_SHAPE)]
  first_value = (
    None if first_index is None else
    repr(got_array[tuple(first_index)].item()))
  return {
    "touched_shape": [rows, cols],
    "untouched_element_count": int(untouched_mask.sum()),
    "untouched_nonzero_count": untouched_nonzero_count,
    "first_untouched_nonzero_index": first_index,
    "first_untouched_nonzero_value": first_value,
    "touched_comparison": touched,
    "output_sha256": _array_sha256(got_array),
    "reference_sha256": _array_sha256(reference_array),
  }


def run_ffn_reduced_grid_child(
    request: FFNReducedGridRequest,
    ) -> dict[str, Any]:
  """Execute and check exactly one allowlisted reduced FFN PM4 grid."""
  if not isinstance(request, FFNReducedGridRequest):
    raise TypeError("FFN reduced-grid child requires a typed request")
  grid = _ffn_reduced_global_size(request.diagnostic_global_size)
  environment = {"DEV": "AMD", "AMD_AQL": "0", "PROFILE": "0"}
  os.environ.update(environment)
  gx, gy, _ = grid
  touched_shape = [gy * 128, gx * 128]
  touched_elements = math.prod(touched_shape)
  base = {
    "schema": FFN_REDUCED_GRID_SCHEMA, "status": "BLOCKED",
    "exact_blocker": None, "queue_mode": "PM4", "prefix_epochs": 1,
    "diagnostic_global_size": list(grid),
    "frozen_global_size": list(FFN_FROZEN_GLOBAL_SIZE),
    "local_size": list(FFN_FROZEN_LOCAL_SIZE),
    "touched_shape": touched_shape,
    "touched_element_count": touched_elements,
    "untouched_element_count": OUTPUT_ELEMENTS - touched_elements,
    "no_retry": True, "retry_count": 0, "no_fallback": True,
    "compile_performed": False, "requires_recompile": False,
    "promotion_evidence_eligible": False,
    "ordinary_attestation_performed": False,
    "full_grid_validation_performed": False,
    "full_prefix1_reference_constructed": False,
    "target_dispatch_submitted": False,
    "target_submit_entered": False, "target_submit_returned": False,
    "invocation_count": 0, "receipt_count": 0,
    "readback_performed": False, "numeric_validation_performed": False,
    "producer_attestation_performed": False,
    "config_identity": _identity(dict(request.config)),
    "request_identity": _ffn_reduced_grid_request_identity(request),
    "environment": environment,
  }
  receipts: list[Mapping[str, Any]] = []
  submit_facts: dict[str, Any] = {
    "target_submit_entered": False,
    "target_submit_returned": False,
    "target_dispatch_submitted": False,
  }
  attempt: dict[str, Any] = {
    "phase": "runtime_construction", "invocation_count": 0,
    "receipt_count": 0}
  try:
    runtime = request.runtime_builder(
      dict(_mapping(request.config, "FFN reduced-grid config")),
      queue_mode="PM4", prefix_epochs=1,
      diagnostic_global_size=grid,
      diagnostic_dispatch_receipt_sink=receipts)
    if not isinstance(runtime, CandidatePrefixRuntime):
      raise TypeError("FFN reduced-grid builder returned no typed runtime")
    runtime.validate("PM4", 1)
    invoke_diagnostic = getattr(
      runtime.session, "invoke_diagnostic_one_epoch", None)
    complete_diagnostic = getattr(
      runtime.session, "complete_diagnostic_post_sync", None)
    if not callable(invoke_diagnostic) or not callable(complete_diagnostic):
      raise TypeError(
        "FFN reduced-grid runtime requires diagnostic low-level methods")
    attempt["phase"] = "diagnostic_invocation"
    submit_facts["target_dispatch_submitted"] = None
    invocation = invoke_diagnostic(grid)
    submit_facts = {
      "target_submit_entered": True,
      "target_submit_returned": True,
      "target_dispatch_submitted": True,
    }
    output = getattr(invocation, "output", None)
    if output is None:
      raise TypeError("FFN reduced-grid invocation omitted output")
    attempt["invocation_count"] = 1
    if receipts:
      raise ValueError(
        "FFN reduced-grid receipt appeared before post-dispatch sync")
    runtime.synchronize()
    attempt["phase"] = "diagnostic_post_sync_receipt"
    completed = complete_diagnostic(invocation, "PM4")
    attempt["receipt_count"] = len(receipts)
    if len(receipts) != 1:
      raise ValueError(
        "FFN reduced-grid invocation requires exactly one diagnostic receipt")
    receipt = _validate_ffn_reduced_receipt(
      completed, runtime=runtime, diagnostic_global_size=grid)
    sink_receipt = _validate_ffn_reduced_receipt(
      receipts[0], runtime=runtime, diagnostic_global_size=grid)
    if sink_receipt != receipt:
      raise ValueError(
        "FFN reduced-grid returned/sink diagnostic receipts differ")
    attempt["diagnostic_receipt"] = receipt
    attempt["phase"] = "producer_readback"
    producer = _candidate_producer_attestation(runtime)
    attempt["q8_producer"] = producer
    attempt["phase"] = "output_readback_and_reduced_comparison"
    got = runtime.readback(output)
    reference = runtime.reference(1)
    facts = _ffn_reduced_output_facts(
      got, reference, diagnostic_global_size=grid,
      comparator=runtime.comparator)
    attempt.update(facts)
    comparison = _validate_ffn_reduced_comparison(
      facts["touched_comparison"], diagnostic_global_size=grid)
    if facts["untouched_nonzero_count"] != 0:
      raise ValueError(
        "FFN reduced-grid output modified an untouched sentinel element")
    attempt["phase"] = "complete"
    payload = {
      **base, "status": "PASS", "exact_blocker": None,
      "full_prefix1_reference_constructed": True,
      "target_dispatch_submitted": True,
      "target_submit_entered": True, "target_submit_returned": True,
      "invocation_count": 1, "receipt_count": 1,
      "readback_performed": True, "numeric_validation_performed": True,
      "producer_attestation_performed": True,
      "family_identity": runtime.family_identity,
      "fixture_identity": runtime.fixture_identity,
      "workload_identity": runtime.workload_identity,
      "input_identity": runtime.input_identity,
      "logical_q4_identity": runtime.logical_q4_identity,
      "resident_fp16_activation_identity":
        runtime.resident_fp16_activation_identity,
      "candidate_executable_identity":
        runtime.candidate_executable_identity,
      "program_key": runtime.program_key,
      "binary_sha256": runtime.binary_sha256,
      "c4_canary_identity": runtime.c4_canary_identity,
      "q8_producer": producer,
      "consumer_reference_q8_sha256":
        producer["consumer_reference_q8_sha256"],
      "diagnostic_receipt": receipt,
      "touched_comparison": comparison,
      "untouched_nonzero_count": 0,
      "first_untouched_nonzero_index": None,
      "first_untouched_nonzero_value": None,
      "output_sha256": facts["output_sha256"],
      "reference_sha256": facts["reference_sha256"],
      "attempt": attempt,
    }
  except BaseException as exc:
    attempt["receipt_count"] = len(receipts)
    low_level_failure = getattr(
      exc, LOW_LEVEL_INVOCATION_FAILURE_ATTR, None)
    if isinstance(low_level_failure, Mapping):
      attempt["invocation_failure"] = dict(low_level_failure)
      dispatch_failure = low_level_failure.get("dispatch_failure")
      if isinstance(dispatch_failure, Mapping):
        observed_submit = dispatch_failure.get("submit_evidence")
        if isinstance(observed_submit, Mapping):
          submit_facts = {
            "target_submit_entered":
              observed_submit.get("target_submit_entered"),
            "target_submit_returned":
              observed_submit.get("target_submit_returned"),
            "target_dispatch_submitted":
              observed_submit.get("target_dispatch_submitted"),
          }
    payload = {
      **base, **submit_facts, "exact_blocker":
        "PM4 FFN reduced-grid diagnostic failed closed: "
        f"{type(exc).__name__}: {exc}",
      "exception": type(exc).__name__, "failed_attempt": attempt,
    }
  return {**payload, "evidence_identity": _identity(payload)}


@dataclass(frozen=True)
class DirectCorrectnessRuntime:
  queue_mode: str
  family_identity: str
  fixture_identity: str
  workload_identity: str
  input_identity: str
  logical_q4_identity: str
  resident_fp16_activation_identity: str
  c4_canary_identity: str
  bindings: Any
  capture: Any
  invoke_lazy: Callable[[], Any]
  synchronize: Callable[[], None]
  readback: Callable[[Any], Any]
  dense_reference: Callable[[], Any]
  comparator: Callable[[Any, Any], Mapping[str, Any]]

  def validate(self, queue_mode: str) -> "DirectCorrectnessRuntime":
    if self.queue_mode != _queue(queue_mode):
      raise ValueError("direct runtime queue differs")
    for value, label in (
        (self.family_identity, "family identity"),
        (self.fixture_identity, "fixture identity"),
        (self.workload_identity, "workload identity"),
        (self.input_identity, "input identity"),
        (self.logical_q4_identity, "logical Q4 identity"),
        (self.resident_fp16_activation_identity,
         "resident FP16 activation identity"),
        (self.c4_canary_identity, "C4 canary identity")):
      _content_identity(value, label)
    if getattr(self.bindings, "queue_mode", None) != queue_mode or \
       not callable(getattr(self.capture, "realize_output", None)) or \
       not callable(getattr(self.capture, "observation_post_sync", None)):
      raise TypeError("direct runtime lacks real executable capture")
    for callback in (
        self.invoke_lazy, self.synchronize, self.readback,
        self.dense_reference, self.comparator):
      if not callable(callback):
        raise TypeError("direct runtime callback differs")
    return self


@dataclass(frozen=True)
class DirectCorrectnessRequest:
  config: Mapping[str, Any]
  queue_mode: str
  candidate_full_evidence: FrozenCorrectnessEvidenceRef
  runtime_builder: Callable[..., DirectCorrectnessRuntime]


def _direct_request_identity(request: DirectCorrectnessRequest) -> str:
  return _identity({
    "schema": f"{SCHEMA}.direct_request",
    "queue_mode": request.queue_mode,
    "config_identity": _identity(dict(request.config)),
    "candidate_full_evidence_identity":
      getattr(request.candidate_full_evidence, "evidence_identity", None),
  })


def validate_direct_evidence(value: Any, *, queue_mode: str | None = None) -> dict[str, Any]:
  row = dict(_mapping(value, "direct correctness evidence"))
  _exact_keys(row, {
    "schema", "queue_mode", "status", "exact_blocker", "no_retry",
    "retry_count", "no_fallback", "promotion_evidence_eligible",
    "config_identity", "request_identity", "family_identity",
    "fixture_identity", "workload_identity", "input_identity",
    "logical_q4_identity", "resident_fp16_activation_identity",
    "c4_canary_identity", "candidate_full_evidence_identity",
    "executable_observation", "executable_evidence",
    "direct_executable_identity", "comparison_authority", "comparison",
    "output_sha256", "reference_sha256",
    "post_sync_before_observation_and_readback", "readback_performed",
    "attempt", "evidence_identity",
  }, "direct correctness evidence")
  if not _identity_valid(row) or row.get("schema") != DIRECT_SCHEMA or \
     row.get("status") != "PASS" or \
     row.get("exact_blocker") is not None or row.get("retry_count") != 0 or \
     row.get("promotion_evidence_eligible") is not False or \
     row.get("no_retry") is not True or row.get("no_fallback") is not True:
    raise ValueError("direct correctness evidence identity/state differs")
  queue = _queue(row.get("queue_mode"))
  if queue_mode is not None and queue != queue_mode:
    raise ValueError("direct correctness evidence queue differs")
  for field in (
      "family_identity", "fixture_identity", "workload_identity",
      "input_identity", "logical_q4_identity",
      "resident_fp16_activation_identity", "c4_canary_identity",
      "direct_executable_identity", "request_identity", "config_identity",
      "candidate_full_evidence_identity"):
    _content_identity(row.get(field), f"direct {field}")
  if row.get("comparison_authority") != \
       "independent_dense_fp16_activation_q4k_dequant_oracle_v1" or \
     row.get("post_sync_before_observation_and_readback") is not True or \
     row.get("readback_performed") is not True:
    raise ValueError("direct oracle/readback authority differs")
  _validate_full_comparison(row.get("comparison"), "direct comparison")
  for field in ("output_sha256", "reference_sha256"):
    _hex_digest(row.get(field), f"direct {field}")
  observation = _mapping(
    row.get("executable_observation"), "direct executable observation")
  observation_payload = {
    key: item for key, item in observation.items()
    if key != "observation_identity"}
  if observation.get("observation_identity") != _identity(
      observation_payload) or \
     observation.get("status") != "PASS" or \
     observation.get("queue_mode") != row["queue_mode"] or \
     observation.get("runtime_cache_join_verified") is not True or \
     observation.get("post_sync_attestation") is not True:
    raise ValueError("direct executable observation differs")
  manifest = _mapping(observation.get("manifest"), "direct manifest")
  manifest_payload = {
    key: item for key, item in manifest.items()
    if key != "executable_identity"}
  if manifest.get("executable_identity") != _identity(manifest_payload) or \
     manifest["executable_identity"] != row["direct_executable_identity"]:
    raise ValueError("direct executable manifest identity differs")
  executable = _mapping(
    row.get("executable_evidence"), "direct executable evidence")
  executable_payload = {
    key: item for key, item in executable.items()
    if key != "evidence_identity"}
  if executable.get("evidence_identity") != _identity(executable_payload) or \
     executable.get("status") != "PASS" or \
     executable.get("queue_mode") != row["queue_mode"] or \
     executable.get("input_identity") != row["input_identity"] or \
     executable.get("workload_identity") != row["workload_identity"] or \
     executable.get("executable_identity") != \
       row["direct_executable_identity"]:
    raise ValueError("direct executable evidence differs")
  attempt = _mapping(row.get("attempt"), "direct attempt")
  _exact_keys(attempt, {
    "phase", "executable_observation", "executable_evidence", "comparison",
    "output_sha256", "reference_sha256",
  }, "direct attempt")
  if attempt.get("phase") != "complete" or \
     attempt.get("executable_observation") != observation or \
     attempt.get("executable_evidence") != executable or \
     attempt.get("comparison") != row["comparison"] or \
     attempt.get("output_sha256") != row["output_sha256"] or \
     attempt.get("reference_sha256") != row["reference_sha256"]:
    raise ValueError("direct nested attempt differs")
  return row


def run_direct_correctness_child(request: DirectCorrectnessRequest) -> dict[str, Any]:
  """One direct-packed full-role invocation with independent oracle/capture."""
  queue_mode = _queue(request.queue_mode)
  os.environ.update({
    "AMD_AQL": "1" if queue_mode == "AQL" else "0", "DEV": "AMD"})
  base = {
    "schema": DIRECT_SCHEMA, "queue_mode": queue_mode, "status": "BLOCKED",
    "exact_blocker": None, "no_retry": True, "retry_count": 0,
    "no_fallback": True, "promotion_evidence_eligible": False,
    "config_identity": _identity(dict(request.config)),
    "request_identity": _direct_request_identity(request),
  }
  attempt: dict[str, Any] = {"phase": "admission"}
  try:
    candidate = validate_candidate_prefix_evidence(
      _load_frozen_stage(
        request.candidate_full_evidence, operation_schema=CANDIDATE_SCHEMA,
        queue_mode=queue_mode),
      queue_mode=queue_mode,
      prefix_epochs=20)
    runtime = request.runtime_builder(
      dict(_mapping(request.config, "direct config")),
      queue_mode=queue_mode)
    if not isinstance(runtime, DirectCorrectnessRuntime):
      raise TypeError("direct builder returned no typed runtime")
    runtime.validate(queue_mode)
    if any(getattr(runtime, field) != candidate[field] for field in (
        "family_identity", "fixture_identity", "workload_identity",
        "input_identity", "logical_q4_identity",
        "resident_fp16_activation_identity")):
      raise ValueError("direct runtime differs from candidate full-role inputs")
    attempt["phase"] = "invocation"
    runtime.synchronize()
    output = runtime.invoke_lazy()
    if output is None:
      raise TypeError("production direct-packed route returned no output")
    runtime.capture.realize_output(output)
    runtime.synchronize()
    attempt["phase"] = "post_sync_executable_observation"
    observation = runtime.capture.observation_post_sync(output, queue_mode)
    from extra.qk.direct_packed_executable_attestor import \
      build_direct_packed_fallback_evidence
    executable = build_direct_packed_fallback_evidence(
      observation, runtime.bindings)
    attempt["executable_observation"] = observation
    attempt["executable_evidence"] = executable
    attempt["phase"] = "output_readback_and_dense_comparison"
    got, reference = runtime.readback(output), runtime.dense_reference()
    attempt.update({
      "comparison": dict(_mapping(
        runtime.comparator(got, reference), "raw direct comparison")),
      "output_sha256": _array_sha256(got),
      "reference_sha256": _array_sha256(reference)})
    comparison = _validate_full_comparison(
      attempt["comparison"], "independent direct comparison")
    attempt["phase"] = "complete"
    payload = {
      **base, "status": "PASS", "exact_blocker": None,
      "family_identity": runtime.family_identity,
      "fixture_identity": runtime.fixture_identity,
      "workload_identity": runtime.workload_identity,
      "input_identity": runtime.input_identity,
      "logical_q4_identity": runtime.logical_q4_identity,
      "resident_fp16_activation_identity":
        runtime.resident_fp16_activation_identity,
      "c4_canary_identity": runtime.c4_canary_identity,
      "candidate_full_evidence_identity":
        request.candidate_full_evidence.envelope_evidence_identity,
      "executable_observation": observation,
      "executable_evidence": executable,
      "direct_executable_identity": executable["executable_identity"],
      "comparison_authority":
        "independent_dense_fp16_activation_q4k_dequant_oracle_v1",
      "comparison": comparison, "output_sha256": _array_sha256(got),
      "reference_sha256": _array_sha256(reference),
      "post_sync_before_observation_and_readback": True,
      "readback_performed": True,
      "attempt": attempt,
    }
  except BaseException as exc:
    payload = {
      **base, "exact_blocker":
        f"{queue_mode} direct correctness failed closed: "
        f"{type(exc).__name__}: {exc}",
      "exception": type(exc).__name__,
      "failed_attempt": attempt,
    }
  return {**payload, "evidence_identity": _identity(payload)}


@dataclass(frozen=True)
class TransitionRequest:
  config: Mapping[str, Any]
  queue_mode: str
  sequence_name: str
  candidate_full_evidence: FrozenCorrectnessEvidenceRef
  direct_evidence: FrozenCorrectnessEvidenceRef
  worker: Callable[..., Mapping[str, Any]]


def _transition_request_identity(request: TransitionRequest) -> str:
  return _identity({
    "schema": f"{SCHEMA}.transition_request",
    "queue_mode": request.queue_mode,
    "sequence_name": request.sequence_name,
    "config_identity": _identity(dict(request.config)),
    "candidate_full_evidence_identity":
      getattr(request.candidate_full_evidence, "evidence_identity", None),
    "direct_evidence_identity":
      getattr(request.direct_evidence, "evidence_identity", None),
  })


def validate_transition_evidence(
    value: Any, *, queue_mode: str | None = None,
    sequence_name: str | None = None,
    ) -> dict[str, Any]:
  row = dict(_mapping(value, "transition evidence"))
  _exact_keys(row, {
    "schema", "queue_mode", "sequence_name", "status", "exact_blocker",
    "no_retry", "retry_count", "no_fallback",
    "promotion_evidence_eligible", "config_identity", "request_identity",
    "family_identity", "fixture_identity", "workload_identity",
    "input_identity", "logical_q4_identity",
    "resident_fp16_activation_identity",
    "candidate_full_evidence_identity", "direct_evidence_identity",
    "sequence", "raw_transition", "attempt", "evidence_identity",
  }, "transition evidence")
  if not _identity_valid(row) or row.get("schema") != TRANSITION_SCHEMA or \
     row.get("status") != "PASS" or \
     row.get("exact_blocker") is not None or row.get("retry_count") != 0 or \
     row.get("promotion_evidence_eligible") is not False or \
     row.get("no_retry") is not True or row.get("no_fallback") is not True:
    raise ValueError("transition evidence identity/state differs")
  queue = _queue(row.get("queue_mode"))
  name = row.get("sequence_name")
  if name not in TRANSITION_SEQUENCES or \
     queue_mode is not None and queue != queue_mode or \
     sequence_name is not None and name != sequence_name:
    raise ValueError("transition queue/sequence differs")
  expected_sequence = [
    {"route": route, "prefix_epochs": prefix}
    for route, prefix in TRANSITION_SEQUENCES[name]]
  if row.get("sequence") != expected_sequence:
    raise ValueError("transition ordered sequence differs")
  for field in (
      "family_identity", "fixture_identity", "workload_identity",
      "input_identity", "logical_q4_identity",
      "resident_fp16_activation_identity",
      "candidate_full_evidence_identity", "direct_evidence_identity",
      "request_identity", "config_identity"):
    _content_identity(row.get(field), f"transition {field}")
  raw = dict(_mapping(row.get("raw_transition"), "raw transition"))
  _exact_keys(raw, {
    "status", "queue_mode", "sequence", "all_outputs_correct",
    "post_route_sync_each_step", "steps", "evidence_identity",
  }, "raw transition")
  if not _identity_valid(raw) or raw.get("status") != "PASS" or \
     raw.get("queue_mode") != queue or \
     raw.get("sequence") != expected_sequence or \
     raw.get("all_outputs_correct") is not True or \
     raw.get("post_route_sync_each_step") is not True:
    raise ValueError("nested transition worker evidence differs")
  steps = raw.get("steps")
  if not isinstance(steps, list) or len(steps) != len(expected_sequence):
    raise ValueError("transition step evidence differs")
  for index, (step, expected) in enumerate(zip(steps, expected_sequence)):
    if not isinstance(step, Mapping):
      raise ValueError(f"transition step {index} differs")
    _exact_keys(step, {
      "ordinal", "route", "prefix_epochs", "status", "comparison_status",
      "mismatch_count", "post_route_sync", "evidence_identity",
    }, f"transition step {index}")
    if not _identity_valid(step) or \
       step.get("ordinal") != index or step.get("route") != \
         expected["route"] or step.get("prefix_epochs") != \
         expected["prefix_epochs"] or step.get("status") != "PASS" or \
       step.get("comparison_status") != "pass" or \
       step.get("mismatch_count") != 0 or \
       step.get("post_route_sync") is not True:
      raise ValueError(f"transition step {index} differs")
  attempt = _mapping(row.get("attempt"), "transition attempt")
  _exact_keys(attempt, {"phase", "raw_transition"}, "transition attempt")
  if attempt.get("phase") != "complete" or \
     attempt.get("raw_transition") != raw:
    raise ValueError("transition nested attempt differs")
  return row


def run_transition_child(request: TransitionRequest) -> dict[str, Any]:
  """One named transition sequence in one otherwise fresh child."""
  queue_mode, name = _queue(request.queue_mode), request.sequence_name
  os.environ.update({
    "AMD_AQL": "1" if queue_mode == "AQL" else "0", "DEV": "AMD"})
  base = {
    "schema": TRANSITION_SCHEMA, "queue_mode": queue_mode,
    "sequence_name": name, "status": "BLOCKED", "exact_blocker": None,
    "no_retry": True, "retry_count": 0, "no_fallback": True,
    "promotion_evidence_eligible": False,
    "config_identity": _identity(dict(request.config)),
    "request_identity": _transition_request_identity(request),
  }
  attempt: dict[str, Any] = {"phase": "admission"}
  try:
    if name not in TRANSITION_SEQUENCES:
      raise ValueError("transition sequence name differs")
    candidate = validate_candidate_prefix_evidence(
      _load_frozen_stage(
        request.candidate_full_evidence, operation_schema=CANDIDATE_SCHEMA,
        queue_mode=queue_mode),
      queue_mode=queue_mode, prefix_epochs=20)
    direct = validate_direct_evidence(
      _load_frozen_stage(
        request.direct_evidence, operation_schema=DIRECT_SCHEMA,
        queue_mode=queue_mode),
      queue_mode=queue_mode)
    if candidate["family_identity"] != direct["family_identity"] or \
       candidate["input_identity"] != direct["input_identity"] or \
       direct["candidate_full_evidence_identity"] != \
         request.candidate_full_evidence.envelope_evidence_identity:
      raise ValueError("transition candidate/direct authorities differ")
    attempt["phase"] = "worker"
    raw = dict(_mapping(request.worker(
      dict(_mapping(request.config, "transition config")),
      queue_mode=queue_mode, sequence=TRANSITION_SEQUENCES[name],
      candidate_evidence=candidate, direct_evidence=direct),
      "transition worker result"))
    if not _identity_valid(raw) or raw.get("status") != "PASS" or \
       raw.get("queue_mode") != queue_mode or \
       raw.get("sequence") != [
         {"route": route, "prefix_epochs": prefix}
         for route, prefix in TRANSITION_SEQUENCES[name]] or \
       raw.get("all_outputs_correct") is not True or \
       raw.get("post_route_sync_each_step") is not True:
      raise ValueError("transition worker evidence differs")
    attempt.update({"phase": "complete", "raw_transition": raw})
    payload = {
      **base, "status": "PASS", "exact_blocker": None,
      "family_identity": candidate["family_identity"],
      "fixture_identity": candidate["fixture_identity"],
      "workload_identity": candidate["workload_identity"],
      "input_identity": candidate["input_identity"],
      "logical_q4_identity": candidate["logical_q4_identity"],
      "resident_fp16_activation_identity":
        candidate["resident_fp16_activation_identity"],
      "candidate_full_evidence_identity":
        request.candidate_full_evidence.envelope_evidence_identity,
      "direct_evidence_identity":
        request.direct_evidence.envelope_evidence_identity,
      "sequence": raw["sequence"], "raw_transition": raw,
      "attempt": attempt,
    }
  except BaseException as exc:
    payload = {
      **base, "exact_blocker":
        f"{queue_mode} transition {name!r} failed closed: "
        f"{type(exc).__name__}: {exc}",
      "exception": type(exc).__name__,
      "failed_attempt": attempt,
    }
  return {**payload, "evidence_identity": _identity(payload)}


def _validate_clear_kernel_fault_evidence(value: Any) -> dict[str, Any]:
  row = dict(_mapping(value, "kernel fault evidence"))
  _exact_keys(row, {
    "schema", "status", "source", "blocks", "relevant_line_count",
    "retained_line_count", "truncated", "limits",
  }, "kernel fault evidence")
  if row != {
      "schema": "tinygrad.amd_kernel_fault_evidence.v1",
      "status": "CLEAR", "source": "kernel_journal_window", "blocks": [],
      "relevant_line_count": 0, "retained_line_count": 0,
      "truncated": False,
      "limits": {
        "max_blocks": 8, "max_lines": 32, "max_line_chars": 512},
      }:
    raise ValueError("kernel fault evidence is not exact CLEAR evidence")
  return row


def validate_guarded_envelope(value: Any) -> dict[str, Any]:
  """Validate the complete containment authority retained between stages."""
  row = dict(_mapping(value, "guarded correctness envelope"))
  expected_keys = {
    "schema", "status", "exact_blocker", "queue_mode", "operation_schema",
    "health_before", "health_after", "kernel_faults",
    "kernel_fault_evidence", "launched", "spawn_count", "child_status",
    "timed_out", "error", "elapsed_seconds", "result", "no_retry",
    "retry_count", "no_queue_fallback", "promotion_evidence_eligible",
    "request_identity", "config_identity", "evidence_identity",
  }
  if set(row) != expected_keys or not _identity_valid(row) or \
     row.get("schema") != ENVELOPE_SCHEMA or row.get("status") != "PASS" or \
     row.get("exact_blocker") is not None:
    raise ValueError("guarded correctness envelope identity/state differs")
  queue = _queue(row.get("queue_mode"))
  _validate_clear_kernel_fault_evidence(row.get("kernel_fault_evidence"))
  if row.get("health_before") is not True or \
     row.get("health_after") is not True or \
     row.get("kernel_faults") != [] or \
     row.get("launched") is not True or row.get("spawn_count") != 1 or \
     row.get("child_status") != "passed" or \
     row.get("timed_out") is not False or row.get("error") is not None or \
     row.get("no_retry") is not True or row.get("retry_count") != 0 or \
     row.get("no_queue_fallback") is not True or \
     row.get("promotion_evidence_eligible") is not False:
    raise ValueError("guarded correctness envelope containment differs")
  elapsed = row.get("elapsed_seconds")
  if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or \
     not math.isfinite(elapsed) or elapsed < 0:
    raise ValueError("guarded correctness envelope elapsed time differs")
  operation = row.get("operation_schema")
  result = row.get("result")
  if operation == CANDIDATE_SCHEMA:
    child = validate_candidate_prefix_evidence(result, queue_mode=queue)
  elif operation == PM4_NO_DOORBELL_SCHEMA:
    if queue != "PM4":
      raise ValueError("PM4 no-doorbell envelope queue differs")
    child = validate_pm4_no_doorbell_evidence(result)
  elif operation == FFN_REDUCED_GRID_SCHEMA:
    if queue != "PM4":
      raise ValueError("FFN reduced-grid envelope queue differs")
    child = validate_ffn_reduced_grid_evidence(result)
  elif operation == DIRECT_SCHEMA:
    child = validate_direct_evidence(result, queue_mode=queue)
  elif operation == TRANSITION_SCHEMA:
    child = validate_transition_evidence(result, queue_mode=queue)
  else:
    raise ValueError("guarded correctness envelope operation differs")
  for field in ("request_identity", "config_identity"):
    _content_identity(row.get(field), f"guarded envelope {field}")
    if row[field] != child[field]:
      raise ValueError(f"guarded envelope child {field} binding differs")
  return row


def _request_identity(
    expected_schema: str, request: Any,
    ) -> str:
  if expected_schema == CANDIDATE_SCHEMA:
    return _candidate_request_identity(request)
  if expected_schema == PM4_NO_DOORBELL_SCHEMA:
    return _pm4_no_doorbell_request_identity(request)
  if expected_schema == FFN_REDUCED_GRID_SCHEMA:
    return _ffn_reduced_grid_request_identity(request)
  if expected_schema == DIRECT_SCHEMA:
    return _direct_request_identity(request)
  if expected_schema == TRANSITION_SCHEMA:
    return _transition_request_identity(request)
  raise ValueError("guarded request operation differs")


def _guarded_envelope(
    *, child: Callable[[Any], Mapping[str, Any]], request: Any,
    expected_schema: str, queue_mode: str, timeout_seconds: float,
    isolated_runner: Callable[..., Any],
    health_probe: Callable[[Mapping[str, str]], bool],
    fault_collector: Callable[[float], tuple[list[str], Mapping[str, Any]]],
    ) -> dict[str, Any]:
  env = {"AMD_AQL": "1" if queue_mode == "AQL" else "0"}
  started = time.time()
  try: health_before = bool(health_probe(env))
  except BaseException: health_before = False
  isolated, runner_error = None, None
  if health_before:
    try:
      isolated = isolated_runner(
        child, args=(request,), timeout_seconds=timeout_seconds,
        start_method="spawn")
    except BaseException as exc:
      runner_error = f"{type(exc).__name__}: {exc}"
  try: health_after = bool(health_probe(env))
  except BaseException: health_after = False
  try:
    faults, fault_evidence = fault_collector(started)
    faults, fault_evidence = list(faults), dict(fault_evidence)
  except BaseException as exc:
    faults = [f"fault collection failed: {type(exc).__name__}: {exc}"]
    fault_evidence = {}
  result = getattr(isolated, "result", None)
  timed_out = bool(getattr(isolated, "timed_out", False))
  child_status = getattr(isolated, "status", None)
  error = runner_error or getattr(isolated, "error", None)
  blocker = None
  if not health_before: blocker = f"{queue_mode} preflight health failed"
  elif timed_out: blocker = f"{queue_mode} child timed out"
  elif runner_error is not None: blocker = runner_error
  elif child_status != "passed" or not isinstance(result, Mapping):
    blocker = error or f"{queue_mode} child returned no structured result"
  elif faults: blocker = f"{queue_mode} kernel fault/reset marker observed"
  elif not health_after: blocker = f"{queue_mode} postflight health failed"
  elif not _identity_valid(result): blocker = f"{queue_mode} child content identity differs"
  elif result.get("schema") != expected_schema or \
       result.get("queue_mode") != queue_mode or \
       result.get("status") != "PASS":
    blocker = result.get("exact_blocker") or f"{queue_mode} child contract differs"
  if blocker is None:
    try:
      if expected_schema == CANDIDATE_SCHEMA:
        candidate = validate_candidate_prefix_evidence(
          result, queue_mode=queue_mode,
          prefix_epochs=request.prefix_epochs)
        if candidate["request_identity"] != \
             _candidate_request_identity(request) or \
           candidate["config_identity"] != _identity(dict(request.config)):
          raise ValueError("candidate request/config binding differs")
      elif expected_schema == PM4_NO_DOORBELL_SCHEMA:
        diagnostic = validate_pm4_no_doorbell_evidence(result)
        if diagnostic["request_identity"] != \
             _pm4_no_doorbell_request_identity(request) or \
           diagnostic["config_identity"] != _identity(dict(request.config)):
          raise ValueError(
            "PM4 no-doorbell request/config binding differs")
      elif expected_schema == FFN_REDUCED_GRID_SCHEMA:
        diagnostic = validate_ffn_reduced_grid_evidence(
          result, diagnostic_global_size=request.diagnostic_global_size)
        if diagnostic["request_identity"] != \
             _ffn_reduced_grid_request_identity(request) or \
           diagnostic["config_identity"] != _identity(dict(request.config)):
          raise ValueError(
            "FFN reduced-grid request/config binding differs")
      elif expected_schema == DIRECT_SCHEMA:
        direct = validate_direct_evidence(result, queue_mode=queue_mode)
        if direct["request_identity"] != _direct_request_identity(request) or \
           direct["config_identity"] != _identity(dict(request.config)):
          raise ValueError("direct request/config binding differs")
      elif expected_schema == TRANSITION_SCHEMA:
        transition = validate_transition_evidence(
          result, queue_mode=queue_mode,
          sequence_name=request.sequence_name)
        if transition["request_identity"] != \
             _transition_request_identity(request) or \
           transition["config_identity"] != _identity(dict(request.config)):
          raise ValueError("transition request/config binding differs")
    except BaseException as exc:
      blocker = (
        f"{queue_mode} nested child evidence failed closed: "
        f"{type(exc).__name__}: {exc}")
  payload = {
    "schema": ENVELOPE_SCHEMA,
    "status": "PASS" if blocker is None else "BLOCKED",
    "exact_blocker": blocker, "queue_mode": queue_mode,
    "operation_schema": expected_schema,
    "health_before": health_before, "health_after": health_after,
    "kernel_faults": faults, "kernel_fault_evidence": fault_evidence,
    "launched": isolated is not None,
    "spawn_count": 1 if isolated is not None else 0,
    "child_status": child_status, "timed_out": timed_out, "error": error,
    "elapsed_seconds": getattr(isolated, "elapsed_seconds", None),
    "result": dict(result) if isinstance(result, Mapping) else None,
    "no_retry": True, "retry_count": 0, "no_queue_fallback": True,
    "promotion_evidence_eligible": False,
    "request_identity": _request_identity(expected_schema, request),
    "config_identity": _identity(dict(request.config)),
  }
  return {**payload, "evidence_identity": _identity(payload)}


def _containment_defaults(
    isolated_runner: Callable[..., Any] | None,
    health_probe: Callable[[Mapping[str, str]], bool] | None,
    fault_collector: Callable[
      [float], tuple[list[str], Mapping[str, Any]]] | None,
    ) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
  if isolated_runner is None:
    from tinygrad.runtime.process_isolated import run_isolated
    isolated_runner = run_isolated
  if health_probe is None:
    from extra.qk.mmq_target_epoch_orchestrator import spawned_tiny_health_probe
    health_probe = spawned_tiny_health_probe
  if fault_collector is None:
    from extra.qk.mmq_target_epoch_orchestrator import \
      collect_kernel_fault_evidence
    fault_collector = collect_kernel_fault_evidence
  if not all(callable(item) for item in (
      isolated_runner, health_probe, fault_collector)):
    raise TypeError("containment callbacks must be callable")
  return isolated_runner, health_probe, fault_collector


def run_guarded_candidate_prefix(
    *, config: Mapping[str, Any], queue_mode: str, prefix_epochs: int,
    runtime_builder: Callable[..., CandidatePrefixRuntime],
    prior_evidence: FrozenCorrectnessEvidenceRef | None = None,
    cross_queue_admission: FrozenCorrectnessEvidenceRef | None = None,
    timeout_seconds: float = 900.0,
    isolated_runner: Callable[..., Any] | None = None,
    health_probe: Callable[[Mapping[str, str]], bool] | None = None,
    fault_collector: Callable[
      [float], tuple[list[str], Mapping[str, Any]]] | None = None,
    ) -> dict[str, Any]:
  """Guard exactly one candidate prefix; never auto-escalate."""
  queue_mode, prefix_epochs = _queue(queue_mode), _prefix(prefix_epochs)
  timeout_seconds = _positive_seconds(timeout_seconds)
  isolated_runner, health_probe, fault_collector = _containment_defaults(
    isolated_runner, health_probe, fault_collector)
  return _guarded_envelope(
    child=run_candidate_prefix_child,
    request=CandidatePrefixRequest(
      dict(_mapping(config, "candidate config")), queue_mode, prefix_epochs,
      prior_evidence, cross_queue_admission, runtime_builder),
    expected_schema=CANDIDATE_SCHEMA, queue_mode=queue_mode,
    timeout_seconds=timeout_seconds, isolated_runner=isolated_runner,
    health_probe=health_probe, fault_collector=fault_collector)


def run_guarded_pm4_no_doorbell(
    *, config: Mapping[str, Any],
    runtime_builder: Callable[..., CandidatePrefixRuntime],
    timeout_seconds: float = 900.0,
    isolated_runner: Callable[..., Any] | None = None,
    health_probe: Callable[[Mapping[str, str]], bool] | None = None,
    fault_collector: Callable[
      [float], tuple[list[str], Mapping[str, Any]]] | None = None,
    ) -> dict[str, Any]:
  """Guard one terminal PM4 command capture with no target submission."""
  timeout_seconds = _positive_seconds(timeout_seconds)
  isolated_runner, health_probe, fault_collector = _containment_defaults(
    isolated_runner, health_probe, fault_collector)
  return _guarded_envelope(
    child=run_pm4_no_doorbell_child,
    request=PM4NoDoorbellRequest(
      dict(_mapping(config, "PM4 no-doorbell config")), runtime_builder),
    expected_schema=PM4_NO_DOORBELL_SCHEMA, queue_mode="PM4",
    timeout_seconds=timeout_seconds, isolated_runner=isolated_runner,
    health_probe=health_probe, fault_collector=fault_collector)


def run_guarded_ffn_reduced_grid(
    *, config: Mapping[str, Any],
    diagnostic_global_size: tuple[int, int, int],
    runtime_builder: Callable[..., CandidatePrefixRuntime],
    timeout_seconds: float = 900.0,
    isolated_runner: Callable[..., Any] | None = None,
    health_probe: Callable[[Mapping[str, str]], bool] | None = None,
    fault_collector: Callable[
      [float], tuple[list[str], Mapping[str, Any]]] | None = None,
    ) -> dict[str, Any]:
  """Guard one PM4-only reduced-grid execution; never escalate or retry."""
  grid = _ffn_reduced_global_size(diagnostic_global_size)
  timeout_seconds = _positive_seconds(timeout_seconds)
  isolated_runner, health_probe, fault_collector = _containment_defaults(
    isolated_runner, health_probe, fault_collector)
  return _guarded_envelope(
    child=run_ffn_reduced_grid_child,
    request=FFNReducedGridRequest(
      dict(_mapping(config, "FFN reduced-grid config")), grid,
      runtime_builder),
    expected_schema=FFN_REDUCED_GRID_SCHEMA, queue_mode="PM4",
    timeout_seconds=timeout_seconds, isolated_runner=isolated_runner,
    health_probe=health_probe, fault_collector=fault_collector)


def run_guarded_direct_correctness(
    *, config: Mapping[str, Any], queue_mode: str,
    candidate_full_evidence: FrozenCorrectnessEvidenceRef,
    runtime_builder: Callable[..., DirectCorrectnessRuntime],
    timeout_seconds: float = 1800.0,
    isolated_runner: Callable[..., Any] | None = None,
    health_probe: Callable[[Mapping[str, str]], bool] | None = None,
    fault_collector: Callable[
      [float], tuple[list[str], Mapping[str, Any]]] | None = None,
    ) -> dict[str, Any]:
  queue_mode, timeout_seconds = _queue(queue_mode), \
    _positive_seconds(timeout_seconds)
  isolated_runner, health_probe, fault_collector = _containment_defaults(
    isolated_runner, health_probe, fault_collector)
  return _guarded_envelope(
    child=run_direct_correctness_child,
    request=DirectCorrectnessRequest(
      dict(_mapping(config, "direct config")), queue_mode,
      candidate_full_evidence, runtime_builder),
    expected_schema=DIRECT_SCHEMA, queue_mode=queue_mode,
    timeout_seconds=timeout_seconds, isolated_runner=isolated_runner,
    health_probe=health_probe, fault_collector=fault_collector)


def run_guarded_transition(
    *, config: Mapping[str, Any], queue_mode: str, sequence_name: str,
    candidate_full_evidence: FrozenCorrectnessEvidenceRef,
    direct_evidence: FrozenCorrectnessEvidenceRef,
    worker: Callable[..., Mapping[str, Any]],
    timeout_seconds: float = 1800.0,
    isolated_runner: Callable[..., Any] | None = None,
    health_probe: Callable[[Mapping[str, str]], bool] | None = None,
    fault_collector: Callable[
      [float], tuple[list[str], Mapping[str, Any]]] | None = None,
    ) -> dict[str, Any]:
  queue_mode, timeout_seconds = _queue(queue_mode), \
    _positive_seconds(timeout_seconds)
  if sequence_name not in TRANSITION_SEQUENCES or not callable(worker):
    raise ValueError("guarded transition request differs")
  isolated_runner, health_probe, fault_collector = _containment_defaults(
    isolated_runner, health_probe, fault_collector)
  return _guarded_envelope(
    child=run_transition_child,
    request=TransitionRequest(
      dict(_mapping(config, "transition config")), queue_mode, sequence_name,
      candidate_full_evidence, direct_evidence, worker),
    expected_schema=TRANSITION_SCHEMA, queue_mode=queue_mode,
    timeout_seconds=timeout_seconds, isolated_runner=isolated_runner,
    health_probe=health_probe, fault_collector=fault_collector)


def compose_guarded_correctness_artifacts(
    *, candidate_by_queue: Mapping[
      str, Mapping[int, FrozenCorrectnessEvidenceRef]],
    direct_by_queue: Mapping[str, FrozenCorrectnessEvidenceRef],
    transitions_by_queue: Mapping[
      str, Mapping[str, FrozenCorrectnessEvidenceRef]],
    joint_c7_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
  """CPU-only exhaustive composition; this function never launches a child."""
  if set(candidate_by_queue) != set(QUEUE_MODES) or \
     set(direct_by_queue) != set(QUEUE_MODES) or \
     set(transitions_by_queue) != set(QUEUE_MODES):
    raise ValueError("correctness composition requires both queue modes")
  queues = {}
  shared = None
  for queue in QUEUE_MODES:
    if set(candidate_by_queue[queue]) != set(PREFIXES) or \
       set(transitions_by_queue[queue]) != set(TRANSITION_SEQUENCES):
      raise ValueError(f"{queue} correctness artifact set is incomplete")
    candidate_refs = candidate_by_queue[queue]
    candidate = {
      prefix: validate_candidate_prefix_evidence(
        _load_frozen_stage(
          candidate_refs[prefix], operation_schema=CANDIDATE_SCHEMA,
          queue_mode=queue),
        queue_mode=queue,
        prefix_epochs=prefix)
      for prefix in PREFIXES}
    if candidate[3]["predecessor_evidence_identity"] != \
         candidate_refs[1].envelope_evidence_identity or \
       candidate[20]["predecessor_evidence_identity"] != \
         candidate_refs[3].envelope_evidence_identity:
      raise ValueError(f"{queue} candidate predecessor chain differs")
    stage_authority_fields = (
      "family_identity", "fixture_identity", "workload_identity",
      "input_identity", "logical_q4_identity",
      "resident_fp16_activation_identity", "candidate_executable_identity",
      "program_key", "binary_sha256")
    if any(candidate[prefix][field] != candidate[1][field]
           for prefix in (3, 20) for field in stage_authority_fields):
      raise ValueError(f"{queue} candidate stage authority differs")
    direct_ref = direct_by_queue[queue]
    direct = validate_direct_evidence(
      _load_frozen_stage(
        direct_ref, operation_schema=DIRECT_SCHEMA, queue_mode=queue),
      queue_mode=queue)
    direct_authority_fields = (
      "family_identity", "fixture_identity", "workload_identity",
      "input_identity", "logical_q4_identity",
      "resident_fp16_activation_identity")
    if direct["candidate_full_evidence_identity"] != \
         candidate_refs[20].envelope_evidence_identity or \
       any(direct[field] != candidate[20][field]
           for field in direct_authority_fields):
      raise ValueError(f"{queue} direct candidate authority differs")
    transitions = {}
    for name, reference in transitions_by_queue[queue].items():
      row = validate_transition_evidence(
        _load_frozen_stage(
          reference, operation_schema=TRANSITION_SCHEMA, queue_mode=queue),
        queue_mode=queue, sequence_name=name)
      if row["candidate_full_evidence_identity"] != \
           candidate_refs[20].envelope_evidence_identity or \
         row["direct_evidence_identity"] != \
           direct_ref.envelope_evidence_identity or \
         any(row[field] != candidate[20][field]
             for field in direct_authority_fields):
        raise ValueError(f"{queue} transition {name} authority differs")
      transitions[name] = row
    authority = tuple(
      candidate[20][field] for field in stage_authority_fields)
    if shared is None: shared = authority
    elif shared != authority:
      raise ValueError("PM4/AQL correctness authority differs")
    queues[queue] = {
      "candidate": candidate, "direct": direct, "transitions": transitions}
  if queues["AQL"]["candidate"][1]["predecessor_evidence_identity"] != \
       candidate_by_queue["PM4"][20].envelope_evidence_identity:
    raise ValueError("AQL admission does not bind complete PM4 candidate")
  c7 = validate_joint_c7_evidence(joint_c7_evidence)
  if c7["family_identity"] != shared[0] or c7["input_identity"] != shared[3]:
    raise ValueError("joint C7 authority differs from correctness artifacts")
  payload = {
    "schema": COMPOSITION_SCHEMA, "status": "PASS",
    "exact_blocker": None, "queues": queues, "joint_c7": c7,
    "execution_performed": False,
    "composition_only": True, "promotion_evidence_eligible": False,
  }
  return {**payload, "evidence_identity": _identity(payload)}


def validate_joint_c7_evidence(value: Any) -> dict[str, Any]:
  row = dict(_mapping(value, "joint C7 evidence"))
  checks = {
    "identity": _identity_valid(row),
    "schema": row.get("schema") == JOINT_C7_SCHEMA,
    "status": row.get("status") == "PASS",
    "census": row.get("physical_allocation_census_complete") is True,
    "lifetimes": row.get("allocation_lifetimes_complete") is True,
    "dense_weight": row.get("dense_fp16_weight_materialization") is False,
    "budget": row.get("within_admitted_budget") is True,
    "promotion": row.get("promotion_evidence_eligible") is False,
  }
  if not all(checks.values()):
    raise ValueError(
      "joint C7 evidence failed checks: "
      f"{sorted(key for key, passed in checks.items() if not passed)!r}")
  _content_identity(row.get("family_identity"), "joint C7 family identity")
  _content_identity(row.get("input_identity"), "joint C7 input identity")
  queues = _mapping(row.get("queues"), "joint C7 queues")
  if set(queues) != set(QUEUE_MODES):
    raise ValueError("joint C7 queue set differs")
  for queue in QUEUE_MODES:
    child = _mapping(queues[queue], f"joint C7 {queue}")
    if not _identity_valid(child) or child.get("status") != "PASS" or \
       child.get("queue_mode") != queue or \
       child.get("family_identity") != row["family_identity"] or \
       child.get("input_identity") != row["input_identity"] or \
       child.get("physical_allocation_census_complete") is not True:
      raise ValueError(f"joint C7 nested {queue} evidence differs")
  return row


def build_production_candidate_prefix_runtime(
    config: Mapping[str, Any], *, queue_mode: str, prefix_epochs: int,
    pm4_submit_policy: str = "execute",
    pm4_no_doorbell_receipt_sink: list[Mapping[str, Any]] | None = None,
    diagnostic_global_size: tuple[int, int, int] | None = None,
    diagnostic_dispatch_receipt_sink:
      list[Mapping[str, Any]] | None = None,
    ) -> CandidatePrefixRuntime:
  """Production default for one candidate stage, including standalone PM4 p1."""
  queue_mode, prefix_epochs = _queue(queue_mode), _prefix(prefix_epochs)
  if os.environ.get("AMD_AQL") != ("1" if queue_mode == "AQL" else "0"):
    raise ValueError("AMD_AQL differs before candidate runtime construction")
  row = dict(_mapping(config, "production candidate config"))
  required = {
    "frozen_bundle", "staged_family_manifest", "execution_fixture_v2",
    "runtime_canary_isolation", "candidate_executable_identity",
  }
  if set(row) != required:
    raise ValueError("production candidate config fields differ")
  from extra.qk.mmq_attn_qo_c6_binding import read_json
  from extra.qk.mmq_exact_role_spec import exact_role_spec
  from extra.qk.mmq_ffn_gate_up_c8_runtime import \
    rebuild_ffn_gate_up_v2_fixture
  from extra.qk.mmq_frozen_staged_family import \
    load_frozen_staged_family_manifest
  from extra.qk.mmq_frozen_staged_family_execution import \
    validate_frozen_staged_runtime_canary_isolation
  from extra.qk.mmq_frozen_staged_low_level_session import (
    FrozenStagedLowLevelSession, FrozenStagedProgramAuthority,
    production_frozen_staged_low_level_dependencies,
  )
  from extra.qk.mmq_llama_five_buffer_gpu_harness import _numeric_comparison
  from tinygrad import Tensor, dtypes
  from tinygrad.device import Device

  role = exact_role_spec("ffn_gate_up")
  family = load_frozen_staged_family_manifest(
    row["staged_family_manifest"], role_spec=role,
    frozen_bundle=row["frozen_bundle"])
  fixture = rebuild_ffn_gate_up_v2_fixture(
    role, read_json(row["execution_fixture_v2"], "execution fixture v2"))
  raw_canary = row["runtime_canary_isolation"]
  if not isinstance(raw_canary, Mapping):
    raw_canary = read_json(raw_canary, f"{queue_mode} C4 canary")
  canary = validate_frozen_staged_runtime_canary_isolation(
    raw_canary, family, queue_mode=queue_mode)
  candidate_identity = _content_identity(
    row["candidate_executable_identity"], "candidate executable identity")
  if candidate_identity != ffn_gate_up_candidate_executable_identity(family):
    raise ValueError("candidate executable identity differs from frozen family")
  resident_fp16 = Tensor(
    np.ascontiguousarray(
      fixture.resident_fp16_activation.reshape(1, role.m, role.k)),
    dtype=dtypes.float16, device="AMD")
  epoch_major_q4 = Tensor(
    fixture.q4_epoch_major, dtype=dtypes.uint32, device="AMD")
  authority = FrozenStagedProgramAuthority.from_binding(
    family.binding, family_identity=family.family_identity,
    candidate_executable_identity=candidate_identity,
    input_identity=fixture.input_identity)
  dependencies = production_frozen_staged_low_level_dependencies(
    authority, pm4_submit_policy=pm4_submit_policy,
    pm4_no_doorbell_receipt_sink=pm4_no_doorbell_receipt_sink,
    diagnostic_global_size=diagnostic_global_size,
    diagnostic_dispatch_receipt_sink=diagnostic_dispatch_receipt_sink)
  produced: list[tuple[Any, Any, Any]] = []
  captured: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
  raw_produce = dependencies.produce_q8

  def retain_q8(*args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    result = raw_produce(*args, **kwargs)
    produced.append(result)
    return result

  dependencies = replace(dependencies, produce_q8=retain_q8)
  session = FrozenStagedLowLevelSession.prepare(
    binding=family.binding, authority=authority,
    common_resident_fp16=resident_fp16,
    q4_epoch_major=epoch_major_q4, dependencies=dependencies)

  def producer_attest(prefix: int) -> Mapping[str, Any]:
    if prefix != prefix_epochs or len(produced) != 1:
      raise ValueError("candidate stage did not produce exactly one Q8 tuple")
    values_t, scales_t, sums_t = produced.pop()
    values, scales, sums = (
      np.ascontiguousarray(values_t.numpy()),
      np.ascontiguousarray(scales_t.numpy()),
      np.ascontiguousarray(sums_t.numpy()))
    if values.shape != fixture.q8_values.shape or \
       scales.shape != fixture.q8_scales.shape or \
       sums.shape != fixture.q8_sums.shape or values.dtype != np.int8 or \
       scales.dtype != np.float32 or sums.dtype != np.float32:
      raise ValueError("physical Q8 shape/dtype differs")
    scale_cmp = _numeric_comparison(scales, fixture.q8_scales)
    sum_cmp = _numeric_comparison(sums, fixture.q8_sums)
    captured[prefix] = (values, scales, sums)
    payload = {
      "schema": PRODUCER_SCHEMA, "status": "PASS",
      "queue_mode": queue_mode, "prefix_epochs": prefix,
      "family_identity": family.family_identity,
      "input_identity": fixture.input_identity,
      "metadata_storage_dtype": "float32",
      "captured_for_consumer_reference": True,
      "consumer_reference_q8_sha256": {
        "values": _array_sha256(values), "scales": _array_sha256(scales),
        "sums": _array_sha256(sums)},
      "fixture_diagnostic": {
        "q8_values_exact": bool(np.array_equal(values, fixture.q8_values)),
        "q8_scales_numeric_match": scale_cmp.get("status") == "pass",
        "q8_sums_numeric_match": sum_cmp.get("status") == "pass",
        "observed_sha256": {
          "values": _array_sha256(values), "scales": _array_sha256(scales),
          "sums": _array_sha256(sums)},
        "fixture_sha256": {
          "values": _array_sha256(fixture.q8_values),
          "scales": _array_sha256(fixture.q8_scales),
          "sums": _array_sha256(fixture.q8_sums)},
        "scales_comparison": scale_cmp, "sums_comparison": sum_cmp,
      },
      "promotion_evidence_eligible": False,
    }
    return {**payload, "evidence_identity": _identity(payload)}

  references: dict[int, np.ndarray] = {}
  def reference(prefix: int) -> np.ndarray:
    if prefix != prefix_epochs:
      raise ValueError("candidate reference stage differs")
    if prefix not in references:
      if prefix not in captured:
        raise ValueError("candidate consumer reference lacks captured Q8")
      references[prefix] = ffn_gate_up_consumer_prefix_reference(
        fixture, prefix, q8_values=captured[prefix][0],
        q8_scales=captured[prefix][1], q8_sums=captured[prefix][2])
    return references[prefix]

  return CandidatePrefixRuntime(
    queue_mode, prefix_epochs, family.family_identity,
    fixture.fixture_identity, fixture.workload_identity,
    fixture.input_identity, fixture.logical_q4_identity,
    fixture.resident_fp16_activation_identity,
    candidate_identity, family.binding.program_key,
    family.binding.binary_sha256, _identity(canary), session,
    producer_attest, Device["AMD"].synchronize,
    lambda output: np.ascontiguousarray(output.numpy()).reshape(OUTPUT_SHAPE),
    reference, _numeric_comparison).validate(queue_mode, prefix_epochs)


__all__ = [
  "CANDIDATE_SCHEMA", "COMPOSITION_SCHEMA", "DIRECT_SCHEMA",
  "ENVELOPE_SCHEMA", "FFN_REDUCED_GLOBAL_SIZE_ALLOWLIST",
  "FFN_REDUCED_GRID_SCHEMA", "OUTPUT_ELEMENTS", "OUTPUT_SHAPE", "PREFIXES",
  "JOINT_C7_SCHEMA", "PM4_NO_DOORBELL_RECEIPT_SCHEMA",
  "PM4_NO_DOORBELL_SCHEMA", "PRODUCER_SCHEMA", "QUEUE_MODES", "SCHEMA",
  "TRANSITION_SCHEMA",
  "TRANSITION_SEQUENCES", "CandidatePrefixRequest",
  "CandidatePrefixRuntime", "FFNReducedGridRequest", "PM4NoDoorbellRequest",
  "DirectCorrectnessRequest",
  "DirectCorrectnessRuntime", "FrozenCorrectnessEvidenceRef",
  "TransitionRequest",
  "build_production_candidate_prefix_runtime",
  "compose_guarded_correctness_artifacts",
  "ffn_gate_up_candidate_executable_identity",
  "ffn_gate_up_consumer_prefix_reference",
  "ffn_gate_up_direct_dense_reference", "freeze_correctness_evidence",
  "load_frozen_correctness_evidence", "run_candidate_prefix_child",
  "run_ffn_reduced_grid_child", "run_guarded_ffn_reduced_grid",
  "run_guarded_pm4_no_doorbell", "run_pm4_no_doorbell_child",
  "run_direct_correctness_child", "run_guarded_candidate_prefix",
  "run_guarded_direct_correctness", "run_guarded_transition",
  "run_transition_child", "validate_candidate_prefix_evidence",
  "validate_direct_evidence", "validate_joint_c7_evidence",
  "validate_ffn_reduced_grid_evidence",
  "validate_pm4_no_doorbell_evidence",
]
