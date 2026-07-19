"""CPU-only composition bootstrap for exact ``ffn_gate_up`` matched C8.

The frozen bundle's target-fixture v1 remains valid PROGRAM provenance.  It is
not, however, matched-input evidence: this bootstrap requires a separate v2
execution fixture whose candidate Q8 bytes are regenerated from the FP16
resident activation after an FP16 -> FP32 roundtrip.  The same resident FP16
bytes are supplied to production direct-packed execution.

No Device is imported or initialized here.  Production Tensor construction and
queue attestation reuse the lazy seams in ``mmq_attn_qo_c8_runtime``.  This
module deliberately selects no timing launcher: a dedicated worker must inject
both untimed route callbacks and wrap them with the ffn outer-wall runner.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from extra.qk.mmq_attn_qo_c6_binding import read_json
from extra.qk.mmq_attn_qo_c8_runtime import (
  DirectPackedObjectBuilder, DirectPackedObjects, build_direct_packed_objects,
  qualification_paths, queue_attestation_bindings,
)
from extra.qk.mmq_exact_role_spec import exact_role_spec
from extra.qk.mmq_ffn_gate_up_matched_timing_contract import (
  CANDIDATE_ROUTE, DIRECT_ROUTE, K, K_LAUNCHES, M, N, QUEUE_MODES,
  validate_ffn_gate_up_matched_complete_role_timing_contract,
)
from extra.qk.mmq_frozen_staged_family import (
  FrozenStagedFamily, load_frozen_staged_family_manifest,
)


FIXTURE_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.ffn_gate_up_execution_fixture.v2"
COMPOSITION_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.ffn_gate_up_c8_runtime_composition.v1"
INPUT_IDENTITY_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.ffn_gate_up_common_resident_fp16_input.v2"
WORKLOAD_IDENTITY_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.ffn_gate_up_complete_role_workload.v1"
Q4_IDENTITY_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.ffn_gate_up_logical_q4.v1"
ACTIVATION_IDENTITY_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.ffn_gate_up_resident_fp16_activation.v1"

FFN_ROLE = "ffn_gate_up"
FFN_SEEDS = {"q4": 20260721, "activation_source": 20260722}
FP16_INPUT_SEMANTICS = "common_resident_fp16_roundtrip_v2"
OUTER_WALL_WRAPPER = \
  "extra.qk.mmq_ffn_gate_up_outer_wall_runner." \
  "run_ffn_gate_up_outer_synchronized_wall"
OUTPUT_REALIZATION_SEMANTICS = \
  "realize_route_output_without_readback_and_return_none"

_HEX = frozenset("0123456789abcdef")
_CONFIG_FIELDS = {
  "composition", "execution_fixture_v2", "matched_timing_contract",
  "frozen_bundle", "staged_family_manifest",
  "qualification_pm4", "qualification_aql",
}
_FIXTURE_FIELDS = {
  "schema", "role", "shape", "total_epochs", "seeds", "activation", "repack",
}
_ACTIVATION_FIELDS = {
  "generation_dtype", "resident_dtype", "resident_shape",
  "resident_fp16_sha256", "roundtrip_dtype", "roundtrip_fp32_sha256",
  "candidate_q8_source",
}
_REPACK_FIELDS = {
  "q4_sha256", "q4_layout", "q8_values_sha256", "q8_scales_sha256",
  "q8_sums_sha256", "q8_layout", "q4_epoch_major_sha256",
  "q4_epoch_major_layout", "q4_epoch_major_dtype",
  "q4_epoch_major_elements",
}
_COMPOSITION_FIELDS = {
  "schema", "status", "role", "family_identity",
  "execution_fixture_identity", "workload_identity", "input_identity",
  "logical_q4_identity", "resident_fp16_activation_identity",
  "candidate_binding", "direct_bindings_by_queue", "c6_by_queue",
  "joint_session_c7_identity", "transition_preflight_bindings_by_queue",
  "runtime_canary_by_queue", "matched_timing_contract_identity",
  "promotion_eligible_on_candidate_win", "composition_identity",
}
_CANDIDATE_FIELDS = {
  "family_identity", "candidate_executable_identity", "program_key",
  "binary_sha256",
}
_DIRECT_FIELDS = {
  "qualification_identity", "executable_identity", "binary_sha256",
}
_C6_FIELDS = {
  "status", "fixture_schema", "input_semantics",
  "legacy_fp32_prequantized", "family_identity", "workload_identity",
  "input_identity", "device_identity", "software_identity",
  "evidence_identity", "candidate_correctness_identity",
  "comparator_identity",
}
_TRANSITION_FIELDS = {
  "candidate_candidate", "direct_direct", "direct_candidate_prefix1",
  "direct_candidate_full_role", "candidate_direct_candidate",
}


def _canonical(value: Any) -> bytes:
  return json.dumps(
    value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(value: Any) -> str:
  return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _bytes_sha256(value: np.ndarray) -> str:
  return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
  if not isinstance(value, Mapping):
    raise ValueError(f"{label} must be a mapping")
  return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
  if set(value) != expected:
    raise ValueError(
      f"{label} fields differ: expected {sorted(expected)!r}, "
      f"got {sorted(value)!r}")


def _digest(value: Any, label: str, *, prefixed: bool = True) -> str:
  if not isinstance(value, str):
    raise ValueError(f"{label} must be a SHA-256 digest")
  raw = value[7:] if prefixed and value.startswith("sha256:") else value
  if prefixed and not value.startswith("sha256:") or \
     len(raw) != 64 or any(char not in _HEX for char in raw):
    raise ValueError(f"{label} must be a lowercase SHA-256 digest")
  return value


def _exact_role(role: Any) -> Any:
  if getattr(role, "role", None) != FFN_ROLE or \
     tuple(getattr(role, "shape", ())) != (M, N, K) or \
     getattr(role, "epochs", None) != K_LAUNCHES:
    raise ValueError("runtime composition is exact to 512x17408x5120 ffn_gate_up")
  return role


def resident_fp16_roundtrip(source: Any) -> tuple[np.ndarray, np.ndarray]:
  """Return the one common resident activation and candidate FP32 source."""
  source_fp32 = np.ascontiguousarray(np.asarray(source, dtype=np.float32))
  if source_fp32.ndim != 2:
    raise ValueError("activation source must be rank-2 [M,K]")
  resident = np.ascontiguousarray(source_fp32.astype(np.float16))
  roundtrip = np.ascontiguousarray(resident.astype(np.float32))
  return resident, roundtrip


def _random_q4_words(n: int, k: int, seed: int) -> np.ndarray:
  if k % 256:
    raise ValueError("Q4_K fixture requires K divisible by 256")
  rng = np.random.default_rng(seed)
  raw = rng.integers(0, 256, size=(n, k // 256, 144), dtype=np.uint8)
  raw[:, :, :4] = np.frombuffer(
    np.array([0.03125, 0.0078125], dtype="<f2").tobytes(),
    dtype=np.uint8)
  return np.ascontiguousarray(raw.reshape(-1).view(np.uint32))


def _pack_q4_epochs_contiguous(blocks: np.ndarray) -> np.ndarray:
  blocks = np.asarray(blocks)
  if blocks.dtype != np.uint8 or blocks.ndim != 3 or blocks.shape[2] != 144:
    raise ValueError("Q4 blocks must be uint8 [N,epoch,144]")
  return np.ascontiguousarray(blocks.transpose(1, 0, 2)).reshape(-1).view(
    np.uint32)


def _production_quantizer(source: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  from extra.qk.mmq_q4k_q8_reference import \
    q8_1_mmq_ds4_quantize_reference
  return q8_1_mmq_ds4_quantize_reference(source)


def _validate_q8_arrays(
    values: Any, scales: Any, sums: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  normalized = (
    np.ascontiguousarray(values), np.ascontiguousarray(scales),
    np.ascontiguousarray(sums))
  expected = (
    ((K // 128, M, 128), np.dtype(np.int8)),
    ((K // 128, M, 4), np.dtype(np.float32)),
    ((K // 128, M, 4), np.dtype(np.float32)),
  )
  for value, (shape, dtype), label in zip(
      normalized, expected, ("values", "scales", "sums")):
    if value.shape != shape or value.dtype != dtype:
      raise ValueError(
        f"v2 fixture Q8 {label} must be {dtype} shaped {shape!r}")
  return normalized


@dataclass(frozen=True)
class FfnGateUpV2Fixture:
  role_spec: Any
  execution_fixture: Mapping[str, Any]
  words: np.ndarray
  resident_fp16_activation: np.ndarray
  roundtrip_fp32: np.ndarray
  q8_values: np.ndarray
  q8_scales: np.ndarray
  q8_sums: np.ndarray
  q4_epoch_major: np.ndarray
  fixture_identity: str
  workload_identity: str
  input_identity: str
  logical_q4_identity: str
  resident_fp16_activation_identity: str


@dataclass(frozen=True)
class FfnGateUpCandidateInputs:
  """Launchable v2 inputs supplied to a production-faithful candidate route.

  Prequantized Q8 arrays are intentionally absent.  The route must run its Q8
  producer from ``resident_fp16_activation`` inside every measured outer wall.
  Retained Q8 hashes are correctness references only.
  """

  role_spec: Any
  words: np.ndarray
  q4_epoch_major: np.ndarray
  resident_fp16_activation: Any
  q8_producer_semantics: str
  q8_reference_sha256: Mapping[str, str]
  fixture_identity: str
  workload_identity: str
  input_identity: str
  logical_q4_identity: str
  resident_fp16_activation_identity: str


@dataclass(frozen=True)
class FfnGateUpNoReadbackOutputRealizer:
  """Typed output realization whose call boundary enforces no returned data."""

  callback: Callable[[Any], Any]
  semantics: str
  readback_performed: bool

  def validate(self) -> "FfnGateUpNoReadbackOutputRealizer":
    if not callable(self.callback) or \
       self.semantics != OUTPUT_REALIZATION_SEMANTICS or \
       self.readback_performed is not False:
      raise ValueError(
        "ffn_gate_up output realization must be callable, no-readback, "
        "and return None")
    return self

  def __call__(self, output: Any) -> None:
    self.validate()
    if self.callback(output) is not None:
      raise ValueError(
        "ffn_gate_up output realization must return None without readback")


@dataclass(frozen=True)
class FfnGateUpRouteCallback:
  """One typed untimed callback that cannot masquerade as a legacy receipt."""

  route_id: str
  queue_mode: str
  input_identity: str
  executable_identity: str
  invoke: Callable[[], Any]
  realize_output: FfnGateUpNoReadbackOutputRealizer
  attest_post_sync: Callable[[Any, str], Mapping[str, Any]]
  outer_wall_wrapper: str
  emits_timing_receipt: bool

  def validate(
      self, *, route_id: str, queue_mode: str, input_identity: str,
      executable_identity: str,
      ) -> "FfnGateUpRouteCallback":
    if self.route_id != route_id or self.queue_mode != queue_mode or \
       self.input_identity != input_identity or \
       self.executable_identity != executable_identity:
      raise ValueError(f"{route_id} callback identity differs")
    if self.queue_mode not in QUEUE_MODES:
      raise ValueError(f"{route_id} callback queue mode differs")
    _digest(self.input_identity, f"{route_id} callback input identity")
    _digest(self.executable_identity, f"{route_id} callback executable identity")
    if not isinstance(
        self.realize_output, FfnGateUpNoReadbackOutputRealizer):
      raise TypeError(
        f"{route_id} callback requires a typed no-readback output realizer")
    self.realize_output.validate()
    if self.outer_wall_wrapper != OUTER_WALL_WRAPPER or \
       self.emits_timing_receipt is not False or not callable(self.invoke) or \
       not callable(self.attest_post_sync):
      raise ValueError(
        f"{route_id} callback must be untimed, observed post-sync, and owned "
        "by the ffn outer wall")
    return self


@dataclass(frozen=True)
class FfnGateUpOuterWallRoutes:
  """Typed invocation callbacks for the dedicated ffn outer-wall worker.

  Each ``invoke`` must return
  ``mmq_ffn_gate_up_outer_wall_runner.RouteInvocation`` when called.  This is
  intentionally not a legacy ``QueueTimingRunners``:
  timing and receipt construction belong to
  ``run_ffn_gate_up_outer_synchronized_wall``.
  """

  candidate: FfnGateUpRouteCallback
  direct_packed: FfnGateUpRouteCallback

  def validate(
      self, *, queue_mode: str, input_identity: str,
      candidate_executable_identity: str,
      direct_executable_identity: str,
      ) -> "FfnGateUpOuterWallRoutes":
    if not isinstance(self.candidate, FfnGateUpRouteCallback) or \
       not isinstance(self.direct_packed, FfnGateUpRouteCallback):
      raise TypeError(
        "ffn_gate_up builders must return typed outer-wall route callbacks")
    self.candidate.validate(
      route_id=CANDIDATE_ROUTE, queue_mode=queue_mode,
      input_identity=input_identity,
      executable_identity=candidate_executable_identity)
    self.direct_packed.validate(
      route_id=DIRECT_ROUTE, queue_mode=queue_mode,
      input_identity=input_identity,
      executable_identity=direct_executable_identity)
    return self


def _validate_fixture_manifest(value: Any, role: Any) -> dict[str, Any]:
  role = _exact_role(role)
  row = dict(_mapping(value, "ffn_gate_up v2 execution fixture"))
  if row.get("schema") != FIXTURE_SCHEMA or "source_sha256" in row or \
     _mapping(row.get("seeds", {}), "ffn_gate_up fixture seeds").get(
       "q8_source") is not None:
    raise ValueError(
      "legacy FP32-prequantized ffn_gate_up fixture is rejected")
  _exact_keys(row, _FIXTURE_FIELDS, "ffn_gate_up v2 execution fixture")
  if row["role"] != FFN_ROLE or row["shape"] != [M, N, K] or \
     row["total_epochs"] != K_LAUNCHES or \
     row["seeds"] != FFN_SEEDS:
    raise ValueError(
      "legacy FP32-prequantized or mismatched ffn_gate_up fixture is rejected")
  activation = _mapping(row["activation"], "v2 fixture activation")
  _exact_keys(activation, _ACTIVATION_FIELDS, "v2 fixture activation")
  expected_activation = {
    "generation_dtype": "float32",
    "resident_dtype": "float16",
    "resident_shape": [1, M, K],
    "roundtrip_dtype": "float32",
    "candidate_q8_source": "resident_fp16_roundtrip_to_float32",
  }
  if any(activation.get(key) != expected
         for key, expected in expected_activation.items()):
    raise ValueError("v2 fixture does not share one resident FP16 activation")
  _digest(
    activation["resident_fp16_sha256"],
    "resident FP16 activation SHA-256", prefixed=False)
  _digest(
    activation["roundtrip_fp32_sha256"],
    "FP16 roundtrip SHA-256", prefixed=False)

  repack = _mapping(row["repack"], "v2 fixture repack")
  _exact_keys(repack, _REPACK_FIELDS, "v2 fixture repack")
  expected_repack = {
    "q4_layout": "q4_k_bytes[n, k_epoch, 144]",
    "q8_layout": "q8_ds4_from_fp16_roundtrip[epoch, m, groups]",
    "q4_epoch_major_layout": "q4_k_bytes[k_epoch, n, 144]",
    "q4_epoch_major_dtype": "uint32",
    "q4_epoch_major_elements": N * K_LAUNCHES * 36,
  }
  if any(repack.get(key) != expected for key, expected in expected_repack.items()):
    raise ValueError("v2 fixture layout or Q8 FP16-roundtrip semantics differ")
  for field in (
      "q4_sha256", "q8_values_sha256", "q8_scales_sha256",
      "q8_sums_sha256", "q4_epoch_major_sha256"):
    _digest(repack[field], f"v2 fixture {field}", prefixed=False)
  return row


def build_ffn_gate_up_v2_fixture_manifest(
    role: Any, *,
    quantizer: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]] =
      _production_quantizer,
    ) -> dict[str, Any]:
  """Build the canonical separate v2 matched-input fixture descriptor."""
  role = _exact_role(role)
  words = _random_q4_words(N, K, FFN_SEEDS["q4"])
  blocks = words.view(np.uint8).reshape(N, K_LAUNCHES, 144)
  source = np.random.default_rng(
    FFN_SEEDS["activation_source"]).standard_normal((M, K), dtype=np.float32)
  resident, roundtrip = resident_fp16_roundtrip(source)
  if not callable(quantizer):
    raise TypeError("fixture quantizer must be callable")
  values, scales, sums = _validate_q8_arrays(*quantizer(roundtrip))
  epoch_major = _pack_q4_epochs_contiguous(blocks)
  manifest = {
    "schema": FIXTURE_SCHEMA, "role": FFN_ROLE,
    "shape": [M, N, K], "total_epochs": K_LAUNCHES,
    "seeds": dict(FFN_SEEDS),
    "activation": {
      "generation_dtype": "float32", "resident_dtype": "float16",
      "resident_shape": [1, M, K],
      "resident_fp16_sha256": _bytes_sha256(resident),
      "roundtrip_dtype": "float32",
      "roundtrip_fp32_sha256": _bytes_sha256(roundtrip),
      "candidate_q8_source": "resident_fp16_roundtrip_to_float32",
    },
    "repack": {
      "q4_sha256": _bytes_sha256(blocks),
      "q4_layout": "q4_k_bytes[n, k_epoch, 144]",
      "q8_values_sha256": _bytes_sha256(values),
      "q8_scales_sha256": _bytes_sha256(scales),
      "q8_sums_sha256": _bytes_sha256(sums),
      "q8_layout": "q8_ds4_from_fp16_roundtrip[epoch, m, groups]",
      "q4_epoch_major_sha256": _bytes_sha256(epoch_major),
      "q4_epoch_major_layout": "q4_k_bytes[k_epoch, n, 144]",
      "q4_epoch_major_dtype": "uint32",
      "q4_epoch_major_elements": N * K_LAUNCHES * 36,
    },
  }
  return _validate_fixture_manifest(manifest, role)


def rebuild_ffn_gate_up_v2_fixture(
    role: Any, execution_fixture: Mapping[str, Any], *,
    quantizer: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]] =
      _production_quantizer,
    ) -> FfnGateUpV2Fixture:
  """Regenerate and byte-check the separate matched-input v2 authority."""
  role = _exact_role(role)
  fixture = _validate_fixture_manifest(execution_fixture, role)
  words = _random_q4_words(N, K, fixture["seeds"]["q4"])
  blocks = words.view(np.uint8).reshape(N, K_LAUNCHES, 144)
  source = np.random.default_rng(
    fixture["seeds"]["activation_source"]).standard_normal(
      (M, K), dtype=np.float32)
  resident, roundtrip = resident_fp16_roundtrip(source)
  if not callable(quantizer):
    raise TypeError("fixture quantizer must be callable")
  values, scales, sums = _validate_q8_arrays(*quantizer(roundtrip))
  epoch_major = _pack_q4_epochs_contiguous(blocks)
  activation, repack = fixture["activation"], fixture["repack"]
  observed = {
    "resident_fp16_sha256": _bytes_sha256(resident),
    "roundtrip_fp32_sha256": _bytes_sha256(roundtrip),
    "q4_sha256": _bytes_sha256(blocks),
    "q8_values_sha256": _bytes_sha256(values),
    "q8_scales_sha256": _bytes_sha256(scales),
    "q8_sums_sha256": _bytes_sha256(sums),
    "q4_epoch_major_sha256": _bytes_sha256(epoch_major),
  }
  expected = {
    key: activation[key] if key in activation else repack[key]
    for key in observed
  }
  if observed != expected:
    drifted = sorted(key for key in observed if observed[key] != expected[key])
    raise ValueError(f"rebuilt ffn_gate_up v2 fixture bytes differ: {drifted!r}")

  fixture_identity = _identity(fixture)
  workload_identity = _identity({
    "schema": WORKLOAD_IDENTITY_SCHEMA, "role": FFN_ROLE,
    "shape": [M, N, K], "k_per_launch": 256,
    "k_launches": K_LAUNCHES, "complete_role": True,
  })
  logical_q4_identity = _identity({
    "schema": Q4_IDENTITY_SCHEMA, "role": FFN_ROLE,
    "shape": [N, K_LAUNCHES, 144],
    "layout": repack["q4_layout"], "sha256": observed["q4_sha256"],
  })
  resident_identity = _identity({
    "schema": ACTIVATION_IDENTITY_SCHEMA, "role": FFN_ROLE,
    "shape": [1, M, K], "dtype": "float16",
    "sha256": observed["resident_fp16_sha256"],
  })
  input_identity = _identity({
    "schema": INPUT_IDENTITY_SCHEMA, "fixture_identity": fixture_identity,
    "workload_identity": workload_identity,
    "logical_q4_identity": logical_q4_identity,
    "resident_fp16_activation_identity": resident_identity,
    "candidate_q8_source": "resident_fp16_roundtrip_to_float32",
    "roundtrip_fp32_sha256": observed["roundtrip_fp32_sha256"],
  })
  return FfnGateUpV2Fixture(
    role, fixture, words, resident, roundtrip, values, scales, sums,
    epoch_major, fixture_identity, workload_identity, input_identity,
    logical_q4_identity, resident_identity)


def _queue_rows(
    value: Any, fields: set[str], label: str,
    ) -> dict[str, dict[str, Any]]:
  rows = _mapping(value, label)
  if set(rows) != set(QUEUE_MODES):
    raise ValueError(f"{label} must contain exactly {QUEUE_MODES!r}")
  normalized = {}
  for queue in QUEUE_MODES:
    row = dict(_mapping(rows[queue], f"{queue} {label}"))
    _exact_keys(row, fields, f"{queue} {label}")
    normalized[queue] = row
  return normalized


def _composition(
    value: Any, *, family: FrozenStagedFamily, fixture: FfnGateUpV2Fixture,
    ) -> dict[str, Any]:
  row = dict(_mapping(value, "ffn_gate_up C8 runtime composition"))
  _exact_keys(row, _COMPOSITION_FIELDS, "ffn_gate_up C8 runtime composition")
  role = _exact_role(family.binding.role_spec)
  expected = {
    "schema": COMPOSITION_SCHEMA, "status": "READY", "role": FFN_ROLE,
    "family_identity": family.family_identity,
    "execution_fixture_identity": fixture.fixture_identity,
    "workload_identity": fixture.workload_identity,
    "input_identity": fixture.input_identity,
    "logical_q4_identity": fixture.logical_q4_identity,
    "resident_fp16_activation_identity":
      fixture.resident_fp16_activation_identity,
    "promotion_eligible_on_candidate_win": False,
  }
  if any(row.get(key) != expected_value
         for key, expected_value in expected.items()):
    raise ValueError("ffn_gate_up composition differs from family/v2 input authority")
  payload = {
    key: value for key, value in row.items() if key != "composition_identity"}
  if row["composition_identity"] != _identity(payload):
    raise ValueError("ffn_gate_up composition content identity differs")

  candidate = dict(_mapping(row["candidate_binding"], "candidate binding"))
  _exact_keys(candidate, _CANDIDATE_FIELDS, "candidate binding")
  if candidate["family_identity"] != family.family_identity or \
     candidate["program_key"] != family.binding.program_key or \
     candidate["binary_sha256"] != family.binding.binary_sha256:
    raise ValueError("candidate binding differs from frozen family PROGRAM")
  _digest(candidate["candidate_executable_identity"], "candidate executable")

  direct = _queue_rows(
    row["direct_bindings_by_queue"], _DIRECT_FIELDS,
    "direct bindings by queue")
  c6 = _queue_rows(row["c6_by_queue"], _C6_FIELDS, "C6 bindings by queue")
  transitions = _queue_rows(
    row["transition_preflight_bindings_by_queue"], _TRANSITION_FIELDS,
    "transition preflight bindings by queue")
  canaries = _mapping(row["runtime_canary_by_queue"], "runtime canaries")
  if set(canaries) != set(QUEUE_MODES):
    raise ValueError("runtime canaries must contain both queue modes")
  for queue in QUEUE_MODES:
    for field in _DIRECT_FIELDS:
      _digest(
        direct[queue][field], f"{queue} direct {field}",
        prefixed=field != "binary_sha256")
    if c6[queue]["status"] != "PASS" or \
       c6[queue]["fixture_schema"] != FIXTURE_SCHEMA or \
       c6[queue]["input_semantics"] != FP16_INPUT_SEMANTICS or \
       c6[queue]["legacy_fp32_prequantized"] is not False or \
       c6[queue]["family_identity"] != family.family_identity or \
       c6[queue]["workload_identity"] != fixture.workload_identity or \
       c6[queue]["input_identity"] != fixture.input_identity:
      raise ValueError(
        f"{queue} C6 is legacy FP32-prequantized or differs from v2 inputs")
    for field in (
        "evidence_identity", "candidate_correctness_identity",
        "comparator_identity"):
      _digest(c6[queue][field], f"{queue} C6 {field}")
    for field in ("device_identity", "software_identity"):
      if not isinstance(c6[queue][field], str) or not c6[queue][field]:
        raise ValueError(f"{queue} C6 {field} must be non-empty")
    for field in _TRANSITION_FIELDS:
      _digest(transitions[queue][field], f"{queue} transition {field}")
    canary = _mapping(canaries[queue], f"{queue} runtime canary")
    if canary.get("schema") != \
         "tinygrad.mmq_q4k_q8_1.frozen_staged_runtime_canary.v1" or \
       canary.get("status") != "PASS" or \
       canary.get("all_checks_pass") is not True or \
       canary.get("queue_mode") != queue or \
       canary.get("family_identity") != family.family_identity or \
       canary.get("program_key") != family.binding.program_key or \
       canary.get("binary_sha256") != family.binding.binary_sha256 or \
       canary.get("compile_performed") is not False or \
       canary.get("requires_recompile") is not False or \
       canary.get("amd_aql_effective") is not (queue == "AQL") or \
       canary.get("exact_blocker") is not None:
      raise ValueError(f"{queue} C4 runtime canary differs")
  if direct["PM4"]["qualification_identity"] == \
     direct["AQL"]["qualification_identity"] or \
     c6["PM4"]["evidence_identity"] == c6["AQL"]["evidence_identity"]:
    raise ValueError("PM4/AQL qualification and C6 identities must be distinct")
  _digest(row["joint_session_c7_identity"], "joint-session C7 identity")
  _digest(
    row["matched_timing_contract_identity"], "matched timing contract identity")
  return row


def build_ffn_gate_up_c8_runtime_composition(
    *, family: FrozenStagedFamily, fixture: FfnGateUpV2Fixture,
    candidate_binding: Mapping[str, Any],
    direct_bindings_by_queue: Mapping[str, Mapping[str, Any]],
    c6_by_queue: Mapping[str, Mapping[str, Any]],
    joint_session_c7_identity: str,
    transition_preflight_bindings_by_queue: Mapping[str, Mapping[str, Any]],
    runtime_canary_by_queue: Mapping[str, Mapping[str, Any]],
    matched_timing_contract_identity: str,
    ) -> dict[str, Any]:
  """Build and self-validate one content-addressed runtime composition."""
  payload = {
    "schema": COMPOSITION_SCHEMA, "status": "READY", "role": FFN_ROLE,
    "family_identity": family.family_identity,
    "execution_fixture_identity": fixture.fixture_identity,
    "workload_identity": fixture.workload_identity,
    "input_identity": fixture.input_identity,
    "logical_q4_identity": fixture.logical_q4_identity,
    "resident_fp16_activation_identity":
      fixture.resident_fp16_activation_identity,
    "candidate_binding": dict(candidate_binding),
    "direct_bindings_by_queue": {
      queue: dict(direct_bindings_by_queue[queue]) for queue in QUEUE_MODES},
    "c6_by_queue": {
      queue: dict(c6_by_queue[queue]) for queue in QUEUE_MODES},
    "joint_session_c7_identity": joint_session_c7_identity,
    "transition_preflight_bindings_by_queue": {
      queue: dict(transition_preflight_bindings_by_queue[queue])
      for queue in QUEUE_MODES},
    "runtime_canary_by_queue": {
      queue: dict(runtime_canary_by_queue[queue]) for queue in QUEUE_MODES},
    "matched_timing_contract_identity": matched_timing_contract_identity,
    "promotion_eligible_on_candidate_win": False,
  }
  row = {**payload, "composition_identity": _identity(payload)}
  return _composition(row, family=family, fixture=fixture)


def _contract_kwargs(composition: Mapping[str, Any]) -> dict[str, Any]:
  return {
    "workload_identity": composition["workload_identity"],
    "input_identity": composition["input_identity"],
    "logical_q4_identity": composition["logical_q4_identity"],
    "resident_fp16_activation_identity":
      composition["resident_fp16_activation_identity"],
    "candidate_binding": composition["candidate_binding"],
    "direct_bindings_by_queue": composition["direct_bindings_by_queue"],
    "joint_session_c7_identity": composition["joint_session_c7_identity"],
    "c6_bindings_by_queue": {
      queue: {
        key: composition["c6_by_queue"][queue][key]
        for key in (
          "evidence_identity", "candidate_correctness_identity",
          "comparator_identity", "workload_identity", "input_identity")
      } for queue in QUEUE_MODES
    },
    "transition_preflight_bindings_by_queue":
      composition["transition_preflight_bindings_by_queue"],
  }


def _bind_qualification_files(
    paths: Mapping[str, Path], composition: Mapping[str, Any],
    ) -> None:
  for queue in QUEUE_MODES:
    value = read_json(paths[queue], f"{queue} direct qualification")
    direct = composition["direct_bindings_by_queue"][queue]
    fallback = _mapping(
      value.get("fallback_evidence"), f"{queue} qualification fallback")
    checks = {
      "qualification_identity":
        value.get("qualification_identity") == direct["qualification_identity"],
      "queue_mode": value.get("queue_mode") == queue,
      "status": value.get("status") == "PASS",
      "executable_identity":
        fallback.get("executable_identity") == direct["executable_identity"],
      "binary_sha256":
        fallback.get("binary_sha256") == direct["binary_sha256"],
      "workload_identity":
        fallback.get("workload_identity") == composition["workload_identity"],
      "input_identity":
        fallback.get("input_identity") == composition["input_identity"],
    }
    if not all(checks.values()):
      failed = sorted(key for key, passed in checks.items() if not passed)
      raise ValueError(
        f"{queue} qualification path binding differs: {failed!r}")


@dataclass(frozen=True)
class FfnGateUpC8RuntimeConfig:
  family: FrozenStagedFamily
  composition: Mapping[str, Any]
  fixture: FfnGateUpV2Fixture
  matched_timing_contract: Mapping[str, Any]
  contract_validation_kwargs: Mapping[str, Any]
  qualification_paths_by_queue: Mapping[str, Path]
  frozen_bundle: Path
  staged_family_manifest: Path


def load_ffn_gate_up_c8_runtime_config(
    config: Mapping[str, Any], *, family: FrozenStagedFamily | None = None,
    family_loader: Callable[..., FrozenStagedFamily] =
      load_frozen_staged_family_manifest,
    fixture_rebuilder: Callable[..., FfnGateUpV2Fixture] =
      rebuild_ffn_gate_up_v2_fixture,
    ) -> FfnGateUpC8RuntimeConfig:
  """Load and cross-bind C4/C6/C7/C8 authorities before any GPU object exists."""
  config = dict(_mapping(config, "ffn_gate_up runtime config"))
  _exact_keys(config, _CONFIG_FIELDS, "ffn_gate_up runtime config")
  frozen_bundle = Path(config["frozen_bundle"]).resolve()
  family_manifest = Path(config["staged_family_manifest"]).resolve()
  if family is None:
    family = family_loader(
      family_manifest, role_spec=exact_role_spec(FFN_ROLE),
      frozen_bundle=frozen_bundle)
  _exact_role(family.binding.role_spec)

  raw_fixture = read_json(
    config["execution_fixture_v2"], "ffn_gate_up v2 execution fixture")
  fixture = fixture_rebuilder(family.binding.role_spec, raw_fixture)
  if not isinstance(fixture, FfnGateUpV2Fixture):
    raise TypeError("fixture rebuilder must return FfnGateUpV2Fixture")
  composition = _composition(
    read_json(config["composition"], "ffn_gate_up C8 composition"),
    family=family, fixture=fixture)
  kwargs = _contract_kwargs(composition)
  contract = validate_ffn_gate_up_matched_complete_role_timing_contract(
    read_json(
      config["matched_timing_contract"],
      "ffn_gate_up matched timing contract"),
    **kwargs)
  if contract["evidence_identity"] != \
     composition["matched_timing_contract_identity"]:
    raise ValueError("matched timing contract identity differs from composition")
  paths = qualification_paths(config)
  _bind_qualification_files(paths, composition)
  return FfnGateUpC8RuntimeConfig(
    family, composition, fixture, contract, kwargs, paths,
    frozen_bundle, family_manifest)


def build_ffn_gate_up_direct_packed_objects(
    loaded: FfnGateUpC8RuntimeConfig, *,
    object_builder: DirectPackedObjectBuilder | None = None,
    ) -> DirectPackedObjects:
  """Build direct-packed objects from the same resident FP16 bytes as C8."""
  if not isinstance(loaded, FfnGateUpC8RuntimeConfig):
    raise TypeError("loaded ffn_gate_up runtime config is required")
  activation = loaded.fixture.resident_fp16_activation.reshape(1, M, K)
  return build_direct_packed_objects(
    role=loaded.family.binding.role_spec, words=loaded.fixture.words,
    activation=activation, activation_dtype="float16",
    object_builder=object_builder)


def ffn_gate_up_queue_attestation_bindings(
    loaded: FfnGateUpC8RuntimeConfig, *, clock_identity: str,
    ) -> dict[str, Any]:
  return queue_attestation_bindings(
    loaded.composition["c6_by_queue"], clock_identity=clock_identity)


def ffn_gate_up_candidate_inputs(
    loaded: FfnGateUpC8RuntimeConfig, *,
    resident_fp16_activation: Any | None = None,
    ) -> FfnGateUpCandidateInputs:
  """Expose validated references with an optional shared resident GPU object."""
  if not isinstance(loaded, FfnGateUpC8RuntimeConfig):
    raise TypeError("loaded ffn_gate_up runtime config is required")
  fixture = loaded.fixture
  if resident_fp16_activation is None:
    resident_fp16_activation = fixture.resident_fp16_activation.reshape(
      1, M, K)
  return FfnGateUpCandidateInputs(
    role_spec=loaded.family.binding.role_spec,
    words=fixture.words, q4_epoch_major=fixture.q4_epoch_major,
    resident_fp16_activation=resident_fp16_activation,
    q8_producer_semantics=
      "per_invocation_from_resident_fp16_inside_outer_synchronized_wall",
    q8_reference_sha256={
      "values": _bytes_sha256(fixture.q8_values),
      "scales": _bytes_sha256(fixture.q8_scales),
      "sums": _bytes_sha256(fixture.q8_sums),
    },
    fixture_identity=fixture.fixture_identity,
    workload_identity=fixture.workload_identity,
    input_identity=fixture.input_identity,
    logical_q4_identity=fixture.logical_q4_identity,
    resident_fp16_activation_identity=
      fixture.resident_fp16_activation_identity)


def compose_ffn_gate_up_queue_runners(
    loaded: FfnGateUpC8RuntimeConfig, *, queue_mode: str,
    clock_identity: str,
    object_builder: DirectPackedObjectBuilder | None = None,
    candidate_route_builder: Callable[..., Any] | None = None,
    direct_route_builder: Callable[..., Any] | None = None,
    ) -> FfnGateUpOuterWallRoutes:
  """Build untimed routes for the dedicated matched outer-wall worker.

  There are deliberately no production defaults here.  Qo's staged candidate
  runner rebuilds the legacy FP32/prequantized fixture and both old timing
  runners emit receipt schemas rejected by the ffn paired collector.  A caller
  must therefore provide route builders that consume these v2 inputs and whose
  callbacks are later wrapped by ``run_ffn_gate_up_outer_synchronized_wall``.
  """
  if not callable(candidate_route_builder) or not callable(direct_route_builder):
    raise ValueError(
      "ffn_gate_up requires explicit production-faithful candidate and direct "
      "outer-wall route builders")
  if queue_mode not in QUEUE_MODES:
    raise ValueError(f"queue_mode must be one of {QUEUE_MODES!r}")
  direct_objects = build_ffn_gate_up_direct_packed_objects(
    loaded, object_builder=object_builder)
  candidate_inputs = ffn_gate_up_candidate_inputs(
    loaded, resident_fp16_activation=direct_objects.activation)
  bindings = ffn_gate_up_queue_attestation_bindings(
    loaded, clock_identity=clock_identity)
  shared = {
    "queue_mode": queue_mode,
    "matched_timing_contract": loaded.matched_timing_contract,
    "contract_validation_kwargs": loaded.contract_validation_kwargs,
  }
  candidate = candidate_route_builder(
    **shared, family=loaded.family,
    frozen_bundle=loaded.frozen_bundle,
    staged_family_manifest=loaded.staged_family_manifest,
    runtime_canary_by_queue=loaded.composition["runtime_canary_by_queue"],
    candidate_inputs=candidate_inputs)
  if not isinstance(candidate, FfnGateUpRouteCallback):
    raise TypeError(
      "ffn_gate_up candidate route builder must return "
      "FfnGateUpRouteCallback")
  direct = direct_route_builder(
    **shared,
    direct_objects=direct_objects,
    qualification_paths_by_queue=loaded.qualification_paths_by_queue,
    bindings_by_queue=bindings)
  if not isinstance(direct, FfnGateUpRouteCallback):
    raise TypeError(
      "ffn_gate_up direct route builder must return FfnGateUpRouteCallback")
  return FfnGateUpOuterWallRoutes(candidate, direct).validate(
    queue_mode=queue_mode, input_identity=loaded.fixture.input_identity,
    candidate_executable_identity=
      loaded.composition["candidate_binding"]["candidate_executable_identity"],
    direct_executable_identity=
      loaded.composition["direct_bindings_by_queue"][queue_mode][
        "executable_identity"])


__all__ = [
  "ACTIVATION_IDENTITY_SCHEMA", "COMPOSITION_SCHEMA", "FFN_ROLE",
  "FFN_SEEDS", "FIXTURE_SCHEMA", "FP16_INPUT_SEMANTICS",
  "OUTER_WALL_WRAPPER", "OUTPUT_REALIZATION_SEMANTICS",
  "FfnGateUpC8RuntimeConfig", "FfnGateUpV2Fixture",
  "FfnGateUpCandidateInputs", "FfnGateUpOuterWallRoutes",
  "FfnGateUpNoReadbackOutputRealizer", "FfnGateUpRouteCallback",
  "build_ffn_gate_up_c8_runtime_composition",
  "build_ffn_gate_up_direct_packed_objects",
  "build_ffn_gate_up_v2_fixture_manifest",
  "compose_ffn_gate_up_queue_runners",
  "ffn_gate_up_candidate_inputs",
  "ffn_gate_up_queue_attestation_bindings",
  "load_ffn_gate_up_c8_runtime_config", "rebuild_ffn_gate_up_v2_fixture",
  "resident_fp16_roundtrip",
]
