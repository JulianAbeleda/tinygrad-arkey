"""Default-off tinygrad runtime adapter for an exact frozen Q4_K/Q8_1 role.

This module is a consumer, not an emitter.  It accepts only an inventory-
admitted role and an already-frozen PROGRAM, retains the model's exact packed
Q4_K storage, and dispatches one in-place K256 epoch at a time through
tinygrad's native runtime.  Hardware operations are isolated behind an
injectable boundary so admission and dispatch structure can be tested without
opening a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from math import prod
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops, UOp

from extra.qk.mmq_exact_role_spec import (
  DEFAULT_INVENTORY, EPOCH_K, ExactRoleSpec, exact_role_spec, load_exact_role_specs,
)
from extra.qk.mmq_frozen_target_artifact import (
  ACCUMULATION, BACKEND_ID, FUNCTION_NAME, LEGACY_SCHEMA, PROGRAM_DEVICE, SCHEMA, FrozenTargetArtifact,
  load_frozen_target_artifact,
)
from extra.qk.q4k_q8_activation_producer import (
  PhysicalDS4Q8ActivationSpec, Q4KQ8ActivationTile, produce_physical_ds4_q8_1,
)


ADAPTER_SCHEMA = "tinygrad.prefill_frozen_exact_role_runtime.v1"
EXECUTION_EVIDENCE_SCHEMA = "tinygrad.prefill_frozen_exact_execution.v1"
LOCAL_SIZE = (256, 1, 1)
Q4_WORDS_PER_EPOCH_ROW = 36
ABI_NAMES = ("output", "q4", "q8_values", "q8_scales", "q8_original_sums")
ABI_DTYPES = (dtypes.float32, dtypes.uint32, dtypes.int8, dtypes.float32, dtypes.float32)


def _sha256(data: bytes) -> str: return hashlib.sha256(data).hexdigest()


def _elements(value: Any) -> int:
  shape = getattr(value, "shape", None)
  if not isinstance(shape, (tuple, list)) or any(type(x) is not int or x < 0 for x in shape):
    raise ValueError("runtime operand must expose a concrete non-negative shape")
  return prod(shape)


def _require_tensor(value: Any, *, name: str, elements: int, dtype: Any) -> None:
  if _elements(value) != elements: raise ValueError(f"{name} element count differs from exact K256 ABI")
  if getattr(value, "dtype", None) != dtype: raise ValueError(f"{name} dtype differs from exact K256 ABI")


def _expected_abi(role_spec: ExactRoleSpec) -> tuple[dict[str, Any], ...]:
  return tuple({
    "slot": slot, "name": name, "dtype": f"{dtype}.ptr({elements})", "elements": elements,
  } for slot, (name, dtype, elements) in enumerate(
    zip(ABI_NAMES, ABI_DTYPES, role_spec.program.abi_elements)))


@dataclass(frozen=True)
class FrozenExactRoleBinding:
  role_spec: ExactRoleSpec
  artifact_role_spec: ExactRoleSpec
  artifact: FrozenTargetArtifact
  candidate_identity: str
  program_key: str
  source_sha256: str
  binary_sha256: str

  @property
  def shared_program_geometry(self) -> bool:
    return self.role_spec.program == self.artifact_role_spec.program


def load_frozen_exact_role_binding(role_spec: ExactRoleSpec, frozen_bundle: str | Path, *,
                                   inventory: str | Path | Mapping[str, Any] = DEFAULT_INVENTORY,
                                   artifact_loader: Callable[[str | Path], FrozenTargetArtifact] = load_frozen_target_artifact
                                   ) -> FrozenExactRoleBinding:
  """Join one admitted role to one validated frozen PROGRAM without compiling."""
  if not isinstance(role_spec, ExactRoleSpec): raise TypeError("exact frozen runtime requires ExactRoleSpec")
  admitted = exact_role_spec(role_spec.role, shape=role_spec.shape, inventory=inventory)
  if admitted != role_spec:
    raise ValueError("exact role spec differs from its inventory-admitted candidate identity")

  artifact = artifact_loader(frozen_bundle)
  if not isinstance(artifact, FrozenTargetArtifact):
    raise TypeError("frozen artifact loader returned the wrong artifact type")
  manifest, program = artifact.manifest, artifact.program
  if manifest.get("schema") not in (SCHEMA, LEGACY_SCHEMA) or manifest.get("state") != "FROZEN":
    raise ValueError("runtime artifact is not a frozen target bundle")
  if manifest.get("backend_id") != BACKEND_ID or manifest.get("accumulation") != ACCUMULATION or \
     manifest.get("accumulate") is not True:
    raise ValueError("runtime artifact is not the exact in-place accumulation backend")
  if manifest.get("gpu_runtime_initialized") is not False or manifest.get("gpu_dispatch_performed") is not False:
    raise ValueError("frozen artifact production crossed a GPU runtime boundary")
  consumer = manifest.get("consumer")
  if not isinstance(consumer, Mapping) or consumer.get("requires_recompile") is not False:
    raise ValueError("frozen artifact does not forbid consumer recompilation")

  artifact_shape = tuple(manifest.get("full_role_shape", ()))
  matches = [row for row in load_exact_role_specs(inventory) if row.shape == artifact_shape]
  if len(matches) != 1: raise ValueError("frozen artifact full-role shape is not uniquely inventory-admitted")
  artifact_role_spec = matches[0]
  if artifact.fixture.get("role", artifact_role_spec.role) != artifact_role_spec.role or \
     tuple(artifact.fixture.get("shape", ())) != artifact_role_spec.shape:
    raise ValueError("frozen fixture differs from its inventory-admitted role")
  if role_spec.program != artifact_role_spec.program:
    raise ValueError("frozen artifact K256 program geometry differs from requested role")
  if tuple(manifest.get("shape", ())) != role_spec.program.shape:
    raise ValueError("frozen artifact program shape differs from requested role")

  if not isinstance(program, UOp) or program.op is not Ops.PROGRAM:
    raise ValueError("frozen artifact does not retain a tinygrad PROGRAM")
  if program.arg.function_name != FUNCTION_NAME or tuple(program.arg.globals) != tuple(range(5)):
    raise ValueError("frozen PROGRAM function or five-buffer globals changed")
  if tuple(program.arg.global_size) != role_spec.program.grid or tuple(program.arg.local_size or ()) != LOCAL_SIZE:
    raise ValueError("frozen PROGRAM launch geometry differs from exact role")
  if len(program.src) < 5 or program.src[1].op is not Ops.DEVICE or program.src[1].arg != PROGRAM_DEVICE:
    raise ValueError("frozen PROGRAM is not a native AMD PROGRAM")
  sources = [u.arg for u in program.src if u.op is Ops.SOURCE]
  binaries = [u.arg for u in program.src if u.op is Ops.BINARY]
  if sources != [artifact.source] or binaries != [artifact.binary]:
    raise ValueError("frozen PROGRAM source or binary payload differs from retained artifact")

  artifacts = manifest.get("artifacts")
  if not isinstance(artifacts, Mapping) or artifacts.get("source_sha256") != _sha256(artifact.source.encode()) or \
     artifacts.get("binary_sha256") != _sha256(artifact.binary):
    raise ValueError("frozen source or binary identity differs from manifest")
  program_manifest = manifest.get("program")
  if not isinstance(program_manifest, Mapping) or program_manifest.get("key") != program.key.hex() or \
     tuple(program_manifest.get("global_size", ())) != role_spec.program.grid or \
     tuple(program_manifest.get("local_size", ())) != LOCAL_SIZE or \
     tuple(program_manifest.get("globals", ())) != tuple(range(5)) or \
     tuple(program_manifest.get("abi", ())) != _expected_abi(role_spec):
    raise ValueError("frozen PROGRAM key, grid, or ABI differs from exact role")

  return FrozenExactRoleBinding(role_spec, artifact_role_spec, artifact, role_spec.candidate_canonical_identity,
                                program.key.hex(), artifacts["source_sha256"], artifacts["binary_sha256"])


@dataclass(frozen=True)
class RuntimeBufferSpec:
  name: str
  elements: int
  dtype: Any

  @property
  def nbytes(self) -> int: return self.elements * self.dtype.itemsize


class FrozenRuntimeBoundary(Protocol):
  device: str
  synchronized_epoch_dispatch: bool
  def allocate(self, spec: RuntimeBufferSpec) -> Any: ...
  def zero(self, output: Any) -> None: ...
  def stage(self, destination: Any, source: Any, *, name: str, epoch: int) -> None: ...
  def create_runtime(self, program: UOp) -> Any: ...
  def execution_evidence(self, runtime: Any, buffers: tuple[Any, ...]) -> Mapping[str, Any]: ...
  def dispatch(self, runtime: Any, buffers: tuple[Any, ...], *, program: UOp, epoch: int) -> None: ...
  def finish(self, output: Any, shape: tuple[int, int, int]) -> Any: ...


def _callable_class_name(value: Any) -> str:
  value = getattr(value, "func", value)
  typ = value if isinstance(value, type) else type(value)
  return f"{typ.__module__}.{typ.__qualname__}"


class TinygradFrozenRuntimeBoundary:
  """Native tinygrad allocation/runtime boundary; construction does not open a device."""
  device = PROGRAM_DEVICE
  synchronized_epoch_dispatch = True

  def allocate(self, spec: RuntimeBufferSpec) -> Tensor:
    return Tensor.empty(spec.elements, dtype=spec.dtype, device=self.device).realize()

  def zero(self, output: Tensor) -> None:
    output.assign(Tensor.zeros(output.shape, dtype=output.dtype, device=self.device)).realize()

  def stage(self, destination: Tensor, source: Tensor, *, name: str, epoch: int) -> None:
    _ = name, epoch
    destination.assign(source.reshape(destination.shape)).realize()

  def create_runtime(self, program: UOp) -> Any:
    from tinygrad.engine.realize import get_runtime
    return get_runtime(self.device, program)

  def execution_evidence(self, runtime: Any, buffers: tuple[Tensor, ...]) -> Mapping[str, Any]:
    dev = getattr(runtime, "dev", None)
    if dev is None or getattr(dev, "device", None) != self.device:
      raise RuntimeError("frozen runtime is not bound to the admitted device")
    handles = tuple(buffer.uop.buffer.get_buf(self.device) for buffer in buffers)
    inputs = [{
      "slot": slot, "name": name, "va": int(handle.va_addr),
      "nbytes": int(buffer.uop.buffer.nbytes), "allocation_nbytes": int(handle.size),
    } for slot, (name, buffer, handle) in enumerate(zip(ABI_NAMES[1:], buffers[1:], handles[1:]), start=1)]
    return {
      "schema": EXECUTION_EVIDENCE_SCHEMA,
      "runtime": {
        "device": self.device,
        "amd_aql_env": os.environ.get("AMD_AQL"),
        "amd_aql_effective": bool(getattr(dev, "is_aql", False)),
        "queue_mode": "AQL" if bool(getattr(dev, "is_aql", False)) else "PM4",
        "queue_class": _callable_class_name(getattr(dev, "hw_compute_queue_t", None)),
        "runtime_class": f"{type(runtime).__module__}.{type(runtime).__qualname__}",
      },
      "staging": {
        "mode": "all_inputs_fixed_va_tinygrad_assign",
        "fixed_va": True,
        "persistent_buffers": True,
        "synchronized_before_overwrite": True,
        "transfer": "tinygrad_runtime_lowering",
        "inputs": inputs,
      },
    }

  def dispatch(self, runtime: Any, buffers: tuple[Tensor, ...], *, program: UOp, epoch: int) -> None:
    _ = epoch
    runtime(*(buffer.uop.buffer.get_buf(self.device) for buffer in buffers),
            global_size=program.arg.global_size, local_size=program.arg.local_size,
            vals=tuple(program.arg.vals({})), wait=True)

  def finish(self, output: Tensor, shape: tuple[int, int, int]) -> Tensor:
    return output.reshape(shape)


@dataclass(frozen=True)
class FrozenExactRoleRun:
  output: Any
  binding: FrozenExactRoleBinding
  evidence: Mapping[str, Any]


def _buffer_specs(role_spec: ExactRoleSpec) -> tuple[RuntimeBufferSpec, ...]:
  return tuple(RuntimeBufferSpec(name, elements, dtype) for name, elements, dtype in
               zip(ABI_NAMES, role_spec.program.abi_elements, ABI_DTYPES))


def validate_frozen_execution_evidence(evidence: Mapping[str, Any], role_spec: ExactRoleSpec) -> dict[str, Any]:
  """Validate queue selection and fixed-address input staging for one live frozen execution."""
  if not isinstance(evidence, Mapping) or evidence.get("schema") != EXECUTION_EVIDENCE_SCHEMA:
    raise ValueError("frozen execution evidence schema is missing or invalid")
  runtime, staging, dispatch = evidence.get("runtime"), evidence.get("staging"), evidence.get("dispatch")
  if not isinstance(runtime, Mapping) or runtime.get("device") != PROGRAM_DEVICE:
    raise ValueError("frozen execution evidence lacks the admitted AMD runtime")
  if runtime.get("queue_mode") not in ("PM4", "AQL") or type(runtime.get("amd_aql_effective")) is not bool or \
     runtime.get("queue_mode") != ("AQL" if runtime.get("amd_aql_effective") else "PM4"):
    raise ValueError("frozen execution queue mode is missing or inconsistent")
  if not all(isinstance(runtime.get(key), str) and runtime[key] for key in ("queue_class", "runtime_class")):
    raise ValueError("frozen execution runtime or queue class identity is missing")
  if not isinstance(staging, Mapping) or staging.get("mode") != "all_inputs_fixed_va_tinygrad_assign" or \
     staging.get("fixed_va") is not True or staging.get("persistent_buffers") is not True or \
     staging.get("synchronized_before_overwrite") is not True or staging.get("transfer") != "tinygrad_runtime_lowering":
    raise ValueError("frozen execution does not attest synchronized fixed-address input staging")
  inputs = staging.get("inputs")
  expected_specs = _buffer_specs(role_spec)[1:]
  if not isinstance(inputs, list) or len(inputs) != len(expected_specs):
    raise ValueError("frozen execution fixed-address input inventory is incomplete")
  for slot, (row, spec) in enumerate(zip(inputs, expected_specs), start=1):
    if not isinstance(row, Mapping) or row.get("slot") != slot or row.get("name") != spec.name or \
       type(row.get("va")) is not int or row["va"] <= 0 or row.get("nbytes") != spec.nbytes or \
       type(row.get("allocation_nbytes")) is not int or row["allocation_nbytes"] < row["nbytes"]:
      raise ValueError("frozen execution fixed-address input identity differs from the exact ABI")
  if len({row["va"] for row in inputs}) != len(inputs):
    raise ValueError("frozen execution input staging addresses are not distinct")
  if not isinstance(dispatch, Mapping) or dispatch.get("mode") != "eager_native_runtime" or \
     dispatch.get("count") != role_spec.epochs or dispatch.get("tinyjit_replay_captured") is not False:
    raise ValueError("frozen execution evidence lacks the exact eager dispatch count and replay boundary")
  return {
    "schema": EXECUTION_EVIDENCE_SCHEMA,
    "runtime": dict(runtime),
    "staging": {**dict(staging), "inputs": [dict(row) for row in inputs]},
    "dispatch": dict(dispatch),
  }


def _q4_epoch(packed_weight: Any, role_spec: ExactRoleSpec, epoch: int) -> Any:
  expected = role_spec.n * role_spec.epochs * Q4_WORDS_PER_EPOCH_ROW
  _require_tensor(packed_weight, name="packed Q4_K weight", elements=expected, dtype=dtypes.uint32)
  # GGUF/tinygrad packed Q4_K is N-major: [N, K256 epoch, 144 bytes].
  return packed_weight.reshape(role_spec.n, role_spec.epochs, Q4_WORDS_PER_EPOCH_ROW)[:, epoch, :].contiguous().reshape(-1)


def _q8_epoch(activation: Any, role_spec: ExactRoleSpec, epoch: int,
              producer: Callable[[Any, PhysicalDS4Q8ActivationSpec], Q4KQ8ActivationTile]) -> Q4KQ8ActivationTile:
  if tuple(getattr(activation, "shape", ())) != (role_spec.m, role_spec.k):
    raise ValueError("activation shape differs from exact admitted role")
  source = activation[:, epoch*EPOCH_K:(epoch+1)*EPOCH_K].cast(dtypes.float32).contiguous()
  tile = producer(source, PhysicalDS4Q8ActivationSpec(role_spec.m, EPOCH_K))
  expected = role_spec.program.abi_elements
  _require_tensor(tile.values, name="Q8 values", elements=expected[2], dtype=dtypes.int8)
  _require_tensor(tile.scales, name="Q8 scales", elements=expected[3], dtype=dtypes.float32)
  _require_tensor(tile.sums, name="Q8 original sums", elements=expected[4], dtype=dtypes.float32)
  return tile


def run_frozen_exact_q4k_research(lin: Any, activation: Any, *, role_spec: ExactRoleSpec,
                                  frozen_bundle: str | Path, enabled: bool = False,
                                  inventory: str | Path | Mapping[str, Any] = DEFAULT_INVENTORY,
                                  artifact_loader: Callable[[str | Path], FrozenTargetArtifact] = load_frozen_target_artifact,
                                  boundary: FrozenRuntimeBoundary | None = None,
                                  activation_producer: Callable[[Any, PhysicalDS4Q8ActivationSpec],
                                                                Q4KQ8ActivationTile] = produce_physical_ds4_q8_1
                                  ) -> FrozenExactRoleRun | None:
  """Run an explicitly enabled frozen research route; disabled means no work."""
  if not enabled: return None
  binding = load_frozen_exact_role_binding(role_spec, frozen_bundle, inventory=inventory, artifact_loader=artifact_loader)
  if getattr(lin, "bias", None) is not None or getattr(lin, "out_features", None) != role_spec.n or \
     getattr(lin, "in_features", None) != role_spec.k or not hasattr(lin, "q4k_storage") or \
     not callable(getattr(lin, "prefill_packed_weight", None)):
    raise ValueError("runtime linear differs from exact bias-free Q4_K role")
  packed_weight = lin.prefill_packed_weight()
  runtime_boundary = TinygradFrozenRuntimeBoundary() if boundary is None else boundary
  if getattr(runtime_boundary, "synchronized_epoch_dispatch", None) is not True:
    raise ValueError("fixed K256 staging requires a synchronized epoch dispatch boundary")
  specs = _buffer_specs(role_spec)
  buffers = tuple(runtime_boundary.allocate(spec) for spec in specs)
  if len({id(buffer) for buffer in buffers}) != len(buffers):
    raise RuntimeError("exact five-buffer ABI requires distinct persistent allocations")
  runtime_boundary.zero(buffers[0])
  runtime = runtime_boundary.create_runtime(binding.artifact.program)
  execution_evidence = dict(runtime_boundary.execution_evidence(runtime, buffers))
  dispatches = []
  for epoch in range(role_spec.epochs):
    q4 = _q4_epoch(packed_weight, role_spec, epoch)
    q8 = _q8_epoch(activation, role_spec, epoch, activation_producer)
    for destination, source, name in zip(buffers[1:], (q4, q8.values, q8.scales, q8.sums), ABI_NAMES[1:]):
      runtime_boundary.stage(destination, source, name=name, epoch=epoch)
    runtime_boundary.dispatch(runtime, buffers, program=binding.artifact.program, epoch=epoch)
    dispatches.append({"epoch": epoch, "global_size": list(role_spec.program.grid),
                       "local_size": list(LOCAL_SIZE), "program_key": binding.program_key})
  execution_evidence["dispatch"] = {
    "mode": "eager_native_runtime",
    "count": len(dispatches),
    # Native AMDProgram submissions are not scheduler UOps and therefore are
    # not captured for TinyJit replay.  Any whole-model authority must re-enter
    # this adapter for each measured prefill.
    "tinyjit_replay_captured": False,
  }
  execution_evidence = validate_frozen_execution_evidence(execution_evidence, role_spec)

  staging_specs = specs[1:]
  evidence = {
    "schema": ADAPTER_SCHEMA, "research_only": True, "default_off": True,
    "mmq_compile_performed": False, "mmq_requires_recompile": False, "hip_used": False,
    "q8_producer_and_staging_use_tinygrad_runtime_lowering": True,
    "q4_gather_and_staging_use_tinygrad_runtime_lowering": True,
    "synchronized_epoch_dispatch": True,
    "role": role_spec.role, "shape": list(role_spec.shape), "program_shape": list(role_spec.program.shape),
    "candidate_identity": binding.candidate_identity, "program_key": binding.program_key,
    "source_sha256": binding.source_sha256, "binary_sha256": binding.binary_sha256,
    "artifact_role": binding.artifact_role_spec.role,
    "shared_program_geometry": binding.shared_program_geometry,
    "execution": execution_evidence,
    "dispatch_count": len(dispatches), "expected_dispatch_count": role_spec.epochs, "dispatches": dispatches,
    "abi": [{"slot": slot, "name": spec.name, "elements": spec.elements,
             "dtype": str(spec.dtype), "nbytes": spec.nbytes} for slot, spec in enumerate(specs)],
    "staging": {
      "mode": "stable_one_k256_epoch", "q4_source_layout": "q4_k_words[n, k256_epoch, 36]",
      "q8_layout": "q8_1_mmq_ds4_transposed_blocks", "epoch_k": EPOCH_K,
      "fixed_va": execution_evidence["staging"]["fixed_va"],
      "persistent_inputs": execution_evidence["staging"]["inputs"],
      "transfer": execution_evidence["staging"]["transfer"],
      "elements": sum(spec.elements for spec in staging_specs),
      "bytes": sum(spec.nbytes for spec in staging_specs),
      "depends_on_epoch_count": False,
    },
  }
  if len(dispatches) != role_spec.epochs: raise RuntimeError("exact K256 dispatch count changed")
  return FrozenExactRoleRun(runtime_boundary.finish(buffers[0], (1, role_spec.m, role_spec.n)), binding, evidence)


__all__ = ["ADAPTER_SCHEMA", "EXECUTION_EVIDENCE_SCHEMA", "FrozenExactRoleBinding", "FrozenExactRoleRun", "FrozenRuntimeBoundary",
           "RuntimeBufferSpec", "TinygradFrozenRuntimeBoundary", "load_frozen_exact_role_binding",
           "run_frozen_exact_q4k_research", "validate_frozen_execution_evidence"]
