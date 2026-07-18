"""Scheduler-owned consumer for an exact frozen Q4_K/Q8_1 PROGRAM.

The frozen artifact already contains the native five-buffer ``Ops.PROGRAM``.
This adapter binds that PROGRAM through ``Tensor.custom_kernel`` and carries
the returned output ``AFTER`` value into the next K256 epoch.  It constructs a
lazy tinygrad graph only: no runtime is created, no device is opened, and no
kernel is dispatched while the route is being built.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops, UOp

from extra.qk.mmq_exact_role_spec import DEFAULT_INVENTORY, ExactRoleSpec
from extra.qk.mmq_frozen_target_artifact import FrozenTargetArtifact, load_frozen_target_artifact
from extra.qk.prefill.frozen_exact_role_runtime import (
  ABI_NAMES, PROGRAM_DEVICE, Q4_WORDS_PER_EPOCH_ROW, FrozenExactRoleBinding, _q4_epoch, _q8_epoch,
  load_frozen_exact_role_binding,
)
from extra.qk.q4k_q8_activation_producer import (
  PhysicalDS4Q8ActivationSpec, Q4KQ8ActivationTile, produce_physical_ds4_q8_1_tensor,
)


SCHEDULE_EVIDENCE_SCHEMA = "tinygrad.prefill_frozen_exact_schedule.v1"


@dataclass(frozen=True)
class FrozenExactRoleSchedule:
  output: Tensor
  binding: FrozenExactRoleBinding
  evidence: Mapping[str, Any]


def _require_amd_tensor(value: Any, *, name: str) -> Tensor:
  if not isinstance(value, Tensor) or not isinstance(value.uop, UOp):
    raise TypeError(f"{name} must be a tinygrad Tensor")
  if value.device != PROGRAM_DEVICE:
    raise ValueError(f"{name} device differs from frozen PROGRAM device")
  return value


def _validate_program_effects(binding: FrozenExactRoleBinding) -> None:
  program = binding.artifact.program
  if program.op is not Ops.PROGRAM or tuple(program.arg.globals) != tuple(range(5)):
    raise ValueError("scheduler route requires the exact five-buffer frozen PROGRAM")
  # Slot zero is an in-place FP32 accumulator. Both declarations are required:
  # the scheduler uses them for dependency tracking and TinyJit graph replay.
  if tuple(program.arg.outs) != (0,) or tuple(program.arg.ins) != tuple(range(5)):
    raise ValueError("frozen PROGRAM lost its in-place accumulator side-effect contract")


def validate_frozen_schedule_evidence(evidence: Mapping[str, Any], role_spec: ExactRoleSpec,
                                      *, expected_dispatch_count: int | None = None) -> dict[str, Any]:
  if not isinstance(evidence, Mapping) or evidence.get("schema") != SCHEDULE_EVIDENCE_SCHEMA:
    raise ValueError("frozen scheduler evidence schema is missing or invalid")
  runtime, operands, dispatch = evidence.get("runtime"), evidence.get("operands"), evidence.get("dispatch")
  if not isinstance(runtime, Mapping) or runtime.get("device") != PROGRAM_DEVICE or \
     runtime.get("created_during_graph_build") is not False or runtime.get("gpu_dispatch_during_graph_build") is not False:
    raise ValueError("frozen scheduler graph crossed or lost its runtime boundary")
  if not isinstance(operands, Mapping) or operands.get("mode") != "scheduler_owned_epoch_tensors" or \
     operands.get("epoch_k") != 256 or operands.get("q4_layout") != "q4_k_words[n, k256_epoch, 36]" or \
     operands.get("q8_layout") != "q8_1_mmq_ds4_transposed_blocks" or \
     operands.get("host_fixed_va_staging") is not False:
    raise ValueError("frozen scheduler operand plan differs from the exact epoch ABI")
  dispatch_count = role_spec.epochs if expected_dispatch_count is None else expected_dispatch_count
  if not isinstance(dispatch, Mapping) or dispatch.get("mode") != "lazy_ops_program_chain" or \
     dispatch.get("count") != dispatch_count or dispatch.get("program_key") in (None, "") or \
     dispatch.get("eager_native_runtime") is not False or dispatch.get("scheduler_visible") is not True or \
     dispatch.get("tinyjit_replay_visible") is not True:
    raise ValueError("frozen scheduler PROGRAM chain is incomplete or not replay-visible")
  return {
    "schema": SCHEDULE_EVIDENCE_SCHEMA,
    "runtime": dict(runtime), "operands": dict(operands), "dispatch": dict(dispatch),
  }


def build_frozen_exact_q4k_schedule(lin: Any, activation: Tensor, *, role_spec: ExactRoleSpec,
                                    frozen_bundle: str | Path, enabled: bool = False,
                                    inventory: str | Path | Mapping[str, Any] = DEFAULT_INVENTORY,
                                    binding: FrozenExactRoleBinding | None = None,
                                    artifact_loader: Callable[[str | Path], FrozenTargetArtifact] =
                                      load_frozen_target_artifact,
                                    activation_producer: Callable[[Any, PhysicalDS4Q8ActivationSpec],
                                                                  Q4KQ8ActivationTile] =
                                      produce_physical_ds4_q8_1_tensor,
                                    epoch_limit: int | None = None,
                                    fixed_scale_stage: bool = False,
                                    ) -> FrozenExactRoleSchedule | None:
  """Build the exact lazy epoch chain; disabled means no artifact or Tensor work."""
  if not enabled: return None
  if binding is None:
    binding = load_frozen_exact_role_binding(
      role_spec, frozen_bundle, inventory=inventory, artifact_loader=artifact_loader)
  elif not isinstance(binding, FrozenExactRoleBinding) or binding.role_spec != role_spec or \
       binding.candidate_identity != role_spec.candidate_canonical_identity:
    raise ValueError("preloaded frozen binding differs from the exact admitted role")
  _validate_program_effects(binding)
  if getattr(lin, "bias", None) is not None or getattr(lin, "out_features", None) != role_spec.n or \
     getattr(lin, "in_features", None) != role_spec.k or not hasattr(lin, "q4k_storage") or \
     not callable(getattr(lin, "prefill_packed_weight", None)):
    raise ValueError("scheduler linear differs from exact bias-free Q4_K role")
  activation = _require_amd_tensor(activation, name="activation")
  if tuple(activation.shape) != (role_spec.m, role_spec.k):
    raise ValueError("activation shape differs from exact admitted role")
  dispatch_count = role_spec.epochs if epoch_limit is None else epoch_limit
  if not isinstance(dispatch_count, int) or isinstance(dispatch_count, bool) or not 1 <= dispatch_count <= role_spec.epochs:
    raise ValueError(f"epoch_limit must be in [1,{role_spec.epochs}]")
  packed_weight = _require_amd_tensor(lin.prefill_packed_weight(), name="packed Q4_K weight")

  # The zero fill and all epoch calls stay lazy. custom_kernel returns slot zero
  # wrapped in AFTER(call); feeding that Tensor into the following epoch is the
  # in-place accumulation ordering edge.
  # Derive the zero accumulator from an admitted operand. A shaped
  # ``UOp.unique_const`` can be inspected as a graph but is not a valid tensor
  # function when the scheduler lowers it. Build the multiply through
  # ``_apply_uop`` so its scalar zero does not consult Device.DEFAULT while
  # @function has ALLOW_DEVICE_USAGE=0; the result retains AMD device lineage
  # and lowers to the ordinary scheduler-owned zero fill.
  output_seed = activation.flatten()[:1].cast(dtypes.float32)
  output = output_seed._apply_uop(lambda u: u.mul(0)).expand(role_spec.m * role_spec.n).contiguous()
  program = binding.artifact.program
  if not isinstance(fixed_scale_stage, bool): raise TypeError("fixed_scale_stage must be a bool")
  scale_stage: Tensor | None = None
  for epoch in range(dispatch_count):
    q4 = _require_amd_tensor(_q4_epoch(packed_weight, role_spec, epoch), name=f"Q4 epoch {epoch}")
    q8 = _q8_epoch(activation, role_spec, epoch, activation_producer)
    q8_values = _require_amd_tensor(q8.values, name=f"Q8 values epoch {epoch}")
    q8_scales = _require_amd_tensor(q8.scales, name=f"Q8 scales epoch {epoch}")
    q8_sums = _require_amd_tensor(q8.sums, name=f"Q8 sums epoch {epoch}")
    if fixed_scale_stage:
      # Diagnostic only: retained two-epoch GPU validation faults even though
      # the published kernargs are correct. The v2 static-offset route replaces
      # this mutable staging design; never enable it from the research route.
      if scale_stage is None: scale_stage = q8_scales.contiguous()
      else: scale_stage.assign(q8_scales)
      call_results = output.custom_kernel(
        q4, q8_values, scale_stage, q8_sums, fxn=lambda *_buffers, program=program: program)
      output, scale_stage = call_results[0], call_results[3]
    else:
      output = output.custom_kernel(
        q4, q8_values, q8_scales, q8_sums, fxn=lambda *_buffers, program=program: program)[0]

  execution = validate_frozen_schedule_evidence({
    "schema": SCHEDULE_EVIDENCE_SCHEMA,
    "runtime": {
      "device": PROGRAM_DEVICE,
      "created_during_graph_build": False,
      "gpu_dispatch_during_graph_build": False,
      "launcher": "tinygrad_scheduler",
    },
    "operands": {
      "mode": "scheduler_owned_epoch_tensors",
      "epoch_k": 256,
      "q4_layout": "q4_k_words[n, k256_epoch, 36]",
      "q8_layout": "q8_1_mmq_ds4_transposed_blocks",
      "host_fixed_va_staging": False,
      "q8_scale_binding": (
        "diagnostic_scheduler_owned_fixed_va_assign" if fixed_scale_stage else "direct_epoch_tensor"),
      "abi_names": list(ABI_NAMES),
    },
    "dispatch": {
      "mode": "lazy_ops_program_chain",
      "count": dispatch_count,
      "program_key": binding.program_key,
      "eager_native_runtime": False,
      "scheduler_visible": True,
      "tinyjit_replay_visible": True,
    },
  }, role_spec, expected_dispatch_count=dispatch_count)
  evidence = {
    "schema": "tinygrad.prefill_frozen_exact_role_scheduler.v1",
    "research_only": True, "default_off": True,
    "mmq_compile_performed": False, "mmq_requires_recompile": False, "hip_used": False,
    "role": role_spec.role, "shape": list(role_spec.shape),
    "program_shape": list(role_spec.program.shape),
    "candidate_identity": binding.candidate_identity, "program_key": binding.program_key,
    "source_sha256": binding.source_sha256, "binary_sha256": binding.binary_sha256,
    "artifact_role": binding.artifact_role_spec.role,
    "shared_program_geometry": binding.shared_program_geometry,
    "execution": execution,
    "dispatch_count": dispatch_count, "expected_dispatch_count": role_spec.epochs,
    "complete_role": dispatch_count == role_spec.epochs,
  }
  return FrozenExactRoleSchedule(output.reshape(1, role_spec.m, role_spec.n), binding, evidence)


__all__ = ["SCHEDULE_EVIDENCE_SCHEMA", "FrozenExactRoleSchedule", "build_frozen_exact_q4k_schedule",
           "validate_frozen_schedule_evidence"]
