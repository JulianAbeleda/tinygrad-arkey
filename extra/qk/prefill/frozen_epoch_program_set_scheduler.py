"""Lazy fixed-base scheduler consumer for a frozen epoch PROGRAM set."""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any, Callable, Mapping

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops, UOp

from extra.qk.mmq_exact_role_spec import DEFAULT_INVENTORY, ExactRoleSpec, admit_exact_role_spec
from extra.qk.mmq_frozen_epoch_program_set import (
  BINDING_SCHEMA, LEGACY_SCHEMA, SCHEMA as ARTIFACT_SCHEMA, FrozenEpochProgramSetBinding,
  load_frozen_epoch_program_set_binding,
)
from extra.qk.mmq_frozen_target_artifact import PROGRAM_DEVICE
from extra.qk.mmq_llama_five_buffer_graph import five_buffer_parameters
from extra.qk.q4k_q8_activation_producer import (
  PhysicalDS4Q8ActivationSpec, Q4KQ8ActivationTile, produce_physical_ds4_q8_1_tensor,
)


SCHEDULE_SCHEMA = "tinygrad.prefill_frozen_epoch_program_set_schedule.v2"
EVIDENCE_SCHEMA = "tinygrad.prefill_frozen_epoch_program_set_execution.v2"
PREPARATION_SCHEMA = "tinygrad.prefill_frozen_epoch_program_set_preparation.v1"
PREPARATION_RECEIPT_SCHEMA = "tinygrad.prefill_frozen_epoch_program_set_preparation_receipt.v1"
ABI_NAMES = ("output", "q4", "q8_values", "q8_scales", "q8_original_sums")


@dataclass(frozen=True)
class FrozenEpochProgramSetPreparation:
  """The exact five ABI tensors before any frozen target PROGRAM is attached."""
  operands: tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
  binding: FrozenEpochProgramSetBinding
  role_spec: ExactRoleSpec
  dispatch_count: int
  require_c1: bool
  evidence: Mapping[str, Any]


@dataclass(frozen=True)
class FrozenEpochProgramSetSchedule:
  output: Tensor
  # The exact five ABI tensors before any frozen target PROGRAM is attached.
  # Realizing these together gives diagnostics a producer/output-initialization
  # phase without manufacturing a second launcher or changing target effects.
  preparation_outputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
  binding: FrozenEpochProgramSetBinding
  evidence: Mapping[str, Any]


def _require_amd_tensor(value: Any, *, name: str, elements: int | None = None,
                        dtype: Any | None = None) -> Tensor:
  if not isinstance(value, Tensor) or not isinstance(value.uop, UOp):
    raise TypeError(f"{name} must be a tinygrad Tensor")
  if value.device != PROGRAM_DEVICE: raise ValueError(f"{name} device differs from frozen PROGRAM device")
  if elements is not None and prod(value.shape) != elements:
    raise ValueError(f"{name} element count differs from the full-role ABI")
  if dtype is not None and value.dtype != dtype:
    raise ValueError(f"{name} dtype differs from the full-role ABI")
  return value


def _validate_binding(binding: FrozenEpochProgramSetBinding, role_spec: ExactRoleSpec, *,
                      require_c1: bool = False) -> None:
  if not isinstance(binding, FrozenEpochProgramSetBinding) or binding.schema != BINDING_SCHEMA:
    raise TypeError("scheduler requires a frozen v2 epoch PROGRAM set binding")
  if binding.role_spec != role_spec or binding.candidate_identity != role_spec.candidate_canonical_identity:
    raise ValueError("frozen v2 binding differs from the requested exact role")
  artifact_schema = binding.artifact.manifest.get("schema")
  certification = binding.artifact.manifest.get("c1_certification")
  current_c1 = artifact_schema == ARTIFACT_SCHEMA and isinstance(certification, Mapping) and \
    certification.get("gate") == "C1" and certification.get("certified") is True and \
    certification.get("content_addressed") is True
  diagnostic_legacy = artifact_schema == LEGACY_SCHEMA and isinstance(certification, Mapping) and \
    certification.get("gate") == "C1" and certification.get("certified") is False and \
    certification.get("status") == "legacy_v2_missing_generation_provenance" and \
    certification.get("content_addressed") is False
  if (not current_c1 and (require_c1 or not diagnostic_legacy)) or \
     len(binding.artifact.programs) != role_spec.epochs or len(binding.program_keys) != role_spec.epochs:
    raise ValueError("frozen v2 binding does not contain the complete exact epoch family")
  if tuple(program.key.hex() for program in binding.artifact.programs) != binding.program_keys:
    raise ValueError("frozen v2 binding PROGRAM keys differ from the loaded family")
  for program in binding.artifact.programs:
    if program.op is not Ops.PROGRAM or tuple(program.arg.globals) != tuple(range(5)) or \
       tuple(program.arg.outs) != (0,) or tuple(program.arg.ins) != tuple(range(5)):
      raise ValueError("frozen v2 PROGRAM lost its five-buffer in-place accumulator effects")


def validate_frozen_epoch_program_set_evidence(evidence: Mapping[str, Any],
                                               role_spec: ExactRoleSpec,
                                               dispatch_count: int) -> dict[str, Any]:
  if not isinstance(evidence, Mapping) or evidence.get("schema") != EVIDENCE_SCHEMA:
    raise ValueError("v2 scheduler evidence schema is missing or invalid")
  runtime, operands, dispatch = evidence.get("runtime"), evidence.get("operands"), evidence.get("dispatch")
  if not isinstance(runtime, Mapping) or runtime.get("device") != PROGRAM_DEVICE or \
     runtime.get("created_during_graph_build") is not False or \
     runtime.get("gpu_dispatch_during_graph_build") is not False or \
     runtime.get("launcher") != "tinygrad_scheduler":
    raise ValueError("v2 scheduler evidence crossed the lazy runtime boundary")
  if not isinstance(operands, Mapping) or operands.get("mode") != "full_role_fixed_base" or \
     operands.get("full_role_ds4_producer_calls") != 1 or operands.get("host_staging") is not False or \
     operands.get("fixed_base_slots") != list(range(5)) or \
     operands.get("all_calls_share_buffer_identity") is not True:
    raise ValueError("v2 scheduler evidence lacks the full-role fixed-base operand contract")
  expected_epochs = list(range(dispatch_count))
  if not isinstance(dispatch, Mapping) or dispatch.get("mode") != "static_offset_program_chain" or \
     dispatch.get("count") != dispatch_count or dispatch.get("selected_epochs") != expected_epochs or \
     dispatch.get("prefix_complete") is not True or \
     dispatch.get("full_role_complete") is not (dispatch_count == role_spec.epochs) or \
     dispatch.get("slot0_ordered") is not True or dispatch.get("eager_native_runtime") is not False:
    raise ValueError("v2 scheduler evidence has an incomplete or unordered PROGRAM prefix")
  keys = dispatch.get("program_keys")
  if not isinstance(keys, list) or len(keys) != dispatch_count or any(not isinstance(key, str) or not key for key in keys):
    raise ValueError("v2 scheduler evidence lacks exact selected PROGRAM identities")
  return {
    "schema": EVIDENCE_SCHEMA, "runtime": dict(runtime),
    "operands": dict(operands), "dispatch": dict(dispatch),
  }


def prepare_frozen_epoch_program_set_schedule(
    lin: Any, activation: Tensor, *, role_spec: ExactRoleSpec,
    frozen_bundle: str | Path, enabled: bool = False,
    prefix_epochs: int | None = None,
    require_c1: bool = False,
    inventory: str | Path | Mapping[str, Any] = DEFAULT_INVENTORY,
    binding: FrozenEpochProgramSetBinding | None = None,
    binding_loader: Callable[..., FrozenEpochProgramSetBinding] = load_frozen_epoch_program_set_binding,
    activation_producer: Callable[[Tensor, PhysicalDS4Q8ActivationSpec], Q4KQ8ActivationTile] =
      produce_physical_ds4_q8_1_tensor,
    ) -> FrozenEpochProgramSetPreparation | None:
  """Build the exact five operands without attaching a frozen target PROGRAM."""
  if not enabled: return None
  role_spec = admit_exact_role_spec(role_spec, inventory=inventory)
  if binding is None:
    binding = binding_loader(
      role_spec, frozen_bundle, inventory=inventory, **({"require_c1": True} if require_c1 else {}))
  _validate_binding(binding, role_spec, require_c1=require_c1)
  if getattr(lin, "bias", None) is not None or getattr(lin, "out_features", None) != role_spec.n or \
     getattr(lin, "in_features", None) != role_spec.k or not hasattr(lin, "q4k_storage") or \
     not callable(getattr(lin, "prefill_packed_weight", None)):
    raise ValueError("v2 scheduler linear differs from the exact bias-free Q4_K role")
  activation = _require_amd_tensor(
    activation, name="activation", elements=role_spec.m * role_spec.k)
  if tuple(activation.shape) != (role_spec.m, role_spec.k):
    raise ValueError("activation shape differs from the exact admitted role")
  dispatch_count = role_spec.epochs if prefix_epochs is None else prefix_epochs
  if not isinstance(dispatch_count, int) or isinstance(dispatch_count, bool) or \
     not 1 <= dispatch_count <= role_spec.epochs:
    raise ValueError(f"prefix_epochs must be in [1,{role_spec.epochs}]")

  parameters = five_buffer_parameters(*role_spec.shape)
  packed_weight = _require_amd_tensor(
    lin.prefill_packed_weight(), name="packed Q4_K weight",
    elements=parameters[1].size, dtype=dtypes.uint32)
  source = activation.cast(dtypes.float32).contiguous()
  ds4 = activation_producer(source, PhysicalDS4Q8ActivationSpec(role_spec.m, role_spec.k))
  q8_values = _require_amd_tensor(
    ds4.values, name="full-role Q8 values", elements=parameters[2].size, dtype=dtypes.int8)
  q8_scales = _require_amd_tensor(
    ds4.scales, name="full-role Q8 scales", elements=parameters[3].size, dtype=dtypes.float32)
  q8_sums = _require_amd_tensor(
    ds4.sums, name="full-role Q8 sums", elements=parameters[4].size, dtype=dtypes.float32)

  # All accumulating variants require a zeroed initial slot zero. Keep this
  # scheduler-owned and lazy while retaining AMD device lineage.
  output_seed = activation.flatten()[:1].cast(dtypes.float32)
  zero = output_seed._apply_uop(lambda u: u.mul(0)).expand(parameters[0].size)
  output = Tensor.empty(parameters[0].size, dtype=dtypes.float32, device=PROGRAM_DEVICE)
  output.assign(zero)
  fixed_inputs = (packed_weight, q8_values, q8_scales, q8_sums)
  preparation_outputs = (output, *fixed_inputs)
  fixed_keys = [output.uop.key.hex(), *(tensor.uop.key.hex() for tensor in fixed_inputs)]
  evidence = {
    "schema": PREPARATION_SCHEMA, "research_only": True, "default_off": True,
    "role": role_spec.role, "shape": list(role_spec.shape),
    "candidate_identity": binding.candidate_identity, "family_identity": binding.family_identity,
    "dispatch_count": dispatch_count, "expected_dispatch_count": role_spec.epochs,
    "target_programs_attached": False, "require_c1": require_c1,
    "initial_tensor_keys": fixed_keys,
    "abi_names": list(ABI_NAMES),
  }
  return FrozenEpochProgramSetPreparation(
    preparation_outputs, binding, role_spec, dispatch_count, require_c1, evidence)


def _attach_frozen_epoch_program_set_schedule(
    preparation: FrozenEpochProgramSetPreparation, *,
    require_realized_operands: bool,
    preparation_receipt: Mapping[str, Any] | None = None,
    ) -> FrozenEpochProgramSetSchedule:
  """Attach the selected PROGRAM prefix to the preparation's current UOps.

  The strict two-stage route calls this only after realizing and synchronizing
  ``preparation.operands``. The ordinary lazy schedule builder below composes
  the same primitives with ``require_realized_operands=False``.
  """
  if not isinstance(preparation, FrozenEpochProgramSetPreparation):
    raise TypeError("attachment requires a frozen epoch PROGRAM set preparation")
  if not isinstance(require_realized_operands, bool):
    raise TypeError("require_realized_operands must be a bool")
  binding, role_spec, dispatch_count = \
    preparation.binding, preparation.role_spec, preparation.dispatch_count
  if not isinstance(preparation.require_c1, bool):
    raise TypeError("frozen epoch preparation require_c1 marker must be a bool")
  _validate_binding(binding, role_spec, require_c1=preparation.require_c1)
  if not isinstance(dispatch_count, int) or isinstance(dispatch_count, bool) or \
     not 1 <= dispatch_count <= role_spec.epochs:
    raise ValueError("frozen epoch preparation dispatch count is invalid")
  expected_evidence = {
    "schema": PREPARATION_SCHEMA, "role": role_spec.role, "shape": list(role_spec.shape),
    "candidate_identity": binding.candidate_identity, "family_identity": binding.family_identity,
    "dispatch_count": dispatch_count, "expected_dispatch_count": role_spec.epochs,
    "target_programs_attached": False, "require_c1": preparation.require_c1,
    "abi_names": list(ABI_NAMES),
  }
  if any(preparation.evidence.get(key) != value for key, value in expected_evidence.items()) or \
     not isinstance(preparation.evidence.get("initial_tensor_keys"), list) or \
     len(preparation.evidence["initial_tensor_keys"]) != 5:
    raise ValueError("frozen epoch preparation evidence is missing or invalid")
  if len(preparation.operands) != 5:
    raise ValueError("frozen epoch preparation lost the exact five-buffer ABI")
  preparation_outputs = preparation.operands
  output, *fixed_input_list = preparation_outputs
  fixed_inputs = tuple(fixed_input_list)
  parameters = five_buffer_parameters(*role_spec.shape)
  for name, tensor, parameter, dtype in zip(
      ABI_NAMES, preparation_outputs, parameters,
      (dtypes.float32, dtypes.uint32, dtypes.int8, dtypes.float32, dtypes.float32)):
    _require_amd_tensor(tensor, name=f"prepared {name}", elements=parameter.size, dtype=dtype)
  if require_realized_operands and any(not tensor.uop.is_realized for tensor in preparation.operands):
    raise ValueError("strict frozen PROGRAM attachment requires all five preparation operands to be realized")

  # Read the Tensor UOps only at attachment time. Realization may replace a
  # lazy expression (notably packed Q4) with a newly allocated BUFFER UOp.
  # Attaching earlier would permanently capture the stale pre-realization UOp.
  attached_buffer_uops = tuple(tensor.uop.buf_uop for tensor in preparation.operands)
  if require_realized_operands and any(uop.op is not Ops.BUFFER for uop in attached_buffer_uops):
    raise ValueError("strict frozen PROGRAM attachment requires five direct BUFFER identities")
  if require_realized_operands:
    if not isinstance(preparation_receipt, Mapping) or \
       preparation_receipt.get("schema") != PREPARATION_RECEIPT_SCHEMA or \
       preparation_receipt.get("phase") != "producer_and_output_initialization" or \
       preparation_receipt.get("status") != "PASS" or \
       preparation_receipt.get("all_checks_pass") is not True or \
       preparation_receipt.get("target_dispatch_allowed") is not True or \
       not isinstance(preparation_receipt.get("checks"), Mapping) or \
       any(preparation_receipt["checks"].get(key) is not True for key in (
         "exact_five_slots", "all_vas_nonzero", "all_vas_distinct",
         "all_extents_nonempty", "all_allocations_cover_tensor_extents")) or \
       not isinstance(preparation_receipt.get("realize"), Mapping) or \
       preparation_receipt["realize"].get("began") is not True or \
       preparation_receipt["realize"].get("returned") is not True or \
       not isinstance(preparation_receipt.get("synchronize"), Mapping) or \
       preparation_receipt["synchronize"].get("began") is not True or \
       preparation_receipt["synchronize"].get("returned") is not True or \
       preparation_receipt["synchronize"].get("failure") is not None:
      raise ValueError("strict frozen PROGRAM attachment requires a successful synchronized preparation receipt")
    allocations = preparation_receipt.get("allocations")
    current_keys = [uop.key.hex() for uop in attached_buffer_uops]
    expected_nbytes = [
      int(parameter.size) * dtype.itemsize
      for parameter, dtype in zip(
        parameters, (dtypes.float32, dtypes.uint32, dtypes.int8, dtypes.float32, dtypes.float32))]
    current_allocations = []
    for slot, (name, tensor, key) in enumerate(zip(ABI_NAMES, preparation.operands, current_keys)):
      buffer = tensor.uop.buffer
      handle = buffer.get_buf(PROGRAM_DEVICE)
      current_allocations.append({
        "slot": slot, "name": name, "buffer_uop_key": key,
        "va": int(handle.va_addr), "nbytes": int(buffer.nbytes),
        "allocation_nbytes": int(handle.size),
      })
    if not isinstance(allocations, list) or len(allocations) != 5 or \
       any(not isinstance(row, Mapping) for row in allocations) or \
       [dict(row) for row in allocations] != current_allocations or \
       [row.get("nbytes") if isinstance(row, Mapping) else None for row in allocations] != expected_nbytes or \
       any(row["va"] <= 0 or row["allocation_nbytes"] < expected_nbytes[slot]
           for slot, row in enumerate(current_allocations)) or \
       len({row["va"] for row in allocations}) != 5:
      raise ValueError("synchronized preparation receipt differs from the current five buffer identities")
  programs = binding.artifact.programs[:dispatch_count]
  for program in programs:
    output = output.custom_kernel(
      *fixed_inputs, fxn=lambda *_buffers, program=program: program)[0]

  execution = validate_frozen_epoch_program_set_evidence({
    "schema": EVIDENCE_SCHEMA,
    "runtime": {
      "device": PROGRAM_DEVICE, "launcher": "tinygrad_scheduler",
      "created_during_graph_build": False, "gpu_dispatch_during_graph_build": False,
    },
    "operands": {
      "mode": "full_role_fixed_base",
      "q4_layout": "q4_k_words[n,k256_epoch,36]",
      "q8_layout": "q8_1_mmq_ds4_transposed_blocks",
      "full_role_ds4_producer_calls": 1, "host_staging": False,
      "fixed_base_slots": list(range(5)), "abi_names": list(ABI_NAMES),
      "all_calls_share_buffer_identity": True,
      "initial_tensor_keys": list(preparation.evidence["initial_tensor_keys"]),
      "attached_after_realization": require_realized_operands,
      "attached_buffer_uop_keys": [uop.key.hex() for uop in attached_buffer_uops],
    },
    "dispatch": {
      "mode": "static_offset_program_chain", "count": dispatch_count,
      "selected_epochs": list(range(dispatch_count)),
      "program_keys": list(binding.program_keys[:dispatch_count]),
      "prefix_complete": True, "full_role_complete": dispatch_count == role_spec.epochs,
      "slot0_ordered": True, "eager_native_runtime": False,
    },
  }, role_spec, dispatch_count)
  evidence = {
    "schema": SCHEDULE_SCHEMA, "research_only": True, "default_off": True,
    "mmq_compile_performed": False, "mmq_requires_recompile": False, "hip_used": False,
    "role": role_spec.role, "shape": list(role_spec.shape),
    "candidate_identity": binding.candidate_identity, "family_identity": binding.family_identity,
    "dispatch_count": dispatch_count, "expected_dispatch_count": role_spec.epochs,
    "prefix_complete": True, "complete_role": dispatch_count == role_spec.epochs,
    "two_stage_realized_attachment": require_realized_operands,
    "two_stage_synchronized_attachment": require_realized_operands,
    "execution": execution,
  }
  return FrozenEpochProgramSetSchedule(
    output.reshape(1, role_spec.m, role_spec.n), preparation_outputs, binding, evidence)


def attach_frozen_epoch_program_set_schedule(
    preparation: FrozenEpochProgramSetPreparation, *,
    preparation_receipt: Mapping[str, Any],
    ) -> FrozenEpochProgramSetSchedule:
  """Strict public attachment after successful realization and synchronization."""
  return _attach_frozen_epoch_program_set_schedule(
    preparation, require_realized_operands=True,
    preparation_receipt=preparation_receipt)


def build_frozen_epoch_program_set_schedule(
    lin: Any, activation: Tensor, *, role_spec: ExactRoleSpec,
    frozen_bundle: str | Path, enabled: bool = False,
    prefix_epochs: int | None = None,
    require_c1: bool = False,
    inventory: str | Path | Mapping[str, Any] = DEFAULT_INVENTORY,
    binding: FrozenEpochProgramSetBinding | None = None,
    binding_loader: Callable[..., FrozenEpochProgramSetBinding] = load_frozen_epoch_program_set_binding,
    activation_producer: Callable[[Tensor, PhysicalDS4Q8ActivationSpec], Q4KQ8ActivationTile] =
      produce_physical_ds4_q8_1_tensor,
    ) -> FrozenEpochProgramSetSchedule | None:
  """Compose preparation and attachment for the existing all-lazy graph API."""
  preparation = prepare_frozen_epoch_program_set_schedule(
    lin, activation, role_spec=role_spec, frozen_bundle=frozen_bundle, enabled=enabled,
    prefix_epochs=prefix_epochs, require_c1=require_c1, inventory=inventory,
    binding=binding, binding_loader=binding_loader, activation_producer=activation_producer)
  if preparation is None: return None
  return _attach_frozen_epoch_program_set_schedule(
    preparation, require_realized_operands=False, preparation_receipt=None)


__all__ = [
  "EVIDENCE_SCHEMA", "PREPARATION_RECEIPT_SCHEMA", "PREPARATION_SCHEMA", "SCHEDULE_SCHEMA",
  "FrozenEpochProgramSetPreparation", "FrozenEpochProgramSetSchedule",
  "attach_frozen_epoch_program_set_schedule", "build_frozen_epoch_program_set_schedule",
  "prepare_frozen_epoch_program_set_schedule", "validate_frozen_epoch_program_set_evidence",
]
