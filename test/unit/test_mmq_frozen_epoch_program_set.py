from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tinygrad import dtypes
from tinygrad.uop.ops import Ops, ProgramInfo, UOp

from extra.qk import mmq_frozen_epoch_program_set as frozen_v2
from extra.qk.mmq_exact_role_spec import ExactRoleSpec, exact_role_spec
from extra.qk.mmq_frozen_target_artifact import FUNCTION_NAME
from extra.qk.mmq_llama_five_buffer_full_kernel import (
  FullGridOwnerCoordinates, FullGridTopology, LlamaFiveBufferEpochOffsetFamily,
  LlamaFiveBufferFullKernel,
)
from extra.qk.mmq_llama_five_buffer_graph import five_buffer_parameters
from extra.qk.mmq_llama_runtime_contract import LLAMA_SOURCE_COMMIT


def _program(role_spec: ExactRoleSpec, epoch: int, *, encoded_epoch: int | None = None) -> UOp:
  encoded_epoch = epoch if encoded_epoch is None else encoded_epoch
  parameters = five_buffer_parameters(*role_spec.shape)
  params = tuple(UOp.param(parameter.slot, parameter.dtype.ptr(parameter.size)) for parameter in parameters)
  records = encoded_epoch * 2
  offsets = (
    0, encoded_epoch * 36, records * role_spec.m * 128,
    records * role_spec.m * 4, records * role_spec.m * 4,
  )
  sink = UOp(Ops.SINK, src=tuple(param.index(UOp.const(dtypes.weakint, offset), ptr=True)
                                  for param, offset in zip(params, offsets)))
  return UOp(Ops.PROGRAM, src=(
    sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR),
    UOp(Ops.SOURCE, arg=f"generated static epoch {epoch}\n"),
    UOp(Ops.BINARY, arg=f"fake-hsaco-epoch-{epoch}".encode()),
  ), arg=ProgramInfo(
    name=FUNCTION_NAME, globals=tuple(range(5)),
    global_size=role_spec.program.grid, local_size=(256, 1, 1),
  ))


def _family(role_spec: ExactRoleSpec, *, wrong_program_epoch: tuple[int, int] | None = None
            ) -> LlamaFiveBufferEpochOffsetFamily:
  parameters = five_buffer_parameters(*role_spec.shape)
  proof = SimpleNamespace(
    facts=SimpleNamespace(m=role_spec.m, n=role_spec.n, k=role_spec.k),
    parameters=parameters,
  )
  topology = FullGridTopology(role_spec.program.grid)
  owners = FullGridOwnerCoordinates(role_spec.m, role_spec.n)
  variants = []
  for epoch in range(role_spec.epochs):
    encoded = wrong_program_epoch[1] if wrong_program_epoch and wrong_program_epoch[0] == epoch else epoch
    program = _program(role_spec, epoch, encoded_epoch=encoded)
    variants.append(LlamaFiveBufferFullKernel(
      proof, topology, program.src[0], owners, LLAMA_SOURCE_COMMIT, tuple(),
      epoch_offset=epoch, blocker="", program=program, emitted=True,
    ))
  return LlamaFiveBufferEpochOffsetFamily(proof, topology, tuple(variants))


def _produce(tmp_path: Path, *, archive: bool = False):
  role_spec = exact_role_spec("ffn_gate_up")
  output = tmp_path / "v2-bundle"
  archive_path = tmp_path / "v2-bundle.tar" if archive else None
  manifest = frozen_v2.produce_frozen_epoch_program_set(
    output, role_spec=role_spec, build_once=lambda: _family(role_spec), archive=archive_path)
  return role_spec, output, archive_path, manifest


def test_v2_program_set_roundtrips_all_epochs_without_compile_or_runtime(tmp_path: Path):
  role_spec, output, archive, manifest = _produce(tmp_path, archive=True)
  assert archive is not None
  assert manifest["schema"] == frozen_v2.SCHEMA
  assert manifest["family_builder_calls"] == 1 and manifest["variant_count"] == role_spec.epochs == 20
  assert manifest["compile_only_cpu"] is True
  assert manifest["gpu_runtime_initialized"] is manifest["gpu_dispatch_performed"] is False
  assert [row["epoch"] for row in manifest["variants"]] == list(range(20))
  assert len({row["program_key"] for row in manifest["variants"]}) == 20
  assert [row["offsets"]["q4"] for row in manifest["variants"]] == [epoch * 36 for epoch in range(20)]
  assert [row["elements"] for row in manifest["shared_program"]["abi"]] == [
    parameter.size for parameter in five_buffer_parameters(*role_spec.shape)]

  directory = frozen_v2.load_frozen_epoch_program_set(output)
  archived = frozen_v2.load_frozen_epoch_program_set(archive)
  assert len(directory.programs) == len(archived.programs) == 20
  assert tuple(program.key for program in directory.programs) == tuple(program.key for program in archived.programs)
  assert directory.manifest["family_identity"] == archived.manifest["family_identity"]
  binding = frozen_v2.load_frozen_epoch_program_set_binding(role_spec, output)
  assert binding.schema == frozen_v2.BINDING_SCHEMA
  assert binding.family_identity == manifest["family_identity"]
  assert binding.program_keys == tuple(row["program_key"] for row in manifest["variants"])


def test_v2_producer_owns_exactly_one_family_builder_call(tmp_path: Path):
  role_spec, calls = exact_role_spec("ffn_gate_up"), []
  def build_once():
    calls.append("build")
    return _family(role_spec)
  manifest = frozen_v2.produce_frozen_epoch_program_set(
    tmp_path / "single-build", role_spec=role_spec, build_once=build_once)
  assert calls == ["build"] and manifest["family_builder_calls"] == 1


def test_v2_producer_rejects_program_whose_structural_offset_differs_from_ordinal(tmp_path: Path):
  role_spec = exact_role_spec("ffn_gate_up")
  with pytest.raises(ValueError, match="compile-time offsets differ"):
    frozen_v2.produce_frozen_epoch_program_set(
      tmp_path / "bad-offset", role_spec=role_spec,
      build_once=lambda: _family(role_spec, wrong_program_epoch=(7, 8)))
  assert not (tmp_path / "bad-offset").exists()


def test_v2_rejects_variant_grid_and_shared_full_role_abi_drift(tmp_path: Path):
  role_spec = exact_role_spec("ffn_gate_up")
  family = _family(role_spec)
  changed_program = family.variants[4].program.replace(arg=ProgramInfo(
    name=FUNCTION_NAME, globals=tuple(range(5)), global_size=(1, 1, 1), local_size=(256, 1, 1)))
  changed_variant = replace(family.variants[4], program=changed_program)
  changed_family = replace(family, variants=family.variants[:4] + (changed_variant,) + family.variants[5:])
  with pytest.raises(ValueError, match="grid or local size"):
    frozen_v2.produce_frozen_epoch_program_set(
      tmp_path / "bad-grid", role_spec=role_spec, build_once=lambda: changed_family)

  _, output, _, _ = _produce(tmp_path / "abi")
  manifest_path = output / "manifest.json"
  manifest = json.loads(manifest_path.read_text())
  manifest["shared_program"]["abi"][2]["elements"] -= 1
  manifest_path.write_text(json.dumps(manifest))
  with pytest.raises(ValueError, match="shared full-role ABI"):
    frozen_v2.load_frozen_epoch_program_set(output)


def test_v2_loader_rejects_duplicate_ordinal_and_manifest_offset_drift(tmp_path: Path):
  _, output, _, _ = _produce(tmp_path)
  manifest_path = output / "manifest.json"
  manifest = json.loads(manifest_path.read_text())
  manifest["variants"][3]["epoch"] = 2
  manifest_path.write_text(json.dumps(manifest))
  with pytest.raises(ValueError, match="ordinal, offsets, or filenames"):
    frozen_v2.load_frozen_epoch_program_set(output)

  manifest["variants"][3]["epoch"] = 3
  manifest["variants"][3]["offsets"]["q8_values"] += 1
  manifest_path.write_text(json.dumps(manifest))
  with pytest.raises(ValueError, match="ordinal, offsets, or filenames"):
    frozen_v2.load_frozen_epoch_program_set(output)


def test_v2_loader_rejects_retained_payload_and_program_key_tampering(tmp_path: Path):
  _, output, _, _ = _produce(tmp_path)
  binary = output / "epoch_005.hsaco"
  binary.write_bytes(binary.read_bytes() + b"tamper")
  with pytest.raises(ValueError, match="file inventory identity mismatch"):
    frozen_v2.load_frozen_epoch_program_set(output)

  _, output2, _, _ = _produce(tmp_path / "second")
  manifest_path = output2 / "manifest.json"
  manifest = json.loads(manifest_path.read_text())
  manifest["variants"][4]["program_key"] = manifest["variants"][5]["program_key"]
  manifest_path.write_text(json.dumps(manifest))
  with pytest.raises(ValueError, match="PROGRAM key differs"):
    frozen_v2.load_frozen_epoch_program_set(output2)


def test_v2_binding_fails_closed_on_forged_candidate_identity_before_loading(tmp_path: Path):
  role_spec, output, _, _ = _produce(tmp_path)
  forged = replace(role_spec, candidate_canonical_identity="0" * 64)
  with pytest.raises(ValueError, match="canonical admitted"):
    frozen_v2.load_frozen_epoch_program_set_binding(forged, output)


def test_v1_schema_remains_distinct_from_v2():
  from extra.qk.mmq_frozen_target_artifact import SCHEMA as V1_SCHEMA
  assert V1_SCHEMA == "tinygrad.mmq_q4k_q8_1.frozen_target_artifact.v1"
  assert frozen_v2.SCHEMA.endswith(".v2") and frozen_v2.SCHEMA != V1_SCHEMA
