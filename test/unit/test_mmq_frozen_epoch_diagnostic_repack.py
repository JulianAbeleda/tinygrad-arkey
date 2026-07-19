from __future__ import annotations

import ctypes
from dataclasses import replace
import json
from pathlib import Path
import pickle
import shutil
import struct
from types import SimpleNamespace

import pytest

from tinygrad.runtime.autogen import hsa
from tinygrad.renderer.amd.elf import assemble_linear
from tinygrad.runtime.autogen.amd.rdna3.ins import s_code_end, s_endpgm
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.autogen import libc
from tinygrad.uop.ops import Ops, UOp

from extra.qk import mmq_frozen_epoch_program_set as frozen
from extra.qk import mmq_frozen_epoch_diagnostic_repack as repack_module
from extra.qk.mmq_frozen_epoch_diagnostic_repack import (
  _guarded_repack_program, repack_guarded_legacy_epoch_program_set,
)
from extra.qk.prefill.frozen_epoch_program_set_scheduler import _validate_binding
from test.unit.test_frozen_epoch_program_set_scheduler import _binding
from test.unit.test_mmq_frozen_epoch_program_set import (
  _downgrade_manifest_to_legacy_v2, _family, _freeze, _program,
)


def _legacy_text_object(program: UOp, linear: UOp) -> bytes:
  guarded = assemble_linear(program, linear, "gfx1100")
  _, sections, _ = elf_loader(guarded)
  text_index, text = next((index, section) for index, section in enumerate(sections) if section.name == ".text")
  code = b"".join(node.arg.to_bytes() for node in linear.src)
  copies = (hsa.AMD_ISA_ALIGN_BYTES-len(code) % hsa.AMD_ISA_ALIGN_BYTES) % hsa.AMD_ISA_ALIGN_BYTES
  legacy_size = len(code) + len(s_code_end().to_bytes())*copies
  assert text.content[:legacy_size] == code + s_code_end().to_bytes()*copies
  header = libc.Elf64_Ehdr.from_buffer_copy(guarded)
  result = bytearray(guarded)
  struct.pack_into("<Q", result, int(header.e_shoff)+text_index*ctypes.sizeof(libc.Elf64_Shdr)+32, legacy_size)
  return bytes(result)


def _legacy_family(role):
  family = _family(role)
  variants = []
  for epoch, variant in enumerate(family.variants):
    # An exact 256-byte stream makes the legacy broken padding count zero,
    # while the guarded gfx11 object still adds three 128-byte cache lines.
    linear = UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=s_endpgm()) for _ in range(64)))
    program = variant.program.replace(src=(
      *variant.program.src[:2], linear,
      UOp(Ops.SOURCE, arg=f"synthetic legacy epoch {epoch}\n"),
      UOp(Ops.BINARY, arg=b"placeholder"),
    ))
    legacy_binary = _legacy_text_object(program, linear)
    program = program.replace(src=program.src[:4] + (UOp(Ops.BINARY, arg=legacy_binary),))
    variants.append(replace(variant, program=program))
  return replace(family, variants=tuple(variants))


def _produce_legacy(tmp_path: Path):
  from extra.qk.mmq_exact_role_spec import exact_role_spec
  role, output = exact_role_spec("ffn_gate_up"), tmp_path / "legacy"
  produced = _freeze(output, role_spec=role, build_once=lambda: _legacy_family(role))
  (output / "manifest.json").write_text(json.dumps(_downgrade_manifest_to_legacy_v2(produced)))
  return role, output


def test_guarded_repack_preserves_exact_final_stream_and_checks_both_padding_policies():
  from extra.qk.mmq_exact_role_spec import exact_role_spec
  role = exact_role_spec("ffn_gate_up")
  program = _program(role, 0).replace(src=(
    *_program(role, 0).src[:2],
    UOp(Ops.LINEAR, src=(UOp(Ops.INS, arg=s_endpgm()),)),
    *_program(role, 0).src[3:],
  ))
  renderer = SimpleNamespace(
    target=SimpleNamespace(arch="gfx1100"),
    _final_linear=lambda linear: linear,
    _assembly_program=lambda value, proof: value,
  )
  code, padding = s_endpgm().to_bytes(), s_code_end().to_bytes()
  old_padding_copies = (hsa.AMD_ISA_ALIGN_BYTES-len(code) % hsa.AMD_ISA_ALIGN_BYTES) % hsa.AMD_ISA_ALIGN_BYTES
  new_padding_nbytes = 128-len(code) + 3*128
  texts = {
    b"old": code + padding * old_padding_copies,
    b"new": code + padding * (new_padding_nbytes//4),
  }
  repacked, evidence = _guarded_repack_program(
    program, b"old", renderer=renderer,
    assemble=lambda _program, _linear, _arch: b"new",
    text_content=texts.__getitem__,
  )
  assert repacked.src[4].arg == b"new"
  assert evidence["final_stream"] == {
    "sha256": frozen._sha256(code), "nbytes": 4,
    "instruction_count": 1, "preserved_byte_for_byte": True,
  }
  assert evidence["padding"]["padding_nbytes"] == new_padding_nbytes


def test_diagnostic_repack_is_atomic_loadable_and_permanently_non_c1(tmp_path: Path):
  role, source = _produce_legacy(tmp_path / "source")
  output, archive = tmp_path / "guarded-diagnostic", tmp_path / "guarded-diagnostic.tar"
  manifest = repack_guarded_legacy_epoch_program_set(
    source, output, archive=archive)
  assert output.is_dir() and archive.is_file()
  assert manifest["schema"] == frozen.DIAGNOSTIC_REPACK_SCHEMA
  assert manifest["family_builder_calls"] == 0
  assert manifest["diagnostic_repack"]["promotion_eligible"] is False
  assert manifest["diagnostic_repack"]["c1_certified"] is False
  assert len(manifest["diagnostic_repack"]["variants"]) == role.epochs == 20
  assert all(row["output"]["current_packer_reassembly"] is True
             for row in manifest["diagnostic_repack"]["variants"])

  loaded = frozen.load_frozen_epoch_program_set(output)
  archived = frozen.load_frozen_epoch_program_set(archive)
  assert loaded.manifest["family_identity"] == archived.manifest["family_identity"]
  assert loaded.manifest["c1_certification"] == {
    "gate": "C1", "certified": False,
    "status": "diagnostic_repack_non_c1_non_promotion",
    "content_addressed": False,
  }
  assert all(binary.startswith(b"\x7fELF") for binary in loaded.binaries)
  assert all((output / f"epoch_{epoch:03d}.legacy.hsaco").is_file()
             for epoch in range(role.epochs))
  assert (output / frozen.DIAGNOSTIC_REPACK_SOURCE_MANIFEST).is_file()
  with pytest.raises(ValueError, match="C1-uncertified"):
    frozen.load_frozen_epoch_program_set(output, require_c1=True)

  # Forge all outer hashes/keys consistently after corrupting guarded .text.
  # Independent loader reconstruction must still reject the executable relation.
  forged = tmp_path / "forged-guarded"
  shutil.copytree(output, forged)
  binary_path, program_path = forged / "epoch_000.hsaco", forged / "epoch_000.program.pkl"
  binary = bytearray(binary_path.read_bytes())
  text = next(section for section in elf_loader(bytes(binary))[1] if section.name == ".text")
  binary[int(text.header.sh_offset)+int(text.header.sh_size)-1] ^= 1
  forged_binary = bytes(binary)
  binary_path.write_bytes(forged_binary)
  program = pickle.loads(program_path.read_bytes())
  program = program.replace(src=program.src[:4] + (UOp(Ops.BINARY, arg=forged_binary),))
  serialized_program = pickle.dumps(program, protocol=pickle.HIGHEST_PROTOCOL)
  program_path.write_bytes(serialized_program)
  forged_manifest_path = forged / "manifest.json"
  forged_manifest = json.loads(forged_manifest_path.read_text())
  binary_identity = {"sha256": frozen._sha256(forged_binary), "nbytes": len(forged_binary)}
  program_identity = {"sha256": frozen._sha256(serialized_program), "nbytes": len(serialized_program)}
  forged_manifest["files"]["epoch_000.hsaco"] = binary_identity
  forged_manifest["files"]["epoch_000.program.pkl"] = program_identity
  forged_manifest["variants"][0]["artifacts"]["binary"] = binary_identity
  forged_manifest["variants"][0]["artifacts"]["program"] = program_identity
  forged_manifest["variants"][0]["program_key"] = program.key.hex()
  forged_manifest["diagnostic_repack"]["variants"][0]["output"].update({
    "program_key": program.key.hex(), "binary_sha256": binary_identity["sha256"],
    "binary_nbytes": binary_identity["nbytes"],
  })
  forged_manifest["family_identity"] = frozen._diagnostic_repack_identity(
    role, [row["sink_key"] for row in forged_manifest["variants"]],
    [row["program_key"] for row in forged_manifest["variants"]],
    forged_manifest["diagnostic_repack"])
  forged_manifest_path.write_bytes(frozen._json_bytes(forged_manifest))
  with pytest.raises(ValueError, match="new \\.text differs|current packer reassembly differs"):
    frozen.load_frozen_epoch_program_set(forged)

  forged_old = tmp_path / "forged-legacy"
  shutil.copytree(output, forged_old)
  old_binary_path = forged_old / "epoch_000.legacy.hsaco"
  old_binary = bytearray(old_binary_path.read_bytes())
  old_text = next(section for section in elf_loader(bytes(old_binary))[1] if section.name == ".text")
  old_binary[int(old_text.header.sh_offset)+int(old_text.header.sh_size)-1] ^= 1
  forged_old_binary = bytes(old_binary)
  old_binary_path.write_bytes(forged_old_binary)
  old_manifest_path = forged_old / "manifest.json"
  old_manifest = json.loads(old_manifest_path.read_text())
  old_identity = {"sha256": frozen._sha256(forged_old_binary), "nbytes": len(forged_old_binary)}
  old_manifest["files"]["epoch_000.legacy.hsaco"] = old_identity
  old_manifest["diagnostic_repack"]["variants"][0]["source"].update({
    "binary_sha256": old_identity["sha256"], "binary_nbytes": old_identity["nbytes"],
  })
  old_manifest["family_identity"] = frozen._diagnostic_repack_identity(
    role, [row["sink_key"] for row in old_manifest["variants"]],
    [row["program_key"] for row in old_manifest["variants"]],
    old_manifest["diagnostic_repack"])
  old_manifest_path.write_bytes(frozen._json_bytes(old_manifest))
  with pytest.raises(ValueError, match="legacy \\.text differs"):
    frozen.load_frozen_epoch_program_set(forged_old)

  manifest_path = output / "manifest.json"
  tampered = json.loads(manifest_path.read_text())
  tampered["diagnostic_repack"]["promotion_eligible"] = True
  manifest_path.write_text(json.dumps(tampered))
  with pytest.raises(ValueError, match="admission flags"):
    frozen.load_frozen_epoch_program_set(output)


def test_scheduler_accepts_diagnostic_repack_only_when_c1_is_not_required():
  from extra.qk.mmq_exact_role_spec import exact_role_spec
  role = exact_role_spec("ffn_gate_up")
  binding = _binding(role)
  manifest = dict(binding.artifact.manifest)
  manifest.update({
    "schema": frozen.DIAGNOSTIC_REPACK_SCHEMA,
    "c1_certification": {
      "gate": "C1", "certified": False,
      "status": "diagnostic_repack_non_c1_non_promotion",
      "content_addressed": False,
    },
  })
  binding = replace(binding, artifact=replace(binding.artifact, manifest=manifest))
  _validate_binding(binding, role)
  with pytest.raises(ValueError, match="complete exact epoch family"):
    _validate_binding(binding, role, require_c1=True)


def test_diagnostic_repack_cli_reports_non_promotion_identity(monkeypatch, capsys, tmp_path: Path):
  monkeypatch.setattr(repack_module, "repack_guarded_legacy_epoch_program_set",
    lambda source, output, *, archive: {
      "schema": frozen.DIAGNOSTIC_REPACK_SCHEMA, "family_identity": "f" * 64})
  assert repack_module.main([
    "--source-bundle", "/frozen/legacy-v2.tar",
    "--output", str(tmp_path / "guarded"),
    "--archive", str(tmp_path / "guarded.tar"),
  ]) == 0
  report = json.loads(capsys.readouterr().out)
  assert report["schema"] == frozen.DIAGNOSTIC_REPACK_SCHEMA
  assert report["promotion_eligible"] is False and report["c1_certified"] is False
