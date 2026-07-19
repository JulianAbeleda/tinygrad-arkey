"""CPU-only guarded repack of one retained legacy frozen epoch family.

This diagnostic deliberately preserves the legacy family's already-retained
sink, source, and authoritative final instruction stream.  It only reassembles
that stream through the current AMD ELF packer so code-end prefetch padding can
be isolated without another graph build or codegen search.  Its distinct schema
is permanently C1-uncertified and promotion-ineligible.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import shutil
import tempfile
from typing import Any, Callable

from tinygrad.helpers import Target, round_up
from tinygrad.renderer.amd.elf import assemble_linear
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.runtime.autogen import hsa
from tinygrad.runtime.autogen.amd.rdna3.ins import s_code_end, s_endpgm
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.uop.ops import Ops, UOp

from extra.qk.mmq_exact_role_spec import DEFAULT_INVENTORY, exact_role_spec
from extra.qk.mmq_frozen_epoch_program_set import (
  DIAGNOSTIC_REPACK_LINEAGE_SCHEMA, DIAGNOSTIC_REPACK_SCHEMA,
  DIAGNOSTIC_REPACK_SOURCE_MANIFEST, LEGACY_SCHEMA,
  _diagnostic_repack_identity, _git_value, _inventory, _json_bytes,
  _program_payload, _read_bundle, _sha256, _validate_program, _variant_files,
  _write_archive, load_frozen_epoch_program_set,
)
from extra.qk.mmq_llama_five_buffer_full_kernel import AMD_ISA_TARGET


def _one_text(binary: bytes) -> bytes:
  _, sections, relocations = elf_loader(binary)
  if relocations: raise ValueError("diagnostic repack does not admit relocatable legacy code")
  matches = [section.content for section in sections if section.name == ".text"]
  if len(matches) != 1: raise ValueError("diagnostic repack requires exactly one .text section")
  return matches[0]


def _guarded_repack_program(program: UOp, old_binary: bytes, *,
                            renderer: AMDISARenderer | None = None,
                            assemble: Callable[[UOp, UOp, str], bytes] = assemble_linear,
                            text_content: Callable[[bytes], bytes] = _one_text,
                            ) -> tuple[UOp, dict[str, Any]]:
  """Reassemble one retained final stream and prove both old and new text boundaries."""
  renderer = renderer or AMDISARenderer(Target.parse(AMD_ISA_TARGET))
  final_linear = renderer._final_linear(program.src[2])
  instructions = tuple(node.arg for node in final_linear.src)
  if not instructions or instructions[-1].to_bytes() != s_endpgm().to_bytes():
    raise ValueError("diagnostic repack final stream does not terminate in s_endpgm")
  code = b"".join(instruction.to_bytes() for instruction in instructions)
  padding_instruction = s_code_end().to_bytes()

  # The legacy packer multiplied a four-byte instruction by a byte count.
  legacy_padding_copies = (
    hsa.AMD_ISA_ALIGN_BYTES - len(code) % hsa.AMD_ISA_ALIGN_BYTES
  ) % hsa.AMD_ISA_ALIGN_BYTES
  if text_content(old_binary) != code + padding_instruction * legacy_padding_copies:
    raise ValueError("legacy HSACO .text differs from its retained authoritative final stream")

  assembly_program = renderer._assembly_program(program, program.src[2].arg)
  new_binary = assemble(assembly_program, final_linear, renderer.target.arch)
  padding_nbytes = round_up(len(code), 128) - len(code) + 3 * 128
  if padding_nbytes % len(padding_instruction):
    raise ValueError("guarded padding is not instruction-aligned")
  if text_content(new_binary) != code + padding_instruction * (padding_nbytes // len(padding_instruction)):
    raise ValueError("current packer output does not preserve the final stream plus exact gfx11 guard")
  rebuilt = assemble(assembly_program, final_linear, renderer.target.arch)
  if rebuilt != new_binary:
    raise ValueError("current packer reassembly is not byte-exact")

  new_program = program.replace(src=program.src[:4] + (UOp(Ops.BINARY, arg=new_binary),))
  return new_program, {
    "final_stream": {
      "sha256": _sha256(code), "nbytes": len(code),
      "instruction_count": len(instructions), "preserved_byte_for_byte": True,
    },
    "padding": {
      "instruction": "s_code_end", "cache_line_bytes": 128,
      "guard_cache_lines": 3, "padding_nbytes": padding_nbytes,
    },
    "new_binary": new_binary,
  }


def _source_lineage() -> dict[str, Any]:
  from tinygrad.renderer.amd import elf as elf_packer
  from tinygrad.renderer.isa import amd as isa_renderer
  commit, tree = _git_value("rev-parse", "HEAD"), _git_value("rev-parse", "HEAD^{tree}")
  return {
    "compile_target": AMD_ISA_TARGET, "arch": "gfx1100",
    "authority": "current_renderer_reassembly_of_retained_final_linear",
    "source_revision": {"vcs": "git", "commit": commit, "tree": tree},
    "source_files": {
      "diagnostic_repacker": _sha256(Path(__file__).read_bytes()),
      "elf_packer": _sha256(Path(elf_packer.__file__).read_bytes()),
      "isa_renderer": _sha256(Path(isa_renderer.__file__).read_bytes()),
    },
  }


def repack_guarded_legacy_epoch_program_set(
    source_bundle: str | Path, output_dir: str | Path, *,
    archive: str | Path | None = None,
    inventory: str | Path | dict[str, Any] = DEFAULT_INVENTORY,
    ) -> dict[str, Any]:
  """Atomically emit a diagnostic-only guarded family from a legacy v2 family."""
  source_path = Path(source_bundle)
  source_files = _read_bundle(source_path)
  if "manifest.json" not in source_files:
    raise ValueError("source bundle does not contain a manifest")
  try: source_manifest = json.loads(source_files["manifest.json"])
  except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise ValueError(f"source manifest is invalid: {exc}") from exc
  if source_manifest.get("schema") != LEGACY_SCHEMA:
    raise ValueError("diagnostic repack requires an explicitly legacy v2 source bundle")
  source = load_frozen_epoch_program_set(source_path, inventory=inventory)
  if source.manifest["c1_certification"] != {
      "gate": "C1", "certified": False, "status": "legacy_v2_missing_generation_provenance",
      "content_addressed": False}:
    raise ValueError("diagnostic repack source is not the canonical non-C1 legacy family")
  role = source_manifest["role"]
  role_spec = exact_role_spec(str(role["name"]), shape=tuple(role["shape"]), inventory=inventory)

  output = Path(output_dir)
  if output.exists(): raise FileExistsError(f"output already exists: {output}")
  archive_path = None if archive is None else Path(archive)
  if archive_path is not None and archive_path.exists():
    raise FileExistsError(f"archive already exists: {archive_path}")
  output.parent.mkdir(parents=True, exist_ok=True)
  staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
  try:
    retained: dict[str, bytes] = {
      DIAGNOSTIC_REPACK_SOURCE_MANIFEST: source_files["manifest.json"],
    }
    variants, lineage_rows = [], []
    for epoch, (sink, program, old_binary, old_source) in enumerate(
        zip(source.sinks, source.programs, source.binaries, source.sources)):
      new_program, evidence = _guarded_repack_program(program, old_binary)
      new_program = _validate_program(new_program, role_spec)
      new_binary, new_source = _program_payload(new_program)
      if new_source != old_source or new_binary != evidence.get("new_binary"):
        raise ValueError("diagnostic repack changed source text or lost the guarded binary")
      names = _variant_files(epoch)
      legacy_binary_name = f"epoch_{epoch:03d}.legacy.hsaco"
      retained[legacy_binary_name] = old_binary
      serialized_sink = source_files[names["sink"]]
      serialized_program = pickle.dumps(new_program, protocol=pickle.HIGHEST_PROTOCOL)
      files = {
        "sink": serialized_sink, "program": serialized_program,
        "source": old_source.encode(), "binary": new_binary,
      }
      retained.update({names[kind]: data for kind, data in files.items()})
      variant = {
        "epoch": epoch, "offsets": source_manifest["variants"][epoch]["offsets"],
        "sink_key": sink.key.hex(), "program_key": new_program.key.hex(), "files": names,
        "artifacts": {
          kind: {"sha256": _sha256(data), "nbytes": len(data)}
          for kind, data in files.items()
        },
      }
      variants.append(variant)
      lineage_rows.append({
        "epoch": epoch,
        "source": {
          "program_key": source_manifest["variants"][epoch]["program_key"],
          "binary_file": legacy_binary_name,
          "binary_sha256": _sha256(old_binary), "binary_nbytes": len(old_binary),
        },
        "final_stream": evidence["final_stream"],
        "output": {
          "program_key": variant["program_key"],
          "binary_sha256": variant["artifacts"]["binary"]["sha256"],
          "binary_nbytes": variant["artifacts"]["binary"]["nbytes"],
          "current_packer_reassembly": True,
        },
        "padding": evidence["padding"],
      })

    sink_keys = [row["sink_key"] for row in variants]
    program_keys = [row["program_key"] for row in variants]
    if len(set(sink_keys)) != role_spec.epochs or len(set(program_keys)) != role_spec.epochs:
      raise ValueError("diagnostic repack sink and PROGRAM identities must remain epoch-unique")
    lineage = {
      "schema": DIAGNOSTIC_REPACK_LINEAGE_SCHEMA,
      "purpose": "isolated_guarded_code_end_repack",
      "promotion_eligible": False, "c1_certified": False,
      "source_bundle": {
        "schema": LEGACY_SCHEMA,
        "family_identity": source_manifest["family_identity"],
        "manifest_file": DIAGNOSTIC_REPACK_SOURCE_MANIFEST,
        "manifest_sha256": _sha256(source_files["manifest.json"]),
        "manifest_nbytes": len(source_files["manifest.json"]),
      },
      "packer": _source_lineage(),
      "variants": lineage_rows,
    }
    manifest = json.loads(json.dumps(source_manifest))
    manifest.update({
      "schema": DIAGNOSTIC_REPACK_SCHEMA,
      "family_builder_calls": 0,
      "compiler_boundary": {
        "authority": "diagnostic_repack_of_retained_legacy_family",
        "offset_authority": "unchanged_retained_pre_lowering_sink",
        "executable_authority": "current_packer_reassembly_of_retained_final_linear",
        "final_program_structural_offsets_claimed": False,
      },
      "variants": variants,
      "files": _inventory(retained),
      "diagnostic_repack": lineage,
    })
    manifest["family_identity"] = _diagnostic_repack_identity(
      role_spec, sink_keys, program_keys, lineage)
    for name, data in retained.items(): (staging / name).write_bytes(data)
    (staging / "manifest.json").write_bytes(_json_bytes(manifest))
    load_frozen_epoch_program_set(staging, inventory=inventory)
    os.replace(staging, output)
    if archive_path is not None:
      archive_path.parent.mkdir(parents=True, exist_ok=True)
      archive_staging_dir = Path(tempfile.mkdtemp(
        prefix=f".{archive_path.name}.", dir=archive_path.parent))
      archive_staging = archive_staging_dir / archive_path.name
      try:
        _write_archive(output, archive_staging)
        load_frozen_epoch_program_set(archive_staging, inventory=inventory)
        os.replace(archive_staging, archive_path)
      finally:
        shutil.rmtree(archive_staging_dir, ignore_errors=True)
    return manifest
  except BaseException:
    shutil.rmtree(staging, ignore_errors=True)
    raise


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
    description="CPU-only guarded diagnostic repack of one legacy frozen epoch family")
  parser.add_argument("--source-bundle", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--archive")
  args = parser.parse_args(argv)
  manifest = repack_guarded_legacy_epoch_program_set(
    args.source_bundle, args.output, archive=args.archive)
  print(_json_bytes({
    "status": "PASS", "schema": manifest["schema"],
    "family_identity": manifest["family_identity"],
    "output": str(Path(args.output).resolve()),
    "archive": None if args.archive is None else str(Path(args.archive).resolve()),
    "promotion_eligible": False, "c1_certified": False,
  }).decode(), end="")
  return 0


__all__ = [
  "repack_guarded_legacy_epoch_program_set",
]


if __name__ == "__main__": raise SystemExit(main())
