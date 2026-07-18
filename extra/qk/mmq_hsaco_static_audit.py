"""Minimal CPU-only control-flow audit for one frozen AMD HSACO.

The frozen-artifact loader owns PROGRAM ABI, descriptor, resource, relocation,
and upload-image validation.  This module adds only the independent checks
needed to rule out malformed ELF bounds or generated control flow.  It consumes
the already-retained disassembly and never compiles, loads, or dispatches code.
"""
from __future__ import annotations

import hashlib
import re
import struct
from typing import Any

from extra.qk.mmq_compile_evidence import analyze_final_isa


SCHEMA = "tinygrad.amd.hsaco_static_audit.v1"
ELF64_EHDR_SIZE = 64
ELF64_SHDR_SIZE = 64
SHT_NOBITS = 8
EM_AMDGPU = 0xE0
S_ENDPGM = 0xBFB00000
S_CODE_END = 0xBF9F0000


def _sha256(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def _section_table(binary: bytes) -> list[dict[str, int | str]]:
  """Parse ELF64 section bounds without trusting offsets from the object."""
  if len(binary) < ELF64_EHDR_SIZE or binary[:4] != b"\x7fELF": raise ValueError("missing ELF magic")
  if binary[4] != 2 or binary[5] not in (0, 1): raise ValueError("only little-endian ELF64 HSACO is supported")
  header = struct.unpack_from("<16sHHIQQQIHHHHHH", binary)
  e_machine, e_shoff, e_shentsize, e_shnum, e_shstrndx = header[2], header[6], header[11], header[12], header[13]
  # assemble_linear's CPU-only test fixture leaves EI_DATA/E_MACHINE and
  # e_shentsize zero; production HSACO must otherwise identify EM_AMDGPU.
  if e_machine not in (0, EM_AMDGPU): raise ValueError(f"not AMDGPU ELF (machine 0x{e_machine:x})")
  if e_shentsize == 0: e_shentsize = ELF64_SHDR_SIZE
  if e_shentsize != ELF64_SHDR_SIZE: raise ValueError("unexpected ELF section-header size")
  if not e_shnum: raise ValueError("ELF has no section headers")
  if e_shoff < ELF64_EHDR_SIZE or e_shoff + e_shnum * e_shentsize > len(binary):
    raise ValueError("section-header table is out of bounds")
  if e_shstrndx >= e_shnum: raise ValueError("section-name string-table index is out of bounds")

  raw = [struct.unpack_from("<IIQQQQIIQQ", binary, e_shoff + i * e_shentsize) for i in range(e_shnum)]
  shstr_offset, shstr_size = raw[e_shstrndx][4:6]
  if shstr_offset + shstr_size > len(binary): raise ValueError("section-name string table is out of bounds")
  names = binary[shstr_offset:shstr_offset + shstr_size]
  sections: list[dict[str, int | str]] = []
  for index, section in enumerate(raw):
    name_offset, section_type, _, address, offset, size, _, _, alignment, _ = section
    if name_offset >= len(names): raise ValueError(f"section {index} name offset is out of bounds")
    name_end = names.find(b"\0", name_offset)
    if name_end < 0: raise ValueError(f"section {index} name is unterminated")
    if section_type != SHT_NOBITS and offset + size > len(binary):
      raise ValueError(f"section {index} contents are out of bounds")
    if alignment and alignment & (alignment - 1): raise ValueError(f"section {index} alignment is not a power of two")
    sections.append({"index": index, "name": names[name_offset:name_end].decode("utf-8", "replace"),
                     "type": int(section_type), "addr": int(address), "offset": int(offset),
                     "size": int(size), "align": int(alignment)})
  return sections


def _one_section(sections: list[dict[str, int | str]], name: str) -> dict[str, int | str]:
  matches = [section for section in sections if section["name"] == name]
  if len(matches) != 1: raise ValueError(f"HSACO requires exactly one {name} section")
  return matches[0]


def _direct_target(row: dict[str, Any]) -> int:
  token = row["operands"].split(",", 1)[0].split(None, 1)[0]
  if not re.fullmatch(r"(?:0x[0-9A-Fa-f]+|[-+]?\d+)", token):
    raise ValueError(f"direct branch {row['mnemonic']} at 0x{row['pc']:x} has no numeric target")
  # llvm-objdump emits signed decimal branch immediates as PC-relative dword
  # offsets.  Its 0x spelling is an absolute destination.
  return int(token, 16) if token.lower().startswith("0x") else row["pc"] + 4 + int(token, 10) * 4


def _control_flow(rows: list[dict[str, Any]], text_start: int, text_end: int) -> dict[str, Any]:
  direct, indirect, errors = [], [], []
  instruction_pcs = {row["pc"] for row in rows}
  for row in rows:
    mnemonic = row["mnemonic"]
    if mnemonic.startswith(("s_branch", "s_cbranch")):
      try: target = _direct_target(row)
      except (IndexError, ValueError) as exc:
        errors.append(str(exc))
        continue
      if target < text_start or target >= text_end or target not in instruction_pcs:
        errors.append(f"direct branch target 0x{target:x} is outside or not an instruction boundary")
      direct.append({"pc": row["pc"], "mnemonic": mnemonic, "target": target})
    if mnemonic.startswith(("s_setpc", "s_swappc", "s_call", "s_rfe")):
      indirect.append({"pc": row["pc"], "mnemonic": mnemonic, "operands": row["operands"]})
  return {"direct": direct, "indirect": indirect, "errors": errors,
          "indirect_control_flow_flag": bool(indirect), "status": "PASS" if not errors else "BLOCKED"}


def _termination(binary: bytes, text: dict[str, int | str], rows: list[dict[str, Any]]) -> dict[str, Any]:
  text_start, text_size, text_offset = int(text["addr"]), int(text["size"]), int(text["offset"])
  if not text_size or text_size % 4: return {"passed": False, "errors": [".text size is empty or not dword-aligned"]}
  pcs = [row["pc"] for row in rows]
  errors: list[str] = []
  if pcs != sorted(set(pcs)): errors.append("disassembly PCs are duplicate or non-monotonic")
  if any(pc < text_start or pc >= text_start + text_size or (pc - text_start) % 4 for pc in pcs):
    errors.append("disassembly contains a PC outside or misaligned for .text")
  end_rows = [row for row in rows if row["mnemonic"] == "s_endpgm"]
  if not end_rows: return {"passed": False, "errors": [*errors, ".text has no s_endpgm"]}
  last_end = end_rows[-1]
  after_end = rows[rows.index(last_end) + 1:]
  nonpadding = [row for row in after_end if row["mnemonic"] not in ("s_code_end", "s_nop")]
  if nonpadding: errors.append("decoded non-padding instruction follows final s_endpgm")

  end_offset = text_offset + last_end["pc"] - text_start
  if end_offset + 4 > text_offset + text_size:
    errors.append("s_endpgm encoding extends beyond .text")
    trailing_words: list[int] = []
  else:
    end_word = struct.unpack_from("<I", binary, end_offset)[0]
    if end_word != S_ENDPGM: errors.append("saved disassembly s_endpgm does not match HSACO bytes")
    trailing_words = list(struct.unpack_from(f"<{(text_size - (end_offset - text_offset) - 4) // 4}I", binary, end_offset + 4))
    # s_nop encodes its immediate in the low 16 bits.
    if any(word != S_CODE_END and word & 0xFFFF0000 != 0xBF800000 for word in trailing_words):
      errors.append("raw .text after final s_endpgm contains non-padding words")
  return {"passed": not errors, "errors": errors, "final_endpgm_pc": last_end["pc"],
          "trailing_padding_dwords": len(trailing_words)}


def audit_hsaco(binary: bytes, disassembly: str) -> dict[str, Any]:
  """Audit immutable HSACO bytes against their already-saved disassembly."""
  result: dict[str, Any] = {
    "schema": SCHEMA, "passed": False, "findings": [],
  }
  try:
    if not isinstance(binary, (bytes, bytearray, memoryview)): raise TypeError("HSACO must be bytes-like")
    if not isinstance(disassembly, str) or not disassembly.strip(): raise ValueError("saved disassembly is empty")
    blob = bytes(binary)
    result.update({"binary_sha256": _sha256(blob), "binary_nbytes": len(blob),
                   "disassembly_sha256": _sha256(disassembly.encode())})
    sections = _section_table(blob)
    text = _one_section(sections, ".text")
    text_start, text_end = int(text["addr"]), int(text["addr"]) + int(text["size"])
    parsed = analyze_final_isa(disassembly)
    rows = parsed["instructions"]
    flow = _control_flow(rows, text_start, text_end)
    termination = _termination(blob, text, rows)
    result.update({"sections": sections, "text": text, "instruction_count": len(rows),
                   "control_flow": flow, "termination": termination})
    result["findings"].extend(flow["errors"])
    if flow["indirect_control_flow_flag"]:
      result["findings"].append("unexpected indirect control flow is present")
    result["findings"].extend(termination["errors"])
    result["passed"] = not result["findings"]
  except Exception as exc:
    result["findings"].append(f"{type(exc).__name__}: {exc}")
  result["verdict"] = "PASS" if result["passed"] else "BLOCKED"
  return result


__all__ = ["SCHEMA", "audit_hsaco"]
