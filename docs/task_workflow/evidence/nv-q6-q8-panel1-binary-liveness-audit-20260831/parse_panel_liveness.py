#!/usr/bin/env python3
"""Deterministic SASS-only Q8 panel liveness audit for the frozen Q6 cubins."""
from __future__ import annotations

import hashlib, json, re
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
INSN_RE = re.compile(r"/\*([0-9a-f]+)\*/\s+(.*?)\s*;\s*$")
REG_RE = re.compile(r"(?<![A-Z0-9_])R(\d+)(?:\.(?:reuse|ROW|COL|64))?")


@dataclass(frozen=True)
class Instruction:
  pc: int
  ordinal: int
  opcode: str
  operands: str
  text: str


@dataclass(frozen=True)
class PanelSpec:
  load_lo: int
  load_hi: int
  store_lo: int
  store_hi: int
  global_offset_base: int
  shared_offset_base: int


@dataclass(frozen=True)
class BodySpec:
  name: str
  binary: str
  symbol: str
  symbol_size_bytes: int
  body_start: int
  body_end: int
  barriers: tuple[int, int, int, int]
  panel0: PanelSpec
  panel1: PanelSpec


LLAMA_BINARY = "docs/task_workflow/evidence/nv-packed-q4k-q8-llama-extracted-20260830/q6k-mmq-dense.sm_120a.cubin"
CANDIDATE_BINARY = "docs/task_workflow/evidence/nv-q6-true-late-q8-panel1-gate7-20260831/artifacts/early_combined_all_partials/early_combined_all_partials.cubin"
LLAMA_SYMBOL = "_Z15dense_mul_mat_qIL9ggml_type14ELi128ELb0EEvPKcPKiPfS5_5uint3iiiiiS6_S6_iiiS6_S6_iiiS6_"
CANDIDATE_SYMBOL = "nv_q6_oracle_broad_cta_prefetch_combined_publish_oracle_publisher_trusted_fp16_packed_ws_segments_in_cta_streamk_s0"

BODIES = (
  BodySpec("llama_direct", LLAMA_BINARY, LLAMA_SYMBOL, 0x21c80, 0x0d80, 0xeb50,
    (0x3840, 0x8620, 0x8890, 0xeb40),
    PanelSpec(0x13b0, 0x15d0, 0x3720, 0x3830, 0x0000, 0x0200),
    PanelSpec(0x80e0, 0x8290, 0x86e0, 0x8870, 0x0000, 0x0200)),
  BodySpec("llama_partial", LLAMA_BINARY, LLAMA_SYMBOL, 0x21c80, 0x12880, 0x20780,
    (0x15480, 0x1a230, 0x1a4d0, 0x20770),
    PanelSpec(0x13110, 0x13720, 0x15350, 0x15470, 0x0000, 0x0200),
    PanelSpec(0x19d00, 0x19e20, 0x1a2f0, 0x1a470, 0x0000, 0x0200)),
  BodySpec("candidate", CANDIDATE_BINARY, CANDIDATE_SYMBOL, 0x14100, 0x09a0, 0x13ad0,
    (0x3060, 0xa930, 0xaab0, 0x109b0),
    PanelSpec(0x1c80, 0x1d90, 0x2f20, 0x3050, 0x0000, 0x9800),
    PanelSpec(0x1e80, 0x2280, 0xa990, 0xaaa0, 0x4800, 0x9800)),
)


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_disassembly(path: Path) -> list[Instruction]:
  out = []
  for raw in path.read_text().splitlines():
    m = INSN_RE.search(raw)
    if m is None: continue
    pc, rendered = int(m.group(1), 16), m.group(2).strip()
    rendered = re.split(r"\s+[&?]", rendered, maxsplit=1)[0].strip()
    if rendered.startswith("@"):
      _, rendered = rendered.split(None, 1)
    opcode, *tail = rendered.split(None, 1)
    out.append(Instruction(pc, pc // 16, opcode, tail[0] if tail else "", raw.strip()))
  assert all(x.pc % 16 == 0 for x in out)
  assert len({x.pc for x in out}) == len(out)
  return out


def regs(text: str) -> set[int]:
  return {int(x) for x in REG_RE.findall(text)}


NO_DEST_PREFIXES = ("ST", "BAR", "BRA", "EXIT", "NOP", "BSSY", "BSYNC", "WARPSYNC", "DEPBAR", "YIELD", "CALL", "RET")


def destination_regs(insn: Instruction) -> set[int]:
  if insn.opcode.startswith(NO_DEST_PREFIXES) or not insn.operands: return set()
  first = insn.operands.split(",", 1)[0]
  m = REG_RE.search(first)
  if m is None: return set()
  base = int(m.group(1))
  if insn.opcode.startswith("IMMA."): width = 4
  elif insn.opcode.startswith("LDSM.") and insn.opcode.endswith(".2"): width = 2
  elif ".128" in insn.opcode: width = 4
  elif ".64" in insn.opcode or ".64" in first: width = 2
  else: width = 1
  return set(range(base, base + width))


def source_regs(insn: Instruction) -> set[int]:
  if insn.opcode.startswith(NO_DEST_PREFIXES): return regs(insn.operands)
  tail = insn.operands.split(",", 1)
  return regs(tail[1]) if len(tail) == 2 else set()


def parse_load(insn: Instruction) -> tuple[int, int] | None:
  if not (insn.opcode == "LDG.E" or insn.opcode == "LDG.E.CONSTANT"): return None
  m = re.match(r"R(\d+),\s*desc\[[^]]+\]\[R\d+\.64(?:(\+0x[0-9a-f]+))?\]", insn.operands)
  return None if m is None else (int(m.group(1)), int(m.group(2) or "+0", 16))


def parse_store(insn: Instruction) -> tuple[int, int] | None:
  if insn.opcode != "STS": return None
  m = re.match(r"\[R\d+(?:\+0x([0-9a-f]+))?\],\s*R(\d+)", insn.operands)
  return None if m is None else (int(m.group(2)), int(m.group(1) or "0", 16))


def summarize_opcodes(instructions: list[Instruction], lo: int, hi: int) -> dict:
  window = [x for x in instructions if lo < x.pc < hi]
  def fam(prefix: str):
    return [x for x in window if x.opcode.startswith(prefix) and not (prefix == "LDS" and x.opcode.startswith("LDSM."))]
  ret = {"exclusive_pc_bounds": [f"0x{lo:x}", f"0x{hi:x}"], "instruction_count": len(window)}
  for prefix in ("IMMA.", "LDSM.", "LDS"):
    vals = fam(prefix)
    ret[prefix.rstrip(".")] = {"count": len(vals), "first_pc": None if not vals else f"0x{vals[0].pc:x}",
      "last_pc": None if not vals else f"0x{vals[-1].pc:x}"}
  return ret


def panel_audit(instructions: list[Instruction], spec: PanelSpec, body: BodySpec) -> dict:
  loads = []
  for insn in instructions:
    if not spec.load_lo <= insn.pc <= spec.load_hi: continue
    if (parsed := parse_load(insn)) is None: continue
    reg, offset = parsed
    logical = offset - spec.global_offset_base
    assert logical >= 0 and logical % 0x400 == 0
    loads.append((logical // 0x400, reg, offset, insn))
  stores = []
  for insn in instructions:
    if not spec.store_lo <= insn.pc <= spec.store_hi: continue
    if (parsed := parse_store(insn)) is None: continue
    reg, offset = parsed
    logical = offset - spec.shared_offset_base
    assert logical >= 0 and logical % 0x400 == 0
    stores.append((logical // 0x400, reg, offset, insn))
  assert len(loads) == len(stores) == 18
  assert {x[0] for x in loads} == {x[0] for x in stores} == set(range(18))
  by_load, by_store = {x[0]: x for x in loads}, {x[0]: x for x in stores}
  rows = []
  for idx in range(18):
    _, load_reg, global_offset, load = by_load[idx]
    _, store_reg, shared_offset, store = by_store[idx]
    assert load_reg == store_reg
    between = [x for x in instructions if load.pc < x.pc < store.pc]
    occurrences = [{"pc": f"0x{x.pc:x}", "opcode": x.opcode,
                    "as_source": load_reg in source_regs(x), "as_destination": load_reg in destination_regs(x), "text": x.text}
                   for x in between if load_reg in regs(x.operands)]
    assert not any(x["as_destination"] for x in occurrences), (body.name, idx, occurrences)
    assert not occurrences, (body.name, idx, occurrences)
    later = [x for x in instructions if store.pc < x.pc <= body.body_end]
    overwrite = next((x for x in later if load_reg in destination_regs(x)), None)
    before_overwrite = later if overwrite is None else [x for x in later if x.pc < overwrite.pc]
    post_store_uses = [x for x in before_overwrite if load_reg in source_regs(x)]
    rows.append({
      "logical_word": idx,
      "global_byte_offset": f"0x{global_offset:x}",
      "shared_byte_offset": f"0x{shared_offset:x}",
      "register": f"R{load_reg}",
      "ldg_pc": f"0x{load.pc:x}",
      "ldg_ordinal": load.ordinal,
      "sts_pc": f"0x{store.pc:x}",
      "sts_ordinal": store.ordinal,
      "ldg_to_sts_span_instructions": store.ordinal - load.ordinal,
      "intervening_register_occurrences": occurrences,
      "last_value_use_pc": f"0x{store.pc:x}" if not post_store_uses else f"0x{post_store_uses[-1].pc:x}",
      "post_store_uses_before_redefinition": [{"pc": f"0x{x.pc:x}", "opcode": x.opcode, "text": x.text} for x in post_store_uses],
      "first_static_redefinition_pc": None if overwrite is None else f"0x{overwrite.pc:x}",
      "first_static_redefinition_opcode": None if overwrite is None else overwrite.opcode,
    })
  ordered_loads, ordered_stores = sorted(loads, key=lambda x: x[3].pc), sorted(stores, key=lambda x: x[3].pc)
  first_load, last_load = ordered_loads[0][3], ordered_loads[-1][3]
  first_store, last_store = ordered_stores[0][3], ordered_stores[-1][3]
  return {
    "load_pc_bounds": [f"0x{first_load.pc:x}", f"0x{last_load.pc:x}"],
    "store_pc_bounds": [f"0x{first_store.pc:x}", f"0x{last_store.pc:x}"],
    "first_load_ordinal": first_load.ordinal,
    "first_store_ordinal": first_store.ordinal,
    "first_load_to_first_store_span_instructions": first_store.ordinal - first_load.ordinal,
    "first_load_to_last_store_span_instructions": last_store.ordinal - first_load.ordinal,
    "intervening_work_first_load_to_first_store": summarize_opcodes(instructions, first_load.pc, first_store.pc),
    "rows_by_logical_word": rows,
  }


def body_audit(body: BodySpec, instructions: list[Instruction]) -> dict:
  panel0, panel1 = panel_audit(instructions, body.panel0, body), panel_audit(instructions, body.panel1, body)
  bars = [x.pc for x in instructions if x.opcode.startswith("BAR.") and body.body_start <= x.pc <= body.body_end]
  assert bars == list(body.barriers), (body.name, bars)
  p1_regs = {int(x["register"][1:]) for x in panel1["rows_by_logical_word"]}
  tensor_direct = []
  for x in instructions:
    if body.panel1.load_lo < x.pc < body.panel1.store_hi and x.opcode.startswith(("IMMA.", "LDSM.")):
      overlap = sorted(p1_regs & source_regs(x))
      if overlap: tensor_direct.append({"pc": f"0x{x.pc:x}", "opcode": x.opcode, "registers": [f"R{r}" for r in overlap], "text": x.text})
  assert not tensor_direct, (body.name, tensor_direct)
  return {
    "symbol": body.symbol,
    "symbol_pc_bounds": ["0x0", f"0x{body.symbol_size_bytes:x}"],
    "normalized_body_pc_bounds": [f"0x{body.body_start:x}", f"0x{body.body_end:x}"],
    "barrier_pcs": [f"0x{x:x}" for x in body.barriers],
    "panel0": panel0,
    "panel1": panel1,
    "panel0_consume_window": summarize_opcodes(instructions, body.barriers[0], body.barriers[1]),
    "panel1_prefetch_tail_before_overwrite_barrier": summarize_opcodes(instructions, int(panel1["load_pc_bounds"][0], 16), body.barriers[1]),
    "panel1_consume_window": summarize_opcodes(instructions, body.barriers[2], body.barriers[3]),
    "panel1_ldg_register_direct_ldsm_imma_uses": tensor_direct,
  }


def main() -> None:
  paths = {
    "llama": ROOT / LLAMA_BINARY,
    "candidate": ROOT / CANDIDATE_BINARY,
  }
  parsed = {
    "llama": parse_disassembly(HERE / "llama.nvdisasm"),
    "candidate": parse_disassembly(HERE / "candidate.nvdisasm"),
  }
  report = {
    "schema": "tinygrad.nv_q6_q8_panel1_binary_liveness_audit.v1",
    "method": "SASS-only fixed-window load/store register pairing; no CUDA-source ordering used",
    "artifacts": {
      "llama": {"path": str(paths["llama"].relative_to(ROOT)), "sha256": sha256(paths["llama"]), "disassembly": str((HERE / "llama.nvdisasm").relative_to(ROOT))},
      "candidate": {"path": str(paths["candidate"].relative_to(ROOT)), "sha256": sha256(paths["candidate"]), "disassembly": str((HERE / "candidate.nvdisasm").relative_to(ROOT))},
    },
    "bodies": {body.name: body_audit(body, parsed[body.binary == CANDIDATE_BINARY and "candidate" or "llama"]) for body in BODIES},
    "parser_limits": [
      "Register occurrence and load-to-store pairing are exact textual SASS facts.",
      "First redefinition expands IMMA destinations to four registers, LDSM.16.M88.2 to two, and .64/.128 destinations to two/four.",
      "The parser does not claim a dynamic latency or scoreboard-wait duration from static control fields.",
      "Downstream shared-memory values are not assigned the identity of their prior global-load register after STS; that value identity ends at publication.",
    ],
  }
  assert report["artifacts"]["llama"]["sha256"] == "04eb9bcb2edef62c672b5496d743a98c57e3236558b88f2ff117964b7fbb91ca"
  assert report["artifacts"]["candidate"]["sha256"] == "6eb663b3a3fd628e3394a0ce8f8780e108e47f40b887b0a75a0756dcf33e9137"
  (HERE / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(json.dumps({name: {
    "panel0": body["panel0"] | {"rows_by_logical_word": len(body["panel0"]["rows_by_logical_word"])},
    "panel1": body["panel1"] | {"rows_by_logical_word": len(body["panel1"]["rows_by_logical_word"])},
    "prefetch_tail": body["panel1_prefetch_tail_before_overwrite_barrier"],
  } for name, body in report["bodies"].items()}, indent=2))


if __name__ == "__main__": main()
