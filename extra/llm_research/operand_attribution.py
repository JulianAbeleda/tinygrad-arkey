"""Bounded ABI-rooted dataflow attribution for the exact shipping AMD final ISA.

Given the parsed final-ISA rows (from `analyze_final_isa`, which already carries per-instruction
`reads`/`writes` in AMD destination-first convention) and the compiled PROGRAM ABI, derive the
semantic operand each memory/WMMA row serves — WITHOUT inferring from route names, source schedules,
LDS totals, or an alternate binary. The only roots are the kernarg pointer loads; taint propagates by
register def-use in program order. Any row whose ownership cannot be traced stays explicit `unknown`
with a named missing discriminator. Never guess.

Operand identity comes only from the ABI: `outs`/`ins` index the kernarg pointers, so arg-index ->
operand-id ("out", and "a"/"b"... for inputs in ABI order). This module emits nothing about serving
tier or transport strategy — that stays BoltBeam's orthogonal job.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

# instruction classes we attribute
_GLOBAL_LOAD = "global_load"
_DS_STORE = "lds_store"
_DS_LOAD = "lds_load"

# mnemonics that only move/compute addresses (taint-propagating, not memory ops)
_ADDR_PROPAGATE_PREFIXES = (
  "v_mov", "v_add", "v_or", "v_lshl", "v_lshr", "v_ashr", "v_and", "v_mad", "v_mul", "v_sub",
  "s_mov", "s_add", "s_lshl", "s_or",
)


def _arg_operand_ids(abi: Mapping[str, Any]) -> dict[int, str]:
  """arg-index -> operand id, from the ABI alone. Output is 'out'; inputs are 'a','b',... in ABI order."""
  outs = tuple(abi.get("outs", ()))
  ins = tuple(abi.get("ins", ()))
  ids: dict[int, str] = {arg: "out" for arg in outs}
  for pos, arg in enumerate(ins):
    ids[arg] = chr(ord("a") + pos)   # first input -> 'a', second -> 'b', ...
  return ids


def _kernarg_load_assignment(row: Mapping[str, Any], operand_of_arg: dict[int, str]) -> dict[str, str]:
  """For one kernarg `s_load`, map each loaded destination SGPR to its operand id.

  The kernarg base is the SGPR pair the load reads; each 64-bit pointer arg sits at byte offset
  arg_index*8. Non-8-aligned or unmapped loads assign nothing (their consumers stay unknown)."""
  offset = _trailing_offset(row["operands"])
  if offset is None or offset % 8 != 0:
    return {}
  dst = [r for r in row["writes"] if r.startswith("s") and r[1:].isdigit()]
  assigned: dict[str, str] = {}
  for i in range(0, len(dst) - 1, 2):
    arg_index = (offset + (i // 2) * 8) // 8
    operand = operand_of_arg.get(arg_index)
    if operand is not None:
      assigned[dst[i]] = operand         # both dwords of the pointer pair carry the identity
      assigned[dst[i + 1]] = operand
  return assigned


def _trailing_offset(operands: str) -> int | None:
  """The immediate byte offset of an s_load ('..., s[0:1], 0x10' -> 16; '..., null' -> 0)."""
  tail = operands.rsplit(",", 1)[-1].strip()
  if tail in ("null", "off", ""):
    return 0
  try:
    return int(tail, 16) if tail.lower().startswith("0x") else int(tail)
  except ValueError:
    return None


def _addr_offset(operands: str) -> int:
  """DS/global explicit `offset:N` (default 0)."""
  import re
  m = re.search(r"offset:(\d+)", operands)
  return int(m.group(1)) if m else 0


def attribute_operands(rows: Sequence[Mapping[str, Any]], abi: Mapping[str, Any]) -> dict[str, Any]:
  """Return per-row operand attribution. Rows are the `analyze_final_isa` instruction list."""
  operand_of_arg = _arg_operand_ids(abi)
  sgpr_roots: dict[str, str] = {}            # reported roots, seeded in program order below
  taint: dict[str, str] = {}                 # register -> operand id (address/data provenance)
  lds_regions: list[tuple[int, str]] = []    # (byte offset, operand) written by ds_store
  attributed: dict[int, dict[str, Any]] = {}

  for row in rows:
    idx, mnem = row["index"], row["mnemonic"]
    reads, writes = row["reads"], row["writes"]
    cls = row["instruction_class"]

    if mnem.startswith("s_load"):
      # kernarg pointer load establishes (or refreshes) SGPR roots at its point in program order
      assigned = _kernarg_load_assignment(row, operand_of_arg)
      for w in writes:
        if w in assigned:
          taint[w] = assigned[w]; sgpr_roots[w] = assigned[w]
        else:
          taint.pop(w, None)               # non-kernarg s_load result carries no operand identity
      continue

    if cls == _GLOBAL_LOAD:
      owners = {taint[r] for r in reads if r in taint}
      operand = _sole(owners)
      if operand:
        attributed[idx] = {"operand_id": operand, "kind": "global_load",
                           "fetch_group": f"{operand}:global", "semantic_ownership": "abi_dataflow"}
        for w in writes:
          taint[w] = operand                 # loaded data carries the operand
      else:
        attributed[idx] = _unknown_row("global_load", "global_load_address_provenance")
        for w in writes:
          taint.pop(w, None)
      continue

    if cls == _DS_STORE:
      # ds_store vBase, vData  -> reads = [base, *data]; data provenance is the operand
      data_owners = {taint[r] for r in reads[1:] if r in taint}
      operand = _sole(data_owners)
      off = _addr_offset(row["operands"])
      if operand:
        lds_regions.append((off, operand))
        attributed[idx] = {"operand_id": operand, "kind": "lds_store", "fetch_group": f"{operand}:lds@{off}",
                           "source_operand_id": operand, "semantic_ownership": "abi_dataflow"}
      else:
        attributed[idx] = _unknown_row("lds_store", "lds_store_data_provenance")
      continue

    if cls == _DS_LOAD:
      off = _addr_offset(row["operands"])
      operand = _lds_region_operand(off, lds_regions)
      if operand:
        attributed[idx] = {"operand_id": operand, "kind": "lds_load", "fetch_group": f"{operand}:lds@{off}",
                           "source_operand_id": operand, "semantic_ownership": "abi_dataflow"}
        for w in writes:
          taint[w] = operand
      else:
        attributed[idx] = _unknown_row("lds_load", "double_buffered_lds_window_binding")
        for w in writes:
          taint.pop(w, None)
      continue

    if "wmma" in mnem or "mfma" in mnem or "_dot" in mnem:
      # v_wmma vD, vSrcA, vSrcB, vC : the two source fragments are the A and B operands
      pieces = [p.strip() for p in row["operands"].split(",")]
      srcs = pieces[1:3] if len(pieces) >= 3 else []
      frag = []
      for pos, piece in enumerate(srcs):
        regs = _expand(piece)
        owners = {taint[r] for r in regs if r in taint}
        frag.append(_sole(owners))
      if all(frag) and len(set(frag)) == len(frag):
        attributed[idx] = {"kind": "wmma", "source_operands": frag, "semantic_ownership": "abi_dataflow"}
      else:
        attributed[idx] = _unknown_row("wmma", "wmma_source_fragment_provenance")
      continue

    # address arithmetic: propagate taint from tainted reads to writes; scrub on merge/overwrite
    if mnem.startswith(_ADDR_PROPAGATE_PREFIXES):
      owners = {taint[r] for r in reads if r in taint}
      if len(owners) == 1:
        for w in writes:
          taint[w] = next(iter(owners))
      else:
        for w in writes:            # untainted or conflicting inputs -> definition is not operand-owned
          taint.pop(w, None)
    else:
      for w in writes:
        taint.pop(w, None)          # any other op redefining a reg clears its provenance

  return {
    "authority": "abi_dataflow_v1",
    "abi": {"operand_of_arg": {str(k): v for k, v in operand_of_arg.items()}},
    "sgpr_pointer_roots": sgpr_roots,
    "rows": attributed,
  }


def _manifest_id(operand: str) -> str:
  """Uppercase manifest id for any semantic operand: 'out' -> 'C' (accumulator), otherwise a->A, b->B, ...
  Generalizes to N operands without a fixed table, so non-GEMM/many-input kernels are handled uniformly."""
  return "C" if operand == "out" else operand.upper()


def operand_paths_for_manifest(attribution: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *,
                               binary_sha256: str) -> list[dict[str, Any]]:
  """Flatten the attribution into `tinygrad.amd_isa_proof_manifest.v1`-shaped operand-path rows, joined
  to the exact binary. Uppercase A/B/C manifest ids; unknown rows are kept explicit with their
  discriminator so no relevant final row is silently dropped."""
  by_index = {r["index"]: r for r in rows}
  out: list[dict[str, Any]] = []
  for idx in sorted(attribution["rows"]):
    a = attribution["rows"][idx]
    row = by_index.get(idx, {})
    base = {"row_index": idx, "pc": row.get("pc"), "kind": a["kind"], "binary_sha256": binary_sha256,
            "semantic_ownership": a.get("semantic_ownership", "unknown")}
    if a["kind"] == "wmma":
      srcs = a.get("source_operands")
      if srcs and all(s and s != "unknown" for s in srcs):
        out.append({**base, "operand_id": "C",
                    "source_operands": [_manifest_id(s) for s in srcs]})
      else:
        out.append({**base, "operand_id": "unknown", "missing": a.get("missing", "wmma_source_fragment_provenance")})
      continue
    operand = a.get("operand_id", "unknown")
    if operand != "unknown":
      mid = _manifest_id(operand)
      path = {**base, "operand_id": mid, "source_operand_id": mid}
      if "fetch_group" in a:
        path["fetch_group"] = a["fetch_group"]
      out.append(path)
    else:
      out.append({**base, "operand_id": "unknown", "missing": a.get("missing", "operand_final_program_attribution")})
  return out


def _sole(owners: set[str]) -> str | None:
  return next(iter(owners)) if len(owners) == 1 else None


def _lds_region_operand(offset: int, regions: list[tuple[int, str]]) -> str | None:
  # Conservative: attribute a ds_load only when its byte offset EXACTLY matches a ds_store region for a
  # single operand. Double-buffered A/B windows are not disambiguable from the shipping final ISA, so a
  # ds_load into an unmatched (second-buffer) offset stays unknown rather than risk a cross-buffer guess.
  exact = {op for off, op in regions if off == offset}
  return _sole(exact)


def _expand(piece: str) -> list[str]:
  import re
  m = re.match(r"v\[(\d+):(\d+)\]", piece)
  if m:
    return [f"v{i}" for i in range(int(m.group(1)), int(m.group(2)) + 1)]
  m = re.match(r"(v\d+)", piece)
  return [m.group(1)] if m else []


def _unknown_row(kind: str, missing: str) -> dict[str, Any]:
  return {"operand_id": "unknown", "kind": kind, "semantic_ownership": "unknown", "missing": missing}
