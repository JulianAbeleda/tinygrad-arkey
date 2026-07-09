#!/usr/bin/env python3
"""A-side LDS fragment-window alias probe for prefill-v2 DBUF candidates.

This is a structural final-ISA probe. It compiles a schedule-search matmul,
tracks coarse address/data origins through the final instruction stream, and
reports whether WMMA src0 A LDS loads have a bounded identity back to A
ds_store_b128 windows.
"""
from __future__ import annotations

import argparse, contextlib, io, json, os, re, sys, traceback
from collections import Counter
from typing import Any

sys.path.insert(0, os.getcwd())

from tinygrad.codegen import to_program_cache
from tinygrad.helpers import Context, getenv
from tinygrad.renderer.amd.elf import group_segment_fixed_size_from_elf
from tinygrad.uop.ops import Ops

from extra.qk.prefill import native_isa_l4_stream_probe as sp
from extra.qk.prefill.hand_vs_generated_shape_matrix import DEFAULT_DBUF_ENV
from extra.qk.prefill_v2_schedule_search import _compile_native_program


DEFAULT_ENV = {k: v for k, v in DEFAULT_DBUF_ENV.items() if k not in ("DEV", "REGALLOC_ADDR_REMAT")}

_PROOF_RE = re.compile(r"'role': '([^']+)'.*?'const': (-?\d+).*?'width': (-?\d+)", re.S)
_BUF_RE = re.compile(r"\('DEFINE_LOCAL',.*?,\s*(-?\d+),\s*\(\)\)")


def _parse_preisel_proof(text: str) -> dict[str, Any]:
  rows = []
  for line in text.splitlines():
    if "LDS_PROOF_KEY" not in line: continue
    m = _PROOF_RE.search(line)
    if m is None: continue
    bm = _BUF_RE.search(line)
    rows.append({
      "role": m.group(1),
      "buf": int(bm.group(1)) if bm is not None else None,
      "const": int(m.group(2)),
      "width": int(m.group(3)),
      "has_lidx1": "lidx1" in line,
    })
  def summarize(stores: list[dict[str, Any]], loads: list[dict[str, Any]]) -> dict[str, Any]:
    store_keys = {(r["buf"], r["const"], r["width"]) for r in stores}
    store_const_counts = {k: len({r["const"] for r in stores if r["buf"] == k}) for k in sorted({r["buf"] for r in stores})}
    load_const_counts = {k: len({r["const"] for r in loads if r["buf"] == k}) for k in sorted({r["buf"] for r in loads})}
    covered_loads = []
    missing_loads = []
    for ld in loads:
      # The proof dump reports LDS offsets in half elements. A frag load is 16 halfs wide; the current b128 store bridge
      # reports two adjacent 8-half store rows. Treat those as covering the load window.
      if ld["width"] == 16 and (ld["buf"], ld["const"], 8) in store_keys and (ld["buf"], ld["const"] + 8, 8) in store_keys:
        covered_loads.append(ld)
      elif (ld["buf"], ld["const"], ld["width"]) in store_keys:
        covered_loads.append(ld)
      else:
        missing_loads.append(ld)
    return {
      "store_count": len(stores),
      "load_count": len(loads),
      "store_count_by_buf": {str(k): sum(1 for r in stores if r["buf"] == k) for k in sorted({r["buf"] for r in stores})},
      "load_count_by_buf": {str(k): sum(1 for r in loads if r["buf"] == k) for k in sorted({r["buf"] for r in loads})},
      "store_const_count_by_buf": {str(k): v for k, v in store_const_counts.items()},
      "load_const_count_by_buf": {str(k): v for k, v in load_const_counts.items()},
      "store_consts": sorted({(r["buf"], r["const"], r["width"]) for r in stores}),
      "load_consts": sorted({(r["buf"], r["const"], r["width"]) for r in loads}),
      "covered_load_count": len(covered_loads),
      "missing_load_count": len(missing_loads),
      "ok": bool(loads) and not missing_loads,
    }
  a_stores = [r for r in rows if r["role"] == "withlocal_store_b128"]
  a_loads = [r for r in rows if r["role"] in ("frag_load_b128_A", "frag_load_b128_pack")]
  all_stores = [r for r in rows if r["role"] in ("withlocal_store_b128", "tilekey_store_b128")]
  all_loads = [r for r in rows if r["role"].startswith("frag_load_b128")]
  a_summary = summarize(a_stores, a_loads)
  all_summary = summarize(all_stores, all_loads)
  return {
    "row_count": len(rows),
    "roles": dict(Counter(r["role"] for r in rows)),
    "a": a_summary,
    "all_operands": all_summary,
    # Back-compat fields used by quick terminal summaries.
    "store_count": a_summary["store_count"],
    "load_count": a_summary["load_count"],
    "store_count_by_buf": a_summary["store_count_by_buf"],
    "load_count_by_buf": a_summary["load_count_by_buf"],
    "store_const_count_by_buf": a_summary["store_const_count_by_buf"],
    "load_const_count_by_buf": a_summary["load_const_count_by_buf"],
    "store_consts": a_summary["store_consts"],
    "load_consts": a_summary["load_consts"],
    "has_bounded_lidx1": any(r["has_lidx1"] for r in rows),
    "covered_load_count": a_summary["covered_load_count"],
    "missing_load_count": a_summary["missing_load_count"],
    "ok": a_summary["ok"] and any(r["has_lidx1"] for r in rows),
    "note": "pre-isel symbolic LDS proof; final physical address registers may still be rematerialized differently",
  }


def _reg_key(reg: Any) -> tuple[str, int] | None:
  if reg is None or not hasattr(reg, "offset"): return None
  return ("v", reg.offset - 256) if 256 <= reg.offset < 512 else ("s", reg.offset)


def _const_value(x: Any) -> int | None:
  if isinstance(x, int): return x
  s = str(x)
  return int(s) if s.lstrip("-").isdigit() else None


def _iadd(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]: return (a[0] + b[0], a[1] + b[1])
def _imul(a: tuple[int, int], k: int) -> tuple[int, int]: return (a[0] * k, a[1] * k) if k >= 0 else (a[1] * k, a[0] * k)
def _ilshl(a: tuple[int, int], k: int) -> tuple[int, int]: return _imul(a, 1 << k)
def _ilshr(a: tuple[int, int], k: int) -> tuple[int, int]: return (max(0, a[0]) >> k, max(0, a[1]) >> k)


AffExpr = tuple[tuple[tuple[str, int], ...], int]


def _expr_const(v: int) -> AffExpr: return (tuple(), v)
def _expr_term(name: str, scale: int=1) -> AffExpr: return (((name, scale),), 0)


def _expr_add(a: AffExpr, b: AffExpr) -> AffExpr:
  terms: dict[str, int] = {}
  for name, scale in a[0] + b[0]: terms[name] = terms.get(name, 0) + scale
  return (tuple(sorted((name, scale) for name, scale in terms.items() if scale)), a[1] + b[1])


def _expr_mul(a: AffExpr, k: int) -> AffExpr:
  return (tuple(sorted((name, scale * k) for name, scale in a[0] if scale * k)), a[1] * k)


def _expr_shr(a: AffExpr, k: int) -> AffExpr | None:
  div = 1 << k
  if a[1] % div: return None
  out = []
  for name, scale in a[0]:
    if scale % div: return None
    out.append((name, scale // div))
  return (tuple(sorted((name, scale) for name, scale in out if scale)), a[1] // div)


def _expr_base_key(e: AffExpr) -> str:
  if not e[0]: return "const"
  return "+".join(f"{scale}*{name}" if scale != 1 else name for name, scale in e[0])


def _expr_named(name: str, e: AffExpr, *args: int) -> AffExpr:
  base = _expr_base_key((e[0], 0))
  suffix = ",".join(str(x) for x in args)
  return _expr_add(_expr_term(f"{name}({base},{e[1]}{',' if suffix else ''}{suffix})"), _expr_const(0))


def _expr_range(e: AffExpr, ranges: dict[str, tuple[int, int]]) -> tuple[int, int] | None:
  lo = hi = e[1]
  for name, scale in e[0]:
    if name not in ranges: return None
    rlo, rhi = ranges[name]
    vals = (rlo * scale, rhi * scale)
    lo += min(vals); hi += max(vals)
  return (lo, hi)


def _offset(inst: Any) -> int:
  for name in ("offset0", "offset"):
    if hasattr(inst, name): return int(getattr(inst, name))
  return sp._trailing_imm(str(inst))


def _width(name: str) -> int:
  if name.endswith("b128"): return 16
  if name.endswith("b64"): return 8
  if name.endswith("b32"): return 4
  if name.endswith("b16") or name.endswith("u16"): return 2
  return 4


def _span_regs(span: dict[str, int] | None) -> list[int]:
  if span is None or span.get("kind") != "v": return []
  return list(range(span["lo"], span["hi"] + 1))


def _window_key(addr_range: tuple[int, int] | None, imm: int, width: int, fallback: str, extent: int | None) -> str:
  if addr_range is None: return fallback
  lo, hi = addr_range[0] + imm, addr_range[1] + imm + width
  if lo < 0 or (extent is not None and hi > extent): return fallback
  return f"bytes:{lo}:{hi}"


def _ds_window(addr_expr: AffExpr | None, addr_range: tuple[int, int] | None, imm: int, width: int, fallback: str,
               extent: int | None, term_ranges: dict[str, tuple[int, int]]) -> dict[str, Any]:
  if addr_expr is None:
    return {"status": "unknown", "window": fallback, "normalized_window": None, "reason": "address_expr_unknown"}
  total = addr_expr[1] + imm
  base_expr = (addr_expr[0], 0)
  dyn_range = _expr_range(base_expr, term_ranges)
  if dyn_range is not None and (dyn_range[0] + total < 0 or (extent is not None and dyn_range[1] + total + width > extent)):
    return {"status": "out_of_bounds", "window": fallback, "normalized_window": None,
            "reason": f"window outside LDS extent {extent}"}
  base = _expr_base_key(base_expr)
  lo, hi = total, total + width
  return {"status": "known", "window": f"{base}:bytes:{lo}:{hi}", "normalized_window": {"base": base, "lo": lo, "hi": hi},
          "reason": "normalized_materialized_const_and_ds_imm", "dynamic_byte_range": dyn_range}


def _origin_summary(origins: list[dict[str, Any]]) -> dict[str, Any]:
  ptrs = sorted({o.get("ptr", "unknown") for o in origins})
  return {
    "ptrs": ptrs,
    "global_indices": sorted({o["idx"] for o in origins if o.get("kind") == "global"}),
    "lds_load_indices": sorted({o["load_idx"] for o in origins if o.get("kind") == "lds_load"}),
  }


def _analyze(insts: list[Any], prg: Any) -> dict[str, Any]:
  info = prg.arg
  group_segment = next((group_segment_fixed_size_from_elf(u.arg) for u in prg.src if u.op is Ops.BINARY), None)
  ptr_names = {0: "C", 1: "A", 2: "B"}
  sgpr_ptr: dict[int, int] = {}
  vrange: dict[tuple[str, int], tuple[int, int]] = {("v", 0): (0, max(info.local_size[0] - 1, 0))}
  srange: dict[tuple[str, int], tuple[int, int]] = {
    ("s", 2): (0, max(info.global_size[0] - 1, 0)),
    ("s", 3): (0, max(info.global_size[1] - 1, 0)),
    ("s", 4): (0, max(info.global_size[2] - 1, 0)),
  }
  vexpr: dict[tuple[str, int], AffExpr] = {("v", 0): _expr_term("v0")}
  sexpr: dict[tuple[str, int], AffExpr] = {}
  term_ranges = {"v0": vrange[("v", 0)], "s2": srange[("s", 2)], "s3": srange[("s", 3)], "s4": srange[("s", 4)]}
  vorig: dict[int, list[dict[str, Any]]] = {}
  stores, loads, wmmas = [], [], []
  unknown_defs = 0

  def rng_of(x: Any) -> tuple[int, int] | None:
    if (cv := _const_value(x)) is not None: return (cv, cv)
    if str(x) == "LIT": return None
    k = _reg_key(x)
    return None if k is None else (vrange.get(k) if k[0] == "v" else srange.get(k))

  def expr_of(x: Any) -> AffExpr | None:
    if (cv := _const_value(x)) is not None: return _expr_const(cv)
    if str(x) == "LIT": return None
    k = _reg_key(x)
    if k is None: return None
    return vexpr.get(k) if k[0] == "v" else sexpr.get(k, _expr_term(f"s{k[1]}"))

  def set_vrange(reg: Any, val: tuple[int, int]) -> None:
    if (k := _reg_key(reg)) is not None and k[0] == "v": vrange[k] = val

  def set_vexpr(reg: Any, val: AffExpr | None) -> None:
    if (k := _reg_key(reg)) is not None and k[0] == "v":
      if val is None: vexpr.pop(k, None)
      else: vexpr[k] = val

  def set_sexpr(reg: Any, val: AffExpr | None) -> None:
    if (k := _reg_key(reg)) is not None and k[0] == "s":
      if val is None: sexpr.pop(k, None)
      else: sexpr[k] = val

  def set_vorig(span: dict[str, int] | None, origins: list[dict[str, Any]]) -> None:
    for r in _span_regs(span): vorig[r] = origins

  def origins_for(span: dict[str, int] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen = set()
    for r in _span_regs(span):
      for item in vorig.get(r, [{"kind": "unknown"}]):
        key = tuple(sorted(item.items()))
        if key not in seen:
          seen.add(key); out.append(item)
    return out

  for idx, inst in enumerate(insts):
    if isinstance(inst, tuple): continue
    name = sp._mn(inst)
    spans = sp._field_spans(inst)
    try:
      if name == "s_load_b64" and getattr(inst, "sbase", None) is not None and getattr(inst, "sbase").offset == 0:
        param_idx = int(getattr(inst, "offset")) // 8
        if (k := _reg_key(getattr(inst, "sdata", None))) is not None: sgpr_ptr[k[1]] = param_idx
      elif name == "s_mov_b32":
        dst = _reg_key(getattr(inst, "sdst", getattr(inst, "sdst0", None)))
        src = getattr(inst, "ssrc0", getattr(inst, "src0", None))
        if dst is not None and (cv := _const_value(src)) is not None:
          srange[dst] = (cv, cv); set_sexpr(getattr(inst, "sdst", getattr(inst, "sdst0", None)), _expr_const(cv))
      elif name == "s_cmp_lt_i32":
        src0, src1 = getattr(inst, "ssrc0", getattr(inst, "src0", None)), getattr(inst, "ssrc1", getattr(inst, "src1", None))
        if (k := _reg_key(src0)) is not None and (cv := _const_value(src1)) is not None:
          srange[k] = (0, max(cv - 1, 0)); term_ranges[f"{k[0]}{k[1]}"] = srange[k]
      elif name == "v_mov_b32_e32":
        src0 = getattr(inst, "src0")
        val = (getattr(inst, "literal"), getattr(inst, "literal")) if str(src0) == "LIT" else rng_of(src0)
        if val is not None: set_vrange(getattr(inst, "vdst"), val)
        set_vexpr(getattr(inst, "vdst"), _expr_const(int(getattr(inst, "literal"))) if str(src0) == "LIT" else expr_of(src0))
        src_span = spans.get("src0")
        if src_span is not None: set_vorig(spans.get("vdst"), origins_for(src_span))
      elif name == "v_and_b32_e32":
        lhs, rhs = rng_of(getattr(inst, "src0")), rng_of(getattr(inst, "vsrc1"))
        if lhs is not None and rhs is not None:
          mask = lhs[0] if lhs[0] == lhs[1] else None
          set_vrange(getattr(inst, "vdst"), (0, min(mask, rhs[1])) if mask is not None and mask >= 0 else (0, max(lhs[1], rhs[1])))
        le, re = expr_of(getattr(inst, "src0")), expr_of(getattr(inst, "vsrc1"))
        lcv, rcv = _const_value(getattr(inst, "src0")), _const_value(getattr(inst, "vsrc1"))
        if lcv is not None and re is not None: set_vexpr(getattr(inst, "vdst"), _expr_named("and", re, int(lcv)))
        elif rcv is not None and le is not None: set_vexpr(getattr(inst, "vdst"), _expr_named("and", le, int(rcv)))
        else: set_vexpr(getattr(inst, "vdst"), None)
      elif name == "v_bfe_u32":
        src, off, bits = expr_of(getattr(inst, "src0")), _const_value(getattr(inst, "src1")), _const_value(getattr(inst, "src2"))
        if off is not None and bits is not None: set_vrange(getattr(inst, "vdst"), (0, max((1 << bits) - 1, 0)))
        set_vexpr(getattr(inst, "vdst"), _expr_named("bfe", src, int(off), int(bits)) if src is not None and off is not None and bits is not None else None)
      elif name in ("v_cmp_lt_i32_e32", "v_cmp_ne_u32_e32", "v_cndmask_b32_e32"):
        set_vexpr(getattr(inst, "vdst", None), None)
      elif name == "v_mul_lo_u32":
        lhs = rng_of(getattr(inst, "src0"))
        rhs = getattr(inst, "literal") if str(getattr(inst, "src1")) == "LIT" else _const_value(getattr(inst, "src1"))
        if lhs is not None and rhs is not None: set_vrange(getattr(inst, "vdst"), _imul(lhs, int(rhs)))
        e = expr_of(getattr(inst, "src0"))
        set_vexpr(getattr(inst, "vdst"), _expr_mul(e, int(rhs)) if e is not None and rhs is not None else None)
      elif name == "v_add_nc_u32_e32":
        lhs = (getattr(inst, "literal"), getattr(inst, "literal")) if str(getattr(inst, "src0")) == "LIT" else rng_of(getattr(inst, "src0"))
        rhs = rng_of(getattr(inst, "vsrc1"))
        if lhs is not None and rhs is not None: set_vrange(getattr(inst, "vdst"), _iadd(lhs, rhs))
        le = _expr_const(int(getattr(inst, "literal"))) if str(getattr(inst, "src0")) == "LIT" else expr_of(getattr(inst, "src0"))
        re = expr_of(getattr(inst, "vsrc1"))
        set_vexpr(getattr(inst, "vdst"), _expr_add(le, re) if le is not None and re is not None else None)
      elif name == "v_lshlrev_b32_e32":
        val, sh = rng_of(getattr(inst, "vsrc1")), _const_value(getattr(inst, "src0"))
        if val is not None and sh is not None: set_vrange(getattr(inst, "vdst"), _ilshl(val, sh))
        e = expr_of(getattr(inst, "vsrc1"))
        set_vexpr(getattr(inst, "vdst"), _expr_mul(e, 1 << sh) if e is not None and sh is not None else None)
      elif name == "v_lshrrev_b32_e32":
        val, sh = rng_of(getattr(inst, "vsrc1")), _const_value(getattr(inst, "src0"))
        if val is not None and sh is not None: set_vrange(getattr(inst, "vdst"), _ilshr(val, sh))
        e = expr_of(getattr(inst, "vsrc1"))
        set_vexpr(getattr(inst, "vdst"), _expr_shr(e, sh) if e is not None and sh is not None else None)
      elif name.startswith("global_load"):
        saddr = _reg_key(getattr(inst, "saddr", None))
        ptr_idx = sgpr_ptr.get(saddr[1]) if saddr is not None else None
        addr_range, imm, width = rng_of(getattr(inst, "addr", None)), _offset(inst), _width(name)
        origin = {"kind": "global", "ptr": ptr_names.get(ptr_idx, str(ptr_idx)), "idx": idx,
                  "window": _window_key(addr_range, imm, width, sp._addr_key({"spans": spans, "text": str(inst)}), None)}
        set_vorig(spans.get("vdst"), [origin])
      elif name.startswith("ds_store"):
        addr, addr_range, imm, width = getattr(inst, "addr", None), rng_of(getattr(inst, "addr", None)), _offset(inst), _width(name)
        family = sp._addr_key({"spans": spans, "text": str(inst)})
        origins = origins_for(spans.get("data0"))
        win = _ds_window(expr_of(addr), addr_range, imm, width, family, group_segment, term_ranges)
        row = {"idx": idx, "op": name, "window": win["window"], "addr_family": family, "addr_range": addr_range,
               "offset": imm, "width": width, "window_status": win["status"], "window_reason": win["reason"],
               "normalized_window": win["normalized_window"], **_origin_summary(origins)}
        stores.append(row)
      elif name.startswith("ds_load"):
        addr, addr_range, imm, width = getattr(inst, "addr", None), rng_of(getattr(inst, "addr", None)), _offset(inst), _width(name)
        family = sp._addr_key({"spans": spans, "text": str(inst)})
        win = _ds_window(expr_of(addr), addr_range, imm, width, family, group_segment, term_ranges)
        row = {"idx": idx, "op": name, "window": win["window"], "addr_family": family, "addr_range": addr_range,
               "offset": imm, "width": width, "window_status": win["status"], "window_reason": win["reason"],
               "normalized_window": win["normalized_window"]}
        loads.append(row)
        set_vorig(spans.get("vdst"), [{"kind": "lds_load", "load_idx": idx, "window": row["window"]}])
      elif name == sp.WMMA_NAME:
        a_origins = origins_for(spans.get("src0"))
        b_origins = origins_for(spans.get("src1"))
        wmmas.append({"idx": idx, "a": _origin_summary(a_origins), "b": _origin_summary(b_origins)})
    except Exception:
      unknown_defs += 1

  a_stores = [s for s in stores if s["op"] == "ds_store_b128" and "A" in s["ptrs"]]
  a_known_stores = [s for s in a_stores if s["window_status"] == "known"]
  a_store_by_window = {s["window"]: s for s in a_known_stores}
  load_by_idx = {l["idx"]: l for l in loads}
  a_wmma_rows = []
  for w in wmmas:
    lds_load_rows = [load_by_idx[i] for i in w["a"]["lds_load_indices"] if i in load_by_idx]
    known_load_windows = sorted({l["window"] for l in lds_load_rows if l["window_status"] == "known"})
    unknown_loads = [l for l in lds_load_rows if l["window_status"] == "unknown"]
    out_of_bounds_loads = [l for l in lds_load_rows if l["window_status"] == "out_of_bounds"]
    missing = [key for key in known_load_windows if key not in a_store_by_window]
    a_wmma_rows.append({"wmma_idx": w["idx"], "a_load_windows": known_load_windows, "missing_store_windows": missing,
                        "known_a_load_windows": known_load_windows, "unknown_a_load_indices": [l["idx"] for l in unknown_loads],
                        "out_of_bounds_a_load_indices": [l["idx"] for l in out_of_bounds_loads],
                        "a_load_indices": w["a"]["lds_load_indices"]})
  use_counts = Counter(key for row in a_wmma_rows for key in row["known_a_load_windows"])
  alias_windows = [{"window": key, "wmma_count": count} for key, count in sorted(use_counts.items()) if count > 1]
  missing_rows = [row for row in a_wmma_rows if row["missing_store_windows"]]
  inconclusive_rows = [row for row in a_wmma_rows if row["unknown_a_load_indices"] or row["out_of_bounds_a_load_indices"]]
  unknown_ds_rows = [r for r in stores + loads if r["window_status"] == "unknown"]
  out_of_bounds_ds_rows = [r for r in stores + loads if r["window_status"] == "out_of_bounds"]
  return {
    "program": str(info),
    "group_segment_bytes": group_segment,
    "instruction_total": len([x for x in insts if not isinstance(x, tuple)]),
    "track_counts": dict(Counter(sp._mn(x) for x in insts if not isinstance(x, tuple))),
    "unknown_analysis_steps": unknown_defs,
    "a_store_b128_count": len(a_stores),
    "a_store_windows": sorted(a_store_by_window),
    "a_unknown_store_count": len([s for s in a_stores if s["window_status"] == "unknown"]),
    "a_wmma_count": len(a_wmma_rows),
    "a_wmma_rows": a_wmma_rows,
    "a_alias_windows": alias_windows,
    "a_missing_store_window_rows": missing_rows,
    "a_inconclusive_window_rows": inconclusive_rows,
    "ds_window_status_counts": dict(Counter(r["window_status"] for r in stores + loads)),
    "ds_unknown_rows": unknown_ds_rows[:32],
    "ds_out_of_bounds_rows": out_of_bounds_ds_rows[:32],
    "a_fragment_window_identity": {
      "ok": bool(a_wmma_rows) and not missing_rows and not alias_windows and not inconclusive_rows,
      "verdict": "covered" if bool(a_wmma_rows) and not missing_rows and not alias_windows and not inconclusive_rows else (
        "unknown" if inconclusive_rows else "alias" if alias_windows else "out_of_bounds" if out_of_bounds_ds_rows else "missing_store"),
      "store_window_count": len(a_store_by_window),
      "load_window_count": len(use_counts),
      "aliased_window_count": len(alias_windows),
      "missing_store_row_count": len(missing_rows),
      "inconclusive_row_count": len(inconclusive_rows),
      "unknown_ds_window_count": len(unknown_ds_rows),
      "out_of_bounds_ds_window_count": len(out_of_bounds_ds_rows),
      "requirement": "each WMMA src0 LDS load window must match one A ds_store_b128 window and no A load window may be reused by multiple WMMA A fragments",
    },
  }


def _case(args: argparse.Namespace, u0: int, u1: int) -> dict[str, Any]:
  getenv.cache_clear(); to_program_cache.clear()
  try:
    old_proof = os.environ.get("PREFILL_DBUF_LDS_PROOF_KEY_DUMP")
    os.environ["PREFILL_DBUF_LDS_PROOF_KEY_DUMP"] = "1"
    proof_stdout = io.StringIO()
    with Context(DEV=args.dev):
      from tinygrad import Device
      with contextlib.redirect_stdout(proof_stdout):
        prg = _compile_native_program(args.m, args.out_f, args.in_f, u0, u1, args.loc, args.unr)
      lin = next(u for u in prg.src if u.op is Ops.LINEAR)
      final = sp._insts_from_uops(sp._final_stream(Device[Device.DEFAULT].renderer, lin.src))
      return {"u0": u0, "u1": u1, "loc": args.loc, "unr": args.unr, "ok": True,
              "preisel_lds_proof": _parse_preisel_proof(proof_stdout.getvalue()), **_analyze(final, prg)}
  except Exception as e:
    return {"u0": u0, "u1": u1, "loc": args.loc, "unr": args.unr, "ok": False, "error": type(e).__name__,
            "message": str(e), "traceback_tail": traceback.format_exc().splitlines()[-8:]}
  finally:
    if "old_proof" in locals():
      if old_proof is None: os.environ.pop("PREFILL_DBUF_LDS_PROOF_KEY_DUMP", None)
      else: os.environ["PREFILL_DBUF_LDS_PROOF_KEY_DUMP"] = old_proof


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--m", type=int, default=512)
  p.add_argument("--out-f", type=int, default=4096)
  p.add_argument("--in-f", type=int, default=4096)
  p.add_argument("--loc", type=int, default=2)
  p.add_argument("--unr", type=int, default=2)
  p.add_argument("--cases", default="2,2;4,2", help="semicolon-separated u0,u1 cases")
  p.add_argument("--dev", default="AMD:ISA")
  p.add_argument("--no-default-env", action="store_true")
  p.add_argument("--indent", type=int, default=2)
  args = p.parse_args()
  if not args.no_default_env:
    for k, v in DEFAULT_ENV.items(): os.environ.setdefault(k, v)
  cases = [tuple(int(x) for x in case.split(",", 1)) for case in args.cases.split(";") if case]
  report = {
    "schema": "a-fragment-window-alias.v1",
    "env": {k: os.environ.get(k) for k in DEFAULT_ENV},
    "shape": {"m": args.m, "out_f": args.out_f, "in_f": args.in_f, "loc": args.loc, "unr": args.unr},
    "cases": [_case(args, u0, u1) for u0, u1 in cases],
  }
  print(json.dumps(report, indent=args.indent, sort_keys=True))


if __name__ == "__main__":
  main()
