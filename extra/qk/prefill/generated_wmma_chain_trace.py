#!/usr/bin/env python3
"""Trace generated native-ISA WMMA chains for full prefill schedule-search shapes.

This is a structural diagnostic for the no-LDS `PREFILL_DBUF=1 u0=4,u1=4`
NaN. It compiles the same generated program as `prefill_v2_schedule_search`
and reports per-WMMA register spans plus nearby operand/store provenance.
"""
from __future__ import annotations

import argparse, json, os, sys
from collections import Counter, defaultdict
from typing import Any, Iterable

sys.path.insert(0, os.getcwd())

from tinygrad import Device
from tinygrad.uop.ops import Ops

from extra.qk.prefill import native_isa_l4_stream_probe as sp
from extra.qk.prefill_v2_schedule_search import _compile_native_program


def _mn(inst: Any) -> str:
  return sp._mn(inst)


def _insts_for(m: int, n: int, k: int, u0: int, u1: int, loc: int, unr: int) -> tuple[list[Any], Any]:
  prg = _compile_native_program(m, n, k, u0, u1, loc, unr)
  lin = next(u for u in prg.src if u.op is Ops.LINEAR)
  ren = Device[Device.DEFAULT].renderer
  final_uops = sp._final_stream(ren, lin.src)
  return sp._insts_from_uops(final_uops), prg


def _span_regs(span: dict[str, int] | None) -> list[int]:
  if span is None or span.get("kind") != "v": return []
  return list(range(span["lo"], span["hi"] + 1))


def _def_spans(inst: Any) -> list[tuple[str, dict[str, int]]]:
  if isinstance(inst, tuple): return []
  name = _mn(inst)
  spans = sp._field_spans(inst)
  fields = ["vdst", "vdata", "sdata", "sdst"]
  if "store" in name: return []
  return [(f, spans[f]) for f in fields if f in spans and spans[f]["kind"] == "v"]


def _def_map(insts: list[Any]) -> dict[int, list[dict[str, Any]]]:
  defs: dict[int, list[dict[str, Any]]] = defaultdict(list)
  for idx, inst in enumerate(insts):
    if isinstance(inst, tuple): continue
    for field, span in _def_spans(inst):
      for reg in _span_regs(span):
        defs[reg].append({"idx": idx, "op": _mn(inst), "field": field, "text": str(inst)})
  return defs


def _last_def(defs: dict[int, list[dict[str, Any]]], reg: int, before: int) -> dict[str, Any] | None:
  rows = defs.get(reg, [])
  out = None
  for row in rows:
    if row["idx"] < before: out = row
    else: break
  return out


def _operand_defs(defs: dict[int, list[dict[str, Any]]], span: dict[str, int] | None, before: int) -> list[dict[str, Any]]:
  out = []
  for reg in _span_regs(span):
    row = _last_def(defs, reg, before)
    out.append({"reg": reg, "def_idx": None if row is None else row["idx"], "def_op": None if row is None else row["op"]})
  return out


def _classify_defs(rows: list[dict[str, Any]]) -> str:
  ops = {r["def_op"] for r in rows}
  if ops == {"global_load_b128"}: return "global_load_b128"
  if ops == {"ds_load_b128"}: return "ds_load_b128"
  if ops == {"v_wmma_f32_16x16x16_f16"}: return "wmma_accum"
  if ops == {None}: return "undef"
  return "mixed:" + ",".join(str(x) for x in sorted(ops, key=str))


def _op_rows(insts: list[Any], name: str, fields: tuple[str, ...]) -> list[dict[str, Any]]:
  return sp._interesting_rows(insts, name, fields)


def _wmma_rows(insts: list[Any]) -> list[dict[str, Any]]:
  defs = _def_map(insts)
  rows = _op_rows(insts, sp.WMMA_NAME, ("vdst", "src0", "src1", "src2"))
  out = []
  for ordinal, row in enumerate(rows):
    spans = row["spans"]
    entry = {"ordinal": ordinal, "idx": row["idx"], "text": row["text"], "spans": spans}
    for operand in ("src0", "src1", "src2"):
      drows = _operand_defs(defs, spans.get(operand), row["idx"])
      entry[f"{operand}_origin"] = _classify_defs(drows)
      entry[f"{operand}_defs"] = drows
    out.append(entry)
  return out


def _global_store_rows(insts: list[Any]) -> list[dict[str, Any]]:
  defs = _def_map(insts)
  rows = _op_rows(insts, "global_store_b16", ("addr", "vaddr", "saddr", "data", "data0"))
  out = []
  for ordinal, row in enumerate(rows):
    spans = row["spans"]
    data_span = spans.get("data") or spans.get("data0")
    drows = _operand_defs(defs, data_span, row["idx"])
    out.append({
      "ordinal": ordinal,
      "idx": row["idx"],
      "text": row["text"],
      "spans": spans,
      "data_origin": _classify_defs(drows),
      "data_defs": drows,
    })
  return out


def _span_key(span: dict[str, int] | None) -> str:
  if span is None: return "none"
  return f"{span.get('kind')}[{span.get('lo')}:{span.get('hi')}]"


def _summary(insts: list[Any], prg: Any, wmma: list[dict[str, Any]], stores: list[dict[str, Any]]) -> dict[str, Any]:
  counts = Counter(_mn(x) for x in insts if not isinstance(x, tuple))
  acc_groups: dict[str, list[int]] = defaultdict(list)
  for row in wmma:
    acc_groups[_span_key(row["spans"].get("vdst"))].append(row["ordinal"])
  store_addr_groups: dict[str, int] = defaultdict(int)
  for row in stores:
    store_addr_groups[_span_key(row["spans"].get("addr") or row["spans"].get("vaddr"))] += 1
  wmma_span_sequence = [f"{_span_key(r['spans'].get('vdst'))}:{r['src2_origin']}" for r in wmma]
  return {
    "program": str(prg.arg),
    "instruction_count": sum(counts.values()),
    "counts": dict(sorted(counts.items())),
    "wmma_count": len(wmma),
    "global_store_b16_count": len(stores),
    "wmma_origin_counts": dict(Counter(f"{r['src0_origin']}/{r['src1_origin']}/{r['src2_origin']}" for r in wmma)),
    "wmma_accum_sequence": wmma_span_sequence,
    "accumulator_span_groups": [{"span": k, "wmma_ordinals": v, "count": len(v)} for k, v in sorted(acc_groups.items())],
    "store_addr_span_groups": [{"span": k, "count": v} for k, v in sorted(store_addr_groups.items())],
  }


def trace_shape(m: int, n: int, k: int, shape: tuple[int, int], loc: int, unr: int, full_rows: bool) -> dict[str, Any]:
  u0, u1 = shape
  insts, prg = _insts_for(m, n, k, u0, u1, loc, unr)
  wmma = _wmma_rows(insts)
  stores = _global_store_rows(insts)
  payload = {"shape": f"{u0}x{u1}", "u0": u0, "u1": u1, **_summary(insts, prg, wmma, stores)}
  if full_rows:
    payload["wmma_rows"] = wmma
    payload["global_store_rows"] = stores
  else:
    def small_wmma(row: dict[str, Any]) -> dict[str, Any]:
      return {"ordinal": row["ordinal"], "idx": row["idx"],
              "vdst": _span_key(row["spans"].get("vdst")), "src0": _span_key(row["spans"].get("src0")),
              "src1": _span_key(row["spans"].get("src1")), "src2": _span_key(row["spans"].get("src2")),
              "src0_origin": row["src0_origin"], "src1_origin": row["src1_origin"], "src2_origin": row["src2_origin"]}
    def small_store(row: dict[str, Any]) -> dict[str, Any]:
      return {"ordinal": row["ordinal"], "idx": row["idx"],
              "addr": _span_key(row["spans"].get("addr") or row["spans"].get("vaddr")),
              "data": _span_key(row["spans"].get("data") or row["spans"].get("data0")),
              "data_origin": row["data_origin"]}
    payload["wmma_rows_sample"] = [small_wmma(r) for r in (wmma[:8] + (wmma[-8:] if len(wmma) > 16 else []))]
    payload["global_store_rows_sample"] = [small_store(r) for r in (stores[:8] + (stores[-8:] if len(stores) > 16 else []))]
  return payload


def _parse_shapes(raw: str) -> list[tuple[int, int]]:
  out = []
  for item in raw.split(";"):
    if not item.strip(): continue
    a, b = item.split(",", 1)
    out.append((int(a), int(b)))
  return out


def main(argv: Iterable[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--shapes", default="2,2;4,2;2,4;4,4")
  ap.add_argument("--m", type=int, default=512)
  ap.add_argument("--n", type=int, default=5120)
  ap.add_argument("--k", type=int, default=5120)
  ap.add_argument("--loc", type=int, default=2)
  ap.add_argument("--unr", type=int, default=2)
  ap.add_argument("--full-rows", action="store_true")
  args = ap.parse_args(list(argv) if argv is not None else None)

  rows = []
  for shape in _parse_shapes(args.shapes):
    try:
      rows.append(trace_shape(args.m, args.n, args.k, shape, args.loc, args.unr, args.full_rows))
    except Exception as e:
      rows.append({"shape": f"{shape[0]}x{shape[1]}", "ok": False, "error": f"{type(e).__name__}: {e}"})
  print(json.dumps({"m": args.m, "n": args.n, "k": args.k, "loc": args.loc, "unr": args.unr, "rows": rows}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
