#!/usr/bin/env python3
"""Audit pre-isel LDS stage ownership for generated prefill WMMA routes.

This inspects the graph after `full_rewrite_to_sink` and before AMD isel. It is
the boundary where generated code should still know which LDS producers feed
which WMMA operands.
"""
from __future__ import annotations

import argparse, json, os, sys
from contextlib import contextmanager
from typing import Any

sys.path.insert(0, os.getcwd())

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import full_rewrite_to_sink, to_program_cache
import tinygrad.codegen as cg
from tinygrad.codegen.opt import postrange
from tinygrad.dtype import AddrSpace, PtrDType
from tinygrad.helpers import getenv
from tinygrad.uop.ops import Ops, UOp

from extra.qk.prefill_v2_schedule_search import _opts_for
from extra.qk.prefill.prefill_route_census import KMAJOR_LDS_ENV


@contextmanager
def patched_env(env: dict[str, str]):
  old = {k: os.environ.get(k) for k in env}
  os.environ.update(env)
  getenv.cache_clear()
  to_program_cache.clear()
  try:
    yield
  finally:
    for k, v in old.items():
      if v is None: os.environ.pop(k, None)
      else: os.environ[k] = v
    getenv.cache_clear()
    to_program_cache.clear()


def const_base(x: UOp) -> tuple[str | None, int]:
  if x.op is Ops.CONST: return None, int(x.arg)
  if x.op is Ops.ADD:
    a, ac = const_base(x.src[0])
    b, bc = const_base(x.src[1])
    if a is None: return b, ac + bc
    if b is None: return a, ac + bc
  return key_uop(x), 0


def key_uop(u: UOp) -> str:
  if u.op is Ops.CONST: return f"CONST:{u.arg}"
  if u.op is Ops.RANGE: return f"RANGE:{u.arg}"
  if u.op is Ops.SPECIAL: return f"SPECIAL:{u.arg}"
  if u.op in (Ops.ADD, Ops.MUL, Ops.SHL, Ops.AND, Ops.CMOD):
    return f"{u.op.name}({','.join(key_uop(s) for s in u.src)})"
  if u.op in (Ops.DEFINE_LOCAL, Ops.DEFINE_REG):
    return f"{u.op.name}:{u.arg}:{u.dtype}"
  return f"{u.op.name}:{u.dtype}:{u.arg}:id{id(u)}"


def tag_dict(tag: Any) -> dict[str, Any]:
  if not isinstance(tag, tuple): return {}
  try:
    return {str(k): repr(v) for k, v in tag[1:]} if tag and isinstance(tag[1], tuple) and len(tag[1]) == 2 else {"tag": repr(tag)}
  except Exception:
    return {"tag": repr(tag)}


def unwrap_index(u: UOp) -> UOp:
  while u.op in (Ops.CAST, Ops.AFTER) and u.src: u = u.src[0]
  return u


def unwrap_buffer(u: UOp) -> UOp:
  while u.op in (Ops.AFTER, Ops.CAST) and u.src: u = u.src[0]
  if u.op in (Ops.EXPAND, Ops.GEP) and u.src: return unwrap_buffer(u.src[0])
  return u


def local_index_info(idx: UOp) -> dict[str, Any] | None:
  idx = unwrap_index(idx)
  if idx.op is not Ops.INDEX or not isinstance(idx.dtype, PtrDType) or idx.dtype.addrspace != AddrSpace.LOCAL: return None
  base, const = const_base(idx.src[1])
  buf = unwrap_buffer(idx.src[0])
  return {
    "buffer_id": id(buf),
    "buffer_key": key_uop(buf),
    "index_id": id(idx),
    "dtype": str(idx.dtype),
    "itemsize": idx.dtype.base.itemsize,
    "base_expr": base,
    "const_elems": const,
    "const_bytes": const * idx.dtype.base.itemsize,
    "index_tag": repr(idx.tag),
    "buffer_tag": repr(buf.tag),
    "buffer_tag_fields": tag_dict(buf.tag),
  }


def store_rows(sink: UOp) -> list[dict[str, Any]]:
  rows = []
  for st in sink.toposort():
    if st.op is not Ops.STORE or len(st.src) < 2: continue
    info = local_index_info(st.src[0])
    if info is None: continue
    val = st.src[1]
    width = getattr(val.dtype, "count", 1) if not isinstance(val.dtype, PtrDType) else 1
    scalar = val.dtype.scalar() if hasattr(val.dtype, "scalar") and not isinstance(val.dtype, PtrDType) else val.dtype
    rows.append({
      "store_id": id(st),
      "store_tag": repr(st.tag),
      "store_tag_fields": tag_dict(st.tag),
      "val_op": val.op.name,
      "val_dtype": str(val.dtype),
      "val_width": width,
      "val_scalar": str(scalar),
      "gate": len(st.src) >= 3,
      **info,
    })
  return rows


def stage_rows(sink: UOp) -> list[dict[str, Any]]:
  rows = []
  for stg in sink.toposort():
    if stg.op is not Ops.STAGE: continue
    rows.append({
      "stage_id": id(stg),
      "stage_dtype": str(stg.dtype),
      "stage_arg": repr(stg.arg),
      "stage_tag": repr(stg.tag),
      "stage_src_ops": [s.op.name for s in stg.src],
      "stage_src_dtypes": [str(s.dtype) for s in stg.src],
      "stage_ranges": [repr(r.arg) for r in sorted(stg.ranges, key=lambda r: repr(r.arg))],
    })
  return rows


def wmma_rows(sink: UOp) -> list[dict[str, Any]]:
  from tinygrad.renderer.isa.amd import _wmma_elems, _wmma_half_addr
  rows = []
  for wi, w in enumerate([u for u in sink.toposort() if u.op is Ops.WMMA]):
    for role, carrier in (("A", w.src[0]), ("B", w.src[1])):
      base_row = {
        "wmma_i": wi, "wmma_id": id(w), "role": role, "carrier_id": id(carrier),
        "carrier_op": carrier.op.name, "carrier_dtype": str(carrier.dtype), "carrier_arg": repr(carrier.arg),
        "carrier_tag": repr(carrier.tag), "carrier_src_ops": [s.op.name for s in carrier.src],
      }
      try:
        elems = _wmma_elems(carrier, 16)
        addrs = [_wmma_half_addr(e) for e in elems]
      except Exception as e:
        rows.append({**base_row, "ok": False, "reason": f"{type(e).__name__}:{e}"})
        continue
      if any(a is None for a in addrs):
        rows.append({**base_row, "ok": False, "reason": "non_memory_operand"})
        continue
      first = addrs[0]
      assert first is not None
      idx0, ptr0, _expr0, c0 = first
      info = local_index_info(idx0)
      consts = [a[3] for a in addrs if a is not None]
      rows.append({
        **base_row,
        "ok": info is not None,
        "first_lane_const": c0,
        "lane_consts": consts,
        "window_const_min": min(consts),
        "window_const_max": max(consts),
        **({} if info is None else info),
      })
  return rows


def compile_full_sink(m: int, n: int, k: int, u0: int, u1: int, loc: int, unr: int, boundary: str) -> UOp:
  postrange._WARMSTART_OPTS = {(frozenset({m, n}), k): _opts_for(u0, u1, loc, unr)}
  postrange._warmstart_stats.update({"match": 0, "apply": 0, "error": 0})
  a = Tensor.empty(m, k, dtype=dtypes.half)
  b = Tensor.empty(n, k, dtype=dtypes.half)
  ast = [u for u in (a @ b.transpose()).schedule_linear().toposort() if u.op is Ops.SINK][0]
  if boundary == "full":
    return full_rewrite_to_sink(ast, Device[Device.DEFAULT].renderer, optimize=True)
  sink = cg.graph_rewrite(ast, cg.pm_mops+cg.pm_syntactic_sugar+cg.pm_store_ranges, ctx=cg.itertools.count(1000),
                          name="audit early movement ops", bottom_up=True)
  sink = cg.graph_rewrite(sink, cg.pm_load_collapse, name="audit load collapse")
  sink = cg.graph_rewrite(sink, cg.pm_split_ranges+cg.pm_flatten_range, ctx={}, name="audit split ranges")
  sink = cg.graph_rewrite(sink, cg.sym+cg.pm_flatten_range, name="audit initial symbolic")
  sink = cg.graph_rewrite(sink, cg.pm_flatten_range+cg.pm_simplify_ranges, ctx={}, name="audit simplify ranges")
  return postrange.apply_opts(sink, Device[Device.DEFAULT].renderer)


def summarize(stores: list[dict[str, Any]], wmma: list[dict[str, Any]]) -> dict[str, Any]:
  store_frag_windows = {(r["buffer_id"], (r["const_bytes"] // 32) * 32, 32) for r in stores}
  store_frag_windows_nobuf = {((r["const_bytes"] // 32) * 32, 32) for r in stores}
  consumer_windows = set()
  consumer_windows_nobuf = set()
  for r in wmma:
    if not r.get("ok"): continue
    consumer_windows.add((r["buffer_id"], r["const_bytes"], 32))
    consumer_windows_nobuf.add((r["const_bytes"], 32))
  return {
    "store_count": len(stores),
    "wmma_operand_count": len(wmma),
    "wmma_operand_ok_count": sum(1 for r in wmma if r.get("ok")),
    "store_frag_window_count": len(store_frag_windows),
    "consumer_window_count": len(consumer_windows),
    "intersection_count": len(store_frag_windows & consumer_windows),
    "window_only_intersection_count": len(store_frag_windows_nobuf & consumer_windows_nobuf),
    "store_buffer_count": len({r["buffer_id"] for r in stores}),
    "consumer_buffer_count": len({r["buffer_id"] for r in wmma if r.get("ok")}),
    "store_tagged_count": sum(1 for r in stores if r.get("store_tag") not in ("None", "")),
    "index_tagged_count": sum(1 for r in stores if r.get("index_tag") not in ("None", "")),
  }


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--shape", default="2,2")
  ap.add_argument("--m", type=int, default=512)
  ap.add_argument("--n", type=int, default=5120)
  ap.add_argument("--k", type=int, default=5120)
  ap.add_argument("--loc", type=int, default=0)
  ap.add_argument("--unr", type=int, default=2)
  ap.add_argument("--json", action="store_true")
  ap.add_argument("--max-rows", type=int, default=12)
  ap.add_argument("--boundary", choices=("postrange", "full"), default="full")
  args = ap.parse_args()
  u0, u1 = (int(x) for x in args.shape.split(",", 1))
  with patched_env({**KMAJOR_LDS_ENV, "PREFILL_STAGE_PRESERVE_TAGS": "1"}):
    sink = compile_full_sink(args.m, args.n, args.k, u0, u1, args.loc, args.unr, args.boundary)
    stages, stores, consumers = stage_rows(sink), store_rows(sink), wmma_rows(sink)
  payload = {
    "shape": f"{u0}x{u1}", "m": args.m, "n": args.n, "k": args.k, "loc": args.loc, "unr": args.unr, "boundary": args.boundary,
    "summary": {**summarize(stores, consumers), "stage_count": len(stages)}, "stages": stages, "stores": stores, "wmma_operands": consumers,
  }
  if args.json:
    print(json.dumps(payload, indent=2))
  else:
    print(json.dumps(payload["summary"], indent=2))
    print("\nStages:")
    for r in stages[:args.max_rows]:
      print(json.dumps(r, sort_keys=True))
    print("\nStores:")
    for r in stores[:args.max_rows]:
      print(json.dumps({k: r.get(k) for k in ("store_id", "buffer_id", "const_bytes", "val_op", "val_dtype", "val_width", "store_tag", "index_tag", "buffer_tag")}, sort_keys=True))
    print("\nWMMA operands:")
    for r in consumers[:args.max_rows]:
      print(json.dumps({k: r.get(k) for k in ("wmma_i", "role", "ok", "reason", "carrier_op", "carrier_dtype", "carrier_arg", "carrier_src_ops", "buffer_id", "const_bytes", "window_const_min", "window_const_max", "carrier_tag", "buffer_tag")}, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
