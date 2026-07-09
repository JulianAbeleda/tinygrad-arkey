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
from tinygrad.schedule import rangeify
from tinygrad.uop.ops import AxisType, Ops, UOp

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


def tag_fields(tag: Any) -> dict[str, Any]:
  if not isinstance(tag, tuple): return {}
  out: dict[str, Any] = {"kind": tag[0] if tag else None}
  for item in tag[1:]:
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
      out[item[0]] = item[1]
  return out


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
    fields = tag_fields(stg.tag)
    rows.append({
      "stage_id": id(stg),
      "stage_dtype": str(stg.dtype),
      "stage_arg": repr(stg.arg),
      "stage_tag": repr(stg.tag),
      "stage_tag_fields": {k: repr(v) for k, v in fields.items()},
      "role": fields.get("role"),
      "lds_buffer_id": fields.get("lds_buffer_id"),
      "nbuf": fields.get("nbuf"),
      "tile_count": fields.get("tile_count"),
      "tile_elems": fields.get("tile_elems"),
      "stage_src_ops": [s.op.name for s in stg.src],
      "stage_src_dtypes": [str(s.dtype) for s in stg.src],
      "stage_ranges": [repr(r.arg) for r in sorted(stg.ranges, key=lambda r: repr(r.arg))],
      "has_reduce_range": any(r.arg[-1] is AxisType.REDUCE for r in stg.ranges),
      "has_global_range": any(r.arg[-1] is AxisType.GLOBAL for r in stg.ranges),
      "has_unroll_range": any(r.arg[-1] is AxisType.UNROLL for r in stg.ranges),
    })
  return rows


def wmma_rows(sink: UOp) -> list[dict[str, Any]]:
  from tinygrad.renderer.isa.amd import _wmma_elems, _wmma_half_addr
  rows = []
  for wi, w in enumerate([u for u in sink.toposort() if u.op is Ops.WMMA]):
    for role, carrier in (("A", w.src[0]), ("B", w.src[1])):
      carrier_fields = tag_fields(carrier.tag)
      base_row = {
        "wmma_i": wi, "wmma_id": id(w), "role": role, "carrier_id": id(carrier),
        "carrier_op": carrier.op.name, "carrier_dtype": str(carrier.dtype), "carrier_arg": repr(carrier.arg),
        "carrier_tag": repr(carrier.tag), "carrier_tag_fields": {k: repr(v) for k, v in carrier_fields.items()},
        "carrier_role": carrier_fields.get("role"), "carrier_nbuf": carrier_fields.get("nbuf"),
        "carrier_lds_buffer_id": carrier_fields.get("lds_buffer_id"),
        "carrier_src_ops": [s.op.name for s in carrier.src],
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


def _range_rows(u: UOp) -> list[dict[str, Any]]:
  return [{"arg": repr(r.arg), "size": int(r.vmax + 1), "axis_type": str(r.arg[-1])}
          for r in sorted(u.ranges, key=lambda r: repr(r.arg))]


def _src_range_args(u: UOp) -> list[str]:
  return [repr(s.arg) for s in u.src if s.op is Ops.RANGE]


def generic_b_stage_contract(sink: UOp) -> dict[str, Any]:
  stages = []
  for stg in sink.toposort():
    if stg.op is not Ops.STAGE: continue
    fields = tag_fields(stg.tag)
    if fields.get("role") != "B": continue
    stages.append({
      "stage_id": id(stg),
      "stage_dtype": str(stg.dtype),
      "stage_shape": [str(x) for x in stg.shape],
      "stage_tag": repr(stg.tag),
      "stage_src_ops": [s.op.name for s in stg.src],
      "stage_src_dtypes": [str(s.dtype) for s in stg.src],
      "stage_src_shapes": [[str(x) for x in s.shape] for s in stg.src],
      "stage_index_range_args": _src_range_args(stg),
      "stage_ranges": _range_rows(stg),
      "contract_arg": repr(stg.src[0].arg) if stg.src and stg.src[0].op is Ops.CONTRACT else None,
      "contract_src_op": stg.src[0].src[0].op.name if stg.src and stg.src[0].op is Ops.CONTRACT and stg.src[0].src else None,
    })

  consumers = []
  for wi, w in enumerate([u for u in sink.toposort() if u.op is Ops.WMMA]):
    b = w.src[1]
    fields = tag_fields(b.tag)
    consumers.append({
      "wmma_i": wi,
      "carrier_id": id(b),
      "carrier_op": b.op.name,
      "carrier_dtype": str(b.dtype),
      "carrier_shape": [str(x) for x in b.shape],
      "carrier_tag": repr(b.tag),
      "carrier_role": fields.get("role"),
      "carrier_src_ops": [s.op.name for s in b.src],
      "carrier_index_range_args": _src_range_args(b),
      "carrier_ranges": _range_rows(b),
      "direct_stage_role": tag_fields(b.src[0].tag).get("role") if b.op is Ops.INDEX and b.src and b.src[0].op is Ops.STAGE else None,
      "direct_stage_id": id(b.src[0]) if b.op is Ops.INDEX and b.src and b.src[0].op is Ops.STAGE else None,
    })

  direct_consumers = [r for r in consumers if r.get("direct_stage_role") == "B"]
  return {
    "stage_count": len(stages),
    "consumer_count": len(consumers),
    "direct_b_stage_consumer_count": len(direct_consumers),
    "ok": len(stages) == 1 and len(direct_consumers) >= 1,
    "expected_owned_stage_shape": "vector payload dtypes.half.vec(16) staged over WARP x LOCAL, not scalar lane16 packing",
    "stages": stages,
    "consumers": consumers,
  }


def compile_full_sink(m: int, n: int, k: int, u0: int, u1: int, loc: int, unr: int, boundary: str) -> UOp:
  rangeify.prefill_dbuf_clear_rotated_stage_lowering_audit()
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


def stage_ownership_summary(stages: list[dict[str, Any]], wmma: list[dict[str, Any]]) -> dict[str, Any]:
  tagged = [r for r in stages if r.get("role") in ("A", "B") and r.get("nbuf") is not None]
  roles = sorted({r.get("role") for r in tagged})
  by_role = {role: [r for r in tagged if r.get("role") == role] for role in roles}
  consumer_tagged = []
  for r in wmma:
    if r.get("carrier_role") in ("A", "B") and r.get("carrier_nbuf") is not None:
      consumer_tagged.append(r)
  return {
    "stage_tagged_count": len(tagged),
    "stage_roles": roles,
    "stage_count_by_role": {str(k): len(v) for k, v in by_role.items()},
    "stage_nbufs": sorted({r.get("nbuf") for r in tagged}),
    "stage_has_reduce_range_count": sum(1 for r in tagged if r.get("has_reduce_range")),
    "stage_has_global_range_count": sum(1 for r in tagged if r.get("has_global_range")),
    "stage_has_unroll_range_count": sum(1 for r in tagged if r.get("has_unroll_range")),
    "wmma_tagged_operand_count": len(consumer_tagged),
    "wmma_tagged_roles": sorted({r.get("carrier_role") for r in consumer_tagged}),
    "pre_lowering_ownership_ready": (
      set(roles) >= {"A", "B"} and all(len(by_role.get(role, [])) == 1 for role in ("A", "B")) and
      set(r.get("nbuf") for r in tagged) == {2} and all(r.get("has_reduce_range") for r in tagged)
    ),
    "full_lowering_ownership_lost": len(stages) == 0 and not consumer_tagged,
    "next_required_object": "RotatedStageOwner(role, lds_buffer_id, nbuf, reduce_epoch, dbuf_slot, producer_phase, consumer_phase)",
  }


def owner_records(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
  records = []
  for r in stages:
    role, nbuf, lds_buffer_id = r.get("role"), r.get("nbuf"), r.get("lds_buffer_id")
    if role not in ("A", "B") or nbuf is None or lds_buffer_id is None: continue
    reduce_ranges = [x for x in r.get("stage_ranges", []) if "AxisType.REDUCE" in x]
    global_ranges = [x for x in r.get("stage_ranges", []) if "AxisType.GLOBAL" in x]
    unroll_ranges = [x for x in r.get("stage_ranges", []) if "AxisType.UNROLL" in x]
    records.append({
      "role": role,
      "lds_buffer_id": lds_buffer_id,
      "nbuf": nbuf,
      "reduce_epoch": reduce_ranges[0] if reduce_ranges else None,
      "dbuf_slot_expr": None if not reduce_ranges else f"({reduce_ranges[0]}) % {nbuf}",
      "tile_count": r.get("tile_count"),
      "tile_elems": r.get("tile_elems"),
      "producer_phase": "prologue_or_body",
      "consumer_phase": "compute",
      "global_ranges": global_ranges,
      "unroll_ranges": unroll_ranges,
      "stage_id": r.get("stage_id"),
    })
  return records


def rotated_lifecycle_plan(records: list[dict[str, Any]]) -> dict[str, Any]:
  roles = sorted({r["role"] for r in records})
  nbufs = sorted({r["nbuf"] for r in records if r.get("nbuf") is not None})
  if roles != ["A", "B"] or nbufs != [2]:
    return {"ok": False, "reason": "requires exactly A+B records with nbuf=2", "roles": roles, "nbufs": nbufs}
  def ops(kind: str, slot: int, epoch: str) -> list[dict[str, Any]]:
    return [{"op": kind, "role": r["role"], "slot": slot, "epoch": epoch, "owner": (r["role"], r["lds_buffer_id"], r["nbuf"])}
            for r in records]
  prologue = ops("produce", 0, "k0") + [{"op": "barrier"}]
  body = (
    ops("consume", 0, "k") +
    ops("produce", 1, "k+1") +
    [{"op": "barrier"}] +
    ops("consume", 1, "k+1") +
    ops("produce", 0, "k+2") +
    [{"op": "barrier"}]
  )
  tail = ops("consume", 1, "last")
  consumers = [x for x in prologue + body + tail if x.get("op") == "consume"]
  producers = [x for x in prologue + body + tail if x.get("op") == "produce"]
  return {
    "ok": True,
    "source": "audit_only_hand_lds2_style_rotation",
    "invariant": "each consume(role,slot,epoch) must be paired with exactly one prior produce(role,slot,epoch-family) after a barrier",
    "prologue": prologue,
    "body": body,
    "tail": tail,
    "producer_count": len(producers),
    "consumer_count": len(consumers),
    "late_suppression_allowed": False,
  }


def p4_readiness(summary: dict[str, Any], plan: dict[str, Any], boundary: str) -> dict[str, Any]:
  if boundary != "postrange":
    return {
      "ready": False,
      "blocked_at": "P4",
      "reason": "destructive rotated construction must start at postrange; full lowering has already lost owner identity",
    }
  if not summary.get("pre_lowering_ownership_ready"):
    return {"ready": False, "blocked_at": "P4", "reason": "missing A/B DBUF owner records"}
  if not plan.get("ok"):
    return {"ready": False, "blocked_at": "P4", "reason": "rotated lifecycle planner failed"}
  return {
    "ready": False,
    "blocked_at": "P4",
    "reason": "no implemented owner-aware STAGE lowering hook; current lowering materializes generic local stores before renderer",
    "required_hook": "lower Ops.STAGE with RotatedStageOwner so legacy duplicate producers are never emitted",
    "forbidden_fallback": "PREFILL_WMMA_KMAJOR_STAGE_KEY_SUPPRESS late deletion",
  }


def lowering_hook_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
  roles = sorted({r.get("role") for r in rows if r.get("role") in ("A", "B")})
  by_role = {role: [r for r in rows if r.get("role") == role] for role in roles}
  return {
    "lowering_owner_count": len(rows),
    "lowering_roles": roles,
    "lowering_count_by_role": {str(k): len(v) for k, v in by_role.items()},
    "lowering_nbufs": sorted({r.get("nbuf") for r in rows if r.get("nbuf") is not None}),
    "lowering_has_reduce_range_count": sum(1 for r in rows if r.get("has_reduce_range")),
    "lowering_hook_owner_ready": set(roles) >= {"A", "B"} and set(r.get("nbuf") for r in rows) == {2},
  }


def summarize(stages: list[dict[str, Any]], stores: list[dict[str, Any]], wmma: list[dict[str, Any]]) -> dict[str, Any]:
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
    **stage_ownership_summary(stages, wmma),
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
    lowering_rows = rangeify.prefill_dbuf_rotated_stage_lowering_audit_rows()
  owners = owner_records(stages)
  summary = {**summarize(stages, stores, consumers), **lowering_hook_summary(lowering_rows), "stage_count": len(stages)}
  plan = rotated_lifecycle_plan(owners)
  b_contract = generic_b_stage_contract(sink)
  payload = {
    "shape": f"{u0}x{u1}", "m": args.m, "n": args.n, "k": args.k, "loc": args.loc, "unr": args.unr, "boundary": args.boundary,
    "summary": summary, "owner_records": owners,
    "lowering_hook_owner_records": lowering_rows,
    "generic_b_stage_contract": b_contract,
    "rotated_lifecycle_plan": plan,
    "p4_readiness": p4_readiness(summary, plan, args.boundary),
    "stages": stages, "stores": stores, "wmma_operands": consumers,
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
