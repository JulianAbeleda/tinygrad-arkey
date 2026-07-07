"""Field-based stream scanner for the generated 4x4 WMMA fault.

This is intentionally independent of regalloc/codegen mutation paths. It captures
the generated I0 stream, builds a fixed-register faultprobe 4x4 comparison
stream, and reports register-span patterns from Inst fields/raw bytes. Disasm is
not parsed; rendered instruction strings are only used for orientation.
"""
from __future__ import annotations

import argparse, os, sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

os.environ.setdefault("ALLOW_DEVICE_USAGE", "1")
sys.path.insert(0, os.getcwd())

from tinygrad.helpers import Target, getenv
from tinygrad.renderer.amd.dsl import Reg
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.uop.ops import Ops, UOp

from extra.qk.prefill.gen4x4_i0_harness import generated_4x4_insts
from extra.qk.prefill.wmma_faultprobe import build_variant, remu_validate


RegKey = tuple[str, int]
Span = frozenset[RegKey]

DEST_FIELDS = {"vdst", "sdst", "sdata", "vdsty"}
LOAD_ADDR_FIELDS = ("addr", "vaddr")
LOAD_DEST_FIELDS = ("vdst", "vdata", "sdata")
WMMA_FIELDS = ("src0", "src1", "src2", "vdst")


@dataclass(frozen=True)
class Row:
  idx: int
  inst: object
  name: str
  raw: bytes
  fields: dict[str, Span]
  defs: Span
  uses: Span


def _name(inst) -> str:
  return getattr(inst, "op_name", type(inst).__name__)


def _unwrap(insts: Iterable) -> list:
  return [x.arg if isinstance(x, UOp) else x for x in insts]


def _finalize(insts: list) -> list:
  ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
  uops = [x if isinstance(x, UOp) else UOp(Ops.INS, arg=x) for x in insts]
  if getenv("AMD_ISA_SCHED", 1): uops = ren._schedule(uops)
  return _unwrap(ren._resolve_labels(ren._insert_waitcnt(uops)))


def generated_stream(final: bool) -> list:
  insts = generated_4x4_insts()
  return _finalize(insts) if final else _unwrap(insts)


def hand4x4_b128_stream(final: bool) -> list:
  # Single-buffer 4x4: low A/B b128 fragments, high contiguous C/D block.
  # VA is fixed at v100..v107 inside build_variant, so C starts at v108.
  insts = build_variant(64, 64, 64, 4, 4,
                        a_bases=[10, 18, 26, 34],
                        b_bases=[42, 50, 58, 66],
                        a_valu=False, b_valu=False, acc_base=108)
  return _finalize(insts) if final else insts


def _reg_span(reg: object, n: int) -> Span:
  if not isinstance(reg, Reg): return frozenset()
  if 256 <= reg.offset < 512: return frozenset(("v", reg.offset - 256 + i) for i in range(n))
  if 0 <= reg.offset < 128: return frozenset(("s", reg.offset + i) for i in range(n))
  return frozenset()


def rows(insts: list) -> list[Row]:
  out: list[Row] = []
  for idx, inst in enumerate(insts):
    if isinstance(inst, tuple): continue
    fields: dict[str, Span] = {}
    defs: set[RegKey] = set()
    uses: set[RegKey] = set()
    is_store = "STORE" in _name(inst)
    for fname, _field in getattr(inst, "_fields", ()):
      span = _reg_span(getattr(inst, fname, None), getattr(inst, "op_regs", {}).get(fname, 1))
      if not span: continue
      fields[fname] = span
      if fname in DEST_FIELDS and not is_store: defs.update(span)
      else: uses.update(span)
    out.append(Row(idx, inst, _name(inst), inst.to_bytes(), fields, frozenset(defs), frozenset(uses)))
  return out


def _fmt_span(span: Span, limit: int = 8) -> str:
  if not span: return "-"
  regs = sorted(span, key=lambda x: (x[0], x[1]))
  runs: list[str] = []
  i = 0
  while i < len(regs):
    k, a = regs[i]
    j = i
    while j + 1 < len(regs) and regs[j + 1][0] == k and regs[j + 1][1] == regs[j][1] + 1: j += 1
    runs.append(f"{k}{a}" if i == j else f"{k}{a}:{regs[j][1]}")
    i = j + 1
  return ",".join(runs[:limit]) + ("..." if len(runs) > limit else "")


def _raw(row: Row) -> str:
  return row.raw.hex()


def _interesting(name: str) -> bool:
  l = name.lower()
  return "global_load" in l or "wmma" in l or "pack" in l


def l3_load_addr_dest(rows_: list[Row], near: int) -> tuple[list[dict], list[dict]]:
  direct, nearby = [], []
  loads = [r for r in rows_ if "LOAD" in r.name]
  for r in loads:
    addr = frozenset().union(*(r.fields.get(f, frozenset()) for f in LOAD_ADDR_FIELDS))
    dst = frozenset().union(*(r.fields.get(f, frozenset()) for f in LOAD_DEST_FIELDS))
    if ov := addr & dst:
      direct.append({"idx": r.idx, "name": r.name, "overlap": ov, "addr": addr, "dst": dst, "row": r})
  for a in loads:
    aaddr = frozenset().union(*(a.fields.get(f, frozenset()) for f in LOAD_ADDR_FIELDS))
    adst = frozenset().union(*(a.fields.get(f, frozenset()) for f in LOAD_DEST_FIELDS))
    for b in loads:
      if a.idx == b.idx or abs(a.idx - b.idx) > near: continue
      baddr = frozenset().union(*(b.fields.get(f, frozenset()) for f in LOAD_ADDR_FIELDS))
      bdst = frozenset().union(*(b.fields.get(f, frozenset()) for f in LOAD_DEST_FIELDS))
      if ov := (adst & baddr) | (aaddr & bdst):
        nearby.append({"a": a, "b": b, "distance": abs(a.idx - b.idx), "overlap": ov})
  nearby.sort(key=lambda x: (x["distance"], x["a"].idx, x["b"].idx))
  return direct, nearby


def l1_recent_writes(rows_: list[Row], lookback: int) -> tuple[list[dict], Counter]:
  findings: list[dict] = []
  summary: Counter = Counter()
  for pos, r in enumerate(rows_):
    if "WMMA" not in r.name: continue
    for role in WMMA_FIELDS:
      span = r.fields.get(role, frozenset())
      if not span: continue
      for prev in reversed(rows_[max(0, pos - lookback):pos]):
        if "WMMA" in prev.name or not prev.defs: continue
        if ov := span & prev.defs:
          dist = r.idx - prev.idx
          summary[(role, prev.name, dist)] += 1
          findings.append({"wmma": r, "writer": prev, "role": role, "distance": dist, "overlap": ov})
          break
  findings.sort(key=lambda x: (x["distance"], x["wmma"].idx, x["role"]))
  return findings, summary


def _role_overlap(a: Row, b: Row) -> str:
  parts = []
  for an, av in a.fields.items():
    for bn, bv in b.fields.items():
      if av & bv: parts.append(f"{an}->{bn}:{_fmt_span(av & bv)}")
  return ";".join(parts[:4]) or "-"


def l4_pairs(rows_: list[Row], radius: int) -> Counter:
  hot: set[int] = set()
  for pos, r in enumerate(rows_):
    if _interesting(r.name):
      hot.update(range(max(0, pos - radius), min(len(rows_), pos + radius + 1)))
  c: Counter = Counter()
  for pos in sorted(hot):
    if pos + 1 not in hot or pos + 1 >= len(rows_): continue
    a, b = rows_[pos], rows_[pos + 1]
    c[(a.name, b.name, _role_overlap(a, b))] += 1
  return c


def l2_backedge(insts: list, tail: int = 64, head: int = 64, limit: int = 8) -> None:
  markers = [(i, u.arg if isinstance(u, UOp) else u) for i, u in enumerate(insts)
             if (isinstance(u, UOp) and isinstance(u.arg, tuple)) or isinstance(u, tuple)]
  tops = {arg[1]: i for i, arg in markers if arg[0] == "label" and isinstance(arg[1], tuple) and arg[1][0] == "top"}
  backs = [(i, arg[2]) for i, arg in markers if arg[0] == "branch" and arg[1] == "s_branch" and arg[2] in tops]
  print("\nL2 backedge physical reuse")
  if not backs:
    print("  no unresolved loop backedge markers in this stream")
    return
  rs = rows(insts)
  by_idx = {r.idx: r for r in rs}
  for bidx, target in backs[:1]:
    tidx = tops[target]
    exit_br = next((i for i, arg in markers if tidx < i < bidx and arg[0] == "branch" and arg[1] == "s_cbranch_scc0"), tidx)
    hrows = [by_idx[i] for i in range(exit_br + 1, min(bidx, exit_br + 1 + head)) if i in by_idx]
    trows = [by_idx[i] for i in range(max(exit_br + 1, bidx - tail), bidx) if i in by_idx]
    tail_defs: dict[RegKey, Row] = {}
    for r in trows:
      for reg in r.defs: tail_defs[reg] = r
    hits = []
    for r in hrows:
      for reg in (r.uses | r.defs) & tail_defs.keys():
        if reg[0] == "s" and reg[1] >= 40: continue
        hits.append((reg, tail_defs[reg], r, "def" if reg in r.defs else "use"))
    print(f"  loop target={target} top={tidx} body_start={exit_br+1} backedge={bidx} tail={tail} head={head} hits={len(hits)}")
    for reg, tr, hr, role in hits[:limit]:
      print(f"  {_fmt_span(frozenset((reg,)))} tail_def@{tr.idx:04d} -> head_{role}@{hr.idx:04d}")
      print(f"        tail {tr.inst}")
      print(f"        head {hr.inst}")


def wmma_producer_spans(rows_: list[Row], limit: int) -> None:
  last_def: dict[RegKey, Row] = {}
  findings: list[dict] = []
  for r in rows_:
    if "WMMA" in r.name:
      for role in ("src0", "src1"):
        span = r.fields.get(role, frozenset())
        if not span: continue
        prods = [last_def[x] for x in sorted(span, key=lambda y: (y[0], y[1])) if x in last_def]
        if not prods: continue
        idxs = [p.idx for p in prods]
        names = frozenset(p.name for p in prods)
        findings.append({"wmma": r, "role": role, "span": span, "prods": prods, "names": names,
                         "producer_span": max(idxs) - min(idxs), "oldest_dist": r.idx - min(idxs),
                         "newest_dist": r.idx - max(idxs)})
    for reg in r.defs: last_def[reg] = r
  findings.sort(key=lambda x: (x["names"] != frozenset(("V_PACK_B32_F16",)), -x["oldest_dist"], -x["producer_span"]))
  print("\nWMMA producer-span ages")
  if not findings:
    print("  none")
    return
  for x in findings[:limit]:
    print(f"  WMMA@{x['wmma'].idx:04d} role={x['role']} operand={_fmt_span(x['span'])} "
          f"names={','.join(sorted(x['names']))} producers={len(x['prods'])} "
          f"span={x['producer_span']} oldest={x['oldest_dist']} newest={x['newest_dist']}")
    for p in x["prods"][:2]:
      print(f"        prod@{p.idx:04d} {p.inst}")
    if len(x["prods"]) > 2: print("        ...")


def _print_examples(title: str, examples: list, limit: int, kind: str) -> None:
  print(f"\n{title}")
  if not examples:
    print("  none")
    return
  for x in examples[:limit]:
    if kind == "l3_direct":
      r = x["row"]
      print(f"  @{r.idx:04d} {r.name} overlap={_fmt_span(x['overlap'])} addr={_fmt_span(x['addr'])} dst={_fmt_span(x['dst'])} raw={_raw(r)}")
      print(f"        {r.inst}")
    elif kind == "l3_near":
      a, b = x["a"], x["b"]
      print(f"  @{a.idx:04d}<->{b.idx:04d} d={x['distance']:2d} overlap={_fmt_span(x['overlap'])}")
      print(f"        A {a.inst}")
      print(f"        B {b.inst}")
    elif kind == "l1":
      w, m = x["writer"], x["wmma"]
      print(f"  WMMA@{m.idx:04d} role={x['role']} writer@{w.idx:04d} d={x['distance']:2d} overlap={_fmt_span(x['overlap'])} raw={_raw(w)}")
      print(f"        writer {w.inst}")
      print(f"        wmma   {m.inst}")


def scan_one(label: str, insts: list, near: int, lookback: int, radius: int, limit: int, producer_spans: bool) -> dict:
  rs = rows(insts)
  direct, nearby = l3_load_addr_dest(rs, near)
  l1, l1sum = l1_recent_writes(rs, lookback)
  l4 = l4_pairs(rs, radius)
  print(f"\n==== {label} ====")
  print(f"insts={len(rs)} bytes={sum(len(r.raw) for r in rs)} "
        f"wmma={sum('WMMA' in r.name for r in rs)} pack={sum('PACK' in r.name for r in rs)} "
        f"global_load={sum('GLOBAL_LOAD' in r.name for r in rs)}")
  print(f"L3 direct load addr/dest overlaps: {len(direct)}")
  print(f"L3 nearby load addr/dest reuses within {near}: {len(nearby)}")
  print(f"L1 recent non-WMMA writes into WMMA operand spans within {lookback}: {len(l1)}")
  _print_examples("L3 direct examples", direct, limit, "l3_direct")
  _print_examples("L3 nearby examples", nearby, limit, "l3_near")
  _print_examples("L1 recent-write examples", l1, limit, "l1")
  print("\nL1 top role/writer/distance")
  for (role, name, dist), cnt in l1sum.most_common(limit):
    print(f"  {cnt:4d} role={role:4s} writer={name} distance={dist}")
  print("\nL4 top adjacent pairs near load/pack/WMMA")
  for (a, b, rel), cnt in l4.most_common(limit):
    print(f"  {cnt:4d} {a} -> {b} overlap={rel}")
  l2_backedge(insts, limit=limit)
  if producer_spans: wmma_producer_spans(rs, limit)
  return {"rows": rs, "l3_direct": direct, "l3_near": nearby, "l1": l1, "l4": l4}


def compare_l4(gen: Counter, hand: Counter, limit: int) -> None:
  only_gen = Counter({k: v for k, v in gen.items() if k not in hand})
  only_hand = Counter({k: v for k, v in hand.items() if k not in gen})
  print("\n==== L4 comparator ====")
  print("Generated-only adjacent summaries")
  for (a, b, rel), cnt in only_gen.most_common(limit):
    print(f"  {cnt:4d} {a} -> {b} overlap={rel}")
  print("Hand-only adjacent summaries")
  for (a, b, rel), cnt in only_hand.most_common(limit):
    print(f"  {cnt:4d} {a} -> {b} overlap={rel}")


def validate_hand_stream() -> None:
  r = remu_validate(hand4x4_b128_stream(final=False), 64, 64, 64)
  print(f"hand4x4_b128 remu: bytes={r['bytes']} rc={r['rc']} nan={r['nanfrac']:.3f} "
        f"rmse={r['rmse']:.5f} bitexact={r['bitexact']}")
  if not r["bitexact"]: raise SystemExit("hand4x4_b128 comparison stream failed remu validation")


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--pre-final", action="store_true", help="scan pre-scheduler/pre-waitcnt streams instead of final streams")
  p.add_argument("--validate-hand", action="store_true", help="remu-validate the constructed faultprobe 4x4 stream")
  p.add_argument("--near", type=int, default=8, help="L3 nearby load addr/dest reuse window")
  p.add_argument("--lookback", type=int, default=32, help="L1 WMMA recent-writer lookback window")
  p.add_argument("--radius", type=int, default=3, help="L4 adjacency hot-window radius around load/pack/WMMA")
  p.add_argument("--limit", type=int, default=8, help="examples/summaries to print per section")
  p.add_argument("--wmma-producer-spans", action="store_true", help="report producer age/span for each WMMA src0/src1 operand")
  args = p.parse_args()

  final = not args.pre_final
  if args.validate_hand: validate_hand_stream()
  gen = scan_one("generated_i0_" + ("final" if final else "prefinal"), generated_stream(final),
                 args.near, args.lookback, args.radius, args.limit, args.wmma_producer_spans)
  hand = scan_one("faultprobe_hand4x4_b128_" + ("final" if final else "prefinal"), hand4x4_b128_stream(final),
                  args.near, args.lookback, args.radius, args.limit, args.wmma_producer_spans)
  compare_l4(gen["l4"], hand["l4"], args.limit)


if __name__ == "__main__":
  main()
