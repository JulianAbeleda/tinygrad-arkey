#!/usr/bin/env python3
"""Shared capture for the 8B decode-gap exhaustion audits (FFN-activation / small-ops / attention-tail).

ONE GPU run dumps, per ctx, for a single decode token:
  - every decode kernel's rendered SOURCE + launch dims (name -> src/global/local), for mapping/identity audits
  - the per-kernel GPU TIMELINE (absolute start/end from ProfileGraphEvent signals), for critical-path / overlap analysis
  - per-kernel GPU-busy us/token (median-of-N), matching qk_decode_time_tax_audit's authority

Critical-path method: with the absolute [start,end] of every kernel we compute busy_union (merged intervals = real
GPU-active span) and, per bucket, SOLO time (only that bucket active) vs OVERLAPPED time. A bucket whose time is solo
is on the serial critical path; a bucket that is overlapped by other work is (partly) hidden -> a wall-clock win there
is bounded by its solo share, not its GPU-busy share. This is pure measurement (no kernel/default change).

  run: DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_audit_common.py
  -> bench/qk-decode-kernel-probe/latest.json   (consumed by the three phase audit tools)
"""
from __future__ import annotations
import collections, json, os, pathlib, re, statistics, sys
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
def _clean(s): return _ANSI.sub("", s)

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-kernel-probe"
CTXS = [512, 1024, 4096]; MAXC = 4608; NSAMP = 7

def capture(ctxs=CTXS, want_src=True):
  from extra.qk_harness_contract import DEFAULT_MODEL
  model = os.environ.get("QK_MODEL", DEFAULT_MODEL)
  from tinygrad import Tensor, UOp, TinyJit, Context, Device, getenv
  from tinygrad.device import Compiled
  from tinygrad.uop.ops import Ops
  from extra.llm_generate import load_model_and_tokenizer
  dev = Device["AMD"]
  m, tok = load_model_and_tokenizer(model, MAXC, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 800)
  ids = (ids * (1 + MAXC // max(1, len(ids))))[:MAXC]
  v = UOp.variable("start_pos", 0, MAXC - 1); temp = Tensor([0.0])
  route = {"Q4K_GEMV_WARP": getenv("Q4K_GEMV_WARP", 0), "Q4K_GEMV_WARP_DOWN": getenv("Q4K_GEMV_WARP_DOWN", 0),
           "FLASH_VARIANT": str(getenv("FLASH_VARIANT", "gqa_coop_vec"))}

  rows = []; sources = {}
  for ck in ctxs:
    for b in m.blk: b._use_flash, b._prefill_v2 = True, False
    sp = TinyJit(m.forward); o2 = Tensor([[int(ids[ck])]], dtype="int32").contiguous()
    with Context(PROFILE=1):
      for i in range(8): o2 = sp(o2, v.bind(ck + i), temp).realize()
      dev.synchronize(); dev._at_profile_finalize()
      # render sources from the captured graph (once, at first ctx is enough but cheap to redo)
      if want_src and not sources:
        def _dims(sz):
          if not sz: return None
          out = []
          for x in sz:
            try: out.append(int(x))
            except Exception: out.append(str(x))
          return out
        for u in sp.captured.linear.toposort():
          if u.op is not Ops.PROGRAM: continue
          pi = u.arg; nm = _clean(pi.name)
          if nm in sources: continue
          # AST fingerprint: op histogram + I/O dtypes -> kernel identity (silu=EXP, q8-quant=int8 STORE+MAX, norm=REDUCE, rope=SIN)
          hist = collections.Counter(); store_dt = set(); load_dt = set()
          try:
            for x in u.src[0].toposort():
              hist[x.op.name] += 1
              if x.op is Ops.STORE and len(x.src) > 1: store_dt.add(str(x.src[1].dtype))
              if x.op is Ops.LOAD: load_dt.add(str(x.dtype))
          except Exception: pass
          sources[nm] = {"global": _dims(pi.global_size), "local": _dims(pi.local_size),
                         "ins": list(pi.ins), "outs": list(pi.outs),
                         "op_hist": dict(sorted(hist.items(), key=lambda x: -x[1])),
                         "store_dtypes": sorted(store_dt), "load_dtypes": sorted(load_dt),
                         "has_exp": hist.get("EXP2",0)+hist.get("EXP",0) > 0,
                         "has_sin": hist.get("SIN",0) > 0, "has_reduce": hist.get("REDUCE",0)+hist.get("REDUCE_AXIS",0) > 0,
                         "has_int8_out": any("char" in d or "int8" in d for d in store_dt),
                         "has_sqrt": hist.get("SQRT",0) > 0, "has_recip": hist.get("RECIP",0) > 0}
      # timeline samples: capture per-kernel absolute [start,end] for one replay, plus median busy us over NSAMP
      tl = None; agg = collections.defaultdict(list)
      for r in range(NSAMP):
        base = len(Compiled.profile_events); sp(o2, v.bind(ck + 20 + r), temp).realize(); dev.synchronize(); dev._at_profile_finalize()
        per = collections.defaultdict(float); intervals = []
        for e in Compiled.profile_events[base:]:
          if type(e).__name__ != "ProfileGraphEvent": continue
          sigs = [float(s) for s in e.sigs]
          for ent in e.ents:
            st, en = sigs[ent.st_id], sigs[ent.en_id]
            per[_clean(str(ent.name))] += en - st
            intervals.append((st, en, _clean(str(ent.name))))
        for k, vv in per.items(): agg[k].append(vv)
        if r == NSAMP - 1: tl = intervals      # keep the last replay's timeline for overlap analysis
    per_kernel = {k: round(statistics.median(vs), 1) for k, vs in agg.items()}   # us/token (median)
    rows.append({"ctx": ck, "per_kernel_us": dict(sorted(per_kernel.items(), key=lambda x: -x[1])),
                 "timeline": [[round(s,3), round(e,3), n] for (s,e,n) in tl] if tl else []})
    print(f"ctx {ck}: {len(per_kernel)} kernels, {len(tl)} timeline entries, busy_sum={sum(per_kernel.values())/1e3:.2f}ms", file=sys.stderr)
  return {"model": os.path.basename(model), "route_flags": route, "sources": sources, "rows": rows,
          "nsamp": NSAMP, "hardware": "RX 7900 XTX / gfx1100"}

def main():
  d = capture()
  OUT.mkdir(parents=True, exist_ok=True)
  d["date"] = "2026-06-22"; d["phase"] = "DECODE_KERNEL_PROBE"; d["default_behavior_changed"] = False
  (OUT / "latest.json").write_text(json.dumps(d, indent=2))
  print(f"artifact: {OUT/'latest.json'} (sources={len(d['sources'])})", file=sys.stderr)

if __name__ == "__main__":
  main()
