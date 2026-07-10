#!/usr/bin/env python3
"""Decode W==D authority harness.

Measures real decode wall/token and dispatch-only replay at fixed contexts:

  W = real decode: TinyJit replay plus one .item() sync per token.
  D = dispatch-only: TinyJit replay with no per-token .item(), one final sync.

Report decode throughput from this harness, not from generate TTFT.
"""
from __future__ import annotations

import argparse, contextlib, io, json, os, pathlib, re, statistics, sys, time

from extra.qk.decode_harness import DEFAULT_MODEL, csv_ints, decode_run_profile

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LINE = re.compile(r"\*\*\*\s+\S+\s+\d+\s+(.+?)\s+arg\s+\d+\s+mem")


def main(argv: list[str] | None = None):
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", default=os.environ.get("QK_MODEL", DEFAULT_MODEL), help="GGUF path")
  ap.add_argument("--ckpts", default=os.environ.get("QK_CKPTS"), help="comma-separated decode checkpoint contexts")
  ap.add_argument("--max-context", type=int, default=int(os.environ.get("QK_MAX_CONTEXT", 4608)), help="model max_context")
  ap.add_argument("--nmeas", type=int, default=int(os.environ.get("QK_NMEAS", 40)), help="measurements per context")
  args = ap.parse_args(argv)
  profile = decode_run_profile(ckpts=csv_ints(args.ckpts) if args.ckpts else None,
                               max_context=args.max_context, nmeas=args.nmeas)
  model = args.model

  from tinygrad import Context, Device, GlobalCounters, Tensor, TinyJit, UOp
  from extra.llm.generate import load_model_and_tokenizer

  dev = Device[Device.DEFAULT]
  m, tok = load_model_and_tokenizer(model, profile.max_context, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True

  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 800)
  ids = (ids * (1 + profile.max_context // max(1, len(ids))))[:profile.max_context]
  v_sp = UOp.variable("start_pos", 0, profile.max_context - 1)
  temp = Tensor([0.0])

  rows = []
  for ck in profile.ckpts:
    use_flash = ck >= int(os.environ.get("FLASH_DECODE_THRESHOLD", "512"))
    for b in m.blk: b._use_flash, b._prefill_v2 = use_flash, False
    step = TinyJit(m.forward)
    tokid = int(ids[ck])

    out = Tensor([[tokid]], dtype="int32").contiguous()
    for i in range(8): out = step(out, v_sp.bind(ck + i), temp).realize()

    out = Tensor([[tokid]], dtype="int32").contiguous()
    W = []
    for i in range(profile.nmeas):
      t0 = time.perf_counter()
      out = step(out, v_sp.bind(ck + i), temp)
      _ = int(out.item())
      W.append(time.perf_counter() - t0)

    out = Tensor([[tokid]], dtype="int32").contiguous()
    dev.synchronize()
    t0 = time.perf_counter()
    for i in range(profile.nmeas):
      out = step(out, v_sp.bind(ck + i), temp)
    dev.synchronize()
    D = (time.perf_counter() - t0) / profile.nmeas

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), Context(DEBUG=2):
      GlobalCounters.reset()
      step(out, v_sp.bind(ck + profile.nmeas), temp).realize()
      gpu_dbg = GlobalCounters.time_sum_s
    progs = sum(1 for l in buf.getvalue().splitlines() if _LINE.search(_ANSI.sub("", l)))

    w_ms, d_ms = statistics.median(W) * 1e3, D * 1e3
    host = max(0.0, w_ms - d_ms)
    rows.append({"ctx": ck, "flash": use_flash, "wall_ms_W": round(w_ms, 3), "dispatch_ms_D": round(d_ms, 3),
                 "host_sync_residual_ms": round(host, 3), "host_sync_pct_of_wall": round(100 * host / w_ms, 1),
                 "tok_s_W": round(1000 / w_ms, 1), "tok_s_D_ceiling": round(1000 / d_ms, 1),
                 "programs_per_token": progs, "debug2_unbatched_gpu_ms": round(gpu_dbg * 1e3, 2),
                 "item_syncs_per_token_W": 1, "item_syncs_per_token_D": 0})
    print(f"ctx {ck:5}{'F' if use_flash else ' '}: W {w_ms:6.2f}ms ({1000/w_ms:.1f} tok/s) | "
          f"D {d_ms:6.2f}ms (ceiling {1000/d_ms:.1f}) | host-sync {host:.2f}ms "
          f"({rows[-1]['host_sync_pct_of_wall']}%) | progs {progs}", file=sys.__stderr__)

  med_host = statistics.median([r["host_sync_pct_of_wall"] for r in rows])
  out = {"model_id": pathlib.Path(model).stem, "hardware": "RX 7900 XTX / gfx1100",
         "ckpts": list(profile.ckpts), "nmeas": profile.nmeas, "max_context": profile.max_context,
         "method": "W=real decode (.item/token), D=dispatch-only (no per-token sync, 1 final sync); host_sync=W-D",
         "rows": rows, "median_host_sync_pct": round(med_host, 1),
         "verdict": ("RUNTIME IS A TARGET (host-sync >20% of wall): a low-sync path could approach the D ceiling"
                     if med_host >= 20 else "runtime NOT the main target (host-sync <10-20%): GPU-bound, stop" if med_host < 10
                     else "borderline (10-20%): marginal runtime target")}
  print(f"\nmedian host-sync {med_host}% of wall | {out['verdict']}", file=sys.__stderr__)
  art = pathlib.Path("bench/qk-decode-runtime-overhead/result.json")
  art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2))
  print(f"artifact: {art}", file=sys.__stderr__)
  print("@@DONE@@")


if __name__ == "__main__":
  main()
