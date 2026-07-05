#!/usr/bin/env python3
"""Arc 4 Phase 0: decode host/runtime overhead accounting. Cleanly isolates per-token host-sync overhead without
the DEBUG=2-unbatch inflation. Three warm measurements per ctx:
  W  = real decode wall/token: jit replay + .item() readback every token (1 host sync/token, the real path).
  D  = dispatch-only wall/token: same jit replayed back-to-back feeding a fixed token, NO per-token .item();
       one final Device.synchronize(). Host dispatch overlaps the GPU -> ~max(GPU, host-dispatch) rate.
  host_sync_residual = W - D  = the cost of the per-token .item() readback sync (the host wait).
If D << W, the per-token sync dominates and a low-sync path would approach D. If D ~= W, sync isn't the lever.
Also: programs/token (one DEBUG=2 eager step) and the DEBUG=2 unbatched GPU sum (proxy only). ctx 128/512/1024
(+4096 flash). Output greedy-correctness preserved (W path is the real decode). No defaults changed.

Run: DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk/decode_runtime_overhead.py
"""
from __future__ import annotations

import argparse, io, json, os, pathlib, re, statistics, sys, time, contextlib
from extra.qk.decode_harness import csv_ints, decode_run_profile
from extra.qk.harness_contract import DEFAULT_MODEL  # tinygrad-free; import before tinygrad to preserve env-ordering

_ANSI = re.compile(r"\x1b\[[0-9;]*m"); _LINE = re.compile(r"\*\*\*\s+\S+\s+\d+\s+(.+?)\s+arg\s+\d+\s+mem")

def main(argv:list[str]|None=None):
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", default=os.environ.get("QK_MODEL", DEFAULT_MODEL), help="GGUF path")
  ap.add_argument("--ckpts", default=os.environ.get("QK_CKPTS"), help="comma-separated decode checkpoint contexts")
  ap.add_argument("--max-context", type=int, default=int(os.environ.get("QK_MAX_CONTEXT", 4608)), help="model max_context")
  ap.add_argument("--nmeas", type=int, default=int(os.environ.get("QK_NMEAS", 40)), help="measurements per context")
  args = ap.parse_args(argv)
  profile = decode_run_profile(ckpts=csv_ints(args.ckpts) if args.ckpts else None,
                               max_context=args.max_context, nmeas=args.nmeas)
  model = args.model
  from tinygrad import Tensor, UOp, TinyJit, Context, GlobalCounters, Device
  from extra.llm.generate import load_model_and_tokenizer
  dev = Device[Device.DEFAULT]
  m, tok = load_model_and_tokenizer(model, profile.max_context, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 800)
  ids = (ids * (1 + profile.max_context // max(1, len(ids))))[:profile.max_context]
  v_sp = UOp.variable("start_pos", 0, profile.max_context - 1); temp = Tensor([0.0])

  rows = []
  for ck in profile.ckpts:
    use_flash = ck >= int(os.environ.get("FLASH_DECODE_THRESHOLD", "512"))   # match the shipped real-generate flash threshold
    for b in m.blk: b._use_flash, b._prefill_v2 = use_flash, False
    step = TinyJit(m.forward)
    tokid = int(ids[ck])
    # warm (compile + clock ramp). Mirror model.generate: feed the device output token back (out=step(out,...)),
    # NO per-step host Tensor creation (that was the harness contamination that halved the rate).
    out = Tensor([[tokid]], dtype="int32").contiguous()
    for i in range(8): out = step(out, v_sp.bind(ck + i), temp).realize()
    # W: real decode -- feed out->out, .item() readback per token (the actual sync path, as generate does)
    out = Tensor([[tokid]], dtype="int32").contiguous(); W = []
    for i in range(profile.nmeas):
      t0 = time.perf_counter(); out = step(out, v_sp.bind(ck + i), temp); _ = int(out.item())
      W.append(time.perf_counter() - t0)
    # D: dispatch-only -- feed out->out, NO per-token .item, one final synchronize
    out = Tensor([[tokid]], dtype="int32").contiguous(); dev.synchronize(); t0 = time.perf_counter()
    for i in range(profile.nmeas): out = step(out, v_sp.bind(ck + i), temp)
    dev.synchronize(); D = (time.perf_counter() - t0) / profile.nmeas
    # GPU proxy (DEBUG=2 unbatched -- inflated) + program count
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), Context(DEBUG=2):
      GlobalCounters.reset(); step(out, v_sp.bind(ck + profile.nmeas), temp).realize()
      gpu_dbg = GlobalCounters.time_sum_s
    progs = sum(1 for l in buf.getvalue().splitlines() if _LINE.search(_ANSI.sub("", l)))
    w_ms, d_ms = statistics.median(W) * 1e3, D * 1e3
    host = max(0.0, w_ms - d_ms)
    rows.append({"ctx": ck, "flash": use_flash, "wall_ms_W": round(w_ms, 3), "dispatch_ms_D": round(d_ms, 3),
                 "host_sync_residual_ms": round(host, 3), "host_sync_pct_of_wall": round(100 * host / w_ms, 1),
                 "tok_s_W": round(1000 / w_ms, 1), "tok_s_D_ceiling": round(1000 / d_ms, 1),
                 "programs_per_token": progs, "debug2_unbatched_gpu_ms": round(gpu_dbg * 1e3, 2),
                 "item_syncs_per_token_W": 1, "item_syncs_per_token_D": 0})
    print(f"ctx {ck:5}{'F' if use_flash else ' '}: W {w_ms:6.2f}ms ({1000/w_ms:.1f} tok/s) | D {d_ms:6.2f}ms (ceiling {1000/d_ms:.1f}) "
          f"| host-sync {host:.2f}ms ({rows[-1]['host_sync_pct_of_wall']}%) | progs {progs}", file=sys.__stderr__)

  med_host = statistics.median([r["host_sync_pct_of_wall"] for r in rows])
  out = {"model_id": pathlib.Path(model).stem, "hardware": "RX 7900 XTX / gfx1100",
         "ckpts": list(profile.ckpts), "nmeas": profile.nmeas, "max_context": profile.max_context,
         "method": "W=real decode (.item/token), D=dispatch-only (no per-token sync, 1 final sync); host_sync=W-D",
         "rows": rows, "median_host_sync_pct": round(med_host, 1),
         "verdict": ("RUNTIME IS A TARGET (host-sync >20% of wall): a low-sync path could approach the D ceiling"
                     if med_host >= 20 else "runtime NOT the main target (host-sync <10-20%): GPU-bound, stop" if med_host < 10
                     else "borderline (10-20%): marginal runtime target")}
  print(f"\nmedian host-sync {med_host}% of wall | {out['verdict']}", file=sys.__stderr__)
  art = pathlib.Path("bench/qk-decode-runtime-overhead/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2)); print(f"artifact: {art}", file=sys.__stderr__)
  print("@@DONE@@")

if __name__ == "__main__":
  main()
