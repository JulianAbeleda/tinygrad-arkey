#!/usr/bin/env python3
"""TG-P9.2 microgate: the live-split generated tile is numerically equivalent to the fixed-L g5 tile, and reduces
tile work at low ctx.

Standalone (no full model): synthetic q + cache_kv at the 8B geometry (Hq=32,Hkv=8,Hd=128). For ctx512 and ctx4096,
compare flash_decode_live_split_block_tile (S fixed, per=ceildiv(Tc,S)) vs flash_decode_g5_block_tile (fixed L=128).
Both compute attention over [0,Tc) via online softmax, so outputs must match up to fp reassoc. Also times each end
to end (tile+gmax+combine) to show the live-split tile scales with Tc while fixed-L is flat.

Symbolic Tc is bound via the carry trick (out + ones[0:vsp.bind(Tc-1)].sum()*0), like the model's JIT binding.

Writes bench/tg-p9-pure-attention-primitive-route/live_split_tile_microgate.json. Verdict
TG_P9_2_PASS_LIVE_SPLIT_TILE / TG_P9_2_BLOCKED_CORRECTNESS / TG_P9_2_REFUTE_LIVE_SPLIT_NO_MOVEMENT.
"""
from __future__ import annotations
import contextlib, io, json, os, pathlib, re

os.environ.setdefault("DEV", "AMD")
import numpy as np

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_TM = re.compile(r"\*\*\* AMD\s+\d+\s+(\S+).*?tm\s+([\d.]+)us")
_ATTN = re.compile(r"(flash_block_tiled\w*|flash_state\w*)")

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/tg-p9-pure-attention-primitive-route"
Hq, Hkv, Hd, MAXC = 32, 8, 128, 4608
S_FIXED = MAXC // 128            # 36 splits: same occupancy as the fixed-L=128 route
CTXS = [int(x) for x in os.environ.get("QK_CKPTS", "512,4096").split(",")]
ITERS = int(os.environ.get("QK_ITERS", "40"))


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  from tinygrad import Tensor, Device, dtypes
  from tinygrad.uop.ops import UOp
  from extra.qk_flash_decode import flash_decode_g5_block_tile
  from extra.qk_live_split_geometry import flash_decode_live_split_block_tile

  rng = np.random.RandomState(0)
  q = Tensor(rng.normal(0, 0.25, (Hq, Hd)).astype(np.float16), device="AMD").realize()
  cache = Tensor(rng.normal(0, 0.25, (2, 1, Hkv, MAXC, Hd)).astype(np.float16), device="AMD").realize()

  def bind_carry(vsp, ctx):
    return Tensor.ones(MAXC, dtype=dtypes.float32)[0:vsp.bind(ctx - 1)].sum().reshape(1, 1) * 0.0

  def run_fixed(ctx):
    vsp = UOp.variable("start_pos", 0, MAXC - 1)
    out = flash_decode_g5_block_tile(q, cache, vsp + 1, vsp + 1, Hd, Hq, Hkv, MAXC, 128, staging="K_ONLY")
    return (out + bind_carry(vsp, ctx))

  def run_live(ctx):
    vsp = UOp.variable("start_pos", 0, MAXC - 1)
    out = flash_decode_live_split_block_tile(q, cache, vsp + 1, Hd, Hq, Hkv, MAXC, S_FIXED, staging="K_ONLY")
    return (out + bind_carry(vsp, ctx))

  from tinygrad import Context, GlobalCounters
  def tile_wall_us(fn, ctx):
    # per-kernel GPU wall for the attention kernels via DEBUG=2 (clean; the end-to-end carry sum would dominate).
    for _ in range(3): fn(ctx).realize()   # warm
    Device["AMD"].synchronize()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), Context(DEBUG=2):
      GlobalCounters.reset(); fn(ctx).realize()
    per = {}
    for line in buf.getvalue().splitlines():
      mm = _TM.search(_ANSI.sub("", line))
      if mm and _ATTN.search(mm.group(1)): per[mm.group(1)] = per.get(mm.group(1), 0.0) + float(mm.group(2))
    return per

  results, all_correct, any_movement, blocked = [], True, False, None
  for ctx in CTXS:
    try:
      a = run_fixed(ctx).realize().numpy()
      b = run_live(ctx).realize().numpy()
    except Exception as e:
      blocked = f"{type(e).__name__}: {str(e)[:200]}"
      results.append({"ctx": ctx, "error": blocked}); all_correct = False; continue
    rel = float(np.abs(a - b).max() / (np.abs(a).max() + 1e-6))
    correct = rel < 2e-2
    all_correct = all_correct and correct
    wf, wl = tile_wall_us(run_fixed, ctx), tile_wall_us(run_live, ctx)
    tf = sum(v for k, v in wf.items() if "flash_block_tiled" in k)
    tl = sum(v for k, v in wl.items() if "flash_block_tiled" in k)
    total_f, total_l = sum(wf.values()), sum(wl.values())
    if tl < tf * 0.9: any_movement = True
    results.append({"ctx": ctx, "max_rel_err": rel, "correct": correct,
                    "fixed_tile_us": round(tf, 2), "live_tile_us": round(tl, 2),
                    "live_tile_over_fixed": round(tl / tf, 4) if tf else None,
                    "fixed_total_us": round(total_f, 2), "live_total_us": round(total_l, 2),
                    "S_fixed": S_FIXED, "per_aligned_at_ctx": ((((ctx + S_FIXED - 1)//S_FIXED)+15)//16)*16})

  if blocked and not any(r.get("correct") for r in results):
    verdict = "TG_P9_2_BLOCKED_CORRECTNESS"
  elif not all_correct:
    verdict = "TG_P9_2_BLOCKED_CORRECTNESS"
  elif any_movement:
    verdict = "TG_P9_2_PASS_LIVE_SPLIT_TILE"
  else:
    verdict = "TG_P9_2_REFUTE_LIVE_SPLIT_NO_MOVEMENT"
  latest = {"scope": "TG-P9.2 live-split generated tile parity + timing (standalone)", "verdict": verdict,
            "geometry": {"Hq": Hq, "Hkv": Hkv, "Hd": Hd, "MAXC": MAXC, "S_fixed": S_FIXED},
            "all_correct": all_correct, "any_low_ctx_movement": any_movement, "blocked": blocked, "cases": results}
  json.dump(latest, open(OUT / "live_split_tile_microgate.json", "w"), indent=2)
  print(verdict, "all_correct=", all_correct, "movement=", any_movement,
        "| " + " ".join(f"ctx{r['ctx']}:{r.get('live_over_fixed','?')}x" for r in results if "error" not in r))
  return 0 if verdict == "TG_P9_2_PASS_LIVE_SPLIT_TILE" else 1


if __name__ == "__main__":
  raise SystemExit(main())
