#!/usr/bin/env python3
"""STEP 0 isolated timing for owned vs generated block-tile decode attention.

Times only the attention tile kernels, not the full decode route/combine, using eager custom_kernel + DEBUG=2.
"""
from __future__ import annotations
import contextlib, io, json, os, pathlib, re, statistics, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-block-tile-isolated-timing"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(r"\*\*\*\s+\S+\s+\d+\s+(.+?)\s+arg\s+\d+\s+mem")
MAXC, Hq, Hkv, Hd, S_OWNED = 4608, 32, 8, 128, 48


def _debug_ms(text: str, prefix: str) -> float | None:
  vals = []
  for raw in text.splitlines():
    line = ANSI.sub("", raw)
    if prefix not in line: continue
    m = re.search(r"tm\s+([0-9]+(?:\.[0-9]+)?)\s*(us|ms)/", line)
    if not m: continue
    val, unit = m.groups()
    vals.append(float(val) / (1000.0 if unit == "us" else 1.0))
  return statistics.median(vals) if vals else None


def _run_generated(ctx: int, S: int) -> dict[str, Any]:
  import numpy as np
  from tinygrad import Context, Tensor, UOp, dtypes
  from extra.qk_flash_decode import flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel
  W = Hd + 2
  L = max(1, (ctx + S - 1) // S)
  s_route = (ctx + L - 1) // L
  rng = np.random.default_rng(20260626 + ctx + S)
  q = Tensor(rng.normal(0, 0.25, (Hq, Hd)).astype(np.float16)).realize()
  cache = Tensor(rng.normal(0, 0.25, (2, 1, Hkv, MAXC, Hd)).astype(np.float16)).realize()
  vsp = UOp.variable("start_pos", 0, MAXC - 1)
  Tc = vsp + 1
  po = Tensor.empty(Hq * s_route * W, dtype=dtypes.float32)
  fxn = flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(Hd, Hq, Hkv, MAXC, L, s_route, Tc)
  def one():
    out = po.custom_kernel(q.reshape(-1), cache, fxn=fxn)[0]
    carry = Tensor.ones(MAXC, dtype=dtypes.float32)[0:vsp.bind(ctx - 1)].sum().reshape(1) * 0.0
    return (out + carry).realize()
  one()
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), Context(DEBUG=2): one()
  txt = buf.getvalue()
  return {"kind": "generated", "ctx": ctx, "S": s_route, "requested_S": S, "L": L,
          "kernel_prefix": "flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128",
          "debug_ms": _debug_ms(txt, "flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128"),
          "debug_tail": txt[-4000:]}


def _run_owned(ctx: int, S: int = S_OWNED) -> dict[str, Any]:
  import numpy as np
  from tinygrad import Context, Tensor, UOp, dtypes
  from extra.qk_owned_flash_decode_graph_node import _kernels, _make_program, Hq as OHq, Hkv as OHkv, Hd as OHd
  assert (OHq, OHkv, OHd) == (Hq, Hkv, Hd)
  rng = np.random.default_rng(20260627 + ctx + S)
  q = Tensor(rng.normal(0, 0.25, (Hq, Hd)).astype(np.float16)).realize()
  cache = Tensor(rng.normal(0, 0.25, (2, 1, Hkv, MAXC, Hd)).astype(np.float16)).realize()
  vsp = UOp.variable("start_pos", 0, MAXC - 1)
  tile_elf, _comb_elf, tile_lds, _comb_lds, tile_sym = _kernels(S, MAXC, whole_cache=True)
  part = Tensor.empty(Hq * S * Hd, dtype=dtypes.float32)
  meta = Tensor.empty(Hq * S * 2, dtype=dtypes.float32)
  def fxn(*ph):
    return _make_program(tile_sym, tile_elf, list(ph), (vsp,), (Hkv, S, 1), (128, 1, 1), outs=(2, 3), ins=(0, 1),
                         group_seg=tile_lds, est_ops=Hq*MAXC*Hd*2, est_mem=Hkv*MAXC*Hd*2*2)
  def one():
    r = Tensor.custom_kernel(q.reshape(-1), cache, part, meta, fxn=fxn)
    carry = Tensor.ones(MAXC, dtype=dtypes.float32)[0:vsp.bind(ctx - 1)].sum().reshape(1) * 0.0
    return (r[2] + carry).realize(), (r[3] + carry).realize()
  one()
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), Context(DEBUG=2): one()
  txt = buf.getvalue()
  return {"kind": "owned", "ctx": ctx, "S": S, "L": (ctx + S - 1) // S, "kernel_prefix": tile_sym,
          "debug_ms": _debug_ms(txt, tile_sym), "debug_tail": txt[-4000:]}


def build() -> dict[str, Any]:
  rows = []
  for ctx in (512, 4096):
    # current generated route equivalent before STEP 1: target_s=48 via MAXC -> L=96, S=ceil(ctx/96)
    gen_route_s = (ctx + ((MAXC + 48 - 1) // 48) - 1) // ((MAXC + 48 - 1) // 48)
    gen = _run_generated(ctx, gen_route_s)
    owned = _run_owned(ctx, S_OWNED)
    ratio = round(gen["debug_ms"] / owned["debug_ms"], 2) if gen["debug_ms"] and owned["debug_ms"] else None
    rows.append({"ctx": ctx, "generated": gen, "owned": owned, "gen_vs_owned_ratio": ratio})
  return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
          "method": "eager custom_kernel + DEBUG=2, tile kernel only", "rows": rows,
          "verdict": "ISOLATED_TILE_TIMING_RECORDED"}


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  (OUT / "latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"isolated-timing-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
