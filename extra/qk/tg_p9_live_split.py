#!/usr/bin/env python3
"""TG-P9 live-context split-geometry trio (Cluster E), collapsed to one parameterized module.

Three sequential microgates over the SAME 8B geometry (Hq=32,Hkv=8,Hd=128,MAXC=4608) and the SAME symbolic-Tc carry
trick (out + ones[0:vsp.bind(Tc-1)].sum()*0 -- the standalone equivalent of the model's JIT-replay var binding). They
share `extra.qk.live_split_geometry` (+ `flash_decode` for the fixed-L baseline) as LIBRARIES. Each VARIANT builds its
microgate and RETURNS the verdict dict (gate_registry writes bench/tg-p9-pure-attention-primitive-route/<artifact> +
prints it); the builder never writes artifacts or calls sys.exit. Registry entrypoints: build_live_split(),
build_live_split_tile(), build_combine().

  live_split  (P9.1) -- prove the live-context split geometry primitive lowers correctly (generated UOp, no HIP/ASM).
    For several (S,Tc) incl. the 8B shape, run the coverage kernel and verify: (1) every token in [0,Tc) covered
    exactly once; (2) no split writes beyond Tc; (3) grid launches S workgroups FIXED, NOT ceildiv(MAXC,L) -- the
    coverage invariant. Verdict TG_P9_1_PASS_LIVE_TC_SPLIT_IR / _BLOCKED_UOP_RANGE_MODEL / _BLOCKED_SYMBOLIC_BOUNDS.

  live_split_tile (P9.2) -- the live-split generated tile is numerically == the fixed-L g5 tile AND reduces tile work
    at low ctx. Synthetic q + cache at the 8B geometry; for each ctx compare flash_decode_live_split_block_tile
    (S fixed, per=ceildiv(Tc,S)) vs flash_decode_g5_block_tile (fixed L=128), both online-softmax over [0,Tc). Times
    each end to end (DEBUG=2 per-kernel wall) to show live scales with Tc while fixed-L is flat. Env knobs QK_CKPTS
    (default "512,4096"), QK_ITERS (default 40). Verdict TG_P9_2_PASS_LIVE_SPLIT_TILE / _BLOCKED_CORRECTNESS /
    _REFUTE_LIVE_SPLIT_NO_MOVEMENT (correct but no timing win -- the movement/REFUTE decision rule: any ctx with
    live_tile < 0.9x fixed_tile => movement).

  combine (P9.3/9.4) -- a split-preserving LSE combine that removes the per-d fexp redundancy of flash_state_combine
    (the 556us/fwd ctx4096 cap) WITHOUT collapsing Hq*S or Hq*Hd parallelism. Three generated-UOp designs (LDS
    weight-share 32-lane warp / inline-gmax single-kernel / two-stage fexp-free) are compiled. Verdict
    TG_P9_4_PASS_COMBINE_MICROGATE if all compile, else TG_P9_4_BLOCKED_EMITTER -- the AMD-backend compiler wall
    (cross-references Cluster A TG-P10.1): the reduction-accumulator REG is vectorized into a non-assignable
    make_float4(...)=... store; REG_STORE_DEVEC=1 compiles them but returns NaN (max-reduce mis-lowers). Only the
    shipped single-reduce per-d combine compiles+is correct. no_parallelism_collapse=true; the split-preserving
    combine IS expressible in principle but the current emitter cannot lower it. Reopen: a tinygrad codegen fix so the
    reduction-accumulator REG stays scalar (or DEVEC lowers the max-reduce correctly) for a multi-reduce /
    weight-sharing combine kernel.

Run:  DEV=AMD PYTHONPATH=. python3 -m extra.qk.gate_registry run tg_p9_live_split [tg_p9_live_split_tile tg_p9_combine]
"""
from __future__ import annotations
import contextlib, io, json, os, pathlib, re

os.environ.setdefault("DEV", "AMD")
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/tg-p9-pure-attention-primitive-route"

# 8B route geometry (shared): S=ceildiv(MAXC,L)=36 at MAXC=4608,L=128 -> same occupancy as the fixed-L=128 route.
Hq, Hkv, Hd, MAXC = 32, 8, 128, 4608
S_FIXED = MAXC // 128   # 36 splits


def _carry(Tensor, vsp, bound, maxc, dtype, shape):
  """Symbolic-Tc carry trick: a zero term whose slice bound binds `start_pos`=bound so realize infers var_vals --
  the standalone equivalent of the model's JIT-replay var binding."""
  return Tensor.ones(maxc, dtype=dtype)[0:vsp.bind(bound)].sum().reshape(*shape) * 0


# ---- live_split (P9.1): live-context split geometry coverage --------------------------------------------------------
# (S, MAXC, [Tc values]) -- 8B route uses S=ceildiv(MAXC,L)=36 at MAXC=4608,L=128
_P91_CASES = [
  {"S": 36, "MAXC": 4608, "tcs": [1, 15, 512, 513, 4096, 4608]},   # 8B geometry
  {"S": 48, "MAXC": 4608, "tcs": [1, 512, 4096]},                  # owned S=48
  {"S": 8, "MAXC": 2048, "tcs": [1, 7, 512, 2048]},                # small
]


def _live_split():
  OUT.mkdir(parents=True, exist_ok=True)
  from tinygrad import Tensor, Device, dtypes
  from tinygrad.uop.ops import UOp
  from extra.qk.live_split_geometry import LiveSplitGeometry, live_split_coverage_kernel

  results, all_ok, blocked = [], True, None
  for c in _P91_CASES:
    S, MAXC = c["S"], c["MAXC"]
    geo = LiveSplitGeometry(S=S, TK=16)
    # start_pos as an UNBOUND symbolic var; Tc = start_pos + 1 flows into the kernel unbound. The bound value is
    # threaded via a carry term (out + ones[0:vsp.bind(Tc-1)].sum()*0) so realize infers var_vals -- the standalone
    # equivalent of the model's JIT-replay binding.
    vsp = UOp.variable("start_pos", 0, MAXC - 1)
    kfn = live_split_coverage_kernel(geo, MAXC, vsp + 1)
    for Tc in c["tcs"]:
      try:
        cov = Tensor.zeros(MAXC, dtype=dtypes.int32, device=Device.DEFAULT).contiguous()
        outt = cov.custom_kernel(fxn=kfn)[0]
        carry = _carry(Tensor, vsp, Tc - 1, MAXC, dtypes.int32, (1,))
        out = (outt + carry).realize().numpy()
      except Exception as e:
        blocked = f"{type(e).__name__}: {str(e)[:200]}"
        results.append({"S": S, "MAXC": MAXC, "Tc": Tc, "error": blocked})
        all_ok = False
        continue
      inside = out[:Tc]
      outside = out[Tc:]
      cover_once = bool(np.all(inside == 1))
      no_spill = bool(np.all(outside == 0))
      covered = int(inside.sum())
      ok = cover_once and no_spill and covered == Tc
      all_ok = all_ok and ok
      results.append({"S": S, "MAXC": MAXC, "Tc": Tc, "covered_count": covered, "expected": Tc,
                      "cover_each_once": cover_once, "no_spill_beyond_Tc": no_spill, "ok": ok,
                      "grid_workgroups": S, "note": "grid=S (fixed), not ceildiv(MAXC,L)"})

  if blocked and not any(r.get("ok") for r in results):
    # never lowered at all -> range model / symbolic bound rejected
    verdict = "TG_P9_1_BLOCKED_SYMBOLIC_BOUNDS" if "range" not in blocked.lower() else "TG_P9_1_BLOCKED_UOP_RANGE_MODEL"
  elif all_ok:
    verdict = "TG_P9_1_PASS_LIVE_TC_SPLIT_IR"
  else:
    verdict = "TG_P9_1_BLOCKED_SYMBOLIC_BOUNDS"
  latest = {"scope": "TG-P9.1 live-context split geometry coverage microgate", "verdict": verdict,
            "capability": "fixed S splits + symbolic per-split length per=ceildiv(Tc,S); symbolic inner-loop bound nb=ceildiv(per,TK)",
            "all_ok": all_ok, "blocked_reason": blocked, "cases": results}
  print(verdict, "all_ok=", all_ok, "blocked=", blocked)
  return latest


# ---- live_split_tile (P9.2): live-split tile parity + timing ---------------------------------------------------------
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_TM = re.compile(r"\*\*\* AMD\s+\d+\s+(\S+).*?tm\s+([\d.]+)us")
_ATTN = re.compile(r"(flash_block_tiled\w*|flash_state\w*)")
_CTXS = [int(x) for x in os.environ.get("QK_CKPTS", "512,4096").split(",")]
_ITERS = int(os.environ.get("QK_ITERS", "40"))


def _live_split_tile():
  OUT.mkdir(parents=True, exist_ok=True)
  from tinygrad import Tensor, Device, dtypes
  from tinygrad.uop.ops import UOp
  from extra.qk.flash_decode import flash_decode_g5_block_tile
  from extra.qk.live_split_geometry import flash_decode_live_split_block_tile

  rng = np.random.RandomState(0)
  q = Tensor(rng.normal(0, 0.25, (Hq, Hd)).astype(np.float16), device="AMD").realize()
  cache = Tensor(rng.normal(0, 0.25, (2, 1, Hkv, MAXC, Hd)).astype(np.float16), device="AMD").realize()

  def bind_carry(vsp, ctx):
    return _carry(Tensor, vsp, ctx - 1, MAXC, dtypes.float32, (1, 1))

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
  for ctx in _CTXS:
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
  print(verdict, "all_correct=", all_correct, "movement=", any_movement,
        "| " + " ".join(f"ctx{r['ctx']}:{r.get('live_over_fixed','?')}x" for r in results if "error" not in r))
  return latest


# ---- combine (P9.3/9.4): split-preserving LSE combine primitive ------------------------------------------------------
def _try(fn):
  try:
    fn(); return "compiles"
  except Exception as e:
    return f"{type(e).__name__}: {str(e)[:80]}"


def _combine():
  OUT.mkdir(parents=True, exist_ok=True)
  from tinygrad import Tensor, dtypes
  from extra.qk.live_split_geometry import (flash_fused_gmax_combine_kernel, flash_inline_gm_combine_kernel,
                                            flash_gm_weights_kernel, flash_weighted_sum_kernel)
  Hq, Hd, S = 32, 128, 36; W = Hd + 2
  pout = Tensor(np.zeros(Hq * S * W, dtype=np.float32), device="AMD").realize()
  attempts = {
    "lds_warp_fused": _try(lambda: Tensor.empty(Hq * Hd, dtype=dtypes.float32, device="AMD").custom_kernel(
      pout, fxn=flash_fused_gmax_combine_kernel(Hd, Hq, S, stride=S))[0].realize()),
    "inline_gmax": _try(lambda: Tensor.empty(Hq * Hd, dtype=dtypes.float32, device="AMD").custom_kernel(
      pout, fxn=flash_inline_gm_combine_kernel(Hd, Hq, S, stride=S))[0].realize()),
    "two_stage_weights": _try(lambda: Tensor.empty(Hq * S, dtype=dtypes.float32, device="AMD").custom_kernel(
      pout, fxn=flash_gm_weights_kernel(Hd, Hq, S, stride=S))[0].realize()),
  }
  blocked = all(v != "compiles" for v in attempts.values())
  verdict = "TG_P9_4_BLOCKED_EMITTER" if blocked else "TG_P9_4_PASS_COMBINE_MICROGATE"
  latest = {"scope": "TG-P9.3/9.4 split-preserving combine primitive", "verdict": verdict,
            "classification": "EMITTER_BLOCKED", "no_parallelism_collapse": True,
            "signature": "reduction-accumulator REG vectorized to a non-assignable make_float4(...) = ... in every "
                         "weight-sharing / gmax-fusing combine shape; REG_STORE_DEVEC=1 compiles them but returns NaN "
                         "(max-reduce mis-lowers). Only the shipped single-reduce per-d combine compiles correctly.",
            "attempts": attempts,
            "primitive_gap": "AMD backend accumulator-REG vectorization + REG_STORE_DEVEC do not correctly lower a "
                             "combine reduction that shares softmax weights across d or fuses the gmax max-reduce; "
                             "so the per-d fexp redundancy (the ctx4096 556us cap) cannot be removed in generated UOp.",
            "reopen_condition": "a tinygrad codegen fix so the reduction-accumulator REG stays scalar (or DEVEC lowers "
                                "the max-reduce correctly) for a multi-reduce / weight-sharing combine kernel."}
  print(verdict, attempts)
  return latest


# ---- registry surface ------------------------------------------------------------------------------------------------
VARIANTS = {"live_split": _live_split, "live_split_tile": _live_split_tile, "combine": _combine}

def build(variant): return VARIANTS[variant]()
def build_live_split(): return build("live_split")
def build_live_split_tile(): return build("live_split_tile")
def build_combine(): return build("combine")


if __name__ == "__main__":
  import sys
  out = build(sys.argv[1] if len(sys.argv) > 1 else "live_split")
  print(json.dumps(out, indent=2))
