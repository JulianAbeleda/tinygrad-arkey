#!/usr/bin/env python3
"""TG-P8.2: bounded geometry search for the generated 8B decode-attention route.

TG-P8.1 classified the delta as SPLIT_GEOMETRY_MISMATCH: the generated whole-cache tile launches smax_route =
ceildiv(MAXC, L) splits regardless of the valid context, and the gmax/combine kernels reduce over those same splits.
The one searchable geometry knob that moves both is L (the split length; FLASH_L env, generated-route only -- owned
uses DECODE_ATTN_AMDGCN_S). Larger L -> fewer splits S -> fewer masked tile workgroups at low ctx AND a narrower
combine/gmax reduction. Smaller S also risks under-occupancy at high ctx. This searches the sweet spot.

Bounded enumeration over L. For each L: measure the generated G4 route (DECODE_FLASH_BLOCK_TILE_G5_8B=1, FLASH_L=L,
K-only) vs the owned oracle at ctx512 and ctx4096 -- token-match, route-bound, tok/s, per-kernel attn wall split.
Promotion bar: >=98% of owned at BOTH protected contexts, token-identical, route-bound.

Writes bench/tg-p8-generated-8b-attention-parity/geometry_search.json. Verdict TG_P8_2_PASS_GEOMETRY_CANDIDATE_SELECTED
/ TG_P8_2_REFUTE_GEOMETRY_SPACE / TG_P8_2_BLOCKED_SEARCH_SPACE_EXPLOSION.
"""
from __future__ import annotations
import contextlib, io, json, os, pathlib, re, statistics, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/tg-p8-generated-8b-attention-parity"
MAXC = int(os.environ.get("QK_MAXC", "4608"))
CKPTS = [int(x) for x in os.environ.get("QK_CKPTS", "512,4096").split(",")]
NMEAS = int(os.environ.get("QK_NMEAS", "40"))
NTOK = int(os.environ.get("QK_NTOK", "24"))
MODEL = os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
L_GRID = [int(x) for x in os.environ.get("QK_L_GRID", "128,256,384,512,576").split(",")]
PROMOTE_PCT = float(os.environ.get("QK_WD_PROMOTION_PCT", "98"))
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_TM = re.compile(r"\*\*\* AMD\s+\d+\s+(\S+).*?tm\s+([\d.]+)us")
_ATTN = re.compile(r"(flash_block_tiled\w*|owned_flash\w*|flash_state\w*|flash_\w*|amdgcn_flash\w*|gqa\w*)")
# route flags: owned (G4 off) vs generated (G4 on). FLASH_L set per-arm in the child env.
ALL_FLAGS = {"DECODE_FLASH_BLOCK_TILE_G5_8B", "DECODE_ATTN_AMDGCN_TILE", "FLASH_L"}


def _spawn(flags, label):
  env = dict(os.environ)
  for k in ALL_FLAGS: env.pop(k, None)
  env.update({k: str(v) for k, v in flags.items()})
  env["QK_MEASURE_ONE"] = "1"; env.setdefault("DEV", "AMD"); env.setdefault("JIT", "1")
  p = subprocess.run([sys.executable, str(pathlib.Path(__file__))], env=env, capture_output=True, text=True, cwd=str(ROOT))
  m = re.search(r"@@RESULT@@(.*)", p.stdout)
  if not m:
    sys.stderr.write(f"[{label}] child failed:\n{p.stdout[-1500:]}\n{p.stderr[-1500:]}\n"); raise SystemExit(2)
  return json.loads(m.group(1))


def measure_one():
  from tinygrad import Tensor, UOp, TinyJit, Context, GlobalCounters
  from extra.llm_generate import load_model_and_tokenizer
  m, tok = load_model_and_tokenizer(MODEL, MAXC, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 800)
  ids = (ids * (1 + MAXC // max(1, len(ids))))[:MAXC]
  v_sp = UOp.variable("start_pos", 0, MAXC - 1); temp = Tensor([0.0]); rows = {}
  for ck in CKPTS:
    for b in m.blk: b._use_flash, b._prefill_v2 = ck >= 512, False
    step = TinyJit(m.forward); tokid = int(ids[ck])
    out = Tensor([[tokid]], dtype="int32").contiguous()
    for i in range(8): out = step(out, v_sp.bind(ck + i), temp).realize()
    out = Tensor([[tokid]], dtype="int32").contiguous(); W, toks = [], []
    for i in range(NMEAS):
      t0 = time.perf_counter(); out = step(out, v_sp.bind(ck + i), temp); tid = int(out.item())
      W.append(time.perf_counter() - t0)
      if i < NTOK: toks.append(tid)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), Context(DEBUG=2):
      GlobalCounters.reset(); m.forward(Tensor([[tokid]], dtype="int32").contiguous(), v_sp.bind(ck), temp).realize()
    per, cnt = {}, {}
    for line in buf.getvalue().splitlines():
      mm = _TM.search(_ANSI.sub("", line))
      if not mm or not _ATTN.search(mm.group(1)): continue
      per[mm.group(1)] = per.get(mm.group(1), 0.0) + float(mm.group(2)); cnt[mm.group(1)] = cnt.get(mm.group(1), 0) + 1
    w_ms = statistics.median(W) * 1e3
    rows[ck] = {"tok_s": round(1000 / w_ms, 1), "w_ms_median": round(w_ms, 3), "tokens": toks,
                "attn_kernels": sorted(per), "attn_total_us": round(sum(per.values()), 1),
                "attn_split_us": {k: round(per[k], 1) for k in per}}
  del m
  print("@@RESULT@@" + json.dumps(rows))


def main():
  if os.environ.get("QK_MEASURE_ONE"):
    measure_one(); return 0
  if len(L_GRID) > 12:
    json.dump({"verdict": "TG_P8_2_BLOCKED_SEARCH_SPACE_EXPLOSION", "n": len(L_GRID)}, open(OUT / "geometry_search.json", "w"), indent=2)
    print("TG_P8_2_BLOCKED_SEARCH_SPACE_EXPLOSION"); return 1
  OUT.mkdir(parents=True, exist_ok=True)
  print("== owned oracle ==", file=sys.stderr)
  owned = _spawn({"DECODE_FLASH_BLOCK_TILE_G5_8B": "0", "DECODE_ATTN_AMDGCN_TILE": "1"}, "owned")
  candidates = []
  for L in L_GRID:
    print(f"== generated G4 L={L} ==", file=sys.stderr)
    g = _spawn({"DECODE_FLASH_BLOCK_TILE_G5_8B": "1", "DECODE_ATTN_AMDGCN_TILE": "1", "FLASH_L": L}, f"L{L}")
    per_ctx, ok = [], True
    for ck in CKPTS:
      o, c = owned[str(ck)], g[str(ck)]
      match = o["tokens"] == c["tokens"]
      bound = any("flash_block_tiled" in k for k in c["attn_kernels"]) and not any("owned_flash" in k for k in c["attn_kernels"])
      pct = round(100 * c["tok_s"] / o["tok_s"], 1)
      ok = ok and match and bound
      per_ctx.append({"ctx": ck, "owned_tok_s": o["tok_s"], "gen_tok_s": c["tok_s"], "pct_of_owned": pct,
                      "token_match": match, "route_bound": bound, "gen_attn_us": c["attn_total_us"],
                      "owned_attn_us": o["attn_total_us"], "gen_attn_split": c["attn_split_us"], "splits": (MAXC + L - 1)//L})
    worst = min(r["pct_of_owned"] for r in per_ctx)
    passes = ok and all(r["pct_of_owned"] >= PROMOTE_PCT for r in per_ctx)
    candidates.append({"L": L, "splits": (MAXC + L - 1)//L, "worst_pct_of_owned": worst, "correct_and_bound": ok,
                       "passes_bar": passes, "per_ctx": per_ctx})

  valid = [c for c in candidates if c["correct_and_bound"]]
  winners = [c for c in valid if c["passes_bar"]]
  best = max(valid, key=lambda c: c["worst_pct_of_owned"], default=None)
  verdict = "TG_P8_2_PASS_GEOMETRY_CANDIDATE_SELECTED" if winners else "TG_P8_2_REFUTE_GEOMETRY_SPACE"
  latest = {"scope": "TG-P8.2 geometry search over L (split length) for generated 8B decode attention",
            "verdict": verdict, "promote_pct": PROMOTE_PCT, "L_grid": L_GRID,
            "selected": (min(winners, key=lambda c: c["L"]) if winners else None),
            "best_effort": best, "candidates": candidates,
            "note": "L is the only generated geometry knob that scales the concrete split count (S=ceildiv(MAXC,L)); "
                    "owned unaffected (uses DECODE_ATTN_AMDGCN_S). Larger L -> fewer splits -> cheaper combine + fewer "
                    "masked tile workgroups, traded against high-ctx occupancy."}
  json.dump(latest, open(OUT / "geometry_search.json", "w"), indent=2)
  md = [f"# TG-P8.2 Geometry Search (L)\n", f"Verdict: **{verdict}**\n",
        "| L | splits | ctx512 %own | ctx4096 %own | worst | correct+bound | passes>=98% |",
        "|---|---|---|---|---|---|---|"]
  for c in candidates:
    p512 = next((r["pct_of_owned"] for r in c["per_ctx"] if r["ctx"] == 512), "-")
    p4096 = next((r["pct_of_owned"] for r in c["per_ctx"] if r["ctx"] == 4096), "-")
    md.append(f"| {c['L']} | {c['splits']} | {p512}% | {p4096}% | {c['worst_pct_of_owned']}% | {c['correct_and_bound']} | {c['passes_bar']} |")
  open(OUT / "geometry_search_summary.md", "w").write("\n".join(md) + "\n")
  print(verdict, "best_worst_pct=", best["worst_pct_of_owned"] if best else None, "winners=", [c["L"] for c in winners])
  return 0 if winners else 1


if __name__ == "__main__":
  raise SystemExit(main())
