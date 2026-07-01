#!/usr/bin/env python3
"""TG-P5 gate: is the generated G5 block-tile flash decode a correct, route-bound, not-slower replacement for the
owned HIP two-kernel attention at the 8B geometry (Hq=32/Hkv=8/Hd=128, G=4)?

Measures BOTH routes in isolated subprocesses (tinygrad getenv is memoized on first read, so a fresh process per
route is the only valid comparison), same session/clock. Per ctx checkpoint: warmup, median decode ms over NMEAS
steps (the in-model W==D timing authority), greedy tokens (correctness gate vs owned), and route attribution (which
attention kernel fired, via DEBUG=2). Methodology mirrors extra/qk_decode_route_attribution_wd.py.

  candidate  = generated G5 block tile   (DECODE_FLASH_BLOCK_TILE_G5_8B=1)  -> flash_block_tiled_*
  comparator = owned HIP tile (oracle)   (DECODE_ATTN_AMDGCN_TILE=1)        -> owned_flash_tile_*

Writes bench/tg-p5-attention-generated-default/{latest.json,microgate.json,resources.json,wd_by_ctx.json}. Verdict
TG_P5_PASS_ATTENTION_GENERATED_DEFAULT / TG_P5_REFUTE_GENERATED_ATTENTION_SLOWER / TG_P5_BLOCKED_HIDDEN_OWNED_FALLBACK.
"""
from __future__ import annotations
import contextlib, io, json, os, pathlib, re, statistics, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/tg-p5-attention-generated-default"
MAXC = int(os.environ.get("QK_MAXC", "4608"))
CKPTS = [int(x) for x in os.environ.get("QK_CKPTS", "512,4096").split(",")]
NMEAS = int(os.environ.get("QK_NMEAS", "40"))
NTOK = int(os.environ.get("QK_NTOK", "24"))
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_ATTN = re.compile(r"(flash_block_tiled\w*|owned_flash\w*|flash_\w*|amdgcn_flash\w*|gqa\w*)")
CANDIDATE = {"DECODE_FLASH_BLOCK_TILE_G5_8B": "1", "DECODE_ATTN_AMDGCN_TILE": "1"}  # g5 fires first; owned stays as the fallthrough
COMPARATOR = {"DECODE_FLASH_BLOCK_TILE_G5_8B": "0", "DECODE_ATTN_AMDGCN_TILE": "1"}
ALL_FLAGS = set(CANDIDATE) | set(COMPARATOR)


def _spawn(flags, label):
  env = dict(os.environ)
  for k in ALL_FLAGS: env.pop(k, None)
  env.update({k: str(v) for k, v in flags.items()})
  env["QK_MEASURE_ONE"] = "1"; env.setdefault("DEV", "AMD"); env.setdefault("JIT", "1")
  p = subprocess.run([sys.executable, str(pathlib.Path(__file__))], env=env, capture_output=True, text=True, cwd=str(ROOT))
  m = re.search(r"@@RESULT@@(.*)", p.stdout)
  if not m:
    sys.stderr.write(f"[{label}] child failed:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}\n"); raise SystemExit(2)
  return json.loads(m.group(1))


def measure_one():
  from tinygrad import Tensor, UOp, TinyJit, Context, GlobalCounters
  from extra.llm_generate import load_model_and_tokenizer
  model = os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  m, tok = load_model_and_tokenizer(model, MAXC, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 800)
  ids = (ids * (1 + MAXC // max(1, len(ids))))[:MAXC]
  v_sp = UOp.variable("start_pos", 0, MAXC - 1); temp = Tensor([0.0]); rows = {}
  for ck in CKPTS:
    use_flash = ck >= int(os.environ.get("FLASH_DECODE_THRESHOLD", "512"))
    for b in m.blk: b._use_flash, b._prefill_v2 = use_flash, False
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
    kernels = sorted({_ANSI.sub("", l) for l in buf.getvalue().splitlines()})
    attn = sorted({mm.group(1) for l in kernels if (mm := _ATTN.search(l))})
    w_ms = statistics.median(W) * 1e3
    rows[ck] = {"tok_s": round(1000 / w_ms, 1), "w_ms_median": round(w_ms, 3),
                "w_ms_stdev": round(statistics.pstdev(W) * 1e3, 3), "nmeas": NMEAS, "tokens": toks, "attn_kernels": attn}
  del m
  print("@@RESULT@@" + json.dumps(rows))


def main():
  if os.environ.get("QK_MEASURE_ONE"):
    measure_one(); return 0
  OUT.mkdir(parents=True, exist_ok=True)
  print("== comparator (owned oracle) ==", file=sys.stderr)
  owned = _spawn(COMPARATOR, "owned")
  print("== candidate (generated G5 8B) ==", file=sys.stderr)
  gen = _spawn(CANDIDATE, "g5_8b")
  WD_PCT = float(os.environ.get("QK_WD_PROMOTION_PCT", "98"))  # not-slower tolerance: candidate >= 98% of owned tok/s
  per_ctx, all_match, all_bound, worst_pct = [], True, True, 1e9
  for ck in CKPTS:
    o, c = owned[str(ck)] if str(ck) in owned else owned[ck], gen[str(ck)] if str(ck) in gen else gen[ck]
    match = o["tokens"] == c["tokens"]
    cand_is_g5 = any("flash_block_tiled" in k for k in c["attn_kernels"])
    owned_not_in_cand = not any("owned_flash" in k for k in c["attn_kernels"])
    bound = cand_is_g5 and owned_not_in_cand
    pct = round(100 * c["tok_s"] / o["tok_s"], 1)
    worst_pct = min(worst_pct, pct)
    all_match = all_match and match; all_bound = all_bound and bound
    per_ctx.append({"ctx": ck, "candidate_tok_s": c["tok_s"], "owned_tok_s": o["tok_s"], "pct_of_owned": pct,
                    "token_match": match, "route_bound": bound, "candidate_attn_kernels": c["attn_kernels"],
                    "owned_attn_kernels": o["attn_kernels"], "candidate_w_ms": c["w_ms_median"], "owned_w_ms": o["w_ms_median"]})
  if not all_bound: verdict = "TG_P5_BLOCKED_HIDDEN_OWNED_FALLBACK"
  elif not all_match: verdict = "TG_P5_REFUTE_GENERATED_ATTENTION_TOKEN_MISMATCH"
  elif worst_pct >= WD_PCT: verdict = "TG_P5_PASS_ATTENTION_GENERATED_DEFAULT"
  else: verdict = "TG_P5_REFUTE_GENERATED_ATTENTION_SLOWER"
  latest = {"scope": "TG-P5 generated G5 8B attention vs owned HIP (in-model W==D + route attribution + token match)",
            "verdict": verdict, "geometry": {"Hq": 32, "Hkv": 8, "Hd": 128, "G": 4},
            "not_slower_tol_pct": WD_PCT, "worst_pct_of_owned": worst_pct, "all_token_match": all_match,
            "all_route_bound": all_bound, "per_ctx": per_ctx}
  json.dump(latest, open(OUT / "latest.json", "w"), indent=2)
  json.dump({"per_ctx": per_ctx}, open(OUT / "wd_by_ctx.json", "w"), indent=2)
  json.dump({"note": "token-match correctness gate vs owned oracle", "all_token_match": all_match,
             "per_ctx": [{"ctx": r["ctx"], "token_match": r["token_match"]} for r in per_ctx]}, open(OUT / "microgate.json", "w"), indent=2)
  print(verdict, "worst_pct_of_owned=", worst_pct, "all_match=", all_match, "all_bound=", all_bound)
  return 0 if verdict == "TG_P5_PASS_ATTENTION_GENERATED_DEFAULT" else 1


if __name__ == "__main__":
  raise SystemExit(main())
