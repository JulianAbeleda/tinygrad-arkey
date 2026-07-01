#!/usr/bin/env python3
"""TG-P8.0: evidence-refresh authority for the 8B decode-attention parity blocker.

Pins the owned vs generated-G4 baseline before any code change. For each route (owned HIP two-kernel; generated G4
block-tile K-only) at ctx512 and ctx4096, in ISOLATED subprocesses (tinygrad getenv is memoized), it captures:
  - tok/s + median decode wall (in-model W==D timing authority),
  - greedy tokens (token-equivalence gate vs owned),
  - route-bound attention kernel names (DEBUG=2 route attribution),
  - PER-KERNEL attention wall split (DEBUG=2 `tm` per kernel, summed over the forward's layers -> per-forward and
    per-occurrence us), so TG-P8.1 can attribute the gap.

Writes bench/tg-p8-generated-8b-attention-parity/{baseline.json,summary.md}. Verdict TG_P8_0_PASS_BASELINE_PINNED /
TG_P8_0_BLOCKED_ROUTE_ATTRIBUTION.
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
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# DEBUG=2 line: "*** AMD  <n> <kernel_name> arg N mem X GB tm  <us>us/ <ms>ms (...)"
_TM = re.compile(r"\*\*\* AMD\s+\d+\s+(\S+).*?tm\s+([\d.]+)us")
_ATTN_NAME = re.compile(r"(flash_block_tiled\w*|owned_flash\w*|flash_state\w*|flash_\w*|amdgcn_flash\w*|gqa\w*)")
OWNED = {"DECODE_FLASH_BLOCK_TILE_G5_8B": "0", "DECODE_ATTN_AMDGCN_TILE": "1"}
GENERATED = {"DECODE_FLASH_BLOCK_TILE_G5_8B": "1", "DECODE_ATTN_AMDGCN_TILE": "1"}
ALL_FLAGS = set(OWNED) | set(GENERATED)


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
    # per-kernel wall via DEBUG=2 eager forward
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), Context(DEBUG=2):
      GlobalCounters.reset(); m.forward(Tensor([[tokid]], dtype="int32").contiguous(), v_sp.bind(ck), temp).realize()
    per_kernel_us, counts = {}, {}
    for line in buf.getvalue().splitlines():
      mm = _TM.search(_ANSI.sub("", line))
      if not mm: continue
      name, us = mm.group(1), float(mm.group(2))
      if not _ATTN_NAME.search(name): continue
      per_kernel_us[name] = per_kernel_us.get(name, 0.0) + us
      counts[name] = counts.get(name, 0) + 1
    attn = sorted(per_kernel_us)
    split = {name: {"total_us_per_forward": round(per_kernel_us[name], 2), "occurrences": counts[name],
                    "us_per_occurrence": round(per_kernel_us[name] / max(1, counts[name]), 3)} for name in attn}
    attn_total = round(sum(per_kernel_us.values()), 2)
    w_ms = statistics.median(W) * 1e3
    rows[ck] = {"tok_s": round(1000 / w_ms, 1), "w_ms_median": round(w_ms, 3),
                "w_ms_stdev": round(statistics.pstdev(W) * 1e3, 3), "nmeas": NMEAS, "tokens": toks,
                "attn_kernels": attn, "attn_wall_split_us": split, "attn_total_us_per_forward": attn_total}
  del m
  print("@@RESULT@@" + json.dumps(rows))


def main():
  if os.environ.get("QK_MEASURE_ONE"):
    measure_one(); return 0
  OUT.mkdir(parents=True, exist_ok=True)
  print("== owned oracle ==", file=sys.stderr); owned = _spawn(OWNED, "owned")
  print("== generated G4 ==", file=sys.stderr); gen = _spawn(GENERATED, "gen")
  per_ctx, all_match, all_bound = [], True, True
  for ck in CKPTS:
    o, c = owned[str(ck)] if str(ck) in owned else owned[ck], gen[str(ck)] if str(ck) in gen else gen[ck]
    match = o["tokens"] == c["tokens"]
    cand_is_g4 = any("flash_block_tiled" in k for k in c["attn_kernels"])
    owned_is_owned = any("owned_flash" in k for k in o["attn_kernels"])
    no_owned_in_cand = not any("owned_flash" in k for k in c["attn_kernels"])
    bound = cand_is_g4 and owned_is_owned and no_owned_in_cand
    all_match = all_match and match; all_bound = all_bound and bound
    per_ctx.append({"ctx": ck, "owned_tok_s": o["tok_s"], "generated_tok_s": c["tok_s"],
                    "pct_of_owned": round(100 * c["tok_s"] / o["tok_s"], 1), "token_match": match, "route_bound": bound,
                    "owned_attn_total_us": o["attn_total_us_per_forward"], "generated_attn_total_us": c["attn_total_us_per_forward"],
                    "owned_attn_split": o["attn_wall_split_us"], "generated_attn_split": c["attn_wall_split_us"],
                    "owned_attn_kernels": o["attn_kernels"], "generated_attn_kernels": c["attn_kernels"]})
  verdict = "TG_P8_0_PASS_BASELINE_PINNED" if all_bound and all_match else \
            ("TG_P8_0_BLOCKED_ROUTE_ATTRIBUTION" if not all_bound else "TG_P8_0_BLOCKED_TOKEN_MISMATCH")
  latest = {"scope": "TG-P8.0 evidence refresh: owned vs generated-G4 8B decode attention", "verdict": verdict,
            "geometry": {"Hq": 32, "Hkv": 8, "Hd": 128, "G": 4}, "model": MODEL, "nmeas": NMEAS,
            "all_token_match": all_match, "all_route_bound": all_bound, "per_ctx": per_ctx}
  json.dump(latest, open(OUT / "baseline.json", "w"), indent=2)
  md = [f"# TG-P8.0 8B Attention Baseline\n", f"Verdict: **{verdict}**\n",
        "| ctx | owned tok/s | gen tok/s | % owned | owned attn us/fwd | gen attn us/fwd | token_match | route_bound |",
        "|---|---|---|---|---|---|---|---|"]
  for r in per_ctx:
    md.append(f"| {r['ctx']} | {r['owned_tok_s']} | {r['generated_tok_s']} | {r['pct_of_owned']}% | "
              f"{r['owned_attn_total_us']} | {r['generated_attn_total_us']} | {r['token_match']} | {r['route_bound']} |")
  md += ["", "## Per-kernel attention wall split (us per forward, summed over layers)", ""]
  for r in per_ctx:
    md += [f"### ctx {r['ctx']}", "owned:"]
    for k, v in r["owned_attn_split"].items(): md.append(f"- {k}: {v['total_us_per_forward']}us ({v['occurrences']}x, {v['us_per_occurrence']}us/occ)")
    md.append("generated:")
    for k, v in r["generated_attn_split"].items(): md.append(f"- {k}: {v['total_us_per_forward']}us ({v['occurrences']}x, {v['us_per_occurrence']}us/occ)")
    md.append("")
  open(OUT / "summary.md", "w").write("\n".join(md) + "\n")
  print(verdict, "all_match=", all_match, "all_bound=", all_bound)
  return 0 if verdict == "TG_P8_0_PASS_BASELINE_PINNED" else 1


if __name__ == "__main__":
  raise SystemExit(main())
