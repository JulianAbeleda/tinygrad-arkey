"""Machine-searchable prefill GEMM emit-config search.

Searches the additive emit-config knobs in extra/qk_prefill_graph_gemm_route.py
(PREFILL_GEMM_{DBUF,BK,PLRA,PLRAB,8WAVE,LEANADDR,RELOC,RELOC_MAX_WGS}) by whole-prefill throughput, with repeats +
significance vs the current route default. Infeasible configs (VGPR/LDS/tile overflow) are caught at build time and
reported INFEASIBLE rather than crashing the search. This is the tool that found the DBUF default-flip win
(docs/prefill-structural-emit-search-result-20260623.md): structural emit changes transfer in-model where scheduling
does not, so candidates are ranked on WHOLE-PREFILL (the authority), never isolated kernels.

Usage (driver):
  DEV=AMD JIT=1 PREFILL_V2=1 PYTHONPATH=. .venv/bin/python extra/qk_prefill_emit_search.py \
      [--candidates default|grid] [--spec cfg.json] [--repeats 6] [--contexts 512,1024,2048,4096,8192] \
      [--maxc 8704] [--out /tmp/prefill-emits] [--quick]
Each candidate runs as an isolated subprocess (one model load); the driver aggregates median/mean/std/95%CI, computes
paired-free significance (|Δ|>1% AND |Δ|>2·CI-band AND p<0.05), ranks by whole-prefill@max-ctx, and writes JSON+CSV+MD.
"""
from __future__ import annotations
import os, sys, json, time, subprocess, statistics, math, bisect, csv, datetime, argparse, itertools

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------------------------------------------------
# SEARCH SPACE: the route's additive emit knobs and their candidate domains. Extend here to widen the search.
# ---------------------------------------------------------------------------------------------------------------------
SEARCH_SPACE = {
  "PREFILL_GEMM_DBUF": [0, 1],            # cross-iteration double-buffer (block-level pipeline) -- the validated lever
  "PREFILL_GEMM_PLRA": [0, 1],            # substep A-prefetch (single-buffer); mutually-better-or-worse vs DBUF
  "PREFILL_GEMM_BK": [16, 32],            # DepthU; 64 is VGPR-walled (268>256) -> auto-INFEASIBLE
  "PREFILL_GEMM_8WAVE": [0, 1],           # 8-wave PLRAB+DBUF tile retime (unstable in practice)
  "PREFILL_GEMM_LEANADDR": [0, 1],        # SALU address progression (VGPR-walled at the default tile)
  "PREFILL_GEMM_RELOC": [0, 1],           # waitcnt relocation (occupancy-gated)
  "PREFILL_GEMM_RELOC_MAX_WGS": [1, 4],   # relocation occupancy threshold
}

# Curated named candidates covering the emit classes + the known winner (kept for regression/repro).
DEFAULT_CANDIDATES = [
  ("baseline_current_default", {}),                                                    # = route default (now DBUF)
  ("old_plra",        {"PREFILL_GEMM_DBUF": "0", "PREFILL_GEMM_PLRA": "1"}),            # the pre-flip default
  ("dbuf_only",       {"PREFILL_GEMM_DBUF": "1", "PREFILL_GEMM_PLRA": "0"}),
  ("dbuf_reloc",      {"PREFILL_GEMM_DBUF": "1", "PREFILL_GEMM_PLRA": "0", "PREFILL_GEMM_RELOC": "1", "PREFILL_GEMM_RELOC_MAX_WGS": "4"}),
  ("dbuf_bk16",       {"PREFILL_GEMM_DBUF": "1", "PREFILL_GEMM_PLRA": "0", "PREFILL_GEMM_BK": "16"}),
  ("plra_bk16",       {"PREFILL_GEMM_PLRA": "1", "PREFILL_GEMM_DBUF": "0", "PREFILL_GEMM_BK": "16"}),
  ("eightwave",       {"PREFILL_GEMM_8WAVE": "1"}),
  ("leanaddr",        {"PREFILL_GEMM_LEANADDR": "1"}),
]

def grid_candidates():
  """Enumerate a bounded feasible grid over the highest-leverage knobs (dbuf/plra/bk/reloc/reloc_wgs). Drops the
  obviously-degenerate dbuf==plra==0 and dbuf==plra==1 combos; build-time feasibility filters the rest."""
  out = [("baseline_current_default", {})]
  for dbuf, plra, bk, reloc in itertools.product([0, 1], [0, 1], [16, 32], [0, 1]):
    if dbuf == plra: continue                                # need exactly one of the two prefetch styles
    env = {"PREFILL_GEMM_DBUF": str(dbuf), "PREFILL_GEMM_PLRA": str(plra), "PREFILL_GEMM_BK": str(bk)}
    name = f"d{dbuf}p{plra}bk{bk}"
    if reloc:
      for wgs in (1, 4):
        out.append((f"{name}_reloc{wgs}", {**env, "PREFILL_GEMM_RELOC": "1", "PREFILL_GEMM_RELOC_MAX_WGS": str(wgs)}))
    else:
      out.append((name, env))
  return out

# ---------------------------------------------------------------------------------------------------------------------
# WORKER (subprocess): one model load, measure chunk_ms[sp] across R repeats for the env-selected config.
# Fresh TinyJit per start_pos (attention key-extent is baked at capture). Build failures -> WORKER_FAIL (INFEASIBLE).
# ---------------------------------------------------------------------------------------------------------------------
def run_worker():
  from tinygrad import Tensor, Device, TinyJit
  os.environ.setdefault("PREFILL_V2", "1")
  from extra.llm_generate import load_model_and_tokenizer
  from extra.qk_harness_contract import DEFAULT_MODEL
  MAXC = int(os.environ["SEARCH_MAXC"]); R = int(os.environ["SEARCH_R"])
  SPS = [int(x) for x in os.environ["SEARCH_SPS"].split(",")]
  dev = Device["AMD"]
  try:
    m, tok = load_model_and_tokenizer(DEFAULT_MODEL, MAXC, seed=20260617)
  except Exception as e:
    print("WORKER_FAIL load:" + str(e)[:120]); return
  for b in m.blk: b._use_flash, b._prefill_v2 = True, True
  temp = Tensor([0.0]); N = 512
  chunk = Tensor([[(i*7) % 1000 for i in range(N)]], dtype="int32").contiguous()
  def measure_sp(sp, allow_fail):
    j = TinyJit(m.forward)
    try:
      for _ in range(4): j(chunk, sp, temp).realize()
      dev.synchronize()
    except Exception as e:
      if allow_fail: print("WORKER_FAIL build:" + str(e)[:160]); sys.exit(0)
      raise
    def burst(K=8):
      ts = []
      for _ in range(3):
        dev.synchronize(); t0 = time.perf_counter()
        for _ in range(K): j(chunk, sp, temp).realize()
        dev.synchronize(); ts.append((time.perf_counter()-t0)/K*1e3)
      return min(ts)
    return [burst() for _ in range(R)]
  samples = {SPS[0]: measure_sp(SPS[0], True)}
  for sp in SPS[1:]: samples[sp] = measure_sp(sp, False)
  reps = [{str(sp): samples[sp][r] for sp in SPS} for r in range(R)]
  print("WORKER_JSON " + json.dumps({"sps": SPS, "reps": reps}))

# ---------------------------------------------------------------------------------------------------------------------
# Driver helpers
# ---------------------------------------------------------------------------------------------------------------------
def whole(rep, L):
  pts = sorted(((int(k), v) for k, v in rep.items())); xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
  def interp(s):
    if s <= xs[0]: return ys[0]
    if s >= xs[-1]: return ys[-1]
    i = bisect.bisect_right(xs, s) - 1
    return ys[i] + (ys[i+1]-ys[i]) * (s-xs[i]) / (xs[i+1]-xs[i])
  return L / sum(interp(s) for s in range(0, L, 512)) * 1e3

def stats(xs):
  n = len(xs); sd = statistics.stdev(xs) if n > 1 else 0.0
  return {"n": n, "median": statistics.median(xs), "mean": statistics.mean(xs), "std": sd,
          "ci95": 1.96*sd/math.sqrt(n) if n else 0.0}

def welch_p(a, b):
  if len(a) < 2 or len(b) < 2: return 1.0
  va, vb = statistics.variance(a), statistics.variance(b)
  se = math.sqrt(va/len(a) + vb/len(b))
  if se == 0: return 1.0
  t = abs(statistics.mean(a) - statistics.mean(b)) / se
  return 2 * (1 - 0.5 * (1 + math.erf(t / math.sqrt(2))))

def run_candidate(env, sps, R, maxc, timeout):
  e = dict(os.environ)
  e.update({"DEV": "AMD", "JIT": "1", "PREFILL_V2": "1", "PYTHONPATH": ".", "SEARCH_WORKER": "1",
            "SEARCH_R": str(R), "SEARCH_SPS": ",".join(map(str, sps)), "SEARCH_MAXC": str(maxc)})
  for k in list(e):
    if k.startswith("PREFILL_GEMM_"): del e[k]            # clean slate; candidate sets its own
  e.update(env)
  try:
    p = subprocess.run([sys.executable, os.path.abspath(__file__)], cwd=ROOT, env=e, capture_output=True, text=True, timeout=timeout)
  except subprocess.TimeoutExpired:
    return {"status": "INFEASIBLE", "reason": "timeout"}
  jl = next((l for l in p.stdout.splitlines() if l.startswith("WORKER_JSON")), None)
  if jl is None:
    fl = next((l for l in p.stdout.splitlines() if l.startswith("WORKER_FAIL")), None)
    return {"status": "INFEASIBLE", "reason": (fl or (p.stderr.strip().splitlines() or ["no output"])[-1])[:160]}
  return {"status": "OK", "data": json.loads(jl[len("WORKER_JSON "):])}

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--candidates", choices=["default", "grid"], default="default")
  ap.add_argument("--spec", help="JSON file: [[name, {env}], ...]")
  ap.add_argument("--repeats", type=int, default=6)
  ap.add_argument("--discard", type=int, default=1)
  ap.add_argument("--contexts", default="512,1024,2048,4096,8192")
  ap.add_argument("--maxc", type=int, default=8704)
  ap.add_argument("--timeout", type=int, default=1200)
  ap.add_argument("--out", default="/tmp/prefill-emits")
  ap.add_argument("--quick", action="store_true")
  args = ap.parse_args()
  if args.quick:
    args.repeats, args.contexts, args.maxc = 3, "512,4096", 4608
  CTX = [int(x) for x in args.contexts.split(",")]
  SPS = sorted(set([0] + [sp for L in CTX for sp in range(0, L, 512)]))
  SPS = [sp for sp in SPS if sp < max(CTX)] or [0]
  if args.spec: CANDS = json.load(open(args.spec))
  elif args.candidates == "grid": CANDS = grid_candidates()
  else: CANDS = DEFAULT_CANDIDATES

  os.makedirs(args.out, exist_ok=True)
  results = {}
  for name, env in CANDS:
    sys.stderr.write(f"[search] {name} {env}\n"); sys.stderr.flush()
    results[name] = run_candidate(env, SPS, args.repeats, args.maxc, args.timeout)

  base = results[CANDS[0][0]]
  base_wl = {L: [whole(r, L) for r in base["data"]["reps"][args.discard:]] for L in CTX} if base["status"] == "OK" else None
  rows, ranking = [], []
  for name, env in CANDS:
    res = results[name]
    if res["status"] != "OK":
      rows.append({"candidate": name, "status": res["status"], "reason": res.get("reason"), "env": env}); continue
    wl = {L: [whole(r, L) for r in res["data"]["reps"][args.discard:]] for L in CTX}
    cand = {"candidate": name, "status": "OK", "env": env, "ctx": {}}
    for L in CTX:
      s = stats(wl[L]); row = {k: round(v, 2) for k, v in s.items() if k != "n"}
      if base_wl is not None and name != CANDS[0][0]:
        bs = stats(base_wl[L]); d = (s["median"]/bs["median"]-1)*100
        band = 2*(s["ci95"]+bs["ci95"])/bs["median"]*100; p = welch_p(wl[L], base_wl[L])
        row.update({"delta_pct": round(d, 2), "ci_band_pct": round(band, 2), "p": round(p, 4),
                    "significant": bool(abs(d) > 1.0 and abs(d) > band and p < 0.05)})
      cand["ctx"][L] = row
    rows.append(cand)
    if name != CANDS[0][0] and base_wl is not None:
      Lm = max(CTX); r = cand["ctx"][Lm]
      ranking.append({"candidate": name, "delta_pct@%d" % Lm: r.get("delta_pct", 0.0),
                      "significant": r.get("significant", False), "std": r["std"], "env": env})
  ranking.sort(key=lambda x: (-x["delta_pct@%d" % max(CTX)], x["std"]))

  ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
  summary = {"timestamp": ts, "git": git, "repeats": args.repeats, "discard": args.discard, "contexts": CTX,
             "start_pos": SPS, "baseline": CANDS[0][0], "rows": rows, "ranking": ranking}
  jpath = f"{args.out}/emit-search-{ts}.json"
  with open(jpath, "w") as f: json.dump(summary, f, indent=2)
  with open(f"{args.out}/emit-search-{ts}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["candidate", "ctx", "status", "median", "mean", "std", "ci95", "delta_pct", "p", "significant"])
    for r in rows:
      if r["status"] != "OK": w.writerow([r["candidate"], "", r["status"], r.get("reason")]); continue
      for L, c in r["ctx"].items():
        w.writerow([r["candidate"], L, "OK", c.get("median"), c.get("mean"), c.get("std"), c.get("ci95"),
                    c.get("delta_pct"), c.get("p"), c.get("significant")])
  # markdown
  with open(f"{args.out}/emit-search-{ts}.md", "w") as f:
    f.write(f"# Prefill emit-config search ({ts}, git {git}, {args.repeats} repeats)\n\nBaseline: `{CANDS[0][0]}`\n\n")
    f.write("| candidate | status | " + " | ".join(f"@{L} Δ%" for L in CTX) + " |\n")
    f.write("|---|---|" + "|".join(["---"]*len(CTX)) + "|\n")
    for r in rows:
      if r["status"] != "OK": f.write(f"| {r['candidate']} | {r['status']}: {r.get('reason','')[:40]} | " + " | ".join([""]*len(CTX)) + " |\n"); continue
      cells = []
      for L in CTX:
        c = r["ctx"][L]
        cells.append(f"{c.get('median',0):.0f}" if r["candidate"] == CANDS[0][0]
                     else f"{c.get('delta_pct',0):+.1f}{'*' if c.get('significant') else ''}")
      f.write(f"| {r['candidate']} | {r['status']} | " + " | ".join(cells) + " |\n")
    f.write("\n## Ranking (by Δ% @ max ctx)\n")
    for i, r in enumerate(ranking):
      f.write(f"{i+1}. `{r['candidate']}` {r['delta_pct@%d' % max(CTX)]:+.2f}%{' *SIG*' if r['significant'] else ''} std={r['std']:.1f} env={r['env']}\n")
  print("SEARCH_DONE " + jpath)
  print(json.dumps({"baseline": CANDS[0][0], "ranking": ranking}, indent=2))

if __name__ == "__main__":
  if os.environ.get("SEARCH_WORKER"): run_worker()
  else: main()
