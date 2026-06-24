"""Machine-searchable prefill GEMM emit-config search.

Searches the additive emit-config knobs in extra/qk_prefill_graph_gemm_route.py
(PREFILL_GEMM_{DBUF,BK,PLRA,PLRAB,8WAVE,LEANADDR,PIPELINE,PIPELINE_TM,PIPELINE_TN,RELOC,RELOC_MAX_WGS})
by whole-prefill throughput, with repeats +
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
  "PREFILL_GEMM_PAD": [0, 16, 32],
  "PREFILL_GEMM_PIPELINE": [0, 1],        # structural emit frontier probe (non-LDS, 1WG)
  "PREFILL_GEMM_PIPELINE_TM": [2, 4],     # PIPELINE tile m-dimension (16*TM, 16*TN threads)
  "PREFILL_GEMM_PIPELINE_TN": [2, 4],     # PIPELINE tile n-dimension
  "PREFILL_GEMM_RELOC": [0, 1],           # waitcnt relocation (occupancy-gated)
  "PREFILL_GEMM_RELOC_MAX_WGS": [1, 4],   # relocation occupancy threshold
}

# Curated named candidates covering the emit classes + the known winner (kept for regression/repro).
DEFAULT_CANDIDATES = [
  ("baseline_current_default", {}),                                                    # = route default (now DBUF)
  ("old_plra",        {"PREFILL_GEMM_DBUF": "0", "PREFILL_GEMM_PLRA": "1"}),            # the pre-flip default
  ("dbuf_only",       {"PREFILL_GEMM_DBUF": "1", "PREFILL_GEMM_PLRA": "0"}),
  ("dbuf_reloc",      {"PREFILL_GEMM_DBUF": "1", "PREFILL_GEMM_PLRA": "0", "PREFILL_GEMM_RELOC": "1", "PREFILL_GEMM_RELOC_MAX_WGS": "1"}),
  ("dbuf_reloc_wgs4", {"PREFILL_GEMM_DBUF": "1", "PREFILL_GEMM_PLRA": "0", "PREFILL_GEMM_RELOC": "1", "PREFILL_GEMM_RELOC_MAX_WGS": "4"}),
  ("dbuf_bk16",       {"PREFILL_GEMM_DBUF": "1", "PREFILL_GEMM_PLRA": "0", "PREFILL_GEMM_BK": "16"}),
  ("dbuf_pad0",       {"PREFILL_GEMM_DBUF": "1", "PREFILL_GEMM_PLRA": "0", "PREFILL_GEMM_PAD": "0"}),
  ("dbuf_pad32",      {"PREFILL_GEMM_DBUF": "1", "PREFILL_GEMM_PLRA": "0", "PREFILL_GEMM_PAD": "32"}),
  ("plra_bk16",       {"PREFILL_GEMM_PLRA": "1", "PREFILL_GEMM_DBUF": "0", "PREFILL_GEMM_BK": "16"}),
  ("eightwave",       {"PREFILL_GEMM_8WAVE": "1"}),
  ("leanaddr",        {"PREFILL_GEMM_LEANADDR": "1"}),
  ("pipe_tm2_tn2",    {"PREFILL_GEMM_PIPELINE": "1", "PREFILL_GEMM_PIPELINE_TM": "2", "PREFILL_GEMM_PIPELINE_TN": "2"}),
  ("pipe_tm2_tn4",    {"PREFILL_GEMM_PIPELINE": "1", "PREFILL_GEMM_PIPELINE_TM": "2", "PREFILL_GEMM_PIPELINE_TN": "4"}),
  ("pipe_tm4_tn2",    {"PREFILL_GEMM_PIPELINE": "1", "PREFILL_GEMM_PIPELINE_TM": "4", "PREFILL_GEMM_PIPELINE_TN": "2"}),
  ("pipe_tm4_tn4",    {"PREFILL_GEMM_PIPELINE": "1", "PREFILL_GEMM_PIPELINE_TM": "4", "PREFILL_GEMM_PIPELINE_TN": "4"}),
]

def grid_candidates():
  """Enumerate a bounded feasible grid over the highest-leverage knobs:
    - LDS-structured path: dbuf/plra/bk/reloc/reloc_wgs (dominant, known-good region)
    - PIPELINE path: tm/tn structural emit frontier probe (non-default, low-risk exploratory).
  Drops the obviously-degenerate dbuf==plra==0 and dbuf==plra==1 combos; build-time feasibility filters the rest."""
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
  # Structural PIPELINE frontier: single-wave emit; not combinable with LDS-specific dbuf/plra knobs.
  for tm, tn in itertools.product([2, 4], [2, 4]):
    out.append((f"pipe_tm{tm}_tn{tn}", {
      "PREFILL_GEMM_PIPELINE": "1",
      "PREFILL_GEMM_PIPELINE_TM": str(tm),
      "PREFILL_GEMM_PIPELINE_TN": str(tn),
    }))
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
def _to_float(v, default=0.0):
  try:
    return float(v)
  except (TypeError, ValueError):
    return default


def _coerce_spec(spec_path):
  """Read --spec and guarantee a stable baseline-first candidate contract."""
  raw = json.load(open(spec_path))
  if not isinstance(raw, list):
    raise ValueError(f"spec must be a JSON list, got {type(raw).__name__}")

  out = []
  for i, row in enumerate(raw):
    if not isinstance(row, (list, tuple)) or len(row) != 2:
      raise ValueError(f"spec row {i} must be [name, env], got {repr(row)}")
    name, env = row
    if not isinstance(name, str):
      raise ValueError(f"spec row {i} name must be string, got {type(name).__name__}")
    if not isinstance(env, dict):
      raise ValueError(f"spec row {i} env must be object/dict, got {type(env).__name__}")
    out.append((name, _normalize_candidate_env(env)))

  if not out:
    raise ValueError("spec is empty: provide at least baseline and one candidate")

  for idx, (name, _) in enumerate(out):
    if name == "baseline_current_default":
      if idx == 0:
        return out
      baseline = out.pop(idx)
      out.insert(0, baseline)
      print("[guardrail] Re-ordered --spec so baseline_current_default is first candidate", file=sys.stderr)
      return out

  out.insert(0, ("baseline_current_default", {}))
  print("[guardrail] Injected baseline_current_default={} as candidate #1 because spec lacked an explicit baseline", file=sys.stderr)
  return out


def _normalize_env_bool(v, default="0"):
  if isinstance(v, bool): return "1" if v else "0"
  if v is None: return default
  return "1" if str(v).strip().lower() in {"1", "true", "yes", "on"} else "0"

def _normalize_candidate_env(env):
  out = {}
  for k, v in env.items():
    if not k.startswith("PREFILL_GEMM_"): 
      continue
    if k in ("PREFILL_GEMM_RELOC", "PREFILL_GEMM_PIPELINE", "PREFILL_GEMM_DBUF", "PREFILL_GEMM_PLRA", "PREFILL_GEMM_8WAVE", "PREFILL_GEMM_LEANADDR", "PREFILL_GEMM_PLRAB"):
      out[k] = _normalize_env_bool(v, "0")
    else:
      out[k] = str(v)
  return out

def _candidate_class(name, env):
  if str(env.get("PREFILL_GEMM_PIPELINE", "0")).lower() in {"1", "true", "yes", "on"}:
    return "pipeline"
  if env.get("PREFILL_GEMM_8WAVE", "0") == "1":
    return "legacy_8wave"
  if env.get("PREFILL_GEMM_LEANADDR", "0") == "1":
    return "legacy_leanaddr"
  if env.get("PREFILL_GEMM_PAD", "16") not in {"", "16"}:
    return "pad_tuning"
  return "lds_structural"

def _risk_score(category):
  if category == "pipeline":
    return 3
  if category in {"legacy_8wave", "legacy_leanaddr"}:
    return 2
  if category == "pad_tuning":
    return 1
  return 0

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

def _bh_fdr(pvals, alpha):
  if not pvals: return {}
  vals = sorted((v, i) for i, v in enumerate(pvals))
  n = len(vals)
  cutoff_idx = -1
  for rank, (_, i) in enumerate(vals, start=1):
    _, pv = vals[rank-1]
    if pv <= alpha * rank / n:
      cutoff_idx = rank
  if cutoff_idx == -1:
    return {i: False for _, i in vals}
  threshold = alpha * cutoff_idx / n
  return {i: (p <= threshold) for p, i in vals}

def _corr_alpha(alpha, corr_mode, candidates_n):
  if candidates_n <= 1:
    return alpha
  if corr_mode == "bonferroni":
    return max(1e-12, alpha / candidates_n)
  return alpha

def _eval_candidate_ctx(cand_samples, base_samples):
  if not cand_samples or not base_samples or len(cand_samples) < 2 or len(base_samples) < 2:
    return None
  cs = stats(cand_samples); bs = stats(base_samples)
  if bs["median"] <= 0:
    return None
  if any(not math.isfinite(x) for x in [cs["median"], cs["mean"], bs["median"], bs["mean"], cs["ci95"], bs["ci95"]]):
    return None
  delta = (cs["median"] / bs["median"] - 1) * 100
  band = 2 * (cs["ci95"] + bs["ci95"]) / bs["median"] * 100
  p = welch_p(cand_samples, base_samples)
  return {
    **{k: round(v, 2) for k, v in cs.items() if k != "n"},
    "delta_pct": round(delta, 2),
    "ci_band_pct": round(band, 2),
    "p": round(p, 4),
    "abs_delta": abs(delta),
    "raw_delta": delta,
    "raw_band": band,
    "raw_p": p,
    "significant": False,
  }

def _passes_filters(ctx_rows, args, candidate_class):
  ordered_keys = sorted(ctx_rows)
  deltas = [ctx_rows[k].get("raw_delta") for k in ordered_keys if isinstance(ctx_rows[k], dict)]
  if not deltas:
    return {"feasible": False, "reason": "no_ctx"}
  if any(d is None for d in deltas):
    return {"feasible": False, "reason": "bad_ctx"}
  ctx_count = len(ctx_rows)
  max_ctx = max(ctx_rows)
  max_d = ctx_rows[max_ctx]["raw_delta"]
  min_allowed_regress = -abs(_to_float(args.max_regress, 0.5))
  if args.strict:
    min_ctx_pos = max(1, math.ceil(ctx_count * 0.75))
    if candidate_class == "pipeline":
      min_ctx_pos = math.ceil(ctx_count * 0.67)
  else:
    min_ctx_pos = max(2, math.ceil(ctx_count * 0.4))
    if min_ctx_pos > ctx_count:
      min_ctx_pos = ctx_count
  same_sign = sum(1 for d in deltas if d >= 0)
  no_large_drop = min(deltas) >= min_allowed_regress
  max_ctx_improves = max_d >= _to_float(args.min_max_ctx_delta, 0.0)
  # candidate should not be unstable under strict mode
  stable = (max(deltas) - min(deltas)) <= abs(_to_float(args.max_ctx_span, 3.0))
  if args.strict:
    min_delta = 1.5 if candidate_class == "pipeline" else 1.0
  else:
    min_delta = 0.75
  max_ctx_significant = ctx_rows[max_ctx].get("significant", False)
  max_ctx_positive = ctx_rows[max_ctx].get("raw_delta", 0.0) > 0
  max_ctx_min_delta = ctx_rows[max_ctx].get("abs_delta", 0.0) >= min_delta
  max_ctx_ok = max_ctx_positive and (max_ctx_min_delta if args.strict else True)
  max_ctx_sig = max_ctx_significant if args.strict else max_ctx_positive
  return {
    "feasible": True,
    "passes_ctx_stability": same_sign >= min_ctx_pos,
    "passes_max_ctx": bool(max_ctx_improves),
    "passes_regress_cap": bool(no_large_drop),
    "passes_ctx_span": bool(stable),
    "passes_strict_delta": bool(max_d >= _to_float(args.min_max_ctx_delta, 0.0) or not args.strict),
    "passes_min_delta_max_ctx": bool(max_ctx_min_delta),
    "passes_signif_at_max": bool(max_ctx_sig),
    "passes": same_sign >= min_ctx_pos and no_large_drop and max_ctx_improves and max_ctx_ok and (stable if args.strict else True) and bool(max_ctx_sig if args.strict else True),
    "min_ctx_pos": min_ctx_pos,
    "max_ctx_delta": max_d,
    "min_delta_thresh": min_delta,
    "same_sign": same_sign,
  }

def _pairwise_confirm(base_env, cand_env, sps, repeats, maxc, timeout):
  base = run_candidate(base_env, sps, repeats, maxc, timeout)
  cand = run_candidate(cand_env, sps, repeats, maxc, timeout)
  if base.get("status") != "OK" or cand.get("status") != "OK":
    return None
  return {
    "base": base.get("data", {}).get("reps", []),
    "cand": cand.get("data", {}).get("reps", []),
    "status": "OK",
  }

def _collect_ctx_wl(res, ctxs, discard):
  reps = res.get("data", {}).get("reps")
  if not reps:
    return None
  usable = reps[discard:]
  if len(usable) < 2:
    return None
  try:
    return {L: [whole(r, L) for r in usable] for L in ctxs}
  except Exception:
    return None

def run_candidate(env, sps, R, maxc, timeout):
  e = dict(os.environ)
  e.update({"DEV": "AMD", "JIT": "1", "PREFILL_V2": "1", "PYTHONPATH": ".", "SEARCH_WORKER": "1",
            "SEARCH_R": str(R), "SEARCH_SPS": ",".join(map(str, sps)), "SEARCH_MAXC": str(maxc)})
  for k in list(e):
    if k.startswith("PREFILL_GEMM_"): del e[k]            # clean slate; candidate sets its own
  e.update(_normalize_candidate_env(env))
  try:
    p = subprocess.run([sys.executable, os.path.abspath(__file__)], cwd=ROOT, env=e, capture_output=True, text=True, timeout=timeout)
  except subprocess.TimeoutExpired:
    return {"status": "INFEASIBLE", "reason": "timeout", "returncode": None, "stdout_tail": "", "stderr_tail": ""}

  out_lines = [l for l in (p.stdout or "").splitlines() if l]
  err_lines = [l for l in (p.stderr or "").splitlines() if l]
  jl = next((l for l in p.stdout.splitlines() if l.startswith("WORKER_JSON")), None)
  if jl is None:
    fl = next((l for l in p.stdout.splitlines() if l.startswith("WORKER_FAIL")), None)
    if fl is None and p.returncode == 0 and err_lines:
      fl = err_lines[-1]
    if fl is None:
      if p.returncode != 0:
        fl = f"returncode={p.returncode}"
      elif not out_lines and not err_lines:
        fl = "no worker output"
      else:
        fl = "worker output parse failed"
    return {
      "status": "INFEASIBLE",
      "reason": fl[:220],
      "returncode": p.returncode,
      "stdout_tail": "\\n".join(out_lines[-3:]),
      "stderr_tail": "\\n".join(err_lines[-3:]),
    }
  try:
    data = json.loads(jl[len("WORKER_JSON "):])
  except json.JSONDecodeError as e:
    return {
      "status": "INFEASIBLE",
      "reason": f"bad_worker_json:{str(e)}",
      "returncode": p.returncode,
      "stdout_tail": "\\n".join(out_lines[-3:]),
      "stderr_tail": "\\n".join(err_lines[-3:]),
    }
  return {"status": "OK", "data": data}

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
  ap.add_argument("--strict", action="store_true")
  ap.add_argument("--corr-mode", choices=["none", "bonferroni", "fdr"], default="none")
  ap.add_argument("--alpha", type=float, default=0.05)
  ap.add_argument("--min-max-ctx-delta", type=float, default=0.0)
  ap.add_argument("--max-regress", type=float, default=0.5)
  ap.add_argument("--max-ctx-span", type=float, default=3.0)
  ap.add_argument("--confirm-k", type=int, default=3)
  ap.add_argument("--confirm-repeats", type=int, default=10)
  ap.add_argument("--confirm-timeout", type=int, default=1800)
  args = ap.parse_args()
  if args.quick:
    args.repeats, args.contexts, args.maxc = 3, "512,4096", 4608
  CTX = sorted(set(int(x) for x in args.contexts.split(",")))
  SPS = sorted(set([0] + [sp for L in CTX for sp in range(0, L, 512)]))
  SPS = [sp for sp in SPS if sp < max(CTX)] or [0]
  if args.spec:
    try:
      CANDS = _coerce_spec(args.spec)
    except Exception as e:
      raise SystemExit(f"[spec] {str(e)}")
  elif args.candidates == "grid":
    CANDS = grid_candidates()
  else:
    CANDS = DEFAULT_CANDIDATES
  CANDS = [(name, _normalize_candidate_env(env)) for name, env in CANDS]

  if args.confirm_k > 0 and len(CANDS) < 2:
    print("[guardrail] --confirm-k requested but no comparator candidates after baseline normalization; forcing --confirm-k 0", file=sys.stderr)
    args.confirm_k = 0

  os.makedirs(args.out, exist_ok=True)
  results = {}
  for name, env in CANDS:
    sys.stderr.write(f"[search] {name} {env}\n"); sys.stderr.flush()
    results[name] = run_candidate(env, SPS, args.repeats, args.maxc, args.timeout)

  baseline_name = CANDS[0][0]
  baseline_env = dict(CANDS[0][1])
  base_res = results.get(baseline_name)
  base_wl = _collect_ctx_wl(base_res, CTX, args.discard) if base_res and base_res.get("status") == "OK" else None
  if base_wl is None:
    if base_res:
      reason = base_res.get("reason", "unknown")
      rc = base_res.get("returncode", "n/a")
      print("BASELINE_INFEASIBLE", reason, f"returncode={rc}")
      if base_res.get("stderr_tail"):
        print("BASELINE_ERROR", base_res["stderr_tail"])
    else:
      print("BASELINE_INFEASIBLE missing baseline result")
    return

  rows = []
  for name, env in CANDS:
    res = results[name]
    if res.get("status") != "OK":
      rows.append({"candidate": name, "status": res.get("status"), "reason": res.get("reason"), "env": env, "class": "n/a", "risk_score": 99})
      continue
    wl = _collect_ctx_wl(res, CTX, args.discard)
    if wl is None:
      rows.append({"candidate": name, "status": "INFEASIBLE", "reason": "bad_workload", "env": env, "class": "n/a", "risk_score": 99})
      continue
    cclass = _candidate_class(name, env)
    cand_row = {"candidate": name, "status": "OK", "env": env, "class": cclass, "risk_score": _risk_score(cclass), "ctx": {}}
    if name == baseline_name:
      for L in CTX:
        cand_row["ctx"][L] = {k: round(v, 2) for k, v in stats(wl[L]).items() if k != "n"}
      cand_row["decision"] = "baseline"
      rows.append(cand_row)
      continue
    for L in CTX:
      ctx_row = _eval_candidate_ctx(wl[L], base_wl[L])
      if ctx_row is None:
        cand_row["status"] = "INFEASIBLE"
        cand_row["reason"] = "ctx_eval_failed"
        cand_row["ctx"] = {}
        break
      cand_row["ctx"][L] = ctx_row
    if cand_row["status"] != "OK":
      rows.append(cand_row); continue
    rows.append(cand_row)

  n_compare = max(1, len([r for r in rows if r["status"] == "OK" and r["candidate"] != baseline_name and r.get("ctx")]))
  alpha_bonf = _corr_alpha(args.alpha, "bonferroni", n_compare)
  fdr_keep = {}
  if args.corr_mode == "fdr":
    compare_indices = [idx for idx, r in enumerate(rows) if r["status"] == "OK" and r["candidate"] != baseline_name and r.get("ctx")]
    for L in CTX:
      ctx_indices = [idx for idx in compare_indices if L in rows[idx].get("ctx", {})]
      pvals = [rows[idx]["ctx"][L]["raw_p"] for idx in ctx_indices]
      keep = _bh_fdr(pvals, args.alpha)
      for pos, idx in enumerate(ctx_indices):
        fdr_keep[(idx, L)] = bool(keep.get(pos, False))

  for r_index, r in enumerate(rows):
    if r.get("status") != "OK" or r["candidate"] == baseline_name:
      continue
    if not r.get("ctx"):
      r["filters"] = {"feasible": False, "reason": "no_ctx"}
      r["decision"] = "rejected"
      continue
    for L in CTX:
      if L not in r["ctx"]:
        continue
      cctx = r["ctx"][L]
      min_delta = 1.5 if (r["class"] == "pipeline" and args.strict) else (1.0 if args.strict else 0.75)
      if args.corr_mode == "bonferroni":
        alpha_ok = cctx["raw_p"] <= alpha_bonf
      elif args.corr_mode == "fdr":
        alpha_ok = bool(fdr_keep.get((r_index, L), False))
      else:
        alpha_ok = cctx["raw_p"] <= args.alpha
      cctx["significant"] = bool(abs(cctx["raw_delta"]) >= min_delta and abs(cctx["raw_delta"]) > cctx["raw_band"] and alpha_ok)
      cctx["alpha"] = alpha_bonf if args.corr_mode == "bonferroni" else args.alpha
      cctx["min_delta_used"] = min_delta
    r["filters"] = _passes_filters(r["ctx"], args, r["class"])
    if r["filters"].get("passes", False):
      r["decision"] = "needs_confirm" if args.strict else "pass"
    else:
      r["decision"] = "needs_review" if args.strict else "rejected"

  ranking = []
  for r in rows:
    if r.get("status") != "OK" or r["candidate"] == baseline_name:
      continue
    max_ctx = max(CTX)
    if max_ctx not in r.get("ctx", {}):
      continue
    rc = r["ctx"][max_ctx]
    ranking.append({
      "candidate": r["candidate"],
      "delta_pct@%d" % max_ctx: rc.get("delta_pct", 0.0),
      "significant": rc.get("significant", False),
      "passes": r.get("filters", {}).get("passes", False),
      "decision": r.get("decision", "unknown"),
      "risk_score": r["risk_score"],
      "std": rc.get("std", 0.0),
      "env": r["env"]
    })
  ranking.sort(key=lambda x: (-x["delta_pct@%d" % max(CTX)], x["risk_score"], x["std"]))

  if args.strict and args.confirm_k > 0:
    confirm_pool = [r for r in rows if r.get("status") == "OK" and r["candidate"] != baseline_name and r.get("filters", {}).get("passes", False)]
    confirm_pool.sort(key=lambda r: (r["ctx"][max(CTX)]["delta_pct"] if r.get("ctx") else -1e9), reverse=True)
    for cand in confirm_pool[: args.confirm_k]:
      sys.stderr.write(f"[confirm] {cand['candidate']}\n"); sys.stderr.flush()
      c2 = _pairwise_confirm(baseline_env, cand["env"], SPS, args.confirm_repeats, args.maxc, args.confirm_timeout)
      if c2 is None:
        cand["confirm"] = {"status": "INFEASIBLE", "reason": "run_failed"}
        cand["decision"] = "rejected"
        continue
      cbase_wl = _collect_ctx_wl({"data": {"reps": c2["base"]}}, CTX, 0)
      ccand_wl = _collect_ctx_wl({"data": {"reps": c2["cand"]}}, CTX, 0)
      if cbase_wl is None or ccand_wl is None:
        cand["confirm"] = {"status": "INFEASIBLE", "reason": "confirm_workload"}
        cand["decision"] = "rejected"
        continue
      cctx = {}
      for L in CTX:
        cr = _eval_candidate_ctx(ccand_wl[L], cbase_wl[L])
        if cr is None:
          continue
        min_delta = 1.0 if args.strict else 0.75
        cr["significant"] = bool(abs(cr["raw_delta"]) >= min_delta and abs(cr["raw_delta"]) > cr["raw_band"] and cr["raw_p"] <= args.alpha)
        cctx[L] = cr
      if not cctx:
        cand["confirm"] = {"status": "INFEASIBLE", "reason": "confirm_no_ctx"}
        cand["decision"] = "rejected"
        continue
      cand["confirm"] = {"status": "OK", "ctx": cctx, "filters": _passes_filters(cctx, args, cand["class"])}
      cand["decision"] = "confirmed" if cand["confirm"]["filters"].get("passes", False) else "rejected"

  ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
  summary = {
    "timestamp": ts,
    "git": git,
    "repeats": args.repeats,
    "discard": args.discard,
    "contexts": CTX,
    "start_pos": SPS,
    "baseline": baseline_name,
    "baseline_env": baseline_env,
    "search": {"strict": args.strict, "corr": args.corr_mode, "alpha": args.alpha},
    "rows": rows,
    "ranking": ranking,
  }
  jpath = f"{args.out}/emit-search-{ts}.json"
  with open(jpath, "w") as f:
    json.dump(summary, f, indent=2)
  with open(f"{args.out}/emit-search-{ts}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["candidate", "class", "decision", "status", "ctx", "median", "mean", "std", "ci95", "delta_pct", "p", "significant", "passes"])
    for r in rows:
      if r.get("status") != "OK" or not r.get("ctx"):
        w.writerow([r["candidate"], r.get("class"), r.get("decision", r.get("status")), r.get("status"), "", "", "", "", "", "", "", ""])
        continue
      for L, c in r["ctx"].items():
        w.writerow([r["candidate"], r.get("class"), r.get("decision", ""), r.get("status"), L, c.get("median"), c.get("mean"), c.get("std"), c.get("ci95"),
                    c.get("delta_pct"), c.get("p"), c.get("significant"), r.get("filters", {}).get("passes", False)])
  with open(f"{args.out}/emit-search-{ts}.md", "w") as f:
    f.write(f"# Prefill emit-config search ({ts}, git {git}, {args.repeats} repeats)\n\nBaseline: `{baseline_name}`\n\n")
    f.write("| candidate | class | decision | status | " + " | ".join(f"@{L} Δ%" for L in CTX) + " |\n")
    f.write("|---|---|---|---|" + "|".join(["---"]*len(CTX)) + "|\n")
    for r in rows:
      if r.get("status") != "OK" or not r.get("ctx"):
        f.write(f"| {r['candidate']} | {r.get('class','n/a')} | {r.get('decision', r.get('status'))} | {r['status']} | " + " | ".join([""]*len(CTX)) + " |\n")
        continue
      cells = []
      for L in CTX:
        c = r["ctx"][L]
        cells.append(f"{c.get('median',0):.0f}" if r["candidate"] == baseline_name else f"{c.get('delta_pct',0):+.1f}{'*' if c.get('significant') else ''}")
      f.write(f"| {r['candidate']} | {r.get('class','n/a')} | {r.get('decision','ok')} | {r['status']} | " + " | ".join(cells) + " |\n")
    f.write("\n## Ranking (by Δ% @ max ctx)\n")
    for i, r in enumerate(ranking):
      f.write(f"{i+1}. `{r['candidate']}` {r['delta_pct@%d' % max(CTX)]:+.2f}%{' *SIG*' if r['significant'] else ''} pass={r['passes']} risk={r['risk_score']} env={r['env']} decision={r['decision']}\n")
  print("SEARCH_DONE " + jpath)
  print(json.dumps({"baseline": baseline_name, "ranking": ranking}, indent=2))

if __name__ == "__main__":
  if os.environ.get("SEARCH_WORKER"): run_worker()
  else: main()
