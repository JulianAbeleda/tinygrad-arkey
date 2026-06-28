#!/usr/bin/env python3
"""Route-attributed in-model W==D harness for the generated decode block tile (owned tile = comparator/oracle).

This is the missing measurement primitive for parity closure (scope:
docs/decode-attention-block-tile-route-binding-scope-20260627.md, 2d). It exists because a bare tok/s number was a
phantom: the block-tile flags fell back to owned, so W==D measured the wrong kernel. Built to the repo's
"Harnesses Are Performance Primitives Too" bar (performance-primitive-research-principles.md): it captures workload,
candidate id + primitive class, comparator id + why, exact env, git commit/dirty, hardware, warmup, repeats +
median + spread, a CORRECTNESS gate (in-model token-match vs the comparator), the in-model W==D timing authority,
ROUTE ATTRIBUTION (which decode-attention kernel actually fired), a pass/fail threshold, and a verdict + stop reason.

It measures BOTH routes IN THE SAME SESSION (same clock state) for a valid comparison -- never against a stale
canonical number -- with a fresh model load per route (the KV-cache dtype gate depends on DECODE_ATTN_AMDGCN_TILE).

Emits two artifacts:
  - bench/qk-owned-oracle-parity/route_attribution.json   (consumed by qk_owned_oracle_parity_audit.py route_bound)
  - bench/qk-pure-search-gap/transfer_snapshot_<ts>.json   (harness-measured; replaces the session-reported one)

Run (full):  DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_route_attribution_wd.py
Run (smoke): QK_NMEAS=8 QK_NTOK=8 QK_CKPTS=512 ... (validates artifact shape fast)
"""
from __future__ import annotations
import io, json, os, pathlib, re, statistics, subprocess, sys, time, contextlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LINE = re.compile(r"\*\*\*\s+\S+\s+\d+\s+(.+?)\s+arg\s+\d+\s+mem")
# decode-attention kernel signatures: the generated block tile vs the owned tile vs the gqa_coop fallback.
_ATTN = re.compile(r"(flash_block_tiled_xlane_score_pv_tile_whole_cache\w*|owned_flash_tile_gqa_whole\w*|flash_partial_coop_vec\w*)")

CKPTS = [int(x) for x in os.environ.get("QK_CKPTS", "512,4096").split(",")]
NMEAS = int(os.environ.get("QK_NMEAS", "40"))
NTOK = int(os.environ.get("QK_NTOK", "24"))
MAXC = 4608
WD_PROMOTION_PCT = 90.0
# The full route-binding stack (scope 1). The hybrid guard (model.py) refuses a partial stack, so this is the only
# config that binds the block tile in-model.
CANDIDATE_FLAGS = {
  "DECODE_ATTN_AMDGCN_TILE": "0", "DECODE_ATTN_GENERATED_WHOLECACHE": "1",
  "DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE": "1", "DECODE_ATTN_BLOCK_TILE": "1",
  "DECODE_STAGE_COALESCE": "4", "COALESCED_LOAD_LOWERING": "1", "SCHED_UNROLL": "8",
  "SCHED_LIST": "1", "DECODE_FAST_EXP2": "1",
}
COMPARATOR_FLAGS = {"DECODE_ATTN_AMDGCN_TILE": "1"}   # owned hand-AMDGCN tile = the shipped default / oracle

def _git():
  try:
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    return sha, dirty
  except Exception: return "unknown", None

def _spawn(flags, label):
  """Measure one route in a FRESH SUBPROCESS. Required: tinygrad's getenv memoizes on first read, so two route
  configs cannot coexist in one process (the second silently inherits the first's cached flags -> fallback). Process
  isolation is the only correct way to compare two route configs -- a harness best practice (clean lifecycle)."""
  env = dict(os.environ)
  for k in set(CANDIDATE_FLAGS) | set(COMPARATOR_FLAGS): env.pop(k, None)   # clear all route flags
  env.update({k: str(v) for k, v in flags.items()})                         # set this route's flags
  env["QK_MEASURE_ONE"] = "1"
  p = subprocess.run([sys.executable, str(pathlib.Path(__file__))], env=env, capture_output=True, text=True, cwd=str(ROOT))
  for line in p.stdout.splitlines():
    if line.startswith("@@RESULT@@"):
      rows = json.loads(line[len("@@RESULT@@"):])
      for ck, r in rows.items(): print(f"  [{label}] ctx{ck}: {r['tok_s']} tok/s (+-{r['w_ms_stdev']}ms) attn={r['attn_kernels']}", file=sys.__stderr__)
      return {int(k): v for k, v in rows.items()}
  raise RuntimeError(f"[{label}] measure-one subprocess produced no @@RESULT@@:\n{p.stderr[-2500:]}")

def measure_one():
  """Runs in the isolated subprocess (env flags already set by the parent). Loads the model ONCE, measures all ctx."""
  from extra.qk_harness_contract import DEFAULT_MODEL
  from tinygrad import Tensor, UOp, TinyJit, Context, GlobalCounters
  from extra.llm_generate import load_model_and_tokenizer
  model = os.environ.get("QK_MODEL", DEFAULT_MODEL)
  m, tok = load_model_and_tokenizer(model, MAXC, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 800)
  ids = (ids * (1 + MAXC // max(1, len(ids))))[:MAXC]
  v_sp = UOp.variable("start_pos", 0, MAXC - 1); temp = Tensor([0.0])
  rows = {}
  for ck in CKPTS:
    use_flash = ck >= int(os.environ.get("FLASH_DECODE_THRESHOLD", "512"))
    for b in m.blk: b._use_flash, b._prefill_v2 = use_flash, False
    step = TinyJit(m.forward); tokid = int(ids[ck])
    out = Tensor([[tokid]], dtype="int32").contiguous()
    for i in range(8): out = step(out, v_sp.bind(ck + i), temp).realize()   # warm (compile + clock ramp)
    out = Tensor([[tokid]], dtype="int32").contiguous(); W, toks = [], []
    for i in range(NMEAS):   # W: real decode (.item/token) -- timing authority + greedy tokens together
      t0 = time.perf_counter(); out = step(out, v_sp.bind(ck + i), temp); tid = int(out.item())
      W.append(time.perf_counter() - t0)
      if i < NTOK: toks.append(tid)
    buf = io.StringIO()   # route attribution: an EAGER forward (NOT the jit replay, which shows graph nodes, not
    with contextlib.redirect_stdout(buf), Context(DEBUG=2):   # individual kernel names) so DEBUG=2 emits flash_*/owned_* lines
      GlobalCounters.reset(); m.forward(Tensor([[tokid]], dtype="int32").contiguous(), v_sp.bind(ck), temp).realize()
    kernels = sorted({_ANSI.sub("", l) for l in buf.getvalue().splitlines()})
    attn = sorted({mm.group(1) for l in kernels if (mm := _ATTN.search(l))})
    w_ms = statistics.median(W) * 1e3
    rows[ck] = {"tok_s": round(1000 / w_ms, 1), "w_ms_median": round(w_ms, 3),
                "w_ms_stdev": round(statistics.pstdev(W) * 1e3, 3), "nmeas": NMEAS,
                "tokens": toks, "attn_kernels": attn}
  del m
  print("@@RESULT@@" + json.dumps(rows))   # consumed by the parent _spawn()

def main() -> int:
  if os.environ.get("QK_MEASURE_ONE"):   # isolated child: env flags already set; measure one route and emit
    measure_one(); return 0
  sha, dirty = _git()
  print("== comparator (owned oracle) [subprocess] ==", file=sys.__stderr__)
  comp = _spawn(COMPARATOR_FLAGS, "owned")
  print("== candidate (generated block tile, full stack) [subprocess] ==", file=sys.__stderr__)
  cand = _spawn(CANDIDATE_FLAGS, "block_tile")

  per_ctx, all_bound, all_match = [], True, True
  for ck in CKPTS:
    c, o = cand[ck], comp[ck]
    bound = any("flash_block_tiled" in k for k in c["attn_kernels"]) and not any("owned_flash_tile" in k for k in c["attn_kernels"])
    match = c["tokens"] == o["tokens"]
    all_bound &= bound; all_match &= match
    per_ctx.append({"ctx": ck, "candidate_tok_s": c["tok_s"], "owned_tok_s": o["tok_s"],
                    "pct_of_owned": round(c["tok_s"] / o["tok_s"] * 100.0, 1) if o["tok_s"] else None,
                    "candidate_attn_kernels": c["attn_kernels"], "owned_attn_kernels": o["attn_kernels"],
                    "route_bound": bound, "token_match": match,
                    "candidate_w_ms_stdev": c["w_ms_stdev"], "owned_w_ms_stdev": o["w_ms_stdev"]})
  pct4096 = next((r["pct_of_owned"] for r in per_ctx if r["ctx"] == max(CKPTS)), None)
  promotable = all_bound and all_match and (pct4096 or 0) >= WD_PROMOTION_PCT
  if not all_bound: verdict = "NOT_ROUTE_BOUND__BLOCK_TILE_DID_NOT_FIRE"; stop = "route attribution failed; fix flag-contract/route-binding before any W==D claim"
  elif not all_match: verdict = "ROUTE_BOUND_BUT_TOKEN_MISMATCH"; stop = "block tile is route-bound but produces wrong tokens; correctness bug (dtype/cache/layout)"
  elif promotable: verdict = "ROUTE_BOUND__TOKEN_MATCH__WD_AT_THRESHOLD"; stop = "candidate route-bound + correct + W==D>=threshold -> promotion review"
  else: verdict = "ROUTE_BOUND__TOKEN_MATCH__WD_BELOW_THRESHOLD"; stop = f"honest in-model gap: {pct4096}% of owned @ctx{max(CKPTS)} (<{WD_PROMOTION_PCT}%)"

  ts = time.strftime("%Y%m%d-%H%M%S")
  art = {
    "schema": "qk_decode_route_attribution_wd_v1", "timestamp": ts,
    "git_commit": sha, "git_dirty": dirty, "hardware": "RX 7900 XTX / gfx1100",
    "model_id": pathlib.Path(os.environ.get("QK_MODEL", "")).stem or "Qwen3-8B-Q4_K_M",
    "timing_authority": "in_model_W_equals_D", "comparator": "owned_flash_tile_gqa_whole (shipped default / oracle)",
    "candidate_primitive_class": "llama-style generated decode-attention block tile (whole-cache fused-xlane)",
    "candidate_env_flags": CANDIDATE_FLAGS, "comparator_env_flags": COMPARATOR_FLAGS,
    "workload": {"ckpts": CKPTS, "maxc": MAXC, "nmeas": NMEAS, "ntok": NTOK, "warmup_steps": 8},
    "wd_promotion_threshold_pct_of_owned": WD_PROMOTION_PCT,
    "route_bound": all_bound, "token_match": all_match, "per_ctx": per_ctx,
    "verdict": verdict, "stop_reason": stop,
  }
  # 1) route_attribution.json -- exactly what the parity precheck consumes
  ra_dir = ROOT / "bench/qk-owned-oracle-parity"; ra_dir.mkdir(parents=True, exist_ok=True)
  (ra_dir / "route_attribution.json").write_text(json.dumps({
    "schema": "qk_route_attribution_v1", "timestamp": ts, "git_commit": sha,
    "route_bound": all_bound, "token_match": all_match,
    "kernels": sorted({k for r in per_ctx for k in r["candidate_attn_kernels"]}),
    "flags": CANDIDATE_FLAGS, "per_ctx": [{"ctx": r["ctx"], "route_bound": r["route_bound"],
      "token_match": r["token_match"], "candidate_tok_s": r["candidate_tok_s"]} for r in per_ctx],
    "source": "harness-measured (qk_decode_route_attribution_wd.py)",
  }, indent=2) + "\n")
  # 2) harness-measured transfer snapshot -- replaces the session-reported one (generator globs the latest)
  snap_dir = ROOT / "bench/qk-pure-search-gap"; snap_dir.mkdir(parents=True, exist_ok=True)
  (snap_dir / f"transfer_snapshot_{ts}.json").write_text(json.dumps({
    "schema": "qk_decode_attention_pure_search_transfer_snapshot_v1", "timestamp": ts,
    "source": "harness-measured (qk_decode_route_attribution_wd.py)", "authority": "harness_measured",
    "wd_authority": "harness_measured_w_equals_d", "git_commit": sha, "route_bound": all_bound, "token_match": all_match,
    "arms": [
      {"arm": "owned_baseline", "ctx512_tok_s": comp.get(512, {}).get("tok_s"), "ctx4096_tok_s": comp.get(4096, {}).get("tok_s"), "provenance": "owned_hand_asm_default"},
      {"arm": "block_tile_route_full_stack", "ctx512_tok_s": cand.get(512, {}).get("tok_s"), "ctx4096_tok_s": cand.get(4096, {}).get("tok_s"), "provenance": "generated_route_bound_harness_measured"},
    ],
    "stack_flags": [f"{k}={v}" for k, v in CANDIDATE_FLAGS.items()],
  }, indent=2) + "\n")
  (ra_dir / "wd_attribution_full.json").write_text(json.dumps(art, indent=2) + "\n")
  print(json.dumps({k: art[k] for k in ("route_bound", "token_match", "verdict", "stop_reason")}, indent=2))
  print(f"\nartifacts: bench/qk-owned-oracle-parity/route_attribution.json + bench/qk-pure-search-gap/transfer_snapshot_{ts}.json", file=sys.__stderr__)
  return 0 if (all_bound and all_match) else 1

if __name__ == "__main__":
  raise SystemExit(main())
