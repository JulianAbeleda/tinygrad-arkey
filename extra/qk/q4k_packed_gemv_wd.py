#!/usr/bin/env python3
"""P2.3/M-E W==D for Q4_K scheduler/lane-partition GEMV vs the owned warp GEMV.

Clock-pinned, in-process interleaved, real per-token .item() W==D, tokens_match.  FFN gate/up only.
This is the decision gate from docs/layout-codegen-full-scope-20260625.md:
  - >=90% of owned with tokens_match -> proceed to P3/search generalization.
  - plateau near the historical ~50 tok/s scheduler ceiling -> stop; CUSTOM remains needed for this GEMV target.

  run: DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk/q4k_packed_gemv_wd.py
"""
from __future__ import annotations
import json, os, pathlib, statistics, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-scheduler-gemv-vs-owned"
DOCS = ROOT / "docs"
CTXS = [512, 1024, 2048, 4096]; MAXC = 4608; NMEAS = 30; REPEATS = 3; PROCEED_RATIO = 0.90
ARMS = {
  "owned":          {"Q4K_GEMV_SCHEDULER": "0"},
  "sched_fp16":     {"Q4K_GEMV_SCHEDULER": "1", "MV_ROWS_PER_THREAD": "1"},
  "sched_packed":   {"Q4K_GEMV_SCHEDULER": "2"},
  "generated_skeleton": {"Q4K_GEMV_SCHEDULER": "2"},
  "sched_wordlane": {"Q4K_GEMV_SCHEDULER": "3"},
  "g2_lanemap": {"Q4K_GEMV_SCHEDULER": "5"},
  "g3_lanemap_codegen": {"Q4K_GEMV_SCHEDULER": "6"},
  "lane_partition": {"Q4K_GEMV_SCHEDULER": "4"},
  "bubblebeam_futuresight":   {"BUBBLEBEAM_FUTURESIGHT": "1"},
}
CLEAR_ENV = ("Q4K_GEMV_SCHEDULER", "MV_ROWS_PER_THREAD", "WARP_REDUCE_LOWERING", "BUBBLEBEAM_FUTURESIGHT")

def _try_pin_clock():
  try:
    subprocess.run(["rocm-smi", "--setperflevel", "high"], capture_output=True, timeout=20)
    out = subprocess.run(["rocm-smi", "--showperflevel"], capture_output=True, timeout=20, text=True)
    for ln in out.stdout.splitlines():
      if "GPU" in ln and "Performance Level:" in ln: return ln.split("Performance Level:")[-1].strip()
  except Exception: pass
  return "unknown"

def _program_counts(step) -> dict[str, int]:
  names = [u.src[0].arg.name for u in step.captured.linear.toposort()
           if u.op.name == "CALL" and len(u.src) and u.src[0].op.name == "PROGRAM"]
  return {
    "owned_gateup": sum(n.startswith("q4k_gemv_warp_12288") for n in names),
    "lane_partition_gateup": sum(n.startswith("q4k_lane_partition_gemv_12288") for n in names),
    "g3_lanemap_gateup": sum(n.startswith("q4k_g3_lanemap_gemv_12288") for n in names),
    "owned_down": sum(n.startswith("q4k_gemv_warp_4096_12288") for n in names),
    "g3_lanemap_down": sum(n.startswith("q4k_g3_lanemap_gemv_4096_12288") for n in names),
    "owned_proj": sum(n.startswith("q4k_gemv_warp_4096_4096") for n in names),
    "g3_lanemap_proj": sum(n.startswith("q4k_g3_lanemap_gemv_4096_4096") for n in names),
    "scheduler_programs": sum("q4k_scheduler" in n for n in names),
  }

def _write_doc(ts:str, out:dict):
  rows = out["rows"]
  best = out["best_arm"]
  lines = [
    f"# Coalesced dequant M-E result {ts}", "",
    f"Verdict: `{out['verdict']}`", "",
    "## Throughput", "",
    "| ctx | owned tok/s | sched_packed | generated_skeleton | sched_wordlane | g2_lanemap | g3_lanemap_codegen | lane_partition | bubblebeam_futuresight | best scheduler | best/owned | tokens match |",
    "|---:|---:|---:|---:|---:|---:|---:|---|---:|---|",
  ]
  for c in CTXS:
    r = rows[str(c)]
    lines.append(f"| {c} | {r['tok_s']['owned']} | {r['tok_s']['sched_packed']} | {r['tok_s']['generated_skeleton']} | "
                 f"{r['tok_s']['sched_wordlane']} | {r['tok_s']['g2_lanemap']} | {r['tok_s']['g3_lanemap_codegen']} | {r['tok_s']['lane_partition']} | {r['tok_s']['bubblebeam_futuresight']} | {r['best_scheduler_arm']} | "
                 f"{r['best_vs_owned_ratio']:.3f} | {r['tokens_match_all']} |")
  lines += ["", "## Interpretation", "", out["interpretation"], "", "## Artifact", "", f"- `{out['artifact']}`", ""]
  (DOCS/f"coalesced-dequant-mE-result-{ts}.md").write_text("\n".join(lines))

def main():
  perflevel = _try_pin_clock()
  from extra.qk.harness_contract import DEFAULT_MODEL
  from tinygrad import Tensor, UOp, TinyJit
  from tinygrad.helpers import getenv
  from extra.llm.generate import load_model_and_tokenizer
  m, tok = load_model_and_tokenizer(os.environ.get("QK_MODEL", DEFAULT_MODEL), MAXC, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []): lin.decode_enabled = True
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 800)
  ids = (ids * (1 + MAXC // max(1, len(ids))))[:MAXC]
  v = UOp.variable("start_pos", 0, MAXC - 1); temp = Tensor([0.0])

  def build(env, ck):
    for k in CLEAR_ENV: os.environ.pop(k, None)
    for k, val in env.items(): os.environ[k] = val
    getenv.cache_clear()
    for b in m.blk: b._use_flash, b._prefill_v2 = True, False
    step = TinyJit(m.forward); out = Tensor([[int(ids[ck])]], dtype="int32").contiguous()
    for i in range(8): out = step(out, v.bind(ck + i), temp).realize()
    return step, _program_counts(step)
  def measure(step, ck):
    out = Tensor([[int(ids[ck])]], dtype="int32").contiguous(); W = []; toks = []
    for i in range(NMEAS):
      t0 = time.perf_counter(); out = step(out, v.bind(ck + i), temp); tid = int(out.item())
      W.append((time.perf_counter() - t0) * 1e3); toks.append(tid)
    return statistics.median(W), toks

  rows = {}; arm_names = list(ARMS)
  for ck in CTXS:
    steps, counts = {}, {}
    for a in arm_names: steps[a], counts[a] = build(ARMS[a], ck)
    ms = {a: [] for a in arm_names}; toks = {}
    for r in range(REPEATS):
      order = arm_names[r % len(arm_names):] + arm_names[:r % len(arm_names)]
      for a in order:
        mm, tt = measure(steps[a], ck); ms[a].append(mm); toks[a] = tt
    med = {a: statistics.median(ms[a]) for a in arm_names}
    tps = {a: round(1000/med[a], 1) for a in arm_names}
    sched_arms = [a for a in arm_names if a != "owned"]
    best = max(sched_arms, key=lambda a: tps[a])
    rows[str(ck)] = {"tok_s": tps, "program_counts": counts, "tokens_match_all": all(toks[a] == toks["owned"] for a in arm_names),
                     "best_scheduler_arm": best, "best_vs_owned_ratio": round(tps[best] / tps["owned"], 4),
                     "spread_pct": {a: round(100*(max(ms[a])-min(ms[a]))/med[a], 2) for a in arm_names}}
    print(f"ctx {ck:5}: " + " | ".join(f"{a} {tps[a]}" for a in arm_names) +
          f" tok/s | best {best} ratio {rows[str(ck)]['best_vs_owned_ratio']:.3f} | tokens_match {rows[str(ck)]['tokens_match_all']} | counts {counts}",
          file=sys.__stderr__)

  tok_ok = all(r["tokens_match_all"] for r in rows.values())
  owned_route_ok = all(rows[str(c)]["program_counts"]["owned"]["owned_gateup"] > 0 for c in CTXS)
  lane_route_ok = all(rows[str(c)]["program_counts"]["lane_partition"]["lane_partition_gateup"] > 0 and
                      rows[str(c)]["program_counts"]["lane_partition"]["owned_gateup"] == 0 for c in CTXS)
  bubblebeam_route_ok = all((rows[str(c)]["program_counts"]["bubblebeam_futuresight"]["lane_partition_gateup"] > 0 or
                             rows[str(c)]["program_counts"]["bubblebeam_futuresight"].get("g3_lanemap_gateup", 0) > 0) and
                            rows[str(c)]["program_counts"]["bubblebeam_futuresight"]["owned_gateup"] == 0 for c in CTXS)
  bubblebeam_generated_route_ok = all(rows[str(c)]["program_counts"]["bubblebeam_futuresight"].get("g3_lanemap_gateup", 0) > 0 and
                                      rows[str(c)]["program_counts"]["bubblebeam_futuresight"]["lane_partition_gateup"] == 0 and
                                      rows[str(c)]["program_counts"]["bubblebeam_futuresight"]["owned_gateup"] == 0 for c in CTXS)
  generated_skeleton_route_ok = all(rows[str(c)]["program_counts"]["generated_skeleton"]["lane_partition_gateup"] == 0 and
                                    rows[str(c)]["program_counts"]["generated_skeleton"]["owned_gateup"] == 0 for c in CTXS)
  g2_lanemap_route_ok = all(rows[str(c)]["program_counts"]["g2_lanemap"]["lane_partition_gateup"] == 0 and
                            rows[str(c)]["program_counts"]["g2_lanemap"]["owned_gateup"] == 0 for c in CTXS)
  g3_lanemap_route_ok = all(rows[str(c)]["program_counts"]["g3_lanemap_codegen"]["g3_lanemap_gateup"] > 0 and
                            rows[str(c)]["program_counts"]["g3_lanemap_codegen"]["lane_partition_gateup"] == 0 and
                            rows[str(c)]["program_counts"]["g3_lanemap_codegen"]["owned_gateup"] == 0 for c in CTXS)
  best_ratios = {c: rows[str(c)]["best_vs_owned_ratio"] for c in CTXS}
  g2_lanemap_ratio_by_ctx = {c: round(rows[str(c)]["tok_s"]["g2_lanemap"] / rows[str(c)]["tok_s"]["owned"], 4) for c in CTXS}
  g2_lanemap_verdict = "G2_LANEMAP_PROMOTABLE" if tok_ok and g2_lanemap_route_ok and all(v >= PROCEED_RATIO for v in g2_lanemap_ratio_by_ctx.values()) else "SEARCH_GENERATED_WD_FAIL"
  g3_lanemap_ratio_by_ctx = {c: round(rows[str(c)]["tok_s"]["g3_lanemap_codegen"] / rows[str(c)]["tok_s"]["owned"], 4) for c in CTXS}
  g3_lanemap_verdict = "G3_LANEMAP_PROMOTABLE" if tok_ok and g3_lanemap_route_ok and all(v >= PROCEED_RATIO for v in g3_lanemap_ratio_by_ctx.values()) else "SEARCH_GENERATED_WD_FAIL"
  proceed = tok_ok and owned_route_ok and bubblebeam_route_ok and max(best_ratios.values()) >= PROCEED_RATIO and all(v >= PROCEED_RATIO for v in best_ratios.values())
  ts = time.strftime("%Y%m%d-%H%M%S")
  verdict = "PROCEED_P3_SEARCH_GENERALIZATION" if proceed else "STOP_CUSTOM_NEEDED_FOR_GEMV_TARGET"
  interpretation = ("Best scheduler/lane-partition arm reached the >=90% owned threshold at every ctx with matching tokens. Proceed to P3."
                    if proceed else "Best scheduler/lane-partition arm did not reach the >=90% owned threshold at every ctx. Do not fund P3; record CUSTOM as needed for this GEMV performance target.")
  artifact = OUT / f"coalesced_dequant_mE_{ts}.json"
  out = {"date": "2026-06-25", "timestamp": ts, "phase": "COALESCED_DEQUANT_M_E_DECISION", "perflevel": perflevel,
         "role": "FFN gate/up (Q4_K 4096x12288)", "arms": ARMS, "ctxs": CTXS, "nmeas": NMEAS, "repeats": REPEATS,
         "proceed_ratio": PROCEED_RATIO, "rows": rows, "tokens_match_all_ctx": tok_ok, "owned_route_ok": owned_route_ok,
         "lane_partition_route_ok": lane_route_ok, "bubblebeam_futuresight_route_ok": bubblebeam_route_ok, "bubblebeam_futuresight_generated_route_ok": bubblebeam_generated_route_ok, "generated_skeleton_route_ok": generated_skeleton_route_ok, "g2_lanemap_route_ok": g2_lanemap_route_ok, "g2_lanemap_ratio_by_ctx": g2_lanemap_ratio_by_ctx, "g2_lanemap_verdict": g2_lanemap_verdict, "g3_lanemap_route_ok": g3_lanemap_route_ok, "g3_lanemap_ratio_by_ctx": g3_lanemap_ratio_by_ctx, "g3_lanemap_verdict": g3_lanemap_verdict, "best_ratio_by_ctx": best_ratios, "best_arm": {c: rows[str(c)]["best_scheduler_arm"] for c in CTXS},
         "verdict": verdict, "interpretation": interpretation, "artifact": str(artifact.relative_to(ROOT))}
  OUT.mkdir(parents=True, exist_ok=True)
  artifact.write_text(json.dumps(out, indent=2)); (OUT/"coalesced_dequant_mE_latest.json").write_text(json.dumps(out, indent=2))
  _write_doc(ts, out)
  print(f"\nverdict: {verdict} | tokens_match {tok_ok} | owned_route {owned_route_ok} | lane_route {lane_route_ok} | bubblebeam_route {bubblebeam_route_ok} | generated_skeleton_route {generated_skeleton_route_ok} | {artifact}", file=sys.__stderr__)
  print(json.dumps({"verdict": verdict, "tokens_match": tok_ok, "best_ratio_by_ctx": best_ratios, "best_arm": out["best_arm"]}))

if __name__ == "__main__":
  main()
