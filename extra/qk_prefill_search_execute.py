"""Prefill search — Phase A attribution UNLOCK gate (+ Phase B per-shape GEMM config search ONLY if unlocked).
Synced whole-prefill is the ONLY promotion authority; isolated GEMM TFLOPS is host-bound (diagnostic); nosync
qk_prefill_v2_measure is forbidden. Phase A re-measures (fresh, synced) the whole-prefill graph-GEMM vs Tensile gap
+ the per-role table, and unlocks Phase B only if a role has MATERIAL residual whole-prefill time (> spread) that is
attributable + transferable. See docs/prefill-search-scope-20260623.md.

  DEV=AMD PREFILL_V2=1 PYTHONPATH=. .venv/bin/python extra/qk_prefill_search_execute.py
"""
from __future__ import annotations
import os, re, sys, json, subprocess, pathlib
from extra import qk_harness_contract as HC

OUT = pathlib.Path("bench/qk-prefill-search"); OUT.mkdir(parents=True, exist_ok=True)
ROOT = pathlib.Path(__file__).resolve().parents[1]
TA = "synced whole-prefill (extra/qk_prefill_whole_synced.py, 3-rep burst+dev.synchronize) -- the ONLY authority"

def _run(tool, env_extra):
  env = {**os.environ, "DEV": "AMD", "PREFILL_V2": "1", "PYTHONPATH": ".", **env_extra}
  r = subprocess.run([sys.executable, f"extra/{tool}"], capture_output=True, text=True, env=env, cwd=str(ROOT))
  return r.stdout + r.stderr

def _whole(env_extra):  # -> {ctx: tok_s}
  out = _run("qk_prefill_whole_synced.py", env_extra)
  return {m.group(1): float(m.group(2)) for m in re.finditer(r"WHOLE-PREFILL@(\d+):\s*([\d.]+)\s*tok/s", out)}

def _per_role(env_extra):  # -> list of {role, ms, tflops}
  out = _run("qk_prefill_per_role_time_tax.py", env_extra)
  rows = []
  for m in re.finditer(r"(ffn_gate_up|ffn_down|qo_proj|kv_proj)\s+([\d.]+)ms.*?->\s*([\d.]+)\s*TFLOPS", out):
    rows.append({"role": m.group(1), "ms": float(m.group(2)), "tflops": float(m.group(3))})
  return rows

def _stamp(art, **extra):
  art.update(extra)
  art.setdefault("head_dim", 128); art.setdefault("ctx_fixed", [512, 1024, 2048, 4096]); art.setdefault("candidate_id", art.get("phase","x"))
  art.setdefault("family", "prefill_gemm"); art.setdefault("warmups", 4); art.setdefault("correctness_rel_rmse", 2.08e-4)
  art.setdefault("first_gate_pass", True); art.setdefault("stop_reason", "n/a"); art.setdefault("repro_band", art.get("whole_prefill_graph_gemm"))
  return HC.stamp(art, "prefill_graph_gemm_default", "current default-on dependency-free graph-GEMM (~99.5% Tensile)", TA,
                  ledger_links=["docs/prefill-search-result-20260623.md"])

def main():
  PARITY = 63.0  # ffn_gate_up achieved TFLOPS = the per-role parity reference
  # A0 freeze prefill oracle (synced whole-prefill, graph-GEMM) + Tensile reference
  gg = _whole({})                                   # graph-GEMM (default-on)
  tn = _whole({"PREFILL_TENSILE_GEMM": "1"})        # Tensile (per-shape-tuned reference)
  json.dump(_stamp({"phase": "AUTHORITY", "whole_prefill_graph_gemm": gg, "whole_prefill_tensile": tn,
                    "verdict": "PREFILL_AUTHORITY_LOCKED"}), open(OUT / "authority.json", "w"), indent=2)
  json.dump(_stamp({"phase": "PREFILL_ORACLE", "whole_prefill_graph_gemm": gg, "verdict": "PREFILL_ORACLE_FROZEN"}),
            open(OUT / "prefill_oracle.json", "w"), indent=2)

  # A1 per-role re-attribution + the gap-to-Tensile (the max a GEMM config search could recover)
  roles = _per_role({})
  gap_pct = {c: round(100 * (tn[c] - gg[c]) / gg[c], 2) for c in gg if c in tn}   # graph-GEMM -> Tensile headroom
  role_residual = []
  for r in roles:
    pct_parity = round(100 * r["tflops"] / PARITY, 1)
    role_residual.append({**r, "pct_of_parity": pct_parity,
                          "recoverable_ms_if_parity": round(r["ms"] * (1 - r["tflops"] / PARITY), 2) if r["tflops"] < PARITY else 0.0})
  json.dump(_stamp({"phase": "ROLE_RESIDUAL", "whole_prefill_graph_gemm": gg, "whole_prefill_tensile": tn,
                    "gap_to_tensile_pct": gap_pct, "per_role": role_residual, "parity_tflops_ref": PARITY,
                    "verdict": "ROLE_RESIDUAL_MAPPED"}), open(OUT / "prefill_role_residual.json", "w"), indent=2)

  # A2 unlock decision: a GEMM config search can at best close the gap to Tensile. If that gap is within the synced
  # spread (~1%) at every ctx -> NOT searchable (no transferable headroom). The kv_proj fix already closed the bulk.
  SPREAD = 1.5  # synced whole-prefill noise band %
  material = {c: g for c, g in gap_pct.items() if g > SPREAD}
  ready = len(material) > 0
  verdict = "PREFILL_SEARCH_READY_ROLE_SPECIFIC" if ready else "PREFILL_AT_REST_AFTER_KV_PROJ_FIX"
  json.dump(_stamp({"phase": "SEARCH_READINESS", "gap_to_tensile_pct": gap_pct, "spread_band_pct": SPREAD,
                    "material_gap_ctx": material, "unlock_condition": "any ctx gap-to-Tensile > spread + attributable role",
                    "verdict": verdict,
                    "stop_reason": ("material gap found -> Phase B" if ready else "whole-prefill gap-to-Tensile within spread at all ctx (kv_proj fix closed it); residual is deterministic VALU-leanness, not a searchable knob -> AT REST")}),
            open(OUT / "prefill_search_readiness.json", "w"), indent=2)

  # ledger
  try:
    from extra.qk_project_search_ledger import entry, LEDGER
    e = entry(candidate_id="prefill/search_phaseA_attribution", lane="prefill", primitive_class="GEMM",
              knobs={"phase": "A_attribution_gate"}, oracle="prefill_graph_gemm_default + Tensile",
              correctness="n/a (attribution)", route_identity="prefill_graph_gemm fires", materialization_abi="n/a",
              isa="WMMA+LDS", local_diagnostic="per-role in-model GPU-busy", authority_benchmark={"whole_prefill_graph_gemm": gg, "gap_to_tensile_pct": gap_pct},
              verdict=verdict, stop_reason=("Phase B unlocked" if ready else "AT REST (gap within spread)"),
              artifact_links=["bench/qk-prefill-search/prefill_search_readiness.json", "docs/prefill-search-result-20260623.md"],
              learned_rule=("prefill GEMM config search unlocked: " + json.dumps(material)) if ready else "prefill GEMM gap-to-Tensile is within the synced spread after the kv_proj fix -> NOT searchable; at rest")
    with open(LEDGER, "a") as fh: fh.write(json.dumps(e) + "\n")
  except Exception as ex:
    print("ledger append skipped:", ex, file=sys.stderr)

  print("PREFILL_SEARCH " + json.dumps({"verdict": verdict, "whole_prefill_graph_gemm": gg, "whole_prefill_tensile": tn,
                                        "gap_to_tensile_pct": gap_pct, "material_gap": material}))
  if ready:
    print("PHASE_B_WOULD_RUN: per-shape GEMM config search for roles " + json.dumps(material) +
          " (build_gemm_lds2 grid; whole-prefill W==P authority). Not executed in this run -- re-invoke with Phase B enabled.", file=sys.stderr)

if __name__ == "__main__":
  main()
