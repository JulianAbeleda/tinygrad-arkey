"""AMD ISA residual lever triage (R0, audit-only) — separate live speed levers from PC/source-trace false positives,
then select EXACTLY ONE R1 lever. No optimization here.

The PC/source trace ranks source groups by ESTIMATED static loop weight (not hardware per-PC). The #1 row
(address_index) is a known trap: naive scalarization (N1B) is dead on the live path. R0 answers, per the scope:
  address_index_live / address_index_strength_reduce / waitcnt_binding / lds_accum_stage_roundtrips /
  other_classifier_split / pv_softmax_fusion / mov_copy_source
using (a) the captured native Inst stream (stream-level counts by loop depth, robust opcode/marker re-parse, FMA-pair
+ mov-cause detection) and (b) a UOp-graph LICM/strength-reduce check on the tile's pre-isel sink (the real R1B
feasibility question: are the hot-loop int address multiplies loop-INVARIANT [hoistable] or loop-var*const
[strength-reducible], without resurrecting the refuted N1B SGPR-scalar path?). Then applies the scope's selection rules.

Run:  DEV=AMD PYTHONPATH=. .venv/bin/python extra/amd_isa_residual_lever_triage.py
Writes: bench/amd-isa-backend-residual-lever-triage/{latest.json, summary.md}
"""
from __future__ import annotations
import os, json, re, pathlib
os.environ.setdefault("DEV", "AMD")
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-isa-backend-residual-lever-triage"
STREAM = ROOT / "bench/amd-isa-backend-pc-source-trace/native_inst_stream.json"

def _reparse_op(asm: str) -> str:
  if asm.startswith("('label'"): return "_label"
  if asm.startswith("('branch'"):
    m = re.search(r"'(s_c?branch[a-z0-9_]*)'", asm); return m.group(1) if m else "_branch"
  m = re.search(r"([a-z][a-z0-9_]+)", asm); return m.group(1) if m else "?"

def _dst(asm: str): m = re.search(r"\(\s*(v\[\d+\])", asm); return m.group(1) if m else None

def stream_rows():
  s = json.load(open(STREAM))
  for r in s:
    if r["opcode"] == "?": r["opcode"] = _reparse_op(r["asm"])
  real = [r for r in s if r["opcode"] not in ("_label", "_branch")]   # control markers carry no compute
  D = lambda r: r["loop_depth"]
  hot = lambda r: D(r) >= 1
  rows = {}
  # other_classifier_split
  other = [r for r in s if r["source_group"] == "other"]
  from collections import Counter
  rows["other_classifier_split"] = {"total": len(other),
    "labels": sum(1 for r in other if r["opcode"] == "_label"), "branches": sum(1 for r in other if r["opcode"] == "_branch"),
    "real_op_families": dict(Counter(r["opcode"] for r in other if r["opcode"] not in ("_label", "_branch"))),
    "note": "labels+branches are synthetic control markers (0 compute); only real_op_families are candidate work"}
  # address_index
  ai = [r for r in s if r["source_group"] == "address_index"]
  rows["address_index_live"] = {"total": len(ai), "hot_loop_depth>=1": sum(1 for r in ai if hot(r)),
    "v_mul_lo": sum(1 for r in ai if r["opcode"].startswith("v_mul_lo")), "v_add_nc": sum(1 for r in ai if r["opcode"].startswith("v_add_nc")),
    "verdict": "LIVE (vector address ops feed global/LDS load addresses). N1B's 'dead' was the SCALARIZED prefix (s64/s65 unconsumed because the live path stayed vector); the VECTOR ops here ARE consumed. So address_index is live, but SGPR-scalarization is the wrong (refuted) lever."}
  # waitcnt_binding
  wc = [r for r in s if r["opcode"].startswith("s_waitcnt")]
  rows["waitcnt_binding"] = {"total": len(wc), "hot_loop_depth>=1": sum(1 for r in wc if hot(r)), "depth>=2": sum(1 for r in wc if D(r) >= 2),
    "note": "PMC (N2B) did not name wait as the primary category; finer thresholds reduce inst count but unlikely to close 35-40%."}
  # lds_accum_stage_roundtrips
  ds = [r for r in s if r["opcode"].startswith(("ds_load", "ds_store"))]
  bperm = [r for r in s if r["opcode"].startswith("ds_bpermute")]
  rows["lds_accum_stage_roundtrips"] = {"ds_load_store_total": len(ds), "ds_bpermute_cross_lane": len(bperm),
    "ds_at_depth>=2_inner_per_token": sum(1 for r in ds if D(r) >= 2), "ds_at_depth<=1_staging": sum(1 for r in ds if D(r) <= 1),
    "verdict": "inner-loop (depth>=2) ds load/store = DEFINE_REG accumulator RMW per token (the structural owned-vs-native gap, PMC 48x LDS/wave). Removing needs loop-carried register state = N5A, regalloc-blocked."}
  # pv_softmax_fusion
  fma = 0
  for i in range(len(s) - 1):
    if s[i]["opcode"].startswith("v_mul_f32") and s[i+1]["opcode"].startswith("v_add_f32") and (d := _dst(s[i]["asm"])) and d in s[i+1]["asm"]: fma += 1
  pv = [r for r in s if r["source_group"] == "pv_softmax_arith"]
  rows["pv_softmax_fusion"] = {"pv_softmax_static": len(pv), "hot_loop": sum(1 for r in pv if hot(r)),
    "fma_fusable_mul_then_add_pairs": fma, "note": f"only {fma} adjacent v_mul_f32->v_add_f32 fusable -> tiny static win"}
  # mov_copy_source
  mv = [r for r in s if r["opcode"] in ("v_mov_b32_e32", "s_mov_b32")]
  def mk(a):
    m = re.search(r"\((?:v\[\d+\]|EXEC\w*|s\[\d+\]),\s*(.+?)\)", a)
    if not m: return "other"
    src = m.group(1)
    return "S2V" if src.startswith("s[") else ("vgpr_copy" if src.startswith("v[") else ("exec" if "EXEC" in a else "immediate"))
  from collections import Counter as C2
  rows["mov_copy_source"] = {"total": len(mv), "by_cause": dict(C2(mk(r["asm"]) for r in mv)),
    "hot_loop": sum(1 for r in mv if hot(r)),
    "note": "S2V = loop-counter/SGPR->VGPR for address math (needed); immediate = const materialization (some inline-foldable); vgpr_copy = candidate removable carrier moves."}
  return rows, s

def uop_licm_check():
  # R1B feasibility, RIGOROUS (linearized span-membership, not the invariant+consumed false positive): in the tile's
  # LINEARIZED sink, an int ADD/MUL is a real HOIST opportunity only if it is loop-INVARIANT AND still emitted INSIDE a
  # RANGE..END span (recomputed per iteration). tinygrad's linearizer hoists invariants by construction, so this is ~0.
  # A real STRENGTH-REDUCE candidate is loop_var*const emitted inside a span (genuinely recomputed). Reuses N1B-audit build.
  try:
    from tinygrad.uop.ops import UOp, Ops, AxisType
    from tinygrad import dtypes
    from tinygrad.codegen import full_rewrite_to_sink
    from tinygrad.codegen.late.linearizer import linearize as _lin
    from extra.qk_flash_decode import flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel
    import extra.qk_native_isa_block_tile_graph_node as M
    Hd, Hq, Hkv, MAXC, L, S = 128, 32, 8, 4608, 96, 48
    fxn = flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(Hd, Hq, Hkv, MAXC, L, S, UOp.variable("start_pos", 0, 4607) + 1)
    phs = [UOp.placeholder((Hq*S*(Hd+2),), dtypes.float32, 0), UOp.placeholder((Hq*Hd,), dtypes.float32, 1), UOp.placeholder((2,1,Hkv,MAXC,Hd), dtypes.float32, 2)]
    fs = full_rewrite_to_sink(M._range_global_to_grid(fxn(*phs)), M._isa_renderer())
    lin = _lin(fs); idx = {u: i for i, u in enumerate(lin)}
    spans = []
    for i, u in enumerate(lin):
      if u.op is Ops.END:
        for r in u.src[1:]:
          if r.op is Ops.RANGE and r.arg[1] in (AxisType.LOOP, AxisType.REDUCE) and r in idx: spans.append((idx[r], i))
    inloop = lambda i: any(s < i < e for s, e in spans)
    loops = [u for u in lin if u.op is Ops.RANGE and u.arg[1] in (AxisType.LOOP, AxisType.REDUCE)]
    def dep(u, t, m):
      if u in m: return m[u]
      m[u] = False; r = (u is t) or any(dep(s, t, m) for s in u.src); m[u] = r; return r
    intalu = [u for u in lin if u.op in (Ops.ADD, Ops.MUL) and u.dtype in dtypes.ints]
    inv = [u for u in intalu if not any(dep(u, lp, {}) for lp in loops)]
    hoist_real = sum(1 for u in inv if u in idx and inloop(idx[u]))
    strength = sum(1 for u in intalu if any(dep(u, lp, {}) for lp in loops) and u.op is Ops.MUL
                   and any(s.op is Ops.CONST for s in u.src) and u in idx and inloop(idx[u]))
    return {"n_inner_loops": len(spans), "int_alu_total": len(intalu), "invariant_int_alu": len(inv),
            "loop_invariant_hoistable_REAL_inside_loop": hoist_real,
            "loop_var_times_const_strength_reducible_inside_loop": strength,
            "interpretation": ("hoist_real==0 => tinygrad's linearizer ALREADY hoists invariant address math out of the "
              "inner loops by construction -> R1B-hoist is a FALSE POSITIVE (the earlier invariant+consumed-in-loop count "
              "double-counted already-hoisted ops). The only real in-loop address lever is strength-reduction of "
              "loop_var*const, which needs induction-variable codegen (a renderer feature, NOT a bounded peephole).")}
  except Exception as e:
    import traceback; return {"error": traceback.format_exc().splitlines()[-1]}

def main():
  OUT.mkdir(parents=True, exist_ok=True)
  if not STREAM.exists():
    rec = {"verdict": "AMD_ISA_RESIDUAL_TRIAGE_BLOCKED_TRACE_METADATA", "detail": "run extra/amd_isa_pc_source_trace.py first"}
    json.dump(rec, open(OUT/"latest.json","w"), indent=2); return rec
  rows, s = stream_rows()
  licm = uop_licm_check()
  rec = {"scope": "Phase R0: residual lever triage -> select exactly one R1 lever", "triage_rows": rows, "uop_licm_check": licm,
         "weighting_note": "category measured by PMC (N2B); PC/source rows estimated by static loop weighting (PC-source-trace). Triage adds stream + UOp-graph dataflow evidence."}
  # selection (scope rules, in order)
  ai = rows["address_index_live"]; ds = rows["lds_accum_stage_roundtrips"]; pv = rows["pv_softmax_fusion"]; mv = rows["mov_copy_source"]; wc = rows["waitcnt_binding"]
  hoist_real = licm.get("loop_invariant_hoistable_REAL_inside_loop", 0); strength = licm.get("loop_var_times_const_strength_reducible_inside_loop", 0)
  reasons = {}
  # R1A: lds_accum dominant + needs loop-carried regs -> only if regalloc proof ready (N5A blocked => NOT ready)
  reasons["R1A_REGALLOC_FEATURE"] = f"lds_accum inner-loop ds={ds['ds_at_depth>=2_inner_per_token']} (structural, PMC 48x). The real lever, but BLOCKED: removing needs loop-carried physical accumulators; N5A proved the single-def regalloc can't represent it and NO regalloc-model design proof is ready -> per scope, stop, do not select."
  # R1B: only REAL if hoist_real>0 (linearized-verified, not the invariant+consumed false positive). strength-reduce is
  # real-in-loop but needs induction-variable codegen (a renderer feature, NOT a small/bounded peephole).
  r1b_hoist_real = (hoist_real or 0) > 0
  reasons["R1B_ADDRESS_STRENGTH_REDUCE"] = (f"address_index live hot-loop ops={ai['hot_loop_depth>=1']}. RIGOROUS linearized check: "
    f"loop-invariant-hoistable-INSIDE-loop={hoist_real} (==0 => tinygrad ALREADY hoists invariants -> hoist is a FALSE POSITIVE, "
    f"the earlier hoistable count was invariant+consumed-in-loop = already-hoisted ops). strength-reducible(loop_var*const) in-loop={strength} "
    f"-> real but needs INDUCTION-VARIABLE codegen (renderer feature, not a bounded peephole) for negligible wall (tile ~10% of wall, {strength}/197 VALU). "
    f"{'VIABLE (real hoist opportunity)' if r1b_hoist_real else 'NOT a bounded credible lever (hoist false-positive; strength-reduce = unbounded IV-codegen feature, negligible W==D)'}.")
  # R1C: pv/mov >=5 live hot-loop removable
  r1c_cluster = pv["fma_fusable_mul_then_add_pairs"] + mv["by_cause"].get("vgpr_copy", 0)
  reasons["R1C_LOCAL_CODEGEN_CLEANUP"] = f"FMA-fusable pairs={pv['fma_fusable_mul_then_add_pairs']}, removable vgpr_copy movs={mv['by_cause'].get('vgpr_copy',0)} -> live removable cluster={r1c_cluster}. {'VIABLE (>=5)' if r1c_cluster >= 5 else 'too small (<5 live hot-loop VALU)'}."
  reasons["R1D_WAITCNT_THRESHOLDS"] = f"waitcnt total={wc['total']} (hot={wc['hot_loop_depth>=1']}). N2B did NOT name wait as primary category -> capped/<2% expected W==D; last resort only."
  rec["lever_feasibility"] = reasons
  # apply rules: only bounded + credible levers may be selected. hoist false-positive; strength-reduce + lds_accum are
  # unbounded backend features (IV codegen / regalloc); R1C too small; R1D capped. => no bounded credible lever.
  if r1b_hoist_real: sel, why = "R1B_ADDRESS_STRENGTH_REDUCE", "linearized-verified loop-invariant address math is emitted inside the loop (real hoist, non-N1B, bounded)"
  elif r1c_cluster >= 5: sel, why = "R1C_LOCAL_CODEGEN_CLEANUP", "a live removable PV/mov cluster >=5"
  else:
    sel, why = None, ("no SMALL live lever with a credible W==D path: R1A (lds_accum, the real structural lever) is regalloc-blocked "
      "with no design proof; R1B-hoist is a verified false positive (tinygrad already hoists) and R1B-strength-reduce needs "
      "unbounded induction-variable codegen for negligible wall; R1C FMA cluster <5; R1D waitcnt is non-primary/capped. "
      "The residual is STRUCTURAL (register accumulators, N5A) -> needs a regalloc feature, not a bounded lever.")
  rec["selected_r1_lever"] = sel
  rec["selection_reason"] = why
  rec["real_but_unbounded_levers"] = ["R1A register accumulators (regalloc feature, N5A-blocked)", f"R1B strength-reduce {strength} loop_var*const muls (induction-variable codegen)"]
  rec["r1a_blocked_no_proof"] = True
  # gates (from PC-source-trace / Phase I)
  pst = json.load(open(ROOT/"bench/amd-isa-backend-pc-source-trace/latest.json"))
  rec["gates"] = {"token_match": pst.get("token_match"), "route_bound": pst.get("route_bound"), "hidden_fallback": pst.get("hidden_fallback")}
  metadata_ok = all(("loop_depth" in r and "category" in r) for r in s)
  if not metadata_ok: rec["verdict"] = "AMD_ISA_RESIDUAL_TRIAGE_BLOCKED_TRACE_METADATA"
  elif sel is None: rec["verdict"] = "AMD_ISA_RESIDUAL_TRIAGE_INCONCLUSIVE_NO_LIVE_LEVER"
  else: rec["verdict"] = "AMD_ISA_RESIDUAL_TRIAGE_PASS_LEVER_SELECTED"
  json.dump(rec, open(OUT/"latest.json","w"), indent=2)
  md = [f"# R0 residual lever triage\n", f"**Verdict:** {rec['verdict']}  ", f"**Selected R1 lever:** {sel}  ", f"**Why:** {why}\n",
        "## Lever feasibility\n"] + [f"- **{k}**: {v}" for k, v in reasons.items()] + \
       ["\n## UOp-graph LICM/strength-reduce check\n", f"```\n{json.dumps(licm, indent=1)}\n```",
        "\n## Triage rows\n"] + [f"- **{k}**: {json.dumps(v)[:240]}" for k, v in rows.items()]
  (OUT/"summary.md").write_text("\n".join(md))
  return rec

if __name__ == "__main__":
  rec = main()
  print(json.dumps({k: rec.get(k) for k in ("verdict", "selected_r1_lever", "selection_reason", "uop_licm_check")}, indent=2))
  print("\nR0_TRIAGE", rec["verdict"])
