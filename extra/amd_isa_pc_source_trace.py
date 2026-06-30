"""AMD ISA PC/source trace (audit-only) — bridge N4 whole-step + N2B PMC category -> PC/source/lowering rows.

ATT/SQTT per-PC decode is walled under HCQ (N2). So this tool does NOT use hardware per-PC stalls. It captures the
FINAL native_block_tile rdna3 Inst stream from AMDISARenderer (post schedule/waitcnt, with byte PCs), classifies each
instruction into a source GROUP + category + loop depth + EXEC-predication via opcode/structural heuristics, estimates
a DYNAMIC weight from compiler-visible loop/grid structure (NOT measured), and merges the N2B PMC counters as the
MEASURED category truth. Ranks source_hot_rows so the next codegen lever has a concrete site. Reuses N4/N2B/N0
artifacts; does not replace them. Audit-only: no optimization.

  "category measured by PMC, PC/source rows estimated by static loop weighting"

Run:  DEV=AMD PYTHONPATH=. .venv/bin/python extra/amd_isa_pc_source_trace.py
Writes: bench/amd-isa-backend-pc-source-trace/{latest.json, summary.md, native_inst_stream.json, owned_disasm.json}
"""
from __future__ import annotations
import os, json, re, pathlib
os.environ.setdefault("DEV", "AMD")
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-isa-backend-pc-source-trace"
Hd, Hq, Hkv, MAXC, L = 128, 32, 8, 4608, 96
TK = 16; NB = -(-L // TK)          # token-block loop trip (b-loop)
SMAX = -(-MAXC // L)               # 48

def _category(op: str) -> str:
  if op.startswith("ds_"): return "LDS"
  if op.startswith("global_"): return "VMEM"
  if op.startswith("s_load"): return "SMEM"
  if op.startswith("s_waitcnt"): return "WAIT"
  if op.startswith("s_barrier"): return "BARRIER"
  if op.startswith(("s_branch", "s_cbranch")): return "BRANCH"
  if op.startswith("s_and_saveexec") or (op.startswith("s_mov") and "exec" in op.lower()): return "BARRIER"
  if op.startswith("v_"): return "VALU"
  if op.startswith("s_"): return "SALU"
  return "OTHER"

def _source_group(op: str, exec_pred: bool) -> tuple[str, str, str]:
  # (source_group, lowering_site, candidate_lever) -- best-effort from opcode + EXEC context (amd.py lowering paths).
  if op.startswith("v_dot2"):       return "fdot_score", "amd.py:lower_inst V_DOT2 (isel_customi fdot2)", "match owned dot strategy / fewer score passes"
  if op.startswith("v_exp"):        return "exp_softmax", "amd.py:lower_inst V_EXP (N1A hardware exp2)", "already hardware; fuse exp into online-softmax merge"
  if op.startswith("ds_bpermute"):  return "cross_lane_reduce", "amd.py:DS_BPERMUTE (warp reduce ladder)", "amortize/stage warp reduce (N3D, but PMC says LDS-wait~0)"
  if op.startswith(("ds_load", "ds_store")): return "lds_accum_stage", "amd.py:isel_index LDS path / DS_LOAD/DS_STORE (DEFINE_REG accumulators + K/V staging)", "register accumulators (N5A regalloc-blocked) / fewer LDS round-trips"
  if op.startswith(("v_mul_lo", "v_add_nc", "v_lshlrev", "v_bfe", "v_movk")): return "address_index", "amd.py:isel_index / _binop V_IMUL/V_IADD/V_OFFSET", "scalarize uniform prefix (N1B refuted/dead) / strength-reduce"
  if op.startswith("global_load"):  return "kv_load", "amd.py:GLOBAL_LOAD (cache staging load)", "wider/coalesced loads"
  if op.startswith("global_store"): return "output_store", "amd.py:GLOBAL_STORE/GATED_STORE", "fewer partial stores"
  if op.startswith("s_and_saveexec") or (op.startswith("s_mov") and "exec" in op.lower()): return "exec_gated_store", "amd.py:GATED_STORE EXEC region", "reduce gated-store regions"
  if op.startswith("s_waitcnt"):    return "waitcnt", "amd.py:_insert_waitcnt (consumer-only)", "finer waitcnt thresholds"
  if op.startswith("v_max"):        return "softmax_max", "amd.py:V_MAX (online-softmax max)", "fuse max into reduce"
  if op.startswith(("v_fma", "v_add_f32", "v_mul_f32", "v_sub_f32")): return "pv_softmax_arith", "amd.py:V_ADD/V_MUL/V_FMA (PV accumulate + softmax rescale)", "FMA-fuse / reduce rescale ops"
  if op.startswith("v_cvt"):        return "cvt", "amd.py:V_CVT_*", "fewer f16<->f32 converts"
  if op.startswith("v_cmp"):        return "predicate", "amd.py:V_CMP (mask/where)", "fewer predicate evals"
  if op.startswith(("s_mul", "s_add", "s_cmp", "s_lshl")): return "loop_control", "amd.py:lower_range/lower_end (counter)", "unroll / fewer loop iters"
  if op.startswith("v_mov") or op.startswith("s_mov"): return "mov", "amd.py:MOV/MOV_S2V/V_CONST", "reduce copies"
  return "other", "?", "?"

def capture_native():
  from tinygrad.uop.ops import UOp, Ops
  from tinygrad.renderer.isa.amd import AMDISARenderer
  from tinygrad.helpers import getenv
  cap = {}
  _oasm = AMDISARenderer.asm
  def spy(self, prg, lin):
    try:
      ins = list(lin.src)
      if getenv("AMD_ISA_SCHED", 1): ins = self._schedule(ins)
      ins = self._insert_waitcnt(ins)                       # pre-label-resolve: still has ("label"/"branch") markers
      cap["stream"] = [u for u in ins if u.op is Ops.INS]
      cap["markers"] = ins
    except Exception as e: cap["err"] = str(e)
    return _oasm(self, prg, lin)
  AMDISARenderer.asm = spy
  from extra.qk_native_isa_block_tile_graph_node import compile_block_tile_isa, _compile
  _compile.cache_clear()
  compile_block_tile_isa(Hd, Hq, Hkv, MAXC, L, SMAX, UOp.variable("start_pos", 0, MAXC - 1) + 1)
  AMDISARenderer.asm = _oasm
  return cap

def build_rows(cap):
  from tinygrad.uop.ops import Ops
  markers = cap["markers"]
  # loop regions: a backward branch (target label defined earlier) = loop backedge. compute byte PCs + label byte offs.
  pos, labels, off = [], {}, 0
  for u in markers:
    pos.append(off); a = u.arg
    if isinstance(a, tuple) and a and a[0] == "label": labels[a[1]] = off
    elif isinstance(a, tuple) and a and a[0] == "branch": off += 4
    elif u.op is Ops.INS: off += len(a.to_bytes())
  # loop spans: for each branch whose target label byte < branch byte -> [target_byte, branch_byte]
  spans = []
  for i, u in enumerate(markers):
    a = u.arg
    if isinstance(a, tuple) and a and a[0] == "branch":
      tgt = a[2] if len(a) > 2 else None
      if tgt in labels and labels[tgt] <= pos[i]: spans.append((labels[tgt], pos[i]))
  def depth_at(b): return sum(1 for (s, e) in spans if s <= b <= e)
  rows = []; exec_pred = False
  for i, u in enumerate(markers):
    if u.op is not Ops.INS: continue
    asm = str(u.arg); op = re.match(r"\s*([a-z0-9_]+)", asm)
    op = op.group(1) if op else "?"
    if op.startswith("s_and_saveexec"): exec_pred = True
    cat = _category(op); grp, low, lever = _source_group(op, exec_pred)
    pc = pos[i]; d = depth_at(pc)
    rows.append({"pc": pc, "asm": asm[:80], "opcode": op, "category": cat, "source_group": grp,
                 "lowering_site": low, "candidate_lever": lever, "loop_depth": d, "exec_predicated": exec_pred,
                 "src_uop": None})   # exact UOp mapping needs a metadata hook (deferred; best-effort group instead)
    if op.startswith("s_mov") and "exec" in asm.lower() and exec_pred: exec_pred = False
  return rows

def estimate_dynamic(rows, ctx):
  # estimated_dynamic = static * trip(loop_depth) * workgroups(ctx) * waves_per_wg. ESTIMATE, not measured per-PC.
  wg = Hkv * (-(-ctx // L)); waves = 4
  trip = {0: 1, 1: NB, 2: NB * TK, 3: NB * TK}      # depth0 prologue/epilogue; depth1 b-loop; depth2+ per-token inner
  return {r["pc"]: r and (trip.get(r["loop_depth"], NB * TK) * wg * waves) for r in rows}

def main():
  OUT.mkdir(parents=True, exist_ok=True)
  rec = {"verdict": None, "route": "native_dynamic_s",
         "weighting_disclaimer": "category measured by PMC, PC/source rows estimated by static loop weighting (NOT hardware per-PC stalls)"}
  # SQTT per-PC retry (opt-in, never required)
  rec["hardware_per_pc_trace"] = {"available": False,
    "reason": "ATT/SQTT per-PC decode unavailable under HCQ (N2: instructions_size==0 / ATT_DECODER_REPAIR_BLOCKED); PMC category counters used instead"}
  if os.environ.get("AMD_ISA_PC_TRACE_TRY_SQTT"): rec["hardware_per_pc_trace"]["sqtt_retry"] = "still blocked (see N2)"
  cap = capture_native()
  if "stream" not in cap: rec["verdict"] = "AMD_ISA_PC_SOURCE_TRACE_BLOCKED_METADATA_LOSS"; rec["detail"] = cap.get("err"); json.dump(rec, open(OUT/"latest.json","w"), indent=2); return rec
  rows = build_rows(cap)
  json.dump(rows, open(OUT / "native_inst_stream.json", "w"), indent=2)
  dyn512, dyn4096 = estimate_dynamic(rows, 512), estimate_dynamic(rows, 4096)
  # owned disasm
  try:
    from extra.qk_decode_attention_fused_score_state_pv_attribution import _disasm
    import extra.qk_owned_flash_decode_graph_node as O
    oe = O._kernels(48, MAXC, "base", whole_cache=True)[0]
    odis = _disasm(oe); json.dump({"kernel": "owned_flash_tile_gqa_whole", "asm_lines": odis.splitlines()}, open(OUT/"owned_disasm.json","w"), indent=2)
    rec["owned_disasm_lines"] = len(odis.splitlines())
  except Exception as e: rec["owned_disasm_lines"] = f"err: {e}"
  # N2B PMC category ratios (measured truth)
  pmc = {}
  n2b = ROOT / "bench/amd-isa-backend-phase-n2b/latest.json"
  if n2b.exists():
    r = json.load(open(n2b)); rec["pmc_category_reference"] = str(n2b)
    for row in r.get("category_diff_rows", []): pmc[row["row"]] = {"native": row["native"], "ratio": row["ratio"]}
    rec["pmc_measured_rows"] = pmc
  cat_to_pmc = {"LDS": "lds_inst_per_wave", "VALU": "valu_inst_per_wave"}
  # rank source groups
  groups = {}
  for r in rows:
    g = groups.setdefault(r["source_group"], {"source_group": r["source_group"], "category": r["category"], "static_insts": 0,
        "est_dyn_512": 0, "est_dyn_4096": 0, "example_pcs": [], "example_asm": [], "lowering_sites": set(), "candidate_lever": r["candidate_lever"]})
    g["static_insts"] += 1; g["est_dyn_512"] += dyn512[r["pc"]]; g["est_dyn_4096"] += dyn4096[r["pc"]]
    if len(g["example_pcs"]) < 6: g["example_pcs"].append(r["pc"]); g["example_asm"].append(r["asm"][:48])
    g["lowering_sites"].add(r["lowering_site"])
  ranked = sorted(groups.values(), key=lambda g: -g["est_dyn_512"])
  rows_out = []
  for i, g in enumerate(ranked):
    pr = pmc.get(cat_to_pmc.get(g["category"], ""), {})
    rows_out.append({"rank": i + 1, "source_group": g["source_group"], "category": g["category"], "static_insts": g["static_insts"],
      "estimated_dynamic_insts_ctx512": g["est_dyn_512"], "estimated_dynamic_insts_ctx4096": g["est_dyn_4096"],
      "pmc_category_ratio_native_over_owned": pr.get("ratio"), "example_pcs": g["example_pcs"], "example_asm": g["example_asm"][:3],
      "lowering_sites": sorted(g["lowering_sites"]), "candidate_lever": g["candidate_lever"],
      "confidence": "high" if g["category"] in ("LDS", "VALU") and pr else "medium"})
  rec["source_hot_rows"] = rows_out
  rec["static_inst_total"] = len(rows)
  rec["category_static_breakdown"] = {c: sum(1 for r in rows if r["category"] == c) for c in set(r["category"] for r in rows)}
  # gates from N4 / Phase I
  def _r(p):
    f = ROOT / p; return json.load(open(f)) if f.exists() else {}
  n4 = _r("bench/amd-isa-backend-phase-n4/latest.json"); pi = _r("bench/amd-isa-backend-phase-i/latest.json"); h = _r("bench/amd-isa-backend-phase-h/latest.json")
  rec["whole_step_reference"] = "bench/amd-isa-backend-phase-n4/latest.json"
  rec["token_match"] = bool(pi.get("token_match")); rec["route_bound"] = bool(pi.get("route_bound"))
  rec["hidden_fallback"] = not str(pi.get("hidden_fallback_check", "")).startswith("no")
  ok = rec["token_match"] and rec["route_bound"] and not rec["hidden_fallback"] and rows_out and all("pc" in r and r.get("category") for r in rows)
  if not rec["route_bound"]: rec["verdict"] = "AMD_ISA_PC_SOURCE_TRACE_BLOCKED_ROUTE_ATTRIBUTION"
  elif not rec["token_match"]: rec["verdict"] = "AMD_ISA_PC_SOURCE_TRACE_BLOCKED_TOKEN_MATCH"
  elif not rows_out: rec["verdict"] = "AMD_ISA_PC_SOURCE_TRACE_INCONCLUSIVE_DYNAMIC_WEIGHTING"
  else: rec["verdict"] = "AMD_ISA_PC_SOURCE_TRACE_PASS_SOURCE_ROWS_PINNED"
  json.dump(rec, open(OUT / "latest.json", "w"), indent=2)
  # summary.md
  md = [f"# AMD ISA PC/source trace\n", f"**Verdict:** {rec['verdict']}  ", f"**{rec['weighting_disclaimer']}**  ",
        f"hardware per-PC: {rec['hardware_per_pc_trace']['available']} ({rec['hardware_per_pc_trace']['reason']})\n",
        f"native tile: {len(rows)} static insts; category breakdown {rec['category_static_breakdown']}\n",
        "## Ranked source groups (by estimated dynamic insts @ctx512)\n",
        "| rank | source_group | cat | static | est_dyn_ctx512 | est_dyn_ctx4096 | pmc_ratio | candidate_lever |",
        "|---|---|---|---|---|---|---|---|"]
  for r in rows_out: md.append(f"| {r['rank']} | {r['source_group']} | {r['category']} | {r['static_insts']} | {r['estimated_dynamic_insts_ctx512']} | {r['estimated_dynamic_insts_ctx4096']} | {r['pmc_category_ratio_native_over_owned']} | {r['candidate_lever']} |")
  top3 = rows_out[:3]
  md.append("\n## Top 3 source groups + levers\n")
  for r in top3: md.append(f"- **{r['source_group']}** ({r['category']}, est_dyn512={r['estimated_dynamic_insts_ctx512']}, pmc_ratio={r['pmc_category_ratio_native_over_owned']}): {r['candidate_lever']}  \n  sites: {', '.join(r['lowering_sites'])}")
  (OUT / "summary.md").write_text("\n".join(md))
  return rec

if __name__ == "__main__":
  rec = main()
  print(json.dumps({"verdict": rec["verdict"], "static_inst_total": rec.get("static_inst_total"),
                    "top_rows": [(r["rank"], r["source_group"], r["category"], r["estimated_dynamic_insts_ctx512"], r["pmc_category_ratio_native_over_owned"]) for r in rec.get("source_hot_rows", [])[:6]]}, indent=2))
  print("\nPC_SOURCE_TRACE", rec["verdict"])
