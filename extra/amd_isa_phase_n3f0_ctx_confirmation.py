"""AMD ISA backend — Phase N3F.0: ctx4096 PMC confirmation capture (before committing to an N3F dynamic-S fix).

N2B (ctx512) selected N3F: native issues 27.6x VALU / 48.4x LDS instructions per wave vs owned at equal occupancy ->
dynamic instruction/loop VOLUME, not memory/cross-lane. But ctx512 maximizes the FIXED_S whole-cache (MAXC=4608)
sweep redundancy (~9x masked). N3F.0 re-captures at ctx4096 (where the sweep is mostly valid) to decide:
  - if native STILL does >=~20x instructions/wave at ctx4096 -> NOT valid-S-bound; fixing valid-S alone won't close
    the gap (deeper per-token dynamic loop volume -> N3D-ish too).
  - if it NORMALIZES toward owned at ctx4096 -> the gap is dominated by the whole-cache sweep -> dynamic-S is a big
    win at short/mid ctx (N3F is the right, high-value lever).

Reuses extra/amd_isa_phase_n2b_pmc_attribution._capture (native PMC path) with N2B_CTX=4096; compares to the saved
ctx512 captures (bench/amd-isa-backend-phase-n2b/{owned,native}_pmc.json).

Run:  DEV=AMD PYTHONPATH=. .venv/bin/python extra/amd_isa_phase_n3f0_ctx_confirmation.py
Writes: bench/amd-isa-backend-phase-n3/{n3f0_ctx_comparison.json, n3f0_summary.md, owned_pmc_4096.json, native_pmc_4096.json}
"""
import os, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-isa-backend-phase-n3"
N2B = ROOT / "bench/amd-isa-backend-phase-n2b"

def _per_wave(c):
  w = max(1, c.get("SQ_WAVES", 1)); wc = max(1, c.get("SQ_WAVE_CYCLES", 1))
  return {"waves": c.get("SQ_WAVES", 0), "valu_inst_per_wave": round(c.get("SQ_INSTS_VALU",0)/w,1),
          "lds_inst_per_wave": round(c.get("SQ_INSTS_LDS",0)/w,1), "wave_cycles_per_wave": round(c.get("SQ_WAVE_CYCLES",0)/w,1),
          "wait_any_frac": round(c.get("SQ_WAIT_ANY",0)/wc,3), "vmem_cycles_frac": round(c.get("SQ_INST_CYCLES_VMEM",0)/wc,3),
          "lds_bank_conflict_per_wave": round(c.get("SQC_LDS_BANK_CONFLICT",0)/w,1)}

def main():
  OUT.mkdir(parents=True, exist_ok=True)
  import extra.amd_isa_phase_n2b_pmc_attribution as N2BM
  os.environ["N2B_CTX"] = "4096"
  owned4 = N2BM._capture("owned"); native4 = N2BM._capture("native")
  json.dump(owned4, open(OUT/"owned_pmc_4096.json","w"), indent=2); json.dump(native4, open(OUT/"native_pmc_4096.json","w"), indent=2)
  rec = {"scope": "Phase N3F.0: ctx4096 PMC confirmation vs ctx512 (decide valid-S-bound vs deeper dynamic loop volume)"}
  if owned4.get("failed") or owned4.get("dispatches",0)==0 or native4.get("failed") or native4.get("dispatches",0)==0:
    rec["verdict"]="AMD_ISA_PHASE_N3F0_BLOCKED_PMC_CAPTURE"; rec["owned4"]=owned4; rec["native4"]=native4
    json.dump(rec, open(OUT/"n3f0_ctx_comparison.json","w"), indent=2); return rec
  # ctx512 from saved N2B captures
  o512 = json.load(open(N2B/"owned_pmc.json"))["counters"]; n512 = json.load(open(N2B/"native_pmc.json"))["counters"]
  o4, n4 = owned4["counters"], native4["counters"]
  pw = {"ctx512": {"owned": _per_wave(o512), "native": _per_wave(n512)}, "ctx4096": {"owned": _per_wave(o4), "native": _per_wave(n4)}}
  rec["per_wave"] = pw
  def ratio(ctx, metric):
    o = pw[ctx]["owned"][metric]; n = pw[ctx]["native"][metric]; return round(n/o,2) if o else None
  rec["native_over_owned_per_wave"] = {
    "valu_inst": {"ctx512": ratio("ctx512","valu_inst_per_wave"), "ctx4096": ratio("ctx4096","valu_inst_per_wave")},
    "lds_inst":  {"ctx512": ratio("ctx512","lds_inst_per_wave"),  "ctx4096": ratio("ctx4096","lds_inst_per_wave")},
    "wave_cycles":{"ctx512": ratio("ctx512","wave_cycles_per_wave"),"ctx4096": ratio("ctx4096","wave_cycles_per_wave")},
  }
  rec["gates"] = {"owned_native_kernels_4096": {"owned": owned4.get("kernels_seen"), "native": native4.get("kernels_seen")},
                  "native_tile_present_4096": any("native_block_tile" in (k or "") for k in native4.get("kernels_seen",[]))}
  # decision
  valu4 = rec["native_over_owned_per_wave"]["valu_inst"]["ctx4096"]
  lds4  = rec["native_over_owned_per_wave"]["lds_inst"]["ctx4096"]
  wc4   = rec["native_over_owned_per_wave"]["wave_cycles"]["ctx4096"]
  valu512 = rec["native_over_owned_per_wave"]["valu_inst"]["ctx512"]
  dom = max(v for v in (valu4, lds4, wc4) if v is not None)
  if not rec["gates"]["native_tile_present_4096"]:
    rec["verdict"] = "AMD_ISA_PHASE_N3F0_BLOCKED_ROUTE_ATTRIBUTION"
  elif dom >= 20:
    rec["verdict"] = "AMD_ISA_PHASE_N3F0_NOT_VALID_S_BOUND"
    rec["finding"] = (f"at ctx4096 native STILL does ~{dom}x instructions/wave (valu {valu4}x, lds {lds4}x, wave_cycles {wc4}x). "
      f"The dynamic-loop excess persists where the whole-cache sweep is mostly VALID, so fixing valid-S alone will NOT close the gap "
      f"-- there is a deeper per-token dynamic loop-volume / dependency-depth cost (N3D territory) on top of any N3F sweep redundancy.")
    rec["next"] = "N3D-class per-token dataflow/dependency rewrite is needed in addition to (or instead of) N3F dynamic-S; re-scope before implementing."
  elif valu512 and valu512 >= 2 * (valu4 or 1):
    rec["verdict"] = "AMD_ISA_PHASE_N3F0_VALID_S_BOUND"
    rec["finding"] = (f"native instr/wave ratio drops from {valu512}x (ctx512) to {valu4}x (ctx4096) as the sweep becomes valid -> "
      f"the gap at short/mid ctx is dominated by the FIXED_S whole-cache sweep. Dynamic-S (process valid splits) is a high-value N3F win at short/mid ctx.")
    rec["next"] = "Implement N3F dynamic-S (valid-split count), expect large W==D gains at short/mid ctx, smaller at ctx~MAXC."
  else:
    rec["verdict"] = "AMD_ISA_PHASE_N3F0_MIXED"
    rec["finding"] = f"partial normalization (valu {valu512}x@512 -> {valu4}x@4096). N3F dynamic-S helps short/mid ctx but a residual per-token cost remains at long ctx."
    rec["next"] = "N3F dynamic-S first (clear short/mid-ctx win), then re-measure to size the residual N3D per-token cost."
  json.dump(rec, open(OUT/"n3f0_ctx_comparison.json","w"), indent=2)
  md = [f"# Phase N3F.0 ctx confirmation\n", f"**Verdict:** {rec['verdict']}\n", f"{rec.get('finding','')}\n",
        "| metric | ctx512 native/owned | ctx4096 native/owned |", "|---|---|---|"]
  for m in ("valu_inst","lds_inst","wave_cycles"):
    r = rec["native_over_owned_per_wave"][m]; md.append(f"| {m}_per_wave | {r['ctx512']}x | {r['ctx4096']}x |")
  md.append(f"\n**Next:** {rec.get('next','')}\n")
  (OUT/"n3f0_summary.md").write_text("\n".join(md))
  return rec

if __name__ == "__main__":
  rec = main()
  print(json.dumps({k:rec[k] for k in ("verdict","native_over_owned_per_wave","finding","next") if k in rec}, indent=2))
  print("\nPHASE_N3F0", rec["verdict"])
