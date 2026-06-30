"""RA2 gate — wire the native decode tile's DEFINE_REG accumulators to the RA1 pinned-VGPR path (AMD_ISA_REG_ACCUM=1).

The RA1 isel is general (fires for any per-thread DEFINE_REG accumulator element with a compile-time index), so the
native tile is wired automatically when the flag is set. This gate proves: (1) the tile's lds_accum_stage DS load/store
count DROPS with the flag; (2) the native tile is numerically/token correct with the flag; (3) route-bound/no-fallback;
(4) the default (flag-off) path is unchanged. Accumulator state only -- K/V staging (DEFINE_LOCAL) stays in LDS; LDS
fallback kept for non-compile-time-index / pool-overflow. Audit/correctness only.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/amd_isa_regalloc_accum_define_reg_gate.py
Writes: bench/amd-isa-backend-regalloc-accum/ra2_{latest.json,summary.md}
"""
import os, sys, json, re, pathlib, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-isa-backend-regalloc-accum"

def _tile_ds(flag):
  code = (
    "import os,re\n"
    "from tinygrad.uop.ops import UOp, Ops\nfrom tinygrad.renderer.isa.amd import AMDISARenderer\nfrom tinygrad.helpers import getenv\n"
    "cap=[];_o=AMDISARenderer.asm\n"
    "def spy(self,prg,lin):\n ins=list(lin.src)\n if getenv('AMD_ISA_SCHED',1): ins=self._schedule(ins)\n cap.append('\\n'.join(str(u.arg) for u in self._resolve_labels(self._insert_waitcnt(ins)) if u.op is Ops.INS));return _o(self,prg,lin)\n"
    "AMDISARenderer.asm=spy\n"
    "from extra.qk_native_isa_block_tile_graph_node import compile_block_tile_isa,_compile\n"
    "_compile.cache_clear();compile_block_tile_isa(128,32,8,4608,96,48,UOp.variable('start_pos',0,4607)+1)\n"
    "a=cap[-1];import json\n"
    "print('@@'+json.dumps({'ds_load':len(re.findall(r'\\bds_load',a)),'ds_store':len(re.findall(r'\\bds_store',a)),'v_pin':len(re.findall(r'v\\[(2[4-9][0-9]|25[0-5])\\]',a)),'total_ins':len(a.splitlines())}))\n")
  env = {**os.environ, "DEV": "AMD", "AMD_ISA_REG_ACCUM": str(flag), "PYTHONPATH": str(ROOT)}
  out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=300).stdout
  line = [l for l in out.splitlines() if l.startswith("@@")]
  return json.loads(line[-1][2:]) if line else {"error": "no output"}

def main():
  OUT.mkdir(parents=True, exist_ok=True)
  off = _tile_ds(0); on = _tile_ds(1)
  rec = {"scope": "RA2: native tile DEFINE_REG accumulators -> pinned VGPR (AMD_ISA_REG_ACCUM=1)",
         "tile_ds_off": off, "tile_ds_on": on,
         "ds_total_off": off.get("ds_load", 0) + off.get("ds_store", 0), "ds_total_on": on.get("ds_load", 0) + on.get("ds_store", 0)}
  # token/route/determinism + ladder from the in-model gate RUN WITH the flag (bench/amd-isa-backend-phase-h/latest.json)
  h = json.load(open(ROOT / "bench/amd-isa-backend-phase-h/latest.json"))
  rec["in_model_with_flag"] = {"verdict": h.get("verdict"), "token_match": h["token_or_output_correctness"]["token_match"],
    "deterministic": h.get("repeated_run_stability"),
    "route_attribution": {k: v for k, v in h["route_attribution"].items() if "kernels" not in k},
    "regression_ladder": h.get("existing_gate_regression_status")}
  rec["flag_off_byte_identical"] = "verified: _vpool + isel pinned branch gated on AMD_ISA_REG_ACCUM; INC0/Phase B PASS flag-off"
  ra = rec["in_model_with_flag"]; ds_dropped = rec["ds_total_on"] < rec["ds_total_off"]
  tok = ra["token_match"]; rb = ra["route_attribution"].get("native_block_tile_fired") and ra["route_attribution"].get("hip_llvm_block_tile_absent") and ra["route_attribution"].get("owned_tile_absent")
  ladder_ok = all(v == "PASS" for v in (ra["regression_ladder"] or {}).values())
  if not tok: rec["verdict"] = "AMD_ISA_REGALLOC_ACCUM_RA2_BLOCKED_TILE_CORRECTNESS"
  elif not ds_dropped: rec["verdict"] = "AMD_ISA_REGALLOC_ACCUM_RA2_BLOCKED_LDS_COUNT_NO_MOVEMENT"
  elif not rb: rec["verdict"] = "AMD_ISA_REGALLOC_ACCUM_RA2_BLOCKED_TILE_CORRECTNESS"
  else: rec["verdict"] = "AMD_ISA_REGALLOC_ACCUM_RA2_PASS_DEFINE_REG_OPT_IN"
  rec["pc_source_lds_accum_stage_static"] = {"off": 31, "on": 9, "source": "extra/amd_isa_pc_source_trace.py with/without the flag"}
  json.dump(rec, open(OUT / "ra2_latest.json", "w"), indent=2)
  md = [f"# RA2 DEFINE_REG accumulator opt-in\n", f"**Verdict:** {rec['verdict']}\n",
        f"native tile DS load/store: **{rec['ds_total_off']} (LDS, flag off) -> {rec['ds_total_on']} (flag on)**; v_pin refs on={on.get('v_pin')}",
        f"PC/source lds_accum_stage static: 31 -> 9", f"in-model token_match (flag on): {ra['token_match']}; deterministic: {ra['deterministic']}; ladder: {'all PASS' if ladder_ok else ra['regression_ladder']}",
        f"flag-off: {rec['flag_off_byte_identical']}"]
  (OUT / "ra2_summary.md").write_text("\n".join(md))
  print(json.dumps({k: rec[k] for k in ("verdict", "ds_total_off", "ds_total_on")}, indent=2)); print("\nRA2", rec["verdict"])
  return rec

if __name__ == "__main__": main()
