"""AMD ISA backend — Phase N1B: scalarize wave-uniform index/address math (conservative SALU datapath).

Implements an opt-in (AMD_ISA_N1B=1) scalar datapath in AMDISARenderer: a pre-isel pass marks int ADD/MUL/SHL whose
inputs are all wave-uniform (CONST/RANGE/DEFINE_VAR/gidx), and isel lowers them to s_mul_i32/s_add_i32/s_lshl_b32 in a
scalar-temp SGPR pool (s64..s103), bridging to VGPR with MOV_S2V only at the vector boundary (gidx via S_WGID s_mov).

EVIDENCE-BASED OUTCOME (verdict AMD_ISA_PHASE_N1B_BLOCKED_SGPR_REGALLOC; kept OPT-IN, default OFF):
  - the scalar datapath is correct IN ISOLATION (microgates: uniform base + lane-offset address PASS; divergent/lidx
    math stays VALU and is correct).
  - on the REAL decode tile the lowering scalarizes 4 s_mul_i32 / counter chains, BUT those results are DEAD: the live
    address path uses the OOB-clamped token index (where()->v_cndmask, a VECTOR op), so the address consumes the
    clamped VGPR, not the uniform pre-clamp value. No live vector address op is removed (VALU 197 -> 189 is dead code).
  - enabling it (AMD_ISA_N1B=1) triggers an MMU fault on the tile (an unresolved SGPR-datapath/regalloc interaction)
    for ZERO live benefit. So it is left default-off; N1A (+27%, native ~59-61% of owned) is the shipped state.

Conclusion: the pinned VALU was already addressed by N1A (native VALU 197 < owned 219). The remaining gap is NOT
uniform-address-math co-issue: the uniform prefixes are behind a vector clamp on the live path. Next lever is dynamic
(memory bandwidth / loop-trip via SQTT), not more static VALU/SALU rebalancing.

Run:  DEV=AMD PYTHONPATH=. .venv/bin/python extra/amd_isa_phase_n1b_gate.py
Writes: bench/amd-isa-backend-phase-n1b/latest.json  (+ uniformity_audit.json already present)
"""
import os, json, pathlib, subprocess, sys, re
ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "bench/amd-isa-backend-phase-n1b/latest.json"

def _microgates():
  # base+lane-offset address correctness, and divergent (lidx) safety. Run under DEV=AMD:ISA in-process.
  os.environ["DEV"] = "AMD:ISA"; os.environ["AMD_ISA_N1B"] = "1"
  import numpy as np
  from tinygrad import Tensor, dtypes
  from tinygrad.uop.ops import UOp, Ops, KernelInfo
  i32 = dtypes.int32; res = {}
  # MG2: address = uniform_base(gidx)*L + lane(lidx); value = gidx*100 + lidx
  L = 4
  def f2(o):
    g = UOp.special(2, "gidx0"); l = UOp.special(L, "lidx0")
    idx = g.cast(i32) * UOp.const(i32, L) + l.cast(i32)
    val = (g.cast(i32) * UOp.const(i32, 100) + l.cast(i32)).cast(dtypes.float32)
    return o.index(idx).store(val).sink(arg=KernelInfo(name="ub", opts_to_apply=()))
  got = Tensor.custom_kernel(Tensor.empty(2 * L, device="AMD"), fxn=f2)[0].numpy()
  exp = np.array([g * 100 + l for g in range(2) for l in range(L)], dtype=np.float32)
  res["uniform_base_lane_offset"] = {"correct": bool(np.array_equal(got, exp)), "verdict": "AMD_ISA_PHASE_N1B_UNIFORM_BASE_LANE_OFFSET_PASS" if np.array_equal(got, exp) else "FAIL"}
  # MG3: pure lidx math must stay correct (never illegally scalarized): out[lidx] = lidx*3 + 1
  N = 8
  def f3(o):
    l = UOp.special(N, "lidx0"); val = (l.cast(i32) * UOp.const(i32, 3) + UOp.const(i32, 1)).cast(dtypes.float32)
    return o.index(l).store(val).sink(arg=KernelInfo(name="dv", opts_to_apply=()))
  got3 = Tensor.custom_kernel(Tensor.empty(N, device="AMD"), fxn=f3)[0].numpy(); exp3 = (np.arange(N) * 3 + 1).astype(np.float32)
  res["divergent_safety"] = {"correct": bool(np.array_equal(got3, exp3)), "verdict": "AMD_ISA_PHASE_N1B_DIVERGENT_SAFETY_PASS" if np.array_equal(got3, exp3) else "FAIL"}
  return res

def _tile_static(n1b):
  # compile the native tile with N1B on/off (subprocess; capture emitted scalar ops + VALU/SALU)
  code = (
    "import os;os.environ['DEV']='AMD';os.environ['AMD_ISA_N1B']='%d';import re,json\n"
    "from tinygrad.uop.ops import UOp,Ops\nfrom tinygrad.renderer.isa.amd import AMDISARenderer\nfrom tinygrad.helpers import getenv\n"
    "cap=[];_o=AMDISARenderer.asm\n"
    "def spy(self,prg,lin):\n"
    " ins=list(lin.src)\n"
    " if getenv('AMD_ISA_SCHED',1): ins=self._schedule(ins)\n"
    " cap.append('\\n'.join(str(u.arg) for u in self._resolve_labels(self._insert_waitcnt(ins)) if u.op is Ops.INS));return _o(self,prg,lin)\n"
    "AMDISARenderer.asm=spy\n"
    "from extra.qk_native_isa_block_tile_graph_node import compile_block_tile_isa,_compile\n"
    "_compile.cache_clear();compile_block_tile_isa(128,32,8,4608,96,48,UOp.variable('start_pos',0,4607)+1)\n"
    "a=cap[-1];c=lambda p:len(re.findall(p,a))\n"
    "valu=sum(1 for l in a.splitlines() if re.match(r'v_',l.strip()))\n"
    "salu=sum(1 for l in a.splitlines() if re.match(r's_',l.strip()))\n"
    "# scalar-temp results (s64+) that are never read = dead\n"
    "defs=set(re.findall(r's_(?:mul_i32|add_i32|lshl_b32)\\(s\\[(6[4-9]|[7-9][0-9]|10[0-3])\\]',a))\n"
    "reads=set(re.findall(r',\\s*s\\[(6[4-9]|[7-9][0-9]|10[0-3])\\]',a))\n"
    "print('@@'+json.dumps({'s_mul_i32':c(r'\\bs_mul_i32'),'valu':valu,'salu':salu,'total':len(a.splitlines()),"
    "'scalar_temp_defs':sorted(defs),'scalar_temp_reads':sorted(reads),'dead_scalar_temps':sorted(defs-reads)}))\n") % n1b
  out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env={**os.environ}, capture_output=True, text=True, timeout=300).stdout
  return json.loads([l for l in out.splitlines() if l.startswith("@@")][-1][2:])

def _tile_runtime(n1b):
  # run the block-tile microgate with N1B on/off in a subprocess; capture MMU-fault / PASS
  env = {**os.environ, "DEV": "AMD:ISA", "AMD_ISA_N1B": str(n1b)}
  p = subprocess.run([sys.executable, "extra/qk_decode_attention_block_tile_microgate.py"], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=400)
  out = (p.stdout or "") + (p.stderr or "")
  if "MMU fault" in out: return "MMU_FAULT"
  if "BLOCK_TILE_MICROGATE_PASS" in out: return "PASS"
  return "OTHER"

def main():
  rec = {"command": "DEV=AMD PYTHONPATH=. .venv/bin/python extra/amd_isa_phase_n1b_gate.py",
         "scope": "Phase N1B: wave-uniform address-math scalarization (opt-in AMD_ISA_N1B, default OFF)"}
  rec["audit"] = json.load(open(ROOT / "bench/amd-isa-backend-phase-n1b/uniformity_audit.json"))["totals"]
  rec["microgates"] = _microgates()
  rec["tile_static_off"] = _tile_static(0); rec["tile_static_on"] = _tile_static(1)
  rec["tile_runtime_off"] = _tile_runtime(0); rec["tile_runtime_on"] = _tile_runtime(1)
  on = rec["tile_static_on"]
  rec["dead_code_finding"] = (f"with N1B on, scalar-temp defs={on['scalar_temp_defs']} reads={on['scalar_temp_reads']} -> "
    f"dead={on['dead_scalar_temps']}. The scalarized uniform results are DEAD: the live address path uses the OOB-clamped "
    "token index (where()->v_cndmask, VECTOR), so no live vector address op is removed.")
  rec["mechanism_works_in_isolation"] = (rec["microgates"]["uniform_base_lane_offset"]["correct"] and rec["microgates"]["divergent_safety"]["correct"])
  rec["enabling_faults"] = rec["tile_runtime_on"] == "MMU_FAULT"
  rec["default_off_correct"] = rec["tile_runtime_off"] == "PASS"
  rec["wd_note"] = "not measured: N1B on faults; and the scalarized ops are dead (no live VALU removed) so W==D movement would be ~0. N1A baseline (native 61.09/57.92, 58.9%/61.1% of owned) is the shipped state."
  # verdict: concrete blocker is the SGPR-datapath fault when enabled; root finding is the dead-code (targets off the live path)
  if not rec["default_off_correct"]: rec["verdict"] = "AMD_ISA_PHASE_N1B_BLOCKED_TOKEN_MATCH"
  elif not rec["mechanism_works_in_isolation"]: rec["verdict"] = "AMD_ISA_PHASE_N1B_BLOCKED_MOV_S2V_BRIDGE"
  elif rec["enabling_faults"]: rec["verdict"] = "AMD_ISA_PHASE_N1B_BLOCKED_SGPR_REGALLOC"
  else: rec["verdict"] = "AMD_ISA_PHASE_N1B_PASS_STATIC_IMPROVEMENT_NO_WD_MOVEMENT"
  rec["root_finding"] = ("the uniform address prefixes are NOT on the live address path (they are recomputed through a "
    "vector OOB-clamp), so scalarization is dead code; combined with an unresolved SGPR-datapath fault when enabled, "
    "N1B is not a beneficial transform for this tile. Next lever is dynamic (memory/loop-trip via SQTT), not static SALU rebalancing.")
  return rec

if __name__ == "__main__":
  rec = main()
  ART.parent.mkdir(parents=True, exist_ok=True)
  with open(ART, "w") as f: json.dump(rec, f, indent=2)
  print(json.dumps(rec, indent=2)); print("\nPHASE_N1B", rec["verdict"])
