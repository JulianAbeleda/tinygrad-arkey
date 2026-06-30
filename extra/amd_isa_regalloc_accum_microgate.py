"""RA1 microgate — loop-carried PINNED register accumulators (AMD_ISA_REG_ACCUM=1), AMD ISA backend.

Proves the RA0 primitive on 3 minimal microkernels (NOT the decode tile): a per-thread DEFINE_REG accumulator with a
compile-time index becomes a reserved physical VGPR (v240+) carried across the loop, instead of LDS ds_load/ds_store.
Each microkernel: correct vs numpy, emits the pinned ops (v240+ present), FEWER ds_load/ds_store than the LDS path.
Off (LDS baseline) and On (pinned) run in SEPARATE subprocesses (getenv + to_program cache => the flag must be set at
process start). Scheduler on (AMD_ISA_SCHED=1 default) so the in-place RMW ordering is exercised.

Run: DEV=AMD:ISA PYTHONPATH=. .venv/bin/python extra/amd_isa_regalloc_accum_microgate.py
Writes: bench/amd-isa-backend-regalloc-accum/ra1_{latest.json,summary.md}
"""
import os, sys, json, re, pathlib, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-isa-backend-regalloc-accum"

CHILD = r'''
import os, re, json
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, AxisType, AddrSpace, KernelInfo
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.helpers import getenv
F32 = dtypes.float32
_cap = []; _orig = AMDISARenderer.asm
def _spy(self, prg, lin):
  try:
    ins = list(lin.src)
    if getenv("AMD_ISA_SCHED", 1): ins = self._schedule(ins)
    _cap.append("\n".join(str(u.arg) for u in self._resolve_labels(self._insert_waitcnt(ins)) if u.op is Ops.INS))
  except Exception: _cap.append("")
  return _orig(self, prg, lin)
AMDISARenderer.asm = _spy
def counts():
  a = "\n".join(_cap)
  return {"v_pin": len(re.findall(r"v\[(2[4-9][0-9]|25[0-5])\]", a)), "ds_load": len(re.findall(r"\bds_load", a)), "ds_store": len(re.findall(r"\bds_store", a))}
def run(fxn, inputs, n_outs, ref):
  _cap.clear()
  outs = tuple(Tensor.empty(n, device="AMD") for n in n_outs)
  got = Tensor.custom_kernel(*inputs, *outs, fxn=fxn)
  res = [got[len(inputs)+k].numpy() for k in range(len(outs))]
  ok = all(np.allclose(r, e, atol=1e-3, rtol=1e-3) for r, e in zip(res, ref))
  c = counts()
  return {"correct": bool(ok), "v_pin_refs": c["v_pin"], "ds_load": c["ds_load"], "ds_store": c["ds_store"], "sample": [float(r.flat[0]) for r in res]}
LANES, N = 32, 8
rng = np.random.default_rng(0)
x = rng.standard_normal(N*LANES).astype(np.float32); y = rng.standard_normal(N*LANES).astype(np.float32)
xm, ym = x.reshape(N, LANES), y.reshape(N, LANES)
def k_sum(ina, o):
  lane = UOp.special(LANES, "lidx0"); i = UOp.range(N, 0, AxisType.REDUCE)
  acc = UOp.placeholder((1,), F32, 200, addrspace=AddrSpace.REG); acc = acc.after(lane)[0].set(0.0)
  acc = acc[0].set(acc.after(i)[0] + ina.index(i*LANES+lane).load(), end=i)
  return o.index(lane).store(acc[0]).sink(arg=KernelInfo(name="ra1_sum", opts_to_apply=()))
def k_two(xa, ya, o0, o1):
  lane = UOp.special(LANES, "lidx0"); i0 = UOp.range(N, 0, AxisType.REDUCE); i1 = UOp.range(N, 1, AxisType.REDUCE)
  a0 = UOp.placeholder((1,), F32, 201, addrspace=AddrSpace.REG); a1 = UOp.placeholder((1,), F32, 202, addrspace=AddrSpace.REG)
  a0 = a0.after(lane)[0].set(0.0); a0 = a0[0].set(a0.after(i0)[0] + xa.index(i0*LANES+lane).load(), end=i0)
  a1 = a1.after(lane)[0].set(1.0); a1 = a1[0].set(a1.after(i1)[0]*UOp.const(F32, 0.5) + ya.index(i1*LANES+lane).load(), end=i1)
  return o0.index(lane).store(a0[0]).sink(o1.index(lane).store(a1[0]), arg=KernelInfo(name="ra1_two", opts_to_apply=()))
OUTER, INNER = 4, 3; xn = rng.standard_normal(OUTER*INNER*LANES).astype(np.float32)
def k_nested(ina, o):
  lane = UOp.special(LANES, "lidx0"); ro = UOp.range(OUTER, 0, AxisType.REDUCE); ri = UOp.range(INNER, 1, AxisType.REDUCE)
  acc = UOp.placeholder((1,), F32, 203, addrspace=AddrSpace.REG); acc = acc.after(lane)[0].set(0.0)
  acc = acc[0].set(acc.after(ro, ri)[0] + ina.index((ro*INNER+ri)*LANES+lane).load(), end=(ro, ri))
  return o.index(lane).store(acc[0]).sink(arg=KernelInfo(name="ra1_nested", opts_to_apply=()))
a1ref = np.ones(LANES, np.float32)
for i in range(N): a1ref = a1ref*0.5 + ym[i]
r = {"k_sum": run(k_sum, (Tensor(x),), (LANES,), [xm.sum(0)]),
     "k_two": run(k_two, (Tensor(x), Tensor(y)), (LANES, LANES), [xm.sum(0), a1ref]),
     "k_nested": run(k_nested, (Tensor(xn),), (LANES,), [xn.reshape(OUTER*INNER, LANES).sum(0)])}
print("@@"+json.dumps(r))
'''

def _spawn(flag):
  env = {**os.environ, "DEV": "AMD:ISA", "AMD_ISA_REG_ACCUM": str(flag), "PYTHONPATH": str(ROOT)}
  out = subprocess.run([sys.executable, "-c", CHILD], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600).stdout
  line = [l for l in out.splitlines() if l.startswith("@@")]
  return json.loads(line[-1][2:]) if line else {"error": "no output"}

def main():
  OUT.mkdir(parents=True, exist_ok=True)
  off = _spawn(0); on = _spawn(1)
  rec = {"scope": "RA1 loop-carried pinned register accumulator microgates", "lds_baseline_off": off, "pinned_on": on}
  v = "AMD_ISA_REGALLOC_ACCUM_RA1_PASS_MICROGATES"
  if "error" in on or "error" in off: v = "AMD_ISA_REGALLOC_ACCUM_RA1_BLOCKED_CORRECTNESS"
  else:
    for k in ("k_sum", "k_two", "k_nested"):
      g, b = on[k], off[k]
      if not g["correct"]: v = "AMD_ISA_REGALLOC_ACCUM_RA1_BLOCKED_CORRECTNESS"; break
      if g["v_pin_refs"] == 0: v = "AMD_ISA_REGALLOC_ACCUM_RA1_BLOCKED_FIXED_REG_REFERENCE"; break
      if (g["ds_load"] + g["ds_store"]) >= (b["ds_load"] + b["ds_store"]): v = "AMD_ISA_REGALLOC_ACCUM_RA1_BLOCKED_MOVE_OR_LOOP_CARRY"; break
  rec["verdict"] = v
  json.dump(rec, open(OUT/"ra1_latest.json", "w"), indent=2)
  md = [f"# RA1 microgates\n", f"**Verdict:** {v}\n", "| kernel | correct(on) | v_pin refs(on) | ds on | ds off(LDS) |", "|---|---|---|---|---|"]
  if "error" not in on and "error" not in off:
    for k in ("k_sum", "k_two", "k_nested"):
      md.append(f"| {k} | {on[k]['correct']} | {on[k]['v_pin_refs']} | {on[k]['ds_load']+on[k]['ds_store']} | {off[k]['ds_load']+off[k]['ds_store']} |")
  (OUT/"ra1_summary.md").write_text("\n".join(md))
  print(json.dumps(rec, indent=2)); print("\nRA1", v)
  return rec

if __name__ == "__main__": main()
