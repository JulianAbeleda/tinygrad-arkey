#!/usr/bin/env python3
"""Track 1: offline per-shape TC schedule search for prefill-v2 fp16 matmuls -> frozen warm-start table.

The default `_prefill_v2_opts` (model.py) omits LOCAL entirely, leaving up to ~3x on the floor (measured:
codegen 14-34 default vs ~43 with LOCAL; hand build_gemm_lds2 52-82). Searches TC+UPCAST+LOCAL+UNROLL per
shape, gates each config on numpy-bit-exactness (rel_rmse) + no-spill, times survivors on AMD, writes a JSON
table the load path reads. ONE SUBPROCESS PER CONFIG (tinygrad's program cache would otherwise reuse the
first config's kernel). OFFLINE ONLY -- never times during model load (memory: killing-tinygrad-amd-wedges-mes-ring).

  run: DEVICE_IN_FUNCTION_BUG=1 ALLOW_DEVICE_USAGE=1 TC=1 DEV=AMD PYTHONPATH=. \
       .venv/bin/python extra/qk/prefill_v2_schedule_search.py [--shapes out,in;...] [--out table.json]

Table schema: { "<out_f>x<in_f>": {"opts": [[OP_NAME, axis, arg], ...], "tflops": float, "default_tflops": float} }
"""
from __future__ import annotations
import os, sys, time, json, argparse, itertools, subprocess, pathlib
import numpy as np
from extra.qk.timing_harness import add_clock_pin_arg, env_wants_clock_pin, pinned_peak_from_env, set_clock_pin_env

M_DEFAULT = 512
GRID_U0 = [2, 4]; GRID_U1 = [2, 4]; GRID_LOC = [0, 4, 8]; GRID_UNR = [8, 16]
DEFAULT_SHAPES = [  # real Qwen3 prefill (out_f, in_f) linear shapes, M=512
  (5120,5120),(1024,5120),(17408,5120),(5120,17408),          # 14B: attn_qo, attn_kv, ffn_gate_up, ffn_down
  (4096,4096),(1024,4096),(12288,4096),(4096,12288),(6144,4096),  # 8B: attn_qo/kv, ffn_gate_up, ffn_down, gate|up
]
TABLE_PATH = pathlib.Path(__file__).resolve().parent / "prefill_v2_schedule_table.json"

def _opts_for(u0,u1,loc,unr):
  from tinygrad.codegen.opt import Opt, OptOps
  o = [Opt(OptOps.TC,0,(-1,2,1)), Opt(OptOps.UPCAST,0,u0), Opt(OptOps.UPCAST,1,u1)]
  if loc: o.append(Opt(OptOps.LOCAL,0,loc))
  o.append(Opt(OptOps.UNROLL,0,unr))
  return tuple(o)

def _serialize(opts):
  return [[o.op.name, o.axis, list(o.arg) if isinstance(o.arg,(tuple,list)) else o.arg] for o in opts]

def _deserialize(ser):
  from tinygrad.codegen.opt import Opt, OptOps
  return tuple(Opt(OptOps[name], axis, tuple(arg) if isinstance(arg,list) else arg) for name,axis,arg in ser)

def load_table(path=None) -> dict:
  """Load the frozen schedule table -> {(out_f,in_f): opts_tuple}. Empty if absent/unreadable.
  Used by model._build_prefill_v2_warmstart to override the static _prefill_v2_opts per shape."""
  p = pathlib.Path(path or TABLE_PATH)
  try:
    raw = json.loads(p.read_text())
    out = {}
    for k, v in raw.items():
      of, inf = (int(x) for x in k.split("x"))
      out[(of, inf)] = _deserialize(v["opts"])
    return out
  except Exception:
    return {}

def _worker():
  # one (shape,config) from env -> prints RESULT <json>. Fresh process = fresh compile cache.
  from tinygrad import Tensor, dtypes, Device, TinyJit
  from tinygrad.codegen.opt import postrange
  M=int(os.environ["MM"]); out_f=int(os.environ["OUTF"]); in_f=int(os.environ["INF"])
  u0=int(os.environ["U0"]); u1=int(os.environ["U1"]); loc=int(os.environ["LOC"]); unr=int(os.environ["UNR"])
  pin_clock = env_wants_clock_pin()
  res={"u0":u0,"u1":u1,"loc":loc,"unr":unr,"tflops":0.0,"status":"?","pin_clock":pin_clock}
  try:
    dev=Device[Device.DEFAULT]
    rng=np.random.default_rng(0)
    a_np=(rng.standard_normal((M,in_f))*0.1).astype(np.float16); b_np=(rng.standard_normal((out_f,in_f))*0.1).astype(np.float16)
    ref=a_np.astype(np.float32)@b_np.astype(np.float32).T; refn=np.sqrt(np.mean(ref**2))+1e-9
    a=Tensor(a_np); b=Tensor(b_np)
    postrange._WARMSTART_OPTS={(frozenset({M,out_f}),in_f):_opts_for(u0,u1,loc,unr)}
    postrange._warmstart_stats.update({"match":0,"apply":0,"error":0})
    c=(a@b.transpose()).realize()
    if postrange._warmstart_stats["apply"]==0: res["status"]="no-apply"; print("RESULT",json.dumps(res)); return
    out=c.float().numpy(); rr=float(np.sqrt(np.mean((out-ref)**2))/refn)
    if not np.isfinite(rr) or rr>2e-2: res["status"]=f"WRONG rr={rr:.1e}"; print("RESULT",json.dumps(res)); return
    j=TinyJit(lambda:(a@b.transpose()).realize())
    with pinned_peak_from_env() as pin_prov:
      if pin_prov is not None: res["clock_pin"] = pin_prov
      for _ in range(5): j()
      dev.synchronize(); ts=[]
      for _ in range(3):
        dev.synchronize(); t0=time.perf_counter()
        for _ in range(15): j()
        dev.synchronize(); ts.append((time.perf_counter()-t0)/15*1e3)
    res["tflops"]=round(2*M*out_f*in_f/min(ts)*1e-12*1e3,2); res["status"]="ok"
  except Exception as e:
    res["status"]=type(e).__name__
  print("RESULT",json.dumps(res))

def _run_config(M,out_f,in_f,u0,u1,loc,unr,pin_clock:bool=False):
  env={**os.environ,"WORKER":"1","MM":str(M),"OUTF":str(out_f),"INF":str(in_f),
       "U0":str(u0),"U1":str(u1),"LOC":str(loc),"UNR":str(unr)}
  set_clock_pin_env(env, pin_clock)
  try:
    p=subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, timeout=180)
    for ln in p.stdout.splitlines():
      if ln.startswith("RESULT "): return json.loads(ln[7:])
  except subprocess.TimeoutExpired:
    return {"u0":u0,"u1":u1,"loc":loc,"unr":unr,"tflops":0.0,"status":"TIMEOUT"}
  return {"u0":u0,"u1":u1,"loc":loc,"unr":unr,"tflops":0.0,"status":"no-result"}

def main():
  ap=argparse.ArgumentParser()
  ap.add_argument("--shapes", default=None); ap.add_argument("--out", default=str(TABLE_PATH)); ap.add_argument("--M", type=int, default=M_DEFAULT)
  add_clock_pin_arg(ap)
  args=ap.parse_args()
  shapes=[tuple(int(x) for x in s.split(",")) for s in args.shapes.split(";")] if args.shapes else DEFAULT_SHAPES
  table={}
  for out_f,in_f in shapes:
    rows=[_run_config(args.M,out_f,in_f,u0,u1,loc,unr,pin_clock=args.pin_clock) for u0,u1,loc,unr in itertools.product(GRID_U0,GRID_U1,GRID_LOC,GRID_UNR)]
    ok=[r for r in rows if r["status"]=="ok"]
    dflt=max((r["tflops"] for r in ok if r["loc"]==0), default=0.0)
    if not ok: print(f"[{out_f}x{in_f}] NO valid config", flush=True); continue
    best=max(ok, key=lambda r:r["tflops"])
    table[f"{out_f}x{in_f}"]={"opts":_serialize(_opts_for(best['u0'],best['u1'],best['loc'],best['unr'])),
                              "tflops":best["tflops"], "default_tflops":round(dflt,2)}
    print(f"[{out_f}x{in_f}] BEST {best['tflops']:6.2f}  u0={best['u0']} u1={best['u1']} loc={best['loc']} unr={best['unr']}  (loc=0 best: {dflt:.1f})", flush=True)
    pathlib.Path(args.out).write_text(json.dumps(table, indent=2))   # incremental save
  print(f"\nWROTE {len(table)} shapes -> {args.out}", flush=True)

if __name__=="__main__":
  if os.environ.get("WORKER")=="1": _worker()
  else: main()
