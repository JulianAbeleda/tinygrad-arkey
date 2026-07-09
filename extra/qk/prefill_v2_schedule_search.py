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
import os, sys, time, json, argparse, itertools, subprocess, pathlib, traceback
import numpy as np
from extra.qk.timing_harness import add_clock_pin_arg, env_wants_clock_pin, pinned_peak_from_env, set_clock_pin_env

M_DEFAULT = 512
GRID_U0 = [2, 4]; GRID_U1 = [2, 4]; GRID_LOC = [0, 4, 8]; GRID_UNR = [8, 16]
DEFAULT_SHAPES = [  # real Qwen3 prefill (out_f, in_f) linear shapes, M=512
  (5120,5120),(1024,5120),(17408,5120),(5120,17408),          # 14B: attn_qo, attn_kv, ffn_gate_up, ffn_down
  (4096,4096),(1024,4096),(12288,4096),(4096,12288),(6144,4096),  # 8B: attn_qo/kv, ffn_gate_up, ffn_down, gate|up
]
TABLE_PATH = pathlib.Path(__file__).resolve().parent / "prefill_v2_schedule_table.json"

def _allow_parked_4x4() -> bool:
  return os.environ.get("PREFILL_ALLOW_PARKED_4X4", "0").strip() == "1"

def _candidate_grid():
  for u0,u1,loc,unr in itertools.product(GRID_U0,GRID_U1,GRID_LOC,GRID_UNR):
    if (u0,u1) == (4,4) and not _allow_parked_4x4(): continue
    yield u0,u1,loc,unr

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

def _compile_resource_summary(M,out_f,in_f,u0,u1,loc,unr):
  from tinygrad import Tensor, dtypes, Device
  from tinygrad.codegen import to_program, to_program_cache
  from tinygrad.codegen.opt import postrange
  from tinygrad.uop.ops import Ops
  from extra.qk.prefill.native_isa_l4_stream_probe import _resource_summary
  prg = _compile_native_program(M,out_f,in_f,u0,u1,loc,unr)
  out=_resource_summary(prg, 65536)
  out.pop("define_rows", None)
  out["warmstart_apply_count"]=postrange._warmstart_stats["apply"]
  return out

def _compile_native_program(M,out_f,in_f,u0,u1,loc,unr):
  from tinygrad import Tensor, dtypes, Device
  from tinygrad.codegen import to_program, to_program_cache
  from tinygrad.codegen.opt import postrange
  from tinygrad.uop.ops import Ops
  to_program_cache.clear()
  postrange._WARMSTART_OPTS={(frozenset({M,out_f}),in_f):_opts_for(u0,u1,loc,unr)}
  postrange._warmstart_stats.update({"match":0,"apply":0,"error":0})
  a=Tensor.empty(M,in_f,dtype=dtypes.half); b=Tensor.empty(out_f,in_f,dtype=dtypes.half)
  ast=[u for u in (a@b.transpose()).schedule_linear().toposort() if u.op is Ops.SINK][0]
  return to_program(ast, Device[Device.DEFAULT].renderer)

def _reg_key(reg):
  if reg is None or not hasattr(reg, "offset"): return None
  return ("v", reg.offset-256) if 256 <= reg.offset < 512 else ("s", reg.offset)

def _const_value(x):
  if isinstance(x, int): return x
  s = str(x)
  return int(s) if s.lstrip("-").isdigit() else None

def _interval_add(a, b): return (a[0]+b[0], a[1]+b[1])
def _interval_mul(a, k): return (a[0]*k, a[1]*k) if k >= 0 else (a[1]*k, a[0]*k)
def _interval_lshl(a, k): return _interval_mul(a, 1 << k)
def _interval_lshr(a, k): return (max(0, a[0]) >> k, max(0, a[1]) >> k)

def _final_stream_address_proof(M,out_f,in_f,u0,u1,loc,unr):
  from tinygrad.uop.ops import Ops
  from tinygrad.renderer.amd.elf import group_segment_fixed_size_from_elf
  prg = _compile_native_program(M,out_f,in_f,u0,u1,loc,unr)
  lin = next(u for u in prg.src if u.op is Ops.LINEAR)
  info = prg.arg
  group_segment = next((group_segment_fixed_size_from_elf(u.arg) for u in prg.src if u.op is Ops.BINARY), None)
  # Kernarg pointer order for matmul output: C(out), A(lhs), B(rhs).
  extents = {0: M*out_f*2, 1: M*in_f*2, 2: out_f*in_f*2}
  names = {0: "C", 1: "A", 2: "B"}
  sgpr_ptr: dict[int, int] = {}
  vrange: dict[tuple[str, int], tuple[int, int]] = {("v", 0): (0, max(info.local_size[0]-1, 0))}
  srange: dict[tuple[str, int], tuple[int, int]] = {
    ("s", 2): (0, max(info.global_size[0]-1, 0)),
    ("s", 3): (0, max(info.global_size[1]-1, 0)),
    ("s", 4): (0, max(info.global_size[2]-1, 0)),
  }
  unknown_defs, checked, violations, addr64_warnings = 0, [], [], []

  def rng_of(x):
    if (cv := _const_value(x)) is not None: return (cv, cv)
    if str(x) == "LIT": return None
    k = _reg_key(x)
    return vrange.get(k) or srange.get(k)
  def setr(reg, val):
    if (k := _reg_key(reg)) is not None: vrange[k] = val

  for idx, u in enumerate(lin.src):
    inst = u.arg if getattr(u, "op", None) is Ops.INS else u
    name = str(inst).split("(", 1)[0]
    try:
      if name == "s_load_b64" and getattr(inst, "sbase", None) is not None and getattr(inst, "sbase").offset == 0:
        param_idx = int(getattr(inst, "offset")) // 8
        if (k := _reg_key(getattr(inst, "sdata"))) is not None: sgpr_ptr[k[1]] = param_idx
      elif name == "s_mov_b32":
        dst = _reg_key(getattr(inst, "sdst", getattr(inst, "sdst0", None)))
        if dst is not None and (cv := _const_value(getattr(inst, "ssrc0", getattr(inst, "src0", None)))) is not None: srange[dst] = (cv, cv)
      elif name == "s_cmp_lt_i32":
        src0, src1 = getattr(inst, "ssrc0", getattr(inst, "src0", None)), getattr(inst, "ssrc1", getattr(inst, "src1", None))
        if (k := _reg_key(src0)) is not None and (cv := _const_value(src1)) is not None: srange[k] = (0, max(cv-1, 0))
      elif name == "v_mov_b32_e32":
        src0 = getattr(inst, "src0")
        val = (getattr(inst, "literal"), getattr(inst, "literal")) if str(src0) == "LIT" else rng_of(src0)
        if val is not None: setr(getattr(inst, "vdst"), val)
        else: unknown_defs += 1
      elif name == "v_and_b32_e32":
        lhs = rng_of(getattr(inst, "src0"))
        rhs = rng_of(getattr(inst, "vsrc1"))
        if lhs is not None and rhs is not None:
          # Current proof only needs power-of-two masks such as lane & 15.
          mask = lhs[0] if lhs[0] == lhs[1] else None
          setr(getattr(inst, "vdst"), (0, min(mask, rhs[1])) if mask is not None and mask >= 0 else (0, max(lhs[1], rhs[1])))
        else: unknown_defs += 1
      elif name == "v_mul_lo_u32":
        lhs = rng_of(getattr(inst, "src0"))
        rhs = getattr(inst, "literal") if str(getattr(inst, "src1")) == "LIT" else None
        if lhs is not None and rhs is not None: setr(getattr(inst, "vdst"), _interval_mul(lhs, int(rhs)))
        else: unknown_defs += 1
      elif name == "v_add_nc_u32_e32":
        lhs = (getattr(inst, "literal"), getattr(inst, "literal")) if str(getattr(inst, "src0")) == "LIT" else rng_of(getattr(inst, "src0"))
        rhs = rng_of(getattr(inst, "vsrc1"))
        if lhs is not None and rhs is not None: setr(getattr(inst, "vdst"), _interval_add(lhs, rhs))
        else: unknown_defs += 1
      elif name == "v_lshlrev_b32_e32":
        val = rng_of(getattr(inst, "vsrc1"))
        sh = _const_value(getattr(inst, "src0"))
        if val is not None and sh is not None: setr(getattr(inst, "vdst"), _interval_lshl(val, sh))
        else: unknown_defs += 1
      elif name == "v_lshrrev_b32_e32":
        val = rng_of(getattr(inst, "vsrc1"))
        sh = _const_value(getattr(inst, "src0"))
        if val is not None and sh is not None: setr(getattr(inst, "vdst"), _interval_lshr(val, sh))
        else: unknown_defs += 1
      elif name.startswith("global_load") or name.startswith("global_store"):
        addr = getattr(inst, "addr", None)
        addr_rng = rng_of(addr)
        addr_key = _reg_key(addr)
        addr_hi_rng = vrange.get((addr_key[0], addr_key[1] + 1)) if addr_key is not None and addr_key[0] == "v" else None
        saddr = _reg_key(getattr(inst, "saddr", None))
        ptr_idx = sgpr_ptr.get(saddr[1]) if saddr is not None else None
        imm = int(getattr(inst, "offset", 0))
        width = 16 if name.endswith("b128") else (2 if name.endswith("b16") or name.endswith("u16") else 4)
        row = {"idx": idx, "op": name, "ptr": None if ptr_idx is None else names.get(ptr_idx, str(ptr_idx)),
               "addr_range": addr_rng, "addr_high_range_if_64bit": addr_hi_rng, "offset": imm, "width": width,
               "extent": None if ptr_idx is None else extents.get(ptr_idx)}
        if addr_hi_rng is not None and addr_hi_rng != (0, 0):
          addr64_warnings.append(row | {"warning": "adjacent high VGPR is nonzero if GLOBAL addr is interpreted as a 64-bit vaddr pair"})
        ok = addr_rng is not None and ptr_idx in extents and addr_rng[0] + imm >= 0 and addr_rng[1] + imm + width <= extents[ptr_idx]
        row["ok"] = ok
        checked.append(row)
        if not ok: violations.append(row)
      elif name.startswith("ds_load") or name.startswith("ds_store"):
        addr_rng = rng_of(getattr(inst, "addr", None))
        imm = int(getattr(inst, "offset0", getattr(inst, "offset", 0)))
        width = 16 if name.endswith("b128") else (8 if name.endswith("b64") else (2 if name.endswith("b16") or name.endswith("u16") else 4))
        row = {"idx": idx, "op": name, "space": "LDS", "addr_range": addr_rng, "offset": imm, "width": width, "extent": group_segment}
        ok = addr_rng is not None and group_segment is not None and addr_rng[0] + imm >= 0 and addr_rng[1] + imm + width <= group_segment
        row["ok"] = ok
        checked.append(row)
        if not ok: violations.append(row)
    except Exception as e:
      unknown_defs += 1
      violations.append({"idx": idx, "op": name, "error": f"{type(e).__name__}: {e}"})
  return {
    "program": str(info.name), "global_size": info.global_size, "local_size": info.local_size, "group_segment_bytes": group_segment,
    "extents": {names[k]: v for k, v in extents.items()},
    "checked_count": len(checked), "violation_count": len(violations), "unknown_defs": unknown_defs,
    "global_addr64_high_warning_count": len(addr64_warnings),
    "global_addr64_high_warnings": addr64_warnings[:20],
    "ok": len(checked) > 0 and not violations,
    "violations": violations[:20],
    "checked_sample": checked[:20],
  }

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
    res["message"]=str(e)
    res["traceback_tail"]=traceback.format_exc().splitlines()[-8:]
    try: res["compile_resource_summary"]=_compile_resource_summary(M,out_f,in_f,u0,u1,loc,unr)
    except Exception as re:
      res["compile_resource_error"]=f"{type(re).__name__}: {re}"
    try: res["final_stream_address_proof"]=_final_stream_address_proof(M,out_f,in_f,u0,u1,loc,unr)
    except Exception as ae:
      res["final_stream_address_proof_error"]=f"{type(ae).__name__}: {ae}"
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
  return {"u0":u0,"u1":u1,"loc":loc,"unr":unr,"tflops":0.0,"status":"no-result","stdout_tail":p.stdout.splitlines()[-8:],"stderr_tail":p.stderr.splitlines()[-8:]}

def main():
  ap=argparse.ArgumentParser()
  ap.add_argument("--shapes", default=None); ap.add_argument("--out", default=str(TABLE_PATH)); ap.add_argument("--M", type=int, default=M_DEFAULT)
  add_clock_pin_arg(ap)
  args=ap.parse_args()
  shapes=[tuple(int(x) for x in s.split(",")) for s in args.shapes.split(";")] if args.shapes else DEFAULT_SHAPES
  table={}
  for out_f,in_f in shapes:
    rows=[_run_config(args.M,out_f,in_f,u0,u1,loc,unr,pin_clock=args.pin_clock) for u0,u1,loc,unr in _candidate_grid()]
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
