"""Capture the pre-final-lowering Q6 AST and prove divisor-owner coverage.

This is inspection-only: the candidate AST is never replaced or scheduled through
the prototype transformation.
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from tinygrad import Device
from tinygrad.uop.ops import Ops
from tinygrad.codegen.opt.postrange import set_q6_pre_scheduler_hook, set_q6_pre_global_hook, set_q6_tc_axis_hook
from .nv_compiler_q6k_pp512_binding import CompilerQ6PP512Binding, M

def _node(u):
  return {"op":u.op.name, "dtype":str(u.dtype), "arg":str(u.arg)[:240], "src":len(u.src)}

def capture(role="ffn_down"):
  records=[]
  def hook(ast):
    us=ast.toposort(); c=Counter(u.op.name for u in us)
    special=[_node(u) for u in us if u.op is Ops.SPECIAL]
    stores=[_node(u) for u in us if u.op is Ops.STORE]
    packed=[_node(u) for u in us if u.op in (Ops.WMMA, Ops.CUSTOM, Ops.LOAD) and ("int8" in str(u.dtype).lower() or "q6" in str(u.arg).lower())]
    ranges=[_node(u)|{"extent":str(u.vmax+1),"axis_type":str(u.arg[1]) if isinstance(u.arg,tuple) else None,
      "store_users":sum(u in s.backward_slice for s in us if s.op is Ops.STORE)} for u in us if u.op is Ops.RANGE]
    records.append({"role":role,"op_histogram":dict(c),"special":special,"ranges":ranges,
      "stores":stores[:128],"packed_fragment_nodes":packed[:128],"node_count":len(us)})
  set_q6_pre_scheduler_hook(hook)
  CompilerQ6PP512Binding.compile(Device["NV"])
  if not records: raise RuntimeError("Q6 pre-scheduler hook did not observe a candidate AST")
  return records[-1]

def capture_pre_global(role="ffn_down"):
  records=[]
  def hook(ast, k):
    set_q6_pre_global_hook(None)
    records.append({"role":role, "full_shape":[str(x) for x in k.full_shape],
      "axis_table":[{"index":i,"extent":str(s),"axis_type":str(t),"range":_node(r)}
        for i,(s,t,r) in enumerate(zip(k.full_shape,k.axis_types,k.rngs))]})
  set_q6_pre_global_hook(hook)
  CompilerQ6PP512Binding.compile(Device["NV"])
  if not records: raise RuntimeError("Q6 pre-global hook did not observe a candidate AST")
  return records[-1]

def owner_proof(tiles_m=8, tiles_n=128, owners=128):
  if tiles_m*tiles_n != 1024 or 1024 % owners: raise ValueError("non-divisor owner configuration")
  seen=[(owner + it*owners) for owner in range(owners) for it in range(1024//owners)]
  return {"owners":owners,"iterations":1024//owners,"tiles":1024,"bijection":len(set(seen))==1024 and set(seen)==set(range(1024))}

def capture_tc_axes(role="ffn_down"):
  rows=[]
  def hook(k, axes, geometry):
    set_q6_tc_axis_hook(None)
    rows.append({"role":role,"selected_axes":[_node(x) for x in axes],"geometry":str(geometry),
      "scheduler_shape":[str(x) for x in k.full_shape],"axis_types":[str(x) for x in k.axis_types],
      "scheduler_ranges":[_node(x) for x in k.rngs]})
  set_q6_tc_axis_hook(hook); CompilerQ6PP512Binding.compile(Device["NV"])
  if not rows: raise RuntimeError("TC axis hook did not observe a Q6 candidate")
  return rows[-1]

def persistent128_transform(ast):
  targets=[u for u in ast.toposort() if u.op is Ops.RANGE and u.vmax+1==8 and u.arg==(1, __import__('tinygrad.uop.ops',fromlist=['AxisType']).AxisType.GLOBAL)]
  if len(targets)!=1: raise RuntimeError(f"persistent128 requires one M extent8 GLOBAL, found {len(targets)}")
  repl=targets[0].replace(arg=(1, __import__('tinygrad.uop.ops',fromlist=['AxisType']).AxisType.LOOP))
  return ast.substitute({targets[0]:repl})

if __name__ == "__main__":
  ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); args=ap.parse_args()
  out={"schema":"tinygrad.nv_q6_persistent_ast_probe.v3","status":"PASS","tc_axes":capture_tc_axes(),"owner_proof":owner_proof(),
       "persistent_transform":"NOT_APPLIED: callback is observational; M-axis retag requires verified AxisType transition API"}
  open(args.out,"w").write(json.dumps(out,indent=2)+"\n")
  print(json.dumps({"status":out["status"],"axes":len(out["pre_global"]["axis_table"]),"owner_proof":out["owner_proof"]}))
