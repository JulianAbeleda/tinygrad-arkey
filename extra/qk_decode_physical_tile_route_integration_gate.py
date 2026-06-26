#!/usr/bin/env python3
"""Route integration gate for decode_attention_physical_tile_p1.

This is the first actual route candidate using a missing physical primitive in-model:
DECODE_ATTN_PHYSICAL_TILE_P1_SCORE=1 swaps the generated score stage to a lane-sharded/cross-lane q.k score kernel.
The gate validates route hygiene, standalone score numeric, emitted ISA primitives, and records W==D readiness.
"""
from __future__ import annotations

import ctypes, json, os, pathlib, re, subprocess, sys, time
from typing import Any
import numpy as np
from tinygrad import Tensor, dtypes, Device

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-space"
TARGET = "flash_p1_crosslane_score_whole_cache_32_128"
CLEAR_FLAGS = (
  "DECODE_ATTN_GENERATED_SKELETON", "DECODE_ATTN_GENERATED_WHOLECACHE", "DECODE_ATTN_SCORE_VDOT2",
  "DECODE_ATTN_SCORE_XLANE", "DECODE_ATTN_TILE_PLACEHOLDER", "DECODE_ATTN_TILE_SCORE_MAX",
  "DECODE_ATTN_TILE_PROB", "DECODE_ATTN_TILE_PARTIAL_PV", "DECODE_ATTN_TILE_PROB_PARTIAL_PV",
  "DECODE_ATTN_ONLINE_PV_TILE", "DECODE_ATTN_ONLINE_STATE_PV_TILE", "DECODE_ATTN_ONLINE_STATE_PV_TILE_XLANE",
  "DECODE_ATTN_ONLINE_STATE_SPLIT_XLANE", "DECODE_ATTN_FUSED_PV_TILE", "DECODE_ATTN_FUSED_SCORE_STATE_PV_TILE",
  "DECODE_ATTN_PHYSICAL_TILE_P1_SCORE", "V_DOT2_LOWERING", "WARP_REDUCE_LOWERING",
)

def _env(arm:str) -> dict[str,str]:
  env={**os.environ,"PYTHONPATH":str(ROOT),"QK_P1_ROUTE_CHILD":"1","QK_P1_ROUTE_ARM":arm}
  for k in CLEAR_FLAGS: env[k]="0"
  if arm == "p1_route":
    env["DECODE_ATTN_GENERATED_WHOLECACHE"]="1"
    env["DECODE_ATTN_PHYSICAL_TILE_P1_SCORE"]="1"
  return env

def _programs(route:dict[str,Any]) -> list[str]: return list(route["route_fire"]["program_node_names"])

def _signature(route:dict[str,Any]) -> dict[str,Any]:
  names=_programs(route); gen=[n for n in names if n.startswith("flash_")]
  return {"generated_attention_programs":gen,"has_target":any(n.startswith(TARGET) for n in gen),
          "has_old_score":any(n.startswith("flash_score_whole_cache") for n in gen),
          "has_owned":route["route_counts"]["owned_flash_tile_gqa_whole"] or route["route_counts"]["owned_flash_combine"]}

def _child_route(arm:str) -> dict[str,Any]:
  from extra.qk_decode_attention_purity_capture import capture
  route=capture("a2" if arm == "p1_route" else "baseline")
  return {"arm":arm,"route":route,"signature":_signature(route)}

def _run_child(arm:str) -> dict[str,Any]:
  p=subprocess.run([sys.executable,str(pathlib.Path(__file__).resolve())],cwd=ROOT,env=_env(arm),text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  if p.returncode != 0: return {"arm":arm,"failed":True,"returncode":p.returncode,"output_tail":(p.stdout or "")[-8000:]}
  for line in reversed((p.stdout or "").splitlines()):
    try: return json.loads(line)
    except Exception: pass
  return {"arm":arm,"failed":True,"returncode":0,"error":"no json","output_tail":(p.stdout or "")[-8000:]}

def _parse_desc(lib:bytes)->dict[str,Any]:
  from tinygrad.runtime.support.elf import elf_loader
  from tinygrad.runtime.autogen import amdgpu_kd
  image, sections, _ = elf_loader(lib)
  rodata_entry=next((sh.header.sh_addr for sh in sections if sh.name == ".rodata"), -1)
  desc_sz=ctypes.sizeof(amdgpu_kd.llvm_amdhsa_kernel_descriptor_t)
  desc=amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(bytes(image[rodata_entry:rodata_entry+desc_sz]))
  rsrc1=desc.compute_pgm_rsrc1; gran_vgpr=rsrc1 & 0x3f; gran_sgpr=(rsrc1 >> 6) & 0xf
  return {"vgpr":(gran_vgpr+1)*8,"sgpr":(gran_sgpr+1)*8,"lds":desc.group_segment_fixed_size,"scratch":desc.private_segment_fixed_size,"kernarg":desc.kernarg_size,"rsrc1":hex(rsrc1)}

def _disasm(lib:bytes)->str:
  from tinygrad.helpers import system
  objdump="/opt/rocm/llvm/bin/llvm-objdump"
  if not pathlib.Path(objdump).exists(): objdump="llvm-objdump"
  return system(f"{objdump} -d -", input=lib)

def _hist(asm:str)->dict[str,int]:
  h={"total":0,"valu":0,"s_inst":0,"vmem_load":0,"vmem_store":0,"ds":0,"cross_lane":0,"fma_dot":0,"scratch":0}
  for line in asm.splitlines():
    m=re.search(r"\b([sv]_[a-z0-9_]+|global_[a-z0-9_]+|buffer_[a-z0-9_]+|ds_[a-z0-9_]+|scratch_[a-z0-9_]+)\b", line)
    if not m: continue
    op=m.group(1); h["total"]+=1
    if op.startswith("v_"): h["valu"]+=1
    if op.startswith("s_"): h["s_inst"]+=1
    if op.startswith("global_load") or op.startswith("buffer_load"): h["vmem_load"]+=1
    if op.startswith("global_store") or op.startswith("buffer_store"): h["vmem_store"]+=1
    if op.startswith("ds_"): h["ds"]+=1
    if op.startswith(("ds_bpermute","ds_permute","ds_swizzle")) or op.startswith("v_permlane"): h["cross_lane"]+=1
    if "fma" in op or "dot" in op or "mac" in op: h["fma_dot"]+=1
    if op.startswith("scratch_"): h["scratch"]+=1
  return h

def _score_numeric_and_isa() -> dict[str,Any]:
  from extra.qk_flash_decode import flash_p1_crosslane_score_whole_cache_kernel
  dev=Device[Device.DEFAULT]; captured={}; orig=dev.runtime
  def hook(name, lib, **kw):
    if name.startswith(TARGET) and name not in captured: captured[name]=lib
    return orig(name, lib, **kw)
  dev.runtime=hook
  Hq,Hkv,Hd,MAXC,Tc=32,8,128,256,192
  rng=np.random.default_rng(20260626)
  q=rng.normal(0,0.25,(Hq,Hd)).astype(np.float32)
  cache=np.zeros((2,Hkv,MAXC,Hd),np.float32); cache[0]=rng.normal(0,0.25,(Hkv,MAXC,Hd)).astype(np.float32)
  got=Tensor.empty(Hq*MAXC,dtype=dtypes.float32).custom_kernel(Tensor(q.reshape(-1)),Tensor(cache.reshape(-1)),fxn=flash_p1_crosslane_score_whole_cache_kernel(Hd,Hq,Hkv,MAXC,Tc))[0].realize().numpy().reshape(Hq,MAXC)
  ref=np.zeros((Hq,MAXC),np.float32)
  for h in range(Hq): ref[h,:Tc]=(cache[0,h//(Hq//Hkv),:Tc,:] @ q[h])*(1.0/np.sqrt(Hd))
  diff=got[:,:Tc]-ref[:,:Tc]
  kernels={}; OUT.mkdir(parents=True, exist_ok=True)
  for name,lib in captured.items():
    asm=_disasm(lib); (OUT/f"disasm_{name}.txt").write_text(asm)
    d=_parse_desc(lib); d["hist"]=_hist(asm); d["primitive_flags"]={"has_v_dot2":"v_dot2" in asm,"has_lds":bool(re.search(r"\bds_(load|store|read|write)",asm)),"has_cross_lane":bool(re.search(r"\b(ds_bpermute|ds_permute|ds_swizzle|v_permlane)",asm)),"has_vector_global_load":"global_load" in asm or "buffer_load" in asm,"has_spill":bool(re.search(r"\bscratch_(load|store)",asm))}; kernels[name]=d
  return {"checked":True,"numeric":{"max_abs":float(np.max(np.abs(diff))),"rmse":float(np.sqrt(np.mean(diff*diff))),"pass":bool(float(np.max(np.abs(diff))) <= 1e-4)},"kernels":kernels}

def build()->dict[str,Any]:
  baseline=_run_child("baseline"); p1=_run_child("p1_route")
  score=_score_numeric_and_isa()
  route_gate={"checked":True,"pass":False}
  if not baseline.get("failed") and not p1.get("failed"):
    route=p1["route"]; sig=p1["signature"]
    route_gate={"checked":True,"pass": bool(route["verdict"] == "DECODE_ATTENTION_A2_GENERATED_WHOLECACHE_ROUTE_CLEAN" and sig["has_target"] and not sig["has_old_score"] and not sig["has_owned"] and baseline["route"]["tokens_sample"] == route["tokens_sample"] and not route["materialization"]["E_49152_present"]),"baseline":baseline,"p1_route":p1}
  flags=next(iter(score.get("kernels",{}).values()),{}).get("primitive_flags",{})
  if not score["numeric"].get("pass"): verdict="P1_ROUTE_FAIL__SCORE_NUMERIC"
  elif not route_gate.get("pass"): verdict="P1_ROUTE_FAIL__ROUTE_GATE"
  elif not flags.get("has_cross_lane"): verdict="P1_ROUTE_FAIL__NO_CROSSLANE_ISA"
  else: verdict="P1_ROUTE_CLEAN__WD_REQUIRED"
  return {"date":"2026-06-26","timestamp":time.strftime("%Y%m%d-%H%M%S"),"candidate_id":"decode_attention_physical_tile_p1_route","verdict":verdict,"flags":{"DECODE_ATTN_GENERATED_WHOLECACHE":"1","DECODE_ATTN_PHYSICAL_TILE_P1_SCORE":"1"},"route_gate":route_gate,"score_probe":score,"decision":"Run W==D only if route is clean; this is partial route integration (CrossLane/LaneMap), not full all-primitives route."}

def main()->int:
  os.chdir(ROOT)
  if os.environ.get("QK_P1_ROUTE_CHILD") == "1":
    print(json.dumps(_child_route(os.environ.get("QK_P1_ROUTE_ARM","baseline"))))
    return 0
  OUT.mkdir(parents=True, exist_ok=True)
  out=build(); (OUT/"route_p1_latest.json").write_text(json.dumps(out,indent=2)+"\n"); (OUT/f"route-p1-{out['timestamp']}.json").write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2)); return 0 if not out["verdict"].startswith("P1_ROUTE_FAIL") else 1
if __name__ == "__main__": raise SystemExit(main())
