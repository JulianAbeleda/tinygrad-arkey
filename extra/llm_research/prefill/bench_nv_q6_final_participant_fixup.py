#!/usr/bin/env python3
"""Causal A/B for final-participant Q6 Stream-K reduction.

The candidate executes optional owner-head segments first.  The tile-ending
owner then retains its final accumulator values, waits for earlier contributors
to publish, reduces them in the authoritative slot order, and writes output.
No production route is modified.
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib, re, statistics
import numpy as np

from tinygrad import Device, Tensor, dtypes
from tinygrad.runtime.ops_nv import NVProgram
from extra.llm_research.layout import GGML_Q6_K, packed_u16_slice, read_metadata
from extra.llm_research.prefill.bench_nv_q6_oracle_broad_cta import _record as broad_record
from extra.llm_research.prefill.bench_nv_q6_oracle_full_streamk import (
  M, N, K, OWNERS, K256, TILES_M, TILES, ROWS, COLS, TILE_ELEMS, LAUNCH_SHARED_BYTES,
  LLAMA_MAIN_US, LLAMA_FIXUP_US, _buf, _combine_sources, _fixup_source, _ownership, _render)
from extra.llm_research.prefill.nv_compiler_q6k_imma_gate import _record as wide_record
from extra.llm_research.prefill.nv_q6_sass_census import analyze_cubin

AUTHORITATIVE_RESULT = pathlib.Path(
  "docs/task_workflow/evidence/nv-q6-oracle-full-streamk-single-launch-20260831/result.json")
CONTROL_HISTORICAL_US = 311.360


def _stats(xs: list[float]) -> dict[str, object]:
  return {"samples_us": xs, "min_us": min(xs), "median_us": statistics.median(xs), "max_us": max(xs)}


def _device_parts(source: str, index: int) -> tuple[str, str, str]:
  pattern = (r'extern "C" __global__ void __launch_bounds__\(256\) \w+\((.*?)\) \{\n'
             r'  int gidx0 = blockIdx.x; /\* 170 \*/\n')
  match = re.search(pattern, source, re.DOTALL)
  if match is None: raise RuntimeError(f"generated segment {index} signature mismatch")
  return source[:match.start()], match.group(1), source[match.end():]


def _final_participant_source(source0: str, source1: str, artifacts: pathlib.Path) -> tuple[str, str]:
  preamble, params, body0 = _device_parts(source0, 0)
  _, params1, body1 = _device_parts(source1, 1)
  if params1 != params: raise RuntimeError("generated segment ABIs differ")

  tail_start = body0.find("  int alu1169 = ")
  if tail_start < 0: raise RuntimeError("segment-zero writeback root disappeared")
  tail = body0[tail_start:]
  root = re.search(r"  int alu1169 = \((.*?)\);", tail)
  if root is None or "(gidx0<<14)" not in root.group(1): raise RuntimeError("segment-zero slot root changed")
  local_expr = root.group(1).replace("+(gidx0<<14)", "").replace("(gidx0<<14)+", "")
  stores = re.findall(
    r"\*\(data0_5570560\+(?:\(alu1169(?:\+(\d+))?\)|alu1169)\) = (buf\d+);", tail)
  if len(stores) != 64: raise RuntimeError(f"expected 64 final accumulator stores, found {len(stores)}")

  prologue = f'''  int fp_start=(gidx0*{TILES*K256})/{OWNERS};
  int fp_stop=((gidx0+1)*{TILES*K256})/{OWNERS};
  int fp_tile=fp_start/{K256},fp_boundary=(fp_tile+1)*{K256};
  bool fp_final=fp_stop>=fp_boundary;
'''
  reduction = [f"  int fp_local = {local_expr};", "  if (fp_final) {",
    "    if (lidx0==0) {",
    "      int fp_need=(fp_map[fp_tile*3]>=0)+(fp_map[fp_tile*3+1]>=0)+(fp_map[fp_tile*3+2]>=0)-1;",
    "      while (atomicAdd(fp_ready+fp_tile,0)<fp_need) {}",
    "    }", "    __syncthreads();", "  }"]
  for off_text, value in stores:
    off = int(off_text or 0)
    reduction.extend((
      "  if (fp_final) {",
      f"    int fp_z=fp_local+{off}; float fp_v=0.0f;",
      "    #pragma unroll",
      "    for (int fp_j=0;fp_j<3;fp_j++) {",
      "      int fp_slot=fp_map[fp_tile*3+fp_j];",
      f"      if (fp_slot>=0) fp_v+=(fp_slot==gidx0)?{value}:data0_5570560[fp_slot*{TILE_ELEMS}+fp_z];",
      "    }",
      f"    int fp_wr=fp_z/{COLS},fp_mc=fp_z%{COLS},fp_mt=fp_tile%{TILES_M},fp_nt=fp_tile/{TILES_M};",
      f"    fp_out[(fp_mt*{COLS}+fp_mc)*{N}+fp_nt*{ROWS}+fp_wr]=fp_v;",
      "  } else {",
      f"    data0_5570560[gidx0*{TILE_ELEMS}+fp_local+{off}]={value};",
      "  }"))
  reduction.extend((
    "  if (!fp_final) {",
    "    __syncthreads(); __threadfence(); __syncthreads();",
    "    if (lidx0==0) atomicAdd(fp_ready+fp_tile,1);",
    "    __syncthreads();",
    "  }", "}"))
  body0 = prologue + body0[:tail_start] + "\n".join(reduction) + "\n"

  dev0 = (f"__device__ __forceinline__ void q6_final_segment_0({params}, float* fp_out, "
          f"int* fp_ready, const int* fp_map, int gidx0) {{\n" + body0)
  dev1 = f"__device__ __forceinline__ void q6_final_segment_1({params}, int gidx0) {{\n" + body1
  name = "nv_q6_oracle_final_participant_streamk_170"
  wrapper = preamble + dev1 + dev0 + f'''
extern "C" __global__ void __launch_bounds__(256) {name}({params}, float* fp_out, int* fp_ready, const int* fp_map) {{
  int owner=blockIdx.x;
  int fp_start=(owner*{TILES*K256})/{OWNERS},fp_stop=((owner+1)*{TILES*K256})/{OWNERS};
  int fp_tile=fp_start/{K256},fp_boundary=(fp_tile+1)*{K256};
  bool fp_second=fp_stop>fp_boundary;
  q6_final_segment_1(data0_5570560,data1_20643840,data2_1769472,owner);
  __syncthreads(); __threadfence(); __syncthreads();
  if (fp_second && threadIdx.x==0) atomicAdd(fp_ready+fp_tile+1,1);
  __syncthreads();
  q6_final_segment_0(data0_5570560,data1_20643840,data2_1769472,fp_out,fp_ready,fp_map,owner);
}}
'''
  path = artifacts / f"{name}.cu"; path.write_text(wrapper)
  return name, wrapper


def _compile(source: str, name: str, artifacts: pathlib.Path) -> tuple[bytes, dict[str, object]]:
  binary = Device["NV"].compiler.compile(source)
  cubin = artifacts / f"{name}.cubin"; cubin.write_bytes(binary)
  census = analyze_cubin(cubin, artifacts / f"sass_{name}", name)["summary"]
  return binary, {"source": str(artifacts / f"{name}.cu"), "source_bytes": len(source),
                  "cubin": str(cubin), "cubin_sha256": hashlib.sha256(binary).hexdigest(), "sass": census}


def main() -> int:
  ap=argparse.ArgumentParser()
  ap.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--rounds",type=int,default=31)
  ap.add_argument("--out",type=pathlib.Path,required=True)
  ap.add_argument("--artifacts",type=pathlib.Path,required=True)
  args=ap.parse_args()
  if args.rounds<31: raise ValueError("qualification requires R31")
  args.artifacts.mkdir(parents=True,exist_ok=True)

  model=pathlib.Path(args.model); meta=read_metadata(model)
  info=next(x for x in meta.infos if x.name=="blk.0.ffn_down.weight")
  if info.typ!=GGML_Q6_K: raise RuntimeError(f"illegal fixture {info}")
  halfs=packed_u16_slice(model,meta,info,device="NV").contiguous().realize()
  _,q,scales=wide_record(M,K); records=[]
  for mt in range(TILES_M):
    for epoch in range(K256):
      records.append(broad_record(np.ascontiguousarray(q[mt*COLS:(mt+1)*COLS,epoch*256:(epoch+1)*256].T),
                                  np.ascontiguousarray(scales[mt*COLS:(mt+1)*COLS,epoch*8:(epoch+1)*8].T)))
  q8=Tensor(np.concatenate(records,axis=0).reshape(-1),device="NV").contiguous().realize()

  _,b0,c0=_render(0,args.artifacts); _,b1,c1=_render(1,args.artifacts)
  source0=pathlib.Path(c0["source"]).read_text(); source1=pathlib.Path(c1["source"]).read_text()
  control_name,control_binary,control_compiler=_combine_sources(source0,source1,args.artifacts)
  candidate_name,candidate_source=_final_participant_source(source0,source1,args.artifacts)
  candidate_binary,candidate_compiler=_compile(candidate_source,candidate_name,args.artifacts)

  slots,ownership=_ownership(); max_segments=max(map(len,slots))
  if max_segments!=3: raise RuntimeError(f"unexpected contributor bound {max_segments}")
  slot_map=np.full((TILES,max_segments),-1,np.int32)
  for tile,tile_slots in enumerate(slots): slot_map[tile,:len(tile_slots)]=tile_slots
  slot_map_t=Tensor(slot_map.reshape(-1),device="NV").contiguous().realize()

  fix_source=_fixup_source(max_segments); fix_path=args.artifacts/"control_fixup.cu"; fix_path.write_text(fix_source)
  fix_binary,fix_compiler=_compile(fix_source,"q6_oracle_fixup",args.artifacts)
  reset_source=f'''extern "C" __global__ void q6_final_ready_reset(int *ready) {{
    for(int i=threadIdx.x;i<{TILES};i+=blockDim.x) ready[i]=0;
  }}'''
  reset_path=args.artifacts/"q6_final_ready_reset.cu"; reset_path.write_text(reset_source)
  reset_binary,reset_compiler=_compile(reset_source,"q6_final_ready_reset",args.artifacts)

  control=NVProgram(Device["NV"],control_name,control_binary,shared_mem=LAUNCH_SHARED_BYTES)
  candidate=NVProgram(Device["NV"],candidate_name,candidate_binary,shared_mem=LAUNCH_SHARED_BYTES)
  fix=NVProgram(Device["NV"],"q6_oracle_fixup",fix_binary)
  reset=NVProgram(Device["NV"],"q6_final_ready_reset",reset_binary)
  control_partials=Tensor.full((2*OWNERS*TILE_ELEMS,),float("nan"),device="NV").contiguous().realize()
  candidate_partials=Tensor.full((2*OWNERS*TILE_ELEMS,),float("nan"),device="NV").contiguous().realize()
  control_output=Tensor.full((M,N),float("nan"),device="NV").contiguous().realize()
  candidate_output=Tensor.full((M,N),float("nan"),device="NV").contiguous().realize()
  ready=Tensor.zeros((TILES,),dtype=dtypes.int32,device="NV").contiguous().realize()

  control(_buf(control_partials),_buf(halfs),_buf(q8),global_size=(OWNERS,1,1),local_size=(256,1,1),wait=True,timeout=120000)
  fix(_buf(control_output),_buf(control_partials),_buf(slot_map_t),global_size=(TILES,1,1),local_size=(256,1,1),wait=True)
  reset(_buf(ready),global_size=(1,1,1),local_size=(128,1,1),wait=True)
  candidate(_buf(candidate_partials),_buf(halfs),_buf(q8),_buf(candidate_output),_buf(ready),_buf(slot_map_t),
    global_size=(OWNERS,1,1),local_size=(256,1,1),wait=True,timeout=120000)
  control_np,candidate_np=control_output.numpy(),candidate_output.numpy(); diff=np.abs(candidate_np-control_np)

  control_main=[]; control_fix=[]; candidate_reset=[]; candidate_main=[]
  for round_idx in range(args.rounds):
    arms=("control","candidate") if round_idx%2==0 else ("candidate","control")
    for arm in arms:
      if arm=="control":
        control_main.append(control(_buf(control_partials),_buf(halfs),_buf(q8),global_size=(OWNERS,1,1),
          local_size=(256,1,1),wait=True,timeout=120000)*1e6)
        control_fix.append(fix(_buf(control_output),_buf(control_partials),_buf(slot_map_t),global_size=(TILES,1,1),
          local_size=(256,1,1),wait=True)*1e6)
      else:
        candidate_reset.append(reset(_buf(ready),global_size=(1,1,1),local_size=(128,1,1),wait=True)*1e6)
        candidate_main.append(candidate(_buf(candidate_partials),_buf(halfs),_buf(q8),_buf(candidate_output),_buf(ready),
          _buf(slot_map_t),global_size=(OWNERS,1,1),local_size=(256,1,1),wait=True,timeout=120000)*1e6)

  control_total=[a+b for a,b in zip(control_main,control_fix)]
  candidate_total=[a+b for a,b in zip(candidate_reset,candidate_main)]
  recoveries=[a-b for a,b in zip(control_total,candidate_total)]
  authoritative=json.loads(AUTHORITATIVE_RESULT.read_text())
  exact=bool(np.array_equal(candidate_np,control_np))
  causal_pass=bool(exact and statistics.median(candidate_total)<statistics.median(control_total))
  reference_pass=bool(authoritative["correctness"]["reference_allclose_rtol2e5_atol2e3"])
  final_segments=TILES; published_segments=len(ownership)-final_segments
  tile_bytes=TILE_ELEMS*4
  result={"schema":"tinygrad.nv_q6_final_participant_fixup.v1","shape":{"M":M,"N":N,"K":K},
    "lock":"all GPU correctness/timing executed under flock /tmp/nv-q6-oracle-gpu.lock",
    "ownership":{"owners":OWNERS,"segments":len(ownership),"final_participants":final_segments,
      "published_partial_segments":published_segments,"segment_census":{str(n):sum(len(x)==n for x in slots) for n in range(1,4)}},
    "correctness":{"finite":bool(np.isfinite(candidate_np).all()),"candidate_control_bit_exact":exact,
      "candidate_control_max_abs":float(diff.max()),"authoritative_reference_pass":reference_pass,
      "authoritative_reference_max_abs":authoritative["correctness"]["reference_max_abs"]},
    "traffic_bytes":{"control_partial_write":len(ownership)*tile_bytes,"control_partial_read":len(ownership)*tile_bytes,
      "candidate_partial_write":published_segments*tile_bytes,"candidate_partial_read":published_segments*tile_bytes,
      "partial_traffic_saved":2*final_segments*tile_bytes,"output_write":TILES*tile_bytes,"ready_atomic_increments":published_segments},
    "timing":{"control_main":_stats(control_main),"control_fixup":_stats(control_fix),"control_total":_stats(control_total),
      "candidate_reset":_stats(candidate_reset),"candidate_embedded":_stats(candidate_main),"candidate_total":_stats(candidate_total),
      "paired_recovery":_stats(recoveries),"candidate_wins":sum(b<a for a,b in zip(control_total,candidate_total)),
      "alternated_call_order":True,"rounds":args.rounds},
    "baselines":{"historical_control_us":CONTROL_HISTORICAL_US,"llama_pair_us":LLAMA_MAIN_US+LLAMA_FIXUP_US,
      "llama_pair_5pct_us":(LLAMA_MAIN_US+LLAMA_FIXUP_US)*1.05},
    "comparison":{"candidate_vs_same_process_control_ratio":statistics.median(candidate_total)/statistics.median(control_total),
      "candidate_vs_historical_control_ratio":statistics.median(candidate_total)/CONTROL_HISTORICAL_US,
      "candidate_vs_llama_ratio":statistics.median(candidate_total)/(LLAMA_MAIN_US+LLAMA_FIXUP_US)},
    "compiler":{"control":control_compiler,"candidate":candidate_compiler,"fixup":fix_compiler,"reset":reset_compiler,
      "segment0":c0,"segment1":c1},
    "causal_pass":causal_pass,"promoted":bool(causal_pass and reference_pass)}
  args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2)+"\n")
  print(json.dumps(result,sort_keys=True))
  return 0 if causal_pass else 1


if __name__=="__main__": raise SystemExit(main())
