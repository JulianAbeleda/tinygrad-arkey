#!/usr/bin/env python3
"""R31 combined-initial-publication gate on the admitted one-body packed Q6 route."""
from __future__ import annotations
import argparse, collections, hashlib, json, pathlib, re, statistics
import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.uop.ops import Ops, UOp
from extra.llm_research.layout import GGML_Q6_K,packed_u16_slice,read_metadata
from extra.llm_research.prefill.bench_nv_q6_oracle_broad_cta import _record as broad_record
from extra.llm_research.prefill.bench_nv_q6_oracle_full_streamk import _fixup_source
from extra.llm_research.prefill.bench_nv_q6_oracle_streamk_single_body_packed import (
  M,N,K,OWNERS,K256,TILES_M,TILES_N,TILES,TILE_ELEMS,LAUNCH_SHARED_BYTES,
  _ast_proof,_buf,_compile_ast,_paired,_schedule,_windows)
from extra.llm_research.prefill.nv_compiler_q6k_imma_gate import _record as wide_record,_run
from extra.llm_research.prefill.nv_q6_oracle_broad_cta import ROWS,COLS,q6_oracle_broad_cta_kernel
from extra.llm_research.prefill.nv_q6_sass_census import analyze_cubin

EXPECTED_CUBIN_SHA256={
  "early_separate":"1df61553f7ebb9904108c2ed14b0c256abdce067a2ae3a1bfe45fcc86a243e1f",
  "early_combined":"6eb663b3a3fd628e3394a0ce8f8780e108e47f40b887b0a75a0756dcf33e9137",
}
ARM_SPECS={
  "early_separate":{"combined_initial_publish":False},
  "early_combined":{"combined_initial_publish":True},
}
ARMS=tuple(ARM_SPECS)
SASS_INSN_RE=re.compile(r"^\s*/\*([0-9a-fA-F]+)\*/\s+(?:@!?[A-Z0-9]+\s+)?([A-Z][A-Z0-9_.]*)\b")
SHARED_ADDR_RE=re.compile(r"\[R\d+(?:\.[A-Za-z0-9_]+)?(?:\+0x([0-9a-fA-F]+))?\]")
Q8_SHARED_OFFSETS=tuple(0x9800+i*0x400 for i in range(18))

def _ast(combined_initial_publish:bool):
  ph=lambda n,dt,i:UOp.placeholder((n,),dt,i)
  return q6_oracle_broad_cta_kernel(ph(2*OWNERS*TILE_ELEMS,dtypes.float32,0),ph(N*K256*105,dtypes.uint16,1),
    ph(TILES_M*K256*2*COLS*36,dtypes.uint32,2),prefetch_second_panel=True,
    combined_initial_publish=combined_initial_publish,
    factor_dA=False,oracle_publisher=True,weight_scale_contract="trusted_fp16_packed",
    streamk_owners=OWNERS,streamk_segment=0,streamk_segments_in_cta=True)

def _sass_instructions(disassembly:str):
  out=[]
  for line in disassembly.splitlines():
    if (match:=SASS_INSN_RE.match(line)) is None: continue
    addr=SHARED_ADDR_RE.search(line)
    out.append({"ordinal":len(out),"pc":int(match.group(1),16),"opcode":match.group(2),"line":line,
      "shared_offset":int(addr.group(1),16) if addr is not None and addr.group(1) is not None else (0 if addr else None)})
  return out

def _classify_q8_panel1(disassembly:str):
  insns=_sass_instructions(disassembly)
  q8_loads=[x for x in insns if x["opcode"].startswith("LDG.E") and ".U16" not in x["opcode"]]
  q8_stores=[x for x in insns if x["opcode"].split(".",1)[0] == "STS" and x["shared_offset"] in Q8_SHARED_OFFSETS]
  load_count_ok=len(q8_loads)==36
  store_offsets=collections.Counter(x["shared_offset"] for x in q8_stores)
  store_count_ok=len(q8_stores)==36 and store_offsets==collections.Counter({x:2 for x in Q8_SHARED_OFFSETS})
  panel1_loads=q8_loads[-18:] if load_count_ok else []
  panel1_stores=q8_stores[-18:] if store_count_ok else []
  panel1_offsets_ok=bool(panel1_stores and collections.Counter(x["shared_offset"] for x in panel1_stores)==
    collections.Counter(Q8_SHARED_OFFSETS))
  span=(panel1_stores[0]["ordinal"]-panel1_loads[0]["ordinal"] if panel1_loads and panel1_stores else None)
  return {"q8_loads":len(q8_loads),"q8_stores":len(q8_stores),"panel1_loads":len(panel1_loads),
    "panel1_stores":len(panel1_stores),"panel1_offsets_exact":panel1_offsets_ok,
    "panel1_first_load_ordinal":panel1_loads[0]["ordinal"] if panel1_loads else None,
    "panel1_first_store_ordinal":panel1_stores[0]["ordinal"] if panel1_stores else None,
    "panel1_first_load_pc":f"0x{panel1_loads[0]['pc']:x}" if panel1_loads else None,
    "panel1_first_store_pc":f"0x{panel1_stores[0]['pc']:x}" if panel1_stores else None,
    "panel1_load_to_store_span_instructions":span,
    "classified":bool(load_count_ok and store_count_ok and panel1_offsets_ok)}

def _signature(artifact):
  summary=artifact["sass"];resources=summary.get("resources") or {};families=summary.get("families",{})
  placement=_classify_q8_panel1(pathlib.Path(artifact["sass_artifacts"]["disassembly"]).read_text())
  return {"instruction_total":summary.get("instruction_total"),"registers":resources.get("registers"),
    "stack_bytes":resources.get("stack_bytes"),"shared_static_bytes":resources.get("shared_static_bytes"),
    "local_static_bytes":resources.get("local_static_bytes"),
    **{key:families.get(key,0) for key in ("LDL","STL","LDG","STS","IMMA","LDSM","BAR","FMUL","FADD","FFMA")},
    "q8_panel1":placement}

def _cpu_fixup(raw,slots):
  raw=raw.reshape(2*OWNERS,ROWS,COLS);out=np.empty((M,N),np.float32)
  for tile,tile_slots in enumerate(slots):
    reduced=raw[tile_slots[0]].copy()
    for slot in tile_slots[1:]: reduced+=raw[slot]
    mt,nt=tile%TILES_M,tile//TILES_M;out[mt*COLS:(mt+1)*COLS,nt*ROWS:(nt+1)*ROWS]=reduced.T
  return out

def _resource_no_regression(candidate,anchor):
  bounded=("registers","stack_bytes","local_static_bytes","LDL","STL")
  return bool(all(candidate[x] is not None and anchor[x] is not None and candidate[x]<=anchor[x] for x in bounded) and
    candidate["shared_static_bytes"]==anchor["shared_static_bytes"])

def _performance(candidate_samples,anchor_samples):
  paired={kind:_paired(candidate_samples[kind],anchor_samples[kind]) for kind in ("main","fixup","pair")}
  passed=bool(-paired["pair"]["r31"]["median_us"]>=3.0 and paired["pair"]["candidate_wins"]>=24)
  return {"paired":paired,"minimum_pair_median_improvement_us":3.0,"minimum_pair_wins":24,"passed":passed}

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--rounds",type=int,default=31);ap.add_argument("--out",type=pathlib.Path,required=True)
  ap.add_argument("--artifacts",type=pathlib.Path,required=True);a=ap.parse_args()
  if a.rounds != 31: raise ValueError("publication qualification requires exactly R31")
  a.artifacts.mkdir(parents=True,exist_ok=True);model=pathlib.Path(a.model);meta=read_metadata(model)
  info=next(x for x in meta.infos if x.name=="blk.0.ffn_down.weight")
  if info.typ != GGML_Q6_K: raise RuntimeError(info)
  halfs=packed_u16_slice(model,meta,info,device="NV").contiguous().realize()
  wide_host,q,scales=wide_record(M,K);wide_q8=Tensor(wide_host,device="NV").contiguous().realize()
  broad=[]
  for mt in range(TILES_M):
    for epoch in range(K256):
      broad.append(broad_record(np.ascontiguousarray(q[mt*COLS:(mt+1)*COLS,epoch*256:(epoch+1)*256].T),
        np.ascontiguousarray(scales[mt*COLS:(mt+1)*COLS,epoch*8:(epoch+1)*8].T)))
  q8=Tensor(np.concatenate(broad).reshape(-1),device="NV").contiguous().realize()

  wide_artifacts=a.artifacts/"wide";wide_artifacts.mkdir(parents=True,exist_ok=True)
  direct=_run("wide_direct",M,N,K,halfs,wide_q8,a.rounds,wide_artifacts,(128,128,2,4,256))
  direct_source=(wide_artifacts/"wide_direct.cu").read_text();direct_binary=Device["NV"].compiler.compile(direct_source)
  match=re.search(r'__global__ void __launch_bounds__\(256\) (\w+)\(',direct_source)
  if match is None: raise RuntimeError("trusted direct symbol missing")
  reference=Tensor.full((M,N),float("nan"),device="NV").contiguous().realize()
  NVProgram(Device["NV"],match.group(1),direct_binary)(_buf(reference),_buf(wide_q8),_buf(halfs),
    global_size=(32,4,1),local_size=(32,2,4),wait=True)
  expected=reference.numpy()

  asts={arm:_ast(**spec) for arm,spec in ARM_SPECS.items()};states={}
  for arm in ARMS:
    name,binary,artifact,source=_compile_ast(asts[arm],arm,a.artifacts)
    states[arm]={"program":NVProgram(Device["NV"],name,binary,shared_mem=LAUNCH_SHARED_BYTES),
      "compiler":artifact,"source":source,"signature":_signature(artifact),"structure":_ast_proof(asts[arm])}
  frozen={arm:states[arm]["compiler"]["cubin_sha256"]==EXPECTED_CUBIN_SHA256[arm] for arm in ARMS}
  anchor_frozen=all(frozen.values())

  slots,ownership,ownership_invariants=_schedule();max_segments=max(map(len,slots));slot_map=np.full((TILES,max_segments),-1,np.int32)
  for tile,tile_slots in enumerate(slots): slot_map[tile,:len(tile_slots)]=tile_slots
  slot_map_t=Tensor(slot_map.reshape(-1),device="NV").contiguous().realize()
  fix_source=_fixup_source(max_segments);fix_path=a.artifacts/"plane_major_fixup.cu";fix_path.write_text(fix_source)
  fix_binary=Device["NV"].compiler.compile(fix_source);fix_cubin=a.artifacts/"plane_major_fixup.cubin";fix_cubin.write_bytes(fix_binary)
  fix_census=analyze_cubin(fix_cubin,a.artifacts/"sass_fixup","q6_oracle_fixup")
  fix_artifact={"source":str(fix_path),"cubin":str(fix_cubin),"cubin_sha256":hashlib.sha256(fix_binary).hexdigest(),
    "sass":fix_census["summary"],"sass_artifacts":{k:fix_census[k] for k in ("sass_json","disassembly","resources")}}
  fix=NVProgram(Device["NV"],"q6_oracle_fixup",fix_binary)

  correctness={};outputs={};partials={}
  for arm,state in states.items():
    state["partials"]=Tensor.full((2*OWNERS*TILE_ELEMS),float("nan"),device="NV").contiguous().realize()
    state["output"]=Tensor.full((M,N),float("nan"),device="NV").contiguous().realize()
    state["program"](_buf(state["partials"]),_buf(halfs),_buf(q8),global_size=(OWNERS,1,1),
      local_size=(256,1,1),wait=True,timeout=120000)
    fix(_buf(state["output"]),_buf(state["partials"]),_buf(slot_map_t),global_size=(TILES,1,1),local_size=(256,1,1),wait=True)
    got=state["output"].numpy();raw=state["partials"].numpy();cpu=_cpu_fixup(raw,slots);diff=np.abs(got-expected)
    close=np.isclose(got,expected,rtol=2e-5,atol=2e-3);outputs[arm]=got;partials[arm]=raw
    correctness[arm]={"finite":bool(np.isfinite(got).all()),
      "gpu_fixup_cpu_bit_exact":bool(np.array_equal(got.view(np.uint32),cpu.view(np.uint32))),
      "reference_max_abs":float(diff.max()),"reference_mean_abs":float(diff.mean()),
      "reference_failing_count":int(np.count_nonzero(~close)),"reference_passed":bool(close.all())}
  anchor=ARMS[0]
  exactness={arm:{"partials_bit_exact":bool(np.array_equal(partials[arm].view(np.uint32),partials[anchor].view(np.uint32))),
    "final_bit_exact":bool(np.array_equal(outputs[arm].view(np.uint32),outputs[anchor].view(np.uint32)))} for arm in ARMS}

  samples={arm:{"main":[],"fixup":[],"pair":[]} for arm in ARMS}
  for round_idx in range(a.rounds):
    order=ARMS if round_idx%2==0 else tuple(reversed(ARMS))
    for arm in order:
      state=states[arm]
      main_us=state["program"](_buf(state["partials"]),_buf(halfs),_buf(q8),global_size=(OWNERS,1,1),
        local_size=(256,1,1),wait=True,timeout=120000)*1e6
      fix_us=fix(_buf(state["output"]),_buf(state["partials"]),_buf(slot_map_t),global_size=(TILES,1,1),
        local_size=(256,1,1),wait=True)*1e6
      samples[arm]["main"].append(main_us);samples[arm]["fixup"].append(fix_us);samples[arm]["pair"].append(main_us+fix_us)
  timing={arm:{kind:_windows(values) for kind,values in sample.items()} for arm,sample in samples.items()}
  comparisons={
    "early_combined_vs_early_separate":_performance(samples["early_combined"],samples["early_separate"]),
  }

  signatures={arm:state["signature"] for arm,state in states.items()};anchor_sig=signatures[anchor]
  structures={}
  for arm,state in states.items():
    expected_bar=4 if ARM_SPECS[arm]["combined_initial_publish"] else 5;sig=signatures[arm];proof=state["structure"]
    structures[arm]={**proof,"one_global_kernel":len(re.findall(r'extern "C" __global__',state["source"]))==1,
      "no_spliced_helpers":"q6_segment_" not in state["source"],"expected_ast_barriers":proof["barrier_nodes"]==expected_bar,
      "expected_sass_barriers":sig["BAR"]==expected_bar,"panel1_18_loads_18_stores":bool(
        sig["q8_panel1"]["classified"] and sig["q8_panel1"]["panel1_loads"]==18 and sig["q8_panel1"]["panel1_stores"]==18)}
  traffic_keys=("LDG","STS","IMMA","LDSM","FMUL","FADD","FFMA")
  binary_exact={arm:bool(all(signatures[arm][x]==anchor_sig[x] for x in traffic_keys) and
    signatures[arm]["LDG"]==109 and signatures[arm]["STS"]==73 and signatures[arm]["IMMA"]==256 and
    signatures[arm]["LDSM"]==32 and all(structures[arm].values())) for arm in ARMS}
  arm_correct={arm:bool(correctness[arm]["finite"] and correctness[arm]["gpu_fixup_cpu_bit_exact"] and
    correctness[arm]["reference_passed"] and exactness[arm]["partials_bit_exact"] and exactness[arm]["final_bit_exact"])
    for arm in ARMS}
  resources={
    "early_combined_vs_early_separate":_resource_no_regression(signatures["early_combined"],signatures["early_separate"]),
  }
  combined_admitted=bool(anchor_frozen and arm_correct["early_combined"] and binary_exact["early_combined"] and
    resources["early_combined_vs_early_separate"] and comparisons["early_combined_vs_early_separate"]["passed"])
  final_arm="early_combined" if combined_admitted else "early_separate"
  # A candidate missing its span/resource gate is a valid negative experiment,
  # not a harness failure. Measurement integrity covers the frozen anchor,
  # numerics, topology, classified traffic, and invariant arithmetic census;
  # binary_exact/resource/performance remain promotion predicates above.
  experiment_valid=bool(anchor_frozen and all(ownership_invariants.values()) and all(arm_correct.values()) and
    all(all(proof.values()) for proof in structures.values()) and
    all(signatures[arm]["LDG"]==109 and signatures[arm]["STS"]==73 and signatures[arm]["IMMA"]==256 and
      signatures[arm]["LDSM"]==32 and all(signatures[arm][x]==anchor_sig[x] for x in traffic_keys) for arm in ARMS))
  verdict="PROMOTE_COMBINED_ONLY" if combined_admitted else "KEEP_EARLY_SEPARATE"
  result={"schema":"tinygrad.nv_q6_oracle_combined_publication.v1","shape":{"M":M,"N":N,"K":K},
    "launch":{"grid":[OWNERS,1,1],"block":[256,1,1],"shared_bytes":LAUNCH_SHARED_BYTES},
    "arms":ARM_SPECS,"frozen_cubins":{arm:{"expected":EXPECTED_CUBIN_SHA256[arm],
      "actual":states[arm]["compiler"]["cubin_sha256"],"frozen":frozen[arm]} for arm in ARMS},
    "ownership":{"records":ownership,"invariants":ownership_invariants,"plane_major_slot_map":slot_map.tolist()},
    "reference":{"kind":"compiler_wide_direct","result":direct},"correctness":correctness,"exactness":exactness,
    "structure":structures,"signatures":signatures,"binary_exact":binary_exact,"resource_no_regression":resources,
    "timing":timing,"comparisons":comparisons,"selection":{"combined_base":"early_separate",
      "combined_candidate":"early_combined","combined_admitted":combined_admitted,"final_arm":final_arm},
    "compiler":{"arms":{arm:states[arm]["compiler"] for arm in ARMS},"fixup":fix_artifact},
    "gpu_lock":{"mode":"outer_flock_required","path":"/tmp/nv-q6-oracle-gpu.lock"},
    "experiment_valid":experiment_valid,"promotion_eligible":combined_admitted,
    "verdict":verdict,"passed":experiment_valid}
  a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,sort_keys=True))
  return 0 if experiment_valid else 1

if __name__=="__main__": raise SystemExit(main())
