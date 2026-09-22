import fcntl,hashlib,json,pathlib,statistics
import numpy as np
from tinygrad import Device,Tensor,dtypes
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.uop.ops import Ops,UOp
from extra.llm_research.layout import GGML_Q6_K,packed_u16_slice,read_metadata
from extra.llm_research.prefill.bench_nv_q6_oracle_broad_cta import _record as broad_record
from extra.llm_research.prefill.bench_nv_q6_oracle_reduction_policy import LAUNCH_SHARED_BYTES,_buf,_compile_ast
from extra.llm_research.prefill.bench_nv_q6_oracle_full_streamk import _combine_sources
from extra.llm_research.prefill.nv_compiler_q6k_imma_gate import _record as wide_record
from extra.llm_research.prefill.nv_q6_destination_partial import SYMBOL,destination_major_fixup_source
from extra.llm_research.prefill.nv_q6_oracle_reduction_policy import M,N,K,K256,OWNERS,TILES,TILES_M,COLS,TILE_ELEMS,build_packed_one_body_ast,build_reduction_schedule
from extra.llm_research.prefill.nv_q6_sass_census import analyze_cubin
from extra.llm_research.prefill.nv_llama_packed_q6k_down_pp512_binding import SCRATCH_FLOATS,binding_for

ROOT=pathlib.Path("docs/task_workflow/evidence/nv-q6-llama-arithmetic-contract-gate19-20260902")
ART=ROOT/"artifacts"
SPECS={
  "trusted_implicit":{"fp32_contraction":"implicit"},
  "trusted_both":{"fp32_contraction":"trusted_both"},
  "llama_direct":{"fp32_contraction":"implicit","weight_scale_contract":"legacy"},
  "llama_direct_round":{"fp32_contraction":"implicit","weight_scale_contract":"legacy","q6_fragment_schedule":"round"},
  "llama_factor_seq":{"fp32_contraction":"implicit","weight_scale_contract":"legacy","factor_dA":True},
  "llama_factor_seq_round":{"fp32_contraction":"implicit","weight_scale_contract":"legacy","factor_dA":True,"q6_fragment_schedule":"round"},
  "llama_factor_ssa":{"fp32_contraction":"both","weight_scale_contract":"legacy","factor_dA":True,"fp32_p_tree":"ssa_vector"},
  "llama_factor_ssa_round":{"fp32_contraction":"both","weight_scale_contract":"legacy","factor_dA":True,"fp32_p_tree":"ssa_vector","q6_fragment_schedule":"round"},
  "llama_factor_ssa_tile2":{"fp32_contraction":"both","weight_scale_contract":"legacy","factor_dA":True,"fp32_p_tree":"ssa_vector","q6_fragment_schedule":"tile2"},
  "llama_factor_ssa_tile4":{"fp32_contraction":"both","weight_scale_contract":"legacy","factor_dA":True,"fp32_p_tree":"ssa_vector","q6_fragment_schedule":"tile4"},
  "llama_factor_ssa_tile8":{"fp32_contraction":"both","weight_scale_contract":"legacy","factor_dA":True,"fp32_p_tree":"ssa_vector","q6_fragment_schedule":"tile8"},
  "llama_factor_ssa_tile4_meta":{"fp32_contraction":"both","weight_scale_contract":"legacy","factor_dA":True,"fp32_p_tree":"ssa_vector","q6_fragment_schedule":"tile4","q6_metadata_schedule":"phase"},
  "llama_factor_ssa_tile8_meta":{"fp32_contraction":"both","weight_scale_contract":"legacy","factor_dA":True,"fp32_p_tree":"ssa_vector","q6_fragment_schedule":"tile8","q6_metadata_schedule":"phase"},
  "llama_factor_ssa_tile8_late_d":{"fp32_contraction":"both","weight_scale_contract":"legacy","factor_dA":True,"fp32_p_tree":"ssa_vector","q6_fragment_schedule":"tile8","q6_metadata_schedule":"late_d"},
  "llama_factor_ssa_tile8_pair":{"fp32_contraction":"both","weight_scale_contract":"legacy","factor_dA":True,"fp32_p_tree":"ssa_vector","q6_fragment_schedule":"tile8","q6_metadata_schedule":"pair"},
  "llama_factor_register_bank":{"fp32_contraction":"both","weight_scale_contract":"legacy","factor_dA":True,"fp32_p_tree":"register_bank","q6_fragment_schedule":"tile8","q6_metadata_schedule":"phase"},
  "llama_factor_ssa_late_p1_c4":{"fp32_contraction":"both","weight_scale_contract":"legacy","factor_dA":True,"fp32_p_tree":"ssa_vector","q6_fragment_schedule":"tile8","q6_metadata_schedule":"phase","region_load_bridge_q8_panel1":False,"strict_after_q8_panel1":True,"q8_panel1_anchor_cg":4},
  "llama_factor_ssa_late_p1_c5":{"fp32_contraction":"both","weight_scale_contract":"legacy","factor_dA":True,"fp32_p_tree":"ssa_vector","q6_fragment_schedule":"tile8","q6_metadata_schedule":"phase","region_load_bridge_q8_panel1":False,"strict_after_q8_panel1":True,"q8_panel1_anchor_cg":5},
  "llama_factor_ssa_late_p1_c6":{"fp32_contraction":"both","weight_scale_contract":"legacy","factor_dA":True,"fp32_p_tree":"ssa_vector","q6_fragment_schedule":"tile8","q6_metadata_schedule":"phase","region_load_bridge_q8_panel1":False,"strict_after_q8_panel1":True,"q8_panel1_anchor_cg":6},
  "llama_factor_ssa_fp32d":{"fp32_contraction":"both","weight_scale_contract":"legacy","factor_dA":True,"fp32_p_tree":"ssa_vector","q6_fragment_schedule":"tile8","q6_metadata_schedule":"phase","q6_d_storage":"fp32"},
  "llama_factor_ssa_fp32d_late_p1":{"fp32_contraction":"both","weight_scale_contract":"legacy","factor_dA":True,"fp32_p_tree":"ssa_vector","q6_fragment_schedule":"tile8","q6_metadata_schedule":"phase","q6_d_storage":"fp32","region_load_bridge_q8_panel1":False,"strict_after_q8_panel1":True,"q8_panel1_anchor_cg":5},
  "llama_factor_ssa_duplicated":{"fp32_contraction":"both","weight_scale_contract":"legacy","factor_dA":True,"fp32_p_tree":"ssa_vector","q6_fragment_schedule":"tile8","q6_metadata_schedule":"phase","duplicated":True},
  "llama_factor_ssa_prefetch":{"fp32_contraction":"both","weight_scale_contract":"legacy","factor_dA":True,"fp32_p_tree":"ssa_vector","q6_fragment_schedule":"tile8","q6_metadata_schedule":"phase","region_load_bridge_q8_panel1":False},
  "llama_factor":{"fp32_contraction":"both","weight_scale_contract":"legacy","factor_dA":True,"fp32_p_tree":"left"},
}
ARMS=tuple(x for x in __import__("os").environ.get("Q6_ARMS",",".join(SPECS)).split(",") if x)
PROMOTION_ARM="llama_factor_ssa_tile8_meta"
if any(x not in SPECS for x in ARMS): raise ValueError(f"unknown Q6 arm(s): {ARMS!r}")
ANCHOR=__import__("os").environ.get("Q6_ANCHOR","trusted_implicit" if "trusted_implicit" in ARMS else ARMS[0])
if ANCHOR not in ARMS: raise ValueError(f"Q6 anchor {ANCHOR!r} is not selected")
LLAMA_TOTAL_US=209.856
TARGET_US=LLAMA_TOTAL_US*1.05

def stats(xs): return {"samples_us":xs,"min_us":min(xs),"median_us":statistics.median(xs),"max_us":max(xs)}

def compile_fixup():
  path=ART/"destination_fixup";path.mkdir(parents=True,exist_ok=True)
  src=destination_major_fixup_source();(path/"destination_fixup.cu").write_text(src)
  binary=Device["NV"].compiler.compile(src);cubin=path/"destination_fixup.cubin";cubin.write_bytes(binary)
  census=analyze_cubin(cubin,path/"sass",SYMBOL)["summary"]
  return binary,{"cubin_sha256":hashlib.sha256(binary).hexdigest(),"sass":census}

def main(program,partial,halfs,q8):
  return program(_buf(partial),_buf(halfs),_buf(q8),global_size=(OWNERS,1,1),local_size=(256,1,1),wait=True,timeout=120000)*1e6

def fix(program,out,partial,slots,counts):
  return program(_buf(out),_buf(partial),_buf(slots),_buf(counts),global_size=(TILES,4,1),local_size=(128,1,1),wait=True)*1e6

def native_runner(program):
  info=program.arg;binary=next(x.arg for x in program.src if x.op is Ops.BINARY)
  return (NVProgram(Device["NV"],info.name,binary,shared_mem=info.aux[0] if info.aux else 0),
          tuple(v.arg[1] for v in info.vars),info.global_size,info.local_size)

def run():
  ART.mkdir(parents=True,exist_ok=True)
  lock=open("/tmp/nv-q6-oracle-gpu.lock","w");fcntl.flock(lock,fcntl.LOCK_EX)
  compiled={}
  for arm in ARMS:
    spec=dict(SPECS[arm]);region_load_bridge=spec.pop("region_load_bridge_q8_panel1",True);duplicated=spec.pop("duplicated",False)
    if duplicated:
      parts=[]
      for segment in (0,1):
        ast=build_packed_one_body_ast(partial_output_layout="destination_major",region_load_bridge_q8_panel1=region_load_bridge,
          streamk_segments_in_cta=False,streamk_segment=segment,**spec)
        parts.append(_compile_ast(ast,f"q6_{arm}_s{segment}",ART)[2])
      source0,source1=(pathlib.Path(x["source"]).read_text() for x in parts)
      source1=source1.replace("region_bridge_ordered_address","region_bridge_ordered_address_s1").replace(
        "region_bridge_order_dependency","region_bridge_order_dependency_s1")
      program_name,binary,artifact=_combine_sources(source0,source1,ART,arm)
    else:
      ast=build_packed_one_body_ast(partial_output_layout="destination_major",region_load_bridge_q8_panel1=region_load_bridge,
                                    **spec)
      name=f"q6_{arm}";program_name,binary,artifact=_compile_ast(ast,name,ART)
    compiled[arm]={"program":NVProgram(Device["NV"],program_name,binary,shared_mem=LAUNCH_SHARED_BYTES),"artifact":artifact}
  fix_binary,fix_artifact=compile_fixup();fix_program=NVProgram(Device["NV"],SYMBOL,fix_binary)
  schedule=build_reduction_schedule();slot_map,counts,_=schedule.arrays()
  slots=Tensor(slot_map.reshape(-1),device="NV").contiguous().realize();cts=Tensor(counts,device="NV").contiguous().realize()
  model=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf");meta=read_metadata(model)
  info=next(x for x in meta.infos if x.name=="blk.0.ffn_down.weight");assert info.typ==GGML_Q6_K
  halfs=packed_u16_slice(model,meta,info,device="NV").contiguous().realize();_,q,scales=wide_record(M,K);broad=[]
  for mt in range(TILES_M):
    for epoch in range(K256):
      broad.append(broad_record(np.ascontiguousarray(q[mt*COLS:(mt+1)*COLS,epoch*256:(epoch+1)*256].T),
                                np.ascontiguousarray(scales[mt*COLS:(mt+1)*COLS,epoch*8:(epoch+1)*8].T)))
  q8=Tensor(np.concatenate(broad).reshape(-1),device="NV").contiguous().realize()
  llama_words=np.empty((K//128,M,36),dtype=np.uint32)
  llama_words[:,:,:4]=scales.reshape(M,K//128,4).transpose(1,0,2).copy().view(np.uint32)
  llama_words[:,:,4:]=q.reshape(M,K//128,128).transpose(1,0,2).copy().view(np.uint32).reshape(K//128,M,32)
  llama_record=Tensor(llama_words.reshape(-1),device="NV").contiguous().realize()
  llama_out=Tensor.empty(M*N,dtype=dtypes.float32,device="NV")
  llama_workspace=Tensor.empty(SCRATCH_FLOATS,dtype=dtypes.float32,device="NV")
  llama=binding_for()
  halfs,llama_record,llama_out,llama_workspace=halfs.uop_program(llama_record,llama_out,llama_workspace,fxn=lambda *_:llama.main)
  llama_out,llama_workspace=llama_out.uop_program(llama_workspace,fxn=lambda *_:llama.fixup)
  llama_out.realize();reference=llama_out.numpy().reshape(M,N)
  llama_main,llama_main_vals,llama_main_grid,llama_main_block=native_runner(llama.main)
  llama_fixup,llama_fixup_vals,llama_fixup_grid,llama_fixup_block=native_runner(llama.fixup)
  def run_llama_main(): return llama_main(_buf(halfs),_buf(llama_record),_buf(llama_out),_buf(llama_workspace),vals=llama_main_vals,
    global_size=llama_main_grid,local_size=llama_main_block,wait=True,timeout=120000)*1e6
  def run_llama_fixup(): return llama_fixup(_buf(llama_out),_buf(llama_workspace),vals=llama_fixup_vals,
    global_size=llama_fixup_grid,local_size=llama_fixup_block,wait=True)*1e6
  partials={arm:Tensor.full((2*OWNERS*TILE_ELEMS),float("nan"),device="NV").contiguous().realize() for arm in ARMS}
  outs={arm:Tensor.full((M,N),float("nan"),device="NV").contiguous().realize() for arm in ARMS}
  for arm in ARMS:
    main(compiled[arm]["program"],partials[arm],halfs,q8);fix(fix_program,outs[arm],partials[arm],slots,cts)
  correctness={}
  for arm in ARMS:
    got=outs[arm].numpy();close=np.isclose(got,reference,rtol=2e-5,atol=2e-3)
    correctness[arm]={"finite":bool(np.isfinite(got).all()),"bit_exact_to_llama":bool(np.array_equal(got.view(np.uint32),reference.view(np.uint32))),
      "tolerance_failures":int(close.size-np.count_nonzero(close)),"max_abs_to_llama":float(np.max(np.abs(got-reference))),
      "mean_abs_to_llama":float(np.mean(np.abs(got-reference)))}
  for warmup in range(3):
    for arm in ARMS[warmup:]+ARMS[:warmup]:
      main(compiled[arm]["program"],partials[arm],halfs,q8);fix(fix_program,outs[arm],partials[arm],slots,cts)
    run_llama_main();run_llama_fixup()
  samples={arm:{"main":[],"fixup":[],"total":[]} for arm in ARMS};orders=[]
  llama_samples={"main":[],"fixup":[],"total":[]}
  for iteration in range(31):
    shift=iteration%len(ARMS);arm_order=ARMS[shift:]+ARMS[:shift]
    order=(arm_order+("live_llama",)) if iteration%2==0 else (("live_llama",)+arm_order);orders.append(order)
    for arm in order:
      if arm == "live_llama": main_us,fix_us=run_llama_main(),run_llama_fixup();target=llama_samples
      else:
        main_us=main(compiled[arm]["program"],partials[arm],halfs,q8);fix_us=fix(fix_program,outs[arm],partials[arm],slots,cts);target=samples[arm]
      target["main"].append(main_us);target["fixup"].append(fix_us);target["total"].append(main_us+fix_us)
  paired={}
  for arm in ARMS:
    if arm == ANCHOR: continue
    paired[arm]={}
    for phase in ("main","fixup","total"):
      delta=[candidate-anchor for anchor,candidate in zip(samples[ANCHOR][phase],samples[arm][phase])]
      median=statistics.median(delta)
      paired[arm][phase]={"median_us":median,"mad_us":statistics.median(abs(x-median) for x in delta),
                          "wins":sum(x<0 for x in delta),"samples_us":delta}
  medians={arm:{phase:statistics.median(values) for phase,values in phases.items()} for arm,phases in samples.items()}
  live_medians={phase:statistics.median(values) for phase,values in llama_samples.items()}
  paired_live={}
  for arm in ARMS:
    paired_live[arm]={}
    for phase in ("main","fixup","total"):
      delta=[candidate-llama for candidate,llama in zip(samples[arm][phase],llama_samples[phase])]
      median=statistics.median(delta)
      paired_live[arm][phase]={"median_us":median,"mad_us":statistics.median(abs(x-median) for x in delta),
                               "wins":sum(x<0 for x in delta),"samples_us":delta}
  historical_eligible=[arm for arm in ARMS if correctness[arm]["finite"] and correctness[arm]["tolerance_failures"]==0 and medians[arm]["total"]<=TARGET_US]
  eligible=[arm for arm in ARMS if correctness[arm]["finite"] and correctness[arm]["tolerance_failures"]==0 and
            medians[arm]["total"]<=live_medians["total"]*1.05]
  result={"schema":"tinygrad.nv_q6_llama_arithmetic_contract_gate.v1","oracle":{"llama_main_us":201.216,"llama_fixup_us":8.640,
    "llama_total_us":LLAMA_TOTAL_US,"target_5pct_us":TARGET_US,"historical_only":True,
    "live_llama_medians_us":live_medians,"live_target_5pct_us":live_medians["total"]*1.05,
    "numerical_reference":"pinned llama Q6 main and fixup on identical canonical Q6/Q8 records"},
    "configuration":{"samples":31,"warmups":3,"arms":SPECS,"partial_output_layout":"destination_major","region_load_bridge_q8_panel1":True,
    "gpu_lock":"/tmp/nv-q6-oracle-gpu.lock","paired_anchor":ANCHOR},"correctness":correctness,"compiler":{arm:compiled[arm]["artifact"] for arm in ARMS},
    "fixup_compiler":fix_artifact,"timing":{arm:{phase:stats(values) for phase,values in phases.items()} for arm,phases in samples.items()},
    "live_llama_timing":{phase:stats(values) for phase,values in llama_samples.items()},
    "medians_us":medians,"paired_vs_implicit":paired,"paired_vs_live_llama":paired_live,
    "historical_eligible_arms":historical_eligible,"eligible_arms":eligible,
    "promotion_arm":PROMOTION_ARM,"promotion_gate":PROMOTION_ARM in eligible,"orders":orders}
  ROOT.mkdir(parents=True,exist_ok=True);(ROOT/"result.json").write_text(json.dumps(result,indent=2)+"\n")
  print(json.dumps({"correctness":correctness,"medians_us":medians,
    "live_llama_medians_us":live_medians,"paired_vs_implicit":paired,"paired_vs_live_llama":paired_live,
    "historical_eligible_arms":historical_eligible,"eligible_arms":eligible},sort_keys=True))

if __name__ == "__main__": run()
