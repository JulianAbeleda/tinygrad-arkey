import fcntl,hashlib,json,pathlib,statistics
import numpy as np
from tinygrad import Device,Tensor,dtypes
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.uop.ops import UOp
from extra.llm_research.layout import GGML_Q6_K,packed_u16_slice,read_metadata
from extra.llm_research.prefill.bench_nv_q6_oracle_broad_cta import _record as broad_record
from extra.llm_research.prefill.bench_nv_q6_oracle_reduction_policy import LAUNCH_SHARED_BYTES,_buf,_compile_ast
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
  "llama_factor":{"fp32_contraction":"both","weight_scale_contract":"legacy","factor_dA":True,"fp32_p_tree":"left"},
}
ARMS=tuple(SPECS)
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

def run():
  ART.mkdir(parents=True,exist_ok=True)
  lock=open("/tmp/nv-q6-oracle-gpu.lock","w");fcntl.flock(lock,fcntl.LOCK_EX)
  compiled={}
  for arm in ARMS:
    ast=build_packed_one_body_ast(partial_output_layout="destination_major",region_load_bridge_q8_panel1=True,
                                  **SPECS[arm])
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
  samples={arm:{"main":[],"fixup":[],"total":[]} for arm in ARMS};orders=[]
  for iteration in range(31):
    shift=iteration%len(ARMS);order=ARMS[shift:]+ARMS[:shift];orders.append(order)
    for arm in order:
      main_us=main(compiled[arm]["program"],partials[arm],halfs,q8);fix_us=fix(fix_program,outs[arm],partials[arm],slots,cts)
      samples[arm]["main"].append(main_us);samples[arm]["fixup"].append(fix_us);samples[arm]["total"].append(main_us+fix_us)
  paired={}
  for arm in ARMS[1:]:
    paired[arm]={}
    for phase in ("main","fixup","total"):
      delta=[candidate-anchor for anchor,candidate in zip(samples["trusted_implicit"][phase],samples[arm][phase])]
      median=statistics.median(delta)
      paired[arm][phase]={"median_us":median,"mad_us":statistics.median(abs(x-median) for x in delta),
                          "wins":sum(x<0 for x in delta),"samples_us":delta}
  medians={arm:{phase:statistics.median(values) for phase,values in phases.items()} for arm,phases in samples.items()}
  eligible=[arm for arm in ARMS if correctness[arm]["finite"] and correctness[arm]["tolerance_failures"]==0 and medians[arm]["total"]<=TARGET_US]
  result={"schema":"tinygrad.nv_q6_llama_arithmetic_contract_gate.v1","oracle":{"llama_main_us":201.216,"llama_fixup_us":8.640,
    "llama_total_us":LLAMA_TOTAL_US,"target_5pct_us":TARGET_US,"numerical_reference":"pinned llama Q6 main and fixup on identical canonical Q6/Q8 records"},
    "configuration":{"samples":31,"warmups":3,"arms":SPECS,"partial_output_layout":"destination_major","region_load_bridge_q8_panel1":True,
      "gpu_lock":"/tmp/nv-q6-oracle-gpu.lock"},"correctness":correctness,"compiler":{arm:compiled[arm]["artifact"] for arm in ARMS},
    "fixup_compiler":fix_artifact,"timing":{arm:{phase:stats(values) for phase,values in phases.items()} for arm,phases in samples.items()},
    "medians_us":medians,"paired_vs_implicit":paired,"eligible_arms":eligible,"promotion_gate":bool(eligible),"orders":orders}
  ROOT.mkdir(parents=True,exist_ok=True);(ROOT/"result.json").write_text(json.dumps(result,indent=2)+"\n")
  print(json.dumps({"correctness":correctness,"medians_us":medians,"paired_vs_implicit":paired,"eligible_arms":eligible},sort_keys=True))

if __name__ == "__main__": run()
