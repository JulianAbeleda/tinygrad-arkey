"""Static contract harness for the provenance-clean many-row Q6 vocabulary arm.

This is intentionally default-off and GPU-free.  It records the exact ABI that
the existing generated primitive exposes; model integration is a later packet.
"""
from __future__ import annotations
import argparse, json
import time
from pathlib import Path

from tinygrad.llm.q6k_vocab_manyrow import K, Q8_GROUPS, Q8_WORDS, ROWS

CANDIDATE = "nv_vocab_manyrow_q8_q6"

def contract() -> dict:
  return {
    "schema": "nv_q6k_vocab_manyrow_native_candidate/v1",
    "candidate_id": CANDIDATE,
    "default_enabled": False,
    "device": "NV",
    "origin": "tinygrad_generated",
    "binary_provenance": "tinygrad UOp KernelProgram emitters",
    "producer": {"identity": "nv_vocab_manyrow/q8_provider", "input": [1, 1, K],
                 "output": [Q8_WORDS], "dtype": "uint32", "groups": Q8_GROUPS},
    "consumer": {"identity": "nv_vocab_manyrow/q6_mmvq", "weights": {"dtype": "uint16", "shape": [ROWS, K]},
                 "input_packet": [Q8_WORDS], "output": [ROWS], "dtype": "float32"},
    "output": {"shape": [1, 1, ROWS], "logits": ROWS},
    "census": {"producer_programs": 1, "consumer_programs": 1, "physical_main_invocations": 1,
               "one_owner": True, "llama_programs": 0, "evidence_cubins": 0},
    "violations": [],
    "integration": "not_model_integrated",
  }

def run_live(model_path, activation_path, rounds, preserve_input_dtype=False, candidate_kind="manyrow"):
  import collections, hashlib, statistics
  from types import SimpleNamespace
  import numpy as np
  from tinygrad import Tensor, Device, TinyJit, dtypes
  from tinygrad.uop.ops import Ops
  from extra.llm_research.layout import read_metadata, packed_u16_slice
  from extra.llm_research.prefill.nv_llama_q6k_vocab_pp512_binding import binding_for
  from extra.llm_research.prefill.nv_compiler_q4k_qo_72real_proxy import program_key
  from tinygrad.llm.q6k_vocab_manyrow import Q6KVocabManyRowAdmission, q6k_vocab_manyrow_call
  if candidate_kind == "direct-four-warp":
    from tinygrad.llm.q6k_v_mmvq import emit_q6k_v_four_warp_fp16_direct
    from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_research_program
    direct_program=KernelProgram("decode_q6k_vocab_four_warp_fp16","q6k_vocab_four_warp_fp16",KernelProgramProvenance.RESEARCH_ONLY,emit_q6k_v_four_warp_fp16_direct(rows=ROWS),OutputSpec((ROWS,),dtypes.float32))
  if rounds<1: raise ValueError("rounds must be positive")
  path=Path(model_path); md=read_metadata(path)
  info=next(i for i in md.infos if i.name=="output.weight")
  if info.typ!=14 or tuple(reversed(info.dims))!=(ROWS,K): raise ValueError("invalid canonical vocabulary weight")
  weight=packed_u16_slice(path,md,info,device="NV").contiguous().realize()
  linear=SimpleNamespace(q6k_storage=SimpleNamespace(halfs=weight),out_features=ROWS,in_features=K,bias=None)
  oracle=binding_for("NV")
  values=np.fromfile(activation_path,dtype=np.float32)
  if values.shape!=(K,) or not np.isfinite(values).all(): raise ValueError("expected finite FP32 hidden row")
  inputs=[Tensor(v,device="NV").realize() for v in (values,values*np.float32(.7))]
  @TinyJit
  def candidate(x):
    logits=(execute_research_program(Tensor.empty((ROWS,),dtype=dtypes.float32,device="NV"),weight,x.reshape(K).cast(dtypes.float16).contiguous(),program=direct_program).reshape(ROWS)
      if candidate_kind == "direct-four-warp" else q6k_vocab_manyrow_call(Q6KVocabManyRowAdmission(preserve_input_dtype=preserve_input_dtype),linear,x.reshape(1,1,K)))
    if logits is None: raise RuntimeError("candidate admission failed")
    logits=logits.reshape(ROWS)
    return logits,logits.argmax()
  @TinyJit
  def llama(x):
    logits=oracle.project(x,weight)
    return logits,logits.argmax()
  def run(jit,x):
    outputs=jit(x); Tensor.realize(*outputs); Device["NV"].synchronize(); return outputs
  for _ in range(3): run(candidate,inputs[0]); run(llama,inputs[0])
  checks=[]
  for x in inputs:
    c,ct=run(candidate,x); ca=c.numpy().copy(); token=int(ct.item())
    l,lt=run(llama,x); la=l.numpy().copy(); ref_token=int(lt.item())
    checks.append({"finite":bool(np.isfinite(ca).all() and np.isfinite(la).all()),
      "allclose":bool(np.allclose(ca,la,rtol=.02,atol=.5)),"max_abs":float(np.max(np.abs(ca-la))),
      "candidate_token":token,"llama_token":ref_token,"same_token":token==ref_token})
  samples={"candidate":[],"llama":[]}; orders=[]
  for i in range(rounds):
    order=("candidate","llama") if i%2==0 else ("llama","candidate"); orders.append(order)
    for name in order:
      Device["NV"].synchronize(); start=time.perf_counter_ns()
      run(candidate if name=="candidate" else llama,inputs[0])
      samples[name].append((time.perf_counter_ns()-start)/1e6)
  def observed(jit):
    programs=[u.src[0] for u in jit.captured.linear.toposort() if u.op is Ops.CALL and u.src and u.src[0].op is Ops.PROGRAM]
    counts=collections.Counter(program_key(p) for p in programs)
    return [{"identity":list(key),"count":count} for key,count in counts.items()]
  passed=all(c["finite"] and c["allclose"] and c["same_token"] for c in checks)
  return {"schema":"tinygrad.nv.vocab.live.v1","status":"PASS_CORRECTNESS" if passed else "FAIL",
    "qualification":"observed lifecycle; not production promoted", "weight":{"name":info.name,"shape":[ROWS,K],"ggml_type":info.typ},
    "activation":{"path":str(activation_path),"sha256":hashlib.sha256(values.tobytes()).hexdigest(),"dtype":"float32"},
    "correctness":checks,"input_boundary":"FP32 reaches provider, which internally rounds through FP16" if preserve_input_dtype else "FP32 cast to FP16 before provider",
    "preserve_input_dtype":preserve_input_dtype,"census":{"candidate":observed(candidate),"llama":observed(llama)},
    "timing_ms":{"rounds":rounds,"samples":samples,"orders":orders,
      "candidate_median":statistics.median(samples["candidate"]),"llama_median":statistics.median(samples["llama"]),
      "includes":"producer, intermediate unpack, vocabulary projection and argmax; host snapshots excluded"}}


def main() -> int:
  ap=argparse.ArgumentParser()
  ap.add_argument("--out",type=Path)
  ap.add_argument("--fixture",action="store_true")
  ap.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--rounds",type=int,default=31)
  ap.add_argument("--preserve-input-dtype",action="store_true")
  ap.add_argument("--candidate",choices=("manyrow","direct-four-warp"),default="manyrow")
  ap.add_argument("--activation",type=Path,default=Path("docs/task_workflow/evidence/nv-vocab-manyrow-e1-postnorm-fixture-20260829/final-hidden-row.f32"))
  a=ap.parse_args()
  report=run_live(a.model,a.activation,a.rounds,a.preserve_input_dtype,a.candidate) if a.fixture else contract()
  payload=json.dumps(report,indent=2)+"\n"
  if a.out: a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(payload)
  else: print(payload,end="")
  return int(report.get("status")=="FAIL")

if __name__=="__main__": raise SystemExit(main())
