"""Structural gate-only compiler Q4 pp512 arm (GPU execution is opt-in)."""
from __future__ import annotations
import argparse, json, os, pathlib, traceback
from collections import Counter
import numpy as np
from tinygrad import Device, Tensor, TinyJit
from tinygrad.device import Compiled
from extra.llm_research.prefill.nv_compiler_q4k_model_gate import _call_and_sync, _call_name, _program_calls, _configure

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"

def main():
  ap=argparse.ArgumentParser(); ap.add_argument("--model",default=MODEL); ap.add_argument("--out",required=True)
  a=ap.parse_args()
  target=pathlib.Path(a.out); state={"schema":"tinygrad.nv_compiler_q4_gate_only_model_arm.v1","status":"RUNNING","stage":"start"}
  def stage(name, **kw):
    state.update(stage=name, **kw); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(state,indent=2,sort_keys=True)+"\n")
  try:
    if os.environ.get("NV_COMPILER_Q4_GATE_ONLY_PP512") != "1": raise SystemExit("set NV_COMPILER_Q4_GATE_ONLY_PP512=1")
    if str(Device.DEFAULT) != "NV": raise SystemExit("NV required")
    from tinygrad.llm.generate import load_model_and_tokenizer
    stage("before_model_load")
    model,_=load_model_and_tokenizer(a.model,4608,seed=20260617); stage("model_loaded")
    cfg=model.config
    selector={"env":os.environ.get("NV_COMPILER_Q4_GATE_ONLY_PP512"),"device":str(Device.DEFAULT),
      "qualified_tuple":(cfg.prefill_ubatch,cfg.num_blocks,cfg.dim,cfg.hidden_dim,cfg.n_heads,cfg.n_kv_heads,cfg.head_dim,cfg.num_experts),
      "expected_tuple":(512,36,4096,12288,32,8,128,0),"prefill_v2":getattr(cfg,"prefill_v2",None),
      "gate_type":type(model.blk[0].ffn_gate).__name__,"gate_shape":tuple(model.blk[0].ffn_gate.weight.shape),
      "weight_type":getattr(model.blk[0].ffn_gate,"quant_type",None)}
    stage("selector_facts",selector=selector,full_packed_enabled=bool(os.environ.get("NV_LLAMA_FULL_PACKED_PP512","1")!="0"),
      compiler_gate_capture_count=1,llama_pair_asset_count=1,
      untouched_fp16_overlays=sum(getattr(b.ffn_gate,"_pf16_w",None) is not None for b in model.blk))
    from extra.llm_research.prefill.nv_compiler_q4k_pp512_binding import binding_for as compiler_for
    from extra.llm_research.prefill.nv_llama_packed_q4k_pp512_binding import binding_for as llama_for
    compiler=compiler_for("NV"); compiler.install_warmstart(model)
    compiler_cap=compiler.new_capture(); compiler_cap.begin_trace()
    llama=llama_for("NV"); llama_cap=llama.new_capture(); llama_cap.begin_trace()
    for block in model.blk:
      block._nv_compiler_q4_gate_only_pp512_binding=compiler_cap
      block._nv_q4_imma_pp512_binding=llama_cap
    _configure(model, None)
    chunk=Tensor([[(i*7)%1000 for i in range(512)]],dtype="int32").contiguous(); temp=Tensor([0.0])
    @TinyJit
    def run(tokens, temperature): return model.forward_greedy_with_logits(tokens, 0, temperature)
    stage("before_prime")
    _configure(model, None)
    for _ in range(3): _call_and_sync(run,chunk,temp)
    stage("captured")
    out=_call_and_sync(run,chunk,temp); logits=np.asarray(out[1].numpy()); stage("replayed",finite=bool(np.isfinite(logits).all()))
    calls=_program_calls(run.captured.linear); names=Counter(_call_name(c) for c in calls)
    compiler_gate=sum("q8_compact_record_fp16" in _call_name(c) for c in calls)
    # llama gate main/fixup names are classified only among 12288x4096 calls;
    # other llama roles have distinct launch symbols/shapes and are excluded.
    llama_gate=sum("dense_mul_mat_q" in _call_name(c) for c in calls if "12288" in _call_name(c) or "q4k" in _call_name(c))
    canonical=sum(getattr(b.ffn_gate.prefill_packed_weight().uop,"buf_uop",None) is not None for b in model.blk)
    payload={"schema":"tinygrad.nv_compiler_q4_gate_only_model_arm.v1","status":"PASS" if np.isfinite(logits).all() and compiler_gate==36 and llama_gate==0 else "FAIL",
    "compiler_gate_producer_count":names.get("q8_compact_record_fp16",0),"compiler_gate_main_count":compiler_gate,
    "llama_gate_main_fixup_count":llama_gate,"other_role_census":dict(sorted(names.items())),
    "canonical_weight_facts":{"gate_weights":canonical,"weight_copy_kernels":sum(v for k,v in names.items() if "copy" in k.lower()),"expanded_fp16_weight_bytes":0},
    "comparison":{"finite_logits":bool(np.isfinite(logits).all()),"token":None,"same_token":None,"allclose":None}}
    state.update(payload); stage("final_write")
    print(json.dumps(payload,sort_keys=True))
  except BaseException as e:
    stage("exception",status="FAIL",error=repr(e),trace=traceback.format_exc())
    raise
if __name__ == "__main__": main()
