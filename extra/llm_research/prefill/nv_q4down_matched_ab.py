#!/usr/bin/env python3
"""Matched Q4-down candidate vs installed fp16 weight A/B (18 type12 roles)."""
import argparse, hashlib, json, pathlib, re, statistics, time, collections
import numpy as np
from tinygrad import Tensor, Device, dtypes, TinyJit
from tinygrad.uop.ops import Ops
from extra.llm_research.layout import read_metadata, packed_u32_slice
from extra.llm_research.prefill.nv_compiler_q4k_down_pp512_binding import capture_for as compiler_capture_for
from extra.llm_research.prefill.nv_llama_packed_q4k_down_pp512_binding import binding_for as llama_binding_for

MODEL='/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf'
def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--model',default=MODEL); ap.add_argument('--z',required=True); ap.add_argument('--out',required=True); ap.add_argument('--rounds',type=int,default=9); ap.add_argument('--candidate',choices=('wide','streamk'),default='wide'); ap.add_argument('--streamk-unroll',type=int,choices=(1,2,4,8),default=None); a=ap.parse_args()
  md=read_metadata(pathlib.Path(a.model)); infos=[i for i in md.infos if i.name.endswith('.ffn_down.weight') and i.typ==12]
  if len(infos)!=18: raise RuntimeError(f'expected exactly 18 type12 metadata names, found {len(infos)}')
  z=np.load(a.z)
  if z.shape==(1,512,12288): z=z.reshape(512,12288)
  if z.shape!=(512,12288) or z.dtype!=np.float16: raise RuntimeError(f'expected z.npy float16 (512,12288), got {z.shape} {z.dtype}')
  x=Tensor(z,device='NV').contiguous().realize()
  # Metadata order is authoritative; map each exact name to its model block.
  rows=[]
  llama=llama_binding_for()
  weights=[]
  for info in infos:
    mi=re.fullmatch(r'blk\.(\d+)\.ffn_down\.weight',info.name)
    if mi is None: raise RuntimeError(f'unexpected metadata name {info.name}')
    idx=int(mi.group(1)); w=packed_u32_slice(pathlib.Path(a.model),md,info,device='NV').contiguous().realize()
    weights.append(w)
    rows.append({'metadata_name':info.name,'block':idx})
  # Real lifecycle timing: each TinyJit receives a dynamic activation and
  # constructs producer/main/fixup calls inside its captured graph. No host
  # reduction or prebuilt-output sink is included in this window.
  ctim=compiler_capture_for(variant=a.candidate, streamk_unroll=a.streamk_unroll); ltim=llama.new_capture()
  def cfn(act):
    ctim.begin_trace(); ctim.prepare_records(18)
    return tuple(ctim.project(act,w,model_family='qwen3_8b',role='ffn_down') for w in weights)
  def lfn(act):
    ltim.begin_trace()
    return tuple(ltim.project(act,w,model_family='qwen3_8b',role='ffn_down') for w in weights)
  cj,lj=TinyJit(cfn),TinyJit(lfn)
  xa=Tensor(z,device='NV').contiguous().realize(); xb=Tensor((z*0.75).astype(np.float16),device='NV').contiguous().realize()
  def run(j,act):
    out=j(act); Tensor.realize(*out); Device['NV'].synchronize(); return out
  for _ in range(3): run(cj,xa); run(lj,xa)
  live_samples={'candidate':[],'llama':[]}; orders=[]
  for i in range(31):
    order=('candidate','llama') if i%2==0 else ('llama','candidate'); orders.append(order)
    for arm in order:
      Device['NV'].synchronize(); st=time.perf_counter_ns(); run(cj if arm=='candidate' else lj,xa); live_samples[arm].append((time.perf_counter_ns()-st)/1e6)
  ca_live=np.stack([v.numpy() for v in run(cj,xa)]); fa_live=np.stack([v.numpy() for v in run(lj,xa)])
  cb_live=np.stack([v.numpy() for v in run(cj,xb)]); fb_live=np.stack([v.numpy() for v in run(lj,xb)])
  distinct=np.max(np.abs(cb_live-ca_live))>0
  distinct_control=np.max(np.abs(fb_live-fa_live))>0
  def calls(j): return [u for u in j.captured.linear.toposort() if u.op is Ops.CALL and u.src and u.src[0].op is Ops.PROGRAM]
  def names(j): return collections.Counter(getattr(u.src[0].arg,'name','') for u in calls(j))
  cn,ln=names(cj),names(lj)
  candidate_main=sum(v for n,v in cn.items() if (n.startswith('r_') if a.candidate=='wide' else n=='q4_down_streamk'))
  candidate_fixup=sum(v for n,v in cn.items() if n=='q4k_imma_fixup_active')
  expected_calls=36 if a.candidate=='wide' else 54
  expected_fixup=0 if a.candidate=='wide' else 18
  control_main=sum(v for n,v in ln.items() if 'dense_mul_mat_q' in n and 'fixup' not in n)
  census_ok=(len(calls(cj))==expected_calls and len(calls(lj))==54 and candidate_main==18 and candidate_fixup==expected_fixup and control_main==18)
  xb_close=bool(np.allclose(cb_live,fb_live,rtol=0.02,atol=0.5))
  payload={'schema':'tinygrad.nv_q4down_matched_ab.v4','status':'PASS' if np.isfinite(ca_live).all() and np.isfinite(fa_live).all() and np.isfinite(cb_live).all() and np.isfinite(fb_live).all() and np.allclose(ca_live,fa_live,rtol=0.02,atol=0.5) and xb_close and distinct and distinct_control and census_ok else 'FAIL','fixture':{'model':a.model,'z':str(a.z),'z_shape':list(z.shape),'z_dtype':str(z.dtype),'z_sha256':hashlib.sha256(z.tobytes()).hexdigest()},'census':{'metadata_type12_names':rows,'expected_mains':18,'candidate_mains':candidate_main,'control_mains':control_main,'candidate_program_names':dict(cn),'control_program_names':dict(ln),'candidate_calls':len(calls(cj)),'control_calls':len(calls(lj)),'census_ok':census_ok},'correctness':{'finite_candidate':bool(np.isfinite(ca_live).all()),'finite_control':bool(np.isfinite(fa_live).all()),'max_abs':float(np.max(np.abs(ca_live-fa_live))),'mean_abs':float(np.mean(np.abs(ca_live-fa_live))),'distinct_activation_candidate':bool(distinct),'distinct_activation_control':bool(distinct_control),'second_activation_allclose':xb_close,'allclose_rtol_0p02_atol_0p5':bool(np.allclose(ca_live,fa_live,rtol=0.02,atol=0.5))},'timing_ms':{'status':'LIVE_GRAPH_R31','candidate_samples':live_samples['candidate'],'control_samples':live_samples['llama'],'candidate_median':statistics.median(live_samples['candidate']),'control_median':statistics.median(live_samples['llama']),'candidate_delta_median':statistics.median(live_samples['candidate'])-statistics.median(live_samples['llama']),'orders':orders}}
  p=pathlib.Path(a.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload,indent=2))
  if payload['status']!='PASS': raise SystemExit(1)
if __name__=='__main__': main()
