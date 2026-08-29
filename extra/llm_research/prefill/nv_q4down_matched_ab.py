#!/usr/bin/env python3
"""Matched Q4-down candidate vs installed fp16 weight A/B (18 type12 roles)."""
import argparse, hashlib, json, pathlib, re, statistics, time
import numpy as np
from tinygrad import Tensor, Device, dtypes
from extra.llm_research.layout import read_metadata, packed_u32_slice
from extra.llm_research.prefill.nv_compiler_q4k_down_pp512_binding import capture_for
from tinygrad.llm.generate import load_model_and_tokenizer

MODEL='/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf'
def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--model',default=MODEL); ap.add_argument('--z',required=True); ap.add_argument('--out',required=True); ap.add_argument('--rounds',type=int,default=9); a=ap.parse_args()
  md=read_metadata(pathlib.Path(a.model)); infos=[i for i in md.infos if i.name.endswith('.ffn_down.weight') and i.typ==12]
  if len(infos)!=18: raise RuntimeError(f'expected exactly 18 type12 metadata names, found {len(infos)}')
  z=np.load(a.z)
  if z.shape==(1,512,12288): z=z.reshape(512,12288)
  if z.shape!=(512,12288) or z.dtype!=np.float16: raise RuntimeError(f'expected z.npy float16 (512,12288), got {z.shape} {z.dtype}')
  x=Tensor(z,device='NV').contiguous().realize(); m,_=load_model_and_tokenizer(a.model,512,seed=20260617)
  m.realize_prefill_v2_weights()
  # Metadata order is authoritative; map each exact name to its model block.
  rows=[]; cap=capture_for(); cap.begin_trace(); cap.prepare_records(18)
  cand=[]; ctrl=[]
  for info in infos:
    mi=re.fullmatch(r'blk\.(\d+)\.ffn_down\.weight',info.name)
    if mi is None: raise RuntimeError(f'unexpected metadata name {info.name}')
    idx=int(mi.group(1)); lin=m.blk[idx].ffn_down; w=packed_u32_slice(pathlib.Path(a.model),md,info,device='NV').contiguous().realize()
    y=cap.project(x,w,model_family='qwen3_8b',role='ffn_down'); cand.append(y)
    cw=lin._pf16_w
    if cw is None: raise RuntimeError(f'control _pf16_w missing for {info.name}')
    ctrl.append(x.matmul(cw.transpose()).cast(dtypes.float32).contiguous().realize())
    rows.append({'metadata_name':info.name,'block':idx})
  # One device-side sink per output; timing contains no host transfer.
  def sink(vals):
    s=Tensor.zeros(1,device='NV')
    for y in vals: s=(s+y.reshape(-1).sum()).realize()
    return s
  sink(cand); sink(ctrl); Device['NV'].synchronize()
  def timed(vals):
    for _ in range(3): sink(vals); Device['NV'].synchronize()
    ts=[]
    for _ in range(a.rounds):
      Device['NV'].synchronize(); t=time.perf_counter_ns(); sink(vals); Device['NV'].synchronize(); ts.append((time.perf_counter_ns()-t)/1e6)
    return ts
  cts,fts=timed(cand),timed(ctrl)
  ca=np.stack([y.numpy() for y in cand]); fa=np.stack([y.numpy() for y in ctrl]); d=np.abs(ca-fa)
  calls=sum(1 for y in cand if y is not None)
  payload={'schema':'tinygrad.nv_q4down_matched_ab.v1','status':'PASS' if np.isfinite(ca).all() and np.isfinite(fa).all() and np.allclose(ca,fa,rtol=0.02,atol=0.5) else 'FAIL','fixture':{'model':a.model,'z':str(a.z),'z_shape':list(z.shape),'z_dtype':str(z.dtype),'z_sha256':hashlib.sha256(z.tobytes()).hexdigest()},'census':{'metadata_type12_names':rows,'expected_mains':18,'candidate_mains':calls,'control_mains':len(ctrl),'producer_records':cap.cursor},'correctness':{'finite_candidate':bool(np.isfinite(ca).all()),'finite_control':bool(np.isfinite(fa).all()),'max_abs':float(d.max()),'mean_abs':float(d.mean()),'allclose_rtol_0p02_atol_0p5':bool(np.allclose(ca,fa,rtol=0.02,atol=0.5))},'timing_ms':{'candidate_samples':cts,'candidate_min':min(cts),'candidate_median':statistics.median(cts),'control_samples':fts,'control_min':min(fts),'control_median':statistics.median(fts),'candidate_minus_control_min':min(cts)-min(fts),'candidate_minus_control_median':statistics.median(cts)-statistics.median(fts)}}
  p=pathlib.Path(a.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload,indent=2))
  if payload['status']!='PASS': raise SystemExit(1)
if __name__=='__main__': main()
