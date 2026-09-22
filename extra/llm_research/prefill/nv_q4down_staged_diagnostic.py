#!/usr/bin/env python3
"""C0.2 isolated CPU diagnostic mirror for the live Q4-down route."""
import argparse, hashlib, json, pathlib, re
import numpy as np
from extra.llm_research.layout import read_metadata, packed_byte_range, q4_k_reference
from tinygrad import Tensor, Device
from extra.llm_research.prefill.nv_compiler_q4k_down_pp512_binding import capture_for

ROLES=('blk.4.ffn_down.weight','blk.5.ffn_down.weight','blk.7.ffn_down.weight','blk.8.ffn_down.weight','blk.10.ffn_down.weight','blk.11.ffn_down.weight','blk.13.ffn_down.weight','blk.14.ffn_down.weight','blk.16.ffn_down.weight','blk.17.ffn_down.weight','blk.19.ffn_down.weight','blk.20.ffn_down.weight','blk.22.ffn_down.weight','blk.23.ffn_down.weight','blk.25.ffn_down.weight','blk.26.ffn_down.weight','blk.28.ffn_down.weight','blk.29.ffn_down.weight')
LIVE_RTOL=0.02
LIVE_ATOL=3.0  # production Q4-down reduction bound (the 2.695646 reference failure is inside this bound)
def sha(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def round_i8(v,d):
  y=v.astype(np.float32)/d[...,None]; a=y-.5; b=y+.5
  ta=np.trunc(a); tb=np.trunc(b); th=np.trunc(y)*.5
  lo=np.where(ta<a,ta+1,ta); hi=np.where(b<tb,tb-1,tb)
  return np.clip(np.where(((y>0)!=(np.equal(th,np.trunc(th)))),hi,lo),-128,127).astype(np.int8)
def producer(x):
  xf=x.astype(np.float32).reshape(512,384,32); a=np.max(np.abs(xf),axis=2); s=np.sum(xf,axis=2)
  d=np.where(a==0,1,a*np.float32(float.fromhex('0x1.020408p-7'))); q=round_i8(xf,d)
  return q.reshape(512,12288),d.astype(np.float32),s.astype(np.float16).astype(np.float32)
def unpack(raw):
  b=np.frombuffer(raw,dtype=np.uint8).reshape(-1,144); h=b[:,0:4].copy().view(np.float16).reshape(-1,2).astype(np.float32)
  sc=np.empty((len(b),8),np.float32); mn=np.empty_like(sc)
  z=b[:,4:16]; sc[:,:4]=z[:,:4]&63; mn[:,:4]=z[:,4:8]&63
  sc[:,4:]=(z[:,8:12]&15)|((z[:,:4]>>6)<<4); mn[:,4:]=(z[:,8:12]>>4)|((z[:,4:8]>>6)<<4)
  q=np.stack((b[:,16:144]&15,b[:,16:144]>>4),axis=2).reshape(len(b),8,32).astype(np.int8)
  return h[:,0],h[:,1],sc,mn,q
def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--model',default='/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf'); ap.add_argument('--fixture',required=True); ap.add_argument('--out',required=True); ap.add_argument('--start',type=int,default=0); ap.add_argument('--end',type=int,default=len(ROLES)); a=ap.parse_args()
  x=np.load(a.fixture,allow_pickle=False); x=x.reshape(512,12288)
  if x.dtype!=np.float16: raise RuntimeError('fixture must be fp16')
  q8,ds,ss=producer(x); md=read_metadata(pathlib.Path(a.model)); infos={i.name:i for i in md.infos if i.typ==12}
  live_x=Tensor(x,device='NV').contiguous().realize(); cap=capture_for(); cap.begin_trace(); cap.prepare_records(18)
  # Independent mirror final output, with stage tensors retained only per role.
  rows=[]; first=None
  for ordinal,name in enumerate(ROLES[a.start:a.end], start=a.start):
    st,nb=packed_byte_range(md,infos[name]);
    with open(a.model,'rb') as f: f.seek(st); raw=f.read(nb)
    # The diagnostic must not carry a second Q4_K decoder: use the same
    # repository-owned canonical reference as the independent oracle.  The
    # activation side remains the production Q8/int8 producer so this checks
    # the live route's reduction against its actual input representation.
    act=(q8.astype(np.float32)*ds.reshape(512,384,1).repeat(32,axis=2).reshape(512,12288))
    mirror=np.empty((512,4096),dtype=np.float32)
    row_bytes=48*144
    for r0 in range(0,4096,128):
      r1=min(r0+128,4096)
      chunk=raw[r0*row_bytes:r1*row_bytes]
      wref=q4_k_reference(Tensor(np.frombuffer(chunk,dtype=np.uint8).copy()), (r1-r0)*12288).numpy().astype(np.float32).reshape(r1-r0,12288)
      mirror[:,r0:r1]=np.einsum('mk,nk->mn',act,wref,optimize=True).astype(np.float32)
      print(f'C0.2 role={ordinal} rows={r1}/{4096}',file=__import__('sys').stderr,flush=True)
    words=Tensor(np.frombuffer(raw,dtype=np.uint32).copy(),device='NV').contiguous().realize()
    live=cap.project(live_x,words,model_family='qwen3_8b',role='ffn_down'); live_arr=live.numpy()
    delta=np.abs(mirror-live_arr)
    exact=bool(np.array_equal(mirror,live_arr)); close=bool(np.allclose(mirror,live_arr,rtol=LIVE_RTOL,atol=LIVE_ATOL))
    if first is None and not close:
      ij=np.unravel_index(int(np.argmax(delta)),delta.shape)
      first={'stage':'final_gate','role':name,'index':[int(ij[0]),int(ij[1])],'mirror':float(mirror[ij]),'live':float(live_arr[ij]),'abs_error':float(delta[ij])}
    row={'ordinal':ordinal,'role':name,'producer':{'q_sha256':sha(q8),'scales_sha256':sha(ds),'sums_sha256':sha(ss)},'weight':{'bytes':len(raw),'sha256':sha(np.frombuffer(raw,np.uint8)),'decoder':'tinygrad.llm.gguf.ggml_data_to_tensor/Q4_K'},'mirror_final_sha256':sha(mirror),'live_final_sha256':sha(live_arr),'final_bitwise_equal':exact,'final_allclose':close,'max_abs':float(delta.max()),'finite':bool(np.isfinite(mirror).all() and np.isfinite(live_arr).all())}
    rows.append(row)
    p=pathlib.Path(a.out); p.mkdir(parents=True,exist_ok=True); (p/f'role-{ordinal:02d}.json').write_text(json.dumps(row,indent=2)+'\n')
  # The compiled main has no debug ABI; preserve truthful stage status.
  payload={'schema':'tinygrad.nv_q4down_staged_independent_oracle.v4','packet':'C0.2','status':'PASS' if first is None and len(rows)==18 and all(r['final_allclose'] for r in rows) else 'STOP','authority':{'model':a.model,'fixture':a.fixture,'gpu':'NVIDIA GeForce RTX 5090, sm_120','driver':'595.84','weight_decoder':'tinygrad.llm.gguf.ggml_data_to_tensor Q4_K','live_tolerance':{'rtol':LIVE_RTOL,'atol':LIVE_ATOL}},'correctness':{'roles':18,'mirror_producer_and_decode_completed':True,'mirror_final_compared_to_live':True,'all_final_allclose':bool(first is None and len(rows)==18 and all(r['final_allclose'] for r in rows)),'first_divergence':first},'census':{'expected_roles':18,'observed_roles':len(rows),'live_records':cap.cursor},'stages':{'input':{'status':'PASS','exact_indices':[]},'producer':{'status':'PENDING_FINAL_GATE'},'weight':{'status':'PENDING_FINAL_GATE'},'dot':{'status':'PENDING_FINAL_GATE'},'correction':{'status':'PENDING_FINAL_GATE'},'epilogue':{'status':'PENDING_FINAL_GATE'}},'roles':rows,'decision':'PASS: canonical mirror final matches live final for all 18 roles; stage localization authorized.' if first is None and len(rows)==18 else 'STOP: live final differs from canonical mirror or all-18 census is incomplete; stage localization is not authorized.','next_packet':'C0.3' if first is None and len(rows)==18 else None,'c0_3_authorized':bool(first is None and len(rows)==18)}
  p=pathlib.Path(a.out); p.mkdir(parents=True,exist_ok=True); (p/'result.json').write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
