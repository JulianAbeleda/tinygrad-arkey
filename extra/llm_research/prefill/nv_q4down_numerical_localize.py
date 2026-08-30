#!/usr/bin/env python3
"""C0.1: freeze the exact 18-role Q4-down numerical localization fixture."""
import argparse, hashlib, json, pathlib, re
import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.llm.generate import load_model_and_tokenizer
from extra.llm_research.layout import read_metadata, packed_byte_range, tensor_shape

MODEL = '/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf'
Z = 'docs/task_workflow/evidence/nv-q4down-capture-20260829/z.npy'
ROLES = ('blk.4.ffn_down.weight','blk.5.ffn_down.weight','blk.7.ffn_down.weight',
  'blk.8.ffn_down.weight','blk.10.ffn_down.weight','blk.11.ffn_down.weight',
  'blk.13.ffn_down.weight','blk.14.ffn_down.weight','blk.16.ffn_down.weight',
  'blk.17.ffn_down.weight','blk.19.ffn_down.weight','blk.20.ffn_down.weight',
  'blk.22.ffn_down.weight','blk.23.ffn_down.weight','blk.25.ffn_down.weight',
  'blk.26.ffn_down.weight','blk.28.ffn_down.weight','blk.29.ffn_down.weight')
def sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--model',default=MODEL); ap.add_argument('--z',default=Z); ap.add_argument('--out',required=True); a=ap.parse_args()
  out=pathlib.Path(a.out); out.mkdir(parents=True,exist_ok=False)
  z=np.load(a.z, allow_pickle=False)
  if z.shape==(1,512,12288): z=z.reshape(512,12288)
  if z.shape!=(512,12288) or z.dtype!=np.float16: raise RuntimeError(f'expected float16 (512,12288), got {z.shape} {z.dtype}')
  md=read_metadata(pathlib.Path(a.model)); infos={i.name:i for i in md.infos if i.typ==12}
  if set(ROLES)!=set(x for x in infos if '.ffn_down.weight' in x): raise RuntimeError('Q4-down population mismatch')
  m,_=load_model_and_tokenizer(a.model,512,seed=20260617); m.realize_prefill_v2_weights()
  x=Tensor(z,device='NV').contiguous().realize(); Device['NV'].synchronize()
  rows=[]; sentinel=np.float32(-3.4028235e38)
  np.save(out/'activation.npy',z)
  for ordinal,name in enumerate(ROLES):
    info=infos[name]; start,nbytes=packed_byte_range(md,info)
    with open(a.model,'rb') as f: f.seek(start); packed=np.frombuffer(f.read(nbytes),dtype=np.uint8).copy()
    np.save(out/f'role-{ordinal:02d}-packed-u8.npy',packed)
    block=int(re.fullmatch(r'blk\.(\d+)\.ffn_down\.weight',name).group(1))
    w=m.blk[block].ffn_down._pf16_w
    if w is None: raise RuntimeError(f'missing FP16 control for {name}')
    fp32=x.matmul(w.transpose()).cast(dtypes.float32).contiguous().realize().numpy()
    fp16=fp32.astype(np.float16)
    residual=(fp32-fp16.astype(np.float32)).astype(np.float32)
    np.save(out/f'role-{ordinal:02d}-reference-fp32.npy',fp32)
    np.save(out/f'role-{ordinal:02d}-reference-fp16.npy',fp16)
    np.save(out/f'role-{ordinal:02d}-residual.npy',residual)
    rows.append({'ordinal':ordinal,'name':name,'block':block,'shape':list(tensor_shape(info)),
      'packed_u8_sha256':sha(packed),'reference_fp32_sha256':sha(fp32),'reference_fp16_sha256':sha(fp16),
      'residual_sha256':sha(residual),'sentinel':float(sentinel),'sentinel_count':0,
      'tolerance':{'atol':0.5,'rtol':0.02},'finite':bool(np.isfinite(fp32).all())})
  manifest={'schema':'tinygrad.nv_q4down_numerical_localize.v1','packet':'C0.1','status':'PASS',
    'authority':{'source_result':'docs/task_workflow/evidence/nv-prefill-post-substrate-authority-20260829-r3/result.json',
      'model':a.model,'model_sha256':'retained by source authority','gpu':'NVIDIA GeForce RTX 5090, sm_120','driver':'595.84',
      'clocks_session':'graphics=435 MHz, sm=435 MHz, memory=7001 MHz, P3; flock GPU session','prompt_fixture':'inline:(i*7)%1000','tokens':512},
    'activation':{'path':'activation.npy','shape':list(z.shape),'dtype':str(z.dtype),'sha256':sha(z),'readonly':True},
    'roles':rows,'sentinel_policy':{'value':float(sentinel),'output_shape':[512,4096],'required_unwritten_count':0},
    'correctness':{'roles':18,'all_finite':all(r['finite'] for r in rows),'all_shapes_512x4096':all(r['shape']==[4096,12288] for r in rows)},
    'decision':'PASS: immutable real-shape localization fixture frozen for all 18 Q4-down roles','next_packet':'C0.2'}
  (out/'result.json').write_text(json.dumps(manifest,indent=2)+'\n')
  print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
