#!/usr/bin/env python3
"""Focused research sweep for compiler-native Q6_K x compact-Q8 prefill."""
from __future__ import annotations
import argparse, json, pathlib, traceback
import numpy as np
from tinygrad import Tensor
from extra.llm_research.prefill.nv_compiler_q6k_imma_gate import _record, _run
from extra.llm_research.layout import GGML_Q6_K, packed_u16_slice, read_metadata

GEOMETRIES = (
  (64, 64, 1, 1, 128), (64, 64, 2, 2, 128),
  (64, 128, 1, 4, 128), (64, 128, 2, 4, 256),
  (128, 64, 4, 1, 128), (128, 64, 4, 2, 256),
  (128, 128, 2, 4, 256),
)

def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--model', required=True)
  ap.add_argument('--role', default='down', choices=('down','v'))
  ap.add_argument('--rounds', type=int, default=9); ap.add_argument('--out', required=True)
  ap.add_argument('--artifacts', required=True); args=ap.parse_args()
  if args.rounds < 9: raise ValueError('qualification requires R9 or greater')
  m,n,k = (512,4096,12288) if args.role == 'down' else (512,1024,4096)
  name = 'blk.0.ffn_down.weight' if args.role == 'down' else 'blk.0.attn_v.weight'
  artifacts=pathlib.Path(args.artifacts); artifacts.mkdir(parents=True,exist_ok=True)
  meta=read_metadata(pathlib.Path(args.model)); info=next(i for i in meta.infos if i.name==name)
  if info.typ != GGML_Q6_K: raise RuntimeError(f'illegal fixture {info}')
  halfs=packed_u16_slice(pathlib.Path(args.model),meta,info,device='NV').contiguous().realize()
  record_np,_,_=_record(m,k); record=Tensor(record_np,device='NV').contiguous().realize()
  out={'schema':'tinygrad.nv_compiler_q6k_wide_sweep.v1','shape':{'M':m,'N':n,'K':k},'role':args.role,'results':[]}
  for geometry in GEOMETRIES:
    tag='g_'+'_'.join(map(str,geometry))
    try:
      r=_run(tag,m,n,k,halfs,record,args.rounds,artifacts,geometry)
      out['results'].append(r)
      print(json.dumps({'geometry':geometry,'passed':r['passed'],'timing':r['timing'],'global':r['geometry']['global_size']},sort_keys=True),flush=True)
    except Exception as e:
      out['results'].append({'geometry':geometry,'legal':False,'error':str(e),'traceback':traceback.format_exc()})
      print(json.dumps({'geometry':geometry,'legal':False,'error':str(e)},sort_keys=True),flush=True)
  out['passed']=any(x.get('passed',False) for x in out['results'])
  pathlib.Path(args.out).write_text(json.dumps(out,indent=2)+'\n')
  print(json.dumps({'best':min((x for x in out['results'] if x.get('passed')),key=lambda x:x['timing']['min_us'],default=None)},sort_keys=True))

if __name__=='__main__': main()
