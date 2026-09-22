#!/usr/bin/env python3
"""E1 isolated matched control/candidate lifecycle bracket."""
import argparse, json, pathlib, statistics, time
import numpy as np
from types import SimpleNamespace
from tinygrad import Tensor, dtypes
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.q6k_vocab_manyrow import Q6KVocabManyRowAdmission, q6k_vocab_manyrow_call

def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--fixture',type=pathlib.Path,required=True); ap.add_argument('--reps',type=int,default=9); ap.add_argument('--out',type=pathlib.Path,required=True); a=ap.parse_args()
  p=a.fixture; h=np.fromfile(p/'final-hidden-row.f32',dtype=np.float32).reshape(1,1,4096)
  raw=np.fromfile(p/'output.weight.q6_k.bin',dtype=np.uint8)
  dev='NV'; x=Tensor(h,dtype=dtypes.float32,device=dev)
  packed=Tensor(raw,dtype=dtypes.uint8).reshape(-1)
  # Ordinary installed control: canonical GGUF Q6_K dequantization followed by matmul.
  w=ggml_data_to_tensor(packed,151936*4096,14).reshape(151936,4096).to(dev)
  lin=SimpleNamespace(q6k_storage=SimpleNamespace(halfs=packed.bitcast(dtypes.uint16)),out_features=151936,in_features=4096,bias=None)
  def control(): return w.matmul(x.reshape(4096,1)).reshape(1,1,151936).realize()
  def candidate(): return q6k_vocab_manyrow_call(Q6KVocabManyRowAdmission(),lin,x).realize()
  control(); candidate();
  samples={'control':[],'candidate':[]}; checks={}
  for i in range(a.reps):
    for name,fn in (('control',control),('candidate',candidate)):
      t=time.perf_counter(); y=fn(); dt=(time.perf_counter()-t)*1e6; arr=y.numpy().reshape(-1); samples[name].append(dt)
      checks[name]={'shape':list(y.shape),'finite':bool(np.isfinite(arr).all()),'nonzero':bool(np.count_nonzero(arr)==arr.size),'argmax':int(arr.argmax())}
  cm=statistics.median(samples['control']); pm=statistics.median(samples['candidate'])
  result={'schema':'tinygrad.nv_vocab_e1_lifecycle_r9.v1','fixture':str(p),'reps':a.reps,'samples_us':samples,'min_us':{k:min(v) for k,v in samples.items()},'median_us':{'control':cm,'candidate':pm},'candidate_minus_control_us':pm-cm,'checks':checks,'verdict':'PASS' if all(c['finite'] and c['nonzero'] for c in checks.values()) and checks['control']['argmax']==checks['candidate']['argmax'] else 'STOP'}
  a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,sort_keys=True))
if __name__=='__main__': main()
