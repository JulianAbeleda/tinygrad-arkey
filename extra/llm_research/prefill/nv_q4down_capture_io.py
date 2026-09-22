import argparse, hashlib, json, pathlib, numpy as np
from tinygrad import Tensor
from tinygrad.llm.generate import load_model_and_tokenizer

ap=argparse.ArgumentParser(); ap.add_argument('--model',default='/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf'); ap.add_argument('--out',required=True); a=ap.parse_args()
m,_=load_model_and_tokenizer(a.model,512,seed=20260617)
for b in m.blk: b._is_prefill=True; b._prefill_v2=True
m.blk[4]._research_capture_down_io=True
m.logits(Tensor([list(range(512))],device='NV'),0).realize(); b=m.blk[4]
if not hasattr(b,'_research_down_input'): raise RuntimeError('block4 down hook did not capture')
p=pathlib.Path(a.out); p.mkdir(parents=True,exist_ok=True)
for n,t in [('z',b._research_down_input),('fp16',b._research_down_output)]: np.save(p/(n+'.npy'),t.numpy())
o={'schema':'tinygrad.nv_q4down_capture_io.v1','status':'PASS','z_shape':list(b._research_down_input.shape),'out_shape':list(b._research_down_output.shape),'z_sha256':hashlib.sha256(b._research_down_input.numpy().tobytes()).hexdigest(),'out_sha256':hashlib.sha256(b._research_down_output.numpy().tobytes()).hexdigest()}
(p/'capture.json').write_text(json.dumps(o,indent=2)+'\n'); print(json.dumps(o))
