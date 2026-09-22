#!/usr/bin/env python3
"""Reverse wall bracket isolating host scalar readback from autoregressive GPU feedback."""
from __future__ import annotations
import argparse, json, os, pathlib, statistics, subprocess, sys
ROOT=pathlib.Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
MODEL='/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf'; PYTHON=ROOT/'.venv/bin/python'; LOCK='/tmp/tinygrad-gpu.lock'

def run_arm(arm,model_path,depth,count,reps,max_context,out,candidate_mode):
  from tinygrad import Device, Tensor
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load,_prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows
  model=_load(model_path,max_context); gen=model.generate(_prompt(model_path,depth),chunk_size=32,temperature=0.0)
  warm=[int(next(gen)) for _ in range(6)]; dev=Device['NV']; dev.synchronize(); original_item=Tensor.item; original_copyout=dev.allocator._copyout
  if arm=='candidate' and candidate_mode=='skip-item': Tensor.item=lambda self: warm[-1]
  try: settled=_settled_continuous_windows(gen,dev,count,reps)
  finally: Tensor.item=original_item; dev.allocator._copyout=original_copyout; gen.close()
  result={'schema':'tinygrad.nv_token_readback_wall.child.v1','arm':arm,'depth':depth,'count':count,'reps':reps,'max_context':max_context,
          'candidate_mode':candidate_mode,'candidate_contract':(('GPU greedy feedback unchanged; host Tensor.item replaced by constant after six warm tokens' if candidate_mode=='skip-item' else
            'native argmax writes both the ordinary GPU feedback token and a pinned-host mirror; CPU waits the compute timeline then reads the mirror' if candidate_mode=='host-mirror' else
            'normal scalar copyout with only its redundant initial device synchronize suppressed') if arm=='candidate' else 'production generate'),
          'settled':settled}; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); return result

def child(a,arm,root):
  out=root/f'{arm}.json'; env={**os.environ,'DEV':'NV','PROFILE':'0','PYTHONPATH':str(ROOT)}
  if a.candidate_mode=='skip-presync': env['NV_COPYOUT_SKIP_PRESYNC']='1' if arm=='candidate' else '0'
  if a.candidate_mode=='host-mirror': env['NV_ARGMAX_HOST_MIRROR']='1' if arm=='candidate' else '0'
  cmd=['timeout','1800','flock','-w','600',LOCK,str(PYTHON),str(pathlib.Path(__file__).resolve()),'--arm',arm,'--candidate-mode',a.candidate_mode,'--model',a.model,'--depth',str(a.depth),'--count',str(a.count),'--reps',str(a.reps),'--max-context',str(a.max_context),'--out',str(out)]
  p=subprocess.run(cmd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  if p.returncode: raise RuntimeError(f'{arm} failed rc={p.returncode}: {p.stderr[-5000:]}')
  return json.loads(out.read_text())

def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--arm',choices=('control_a','candidate','control_c')); ap.add_argument('--model',default=MODEL)
  ap.add_argument('--depth',type=int,default=512); ap.add_argument('--count',type=int,default=24); ap.add_argument('--reps',type=int,default=7)
  ap.add_argument('--candidate-mode',choices=('skip-presync','skip-item','host-mirror'),default='skip-presync')
  ap.add_argument('--max-context',type=int,default=768); ap.add_argument('--out',type=pathlib.Path,required=True); a=ap.parse_args()
  if a.arm: result=run_arm(a.arm,a.model,a.depth,a.count,a.reps,a.max_context,a.out,a.candidate_mode)
  else:
    root=pathlib.Path(str(a.out).removesuffix('.json')); root.mkdir(parents=True,exist_ok=True)
    arms=[child(a,x,root) for x in ('control_a','candidate','control_c')]; wall=lambda x:x['settled']['median_ms_per_token']*1000
    control=statistics.median((wall(arms[0]),wall(arms[2]))); cand=wall(arms[1])
    result={'schema':'tinygrad.nv_token_readback_wall.v1','mode':'reverse-bracket','candidate_mode':a.candidate_mode,'all_token_hashes_equal':len({x['settled']['token_stream_hash'] for x in arms})==1,'walls_us_per_token':{'control_a':wall(arms[0]),'candidate':cand,'control_c':wall(arms[2]),'control_midpoint':control,'candidate_delta':cand-control},
            'tokens_per_second':{'control':1e6/control,'candidate':1e6/cand,'delta':1e6/cand-1e6/control},'arms':arms}; a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
  print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__': main()
