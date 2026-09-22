#!/usr/bin/env python3
"""Reverse production wall bracket for relaxed internal dependent-QMD membars."""
from __future__ import annotations
import argparse,json,os,pathlib,statistics,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
MODEL='/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf'; LOCK='/tmp/tinygrad-gpu.lock'; PYTHON=ROOT/'.venv/bin/python'

def child(arm,model,depth,count,reps,max_context,out,policy):
 env={**os.environ,'DEV':'NV','PROFILE':'0','PYTHONPATH':str(ROOT)}
 env['NV_RELAX_INTERNAL_QMD_MEMBAR']='1' if policy=='invalidate' or arm=='candidate' else '0'
 env['NV_RELAX_DEPENDENT_QMD_INVALIDATE']='1' if policy=='invalidate' and arm=='candidate' else '0'
 cmd=['timeout','1800','flock','-w','600',LOCK,str(PYTHON),str(pathlib.Path(__file__).resolve()),'--arm',arm,'--policy',policy,'--model',model,'--depth',str(depth),'--count',str(count),'--reps',str(reps),'--max-context',str(max_context),'--out',str(out)]
 p=subprocess.run(cmd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if p.returncode: raise RuntimeError(f'{arm} failed rc={p.returncode}: {p.stderr[-6000:]}')
 return json.loads(out.read_text())

def run_arm(arm,model_path,depth,count,reps,max_context,out):
 from tinygrad import Device
 from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load,_prompt
 from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows
 model=_load(model_path,max_context); gen=model.generate(_prompt(model_path,depth),chunk_size=32,temperature=0.0)
 try: settled=_settled_continuous_windows(gen,Device['NV'],count,reps)
 finally: gen.close()
 result={'schema':'tinygrad.nv_qmd_membar_wall.v1','arm':arm,'model':model_path,'depth':depth,'count':count,'reps':reps,'max_context':max_context,'relax_internal_membar':os.getenv('NV_RELAX_INTERNAL_QMD_MEMBAR','1')!='0','relax_dependent_qmd_invalidate':os.getenv('NV_RELAX_DEPENDENT_QMD_INVALIDATE','0')!='0','settled':settled}; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); return result

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--arm',choices=('control_a','candidate','control_c')); ap.add_argument('--policy',choices=('membar','invalidate'),default='membar'); ap.add_argument('--model',default=MODEL); ap.add_argument('--depth',type=int,default=512); ap.add_argument('--count',type=int,default=24); ap.add_argument('--reps',type=int,default=9); ap.add_argument('--max-context',type=int,default=768); ap.add_argument('--out',type=pathlib.Path,required=True); a=ap.parse_args()
 if a.depth+7+a.count*a.reps>a.max_context: raise ValueError('decode windows exceed context')
 if a.arm: result=run_arm(a.arm,a.model,a.depth,a.count,a.reps,a.max_context,a.out)
 else:
  root=pathlib.Path(str(a.out).removesuffix('.json')); root.mkdir(parents=True,exist_ok=True); arms=[child(x,a.model,a.depth,a.count,a.reps,a.max_context,root/f'{x}.json',a.policy) for x in ('control_a','candidate','control_c')]
  wall=lambda x:float(x['settled']['median_ms_per_token'])*1000; control=statistics.median((wall(arms[0]),wall(arms[2]))); cand=wall(arms[1]); hashes={x['settled']['token_stream_hash'] for x in arms}
  result={'schema':'tinygrad.nv_qmd_membar_wall.v1','mode':'reverse-bracket-unprofiled','policy':a.policy,'all_token_hashes_equal':len(hashes)==1,'token_stream_hashes':sorted(hashes),'walls_us_per_token':{'control_a':wall(arms[0]),'candidate':cand,'control_c':wall(arms[2]),'control_midpoint':control,'candidate_delta':cand-control},'tokens_per_second':{'control_midpoint':1e6/control,'candidate':1e6/cand,'candidate_delta':1e6/cand-1e6/control},'arms':arms}; a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__': main()
