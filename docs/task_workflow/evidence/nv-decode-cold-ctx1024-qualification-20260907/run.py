#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, statistics, subprocess, time

from tinygrad import Device
from tinygrad.engine.jit import observe_graph_admissions
from tinygrad.helpers import Context, GlobalCounters
from tinygrad.llm.generate import load_model_and_tokenizer
from extra.llm_research.decode.decode_harness import DEFAULT_MODEL
from extra.llm_research.decode.decode_runtime_overhead import (
  _captured_program_count, _captured_program_evidence, _decode_jits, _make_prompt,
  _model_identity, _nv_gpu_state, _prefill, _route, _token_evidence, _used_decode_jits)

OUT=pathlib.Path('/tmp/nv_decode_cold_ctx1024.json')
CENSUS=pathlib.Path('/tmp/nv_decode_cold_ctx1024_census.json')
DEPTH=1024; MAX_CONTEXT=1536; STEADY=6

def memory() -> dict:
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip()
  used,total=(int(x.strip()) for x in raw.split(','))
  return {'nvidia_used_mib':used,'nvidia_total_mib':total,
          'tinygrad_live_bytes':int(GlobalCounters.mem_used_per_device[Device.DEFAULT])}

def main() -> None:
  process_start=time.perf_counter(); phases={'process_start':memory()}
  started=time.perf_counter(); model,tok=load_model_and_tokenizer(DEFAULT_MODEL,MAX_CONTEXT,seed=20260617)
  phases['model_construct']={'elapsed_s':time.perf_counter()-started,**memory()}
  base=(tok.prefix() if hasattr(tok,'prefix') else [])+tok.encode('the quick brown fox jumps. '*800)
  prompt=_make_prompt(base,DEPTH); model.reset_generation_state()
  # No request-scoped prewarm: the next token must exercise the first ordinary decode call.
  started=time.perf_counter(); gen,prelude=_prefill(model,prompt,32)
  phases['prefill_and_prelude']={'elapsed_s':time.perf_counter()-started,**memory()}
  before={name:jit.cnt for name,jit in _decode_jits(model).items()}
  with Context(TRACEMETA=1), observe_graph_admissions() as admissions:
    Device[Device.DEFAULT].synchronize(); started=time.perf_counter(); first=int(next(gen)); Device[Device.DEFAULT].synchronize()
    first_s=time.perf_counter()-started; phases['first_decode']={'elapsed_s':first_s,**memory()}
    settled=[]; tokens=[]
    for _ in range(STEADY):
      Device[Device.DEFAULT].synchronize(); started=time.perf_counter(); tokens.append(int(next(gen))); Device[Device.DEFAULT].synchronize()
      settled.append(time.perf_counter()-started)
  gen.close()
  selected=_used_decode_jits(model,before)
  programs={name:_captured_program_evidence(jit) for name,jit in selected.items()}
  sources={}
  for rows in programs.values():
    for row in rows:
      src=row.pop('source_text')
      if src is not None: sources[row['source_sha256']]=src
  census=admissions.to_dict(); census['capture']={'selected_jits':list(selected),'programs_by_jit':{n:_captured_program_count(j) for n,j in selected.items()},
    'program_evidence_by_jit':programs,'source_text_by_sha256':sources,'captured':all(j.captured is not None for j in selected.values())}
  CENSUS.write_text(json.dumps(census,indent=2,sort_keys=True)+'\n')
  from tinygrad import UOp
  route_sp=UOp.variable('cold_reported_start_pos',0,MAX_CONTEXT-1).bind(DEPTH)
  artifact={'schema':'tinygrad.decode.cold_first_request.v1','model':_model_identity(DEFAULT_MODEL),'depth':DEPTH,'max_context':MAX_CONTEXT,
    'route':'flash' if _route(model,route_sp,1) else 'sdpa','ordinary_environment':True,'request_scoped_prewarm':False,'phases':phases,
    'first_decode_ms':first_s*1e3,'steady_per_token_ms':[x*1e3 for x in settled],
    'steady_median_ms':statistics.median(settled)*1e3,'steady_min_ms':min(settled)*1e3,
    'prelude_token':_token_evidence([prelude],include_token_ids=True),'first_decode_token':_token_evidence([first],include_token_ids=True),
    'steady_tokens':_token_evidence(tokens,include_token_ids=True),'selected_jits':list(selected),
    'programs_per_token':{n:_captured_program_count(j) for n,j in selected.items()},'gpu_after':_nv_gpu_state(),
    'process_elapsed_s':time.perf_counter()-process_start}
  OUT.write_text(json.dumps(artifact,indent=2,sort_keys=True)+'\n'); print(json.dumps({'out':str(OUT),'first_decode_ms':first_s*1e3,'steady_median_ms':artifact['steady_median_ms']}))

if __name__=='__main__': main()
