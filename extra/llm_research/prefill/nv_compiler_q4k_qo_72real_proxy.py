#!/usr/bin/env python3
"""Live, paired Q/O lifecycle gate over canonical GGUF weights, one role per process."""
import argparse, collections, hashlib, json, pathlib, statistics, time
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.uop.ops import Ops
from extra.llm_research.layout import read_metadata, packed_u32_slice
from extra.llm_research.prefill.nv_compiler_q4k_qo_binding import binding_for


def program_key(p):
  binaries = [u.arg for u in p.src if u.op is Ops.BINARY]
  if len(binaries) != 1: raise ValueError('expected one retained binary')
  return (p.arg.name, hashlib.sha256(binaries[0]).hexdigest(), tuple(p.arg.global_size), tuple(p.arg.local_size),
          tuple(p.arg.globals), tuple(p.arg.outs), tuple(p.arg.ins), tuple(v.arg[1] for v in p.arg.vars), tuple(p.arg.aux))


def census(jit, expected):
  programs = [u.src[0] for u in jit.captured.linear.toposort() if u.op is Ops.CALL and u.src and u.src[0].op is Ops.PROGRAM]
  actual = collections.Counter(program_key(p) for p in programs)
  wanted = collections.Counter()
  for p, count in expected: wanted[program_key(p)] += count
  return {'exact': actual == wanted, 'calls': len(programs),
          'programs': [{'identity': list(key), 'count': count} for key,count in actual.items()]}


def main():
  ap=argparse.ArgumentParser()
  ap.add_argument('--model',default='/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf')
  ap.add_argument('--role',choices=('q','o'),required=True)
  ap.add_argument('--rounds',type=int,default=31)
  ap.add_argument('--variant',choices=('wide','streamk'),default='wide')
  ap.add_argument('--out',required=True)
  a=ap.parse_args()
  if a.rounds < 1: raise ValueError('rounds must be positive')
  role='attn_q' if a.role=='q' else 'attn_output'
  path=pathlib.Path(a.model); md=read_metadata(path)
  infos=[i for i in md.infos if i.name.endswith('.'+role+'.weight')]
  if len(infos)!=36 or any(i.typ!=12 or tuple(reversed(i.dims))!=(4096,4096) for i in infos):
    raise ValueError('expected exactly 36 Q4_K weights with shape (4096,4096)')
  weights=[packed_u32_slice(path,md,i,device='NV').contiguous().realize() for i in infos]
  weight_ids=[w.uop.buf_uop for w in weights]
  candidate=binding_for('NV',variant=a.variant).new_capture(); candidate.prepare(36)
  if a.role=='q':
    from extra.llm_research.prefill.nv_qkv_packed_pp512_binding import binding_for as llama_binding
    oracle=llama_binding('NV').new_capture()
    oracle_programs=[oracle.asset.ds4,oracle.asset.q4_q_main,oracle.asset.q4_q_fix]
  else:
    from extra.llm_research.prefill.nv_llama_packed_q4k_o_pp512_binding import binding_for as llama_binding
    oracle=llama_binding('NV').new_capture()
    oracle_programs=[oracle.asset.producer,oracle.asset.main,oracle.asset.fixup]
  rng=np.random.default_rng(3)
  x_np=rng.standard_normal((512,4096)).astype(np.float16)
  inputs=[Tensor(v,device='NV').realize() for v in (x_np,(x_np*.7).astype(np.float16))]
  @TinyJit
  def generated(x):
    candidate.begin_trace()
    return tuple(candidate.project(x,w,model_family='qwen3_8b',role=role) for w in weights)
  @TinyJit
  def llama(x):
    oracle.begin_trace()
    return tuple(oracle.project_q(x,w) if a.role=='q' else
                 oracle.project(x,w,model_family='qwen3_8b',role=role) for w in weights)
  def run(jit,x):
    outputs=jit(x); Tensor.realize(*outputs); Device['NV'].synchronize(); return outputs
  def snapshot(outputs): return np.stack([x.numpy().copy() for x in outputs])
  for _ in range(3): run(generated,inputs[0]); run(llama,inputs[0])
  expected=[(candidate.producer,36),(candidate.q_program,36)]
  if a.variant=='streamk': expected.append((candidate.fixup_program,36))
  cc=census(generated,expected)
  lc=census(llama,[(p,36) for p in oracle_programs])
  # Correctness snapshots are deliberately outside the timing window.
  pairs=[]; first_outputs=[]
  for x in inputs:
    got=snapshot(run(generated,x)); ref=snapshot(run(llama,x))
    pairs.append({'finite':bool(np.isfinite(got).all() and np.isfinite(ref).all()),
                  'allclose':bool(np.allclose(got,ref,rtol=.02,atol=.5)),
                  'max_abs':float(np.max(np.abs(got-ref)))})
    first_outputs.append((got[0].copy(),ref[0].copy()))
  distinct=all(not np.array_equal(first_outputs[0][i],first_outputs[1][i]) for i in (0,1))
  stable=all(w.uop.buf_uop is old for w,old in zip(weights,weight_ids))
  samples={'candidate':[],'llama':[]}; orders=[]
  for iteration in range(a.rounds):
    order=('candidate','llama') if iteration%2==0 else ('llama','candidate'); orders.append(order)
    for name in order:
      Device['NV'].synchronize(); start=time.perf_counter_ns()
      run(generated if name=='candidate' else llama,inputs[0])
      samples[name].append((time.perf_counter_ns()-start)/1e6)
  passed=all(p['finite'] and p['allclose'] for p in pairs) and distinct and stable and cc['exact'] and lc['exact']
  result={'schema':'tinygrad.nv.qo.live.v2','status':('PASS_DIRECT' if a.rounds>=31 else 'PASS_SMOKE') if passed else 'FAIL',
          'role':role,'variant':a.variant,'scope':'isolated projection lifecycle; QKV shared producer and O residual are excluded',
          'weights':[{'name':i.name,'ggml_type':i.typ,'shape':list(reversed(i.dims))} for i in infos],
          'fixture':{'kind':'synthetic normal','seed':3,'sha256':hashlib.sha256(x_np.tobytes()).hexdigest()},
          'canonical_buffers_stable':stable,'candidate_census':cc,'llama_census':lc,'correctness':pairs,'distinct':distinct,
          'timing_ms':{'samples':samples,'orders':orders,'rounds':a.rounds,
                       'candidate_median':statistics.median(samples['candidate']),'llama_median':statistics.median(samples['llama'])}}
  out=pathlib.Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps({k:result[k] for k in ('status','role','correctness','distinct')}),flush=True)
  print(json.dumps({'candidate_ms':result['timing_ms']['candidate_median'],'llama_ms':result['timing_ms']['llama_median'],
                    'candidate_census':cc['exact'],'llama_census':lc['exact']}),flush=True)
  if not passed: raise SystemExit(1)

if __name__=='__main__': main()
