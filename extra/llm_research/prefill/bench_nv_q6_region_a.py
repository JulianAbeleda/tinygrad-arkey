#!/usr/bin/env python3
"""Region A probe: canonical Q6/Q8 decode and shared publication only."""
from __future__ import annotations
import argparse, hashlib, json, pathlib, re, statistics
import numpy as np
from tinygrad import Device, Tensor
from tinygrad.runtime.ops_nv import NVProgram
from extra.llm_research.layout import GGML_Q6_K, packed_u16_slice, read_metadata
from extra.llm_research.prefill.nv_compiler_q6k_imma_gate import _buf, _record, _run, _sass
from extra.llm_research.prefill.nv_compiler_q6k_streamk_transform import transform_compiler_q6k_wide_to_streamk

M,N,K=512,4096,12288

def region_a_source(source: str, *, bulk_readback: bool = True, variant: str = 'full') -> str:
  # The first barrier terminates the producer publication prefix in the direct
  # compiler kernel. Checksum shared bytes after it to force materialization.
  start=source.find('  (*(buf0+0)) = 0.0f;')
  barrier0=source.find('    __syncthreads();', start)
  barrier=source.find('    __syncthreads();', barrier0+1)
  if start < 0 or barrier < 0: raise ValueError('producer boundary not found')
  prefix=source[:barrier+len('    __syncthreads();')]
  signature=re.search(r'extern "C" __global__ void __launch_bounds__\(256\) (\w+)\(',prefix)
  if signature is None: raise ValueError('kernel signature not found')
  prefix=prefix.replace('float* data0_2097152, unsigned int* data1_1966080, unsigned short* data2_20643840',
                        'float* data0_2097152, unsigned int* data1_1966080, unsigned short* data2_20643840',1)
  # Direct byte readback prevents cancellation and makes the published shared
  # payload observable without an accumulator or MMA/FP32 epilogue.
  body = ('''\n    for (int z=lidx0+32*lidx1+64*lidx2; z<20480; z+=256)\n      data0_2097152[(blockIdx.x+32*blockIdx.y)*20480 + z] = (float)((unsigned char)buf1[z]);\n  }\n  }\n''' if bulk_readback else '''\n    if (lidx0==0 && lidx1==0 && lidx2==0)\n      data0_2097152[blockIdx.x+32*blockIdx.y] = (float)((unsigned char)buf1[64]);\n  }\n  }\n''')
  if variant == 'q6_decode_no_q8':
    prefix = re.sub(r'(unsigned int val\d+ = )\(\*\([^;]+\);', r'\g<1>0;', prefix)
  elif variant == 'q8_only':
    prefix = re.sub(r'(unsigned short val\d+ = )\(\*\([^;]+\);', r'\g<1>0;', prefix)
  elif variant != 'full': raise ValueError(f'unknown variant: {variant}')
  return prefix+body

def region_a_loads_source(source: str) -> str:
  """Retain producer index arithmetic and global Q6/Q8 loads only."""
  start=source.find('  for (int Ridx0 = 0;')
  if start < 0: raise ValueError('producer loop not found')
  end=source.find('    __syncthreads();', start)
  if end < 0: raise ValueError('producer barrier not found')
  loop=source[start:end]
  # Keep address declarations and global load declarations; all decode and
  # shared publication are deliberately excluded from this causal arm.
  kept=[]
  for line in loop.splitlines():
    stripped=line.strip()
    if stripped.startswith('int alu') or stripped.startswith('unsigned short val') or stripped.startswith('unsigned int val'):
      kept.append(line)
  if not kept: raise ValueError('no global loads retained')
  vals=[re.search(r'\b(val\d+)\b', x).group(1) for x in kept if ' val' in x]
  sink='    unsigned int load_sink = 0;\n'
  sink += ''.join(f'    load_sink ^= (unsigned int){v};\n' for v in vals)
  sink += '    data0_2097152[blockIdx.x+32*blockIdx.y] = (float)load_sink;\n'
  body='  for (int Ridx0 = 0; Ridx0 < 192; Ridx0++) {\n'+'\n'.join(kept)+'\n'+sink+'  }\n  }\n'
  sig_end=source.find('{', start)
  prefix=source[:start]
  return prefix+body

def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument('--model',default='/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf')
  ap.add_argument('--loads-only', action='store_true')
  ap.add_argument('--variant', choices=('full','q6_decode_no_q8','q8_only'), default='full')
  ap.add_argument('--rounds',type=int,default=9); ap.add_argument('--out',required=True); ap.add_argument('--artifacts',required=True)
  a=ap.parse_args();
  if a.rounds<9: raise ValueError('R9 required')
  art=pathlib.Path(a.artifacts); art.mkdir(parents=True,exist_ok=True)
  model=pathlib.Path(a.model); meta=read_metadata(model); info=next(i for i in meta.infos if i.name=='blk.0.ffn_down.weight')
  if info.typ != GGML_Q6_K: raise RuntimeError('fixture is not Q6_K')
  halfs=packed_u16_slice(model,meta,info,device='NV').contiguous().realize(); record=Tensor(_record(M,K)[0],device='NV').contiguous().realize()
  direct=_run('region_a_base',M,N,K,halfs,record,a.rounds,art,(128,128,2,4,256))
  src=region_a_loads_source((art/'region_a_base.cu').read_text()) if a.loads_only else region_a_source((art/'region_a_base.cu').read_text(),variant=a.variant); (art/'region_a.cu').write_text(src)
  binary=Device['NV'].compiler.compile(src); sass=_sass(binary,art/'region_a')
  name=re.search(r'__launch_bounds__\(256\) (\w+)\(',src).group(1); p=NVProgram(Device['NV'],name,binary)
  out=Tensor.full((128 if a.loads_only else 128*20480,),float('nan'),device='NV').contiguous().realize(); buf=out.uop.buffer.get_buf('NV')
  samples=[p(buf,record.uop.buffer.get_buf('NV'),halfs.uop.buffer.get_buf('NV'),global_size=(32,4,1),local_size=(32,2,4),wait=True)*1e6 for _ in range(a.rounds)]
  got=out.numpy(); checksum_slots=np.arange(128,dtype=np.int64) if a.loads_only else np.arange(128*20480,dtype=np.int64)
  timing_src=region_a_loads_source((art/'region_a_base.cu').read_text()) if a.loads_only else region_a_source((art/'region_a_base.cu').read_text(),bulk_readback=False,variant=a.variant); (art/'region_a_timing.cu').write_text(timing_src)
  timing_binary=Device['NV'].compiler.compile(timing_src); timing_sass=_sass(timing_binary,art/'region_a_timing')
  timing_name=re.search(r'__launch_bounds__\(256\) (\w+)\(',timing_src).group(1); timing_p=NVProgram(Device['NV'],timing_name,timing_binary)
  timing_out=Tensor.full((128,),float('nan'),device='NV').contiguous().realize(); timing_buf=timing_out.uop.buffer.get_buf('NV')
  timing_samples=[timing_p(timing_buf,record.uop.buffer.get_buf('NV'),halfs.uop.buffer.get_buf('NV'),global_size=(32,4,1),local_size=(32,2,4),wait=True)*1e6 for _ in range(a.rounds)]
  base_src=(art/'region_a_base.cu').read_text()
  census={'full_ldg_q6':base_src.count('data2_20643840+'),'full_ldg_q8':base_src.count('data1_1966080+'),'full_buf1_stores':base_src.count('*(buf1+'),'producer_ldg_q6':src.count('data2_20643840+'),'producer_ldg_q8':src.count('data1_1966080+'),'producer_buf1_stores':src.count('*(buf1+')}
  result={'schema':'tinygrad.nv_q6_region_a.v1','shape':{'M':M,'N':N,'K':K},'population':{'ctas':128,'k256_blocks_per_cta':48},'structural_census':census,
    'correctness':{'readback_finite':bool(np.isfinite(got.ravel()[checksum_slots]).all()),'decoded_nonzero':int(np.count_nonzero(got.ravel()[checksum_slots])),'decoded_sample':got.ravel()[checksum_slots[:8]].tolist()},
    'timing':{'diagnostic_bulk_readback':{'samples_us':samples,'min_us':min(samples),'median_us':statistics.median(samples),'max_us':max(samples)},'live_sink':{'samples_us':timing_samples,'min_us':min(timing_samples),'median_us':statistics.median(timing_samples),'max_us':max(timing_samples),'sample_finite':bool(np.isfinite(timing_out.numpy()).all())}},
    'compiler':{'sha256':hashlib.sha256(binary).hexdigest(),'sass':sass,'timing_sass':timing_sass,'no_imma':sass['imma']==0,'no_local_spills':sass['local_load']==sass['local_store']==0},'full_route':direct}
  result['passed']=bool(result['correctness']['readback_finite'] and result['correctness']['decoded_nonzero'] and result['compiler']['no_imma'])
  pathlib.Path(a.out).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,sort_keys=True)); return 0 if result['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
