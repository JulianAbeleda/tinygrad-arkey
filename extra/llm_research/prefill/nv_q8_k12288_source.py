"""Fail-closed source specialization for the Q8 FP16 producer."""
from extra.llm_research.prefill.nv_q8_compact_producer_gate import SRC_FP16

def source_k12288():
  src=SRC_FP16.replace('base=row*4096+i','base=row*12288+i').replace('row*128+seg*16+t/8','row*384+seg*16+t/8')
  if 'base=row*4096+i' in src or 'row*128+seg*16+t/8' in src: raise RuntimeError('incomplete K specialization')
  return src.replace('q8_compact_fp16','q8_compact_record_fp16_k12288')

def source_k12288_record():
  """The qualified K12288 math with one packed-record output buffer."""
  src=source_k12288()
  old=("void q8_compact_record_fp16_k12288(const half* __restrict__ x, signed char* __restrict__ q,\n"
       " float* __restrict__ scales,float* __restrict__ sums) {")
  new=("void q8_compact_record_fp16_k12288(const half* __restrict__ x, unsigned int* __restrict__ record) {\n"
       " signed char* __restrict__ q=(signed char*)record;\n"
       " float* __restrict__ scales=(float*)(q+6291456);\n"
       " float* __restrict__ sums=scales+196608;")
  if src.count(old)!=1: raise RuntimeError('K12288 producer source ABI changed')
  out=src.replace(old,new)
  if 'signed char* __restrict__ q,' in out or 'float* __restrict__ scales,float* __restrict__ sums' in out:
    raise RuntimeError('K12288 packed-record adaptation incomplete')
  return out

K=12288; SEGMENTS=24; GROUPS_PER_ROW=384; GRID=(512,SEGMENTS,1); BLOCK=(128,1,1)
