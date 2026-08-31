from __future__ import annotations

import re

OWNERS, OUTPUT_TILES, K_BLOCKS, TILES_N = 170, 384, 64, 96
WORK_UNITS, BOUNDARY_QUANTUM = OUTPUT_TILES*K_BLOCKS, 8
TILE_ELEMENTS, PARTIAL_SLOTS = 128*128, 2*OWNERS

def _partial_store_block(direct_store_block:str) -> str:
  block=direct_store_block
  block=re.sub(r"int alu242 = .*?;", "int alu242 = ((alu5<<1)+(lidx2<<5)+(alu2*128)+(lidx1*8192));", block, count=1)
  block=block.replace("data0_6291456+", "partials+(slot*16384)+")
  for value in sorted({int(x) for x in re.findall(r"alu242\+(\d+)",block)},reverse=True):
    row,column=divmod(value,12288)
    if column >= 128: raise ValueError(f"global output offset {value} escapes its 128-column tile")
    block=block.replace(f"alu242+{value}",f"alu242+{row*128+column}")
  return block

def transform_compiler_q4k_to_streamk(source:str, *, unroll:int|None=None) -> str:
  """Wrap the compiler-owned Q4_K/Q8 tile body in llama-compatible Stream-K ownership.

  The signed-IMMA math and packed input addressing remain compiler emitted.  Only
  launch ownership, the outer K64 range, and terminal output destination change.
  """
  signature=re.search(r'(extern "C" __global__ void __launch_bounds__\(256\) \w+\()'
                      r'(float\* data0_6291456, unsigned int\* data1_655360, unsigned int\* data2_7077888)(\) \{)',source)
  if signature is None: raise ValueError("compiler Q4 kernel signature not found")
  exported='extern "C" __global__ void __launch_bounds__(256) q4k_imma_stream('
  source=source[:signature.start()]+exported+(
    "float* data0_6291456, float* partials, int* partial_ids, "
    "unsigned int* data2_7077888, unsigned int* data1_655360")+signature.group(3)+source[signature.end():]
  source=source.replace("  int gidx0 = blockIdx.x; /* 96 */\n  int gidx1 = blockIdx.y; /* 4 */\n",
                        "  int owner = blockIdx.x; /* 170 persistent owners */\n",1)
  body_start=source.find("  (*(buf0+0)) = 0.0f;")
  store_start=source.find("  int alu242 = ",body_start)
  if body_start < 0 or store_start < 0: raise ValueError("compiler Q4 body/store boundary not found")
  function_end=source.rfind("}")
  if function_end < store_start: raise ValueError("compiler Q4 function terminator not found")
  math=source[body_start:store_start]
  if unroll is not None:
    if unroll not in (1,2,4,8): raise ValueError("unsupported Stream-K outer-K unroll")
    loop="for (int Ridx0 = 0; Ridx0 < 64; Ridx0++) {"
    if loop not in math: raise ValueError("compiler outer-K loop not found")
    math=math.replace(loop,f"#pragma unroll {unroll}\n  {loop}",1)
  math=math.replace("for (int Ridx0 = 0; Ridx0 < 64; Ridx0++) {",
                    "for (int Ridx0 = k_begin; Ridx0 < k_end; Ridx0++) {",1)
  direct=source[store_start:function_end]
  partial=_partial_store_block(direct)
  prefix=source[:body_start]
  owner_loop=f"""  int owner_start = ((owner*{WORK_UNITS}/{OWNERS})/{BOUNDARY_QUANTUM})*{BOUNDARY_QUANTUM};
  if (threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0) {{ partial_ids[owner*2]=-1; partial_ids[owner*2+1]=-1; }}
  int owner_stop = (owner == {OWNERS-1}) ? {WORK_UNITS} : ((((owner+1)*{WORK_UNITS}/{OWNERS})/{BOUNDARY_QUANTUM})*{BOUNDARY_QUANTUM});
  int first_tile = owner_start/{K_BLOCKS}, last_tile = (owner_stop-1)/{K_BLOCKS};
  for (int tile=first_tile; tile<=last_tile; tile++) {{
    int segment_start=max(owner_start,tile*{K_BLOCKS}), segment_stop=min(owner_stop,(tile+1)*{K_BLOCKS});
    int k_begin=segment_start-tile*{K_BLOCKS}, k_end=segment_stop-tile*{K_BLOCKS};
    int gidx0=tile%{TILES_N}, gidx1=tile/{TILES_N};
    bool direct=(k_begin==0 && k_end=={K_BLOCKS});
    bool owner_tail=(segment_stop==owner_stop && k_end!={K_BLOCKS});
    bool owner_has_head=((owner_start%{K_BLOCKS})!=0);
    int slot=owner*2+((owner_tail&&owner_has_head)?1:0);
"""
  stores=("    if (direct) {\n"+direct+"    } else {\n"
          "      if (threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0) partial_ids[slot]=tile;\n"+
          partial+"    }\n")
  return prefix+owner_loop+math+stores+"  }\n}\n"

def active_fixup_source() -> str:
  return r'''extern "C" __global__ void q4k_imma_fixup_active(float *out,const float *partials,const int *map,const int *active,int M,int N) {
    int tile=active[blockIdx.x],s0=map[2*tile],s1=map[2*tile+1],nb=(tile%(N/128))*128,mb=(tile/(N/128))*128;
    for (int z=threadIdx.x;z<16384;z+=256) { int r=z/128,c=z%128;
      out[(mb+r)*N+nb+c]=partials[s0*16384+z]+(s1>=0?partials[s1*16384+z]:0); }
  }'''

__all__=["BOUNDARY_QUANTUM","K_BLOCKS","OWNERS","OUTPUT_TILES","PARTIAL_SLOTS","TILE_ELEMENTS",
         "TILES_N","WORK_UNITS","active_fixup_source","transform_compiler_q4k_to_streamk"]
