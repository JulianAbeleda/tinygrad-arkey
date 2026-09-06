from __future__ import annotations

import re

OWNERS, OUTPUT_TILES, K_BLOCKS, TILES_N = 170, 384, 64, 96
WORK_UNITS, BOUNDARY_QUANTUM = OUTPUT_TILES*K_BLOCKS, 8
TILE_ELEMENTS, PARTIAL_SLOTS = 128*128, 2*OWNERS

def _partial_store_block(direct_store_block:str, *, output_stride:int=12288, output_arg:str="data0_6291456") -> str:
  block=direct_store_block
  block=re.sub(r"int alu242 = .*?;", "int alu242 = ((alu5<<1)+(lidx2<<5)+(alu2*128)+(lidx1*8192));", block, count=1)
  block=block.replace(output_arg+"+", "partials+(slot*16384)+")
  for value in sorted({int(x) for x in re.findall(r"alu242\+(\d+)",block)},reverse=True):
    row,column=divmod(value,output_stride)
    if column >= 128: raise ValueError(f"global output offset {value} escapes its 128-column tile")
    block=block.replace(f"alu242+{value}",f"alu242+{row*128+column}")
  return block

def transform_compiler_q4k_to_streamk(source:str, *, unroll:int|None=None, tiles_n:int=96,
                                     k_blocks:int=64, output_stride:int=12288,
                                     kernel_name:str="q4k_imma_stream") -> str:
  """Wrap the compiler-owned Q4_K/Q8 tile body in llama-compatible Stream-K ownership.

  The signed-IMMA math and packed input addressing remain compiler emitted.  Only
  launch ownership, the outer K64 range, and terminal output destination change.
  """
  if any(x <= 0 for x in (tiles_n,k_blocks,output_stride)): raise ValueError("invalid Stream-K source geometry")
  signature=re.search(r'(extern "C" __global__ void __launch_bounds__\(256\) \w+\()'
                      r'(float\* (data0_\d+), unsigned int\* (data1_\d+), unsigned int\* (data2_\d+))(\) \{)',source)
  if signature is None: raise ValueError("compiler Q4 kernel signature not found")
  if f"Ridx0 < {k_blocks}" not in source: raise ValueError("source K loop does not match requested Stream-K geometry")
  out_name=kernel_name
  exported=f'extern "C" __global__ void __launch_bounds__(256) {out_name}('
  out_arg,w_arg,rec_arg=signature.group(3),signature.group(4),signature.group(5)
  source=source[:signature.start()]+exported+(
    f"float* {out_arg}, float* partials, int* partial_ids, "
    f"unsigned int* {rec_arg}, unsigned int* {w_arg}) {{")+source[signature.end():]
  source=source.replace(f"  int gidx0 = blockIdx.x; /* {tiles_n} */\n  int gidx1 = blockIdx.y; /* 4 */\n",
                        "  int owner = blockIdx.x; /* 170 persistent owners */\n",1)
  body_start=source.find("  (*(buf0+0)) = 0.0f;")
  store_start=source.find("  int alu242 = ",body_start)
  if body_start < 0 or store_start < 0: raise ValueError("compiler Q4 body/store boundary not found")
  function_end=source.rfind("}")
  if function_end < store_start: raise ValueError("compiler Q4 function terminator not found")
  math=source[body_start:store_start]
  loop=f"for (int Ridx0 = 0; Ridx0 < {k_blocks}; Ridx0++) {{"
  if loop not in math: raise ValueError("compiler outer-K loop not found")
  if unroll is not None:
    if unroll not in (1,2,4,8,16,32): raise ValueError("unsupported Stream-K outer-K unroll")
    math=math.replace(loop,f"#pragma unroll {unroll}\n  {loop}",1)
  math=math.replace(loop,"for (int Ridx0 = k_begin; Ridx0 < k_end; Ridx0++) {",1)
  direct=source[store_start:function_end]
  partial=_partial_store_block(direct,output_stride=output_stride,output_arg=signature.group(3))
  prefix=source[:body_start]
  work_units=tiles_n*(4)*k_blocks
  owner_loop=f"""  int owner_start = ((owner*{work_units}/{OWNERS})/{BOUNDARY_QUANTUM})*{BOUNDARY_QUANTUM};
  if (threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0) {{ partial_ids[owner*2]=-1; partial_ids[owner*2+1]=-1; }}
  int owner_stop = (owner == {OWNERS-1}) ? {work_units} : ((((owner+1)*{work_units}/{OWNERS})/{BOUNDARY_QUANTUM})*{BOUNDARY_QUANTUM});
  int first_tile = owner_start/{k_blocks}, last_tile = (owner_stop-1)/{k_blocks};
  for (int tile=first_tile; tile<=last_tile; tile++) {{
    int segment_start=max(owner_start,tile*{k_blocks}), segment_stop=min(owner_stop,(tile+1)*{k_blocks});
    int k_begin=segment_start-tile*{k_blocks}, k_end=segment_stop-tile*{k_blocks};
    int gidx0=tile%{tiles_n}, gidx1=tile/{tiles_n};
    bool direct=(k_begin==0 && k_end=={k_blocks});
    bool owner_tail=(segment_stop==owner_stop && k_end!={k_blocks});
    bool owner_has_head=((owner_start%{k_blocks})!=0);
    int slot=owner*2+((owner_tail&&owner_has_head)?1:0);
"""
  stores=("    if (direct) {\n"+direct+"    } else {\n"
          "      if (threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0) partial_ids[slot]=tile;\n"+
          partial+"    }\n")
  return prefix+owner_loop+math+stores+"  }\n}\n"

def active_fixup_source(*, max_contributors:int=2, sliced:bool=False) -> str:
  if max_contributors < 2: raise ValueError("fixup requires at least two contributors")
  decl=','.join(f"s{i}=map[{max_contributors}*tile+{i}]" for i in range(max_contributors))
  adds=''.join(f"+(s{i}>=0?partials[s{i}*16384+z]:0)" for i in range(1,max_contributors))
  zdecl="int tile=active[blockIdx.x],z=threadIdx.x;" if not sliced else "int tile=active[blockIdx.x],z=blockIdx.y*4096+threadIdx.x;"
  loop="z<16384;z+=256" if not sliced else "z<((blockIdx.y+1)*4096);z+=128"
  return f'''extern "C" __global__ void q4k_imma_fixup_active(float *out,const float *partials,const int *map,const int *active,int M,int N) {{
    {zdecl} int {decl},nb=(tile%(N/128))*128,mb=(tile/(N/128))*128;
    if(s0<0)return;
    for (;{loop}) {{ int r=z/128,c=z%128;
      out[(mb+r)*N+nb+c]=partials[s0*16384+z]{adds}; }}
  }}'''

__all__=["BOUNDARY_QUANTUM","K_BLOCKS","OWNERS","OUTPUT_TILES","PARTIAL_SLOTS","TILE_ELEMENTS",
         "TILES_N","WORK_UNITS","active_fixup_source","transform_compiler_q4k_to_streamk"]
