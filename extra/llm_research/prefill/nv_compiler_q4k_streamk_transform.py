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
                                     kernel_name:str="q4k_imma_stream", restrict_pointers:bool=False,
                                     double_buffer:bool=False, fragment_load_to_use:bool=False,
                                     shared_load_to_pack:bool=False) -> str:
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
  qual=" __restrict__" if restrict_pointers else ""
  source=source[:signature.start()]+exported+(
    f"float*{qual} {out_arg}, float*{qual} partials, int*{qual} partial_ids, "
    f"const unsigned int*{qual} {rec_arg}, const unsigned int*{qual} {w_arg}) {{")+source[signature.end():]
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
    if unroll not in (1,2,4,6,8,10,12,16,32): raise ValueError("unsupported Stream-K outer-K unroll")
    math=math.replace(loop,f"#pragma unroll {unroll}\n  {loop}",1)
  math=math.replace(loop,"for (int Ridx0 = k_begin; Ridx0 < k_end; Ridx0++) {",1)
  if fragment_load_to_use:
    # Keep the emitted loads and arithmetic intact, but shorten the lifetime of
    # the Q4 fragment words by moving them after the Q8 metadata/data loads.
    frag_start=math.find("    unsigned int val0 =")
    frag_stop=math.find("    unsigned int val11 =",frag_start)
    publish=math.find("    __syncthreads();",frag_stop)
    if min(frag_start,frag_stop,publish)<0: raise ValueError("fragment load-to-use schedule markers not found")
    fragment=math[frag_start:frag_stop]
    if fragment.count("unsigned int val")!=11 or "int alu79 =" not in fragment:
      raise ValueError("fragment load-to-use schedule requires exact val0..val10 group")
    math=math[:frag_start]+math[frag_stop:publish]+fragment+math[publish:]
  if shared_load_to_pack:
    # The emitted source declares every scalar shared load before packing any
    # fragment.  Place each single-use load immediately before its pack instead,
    # preserving the pack and all subsequent IMMA/FP32 arithmetic verbatim.
    load_re=re.compile(r"^    signed char (val(?:2[6-9]|[3-9][0-9]|[12][0-9]{2}|3[0-4][0-9]|345)) = \(\*\(buf1.*\);\n",re.M)
    loads={m.group(1):m.group(0) for m in load_re.finditer(math)}
    if len(loads)!=320: raise ValueError(f"shared load-to-pack requires exact val26..val345 set, found {len(loads)}")
    math=load_re.sub("",math)
    pack_re=re.compile(r"^    (?:signed_char(?:8|16)|float) cast(?:1[7-9]|[2-9][0-9]) = .*;$",re.M)
    consumed=set()
    def stage_pack(m):
      names=[x for x in re.findall(r"\bval\d+\b",m.group(0)) if x in loads]
      if not names: return m.group(0)
      if any(x in consumed for x in names): raise ValueError("shared scalar load has multiple pack consumers")
      consumed.update(names)
      return "".join(loads[x] for x in names)+m.group(0)
    math=pack_re.sub(stage_pack,math)
    if consumed!=set(loads): raise ValueError(f"shared load-to-pack left {len(set(loads)-consumed)} loads without a pack")
  if double_buffer:
    shared="__shared__ __align__(16) signed char buf1[20480];"
    if source.count(shared)!=1: raise ValueError("double buffer requires the exact 20 KiB shared tile")
    source=source.replace(shared,"__shared__ __align__(16) signed char buf1[40960];",1)
    if math.count("__syncthreads();")!=2: raise ValueError("double buffer requires exact recycle/publish barriers")
    math=math.replace("__syncthreads();","",1)
    math=math.replace("buf1+","buf1+((Ridx0&1)*20480)+")
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
