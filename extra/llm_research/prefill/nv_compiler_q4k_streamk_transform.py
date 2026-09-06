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
                                     shared_load_to_pack:bool|str=False, interleave_wmma_updates:bool=False) -> str:
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
    mode="all" if shared_load_to_pack is True else shared_load_to_pack
    if mode not in ("all","fragments","scales"): raise ValueError(f"unsupported shared load-to-pack mode {mode!r}")
    load_re=re.compile(r"^    signed char (val(?:2[6-9]|[3-9][0-9]|[12][0-9]{2}|3[0-4][0-9]|345)) = \(\*\(buf1.*\);\n",re.M)
    all_loads={m.group(1):m.group(0) for m in load_re.finditer(math)}
    if len(all_loads)!=320: raise ValueError(f"shared load-to-pack requires exact val26..val345 set, found {len(all_loads)}")
    pack_re=re.compile(r"^    (?:signed_char(?:8|16)|float) cast(?:1[7-9]|[2-9][0-9]) = .*;$",re.M)
    def selected_pack(line):
      cast=int(re.search(r"cast(\d+)",line).group(1))
      return mode=="all" or (mode=="fragments" and cast<=32) or (mode=="scales" and cast>=33)
    selected={x for m in pack_re.finditer(math) if selected_pack(m.group(0)) for x in re.findall(r"\bval\d+\b",m.group(0)) if x in all_loads}
    expected={"all":320,"fragments":192,"scales":128}[mode]
    if len(selected)!=expected: raise ValueError(f"shared load-to-pack {mode} expected {expected} loads, found {len(selected)}")
    loads={x:all_loads[x] for x in selected}
    math=re.sub("|".join(re.escape(loads[x]) for x in sorted(loads)),"",math)
    consumed=set()
    def stage_pack(m):
      names=[x for x in re.findall(r"\bval\d+\b",m.group(0)) if x in loads]
      if not names or not selected_pack(m.group(0)): return m.group(0)
      if any(x in consumed for x in names): raise ValueError("shared scalar load has multiple pack consumers")
      consumed.update(names)
      return "".join(loads[x] for x in names)+m.group(0)
    math=pack_re.sub(stage_pack,math)
    if consumed!=set(loads): raise ValueError(f"shared load-to-pack left {len(set(loads)-consumed)} loads without a pack")
  if interleave_wmma_updates:
    # Compute eight IMMA results and their output-column scale values together.
    # Each accumulator expression remains byte-for-byte unchanged, while 24
    # unrelated int4 IMMA results no longer stay live.
    wmma_re=re.compile(r"^    int4 (wmma\d+) = .*;$",re.M)
    scale_re=re.compile(r"^    float (cast(?:3[3-9]|[4-8][0-9]|9[0-6])) = .*;$",re.M)
    update_re=re.compile(r"^    \(\*\(buf0\+(\d+)\)\) = .*;$",re.M)
    wmmas={m.group(1):m.group(0) for m in wmma_re.finditer(math)}
    scales={m.group(1):m.group(0) for m in scale_re.finditer(math)}
    updates={int(m.group(1)):m.group(0) for m in update_re.finditer(math)}
    if len(wmmas)!=32 or len(scales)!=64 or set(updates)!=set(range(64)):
      raise ValueError(f"interleaved IMMA schedule requires 32 wmmas, 64 scales, and 64 updates; got {len(wmmas)}, {len(scales)}, {len(updates)}")
    insertion=min(m.start() for m in update_re.finditer(math))
    marker="    /* INTERLEAVED_WMMA_UPDATES */\n"
    math=math[:insertion]+marker+math[insertion:]
    math=wmma_re.sub("",math); math=scale_re.sub("",math); math=update_re.sub("",math)
    # Activation scales cast33..64 are reused across all four output-column
    # groups, so materialize them once.  Each group then owns eight IMMA
    # results and its eight weight scales without duplicating arithmetic.
    groups=[*(scales[f"cast{i}"] for i in range(33,65))]; used_wmmas=set(); used_scales={f"cast{i}" for i in range(33,65)}
    for col in range(0,16,4):
      lines=[updates[c+16*i] for c in range(col,col+4) for i in range(4)]
      wnames=[]; snames=[]
      for line in lines:
        wnames.extend(re.findall(r"\bwmma\d+\b",line)); snames.extend(x for x in re.findall(r"\bcast\d+\b",line) if x in scales)
      wnames=list(dict.fromkeys(wnames)); snames=list(dict.fromkeys(snames))
      local_scales=[x for x in snames if int(x[4:])>=65]
      if len(wnames)!=8 or len(local_scales)!=8: raise ValueError("unexpected IMMA accumulator dependency group")
      if any(x in used_wmmas for x in wnames) or any(x in used_scales for x in local_scales):
        raise ValueError("interleaved IMMA declaration would be emitted more than once")
      used_wmmas.update(wnames); used_scales.update(local_scales)
      groups.extend([*(wmmas[x] for x in wnames),*(scales[x] for x in local_scales),*lines])
    if used_wmmas!=set(wmmas) or used_scales!=set(scales): raise ValueError("interleaved IMMA schedule did not consume every declaration")
    if math.count(marker)!=1: raise ValueError("interleaved IMMA insertion marker lost")
    math=math.replace(marker,"\n".join(groups)+"\n",1)
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
