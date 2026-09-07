from __future__ import annotations

import re

OWNERS, OUTPUT_TILES, K_BLOCKS, TILES_N = 170, 384, 64, 96
WORK_UNITS, BOUNDARY_QUANTUM = OUTPUT_TILES*K_BLOCKS, 8
TILE_ELEMENTS, PARTIAL_SLOTS = 128*128, 2*OWNERS

def _pack_q4_publication(math:str) -> str:
  """Replace the exact 32 scalar Q4 nibble publications with eight aligned words."""
  store_re=re.compile(r"^    \*\(buf1\+(?P<addr>[^\n]+)\) = \(\(signed char\)\(\(\((?P<val>val2[23])\.(?P<comp>[xyzw])>>(?P<shift>cast[1-4])\)&15u\)\)\);$",re.M)
  stores=list(store_re.finditer(math))
  if len(stores)!=32: raise ValueError(f"packed Q4 publication requires 32 scalar nibble stores, found {len(stores)}")
  seen=set()
  shifts=("cast4","cast1","cast2","cast3")
  for st in stores:
    bank=0 if st.group("val")=="val22" else 5120; comp="xyzw".index(st.group("comp")); shift=shifts.index(st.group("shift"))
    expected=bank+comp*4+shift; addr=st.group("addr")
    match=None if addr=="alu8" else re.fullmatch(r"\(alu8\+(\d+)\)",addr)
    if addr!="alu8" and match is None: raise ValueError("packed Q4 publication has an unsupported address expression")
    actual=0 if addr=="alu8" else int(match.group(1))
    if actual!=expected or (key:=(st.group("val"),st.group("comp"),st.group("shift"))) in seen:
      raise ValueError("packed Q4 publication lost exact lane/address ownership")
    seen.add(key)
  math=store_re.sub("",math)
  publish=math.find("    __syncthreads();")
  if publish<0: raise ValueError("packed Q4 publication barrier not found")
  packed=[]
  for val,bank in (("val22",0),("val23",5120)):
    for comp_i,comp in enumerate("xyzw"):
      offset=bank+comp_i*4; addr="alu8" if offset==0 else f"(alu8+{offset})"
      packed.append(f"    *reinterpret_cast<unsigned int*>(buf1+{addr}) = ({val}.{comp}>>alu1)&0x0f0f0f0fu;")
  math=math[:publish]+"\n".join(packed)+"\n"+math[publish:]
  if math.count("*reinterpret_cast<unsigned int*>(buf1+")!=8: raise ValueError("packed Q4 publication store census mismatch")
  return math

def _partial_store_block(direct_store_block:str, *, store_index:str="alu242", output_stride:int=12288, output_arg:str="data0_6291456") -> str:
  block=direct_store_block
  block=re.sub(rf"int {re.escape(store_index)} = .*?;", f"int {store_index} = ((alu5<<1)+(lidx2<<5)+(alu2*128)+(lidx1*8192));", block, count=1)
  block=block.replace(output_arg+"+", "partials+(slot*16384)+")
  def remap_offset(match:re.Match) -> str:
    value=int(match.group(1))
    row,column=divmod(value,output_stride)
    if column >= 128: raise ValueError(f"global output offset {value} escapes its 128-column tile")
    return f"{store_index}+{row*128+column}"
  # Rewrite the original tokens in one pass. Iterative replacement can remap a
  # newly produced offset a second time (for example 16384 -> 4096 -> 1024).
  block=re.sub(rf"{re.escape(store_index)}\+(\d+)",remap_offset,block)
  return block

def _logical_transpose_store_blocks(direct_store_block:str, *, output_arg:str="data0_6291456",
                                    logical_stride:int=12288) -> tuple[str,str]:
  """Transpose one exact 128x128 swapped IMMA tile with XOR4 and vector stores."""
  line_re=re.compile(r"^  \*\(\(float2\*\)\(\("+re.escape(output_arg)+
    r"\+\(?alu\d+(?:\+(\d+))?\)?\)\)\) = make_float2\(\(\*\(buf0\+(\d+)\)\),\(\*\(buf0\+(\d+)\)\)\);$",re.M)
  rows=[]
  for m in line_re.finditer(direct_store_block):
    off=int(m.group(1) or 0); row,col=divmod(off,512)
    if col>=128: raise ValueError("swapped output store escapes its physical tile")
    rows.append((row,col,int(m.group(2)),int(m.group(3))))
  if len(rows)!=32: raise ValueError(f"logical transpose requires 32 physical float2 stores, found {len(rows)}")
  table={(row,col):(a,b) for row,col,a,b in rows}
  if len(table)!=32: raise ValueError("logical transpose physical stores overlap")
  pairs=[]
  for row,col,a,b in rows:
    if row%16>=8: continue
    high=table.get((row+8,col))
    if high is None: raise ValueError("logical transpose has no r+8 C-fragment partner")
    pairs.append((row,col,(a,b),high))
  if len(pairs)!=16: raise ValueError("logical transpose requires 16 low/high store pairs")
  def emit(partial:bool)->str:
    lines=[]
    for pair,(row,col,lo,hi) in enumerate(pairs):
      for p in range(2):
        sl,sh,v0,v1=(f"tx{pair}_{p}_{x}" for x in ("sl","sh","0","1"))
        # Every lane must execute both full-mask shuffles before selection;
        # placing the intrinsic in a divergent ternary makes its result undefined.
        lines += [f"  float {sl} = __shfl_xor_sync(0xffffffffu, (*(buf0+{lo[p]})), 4);",
                  f"  float {sh} = __shfl_xor_sync(0xffffffffu, (*(buf0+{hi[p]})), 4);",
                  f"  float {v0} = (alu2&1) ? {sh} : (*(buf0+{lo[p]}));",
                  f"  float {v1} = (alu2&1) ? (*(buf0+{hi[p]})) : {sl};"]
        local_row=f"((lidx2<<5)+(alu5<<1)+{col+p})"
        local_col=f"((lidx1<<6)+{row}+alu2+7*(alu2&1))"
        index=f"(({local_row})*128+{local_col})" if partial else \
          f"(((gidx0*128+{local_row})*{logical_stride})+(gidx1*128)+{local_col})"
        dest=f"partials+(slot*16384)+{index}" if partial else f"{output_arg}+{index}"
        lines.append(f"  *((float2*)({dest})) = make_float2({v0},{v1});")
    return "\n".join(lines)+"\n"
  return emit(False),emit(True)

def coalesce_swapped_direct_source(source:str, *, logical_stride:int) -> str:
  """Rewrite only a compiler wide kernel's physical NxM stores to logical MxN."""
  signature=re.search(r'extern "C" __global__ void __launch_bounds__\(256\) \w+\(float\* (data0_\d+),', source)
  if signature is None: raise ValueError("wide swapped source has no exact output signature")
  output_arg=signature.group(1)
  store_re=re.compile(r"^  \*\(\(float2\*\)\(\("+re.escape(output_arg)+
    r"\+\(?alu\d+(?:\+\d+)?\)?\)\)\) = make_float2\(\(\*\(buf0\+\d+\)\),\(\*\(buf0\+\d+\)\)\);$", re.M)
  stores=list(store_re.finditer(source))
  if len(stores)!=32: raise ValueError(f"wide swapped source requires 32 terminal stores, found {len(stores)}")
  direct,_=_logical_transpose_store_blocks(source,output_arg=output_arg,logical_stride=logical_stride)
  # Compiler terminal stores are contiguous. Replacing one exact span preserves
  # all generated arithmetic, correction, launch geometry, and packed-input ABI.
  between=source[stores[0].start():stores[-1].end()]
  if len(store_re.findall(between))!=32: raise ValueError("wide swapped terminal store span is not contiguous")
  return source[:stores[0].start()]+direct.rstrip()+source[stores[-1].end():]

def transform_compiler_q4k_to_streamk(source:str, *, unroll:int|None=None, tiles_n:int=96,
                                     tiles_m:int=4, k_blocks:int=64, output_stride:int=12288,
                                     kernel_name:str="q4k_imma_stream", restrict_pointers:bool=False,
                                     double_buffer:bool=False, fragment_load_to_use:bool=False,
                                     shared_load_to_pack:bool|str=False, interleave_wmma_updates:bool=False,
                                     operand_order:str="activation_a_weight_b", logical_transpose_output:bool=False,
                                     q8_ds4_packed_loads:bool=False, q4_packed_publication:bool=False) -> str:
  """Wrap the compiler-owned Q4_K/Q8 tile body in llama-compatible Stream-K ownership.

  The signed-IMMA math and packed input addressing remain compiler emitted.  Only
  launch ownership, the outer K64 range, and terminal output destination change.
  """
  if any(x <= 0 for x in (tiles_m,tiles_n,k_blocks,output_stride)): raise ValueError("invalid Stream-K source geometry")
  signature=re.search(r'(extern "C" __global__ void __launch_bounds__\(256\) \w+\()'
                      r'(float\* (data0_\d+), unsigned int\* (data1_\d+), unsigned int\* (data2_\d+))(\) \{)',source)
  if signature is None: raise ValueError("compiler Q4 kernel signature not found")
  if f"Ridx0 < {k_blocks}" not in source: raise ValueError("source K loop does not match requested Stream-K geometry")
  if operand_order not in ("activation_a_weight_b","weight_a_activation_b"): raise ValueError("unsupported packed operand order")
  if logical_transpose_output and (operand_order!="weight_a_activation_b" or tiles_m not in (32,96) or
                                   (tiles_n,output_stride)!=(4,512)):
    raise ValueError("logical transpose output requires exact swapped 32x4 or 96x4 physical grid")
  out_name=kernel_name
  exported=f'extern "C" __global__ void __launch_bounds__(256) {out_name}('
  out_arg,slot1_arg,slot2_arg=signature.group(3),signature.group(4),signature.group(5)
  rec_arg,w_arg=(slot1_arg,slot2_arg) if operand_order=="activation_a_weight_b" else (slot2_arg,slot1_arg)
  qual=" __restrict__" if restrict_pointers else ""
  source=source[:signature.start()]+exported+(
    f"float*{qual} {out_arg}, float*{qual} partials, int*{qual} partial_ids, "
    f"const unsigned int*{qual} {w_arg}, const unsigned int*{qual} {rec_arg}) {{")+source[signature.end():]
  source=source.replace(f"  int gidx0 = blockIdx.x; /* {tiles_n} */\n  int gidx1 = blockIdx.y; /* {tiles_m} */\n",
                        "  int owner = blockIdx.x; /* 170 persistent owners */\n",1)
  if "int owner = blockIdx.x" not in source: raise ValueError("compiler output tile grid does not match requested Stream-K geometry")
  body_start=source.find("  (*(buf0+0)) = 0.0f;")
  store_matches=list(re.finditer(r"^  int (alu\d+) = ",source[body_start:],re.M)) if body_start >= 0 else []
  if body_start < 0 or not store_matches: raise ValueError("compiler Q4 body/store boundary not found")
  store_match=store_matches[-1]
  store_start=body_start+store_match.start(); store_index=store_match.group(1)
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
      # The tile-major Q8 carrier changes compiler numbering and leaves ten Q4
      # words ahead of the first activation-record load. Derive that boundary
      # from the captured ABI instead of mistaking activation val10 for Q4.
      weight_loads=list(re.finditer(rf"^    unsigned int val\d+ = \(\*\({re.escape(w_arg)}.*;$",math,re.M))
      record_loads=list(re.finditer(rf"^    unsigned int val\d+ = \(\*\({re.escape(rec_arg)}.*;$",math,re.M))
      if len(weight_loads)!=10 or not record_loads or weight_loads[-1].end() >= record_loads[0].start():
        raise ValueError("fragment load-to-use schedule requires exact flat or tile-major Q4 group")
      frag_start=weight_loads[0].start(); frag_stop=weight_loads[-1].end()+1
      fragment=math[frag_start:frag_stop]
      if fragment.count("unsigned int val")!=10 or rec_arg in fragment:
        raise ValueError("tile-major fragment load-to-use boundary escaped Q4 words")
      publish=math.find("    __syncthreads();",record_loads[0].start())
      if publish<0: raise ValueError("tile-major fragment load-to-use publish marker not found")
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
  if q8_ds4_packed_loads:
    if operand_order!="weight_a_activation_b": raise ValueError("packed Q8 DS4 loads require activation in MMA B")
    decl_re=re.compile(r"^    signed char (val\d+) = \(\*\(buf1\+\((alu\d+)\+(\d+)\)\)\);\n",re.M)
    decls={m.group(1):(m.group(0),m.group(2),int(m.group(3))) for m in decl_re.finditer(math)}
    remove=set(); replacements={}
    packs=list(re.finditer(r"^    signed_char8 (cast\d+) = make_signed_char8\((val\d+),(val\d+),(val\d+),(val\d+),(val\d+),(val\d+),(val\d+),(val\d+)\);$",math,re.M))
    if len(packs)!=8: raise ValueError(f"packed Q8 DS4 loads require eight char8 fragments, found {len(packs)}")
    for pack in packs:
      cast,*names=pack.groups(); entries=[decls.get(name) for name in names]
      if any(x is None for x in entries): raise ValueError("Q8 fragment escaped scalar declarations")
      bases={x[1] for x in entries}; offsets=[x[2] for x in entries]
      if len(bases)!=1 or offsets[1:4]!=[offsets[0]+i for i in range(1,4)] or offsets[4:]!=[offsets[0]+16+i for i in range(4)] or offsets[0]%4:
        raise ValueError("Q8 fragment addresses are not two aligned dword runs")
      remove.update(x[0] for x in entries); base=entries[0][1]
      replacements[pack.group(0)]=(f"    uint2 {cast} = make_uint2(*reinterpret_cast<const unsigned int*>(buf1+({base}+{offsets[0]})), "
        f"*reinterpret_cast<const unsigned int*>(buf1+({base}+{offsets[4]})));")
    halves=list(re.finditer(r"^    float (cast(?:5[7-9]|[6-7][0-9]|8[0-8])) = (.*);$",math,re.M))
    if len(halves)!=32: raise ValueError(f"packed Q8 DS4 loads require 32 half values, found {len(halves)}")
    for half in halves:
      names=re.findall(r"\bval\d+\b",half.group(2))
      if len(names)!=2 or any(name not in decls for name in names): raise ValueError("Q8 half escaped scalar declarations")
      lo,hi=decls[names[0]],decls[names[1]]
      if lo[1]!=hi[1] or hi[2]!=lo[2]+1 or lo[2]%2: raise ValueError("Q8 half address is not aligned")
      remove.update((lo[0],hi[0])); replacements[half.group(0)]=f"    float {half.group(1)} = ((float)(*reinterpret_cast<const half*>(buf1+({lo[1]}+{lo[2]}))));"
    if len(remove)!=128: raise ValueError(f"packed Q8 DS4 loads require 128 unique declarations, found {len(remove)}")
    for line in remove:
      if math.count(line)!=1: raise ValueError("Q8 scalar declaration ownership is not unique")
      math=math.replace(line,"",1)
    for old,new in replacements.items():
      if math.count(old)!=1: raise ValueError("Q8 packed consumer ownership is not unique")
      math=math.replace(old,new,1)
    if math.count("reinterpret_cast<const unsigned int*>(buf1+")!=16 or math.count("reinterpret_cast<const half*>(buf1+")!=32:
      raise ValueError("packed Q8 DS4 load census mismatch")
    math=math.replace("__WMMA_8_16_32_signed_char_int(","__WMMA_8_16_32_signed_char_int_q8u2(")
    helper="__device__ int4 __WMMA_8_16_32_signed_char_int(signed_char16 a, signed_char8 b, int4 c){"
    if source.count(helper)!=1: raise ValueError("Q8 uint2 WMMA helper not found")
    source=source.replace(helper,"__device__ int4 __WMMA_8_16_32_signed_char_int_q8u2(signed_char16 a, uint2 b, int4 c){",1)
  if q4_packed_publication:
    if operand_order!="weight_a_activation_b": raise ValueError("packed Q4 publication requires weight in MMA A")
    math=_pack_q4_publication(math)
  if interleave_wmma_updates:
    # Compute eight IMMA results and their output-column scale values together.
    # Each accumulator expression remains byte-for-byte unchanged, while 24
    # unrelated int4 IMMA results no longer stay live.
    wmma_re=re.compile(r"^    int4 (wmma\d+) = .*;$",re.M)
    # Cast numbering differs by four when the typed Q4-A/x4 carrier removes
    # scalar fragment assembly.  The body contract is 64 float scale
    # declarations consumed by the 64 accumulator updates; derive their names
    # instead of coupling this schedule to renderer-local numbering.
    scale_re=re.compile(r"^    float (cast\d+) = .*;$",re.M)
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
    scale_names=sorted(scales,key=lambda x:int(x[4:])); common_scales=set(scale_names[:32])
    groups=[*(scales[x] for x in scale_names[:32])]; used_wmmas=set(); used_scales=set(common_scales)
    for col in range(0,16,4):
      lines=[updates[c+16*i] for c in range(col,col+4) for i in range(4)]
      wnames=[]; snames=[]
      for line in lines:
        wnames.extend(re.findall(r"\bwmma\d+\b",line)); snames.extend(x for x in re.findall(r"\bcast\d+\b",line) if x in scales)
      wnames=list(dict.fromkeys(wnames)); snames=list(dict.fromkeys(snames))
      local_scales=[x for x in snames if x not in common_scales]
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
  if logical_transpose_output: direct,partial=_logical_transpose_store_blocks(direct,output_arg=signature.group(3))
  else: partial=_partial_store_block(direct,store_index=store_index,output_stride=output_stride,output_arg=signature.group(3))
  prefix=source[:body_start]
  work_units=tiles_n*tiles_m*k_blocks
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

def active_fixup_source(*, max_contributors:int=2, sliced:bool=False, transpose_physical_tiles_n:int|None=None) -> str:
  if max_contributors < 2: raise ValueError("fixup requires at least two contributors")
  decl=','.join(f"s{i}=map[{max_contributors}*tile+{i}]" for i in range(max_contributors))
  adds=''.join(f"+(s{i}>=0?partials[s{i}*16384+z]:0)" for i in range(1,max_contributors))
  zdecl="int tile=active[blockIdx.x],z=threadIdx.x;" if not sliced else "int tile=active[blockIdx.x],z=blockIdx.y*4096+threadIdx.x;"
  loop="z<16384;z+=256" if not sliced else "z<((blockIdx.y+1)*4096);z+=128"
  tile_map=(f"nb=(tile/{transpose_physical_tiles_n})*128,mb=(tile%{transpose_physical_tiles_n})*128" if
            transpose_physical_tiles_n is not None else "nb=(tile%(N/128))*128,mb=(tile/(N/128))*128")
  return f'''extern "C" __global__ void q4k_imma_fixup_active(float *out,const float *partials,const int *map,const int *active,int M,int N) {{
    {zdecl} int {decl},{tile_map};
    if(s0<0)return;
    for (;{loop}) {{ int r=z/128,c=z%128;
      out[(mb+r)*N+nb+c]=partials[s0*16384+z]{adds}; }}
  }}'''

__all__=["BOUNDARY_QUANTUM","K_BLOCKS","OWNERS","OUTPUT_TILES","PARTIAL_SLOTS","TILE_ELEMENTS",
         "TILES_N","WORK_UNITS","active_fixup_source","transform_compiler_q4k_to_streamk"]
