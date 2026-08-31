from __future__ import annotations

import re

OWNERS, OUTPUT_TILES, K_BLOCKS, TILES_N = 170, 1024, 192, 128
WORK_UNITS, BOUNDARY_QUANTUM = OUTPUT_TILES*K_BLOCKS, 8
TILE_M, TILE_N, TILE_ELEMENTS, PARTIAL_SLOTS = 64, 32, 64*32, 2*OWNERS

def _partial_store_block(direct:str) -> str:
  block=re.sub(r"int alu116 = .*?;", "int alu116 = ((alu3<<1)+(lidx2<<4)+(alu0*32)+(lidx1*1024));", direct, count=1)
  block=block.replace("data0_2097152+", "partials+(slot*2048)+")
  for value in sorted({int(x) for x in re.findall(r"alu116\+(\d+)",block)},reverse=True):
    row,column=divmod(value,4096)
    if column >= TILE_N: raise ValueError(f"Q6 global output offset {value} escapes its tile")
    block=block.replace(f"alu116+{value}",f"alu116+{row*TILE_N+column}")
  return block

def transform_compiler_q6k_to_streamk(source:str, *, unroll:int|None=None) -> str:
  signature=re.search(r'(extern "C" __global__ void __launch_bounds__\(128\) \w+\()'
                      r'(float\* data0_2097152, unsigned int\* data1_1966080, unsigned short\* data2_20643840)(\) \{)',source)
  if signature is None: raise ValueError("compiler Q6-down signature not found")
  exported='extern "C" __global__ void __launch_bounds__(128) q6k_imma_stream('
  source=source[:signature.start()]+exported+(
    "float* data0_2097152, float* partials, int* partial_ids, "
    "unsigned short* data2_20643840, unsigned int* data1_1966080")+signature.group(3)+source[signature.end():]
  source=source.replace("  int gidx0 = blockIdx.x; /* 128 */\n  int gidx1 = blockIdx.y; /* 8 */\n",
                        "  int owner = blockIdx.x; /* 170 persistent owners */\n",1)
  body_start=source.find("  (*(buf0+0)) = 0.0f;"); store_start=source.find("  int alu116 = ",body_start); function_end=source.rfind("}")
  if min(body_start,store_start,function_end)<0: raise ValueError("compiler Q6 body/store boundary not found")
  math=source[body_start:store_start]
  loop="for (int Ridx0 = 0; Ridx0 < 192; Ridx0++) {"
  if unroll is not None:
    if unroll not in (1,2,4,8): raise ValueError("unsupported Q6 Stream-K outer-K unroll")
    math=math.replace(loop,f"#pragma unroll {unroll}\n  {loop}",1)
  math=math.replace(loop,"for (int Ridx0 = k_begin; Ridx0 < k_end; Ridx0++) {",1)
  direct=source[store_start:function_end]; partial=_partial_store_block(direct); prefix=source[:body_start]
  owner_loop=f"""  int owner_start = ((owner*{WORK_UNITS}/{OWNERS})/{BOUNDARY_QUANTUM})*{BOUNDARY_QUANTUM};
  if (threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0) {{ partial_ids[owner*2]=-1; partial_ids[owner*2+1]=-1; }}
  int owner_stop = (owner == {OWNERS-1}) ? {WORK_UNITS} : ((((owner+1)*{WORK_UNITS}/{OWNERS})/{BOUNDARY_QUANTUM})*{BOUNDARY_QUANTUM});
  int first_tile=owner_start/{K_BLOCKS},last_tile=(owner_stop-1)/{K_BLOCKS};
  for (int tile=first_tile;tile<=last_tile;tile++) {{
    int segment_start=max(owner_start,tile*{K_BLOCKS}),segment_stop=min(owner_stop,(tile+1)*{K_BLOCKS});
    int k_begin=segment_start-tile*{K_BLOCKS},k_end=segment_stop-tile*{K_BLOCKS};
    int gidx0=tile%{TILES_N},gidx1=tile/{TILES_N}; bool direct=(k_begin==0&&k_end=={K_BLOCKS});
    bool owner_tail=(segment_stop==owner_stop&&k_end!={K_BLOCKS}),owner_has_head=((owner_start%{K_BLOCKS})!=0);
    int slot=owner*2+((owner_tail&&owner_has_head)?1:0);
"""
  stores=("    if (direct) {\n"+direct+"    } else {\n"
          "      if (threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0) partial_ids[slot]=tile;\n"+partial+"    }\n"
          "    __syncthreads(); /* protect shared fragments across owner tile segments */\n")
  return prefix+owner_loop+math+stores+"  }\n}\n"

def active_fixup_source() -> str:
  return r'''extern "C" __global__ void q6k_imma_fixup_active(float *out,const float *partials,const int *map,const int *active,int M,int N) {
    int tile=active[blockIdx.x],s0=map[2*tile],s1=map[2*tile+1],nb=(tile%(N/32))*32,mb=(tile/(N/32))*64;
    for (int z=threadIdx.x;z<2048;z+=256) { int r=z/32,c=z%32;
      out[(mb+r)*N+nb+c]=partials[s0*2048+z]+(s1>=0?partials[s1*2048+z]:0); }
  }'''

def transform_compiler_q6k_wide_to_streamk(source:str, *, unroll:int|None=None, owners:int=170, force_partials:bool=False) -> str:
  if not 1 <= owners <= 256: raise ValueError("wide Q6 Stream-K owners must be in [1, 256]")
  signature=re.search(r'(extern "C" __global__ void __launch_bounds__\(256\) \w+\()'
                      r'(float\* data0_2097152, unsigned int\* data1_1966080, unsigned short\* data2_20643840)(\) \{)',source)
  if signature is None: raise ValueError("compiler wide Q6-down signature not found")
  source=(source[:signature.start()]+'extern "C" __global__ void __launch_bounds__(256) q6k_imma_stream('
          'float* data0_2097152, float* partials, int* partial_ids, unsigned short* data2_20643840, unsigned int* data1_1966080'
          +signature.group(3)+source[signature.end():])
  source=source.replace("  int gidx0 = blockIdx.x; /* 32 */\n  int gidx1 = blockIdx.y; /* 4 */\n",
                        "  int owner = blockIdx.x; /* 170 persistent owners */\n",1)
  body_start=source.find("  (*(buf0+0)) = 0.0f;"); store_start=source.find("  int alu233 = ",body_start); function_end=source.rfind("}")
  if min(body_start,store_start,function_end)<0: raise ValueError("compiler wide Q6 body/store boundary not found")
  math=source[body_start:store_start]; loop="for (int Ridx0 = 0; Ridx0 < 192; Ridx0++) {"
  if unroll is not None:
    if unroll not in (1,2,4,8): raise ValueError("unsupported wide Q6 Stream-K outer-K unroll")
    math=math.replace(loop,f"#pragma unroll {unroll}\n  {loop}",1)
  math=math.replace(loop,"for (int Ridx0 = k_begin; Ridx0 < k_end; Ridx0++) {",1)
  direct=source[store_start:function_end]
  partial=re.sub(r"int alu233 = .*?;","int alu233 = ((alu3<<1)+(lidx2<<5)+(alu0*128)+(lidx1*8192));",direct,count=1)
  partial=partial.replace("data0_2097152+","partials+(slot*16384)+")
  for value in sorted({int(x) for x in re.findall(r"alu233\+(\d+)",partial)},reverse=True):
    row,column=divmod(value,4096)
    if column>=128: raise ValueError(f"wide Q6 output offset {value} escapes its tile")
    partial=partial.replace(f"alu233+{value}",f"alu233+{row*128+column}")
  work_units=128*192
  owner_loop=f"""  int owner_start=((owner*{work_units}/{owners})/8)*8;
  if (threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0) {{ partial_ids[owner*2]=-1; partial_ids[owner*2+1]=-1; }}
  int owner_stop=(owner=={owners-1})?{work_units}:((((owner+1)*{work_units}/{owners})/8)*8);
  int first_tile=owner_start/192,last_tile=(owner_stop-1)/192;
  for (int tile=first_tile;tile<=last_tile;tile++) {{
    int segment_start=max(owner_start,tile*192),segment_stop=min(owner_stop,(tile+1)*192);
    int k_begin=segment_start-tile*192,k_end=segment_stop-tile*192,gidx0=tile%32,gidx1=tile/32;
    bool direct=(k_begin==0&&k_end==192),owner_tail=(segment_stop==owner_stop&&k_end!=192),owner_has_head=((owner_start%192)!=0);
    int slot=owner*2+((owner_tail&&owner_has_head)?1:0);
"""
  if force_partials: owner_loop=owner_loop.replace("bool direct=(k_begin==0&&k_end==192)", "bool direct=false")
  stores=("    if (direct) {\n"+direct+"    } else {\n"
          "      if (threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0) partial_ids[slot]=tile;\n"+partial+"    }\n")
  return source[:body_start]+owner_loop+math+stores+"  }\n}\n"

def wide_active_fixup_source() -> str:
  return r'''extern "C" __global__ void q6k_imma_fixup_active(float *out,const float *partials,const int *map,const int *active,int M,int N) {
    int tile=active[blockIdx.x],s0=map[3*tile],s1=map[3*tile+1],s2=map[3*tile+2],nb=(tile%(N/128))*128,mb=(tile/(N/128))*128;
    for (int z=threadIdx.x;z<16384;z+=256) { int r=z/128,c=z%128;
      float v=partials[s0*16384+z]; if(s1>=0)v+=partials[s1*16384+z]; if(s2>=0)v+=partials[s2*16384+z];
      out[(mb+r)*N+nb+c]=v; }
  }'''

def transform_compiler_q6k_wide_persistent_b(source:str) -> str:
  """Cache one canonical 128-row Q6_K K256 block across four K64 phases."""
  declaration="  __shared__ __align__(16) signed char buf1[20480];\n"
  if source.count(declaration) != 1: raise ValueError("compiler wide Q6 shared declaration not found")
  source=source.replace(declaration, declaration+"  __shared__ __align__(16) unsigned short q6_cache[13440];\n", 1)
  old_alu="    int alu77 = ((lidx1*161280)+(lidx2*40320)+(alu0*5040)+(gidx0*645120)+((Ridx0>>2)*105));"
  new_alu="    int alu77 = ((lidx1*3360)+(lidx2*840)+(alu0*105));"
  if source.count(old_alu) != 1: raise ValueError("compiler wide Q6 packed address root not found")
  source=source.replace(old_alu,new_alu,1)
  source=source.replace("data2_20643840+", "q6_cache+")
  source=re.sub(r"\b322(?:5[6-9]\d|6[0-6]\d)\b", lambda m:str(int(m.group(0))-315840), source)
  escaped=[int(x) for x in re.findall(r"q6_cache\+[^;\n]*?\b(\d{6})\b",source)]
  if escaped: raise ValueError(f"compiler wide Q6 cache retained global offsets {escaped}")
  loop="  for (int Ridx0 = k_begin; Ridx0 < k_end; Ridx0++) {\n"
  if source.count(loop) != 1: raise ValueError("compiler wide Q6 phase loop not found")
  cache=loop+'''    if ((Ridx0&3)==0) {
      __syncthreads();
      int q6_tid=lidx0+32*lidx1+64*lidx2;
      for (int q6_i=q6_tid;q6_i<13440;q6_i+=256) {
        int q6_row=q6_i/105,q6_word=q6_i-q6_row*105;
        q6_cache[q6_i]=data2_20643840[(gidx0*128+q6_row)*5040+(Ridx0>>2)*105+q6_word];
      }
      __syncthreads();
    }
'''
  return source.replace(loop,cache,1)

def transform_compiler_q6k_wide_live_publication(source:str) -> str:
  """Publish only bytes consumed by the wide Q6 MMA body.

  The producer's four scale planes reserve sixteen bytes per 80-byte row.
  Collective consumer-address enumeration proves that planes zero and one use
  bytes {0,1,8,9}, while planes two and three use bytes {0..3,8..11}.
  The payload planes remain byte-complete and are deliberately untouched.
  """
  planes = (
    (64, 2), (5184, 2), (10304, 4), (15424, 4),
  )
  for base, width in planes:
    lines = []
    for off in range(4):
      match = re.search(rf"^    \*\(buf1\+\(alu5\+{base+off}\)\) = .*;$", source, re.MULTILINE)
      if match is None: raise ValueError(f"wide Q6 publication store {base+off} not found")
      if off < width: lines.append(match.group(0))
    first = re.search(rf"^    \*\(buf1\+\(alu5\+{base}\)\) = .*;$", source, re.MULTILINE)
    last = re.search(rf"^    \*\(buf1\+\(alu5\+{base+3}\)\) = .*;$", source, re.MULTILINE)
    assert first is not None and last is not None
    replacement = "    if ((alu3&1)==0) {\n" + "\n".join("  "+line for line in lines) + "\n    }"
    source = source[:first.start()] + replacement + source[last.end():]
  return source

__all__=["active_fixup_source","transform_compiler_q6k_to_streamk","transform_compiler_q6k_wide_to_streamk",
         "transform_compiler_q6k_wide_live_publication","transform_compiler_q6k_wide_persistent_b","wide_active_fixup_source"]
