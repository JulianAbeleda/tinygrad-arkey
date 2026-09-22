import os

from tinygrad.codegen.opt import tc
from tinygrad.dtype import AddrSpace, DType, dtypes
from tinygrad.helpers import NV_FLASH_LOAD_SCHEDULE, Target, dedup, prod
from tinygrad.renderer.cstyle import CStyleLanguage, base_rewrite, create_non_native_float_pats, uops_to_dtypes, wmma_args, _install_native_attention_bindings
from tinygrad.uop.ops import LoadSchedule, Ops, PatternMatcher, RegionLoad, StrictAfter, UPat, UOp

_nms = list("xyzwabcdefghijkl") + [f'v{i}' for i in range(16, 32)]

# Programmatic dependent launch (PDL) emission, env-gated and name-pinned.
# Off by default: when the lists are empty no kernel source changes, so
# unmarked programs stay byte-identical. A consumer program gets
# `griddepcontrol.wait` (SASS ACQBULK) at the top of its body; a producer
# program gets `griddepcontrol.launch_dependents` (SASS PREEXIT). The
# instruction is emitted at the END of the producer body by default (the
# native last-CTA semantics); set NV_PDL_TRIGGER_POSITION=start to emit it at
# the TOP, which is llama's `cudaTriggerProgrammaticLaunchCompletion` at
# kernel-start semantics. Unset NV_PDL_TRIGGER_POSITION keeps the default
# end placement and therefore byte-identical output. The matching supports
# exact names and `prefix:` rules, same vocabulary as the NV multi-queue
# admission policy in tinygrad/runtime/graph/hcq.py.
def _nv_pdl_match(name:str, spec:frozenset[str]) -> bool:
  return name in spec or any(rule.startswith("prefix:") and name.startswith(rule.removeprefix("prefix:")) for rule in spec)

def _nv_pdl_body(name:str, kernel:list[str]) -> list[str]:
  """Prepend/append PDL asm to a rendered kernel body for marked programs."""
  consumers = frozenset(x for x in os.environ.get("NV_PDL_CONSUMER_PROGRAMS", "").split(",") if x)
  producers = frozenset(x for x in os.environ.get("NV_PDL_PRODUCER_PROGRAMS", "").split(",") if x)
  if consumers and _nv_pdl_match(name, consumers):
    kernel = ['  asm volatile("griddepcontrol.wait;");', *kernel]
  if producers and _nv_pdl_match(name, producers):
    launch = '  asm volatile("griddepcontrol.launch_dependents;");'
    kernel = ([launch, *kernel] if os.environ.get("NV_PDL_TRIGGER_POSITION", "end") == "start" else [*kernel, launch])
  return kernel


def _nv_pdl_body_split_phase(name:str, kernel:list[str]) -> list[str]:
  """Policy-driven PDL emission behind NV_SPLIT_PHASE=1 (interim scaffolding).

  Reads the versioned NV_SPLIT_PHASE_POLICY JSON file via the memoized loader
  in extra/llm_research/decode/nv_edge_aware_pdl_render_policy.py instead of
  the process-wide NV_PDL_* name prefixes. This is not the production
  interface: the target is per-program split policy carried in graph metadata.
  """
  try:
    from extra.llm_research.decode.nv_edge_aware_pdl_render_policy import apply_nv_split_phase_policy
  except ImportError as e:
    raise ImportError("NV_SPLIT_PHASE=1 requires extra/llm_research/decode/nv_edge_aware_pdl_render_policy importable (run from the repo root)") from e
  return apply_nv_split_phase_policy(name, kernel)


def _nv_l2_streaming_weight_source(name:str, source:str) -> str:
  """Research gate: render direct loads from the second kernel buffer with ``__ldcs``.

  Rules identify the immutable weight-buffer indices for hand-authored decode
  GEMVs. CUDA's streaming-load intrinsic gives those lines evict-first priority
  instead of letting them displace the KV working set. The rewrite is
  exact-name/env gated and rejects a matched kernel with no eligible loads, so
  the default renderer remains byte-identical.
  """
  # Rule spelling is ``program@1+2`` (buffer indices); ``@...`` omitted means
  # buffer 1.  Exact names and ``prefix:`` keep the existing NV policy idiom.
  indices = None
  for raw in (x for x in os.environ.get("NV_L2_STREAMING_WEIGHT_PROGRAMS", "").split(",") if x):
    rule, sep, fields = raw.rpartition("@")
    if not sep: rule, fields = raw, "1"
    if _nv_pdl_match(name, frozenset((rule,))):
      indices = tuple(int(x) for x in fields.split("+") if x)
      break
  if indices is None: return source
  needles = tuple(f"data{i}_" for i in indices)
  out, replaced = [], 0
  for line in source.splitlines():
    if " = (*" in line and any(x in line for x in needles) and line.endswith(";"):
      lhs, rhs = line.rsplit(" = ", 1)
      expr = rhs[:-1]
      if expr.startswith("(*") and expr.endswith(")"):
        line = f"{lhs} = __ldcs({expr[2:-1]});"
        replaced += 1
    out.append(line)
  if replaced == 0: raise RuntimeError(f"NV L2 streaming-weight policy matched {name!r} but rewrote no data1 loads")
  return "\n".join(out)


def _nv_l2_q6_payload_source(name:str, source:str) -> str:
  """Research gate: stream Q6 ql/qh payload while retaining its reused metadata.

  Unlike the split-pointer experiment this preserves the production ABI.  The
  exact Q6 decode grammar emits qh as val0..7, scales as val8..15, ql as
  val16..23, and block scales as val24..27.
  """
  programs=frozenset(x for x in os.environ.get("NV_L2_STREAMING_Q6_PAYLOAD_PROGRAMS", "").split(",") if x)
  if not programs or not _nv_pdl_match(name,programs): return source
  out,replaced=[],0
  for line in source.splitlines():
    stripped=line.strip()
    selected=any(stripped.startswith(f"unsigned short val{i} = (*") for i in (*range(8),*range(16,24)))
    if selected and "data1_" in line and line.endswith(";"):
      lhs,rhs=line.rsplit(" = ",1);expr=rhs[:-1]
      if expr.startswith("(*") and expr.endswith(")"):
        line=f"{lhs} = __ldcs({expr[2:-1]});";replaced+=1
    out.append(line)
  if replaced != 16: raise RuntimeError(f"NV Q6 payload policy matched {name!r}, expected 16 rewrites but saw {replaced}")
  return "\n".join(out)


def _nv_flash_kv_evict_last_source(name:str, source:str) -> str:
  """Research gate for explicit evict-last policy on wide Flash K/V vectors.

  HCQ submits native QMDs and therefore cannot consume CUDA stream access
  policy windows.  This source lease tests the closest dispatch-independent
  mechanism: a cache-hinted load policy carried by each K/V instruction.
  """
  dynamic=bool(os.environ.get("NV_FLASH_KV_EVICT_LAST", ""));static_raw=os.environ.get("NV_FLASH_KV_STATIC_PERSISTING", "");static=bool(static_raw)
  if not (dynamic or static) or not name.startswith("flash_vec_llama_score_pv_"): return source
  if static:
    numerator=int(static_raw,0);assert 0 <= numerator <= 15
    descriptor=0x1400000000000000 | (numerator<<52)
  setup=(f'unsigned long long policy = 0x{descriptor:016X}ull;' if static else
    'unsigned long long policy; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));')
  helper=f'''__device__ __forceinline__ uint4 tg_nv_ld_evict_last(const uint4 *p) {{
  uint4 v; {setup}
  asm volatile("ld.global.L2::cache_hint.v4.u32 {{%0, %1, %2, %3}}, [%4], %5;"
    : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "l"(policy));
  return v;
}}
'''
  out,replaced=[],0
  for line in source.splitlines():
    if " = (*" in line and "data2_" in line and "uint4" in line and line.endswith(";"):
      lhs,rhs=line.rsplit(" = ",1);expr=rhs[:-1]
      if expr.startswith("(*") and expr.endswith(")"):
        line=f"{lhs} = tg_nv_ld_evict_last({expr[2:-1]});";replaced+=1
    out.append(line)
  if replaced == 0: raise RuntimeError(f"NV Flash K/V evict-last policy matched {name!r} but rewrote no wide loads")
  return helper+"\n"+"\n".join(out)


def _nv_fast_math_source(name:str, source:str) -> str:
  """Research lease for program-scoped NVRTC fast math.

  The marker changes the source/cache identity and is consumed by
  ``NVRTCCompiler``. Empty policy remains byte-identical.
  """
  programs = frozenset(x for x in os.environ.get("NV_FAST_MATH_PROGRAMS", "").split(",") if x)
  if not programs or not _nv_pdl_match(name, programs): return source
  return "#define TINYGRAD_NV_USE_FAST_MATH 1\n" + source


def _nv_min_blocks_source(name:str, source:str) -> str:
  """Research lease for CUDA's two-argument ``__launch_bounds__`` contract.

  The second argument lets ptxas spend registers up to the one-block residency
  boundary to expose per-thread ILP.  Keep this exact-name/prefix gated: it is
  a compiler scheduling contract, not a generally safe renderer default.
  """
  programs = frozenset(x for x in os.environ.get("NV_MIN_BLOCKS_PROGRAMS", "").split(",") if x)
  admitted = bool(NV_FLASH_LOAD_SCHEDULE) and name.startswith("flash_vec_llama_score_pv_")
  if not admitted and (not programs or not _nv_pdl_match(name, programs)): return source
  marker = "__launch_bounds__("
  start = source.find(marker)
  if start < 0: raise RuntimeError(f"NV min-blocks policy matched {name!r} but found no launch bounds")
  end = source.find(")", start + len(marker))
  if end < 0 or "," in source[start:end]:
    raise RuntimeError(f"NV min-blocks policy matched {name!r} but launch bounds were not single-argument")
  return source[:end] + ", 1" + source[end:]


class CUDARenderer(CStyleLanguage):
  supports_strict_after = True
  supports_load_schedule = True
  supports_region_load = True
  supports_region_load_bridge = True
  supports_const_restrict_pointer = True
  region_load_bridge_owns_barrier = False
  region_load_bridge_warp_fence = False
  region_load_bridge_loads_after_barrier = False
  region_load_bridge_group_words = 18

  def render_region_load_bridge(self, pairs:list[tuple[UOp,UOp]], order_dependency:UOp|None=None) -> tuple[list[str],list[str]]:
    if len(pairs) != 18: raise RuntimeError("CUDA region load bridge requires exactly 18 copies")
    def delta(a:UOp,b:UOp) -> int:
      d=(a-b).simplify()
      if d.vmin != d.vmax or not isinstance(d.vmin,int):
        raise RuntimeError("CUDA region load bridge requires constant affine INDEX deltas")
      return d.vmin
    def offset(x:int) -> str: return f"+{x}" if x > 0 else str(x) if x < 0 else ""
    global_base=pairs[0][0].src[0].src[0]
    local_base=pairs[0][1].src[0].src[0]
    if any(load.src[0].src[0] is not global_base or store.src[0].src[0] is not local_base for load,store in pairs):
      raise RuntimeError("CUDA region load bridge requires common direct GLOBAL and LOCAL bases")
    ref_global=pairs[0][0].src[0].src[1]
    global_deltas=[delta(load.src[0].src[1],ref_global) for load,_ in pairs]
    pairs=[pair for _,pair in sorted(zip(global_deltas,pairs),key=lambda x:x[0])]
    ref_global=pairs[0][0].src[0].src[1]
    ref_local=min(pairs,key=lambda pair:delta(pair[1].src[0].src[1],pairs[0][1].src[0].src[1]))[1].src[0].src[1]
    global_offsets=[delta(load.src[0].src[1],ref_global)*4 for load,_ in pairs]
    local_offsets=[delta(store.src[0].src[1],ref_local)*4 for _,store in pairs]
    if len(set(global_offsets)) != 18 or len(set(local_offsets)) != 18 or any(abs(x) >= 2**31 for x in (*global_offsets,*local_offsets)):
      raise RuntimeError("CUDA region load bridge requires 18 distinct signed-32-bit affine offsets")
    global_address=self[pairs[0][0].src[0]]
    local_ref=next(store for _,store in pairs if store.src[0].src[1] is ref_local)
    shared_address=self[local_ref.src[0]]
    names=[f"region_bridge_copy{i}" for i in range(18)]
    dep_expr=None
    if order_dependency is not None:
      dep_expr=(f"__float_as_uint({self[order_dependency]})" if order_dependency.dtype is dtypes.float else
                f"((unsigned int)({self[order_dependency]}))")
    def ordered_address(address_operand:str, dependency_operand:str) -> list[str]:
      return (["  .reg .u64 region_bridge_ordered_address;\\n\\t",
               "  .reg .u64 region_bridge_order_dependency;\\n\\t",
               f"  cvt.u64.u32 region_bridge_order_dependency, {dependency_operand};\\n\\t",
               f"  xor.b64 region_bridge_ordered_address, {address_operand}, region_bridge_order_dependency;\\n\\t",
               "  xor.b64 region_bridge_ordered_address, region_bridge_ordered_address, region_bridge_order_dependency;\\n\\t"]
              if dep_expr is not None else [])
    if self.region_load_bridge_owns_barrier:
      fused=["asm volatile(", '  "{\\n\\t"', '  ".reg .u32 region_bridge_copy<18>;\\n\\t"']
      fused += [f'  "{line}"' for line in ordered_address("%0","%2")]
      if self.region_load_bridge_warp_fence: fused += ['  "bar.warp.sync 0xffffffff;\\n\\t"']
      if self.region_load_bridge_loads_after_barrier: fused += ['  "bar.sync 0;\\n\\t"']
      group=self.region_load_bridge_group_words
      if not isinstance(group,int) or group < 1 or group > 18: raise RuntimeError("CUDA region load bridge group must be 1..18 words")
      if self.region_load_bridge_loads_after_barrier:
        for start in range(0,18,group):
          fused += [f'  "ld.global.nc.u32 region_bridge_copy{i}, [{"region_bridge_ordered_address" if dep_expr is not None else "%0"}{offset(global_offsets[i])}];\\n\\t"' for i in range(start,min(start+group,18))]
          fused += [f'  "st.shared.u32 [%1{offset(local_offsets[i])}], region_bridge_copy{i};\\n\\t"' for i in range(start,min(start+group,18))]
      else:
        fused += [f'  "ld.global.u32 region_bridge_copy{i}, [{"region_bridge_ordered_address" if dep_expr is not None else "%0"}{offset(off)}];\\n\\t"' for i,off in enumerate(global_offsets)]
        fused += ['  "bar.sync 0;\\n\\t"']
        fused += [f'  "st.shared.u32 [%1{offset(off)}], region_bridge_copy{i};\\n\\t"' for i,off in enumerate(local_offsets)]
      fused += ['  "}\\n"', '  :',
                f'  : "l"((unsigned long long)({global_address})), "r"((unsigned int)__cvta_generic_to_shared({shared_address}))'+
                (f', "r"({dep_expr})' if dep_expr is not None else ''),
                '  : "memory");']
      return fused,[]
    before=[f"unsigned int {', '.join(names)};", "asm volatile("]
    before += [f'  "{line}"' for line in ordered_address("%18","%19")]
    before += [f'  "ld.global.u32 %{i}, [{"region_bridge_ordered_address" if dep_expr is not None else "%18"}{offset(off)}];\\n\\t"' for i,off in enumerate(global_offsets)]
    before += ["  : "+", ".join(f'"=r"({name})' for name in names),
               f'  : "l"((unsigned long long)({global_address}))'+(f', "r"({dep_expr})' if dep_expr is not None else ''),
               '  : "memory");']
    after=["asm volatile("]
    after += [f'  "st.shared.u32 [%0{offset(off)}], %{i+1};\\n\\t"' for i,off in enumerate(local_offsets)]
    after += ["  :", f'  : "r"((unsigned int)__cvta_generic_to_shared({shared_address})), '+
              ", ".join(f'"r"({name})' for name in names), '  : "memory");']
    return before,after

  def render_const_restrict_pointer(self, u:UOp) -> str:
    if u.addrspace is not AddrSpace.GLOBAL or u.dtype not in {dtypes.int, dtypes.uint, dtypes.float} or u.dtype.vcount != 1:
      raise RuntimeError(f"CUDA const_restrict requires a scalar 32-bit GLOBAL parameter, got {u.dtype}")
    return f"const {self.render_scalar_dtype(u.dtype)} *__restrict__"

  def render_load_schedule(self, u:UOp, name:str) -> str:
    if len(u.src) != 2 or u.src[1].op is not Ops.AFTER or not isinstance(u.src[1].arg, LoadSchedule):
      raise RuntimeError("CUDA schedule_after requires one unmasked LOAD and one opaque phase token")
    idx, token = u.src
    if idx.addrspace is not AddrSpace.GLOBAL: raise RuntimeError("CUDA schedule_after requires an immutable GLOBAL LOAD")
    if u.dtype not in {dtypes.int, dtypes.uint, dtypes.float} or u.dtype.vcount != 1:
      raise RuntimeError(f"CUDA schedule_after requires a scalar 32-bit LOAD, got {u.dtype}")
    if len(idx.src) != 2 or idx.src[1].dtype not in {dtypes.weakint, dtypes.int, dtypes.uint} or idx.src[1].dtype.vcount != 1:
      raise RuntimeError("CUDA schedule_after requires one scalar 32-bit INDEX")
    dep=token.src[0]
    if dep.dtype not in {dtypes.int, dtypes.uint, dtypes.float} or dep.dtype.vcount != 1:
      raise RuntimeError(f"CUDA schedule_after requires a scalar 32-bit phase token, got {dep.dtype}")
    dep_expr=f"__float_as_uint({self[dep]})" if dep.dtype is dtypes.float else self[dep]
    index_expr, address=self[idx.src[1]], self[idx]
    if not index_expr or address.count(index_expr) != 1:
      raise RuntimeError("CUDA schedule_after cannot isolate the rendered scalar INDEX")
    index_name=f"{name}_schedule_index"
    ordered_address=address.replace(index_expr, index_name, 1)
    return (f'{self.render_type(idx.src[1])} {index_name} = {index_expr}; '
      f'asm volatile("xor.b32 %0, %0, %1; xor.b32 %0, %0, %1;" : "+r"({index_name}) : "r"({dep_expr}) : "memory"); '
      f'{self.render_type(u)} {name} = (*{ordered_address});')

  def render_strict_after(self, u:UOp) -> tuple[str, str]:
    if len(u.src) != 2: raise RuntimeError("strict_after requires exactly one dependency")
    value, dep = u.src
    if value.dtype not in {dtypes.weakint, dtypes.int, dtypes.uint} or value.dtype.vcount != 1:
      raise RuntimeError(f"CUDA strict_after requires a scalar 32-bit integer value, got {value.dtype}")
    if dep.dtype not in {dtypes.int, dtypes.uint, dtypes.float} or dep.dtype.vcount != 1:
      raise RuntimeError(f"CUDA strict_after requires a scalar 32-bit dependency, got {dep.dtype}")
    dep_expr = f"__float_as_uint({self[dep]})" if dep.dtype is dtypes.float else self[dep]
    name = f"strict_after_{sum(x.op is Ops.AFTER and isinstance(x.arg, StrictAfter) for x in self.r)}"
    statement = (f'{self.render_type(u)} {name} = {self[value]}; '
      f'asm volatile("xor.b32 %0, %0, %1; xor.b32 %0, %0, %1;" : "+r"({name}) : "r"({dep_expr}) : "memory");')
    return statement, name

  supports_post_barrier_regions = True
  global_max = (2147483647, 65535, 65535)
  local_max = (1024, 1024, 64)
  shared_max = 49152

  def __init__(self, target:Target, use_nvcc=False):
    super().__init__(target)
    from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler, NVCCCompiler
    iface, dev, arch = target.interface, target.device, target.arch
    self.compiler = (NVCCCompiler if use_nvcc else NVRTCCompiler)(arch, ptx=iface.startswith("MOCK") or dev == "CUDA", cache_key=dev.lower())
    self.tensor_cores = tc.get_cuda(arch)
    # CUDA warps are 32 lanes across every CUDA-capable device (the PTX model fixes warpSize at 32), and this
    # renderer's own warp_shfl_xor lowering was compiled and run at that width (TG1), including the Q4_K/Q6_K
    # fused-primitive probe on sm_120. Reporting a verified width is not the silent defaulting scope forbids
    # (target-capability-policy-decoupling-scope-20260730.md section 3.3): that rule bars inventing a value for
    # an unknown target, and this one is known. Same discipline as Metal's `self.wave_size = 32` (cstyle.py).
    self.wave_size = 32
    # Fused prefill attention is a renderer capability, not an nvcc-compiler
    # capability.  Both NVRTC (the production DEV=NV renderer) and NVCC lower
    # the same CUDA source after these typed native-attention ops are expanded.
    # Historically this binding lived only on NVCCRenderer, which made the
    # measured fast path depend on the undocumented DEV=NV:CC selection.
    if arch.split(":")[0] == "sm_120": _install_native_attention_bindings(self)

  kernel_typedef = 'extern "C" __global__ void __launch_bounds__({launch_bounds})'
  smem_prefix = "__shared__ __align__(16) "
  runtime_local_prefix = "extern __shared__ __align__({alignment}) "
  smem_prefix_for_cast = False
  barrier = "__syncthreads();"
  float4 = "make_float4"
  gep_arr_threshold = 8
  code_for_workitem = {"g": lambda x: f"blockIdx.{chr(120+int(x))}", "l": lambda x: f"threadIdx.{chr(120+int(x))}",
                       "i": lambda x: f"(blockIdx.{chr(120+int(x))}*blockDim.{chr(120+int(x))}+threadIdx.{chr(120+int(x))})"}
  # __shfl_xor_sync takes the lane mask directly (like Metal's simd_shuffle_xor) -- no per-lane address needed,
  # so `lane` is unused. Full-warp mask: every call site in this repo shuffles across the whole warp (WARP=32).
  warp_shfl_xor = staticmethod(lambda val, offset, lane: UOp(Ops.CUSTOMI, val.dtype, (val,), arg=f"__shfl_xor_sync(0xffffffffu, {{0}}, {offset})"))
  # Byte-address variant (fused-attention row softmax): the caller carries the source lane's register byte
  # address; CUDA cannot address registers by byte, but the address IS the source lane index times four, so
  # __shfl_sync's srcLane is `addr >> 2` -- correct for both the XOR butterfly and the half-wave broadcast.
  warp_bpermute = staticmethod(lambda addr, value: UOp(Ops.CUSTOMI, value.dtype, (addr, value),
    arg="__shfl_sync(0xffffffffu, {1}, (({0}) >> 2))"))
  # TG7: exp2f is a native CUDA device function, a one-liner. fdot2 has no native packed-fp16x2 dot-accumulate
  # CUDA builtin either, so it reuses the exact two-fp32-FMA substitute Metal's provider uses (cstyle.py):
  # CUDA's half2 exposes .x/.y and float() conversions just like MSL, so the identical template is the asset
  # being reused, not a CUDA-specific string. fp32 accumulate throughout, same as the AMD builtin's contract.
  exp2f = staticmethod(lambda x: UOp(Ops.CUSTOMI, x.dtype, (x,), arg="exp2f({0})"))
  fdot2 = staticmethod(lambda acc, a, b: UOp(Ops.CUSTOMI, dtypes.float32, (acc, a, b),
    arg="({0}) + float({1}.x) * float({2}.x) + float({1}.y) * float({2}.y)"))
  int8x4_dot = staticmethod(lambda acc, a, b: UOp(Ops.CUSTOMI, dtypes.int32, (acc, a, b),
    arg="__dp4a((int){1}, (int){2}, {0})"))
  native_fragment_x4 = staticmethod(lambda buffer,index: UOp(Ops.CUSTOMI, dtypes.uint32.vec(4), (buffer,index),
    arg="tg_ldmatrix_x4((const void*)({0}+{1}))"))
  native_fragment_x2 = staticmethod(lambda buffer,index: UOp(Ops.CUSTOMI, dtypes.uint32.vec(2), (buffer,index),
    arg="tg_ldmatrix_x2((const void*)((const char*)({0})+({1})*4))"))
  packed_i8_sub = staticmethod(lambda value,bias: UOp(Ops.CUSTOMI,dtypes.uint32,(value,bias),arg="__vsubss4({0},{1})"))
  native_fragment_bitcast = staticmethod(lambda value,dtype: UOp(Ops.CUSTOMI,dtype,(value,),arg=f"tg_bitcast<signed_char{dtype.count}>({{0}})"))
  code_for_op = { **CStyleLanguage.code_for_op,
    Ops.TRUNC: lambda x,dtype: f"htrunc({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"trunc({x})",
    Ops.SIN: lambda x,dtype: f"hsin({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"sin({x})",
    Ops.LOG2: lambda x,dtype: f"hlog2({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"log2({x})",
    Ops.EXP2: lambda x,dtype: f"hexp2({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"exp2({x})",
    Ops.SQRT: lambda x,dtype: f"hsqrt({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"sqrt({x})",
    Ops.RECIPROCAL: lambda x,dtype: f"hrcp({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"(1/{x})",
    Ops.PRECISE_DIV: lambda a,b,dtype: f"({a}/{b})", Ops.ROUND_AWAY: lambda x,dtype: f"roundf({x})" }
  type_map = {dtypes.bfloat16: "nv_bfloat16", dtypes.fp8e4m3: "__nv_fp8_e4m3", dtypes.fp8e5m2: "__nv_fp8_e5m2"}
  # CUDA ships native vector types (and make_ constructors) for 2/3/4 lanes of the common
  # scalars; half is native only as half2. Everything else gets the custom struct emitted
  # by render_vector_prefix.
  native_vector_types = {dtypes.char: "char", dtypes.uchar: "uchar", dtypes.short: "short", dtypes.ushort: "ushort",
                         dtypes.int: "int", dtypes.uint: "uint", dtypes.int64: "long", dtypes.uint64: "ulong",
                         dtypes.float: "float", dtypes.double: "double", dtypes.half: "half"}
  native_vector_lanes = {dtypes.half: (2,)}
  default_native_lanes = (2, 3, 4)

  def render_vector_dtype(self, dtype: DType, lanes: int) -> str:
    native = self.native_vector_types.get(dtype.scalar())
    if native is not None and lanes in self.native_vector_lanes.get(dtype.scalar(), self.default_native_lanes):
      return f"{native}{lanes}"
    return super().render_vector_dtype(dtype, lanes)

  extra_matcher = create_non_native_float_pats(dtypes.fp8s, casting=False) + PatternMatcher([
    (UPat(Ops.CAST, dtypes.fp8s, UPat.var("x", dtypes.fp8s), name='y'), lambda x,y: x.cast(dtypes.float).cast(y.dtype) if x.dtype!=y.dtype else None),
  ])
  string_rewrite = PatternMatcher([
    (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"tg_bitcast<{ctx.render_type(x)}>(({ctx.render_type(x.src[0])})({ctx[x.src[0]]}))"),
  ]) + base_rewrite

  def render_vector_prefix(self, dt:DType) -> str:
    vec, scal = self.render_vector_dtype(dt, dt.count), self.render_dtype(dt)
    names = _nms[:dt.count] if dt.count <= len(_nms) else [f"v{i}" for i in range(dt.count)]
    elems, header = ', '.join(names), ', '.join([f"{scal} {x}" for x in names])
    # nvcc rejects alignment values above 128, so cap there; wider structs stay correct,
    # just without their ideal alignment.
    align = min(dt.itemsize, 128)
    return f"struct __align__({align}) {vec} {{ {scal} {elems}; }}; __device__ {vec} make_{vec}({header}) {{ {vec} r={{{elems}}}; return r; }}"

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None):
    prefix = ["#define INFINITY (__int_as_float(0x7f800000))", "#define NAN (__int_as_float(0x7fffffff))",
              "template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }"]
    if any(u.op is Ops.CUSTOMI and isinstance(u.arg,str) and "tg_ldmatrix_x4(" in u.arg for u in uops):
      prefix.append('''__device__ __forceinline__ uint4 tg_ldmatrix_x4(const void *p) {
  uint4 r; asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
    : "=r"(r.x),"=r"(r.y),"=r"(r.z),"=r"(r.w) : "l"(p)); return r;
}''')
    if any(u.op is Ops.CUSTOMI and isinstance(u.arg,str) and "tg_ldmatrix_x2(" in u.arg for u in uops):
      prefix.append('''__device__ __forceinline__ uint2 tg_ldmatrix_x2(const void *p) {
  uint2 r; asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1},[%2];"
    : "=r"(r.x),"=r"(r.y) : "l"(p)); return r;
}''')
    if os.environ.get("NV_SPLIT_PHASE", "") not in ("", "0"):
      kernel = _nv_pdl_body_split_phase(function_name, kernel)
    else:
      kernel = _nv_pdl_body(function_name, kernel)
    # Buffer-argument dtypes count too: a kernel whose ONLY fp16/fp8/bf16 element is a `half*` parameter
    # (e.g. an fp16 KV cache stored from fp32 values) previously missed the header include and rendered
    # `half` undefined in the signature (NVRTC compile error). Body uops drive vector-prefix emission, so
    # keep that list first and append the param dtypes; scalar params are no-ops for both loops below.
    wmma_vector_dtypes=[]
    for _,_,dtype_in,dtype_out,_,_,upcast_axes,_ in wmma_args(uops):
      sizes=[prod(size for _,size in axes) for axes in upcast_axes]
      wmma_vector_dtypes += [dtype.vec(size) for dtype,size in zip((dtype_in,dtype_in,dtype_out),sizes)]
    used_dtypes = dedup([*uops_to_dtypes(uops), *wmma_vector_dtypes, *[u.dtype for _, (u, _) in bufs]])
    if any(dt.scalar() in dtypes.fp8s for dt in used_dtypes): prefix.append("#include <cuda_fp8.h>")
    if any(dt.scalar() == dtypes.half for dt in used_dtypes): prefix.append("#include <cuda_fp16.h>")
    if any(dt.scalar() == dtypes.bfloat16 for dt in used_dtypes): prefix.append("#include <cuda_bf16.h>")
    prefix += [self.render_vector_prefix(dt) for dt in used_dtypes if dt.count > 1 and
      (dt.scalar() not in self.native_vector_types or
       dt.count not in self.native_vector_lanes.get(dt.scalar(), self.default_native_lanes))]
    dt_map_in = { dtypes.float: "tf32", dtypes.half: "f16", dtypes.bfloat16: "bf16", dtypes.fp8e4m3: "e4m3",
                  dtypes.fp8e5m2: "e5m2", dtypes.char: "s8" }
    dt_map_out = { dtypes.float: "f32", dtypes.half: "f16", dtypes.int: "s32" }
    for name, (N, M, K), dtype_in, dtype_out, _, _, upcast_axes, _ in wmma_args(uops):
      upcast_sizes = [prod(size for _, size in upcast) for upcast in upcast_axes]
      wmma_dtypes = [self.render_vector_dtype(dtype, size) for dtype, size in zip([dtype_in, dtype_in, dtype_out], upcast_sizes)]
      n_operands = [size*dtype.itemsize//4 for dtype, size in zip([dtype_in, dtype_in, dtype_out], upcast_sizes)]
      operands = [f"%{i}" for i in range(sum(n_operands))]
      prefix.append(f"""__device__ {wmma_dtypes[2]} __{name}({wmma_dtypes[0]} a, {wmma_dtypes[1]} b, {wmma_dtypes[2]} c){{
  int *a_pk = (int *)(&a), *b_pk = (int *)(&b), *c_pk = (int *)(&c);
  asm("mma.sync.aligned.m{M}n{N}k{K}.row.col.{dt_map_out[dtype_out]}.{dt_map_in[dtype_in]}.{dt_map_in[dtype_in]}.{dt_map_out[dtype_out]}"
      "{{{", ".join(operands[:n_operands[2]])}}}, {{{", ".join(operands[n_operands[2]:n_operands[2]+n_operands[0]])}}},"
      "{{{", ".join(operands[-n_operands[1]:])}}}, {{{", ".join(operands[:n_operands[2]])}}};"
    : {", ".join([f'"+r"(c_pk[{i}])' for i in range(n_operands[2])])}
    : {", ".join([f'"r"(a_pk[{i}])' for i in range(n_operands[0])])}, {", ".join([f'"r"(b_pk[{i}])' for i in range(n_operands[1])])});
  return c;\n}}""")

    source = super().render_kernel(function_name, kernel, bufs, uops, prefix=prefix)
    return _nv_min_blocks_source(function_name,
      _nv_fast_math_source(function_name, _nv_flash_kv_evict_last_source(function_name,
        _nv_l2_q6_payload_source(function_name, _nv_l2_streaming_weight_source(function_name, source)))))

  def supported_dtypes(self):
    ver = int(self.target.arch[3:])
    return {d for d in super().supported_dtypes() if (d != dtypes.half or ver >= 53) and (d != dtypes.bfloat16 or ver >= 80)
            and (d not in dtypes.fp8_ocp or ver >= 89) and d not in dtypes.fp8_fnuz}

class NVCCRenderer(CUDARenderer):
  def __init__(self, target:Target):
    super().__init__(target, use_nvcc=True)
