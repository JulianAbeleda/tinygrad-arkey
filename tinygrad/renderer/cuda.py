from tinygrad.codegen.opt import tc
from tinygrad.dtype import DType, dtypes
from tinygrad.helpers import Target, prod
from tinygrad.renderer.cstyle import CStyleLanguage, base_rewrite, create_non_native_float_pats, uops_to_dtypes, wmma_args, _install_native_attention_bindings
from tinygrad.uop.ops import Ops, PatternMatcher, UPat, UOp

_nms = list("xyzwabcdefghijkl") + [f'v{i}' for i in range(16, 32)]

class CUDARenderer(CStyleLanguage):
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

  kernel_typedef = 'extern "C" __global__ void __launch_bounds__({launch_bounds})'
  smem_prefix = "__shared__ __align__(16) "
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
  code_for_op = { **CStyleLanguage.code_for_op,
    Ops.TRUNC: lambda x,dtype: f"htrunc({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"trunc({x})",
    Ops.SIN: lambda x,dtype: f"hsin({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"sin({x})",
    Ops.LOG2: lambda x,dtype: f"hlog2({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"log2({x})",
    Ops.EXP2: lambda x,dtype: f"hexp2({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"exp2({x})",
    Ops.SQRT: lambda x,dtype: f"hsqrt({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"sqrt({x})",
    Ops.RECIPROCAL: lambda x,dtype: f"hrcp({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"(1/{x})" }
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
    used_dtypes = uops_to_dtypes(uops)
    if any(dt.scalar() in dtypes.fp8s for dt in used_dtypes): prefix.append("#include <cuda_fp8.h>")
    if any(dt.scalar() == dtypes.half for dt in used_dtypes): prefix.append("#include <cuda_fp16.h>")
    if any(dt.scalar() == dtypes.bfloat16 for dt in used_dtypes): prefix.append("#include <cuda_bf16.h>")
    prefix += [self.render_vector_prefix(dt) for dt in used_dtypes if dt.count > 1 and
      (dt.scalar() not in self.native_vector_types or
       dt.count not in self.native_vector_lanes.get(dt.scalar(), self.default_native_lanes))]
    dt_map_in = { dtypes.float: "tf32", dtypes.half: "f16", dtypes.bfloat16: "bf16", dtypes.fp8e4m3: "e4m3", dtypes.fp8e5m2: "e5m2" }
    dt_map_out = { dtypes.float: "f32", dtypes.half: "f16" }
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

    return super().render_kernel(function_name, kernel, bufs, uops, prefix=prefix)

  def supported_dtypes(self):
    ver = int(self.target.arch[3:])
    return {d for d in super().supported_dtypes() if (d != dtypes.half or ver >= 53) and (d != dtypes.bfloat16 or ver >= 80)
            and (d not in dtypes.fp8_ocp or ver >= 89) and d not in dtypes.fp8_fnuz}

class NVCCRenderer(CUDARenderer):
  def __init__(self, target:Target):
    super().__init__(target, use_nvcc=True)
    if target.arch.split(":")[0] == "sm_120":
      # The fused-prefill-attention native matchers, bound on NVCC exactly as HIP binds them on gfx1100
      # (same shared install function -- see cstyle.py). Literal-first: the lane math is still AMD-shaped
      # until the fragment-model package (P3); this proves the NVCC path can consume the same expansions.
      _install_native_attention_bindings(self)
