from typing import Literal, Callable, cast
import math, sys, struct, re
from collections import defaultdict, Counter
from tinygrad.codegen.opt import tc
from tinygrad.uop.ops import GroupOp, Ops, UOp, PatternMatcher, UPat, range_str, axis_letters
from tinygrad.helpers import strip_parens, getenv, prod, dedup, Target, CPU_COUNT, IMAGE, FLOAT16
from tinygrad.dtype import ImageDType, dtypes, DType, PtrDType, AddrSpace, truncate, float_to_bf16
from tinygrad.renderer import Renderer
from tinygrad.codegen.late.devectorizer import no_vectorized_alu


def _render_arg_format(ctx, x:UOp) -> str:
  # CUSTOM/CUSTOMI args are str.format templates with positional {0},{1},... placeholders;
  # literal C braces in the body must be doubled ({{ }}). Surface a clear, actionable error if they aren't,
  # instead of a bare IndexError/ValueError from .format. Success path is byte-identical to x.arg.format(...).
  try:
    return x.arg.format(*[ctx[y] for y in x.src])
  except (IndexError, KeyError, ValueError) as e:
    raise RuntimeError(f"{x.op} arg failed to format ({type(e).__name__}: {e}); "
                       f"literal C braces must be doubled and placeholders must match len(src)={len(x.src)}. "
                       f"arg={x.arg!r}") from e

def _render_hip_wait(x:UOp) -> str:
  from tinygrad.codegen.opt.compiler_policies import WaitCount
  if not isinstance(x.arg, WaitCount): raise ValueError("HIP WAIT lowering requires a typed WaitCount")
  return f"__builtin_amdgcn_s_waitcnt({x.arg.simm16});"

def _render_hip_barrier(ctx, x:UOp) -> str:
  from tinygrad.codegen.opt.compiler_policies import WaveLDSFence
  if isinstance(x.arg,WaveLDSFence): return _render_hip_wait(x)
  if x.arg is not None: raise ValueError("HIP BARRIER has an unsupported typed payload")
  return ctx.barrier


base_rewrite = PatternMatcher([
  # local/reg buffers
  (UPat(Ops.BUFFER, name="x"), lambda ctx,x: ctx.render_buffer(x)),

  # range/if/endif
  (UPat(Ops.RANGE, name="x"),
   lambda ctx,x: f"for ({ctx.render_dtype(x.dtype)} {ctx[x]} = 0; {ctx[x]} < {ctx[x.src[0]]}; {ctx[x]}++) {{"),
  (UPat(Ops.IF, name="x"), lambda ctx,x: f"if ({ctx[x.src[0]]}) {{"),
  (UPat((Ops.ENDIF, Ops.END)), lambda ctx: "}"),

  # casting
  (UPat(Ops.CAST, name="x"), lambda ctx,x: f"__builtin_convertvector({ctx[x.src[0]]}, {ctx.render_type(x)})" \
    if x.max_numel() > 1 and x.addrspace is AddrSpace.REG else None),
  (UPat(Ops.CAST, name="x"), lambda ctx,x: f"({ctx.render_cast(x, ctx[x.src[0]])})"),
  (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"__builtin_bit_cast({ctx.render_type(x)}, ({ctx.render_type(x.src[0])})({ctx[x.src[0]]}))"),

  # GPU stuff
  (UPat(Ops.BARRIER), lambda ctx: ctx.barrier),
  (UPat(Ops.SPECIAL, name="x"), lambda ctx,x: f"{ctx.code_for_workitem[x.arg[0]](x.arg[-1])}; /* {(x.src[0]).render()} */"),

  # const
  (UPat(Ops.CONST, arg=math.inf, name="x"), lambda ctx, x: f"({ctx.render_cast(x, ctx.infinity)})"),
  (UPat(Ops.CONST, arg=-math.inf, name="x"), lambda ctx, x: f"({ctx.render_cast(x, f'-{ctx.infinity}')})"),
  (UPat(Ops.CONST, dtype=dtypes.floats, name="x"), lambda ctx,x: f"({ctx.render_cast(x, ctx.nan)})" if math.isnan(x.arg) else None),
  (UPat(Ops.CONST, dtype=dtypes.float, name="x"), lambda ctx,x: f"{x.arg}f"),
  (UPat(Ops.CONST, dtype=dtypes.int64, name="x"), lambda ctx,x: f"{x.arg}ll"),
  (UPat(Ops.CONST, dtype=dtypes.uint64, name="x"), lambda ctx,x: f"{truncate[x.dtype](x.arg)}ull"),
  (UPat(Ops.CONST, dtype=dtypes.uint32, name="x"), lambda ctx,x: f"{truncate[x.dtype](x.arg)}u"),
  (UPat(Ops.CONST, dtype=dtypes.bool, name="x"), lambda ctx,x: "1" if x.arg else "0"),
  # consts are rendered to larger type and casted
  (UPat(Ops.CONST, (*dtypes.fp8s, dtypes.bfloat16, dtypes.half), name="x"), lambda ctx,x: f"({ctx.render_cast(x, f'{x.arg}f')})"),
  (UPat(Ops.CONST, (dtypes.uint8, dtypes.uint16), name="x"), lambda ctx,x: f"({ctx.render_cast(x, f'{x.arg}u')})"),
  (UPat(Ops.CONST, (dtypes.int8, dtypes.int16), name="x"), lambda ctx,x: f"({ctx.render_cast(x, str(x.arg))})"),
  # default const render
  (UPat(Ops.CONST, name="x"), lambda ctx,x: str(x.arg)),

  # SHRINK/INDEX
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var('idx')), name="x"), lambda ctx,**kwargs: ctx.render_index(**kwargs)),
  # zero-dim INDEX (scalar access)
  (UPat(Ops.INDEX, src=(UPat.var("buf"),), name="x"), lambda ctx,buf,x: ctx[buf] if isinstance(buf.dtype, PtrDType) else ctx[buf]),
  (UPat(Ops.SHRINK, src=(UPat.var("buf"), UPat.var('idx'), UPat.cvar()), name="x"), lambda ctx,**kwargs: ctx.render_index(**kwargs)),
  (UPat(Ops.STACK, name="x"),
   lambda ctx,x: f"{ctx.float4.replace('float4', ctx.render_type(x))}" + \
                 f"{ctx.float4_style[0]}{','.join([ctx[y] for y in x.src])}{ctx.float4_style[1]}"),

  # load/store
  (UPat(Ops.LOAD, src=(UPat.var('bidx'),)), lambda ctx,bidx: f"({ctx.render_access(bidx)})"),
  (UPat(Ops.LOAD, src=(UPat.var("bidx"), UPat.var("var"), UPat.var("gate"))),
   lambda ctx,bidx,var,gate: f"({ctx[gate]}?{ctx.render_access(bidx)}:{ctx[var]})"),
  (UPat(Ops.STORE, src=(UPat.var('bidx'), UPat.var("var"))), lambda ctx,bidx,var: f"{ctx.render_access(bidx)} = {ctx[var]};"),

  # alu/gep
  (UPat(Ops.WMMA, name="x"), lambda ctx,x: f"__{x.arg[0]}({ctx[x.src[0]]}, {ctx[x.src[1]]}, {ctx[x.src[2]]})"),
  (UPat(GroupOp.ALU, name="x"), lambda ctx,x: ctx.code_for_op[x.op](
    *([strip_parens(ctx[v]) if v.op == x.op and x.op in {Ops.ADD, Ops.MUL, Ops.XOR, Ops.OR, Ops.AND} else ctx[v] for v in x.src]), x.dtype)),

  # custom passes through with format
  (UPat((Ops.CUSTOM, Ops.CUSTOMI), name="x"), lambda ctx,x: _render_arg_format(ctx, x)),
])

extra_pm = PatternMatcher([
  # devectorize any bools
  (UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.INDEX), dtype=dtypes.bool, name="alu"), no_vectorized_alu),
  # CAST (from bool) can't be vectorized
  (UPat(Ops.CAST, src=(UPat(dtype=dtypes.bool),), name="alu"), no_vectorized_alu),
  # WHERE can't be vectorized
  (UPat(Ops.WHERE, name="alu"), no_vectorized_alu),
])

_HIP_BPERMUTE_F32 = "__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute({0}, __builtin_bit_cast(unsigned int, {1})))"

def _hip_native_bpermute_max(x:UOp) -> UOp|None:
  if x.dtype != dtypes.float or len(x.src) != 2: return None
  peers = [s for s in x.src if s.op is Ops.CUSTOMI and s.dtype == dtypes.float and s.arg in ("bpermute", _HIP_BPERMUTE_F32)]
  if len(peers) != 1: return None
  return UOp(Ops.CUSTOMI, dtypes.float, x.src, "__builtin_fmaxf({0}, {1})")

def _hip_native_row_state(x:UOp) -> UOp|None:
  if not isinstance(x.arg,tuple) or not x.arg: return None
  if x.arg[0] == "amd_gfx1100_row_state_write_v1": return x.replace(arg="do {{ (void)({0}); }} while (0)")
  if x.arg[0] == "amd_gfx1100_row_state_read_v1" and x.src and x.src[0].op in {Ops.NOOP,Ops.CUSTOMI} and x.src[0].src:
    return x.src[0].src[0].after(*x.src[1:])
  return None

def _hip_expand_native_row_softmax(ctx, x:UOp) -> UOp:
  from tinygrad.renderer.isa.amd import expand_native_row_softmax_repack
  return expand_native_row_softmax_repack(ctx,x,native_state=False)

def _hip_expand_attention_loop_state(x:UOp) -> UOp:
  from tinygrad.uop.ops import AMDLoopStateSpec
  if not isinstance(x.arg, AMDLoopStateSpec): raise ValueError("HIP attention loop state is missing its typed ABI")
  x.arg.validate()
  if x.arg.access in {"init","write"}: return x.src[0]
  reg=x.src[0]; offset=x.arg.block*8+x.arg.lane if x.arg.role=="acc" else x.arg.lane
  addr=reg.after(*x.src[1:]).index(UOp.const(dtypes.weakint,offset))
  return addr.load()

def _hip_expand_loop_fragment(x:UOp) -> UOp:
  from tinygrad.renderer.isa.amd import expand_loop_fragment
  return expand_loop_fragment(x)

def _hip_expand_attention_output_drain(x:UOp) -> UOp:
  """Expand the typed native-output ABI to ordinary HIP SSA stores."""
  from tinygrad.uop.ops import AMDAttentionOutputDrainSpec
  if not isinstance(x.arg, AMDAttentionOutputDrainSpec): raise ValueError("HIP attention output drain is missing its typed ABI")
  x.arg.validate()
  grid=x.arg.grid
  if len(x.src) != (3+x.arg.blocks if grid is not None else 2+x.arg.blocks) or x.dtype != dtypes.void: raise ValueError("HIP attention output drain has malformed sources")
  out, *rest=x.src
  group, l, acc = (rest[0],rest[1],rest[2:]) if grid is not None else (None,rest[0],rest[1:])
  lane=UOp.special(32,"lidx0"); col=lane.alu(Ops.AND,UOp.const(dtypes.weakint,15)); half=lane.alu(Ops.SHR,UOp.const(dtypes.weakint,4))
  stores=[]
  for j in range(x.arg.blocks):
    for e in range(8):
      den=l.gep(e); recip=den.ne(UOp.const(dtypes.float,0)).where(UOp.const(dtypes.float,1)/den,UOp.const(dtypes.float,0))
      base=group*UOp.const(dtypes.weakint,2048) if group is not None else UOp.const(dtypes.weakint,0)
      dst=out.index(base+(UOp.const(dtypes.weakint,2*e)+half)*128+(j+x.arg.output_block_base)*16+col)
      stores.append(dst.store((acc[j].gep(e)*recip).cast(dtypes.half)))
  return UOp.group(*stores)

def _hip_expand_attention_stats_drain(x:UOp) -> UOp:
  from tinygrad.uop.ops import AMDAttentionStatsDrainSpec
  if not isinstance(x.arg,AMDAttentionStatsDrainSpec) or len(x.src) != 4: raise ValueError("HIP attention stats drain is malformed")
  x.arg.validate(); stats,group,m,l=x.src; lane=UOp.special(32,"lidx0"); half=lane.alu(Ops.SHR,UOp.const(dtypes.weakint,4)); stores=[]
  for e in range(8):
    base=group*UOp.const(dtypes.weakint,32)+(UOp.const(dtypes.weakint,2*e)+half)*UOp.const(dtypes.weakint,2)
    stores += [stats.index(base).store(m.gep(e)),stats.index(base+UOp.const(dtypes.weakint,1)).store(l.gep(e))]
  return UOp.group(*stores)

hip_native_repack_pm = PatternMatcher([
  (UPat(Ops.MAX, name="x"), _hip_native_bpermute_max),
  (UPat(Ops.CUSTOMI, name="x"), _hip_native_row_state),
  (UPat(Ops.CUSTOMI, name="x"), lambda x: x.replace(
    arg=_HIP_BPERMUTE_F32)
    if x.arg == "bpermute" and x.dtype == dtypes.float else None),
])

def create_non_native_float_pats(dts:tuple[DType, ...], casting:bool=True):
  patterns = PatternMatcher([
    (UPat(Ops.WHERE, src=(UPat.var("b"), UPat.var("x", dtype=dts), UPat.var("y", dtype=dts))),
     lambda b,x,y: UOp(Ops.WHERE, dtype=dtypes.float, src=(b,x.cast(dtypes.float),y.cast(dtypes.float))).cast(x.dtype)),
    (UPat(GroupOp.ALU, dtype=dts, name="x"),
     lambda x: UOp(x.op, dtypes.float, tuple(vv.cast(dtypes.float) for vv in x.src), x.arg).cast(x.dtype)),
    (UPat(GroupOp.ALU, dtypes.bool, name="alu", src=(UPat.var("x", dtype=dts), UPat.var("y", dtype=dts))),
     lambda alu,x,y: UOp(alu.op, dtypes.bool, (x.cast(dtypes.float), y.cast(dtypes.float)), alu.arg))])
  if casting:
    # add float intermediate casting
    patterns += PatternMatcher([
      (UPat(Ops.CAST, dts, (UPat.var("x"),), name="y"), lambda x,y: x.cast(dtypes.float).cast(y.dtype) if x.dtype!=dtypes.float else None),
      (UPat(Ops.CAST, name="x", src=(UPat.var("y", dts),)), lambda x,y: y.cast(dtypes.float).cast(x.dtype) if x.dtype!=dtypes.float else None)])
  return patterns

def cast_float_to_bf16(x: UOp) -> UOp:
  assert x.dtype == dtypes.float, "cast float -> bf16 must start with float"
  x = x.bitcast(dtypes.uint)
  x = (-x & 0x7f800000).ne(0).where(x + ((x >> 16) & 1) + 0x7fff, (x & 0xffff).ne(0).where((x | 0x10000), x))
  return (x >> 16).cast(dtypes.ushort).bitcast(dtypes.bfloat16)

# manual bfloat16 casting patterns (shared between LLVM, Clang, and AMD renderers to avoid compiler intrinsics)
pm_manual_bf16_cast = PatternMatcher([
  (UPat(Ops.CAST, dtypes.float, (UPat.var("x", dtypes.bfloat16),)),
   lambda x: (x.bitcast(dtypes.ushort).cast(dtypes.uint)<<16).bitcast(dtypes.float)),
  (UPat(Ops.CAST, dtype=dtypes.bfloat16, src=(UPat.var("x", dtype=dtypes.float),)), cast_float_to_bf16),
])

def uops_to_dtypes(uops:list[UOp]) -> list[DType]:
  ret = []
  seen = set()
  for u in uops:
    if u.addrspace in (AddrSpace.REG, None) and u.dtype != dtypes.void and u._shape is not None and (key:=(u.dtype, u.max_numel())) not in seen:
      # TODO: this eventually needs to be removed
      ret.append(u.dtype.vec(u.max_numel()))
      seen.add(key)
  return ret

# (name, dims, dtype_in, dtype_out, device, threads, upcast_axes, reduce_axes)
def wmma_args(uops:list[UOp]):
  return dedup((uop.arg[0], uop.arg[1], uop.arg[2], uop.dtype.scalar(), *(uop.arg[4:8])) for uop in uops if uop.op is Ops.WMMA)

class CStyleLanguage(Renderer):
  new_style = True
  kernel_typedef: str = "void"
  buffer_prefix: str = ""
  buffer_suffix: str = ""
  smem_align: str = ""
  smem_prefix: str = ""
  smem_prefix_for_cast: bool = True
  arg_int_prefix: str = "const int"
  barrier: str = ""
  code_for_workitem: dict[Literal["g", "l", "i"], Callable] = {}
  extra_args: list[str] = []
  float4: str|None = None
  float4_style: tuple[str, str] = ('(', ')')
  gep_arr_threshold: int = 4
  type_map: dict[DType, str] = {}
  infinity: str = "INFINITY"
  nan: str = "NAN"
  code_for_op: dict = {
    Ops.SQRT: lambda x,dtype: f"sqrt({x})", Ops.RECIPROCAL: lambda x,dtype: f"(1/{x})", Ops.NEG: lambda x,dtype: f"-{x}",
    Ops.EXP2: lambda x,dtype: f"exp2({x})", Ops.LOG2: lambda x,dtype: f"log2({x})", Ops.SIN: lambda x,dtype: f"sin({x})",
    Ops.TRUNC: lambda x,dtype: f"trunc({x})",
    Ops.AND: lambda a,b,dtype: f"({a}&{b})", Ops.XOR: lambda a,b,dtype: f"({a}^{b})", Ops.OR: lambda a,b,dtype: f"({a}|{b})",
    Ops.ADD: lambda a,b,dtype: f"({a}+{b})", Ops.SUB: lambda a,b,dtype: f"({a}-{b})", Ops.MUL: lambda a,b,dtype: f"({a}*{b})",
    Ops.CMOD: lambda a,b,dtype: f"({a}%{b})", Ops.CDIV: lambda a,b,dtype: f"({a}/{b})", Ops.CMPNE: lambda a,b,dtype: f"({a}!={b})",
    Ops.SHR: lambda a,b,dtype: f"({a}>>{b})", Ops.SHL: lambda a,b,dtype: f"({a}<<{b})", Ops.CMPLT: lambda a,b,dtype: f"({a}<{b})",
    Ops.WHERE: lambda a,b,c,dtype: f"({a}?{b}:{c})", Ops.CMPEQ: lambda a,b,dtype: f"({a}=={b})"}

  string_rewrite = base_rewrite
  extra_matcher = extra_pm

  def render_kernel(self, function_name:str, kernel:list[str], bufs:list[tuple[str,tuple[UOp,bool]]], uops:list[UOp], prefix=None) -> str:
    tmp = ""
    if any(isinstance(u.dtype, ImageDType) for _,(u,_) in bufs):
      tmp = "const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"
    buftypes = [(name, self._render_dtype(u.dtype, sz=1, addrspace=u.addrspace, mutable=mutable)+self.buffer_suffix \
                 if u.addrspace == AddrSpace.GLOBAL else self.arg_int_prefix if u.dtype == dtypes.int else None) for name,(u,mutable) in bufs]
    local_dims = [u.src[0] for u in uops if u.op is Ops.SPECIAL and u.arg[0] == "l"]
    launch_bounds = prod([d.vmax for d in local_dims])
    prg = ''.join([f"{self.kernel_typedef.format(launch_bounds=launch_bounds)} {function_name}(",] +
    [', '.join([f'{t} {name}' for name,t in buftypes] + self.extra_args)] +
    [") {\n" + tmp] + ['\n'.join(kernel), "\n}"])
    return prg if prefix is None else "\n".join(prefix)+f"\n{prg}"

  def render_index(self, x:UOp, buf:UOp, idx:UOp):
    if buf.addrspace == AddrSpace.REG and buf.op not in {Ops.AFTER, Ops.BUFFER}:
      # this is lane access in C
      assert idx.op is Ops.CONST, f"{idx.op} must be CONST"
      return self[buf]+(f"[{idx.arg}]" if buf.max_numel() > self.gep_arr_threshold else f".{'xyzwabcd'[idx.arg]}")
    ptr = f"({self[buf]}+{strip_parens(self[idx]) if idx.arg == Ops.ADD else self[idx]})"
    if buf.addrspace != AddrSpace.REG: return ptr
    # REG buffers have no LOAD, so the access is rendered at the INDEX. the cast handles vector access, same as render_access
    return f"(*(({self.render_type(x)}*)({ptr})))" if x.max_numel() > 1 else f"(*{ptr})"

  def render_buffer(self, x:UOp):
    shp = x.src[0].as_shape
    lanes = 1
    prefix = f"{self.smem_align}{self.smem_prefix}" if x.addrspace == AddrSpace.LOCAL else ""
    suffix = f"[{shp[0]}]" if len(shp) else ""
    return f"{prefix}{self._render_dtype(x.dtype, sz=lanes)} {self[x]}{suffix};"

  def _render_dtype(self, dtype:DType, sz:int=1, addrspace=AddrSpace.REG, mutable=True):
    if isinstance(dtype, ImageDType): return f"{'write_only' if mutable else 'read_only'} image2d_t"
    prefix, suffix = "", ""
    if addrspace in (AddrSpace.LOCAL, AddrSpace.GLOBAL):
      if addrspace == AddrSpace.LOCAL and self.smem_prefix_for_cast: prefix = self.smem_prefix
      if addrspace == AddrSpace.GLOBAL: prefix = self.buffer_prefix
      suffix = "*"
    if sz > 1:
      return prefix + self.type_map.get(scalar:=dtype.scalar(), scalar.name).replace(" ", "_") + str(sz) + suffix
    return prefix + self.type_map.get(scalar:=dtype.scalar(), scalar.name) + suffix

  def render_type(self, u:UOp): return self._render_dtype(u.dtype, u.max_numel(), u.addrspace)
  def render_access(self, u:UOp):
    if u.addrspace in (AddrSpace.GLOBAL, AddrSpace.LOCAL):
      if u.max_numel() > 1: return f"*(({self.render_type(u)})({self[u]}))"
      else: return f"*{self[u]}"
    return self[u]
  def render_cast(self, u:UOp, val:str) -> str: return f"({self.render_type(u)})({val})"

  # LEGACY
  def render_dtype(self, dt:DType, mutable=True) -> str:
    return self._render_dtype(dt, dt.count, dt.addrspace if isinstance(dt, PtrDType) else AddrSpace.REG)

  def __getitem__(self, key): return self.r[key]  # hacky helper
  def _render(self, uops:list[UOp]) -> tuple[str, list[str], list[tuple[str,tuple[UOp,bool]]]]:
    r: dict[UOp, str] = {}
    self.r = r

    child_count = Counter(v for ru in uops for v in ru.src)
    # find which PARAMs are stored to with a single toposort
    writable_params = {u for u in UOp.sink(*[u.src[0] for u in uops if u.op is Ops.STORE]).toposort(lambda u: u.op != Ops.END) if u.op is Ops.PARAM}
    bufs: dict[UOp, tuple[str, tuple[UOp, bool]]] = {}
    kernel = []
    depth = 1
    c: defaultdict[str, int] = defaultdict(int)
    name = "test"
    for u in uops:
      if u.op in {Ops.NOOP, Ops.GROUP}: continue
      if u.op == Ops.STACK and len(u.src) == 0: continue
      if u.op is Ops.AFTER:
        r[u] = r[u.src[0]]
        continue
      if u.op is Ops.SINK:
        if u.arg is not None: name = u.arg.function_name
        continue
      if u.op in (Ops.PARAM, Ops.DEFINE_VAR):
        if u.op is not Ops.PARAM: r[u] = u.arg[0]
        elif isinstance(u.dtype, ImageDType): r[u] = f"data{u.arg.slot}_{u.dtype.shape[0]}x{u.dtype.shape[1]}"
        else: r[u] = f"data{u.arg.slot}_{sz}" if (sz:=u.max_numel()) > 0 else f"data{u.arg.slot}"
        bufs[u] = (r[u], (u, u in writable_params))
        continue

      # naming
      prefix = None
      if u.op is Ops.SPECIAL: r[u] = u.arg
      elif u.op is Ops.RANGE: r[u] = f"{axis_letters[u.arg[-1]]}idx"+range_str(u)
      else:
        prefix = {Ops.WMMA: "wmma", Ops.DEFINE_LOCAL: "temp", Ops.CONST: "const", Ops.BUFFER: "buf",
                  Ops.CAST: "cast", Ops.BITCAST: "cast", Ops.GEP: "gep", Ops.STACK: "cast",
                  Ops.INDEX: "bidx", Ops.DEFINE_REG: "acc", Ops.LOAD: "val"}.get(u.op, "alu")
        r[u] = f"{prefix}{c[prefix]}"

      l = cast(str, self.string_rewrite.rewrite(u, ctx=self))
      assert l is not None, f"failed to render {u.op} {u.dtype} {[(x.op,x.dtype) for x in u.src]} {u.arg}"

      if u.op in {Ops.ENDIF, Ops.END}: depth -= 1
      if (u.op is not Ops.CAST or u.dtype.vcount == 1) and (u.op in {Ops.CONST, Ops.GEP, Ops.INDEX, Ops.SHRINK, Ops.CUSTOMI} or \
        (u.op is Ops.LOAD and u.src[0].addrspace == AddrSpace.REG) or \
        (u.op is Ops.CAST and u.addrspace in (AddrSpace.GLOBAL, AddrSpace.LOCAL)) or \
        (u.op in {Ops.STACK, *(GroupOp.ALU-{Ops.WHERE}), Ops.CAST, Ops.BITCAST} and child_count[u] == 1 and not getenv("EXPAND_SSA"))):
        r[u] = l
      else:
        if u.op not in {Ops.RANGE, Ops.DEFINE_LOCAL, Ops.STORE, Ops.DEFINE_REG, Ops.BUFFER} and u.dtype != dtypes.void:
          l = f"{self.render_type(u)} {r[u]} = {l}" + (";" if u.op is not Ops.SPECIAL else "")
        kernel.append("  "*depth + l)
        if prefix: c[prefix] += 1  # if it was used, increment
      if u.op in {Ops.IF, Ops.RANGE}: depth += 1
    del self.r

    # NOTE: this relies on bufs dict preserving order
    return (name, kernel, list(bufs.values()))
  def render(self, uops:list[UOp]) -> str: return self.render_kernel(*self._render(uops), uops)

class ClangRenderer(CStyleLanguage):
  float4 = "(float4)"
  float4_style = ('{', '}')
  gep_arr_threshold = 0
  has_local = False
  has_threads = bool(getenv("THREADS", 1))
  global_max = (CPU_COUNT.value, 0, 0)
  infinity = "__builtin_inff()"
  nan = '__builtin_nanf("")'

  # language options
  buffer_suffix = " restrict"
  type_map = {dtypes.bool:"_Bool", dtypes.half:"__fp16"}
  code_for_op = {**({k:v for k,v in CStyleLanguage.code_for_op.items() if k not in [Ops.EXP2, Ops.SIN, Ops.LOG2, Ops.TRUNC, Ops.RECIPROCAL]}),
                 Ops.SQRT: lambda x,dtype: f"__builtin_sqrt({x})" if dtype == dtypes.float64 else f"__builtin_sqrtf({x})",
                 Ops.TRUNC: lambda x,dtype: f"__builtin_trunc({x})" if dtype == dtypes.float64 else f"__builtin_truncf({x})",
                 Ops.FDIV: lambda a,b,dtype: f"({a}/{b})"}

  # LLVM legalizes double => half/bf16 cast on systems that don't support it natively (like x86 cpus without AVX512-FP16) into a compiler-rt libcall.
  # there is also no native bfl16 <-> fp16 conversion on those CPUs
  extra_matcher = PatternMatcher([(UPat.var("x", dtypes.float64).cast(dtypes.float16), lambda x: x.cast(dtypes.float32).cast(dtypes.float16)),
                                 (UPat.var("x", dtypes.float64).cast(dtypes.bfloat16), lambda x: x.cast(dtypes.float32).cast(dtypes.bfloat16)),
                                 (UPat.var("x", dtypes.bfloat16).cast(dtypes.float16), lambda x: x.cast(dtypes.float32).cast(dtypes.float16)),
    (UPat((Ops.SQRT, Ops.TRUNC), name="alu"), no_vectorized_alu)]) + create_non_native_float_pats((dtypes.bfloat16,)) + pm_manual_bf16_cast + \
    CStyleLanguage.extra_matcher

  if sys.platform == 'win32':
    kernel_typedef = "__attribute__((ms_abi)) void"
  def render_vector_prefix(self, dt:DType) -> str:
    # round (down) to power of two (this is actually the default clang behavior)
    alignment = 2**int(math.log2(dt.itemsize)) if getenv("ALIGNED", 1) and not dtypes.is_bool(dt) else 1
    return f"typedef {self.render_dtype(dt.scalar())} {self.render_dtype(dt)} __attribute__((aligned({alignment}),ext_vector_type({dt.count})));"

  def _render_defines(self, uops) -> list[str]: return [self.render_vector_prefix(dt) for dt in uops_to_dtypes(uops) if dt.count > 1]
  def _render_body(self, function_name, kernel, bufs, uops, pref=None) -> str: return super().render_kernel(function_name, kernel, bufs, uops, pref)
  def _render_entry(self, function_name:str, bufs:list[tuple[str,tuple[UOp,bool]]]) -> str: return ""

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
    defines = '\n'.join(self._render_defines(uops))
    return defines + "\n" + self._render_body(function_name, kernel, bufs, uops, prefix) + "\n" + self._render_entry(function_name, bufs)

  def supported_dtypes(self):
    return {d for d in super().supported_dtypes() if (d != dtypes.bfloat16 or self.target.arch.startswith(("x86", "arm"))) and d not in dtypes.fp8s}

  def __init__(self, target:Target):
    super().__init__(target)
    from tinygrad.runtime.support.compiler_cpu import ClangCompiler
    self.compiler = ClangCompiler(target.arch.split(","))

_nms = list("xyzwabcdefghijkl") + [f'v{i}' for i in range(16, 32)]

def fp8_index(dtype: DType): return (dtypes.fp8e4m3, dtypes.fp8e5m2).index(dtype.scalar())
def _ocml(op): return lambda x,dtype: f"__ocml_{op}_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})"

class HIPRenderer(CStyleLanguage):
  local_store_vector_widths = {dtypes.half: (8, 4, 2)}
  local_store_requires_static_alignment = False
  shared_max = 65536
  # NOTE: this is only really needed on gfx12, even though gfx11 reports the same limitation
  global_max = (2147483647, 65535, 65535)
  global_prod_max = (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)

  @staticmethod
  def is_cdna(arch): return arch.split(":")[0] in {"gfx942", "gfx950"}
  @staticmethod
  def is_cdna4(arch): return arch.split(":")[0] == "gfx950"
  def __init__(self, target:Target, use_hipcc=False): # gfx942 => MI300, gfx1100 => RX 7900, gfx1201 => RX 9700
    super().__init__(target)
    from tinygrad.runtime.support.compiler_amd import HIPCompiler, HIPCCCompiler
    self.compiler, self.tensor_cores = (HIPCCCompiler if use_hipcc else HIPCompiler)(target.arch), tc.get_amd(target.arch)
    if not self.is_cdna4(target.arch): self.extra_matcher += pm_manual_bf16_cast + extra_pm
    if target.arch.split(":")[0] == "gfx1100":
      # Exact native attention loop address expressions retain weakint until HIP source rendering.
      self.type_map = {**self.type_map, dtypes.weakint:"int"}
      # The scheduler-owned expansion is shared with the native ISA renderer;
      # HIP only supplies source spelling for its existing bpermute marker.
      from tinygrad.renderer.isa.amd import native_repack_matcher
      from tinygrad.renderer.isa.amd import native_state_lane_matcher
      self.native_repack_matcher = PatternMatcher([(UPat(Ops.AMD_ATTENTION_OUTPUT_DRAIN,name="x"), _hip_expand_attention_output_drain),
        (UPat(Ops.AMD_ATTENTION_STATS_DRAIN,name="x"), _hip_expand_attention_stats_drain),
        (UPat(Ops.AMD_ATTENTION_LOOP_STATE,name="x"), _hip_expand_attention_loop_state),
        (UPat(Ops.AMD_ROW_SOFTMAX_REPACK,name="x"), _hip_expand_native_row_softmax)]) + native_repack_matcher + \
        PatternMatcher([(UPat(Ops.MAX, name="x"), _hip_native_bpermute_max)])
      self.native_state_lane_matcher = native_state_lane_matcher
      self.native_state_lane_matcher = PatternMatcher([(UPat(Ops.AMD_ATTENTION_LOOP_STATE,name="x"), _hip_expand_attention_loop_state)]) + native_state_lane_matcher
      self.native_loop_fragment_matcher = PatternMatcher([(UPat(Ops.AMD_PACKED_FRAGMENT_LOAD,name="x"), _hip_expand_loop_fragment)])
      self.extra_matcher += hip_native_repack_pm
    if self.is_cdna(target.arch):
      self.string_rewrite = PatternMatcher([
        (UPat(Ops.WMMA, name="x"), lambda ctx,x: f"__{x.arg[0]}({ctx[x.src[0]]}, {ctx[x.src[1]]}, {ctx[x.src[2]]},"
          f" {fp8_index(x.src[0].dtype)}, {fp8_index(x.src[0].dtype)}, 0, 0, 0, 0)" if x.arg[1][2] == 128 else None),
        (UPat(Ops.WMMA, name="x"), lambda ctx,x: f"__{x.arg[0]}({ctx[x.src[0]]}, {ctx[x.src[1]]}, {ctx[x.src[2]]}, 0, 0, 0)"),
        (UPat(Ops.CONST, dtypes.fp8s, name="x"), lambda ctx,x: f"f32_to_fp8({ctx.nan}, {fp8_index(x.dtype)})" if math.isnan(x.arg) else None),
        (UPat(Ops.CONST, dtypes.fp8s, arg=math.inf, name="x"), lambda ctx,x: f"f32_to_fp8({ctx.infinity}, {fp8_index(x.dtype)})"),
        (UPat(Ops.CONST, dtypes.fp8s, arg=-math.inf, name="x"), lambda ctx,x: f"f32_to_fp8(-{ctx.infinity}, {fp8_index(x.dtype)})"),
        (UPat(Ops.CONST, dtypes.fp8s, name="x"), lambda ctx,x: f"f32_to_fp8({x.arg}f, {fp8_index(x.dtype)})"),
        (UPat(Ops.CAST, dtypes.fp8s, (UPat(dtype=dtypes.float),), name="x",),
          lambda ctx,x: f"f32_to_fp8({ctx[x.src[0]]}, {fp8_index(x.dtype)})"),
        (UPat(Ops.CAST, dtypes.float, (UPat.var("y", dtypes.fp8s),), name="x",),
          lambda ctx,x,y: f"__builtin_amdgcn_cvt_f32_{('fp8', 'bf8')[fp8_index(y.dtype)]}((unsigned int){ctx[x.src[0]]}, 0)"),
      ]) + base_rewrite

  # https://clang.llvm.org/docs/AttributeReference.html#amdgpu-flat-work-group-size
  # NOTE: this makes hlb_cifar10 twice as fast, there may be more gains in tweaking these parameters
  kernel_typedef = 'extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, {launch_bounds})))'
  code_for_workitem = {"g": lambda x: f"__ockl_get_group_id({x})", "l": lambda x: f"__ockl_get_local_id({x})",
                       "i": lambda x: f"(__ockl_get_group_id({x})*__ockl_get_local_size({x})+__ockl_get_local_id({x}))"}
  code_for_op = {**CStyleLanguage.code_for_op, Ops.TRUNC: _ocml("trunc"), Ops.SIN: _ocml("sin"),
                 Ops.LOG2: _ocml("log2"), Ops.EXP2: _ocml("exp2"), Ops.SQRT: _ocml("sqrt")}
  smem_prefix = "__attribute__((shared, aligned(16)))"
  smem_prefix_for_cast: bool = False
  barrier = '__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");' + '__builtin_amdgcn_s_barrier();' + \
            '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");'
  string_rewrite = PatternMatcher([(UPat(Ops.WAIT, name="x"), _render_hip_wait),
                                   (UPat(Ops.BARRIER, name="x"), _render_hip_barrier)]) + base_rewrite
  float4 = "make_float4"
  type_map = {dtypes.bfloat16: "hip_bfloat16", dtypes.fp8e4m3: "hip_fp8", dtypes.fp8e5m2: "hip_bf8"}
  extra_matcher = create_non_native_float_pats((dtypes.bfloat16, *dtypes.fp8s)) + PatternMatcher([
    (UPat(Ops.WMMA, name="x", dtype=dtypes.float.vec(4)),
      lambda x: UOp(Ops.WMMA, x.dtype, (x.src[0].bitcast(dtypes.uint64), x.src[1].bitcast(dtypes.uint64),
        x.src[2]), (*x.arg,)) if x.src[0].dtype in (dtypes.fp8e4m3.vec(8), dtypes.fp8e5m2.vec(8)) else None),
    # bfloat16 constant casting
    (UPat.cvar('x', dtypes.bfloat16), lambda x: cast_float_to_bf16(UOp.const(dtypes.float, x.arg))),
  ])

  def asm(self, prg:UOp, lin:UOp) -> bytes:
    from tinygrad.renderer.amd.elf import assemble_linear
    return assemble_linear(prg, lin, self.target.arch)

  def render_vector_prefix(self, dtype:DType) -> str:
    vec, scal = self.render_dtype(dtype), self.render_dtype(dtype.scalar())
    names = _nms[:dtype.count] if dtype.count <= len(_nms) else [f"v{i}" for i in range(dtype.count)]
    return f"typedef {scal} {vec} __attribute__((ext_vector_type({dtype.count})));\nstatic inline __attribute__((device)) "+ \
           f"{vec} make_{vec}({', '.join([f'{scal} {x}' for x in names])}) {{ return {{ {', '.join(names)} }}; }}"

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
    prefix, ockl = [], []
    # gated DP4A helper: emit the schedulable udot4 device helper only when a CUSTOM body references _dp4a
    # (so the builtin's required target("dot-insts") attr lives on the helper, not the generated kernel).
    # match an actual `_dp4a(` call (not a longer identifier like my_dp4a) so we don't emit a stray helper
    if any(u.op in (Ops.CUSTOM, Ops.CUSTOMI) and isinstance(u.arg, str) and re.search(r"(?<!\w)_dp4a\s*\(", u.arg) for u in uops):
      prefix.append('__attribute__((device)) __attribute__((target("dot-insts"))) unsigned int '
                    '_dp4a(unsigned int a, unsigned int b, unsigned int c){ return __builtin_amdgcn_udot4(a, b, c, false); }')
    type_map = { dtypes.bfloat16: "bf16", dtypes.float: "f32", dtypes.half: "f16", dtypes.fp8e4m3: "_fp8_fp8", dtypes.fp8e5m2: "_bf8_bf8" }
    used_dtypes = uops_to_dtypes(uops)
    if any(u.op is Ops.CONST and not math.isfinite(u.arg) for u in uops) or \
       any(u.op in {Ops.CUSTOM,Ops.CUSTOMI} and "INFINITY" in str(u.arg) for u in uops):
      prefix += ["#define INFINITY (__builtin_inff())", "#define NAN (__builtin_nanf(\"\"))"]
    if any(u.op is Ops.SPECIAL for u in uops):
      prefix.append("typedef long unsigned int size_t;")
      ockl = [(f"__ockl_get_{name}", "unsigned int", "size_t", "const") for name in ["local_id", "group_id", "local_size"]]
    ocml_ops = {Ops.EXP2: ("exp2", "pure"), Ops.LOG2: ("log2", "pure"), Ops.SQRT: ("sqrt", "const"), Ops.SIN: ("sin", ""), Ops.TRUNC: ("trunc", "")}
    ocml = [(f"__ocml_{ocml_ops[op][0]}_f{dt.bitsize}", dt.name, dt.name, ocml_ops[op][1])
      for op, dt in dedup((u.op, u.dtype.scalar()) for u in uops) if op in ocml_ops and dt in (dtypes.half, dtypes.float, dtypes.double)]
    if any(dt.scalar() == dtypes.bfloat16 for dt in used_dtypes):
      prefix.append(f"typedef {'__bf16' if self.is_cdna4(self.target.arch) else 'unsigned short'} hip_bfloat16;")
    def _has_half_dtype(dt:DType) -> bool:
      return dt.scalar() == dtypes.half or (isinstance(dt, PtrDType) and dt.base.scalar() == dtypes.half)
    if any(_has_half_dtype(dt) for dt in used_dtypes) or any(_has_half_dtype(u.dtype) for _,(u,_) in bufs):
      prefix.append("#define half _Float16")
    if any(dt.scalar() in dtypes.fp8s for dt in used_dtypes):
      prefix += ["typedef unsigned char hip_bf8;", "typedef unsigned char hip_fp8;"]
    if any((u.op is Ops.CAST and u.dtype in dtypes.fp8s and u.src[0].dtype == dtypes.float) or
           (u.op is Ops.CONST and u.dtype in dtypes.fp8s) for u in uops):
      prefix.append("""static inline __attribute__((device)) unsigned char f32_to_fp8(float v, int is_bf8) {
  v = (((*(unsigned*)&v)&0x7F800000)!=0x7F800000)?__builtin_amdgcn_fmed3f(v,is_bf8?57344.0f:448.0f,is_bf8?-57344.0f:-448.0f) : v;
  return (unsigned char)(is_bf8?__builtin_amdgcn_cvt_pk_bf8_f32(v,v,0,false):__builtin_amdgcn_cvt_pk_fp8_f32(v,v,0,false));\n}""")
    prefix += [f'extern "C" __attribute__((device{f", {atr}" if atr else ""})) {dto} {meth}({dti});' for meth,dti,dto,atr in ockl+ocml]
    prefix += [self.render_vector_prefix(dt) for dt in used_dtypes if dt.count > 1]

    _seen_wmma: set[str] = set()
    for name, (N, M, K), dtype_in, dtype_out, _, _, _, _ in wmma_args(uops): # TODO: handle TCs f32_bf16 and bf16_bf16 w/ wrapper
      # wmma_args dedups on the full arg tuple incl. upcast_axes (fresh range-ids per SHAPED_WMMA call), so multiple
      # WMMA ops of the SAME type but different range-ids yield repeat entries -> the name-keyed helper/typedef would be
      # emitted (and C-redefined) more than once. The helper body is fully determined by `name`, so dedup on it.
      if name in _seen_wmma: continue
      _seen_wmma.add(name)
      if self.is_cdna(self.target.arch):
        if (N, M, K) == (16, 16, 16): type_map[dtypes.bfloat16] = 'bf16_1k'
        elif (N, M, K) == (16, 16, 32): type_map = {**type_map, dtypes.bfloat16: "_bf16", dtypes.half: "_f16"}
        elif (N, M, K) == (16, 16, 128): type_map = {**type_map, dtypes.fp8e4m3: "_f8f6f4", dtypes.fp8e5m2: "_f8f6f4"}
        prefix.append(f"#define __{name} __builtin_amdgcn_mfma_{'scale_' if K == 128 else ''}f32_{N}x{M}x{K}{type_map[dtype_in]}")
      # #define __WMMA_16_16_16_half_half __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12
      elif self.tensor_cores == tc.amd_rdna4:
        prefix.append(f"#define __{name} __builtin_amdgcn_wmma_{type_map[dtype_out]}_16x16x16_{type_map[dtype_in]}_w32_gfx12")
      elif dtype_out == dtypes.float:
        prefix.append(f"#define __{name} __builtin_amdgcn_wmma_f32_16x16x16_{'f16' if dtype_in == dtypes.half else 'bf16'}_w32")
      elif dtype_out == dtypes.int:
        # RDNA3 iu8 int8 WMMA (Q4_K MMQ). A/B arrive as 16 int8/lane; the builtin wants v4i32 (16 int8 packed)
        # -> bitcast. C/D = 8 int32. signed*signed: the 1st/3rd bool args are the per-operand SIGN flags where
        # true == SIGNED (AMD-VALIDATED bit-exact on gfx1100 -- passing false treats int8 as UNSIGNED, which
        # blows negatives up to 255 and gives garbage; cf. the sudot4 true=signed convention above).
        # NOTE: use render_dtype for the vector type spellings -- char.vec(16) renders as "signed_char16"
        # (NOT "char16"), and int.vec(4) ("int4") is needed for the pack but is not a naturally-used dtype, so
        # emit its typedef here if the render_kernel prologue didn't already (avoids a duplicate typedef).
        c16, i8, i4 = self.render_dtype(dtypes.char.vec(16)), self.render_dtype(dtypes.int.vec(8)), self.render_dtype(dtypes.int.vec(4))
        if dtypes.int.vec(4) not in used_dtypes: prefix.append(self.render_vector_prefix(dtypes.int.vec(4)))
        prefix.append(f"static inline __attribute__((device)) {i8} __{name}({c16} a, {c16} b, {i8} c) {{\n"
                      f"  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, *({i4}*)&a, true, *({i4}*)&b, c, false);\n}}")
      else: prefix.append(f"static inline __attribute__((device)) half8 __{name}"+"""(half16 a, half16 b, half8 c) {
  half16 c_frag = {}; half8 d; for (int n = 0; n < 8; n++) { c_frag[n*2] = c[n]; }
  c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c_frag, false);
  for (int n = 0; n < 8; n++) { d[n] = c_frag[n*2]; } return d;\n}""")
    return super().render_kernel(function_name, kernel, bufs, uops, prefix)

  def supported_dtypes(self): return {d for d in super().supported_dtypes()
                                      if (d not in dtypes.fp8_ocp or self.target.arch == "gfx950") and d not in dtypes.fp8_fnuz}

class HIPCCRenderer(HIPRenderer):
  def __init__(self, target:Target): super().__init__(target, use_hipcc=True)
