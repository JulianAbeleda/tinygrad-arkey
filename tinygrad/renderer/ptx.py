from typing import cast, Callable
import struct
from collections import defaultdict
from tinygrad.codegen.opt import tc
from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat, GroupOp, RuntimeLocalAllocation
from tinygrad.dtype import dtypes, DType, PtrDType, AddrSpace
from tinygrad.renderer import Renderer
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.helpers import flatten, prod, unwrap, Target

def render_val(x, dtype):
  if dtypes.is_float(dtype):
    if dtype == dtypes.double: return "0d%02X%02X%02X%02X%02X%02X%02X%02X" % tuple(struct.pack("d",x)[::-1])
    if dtype == dtypes.half: return "0x%02X%02X" % tuple(struct.pack("e",x)[::-1])
    return "0f%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])
  return str(int(x)) + ("U" if dtypes.is_unsigned(dtype) else "")

asm_for_op: dict[Ops, Callable] = {
  Ops.RECIPROCAL: lambda d,a,dt,name: f"rcp{'.approx' if dtypes.is_float(dt) else ''}.{name} {d}, {a};",
  Ops.EXP2: lambda d,a,dt,name: f"ex2.approx.{name} {d}, {a};", Ops.LOG2: lambda d,a,dt,name: f"lg2.approx.{name} {d}, {a};",
  Ops.SIN: lambda d,a,dt,name: f"sin.approx.{name} {d}, {a};", Ops.SQRT: lambda d,a,dt,name: f"sqrt.approx.{name} {d}, {a};",
  Ops.TRUNC: lambda d,a,dt,name: f"cvt.rzi.{name}.{name} {d}, {a};",
  Ops.SHR: lambda d,a,b,dt,name: f"shr.{name} {d}, {a}, {b};", Ops.SHL: lambda d,a,b,dt,name: f"shl.b{name[1:]} {d}, {a}, {b};",
  Ops.ADD: lambda d,a,b,dt,name: f"{'or' if dt == dtypes.bool else 'add'}.{name} {d}, {a}, {b};",
  Ops.MUL: lambda d,a,b,dt,name: f"{'and' if dt == dtypes.bool else 'mul'}{'.lo' if dtypes.is_int(dt) else ''}.{name} {d}, {a}, {b};",
  Ops.XOR: lambda d,a,b,dt,name: f"xor.pred {d}, {a}, {b};" if dt == dtypes.bool else f"xor.b{name[1:]} {d}, {a}, {b};",
  Ops.AND: lambda d,a,b,dt, name: f"and.pred {d}, {a}, {b};" if dt == dtypes.bool else f"and.b{name[1:]} {d}, {a}, {b};",
  Ops.OR: lambda d,a,b,dt, name: f"or.pred {d}, {a}, {b};" if dt == dtypes.bool else f"or.b{name[1:]} {d}, {a}, {b};",
  Ops.CDIV: lambda d,a,b,dt,name: f"div.{name} {d}, {a}, {b};", Ops.CMOD: lambda d,a,b,dt,name: f"rem.{name} {d}, {a}, {b};",
  Ops.MAX: lambda d,a,b,dt,name: f"max.{name} {d}, {a}, {b};", Ops.CMPEQ: lambda d,a,b,dt,name: f"setp.eq.{name} {d}, {a}, {b};",
  Ops.CMPLT: lambda d,a,b,dt,name: f"setp.lt.{name} {d}, {a}, {b};",
  Ops.CMPNE: lambda d,a,b,dt,name: f"setp.{'neu' if dtypes.is_float(dt) else 'ne'}.{name} {d}, {a}, {b};",
  Ops.MULACC: lambda d,a,b,c,dt,name: f"{'fma.rn' if dtypes.is_float(dt) else 'mad.lo'}.{name} {d}, {a}, {b}, {c};",
  Ops.WHERE: lambda d,a,b,c,dt,name: [f"@{a} mov.{name} {d}, {b};", f"@!{a} mov.{name} {d}, {c};"] if dt == dtypes.bool else \
    f"selp.{'b16' if name == 'f16' else name} {d}, {b}, {c}, {a};"
}

supports_half = (Ops.EXP2, Ops.ADD, Ops.MUL, Ops.MAX, Ops.CMPLT, Ops.WHERE, Ops.TRUNC)
doesnt_support_half: tuple[Ops, ...] = tuple(op for op in asm_for_op.keys() if op not in supports_half)
ptx_matcher = PatternMatcher([
  # bool CMPNE is XOR, bool CMPLT is XOR+AND (universal makes this slow, this is for renderer only)
  (UPat.var('x', dtype=dtypes.bool).ne(UPat.var('y')), lambda x,y: x^y),
  (UPat.var('x', dtype=dtypes.bool).alu(Ops.CMPEQ, UPat.var('y')), lambda x,y: (x^y)^True),
  (UPat.var('x', dtype=dtypes.bool)<UPat.var('y'), lambda x,y: (x^True)&y),
  # upcast to float32 all the ops that don't support half
  (UPat(doesnt_support_half, dtype=dtypes.half, name="x"),
    lambda x: (UOp(x.op, dtypes.float32, tuple(vv.cast(dtypes.float32) for vv in x.src), x.arg).cast(dtypes.half))),
  # load/store bool -> uint8 (only for memory, not registers)
  (UPat(Ops.LOAD, dtypes.bool, src=(UPat(name="idx"),), name="x", allow_any_len=True),
   lambda x,idx: UOp(x.op, dtypes.uint8, x.src[0:1] + ((x.src[1].cast(dtypes.uint8),) if len(x.src) >= 2 else ()) + x.src[2:]).cast(dtypes.bool) \
     if idx.addrspace != AddrSpace.REG else None),
  (UPat(Ops.STORE, src=(UPat(name="idx"), UPat(dtype=dtypes.bool)), name="x", allow_any_len=True),
   lambda x,idx: UOp(x.op, dtypes.void, (x.src[0], x.src[1].cast(dtypes.uint8))+x.src[2:]) if idx.addrspace != AddrSpace.REG else None),
  # ptx shr and shl instructions require y to be uint
  (UPat.var("x") << UPat.var("y"), lambda x,y: UOp(Ops.SHL, x.dtype, (x,y.cast(dtypes.uint))) if y.dtype != dtypes.uint else None),
  (UPat.var("x") >> UPat.var("y"), lambda x,y: UOp(Ops.SHR, x.dtype, (x,y.cast(dtypes.uint))) if y.dtype != dtypes.uint else None),
])

def mem_type(x:UOp) -> str: return 'shared' if x.addrspace == AddrSpace.LOCAL else 'global'

def render_wmma(ctx: "PTXRenderer", wmma: UOp):
  assert ctx.wmma_r, "registry values for wmma must be populated"
  (N, M, K), dtype_in, dtype_out = wmma.arg[1], wmma.arg[2], wmma.arg[3]

  q6_native = dtype_in is dtypes.char and dtype_out is dtypes.int and isinstance(wmma.src[0].tag,tuple) and \
    wmma.src[0].tag[:3] == ("native_fragment_carrier_v1",2,"bitcast")
  if q6_native:
    vals=ctx.r[wmma.src[1]]; packed=ctx.wmma_r[1]
    for i,reg in enumerate(packed):
      lanes=vals[i*4:(i+1)*4]; temps=ctx.wmma_pack[1][i*2:(i+1)*2]
      yield f"mov.b32 {temps[0]}, {{{lanes[0]}, {lanes[1]}}};"
      yield f"mov.b32 {temps[1]}, {{{lanes[2]}, {lanes[3]}}};"
      yield f"prmt.b32 {reg}, {temps[0]}, {temps[1]}, 0x6420;"
    yield f'mma.sync.aligned.m{M}n{N}k{K}.row.col.s32.s8.s8.s32 '+\
      f'{{{", ".join(ctx.r[wmma])}}}, {{{", ".join(ctx.r[wmma.src[0]])}}}, {{{", ".join(packed)}}}, '+\
      f'{{{", ".join(ctx.r[wmma.src[2]])}}};'
    return

  for src_idx,(src, regs) in enumerate(zip(wmma.src, ctx.wmma_r)):
    if isinstance(src.tag,tuple) and src.tag[:3] == ("native_fragment_carrier_v1",2,"bitcast"):
      for i,reg in enumerate(regs): yield f"mov.b32 {reg}, {ctx.r[src][i]};"
      continue
    if src.dtype.scalar() is dtypes.char:
      for i,reg in enumerate(regs):
        vals=ctx.r[src][i*4:(i+1)*4]; temps=ctx.wmma_pack[src_idx][i*2:(i+1)*2]
        yield f"mov.b32 {temps[0]}, {{{vals[0]}, {vals[1]}}};"
        yield f"mov.b32 {temps[1]}, {{{vals[2]}, {vals[3]}}};"
        yield f"prmt.b32 {reg}, {temps[0]}, {temps[1]}, 0x6420;"
      continue
    for i, reg in enumerate(regs): # pack input and acc registers
      if (elems_per_reg := 4 // src.dtype.scalar().itemsize) == 1: yield f"mov.b32 {reg}, {ctx.r[src][i]};"
      else: yield f"mov.b32 {reg}, {{{', '.join(ctx.r[src][i * elems_per_reg : (i+1) * elems_per_reg])}}};"

  dt_map_in, dt_map_out = {dtypes.float: "tf32", dtypes.half: "f16", dtypes.char: "s8"}, \
                           {dtypes.float: "f32", dtypes.half: "f16", dtypes.int: "s32"}
  yield f'mma.sync.aligned.m{M}n{N}k{K}.row.col.{dt_map_out[dtype_out]}.{dt_map_in[dtype_in]}.{dt_map_in[dtype_in]}.{dt_map_out[dtype_out]}{" "*12}'+\
        f'{{{", ".join(ctx.wmma_r[2])}}}, {{{", ".join(ctx.wmma_r[0])}}}, {{{", ".join(ctx.wmma_r[1])}}}, {{{", ".join(ctx.wmma_r[2])}}};'

  for i, reg in enumerate(ctx.wmma_r[2]): # unpack acc registers
    if (elems_per_reg := 4 // dtype_out.itemsize) == 1: yield f"mov.b32 {ctx.r[wmma][i]}, {reg};"
    else: yield f"mov.b32 {{{', '.join(ctx.r[wmma][i * elems_per_reg : (i+1) * elems_per_reg])}}}, {reg};"

def modifier(a: DType, b: DType): return '.rzi' if dtypes.is_int(a) and dtypes.is_float(b) else '.rn' if dtypes.is_float(a) and \
  (a.itemsize < b.itemsize or dtypes.is_int(b) or b == dtypes.bool) else ''

def render_customi(ctx:"PTXRenderer", x:UOp):
  if x.arg == ("ptx_ldmatrix_x2_v1",):
    return [f"cvt.s64.{ctx.types[x.src[1].dtype]} {ctx.fragment_addr[x]}, {ctx.r[x.src[1]]};",
      f"mad.lo.s64 {ctx.fragment_addr[x]}, {ctx.fragment_addr[x]}, {x.src[0].dtype.base.itemsize}, {ctx.r[x.src[0]]};",
      f"ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{{', '.join(ctx.r[x])}}}, [{ctx.fragment_addr[x]}];"]
  if x.arg == ("ptx_fragment_bitcast_v1",): return []
  exact={"__fmul_rn({0},{1})":"mul.rn.f32", "__fadd_rn({0},{1})":"add.rn.f32", "__fmaf_rn({0},{1},{2})":"fma.rn.f32"}
  if x.arg in exact: return f"{exact[x.arg]} {ctx.r[x]}, {', '.join(ctx.r[s] for s in x.src)};"
  return None

string_rewrite = PatternMatcher([
  (UPat.cvar("x", dtypes.bool), lambda ctx, x: f"setp.ne.s16 {ctx.r[x]}, {render_val(x.arg, x.dtype)}, 0;"),
  (UPat.cvar("x"), lambda ctx, x: f"mov.b{ctx.types[x.dtype][1:]} {ctx.r[x]}, {render_val(x.arg, x.dtype)};"),
  (UPat(Ops.SPECIAL, name="x"), lambda ctx,x: f"mov.u32 %{x.arg}, %{'ctaid' if x.arg[0] == 'g' else 'tid'}.{chr(120+int(x.arg[-1]))};"),
  (UPat(Ops.CUSTOMI, name="x"), render_customi),
  (UPat(Ops.PARAM, name="x"), lambda ctx, x:
   f"ld.param.{ctx.types[dtypes.ulong] if x.addrspace is AddrSpace.GLOBAL else ctx.mem_types[x.dtype]} {ctx.r[x]}, [data{x.arg.slot}+0];"),
  # address computation: addr = buf + idx*itemsize
  (UPat((Ops.INDEX, Ops.SHRINK), src=(UPat.var("buf"), UPat.var("idx")), allow_any_len=True, name="x"), lambda ctx, x, buf, idx:
   [f"cvt.s64.{ctx.types[idx.dtype]} {ctx.r[x]}, {ctx.r[idx]};", f"mad.lo.s64 {ctx.r[x]}, {ctx.r[x]}, {x.dtype.itemsize}, {ctx.r[buf]};"]),
  (UPat((Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ), name="x", allow_any_len=True, src=(UPat.var("src0"),)),
    lambda ctx, x, src0: ctx.code_for_op[x.op](ctx.r[x], *[ctx.r[v] for v in x.src], src0.dtype, ctx.types[src0.dtype])),
  (UPat(GroupOp.ALU, name="x"), lambda ctx, x: ctx.code_for_op[x.op](ctx.r[x], *[ctx.r[v] for v in x.src], x.dtype, ctx.types[x.dtype])),
  (UPat(Ops.BITCAST, name="x", src=(UPat.var("a"),), allow_any_len=True), lambda ctx, x, a: f"mov.b{ctx.types[x.dtype][1:]} {ctx.r[x]}, {ctx.r[a]};"),
  (UPat(Ops.CAST, name="x", src=(UPat(dtype=dtypes.bool, name="a"),)),
   lambda ctx, x, a: f"selp.b{ctx.types[x.dtype][1:]} {ctx.r[x]}, {render_val(1, x.dtype)}, {render_val(0, x.dtype)}, {ctx.r[a]};"),
  (UPat(Ops.CAST, name="x", src=(UPat.var("a"),)),
   lambda ctx, x, a: f"cvt{modifier(x.dtype, a.dtype)}.{ctx.cast_types[x.dtype]}.{ctx.cast_types[a.dtype]} {ctx.r[x]}, {ctx.r[a]};"),
  # store / gated load / load
  (UPat(Ops.STORE, src=(UPat(name="loc"), UPat.var("var"))), lambda ctx, loc, var:
   f"mov.{'pred' if var.dtype == dtypes.bool else 'b'+ctx.types[var.dtype][1:]} {ctx.r[loc]}, {ctx.r[var]};" \
     if loc.addrspace == AddrSpace.REG else None),
  (UPat(Ops.STORE, src=(UPat((Ops.INDEX, Ops.SHRINK, Ops.CAST), name="loc"), UPat.var("var"))),
   lambda ctx, loc, var: f"st.{mem_type(loc)}" + \
    f"{f'.v{cnt}' if ((cnt:=var.max_numel())>1) else ''}.{ctx.mem_types[var.dtype.scalar()]} " + \
    f"[{ctx.r[loc]}+0], {('{' + ', '.join(ctx.r[var]) + '}') if var.max_numel() > 1 else ctx.r[var]};"),
  (UPat(Ops.LOAD, name="x", src=(UPat((Ops.INDEX, Ops.SHRINK, Ops.CAST), name="loc"), UPat.var("alt"), UPat.var("gate"))),
    lambda ctx, x, loc, alt, gate: flatten([
    [f"mov.{ctx.mem_types[x.dtype.scalar()]} {v}, {render_val(0, x.dtype.scalar())};" for v in ctx.r[x]],
    [f"@{ctx.r[gate]} ld.{mem_type(loc)}.v{x.max_numel()}.{ctx.mem_types[x.dtype.scalar()]} {{{', '.join(ctx.r[x])}}}, [{ctx.r[loc]}+0];"]
  ]) if alt.max_numel() > 1 else [
    f"@{ctx.r[gate]} ld.{mem_type(loc)}.{ctx.mem_types[x.dtype.scalar()]} {ctx.r[x]}, [{ctx.r[loc]}+0];",
    f"@!{ctx.r[gate]} mov.b{ctx.types[x.dtype.scalar()][1:]} {ctx.r[x]}, {ctx.r[alt]};"]),
  (UPat(Ops.LOAD, name="x", src=(UPat((Ops.INDEX, Ops.SHRINK, Ops.CAST), name="loc"),)),
    lambda ctx, x, loc: f"ld.{mem_type(loc)}.v{x.max_numel()}.{ctx.mem_types[x.dtype.scalar()]} {{{', '.join(ctx.r[x])}}}, [{ctx.r[loc]}+0];" \
     if x.max_numel() > 1 else f"ld.{mem_type(loc)}.{ctx.mem_types[x.dtype]} {ctx.r[x]}, [{ctx.r[loc]}+0];"),
  # simple
  (UPat(Ops.BUFFER, name="x"), lambda ctx, x: [] if x.addrspace == AddrSpace.REG else [
    f".shared .align 16 .b8 local{x.arg.slot}[{x.max_numel()*x.dtype.itemsize}];", f"mov.u64 {ctx.r[x]}, local{x.arg.slot}[0];"]),
  (UPat(Ops.RANGE, name="r"), lambda ctx, r: [
    f"mov.u32 {ctx.r[r]}, -1;",
    f"bra END_{ctx.r[r][1:]};",
    "LOOP_" + f"{ctx.r[r][1:]}:"]),
  (UPat(Ops.END, name="x", src=(UPat(), UPat(Ops.RANGE, name="r"))), lambda ctx, x, r: [
    "END_" + f"{ctx.r[r][1:]}:",
    ctx.code_for_op[Ops.ADD](ctx.r[r], ctx.r[r], "1", dtypes.int, ctx.types[dtypes.int]),
    ctx.code_for_op[Ops.CMPLT](ctx.r[x], ctx.r[r], ctx.r[r.src[0]], dtypes.int, ctx.types[dtypes.int]),
    f"@{ctx.r[x]} bra LOOP_{ctx.r[r][1:]};"]),
  (UPat(Ops.IF, name="x"), lambda ctx, x: f"@!{ctx.r[x.src[0]]} bra IF_{ctx.r[x.src[0]][1:]}_{ctx.uops.index(x)};"),
  (UPat(Ops.ENDIF, name="x"), lambda ctx, x: f"IF_{ctx.r[x.src[0].src[0]][1:]}_{ctx.uops.index(x.src[0])}:"),
  (UPat(Ops.WMMA, name="x"), lambda ctx, x: list(render_wmma(ctx, x))),
  (UPat(Ops.BARRIER), lambda ctx: ctx.barrier),
])

class PTXRenderer(Renderer):
  supports_post_barrier_regions = True
  suffix = "PTX"
  global_max, local_max, shared_max = CUDARenderer.global_max, CUDARenderer.local_max, CUDARenderer.shared_max
  tc_sm80 = [x for x in tc.cuda_sm80 if x.dtype_in in [dtypes.half, dtypes.float]]
  code_for_op = asm_for_op
  extra_matcher = ptx_matcher
  # Q6_K staging subtracts 0x20 from bytes known to be in [0, 63]. Setting
  # each byte's high bit prevents inter-byte borrow; toggling it afterward
  # restores the exact signed-byte two's-complement result.
  packed_i8_sub = staticmethod(lambda value,bias:
    (((value | UOp.const(dtypes.uint32,0x80808080))-bias) ^ UOp.const(dtypes.uint32,0x80808080)))
  native_fragment_x2 = staticmethod(lambda buffer,index:
    UOp(Ops.CUSTOMI,dtypes.uint32.vec(2),(buffer,index.cast(dtypes.int32)),arg=("ptx_ldmatrix_x2_v1",)))
  native_fragment_bitcast = staticmethod(lambda value,dtype:
    UOp(Ops.CUSTOMI,dtype,(value,),arg=("ptx_fragment_bitcast_v1",)))
  def __init__(self, target:Target):
    super().__init__(target)
    from tinygrad.runtime.support.compiler_cuda import NVPTXCompiler, PTXCompiler
    self.compiler = (PTXCompiler if target.interface.startswith("MOCK") or target.device == "CUDA" else NVPTXCompiler)(target.arch)
    self.tensor_cores = PTXRenderer.tc_sm80 if (ver:=int(target.arch[3:])) >= 80 else tc.cuda_sm75 if ver >= 75 else []

  # language options
  kernel_prefix = """.version VERSION
.target TARGET
.address_size 64
.visible .entry"""
  barrier = "bar.sync\t0;"
  types: dict[DType, str] = { dtypes.int8: "s16", dtypes.int16: "s16", dtypes.int32: "s32", dtypes.int64: "s64",
                              dtypes.uint8: "u16", dtypes.uint16: "u16", dtypes.uint32: "u32", dtypes.uint64: "u64",
                              dtypes.float16: "f16", dtypes.float32: "f32", dtypes.float64: "f64", dtypes.bool: "pred",
                              dtypes.weakint: "s32" }

  mem_types: dict[DType, str] = {**types, dtypes.int8: "s8", dtypes.uint8: "u8", dtypes.bool: "u8", dtypes.float16: "b16"}
  cast_types: dict[DType, str] = {**types, dtypes.int8: "s8", dtypes.uint8: "u8"}

  def render_kernel(self, kernel, function_name, bufs, regs, uops) -> str:
    def fmt(line): return line if line[0]=="$" else "\t" + line.replace(" ", "\t" if len(line.split(" ")[0]) > 7 else "\t\t", 1)
    kernel = '\n'.join(map(fmt, [f".reg .{reg.split('_')[-2]} %{reg}<{cnt}>;" for reg,cnt in regs] + kernel + ["ret;"]))
    local_dims = [u.src[0] for u in uops if u.op is Ops.SPECIAL and u.arg[0] == "l"]
    launch_bounds = prod([d.vmax for d in local_dims])
    params = ',\n\t'.join([f".param .{'u64' if u.addrspace is AddrSpace.GLOBAL else self.types[u.dtype]} {name}" for name,u in bufs])
    prefix=self.kernel_prefix.format(launch_bounds=launch_bounds)
    if self.module_shared: prefix=prefix.replace(".visible .entry",'\n'.join(self.module_shared)+"\n.visible .entry")
    return f"{prefix} {function_name} (\n\t{params}\n)\n.maxntid {launch_bounds}\n{{\n{kernel}\n}}"

  def render(self, uops:list[UOp]) -> str:
    kernel:list[str] = []
    bufs = []
    self.module_shared:list[str] = []

    c: defaultdict[str, int] = defaultdict(int)
    r: dict[UOp, list[str]|str] = {}
    self.r = r
    self.fragment_addr: dict[UOp,str] = {}
    self.uops = uops

    def ssa(prefix:str, u:UOp|None=None, dtype:str|None=None) -> str:
      nonlocal c
      prefix += f"_{dtype if dtype is not None else self.types[unwrap(u).dtype.base]}_"
      c[prefix] += 1
      return f"%{prefix}{c[prefix]-1}"

    name = "test"
    for u in uops:
      if u.op in {Ops.NOOP, Ops.GROUP}: continue
      if u.op is Ops.AFTER:
        self.r[u] = self.r[u.src[0]]
        continue
      if u.op is Ops.CAST and isinstance(u.dtype, PtrDType) and isinstance(u.src[0].dtype, PtrDType):
        # Pointer-to-pointer CAST is an address no-op in PTX; the devectorizer emits
        # INDEX(...).cast(vecN.ptr(...)) for vectorized loads, so alias the u64 address.
        r[u] = r[u.src[0]]
        continue
      if u.op is Ops.SINK:
        if u.arg is not None: name = u.arg.function_name
        continue
      if u.op is Ops.STACK:
        r[u] = [cast(str,r[x]) for x in u.src]
        continue
      if u.op is Ops.GEP and len(u.src) == 1 and isinstance(u.arg, tuple) and len(u.arg) == 1 and isinstance(u.arg[0], int):
        # Vector lane select: the devectorizer keeps vec loads/WMMA results as register lists
        # and extracts lanes with GEP(vec, (i,)).
        r[u] = r[u.src[0]][u.arg[0]]
        continue
      if u.op is Ops.BUFFER and u.addrspace == AddrSpace.REG:
        r[u] = [ssa("reg", u, self.types[u.dtype.base.scalar()]) for _ in range(u.max_numel())]
        continue
      if u.op is Ops.DEFINE_REG and u.addrspace == AddrSpace.REG:
        # The stage-1 precontract pipeline builds its register-resident accumulator as
        # DEFINE_REG (AddrSpace.REG).  Same per-scalar register-array contract the BUFFER
        # form gets after the new-style rewrite; PTX (old-style) sees the define directly.
        r[u] = [ssa("reg", u, self.types[u.dtype.base.scalar()]) for _ in range(u.max_numel())]
        continue
      if u.op is Ops.DEFINE_LOCAL:
        # LDS windows arrive as DEFINE_LOCAL on the old-style path.  Emit the shared
        # declaration (byte-sized from the element dtype, not the pointer) plus the u64 base.
        slot = u.arg
        if isinstance(u.tag,RuntimeLocalAllocation): self.module_shared.append(f".extern .shared .align 16 .b8 local{slot}[];")
        else: kernel.append(f".shared .align 16 .b8 local{slot}[{u.max_numel()*u.dtype.base.itemsize}];")
        r[u] = ssa("local", u, "u64")
        kernel.append(f"mov.u64 {r[u]}, local{slot}[0];")
        continue
      if u.op in {Ops.INDEX, Ops.SHRINK, Ops.LOAD} and u.src[0].addrspace == AddrSpace.REG:
        # on REG, INDEX/SHRINK pick the register (must be CONST) and LOAD is a noop
        r[u] = r[u.src[0]] if u.op is Ops.LOAD else r[u.src[0]][u.src[1].arg]
        continue
      if u.op is Ops.SPECIAL: r[u] = "%" + u.arg
      elif u.op is Ops.LOAD:
        r[u] = [ssa('val', dtype=self.types[u.dtype.scalar()]) for _ in range(u.max_numel())] if u.max_numel() > 1 else ssa('val', u)
      elif u.op is Ops.PARAM: bufs.append((f"data{u.arg.slot}", u))
      elif u.op is Ops.WMMA:
        # registers for packing/unpacking input and acc
        packed_count=lambda src: len(r[src]) if isinstance(src.tag,tuple) and src.tag[:3] == \
          ("native_fragment_carrier_v1",2,"bitcast") else len(range(0,len(r[src]),4//src.dtype.scalar().itemsize))
        self.wmma_r = [[ssa("wmma_in", dtype="b32") for _ in range(packed_count(u.src[0]))],
                       [ssa("wmma_in", dtype="b32") for _ in range(packed_count(u.src[1]))],
                       [ssa("wmma_acc", dtype="b32") for _ in range(0, len(r[u.src[2]]), 4 // u.dtype.scalar().itemsize)]]
        self.wmma_pack = [[ssa("wmma_pack",dtype="b32") for _ in range(2*packed_count(src))]
                          if src.dtype.scalar() is dtypes.char and not (isinstance(src.tag,tuple) and src.tag[:3] ==
                            ("native_fragment_carrier_v1",2,"bitcast")) else [] for src in u.src]
        r[u] = [ssa("wmma", dtype=self.types[u.dtype.scalar()]) for _ in range(u.max_numel())]
      elif u.op is Ops.CUSTOMI and isinstance(u.tag,tuple) and u.tag[:2] == ("native_fragment_carrier_v1",2):
        if len(u.tag)>=3 and u.tag[2]=="bitcast": r[u] = r[u.src[0]]
        else:
          r[u] = [ssa("fragment",dtype="u32") for _ in range(2)]
          self.fragment_addr[u] = ssa("fragment_addr",dtype="s64")
      elif u.op is Ops.CUSTOMI: r[u] = ssa("custom",u)
      prefix, dtype = {Ops.CAST: ("cast", None), Ops.BITCAST: ("cast", None), Ops.END: ("pred", "pred"), Ops.RANGE: ("ridx", None),
        Ops.CONST: ("const", None), Ops.BUFFER: ("local", "u64"), Ops.INDEX: ("bidx", "u64"), Ops.SHRINK: ("bidx", "u64"),
        Ops.PARAM: ("dat", "u64" if u.addrspace is AddrSpace.GLOBAL else None), **{op: ("alu", None) for op in GroupOp.ALU}}.get(u.op, (None, None))
      if prefix: r[u] = ssa(prefix, u, dtype)

      l: str|list[str]|None = string_rewrite.rewrite(u, ctx=self)
      if l is None:
        raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")
      kernel.extend([l] if isinstance(l, str) else l)

      if u.op is Ops.SPECIAL: kernel = [f".reg .u32 %{u.arg};"] + kernel
    return self.render_kernel(kernel, name, bufs, c.items(), uops)

  def supported_dtypes(self): return {d for d in super().supported_dtypes()
                                      if (d != dtypes.half or int(self.target.arch[3:]) >= 53) and d not in dtypes.fp8s+(dtypes.bfloat16,)}
