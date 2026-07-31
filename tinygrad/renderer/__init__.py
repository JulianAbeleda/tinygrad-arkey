from __future__ import annotations
from typing import Callable, cast
from dataclasses import dataclass
from tinygrad.helpers import prod, Target, EMULATED_DTYPES
from tinygrad.uop.ops import Ops, UOp, sint, ssimplify, smin, GroupOp, PatternMatcher
from tinygrad.dtype import AddrSpace, DType, dtypes
from tinygrad.codegen.opt.tc import TensorCore
from tinygrad.device import Compiler

@dataclass(frozen=True)
class Estimates:
  # number of FLOPS used in the Kernel
  ops:sint = 0
  # bytes accessed in loads and stores
  lds:sint = 0
  # total bytes accessed, counting only once for bytes that are accessed multiple times
  mem:sint = 0
  def __add__(self, o:Estimates): return Estimates(self.ops + o.ops, self.lds + o.lds, self.mem + o.mem)
  def simplify(self): return Estimates(ssimplify(self.ops), ssimplify(self.lds), ssimplify(self.mem))
  @staticmethod
  def from_uops(uops:tuple[UOp, ...], ignore_indexing=False) -> Estimates:
    flops: sint = 0
    lds: sint = 0
    mem: dict[tuple[UOp, Ops], sint] = {}
    mults: sint = 1
    mult_stack: list[sint] = []
    for u in uops:
      if u.op in {Ops.LOAD, Ops.STORE}:
        buf = u
        while len(buf.src) and buf.op is not Ops.PARAM: buf = buf.src[0]
        if buf.op is Ops.PARAM:
          # u.src[0] is INDEX, cap at buffer size for re-reads (e.g. matmul)
          accessed = mem.get((buf, u.op), 0) + u.src[0].max_numel() * u.src[0].dtype.base.scalar().itemsize * mults
          mem[(buf, u.op)] = smin(accessed, buf.max_numel() * buf.dtype.scalar().itemsize)
      if u.op is Ops.RANGE:
        mult_stack.append(mults)
        mults *= cast(sint, u.src[0].ssimplify())
        # SPECIAL are already counted in mults
        mults = mults.substitute({x:x.const_like(0) for x in mults.toposort() if x.op is Ops.SPECIAL}) if isinstance(mults, UOp) else mults
      elif u.op is Ops.END: mults = mult_stack.pop(-1)
      elif u.op is Ops.SPECIAL: mults *= cast(sint, u.src[0].ssimplify()) # NOTE: we don't push to the mult_stack here, you can't end these
      elif u.op is Ops.DEFINE_VAR and u.arg[0] == 'core_id': mults *= u.arg[2] + 1
      elif u.op is Ops.LOAD and u.src[0].addrspace != AddrSpace.REG:
        lds += u.max_numel() * u.dtype.scalar().itemsize * mults
      elif u.op is Ops.STORE and u.src[0].addrspace != AddrSpace.REG:
        lds += u.max_numel() * u.src[1].dtype.scalar().itemsize * mults
      elif u.op in GroupOp.ALU and (not ignore_indexing or u.addrspace is not None):
        flops += (mults * (2 if u.op is Ops.MULACC else 1)) * u.max_numel()
      elif u.op is Ops.WMMA and (not ignore_indexing or u.addrspace is not None):
        flops += 2 * prod(u.arg[1]) // u.arg[5] * mults
    return Estimates(flops, lds, sum(mem.values()))

class Renderer:
  target: Target
  suffix: str = ""
  # TODO: make this generic with a list of supported types
  supports_float4: bool = True
  local_store_vector_widths: dict[DType, tuple[int, ...]] = {}
  local_store_requires_static_alignment: bool = True
  has_local: bool = True
  has_threads: bool = False
  has_shared: bool = True
  has_aux: bool = False # additional program info, eg. image shapes
  # NOTE: these two should be in (x,y,z) order to match the max_sizes argument in get_grouped_dims
  global_max: tuple[int, ...]|None = (0x8FFFFFFF,) * (3) # TODO: Ops.SPECIAL int32 indexes right now
  local_max: tuple[int, ...]|None = (0x8FFFFFFF,) * (3) # TODO: Ops.SPECIAL int32 indexes right now
  global_prod_max: tuple[int, ...]|None = None
  shared_max: int = 32768
  tensor_cores: list[TensorCore] = []
  # Target capability facts (see docs/task_workflow/input/target-capability-policy-decoupling-scope-20260730.md
  # TG2). Declarative only -- no admission/eligibility logic reads these here.
  # Lane (wavefront/simdgroup) width. None means unreported, never a silent 32 -- an unreported width must stay
  # distinguishable from a known 32 (e.g. Metal's simdgroup is 32-wide in hardware but is not modelled here
  # because it is not verified through this renderer's own reporting path).
  wave_size: int|None = None
  # Indirect-command-buffer byte-offset limit. None means "no such constraint on this backend" -- never a
  # sentinel number that could be mistaken for a real bound (Metal's is METAL_ICB_OFFSET_MAX, set below).
  max_indirect_buffer_offset: int|None = None
  # Threadgroup/LDS memory bank structure (docs/task_workflow/input/precontract-target-generalization-
  # scope-20260730.md PG1). Same discipline as wave_size above: None means unreported, never a silent
  # AMD-shaped default. `lds_bank_dwords` is the interleaved bank count (each bank one dword wide; AMD
  # RDNA3: 32, from AMD's published LDS architecture -- see HIPRenderer). `lds_bank_cycle_lanes` is how
  # many lanes one LDS access cycle services for a full b128 vector (AMD RDNA3: 8, also ISA-documented).
  # These affect only whether kernel_lds.py's cooperative-store row rotation is *beneficial*, never
  # whether it is *correct* -- the rotation is an exact one-writer cover of the tile regardless of bank
  # structure -- so an unknown target simply forgoes the optimization rather than guessing at Apple's
  # undocumented threadgroup memory banking.
  lds_bank_dwords: int|None = None
  lds_bank_cycle_lanes: int|None = None
  pre_matcher: PatternMatcher|None = None
  extra_matcher: PatternMatcher|None = None
  code_for_op: dict[Ops, Callable] = {}
  new_style: bool = False

  compiler: Compiler = Compiler()

  def __init__(self, target:Target): self.target = target
  def __reduce__(self): return self.__class__, (self.target,)
  @property
  def supports_warp_shfl_xor(self) -> bool:
    # One authority: derived from the TG1 provider (codegen/late/warp_reduce.py), not restated. A renderer
    # has the capability iff it declares a `warp_shfl_xor` provider -- CStyleLanguage subclasses that don't
    # provide one leave the class attribute at its None default; renderers with no such attribute at all
    # (e.g. LLVMRenderer, PTXRenderer) read as unavailable via getattr, same as the lowering code does.
    return getattr(self, "warp_shfl_xor", None) is not None
  @property
  def supports_flash_decode_fdot2(self) -> bool:
    # TG7 (docs/task_workflow/input/target-capability-policy-decoupling-scope-20260730.md): same derivation
    # shape as supports_warp_shfl_xor above -- one authority, the codegen/late/flash_decode_intrinsics.py
    # provider, never restated as an independent boolean.
    return getattr(self, "fdot2", None) is not None
  @property
  def supports_flash_decode_exp2f(self) -> bool:
    return getattr(self, "exp2f", None) is not None
  def render(self, uops:list[UOp]) -> str: raise NotImplementedError("needs a renderer")
  def asm(self, prg:UOp, lin:UOp) -> bytes: raise NotImplementedError("needs an assembler")
  def aux(self, uops:list[UOp]) -> dict: raise NotImplementedError("needs aux")
  def supported_dtypes(self) -> set[DType]:
    # double can't be bitcast to anything without long support
    return set(dtypes.all) - {dtypes.weakint} - ({dtypes.double} if dtypes.long in EMULATED_DTYPES.tolist(dtypes) else set())
