from __future__ import annotations
import itertools
from dataclasses import dataclass, field, replace
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import PatternMatcher, UOp, Ops, consumer_map_from_toposort

@dataclass(frozen=True)
class RegisterSpan:
  """Physical-register shape required by one virtual register definition."""
  count: int
  alignment: int = 1

  def __post_init__(self):
    if not isinstance(self.count, int) or isinstance(self.count, bool) or self.count < 1:
      raise ValueError("register span count must be a positive integer")
    if not isinstance(self.alignment, int) or isinstance(self.alignment, bool) or self.alignment < 1:
      raise ValueError("register span alignment must be a positive integer")

@dataclass(frozen=True)
class Register:
  name: str
  index: int
  _cons: tuple[Register, ...] = field(default_factory=tuple)
  _span: RegisterSpan = field(default_factory=lambda: RegisterSpan(1))
  @property
  def cons(self): return self._cons or (self,)
  @property
  def span(self): return self._span
  def __repr__(self): return self.name

@dataclass(frozen=True)
class FixedRegisterUse(Register):
  """An already-owned physical register operand, never a virtual definition."""

class IselContext:
  def __init__(self, sink:UOp):
    self.uses = consumer_map_from_toposort(sink.toposort())
    self.reg_n = itertools.count()
    arg_order = {Ops.PARAM: 0, Ops.DEFINE_VAR: 1, Ops.SPECIAL: 2}
    self.func_args = sorted([u for u in self.uses if u.op in arg_order], key=lambda k: (arg_order[k.op], k.arg))

  def vreg(self, cons:tuple[Register, ...]|Register, span:RegisterSpan|None=None):
    return Register(f"v{next(self.reg_n)}", 0, _cons=cons if isinstance(cons, tuple) else (cons,), _span=span or RegisterSpan(1))

@dataclass
class PreRegAllocContext:
  lock: UOp|None = None
  clobbered: set[UOp] = field(default_factory=set)

@dataclass(frozen=True)
class CompilerRegisterLease:
  """Selection-owned physical register lease carried to final assembly."""
  logical_role: str
  bank: str
  start: int
  end: int
  purpose: str
  fixed: bool
  slots: int
  lifetime: tuple[str, ...]

  def __post_init__(self):
    if self.logical_role not in ("A", "B", "C"): raise ValueError("capture lease role must be A, B, or C")
    if self.bank not in ("vgpr", "sgpr"): raise ValueError("capture lease bank must be vgpr or sgpr")
    if not (isinstance(self.start, int) and isinstance(self.end, int) and 0 <= self.start < self.end):
      raise ValueError("capture lease must be a non-empty half-open interval")
    if not isinstance(self.purpose, str) or not self.purpose: raise ValueError("capture lease purpose is required")
    if self.fixed is not True: raise ValueError("capture leases must be fixed by their compiler owner")
    if not isinstance(self.slots, int) or self.slots <= 0: raise ValueError("capture lease slots must be positive")
    if len(self.lifetime) < 2 or any(not isinstance(x, str) or not x for x in self.lifetime):
      raise ValueError("capture lease lifetime requires named boundaries")

@dataclass(frozen=True)
class CompilerCaptureProof:
  """Hashable A/B/C ownership handoff, finalized only by register allocation."""
  leases: tuple[CompilerRegisterLease, ...]
  authority: str = "instruction_selection"
  regalloc_status: str = "selection_complete"
  scratch_spills: int|None = None
  vgpr_spills: int|None = None
  sgpr_spills: int|None = None
  lds_bytes: int|None = None
  wait_policy: str|None = None
  owned_storage: tuple[UOp, ...] = ()

  def __post_init__(self):
    if not isinstance(self.leases, tuple) or any(not isinstance(x, CompilerRegisterLease) for x in self.leases):
      raise TypeError("compiler capture proof requires typed leases")
    if self.leases and not {"A", "B", "C"}.issubset({x.logical_role for x in self.leases}):
      raise ValueError("compiler capture proof requires A, B, and C ownership")

  def finalize_zero_spill(self) -> CompilerCaptureProof:
    return replace(self, authority="final_regalloc", regalloc_status="post_regalloc", scratch_spills=0, vgpr_spills=0, sgpr_spills=0)

class ISARenderer(Renderer):
  pre_isel_matcher: PatternMatcher
  def is_rematerializable(self, u:UOp) -> bool: return False
  isel_matcher: PatternMatcher
  post_isel_matcher: PatternMatcher|None = None
  pre_regalloc_matcher: PatternMatcher|None = None
  post_regalloc_matcher: PatternMatcher

  def is_two_address(self, x:UOp) -> bool: return False
  def stack_pointer(self) -> UOp: raise NotImplementedError("arch specific")
  def copy(self, x:UOp, reg:Register) -> UOp: raise NotImplementedError("arch specific")
  def spill(self, disp:UOp, x:UOp) -> UOp: raise NotImplementedError("arch specific")
  def fill(self, disp:UOp, x:UOp, reg:Register) -> UOp: raise NotImplementedError("arch specific")
  def asm_str(self, uops:list[UOp], function_name:str) -> str: raise NotImplementedError("arch specific")
  def capture_selection_proof(self, ctx:IselContext) -> CompilerCaptureProof|None: return None
