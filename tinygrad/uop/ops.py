from __future__ import annotations
from typing import Any, Callable, cast, TYPE_CHECKING, Type, Sequence, Iterable, Final, Iterator, NamedTuple
import sys, time, functools, itertools, math, operator, hashlib, os, types, pickle, pathlib, inspect, weakref, collections, struct
from dataclasses import dataclass
from enum import Enum, auto
from tinygrad.uop import Ops, GroupOp, MemorySemanticOwner
from tinygrad.dtype import ConstType, ImageDType, dtypes, DType, DTypeLike, to_dtype, truncate, PtrDType, least_upper_dtype, Invalid, AddrSpace
from tinygrad.dtype import ConstFloat, PyConst, storage_fmt_for_dtype, to_storage_scalar, from_storage_scalar
from tinygrad.device import Buffer, MultiBuffer, canonicalize_device
from tinygrad.helpers import ContextVar, all_int, prod, getenv, all_same, Context, partition, temp, unwrap, T, argfix, Metadata, flatten, TRACEMETA
from tinygrad.helpers import PROFILE, dedup, cdiv, cmod, floordiv, floormod, diskcache_put, to_function_name, cpu_profile, TracingKey
from tinygrad.helpers import VIZ, SPEC, CAPTURE_PROCESS_REPLAY, DISALLOW_BROADCAST, get_shape, fully_flatten
from tinygrad.helpers import colored, ansilen, printable
if TYPE_CHECKING:
  from tinygrad.renderer import Estimates

class AxisType(Enum):
  def __repr__(self): return str(self)
  GLOBAL = auto(); WARP = auto(); LOCAL = auto(); LOOP = auto(); GROUP_REDUCE = auto(); REDUCE = auto(); UPCAST = auto(); UNROLL = auto() # noqa: E702
  THREAD = auto(); PLACEHOLDER = auto() # noqa: E702

@dataclass(frozen=True, order=True)
class ParamArg:
  slot: int
  vmin_vmax: tuple[PyConst, PyConst]|None = None
  name: str|None = None
  addrspace: AddrSpace = AddrSpace.GLOBAL
  axis: int|None = None
  device: str|tuple[str, ...]|None = None
  def __repr__(self):
    fields = (("vmin_vmax", None), ("name", None), ("addrspace", AddrSpace.GLOBAL), ("axis", None), ("device", None))
    args = [repr(self.slot)] + [f"{k}={v!r}" for k,default in fields if (v:=getattr(self, k)) != default]
    return f"ParamArg({', '.join(args)})"
axis_letters = {AxisType.GLOBAL: "g", AxisType.THREAD: "t", AxisType.LOCAL: "l", AxisType.WARP: "w", AxisType.LOOP: "L", AxisType.UPCAST: "u",
                AxisType.GROUP_REDUCE: "G", AxisType.REDUCE: "R", AxisType.UNROLL: "r"}
axis_colors = {AxisType.GLOBAL: "blue", AxisType.THREAD: "BLUE", AxisType.LOCAL: "cyan", AxisType.WARP: "CYAN", AxisType.LOOP: "WHITE",
               AxisType.UPCAST: "yellow", AxisType.GROUP_REDUCE: "RED", AxisType.REDUCE: "red", AxisType.UNROLL: "magenta"}

# NOTE: LOCAL and GROUP_REDUCE have the same priority. the order here matters
axis_to_pos = {AxisType.LOOP: -1, AxisType.THREAD: 0, AxisType.GLOBAL: 0, AxisType.WARP: 1, AxisType.LOCAL: 2, AxisType.UPCAST: 3,
               AxisType.GROUP_REDUCE: 2, AxisType.REDUCE: 4, AxisType.UNROLL: 5}

range_start = {Ops.STAGE: 1, Ops.REDUCE: 1, Ops.WMMA: 3, Ops.END: 1, Ops.CALL: 1, Ops.FUNCTION: 1,
               Ops.COPY: 2, Ops.SLICE: 2, Ops.LINEAR: 0}

# https://en.wikipedia.org/wiki/Identity_element
def identity_element(op:Ops, dt:DType) -> PyConst: return dt.const({Ops.ADD:0, Ops.MUL:1, Ops.MAX:dt.min}[op])

# With True as the default, this matches the old symbolic behavior
def resolve(x:UOp|bool, default:bool=True):
  if isinstance(x, bool): return x
  assert x.dtype == dtypes.bool, "UOp in resolve must be bool"
  # NOTE: generating the text for the exception is expensive, so we do this
  return bool(sx.vmin) if (sx:=x.simplify()).vmin == sx.vmax else default

# smax/smin are replacements for max/min that preserve symbolic
def _suop(lst, uop_fxn, python_fxn):
  uops, nums = partition(lst, lambda x: isinstance(x, UOp))
  return ssimplify(functools.reduce(uop_fxn, uops + ([python_fxn(nums)] if nums else [])))
def smax(*lst) -> sint: return _suop(argfix(*lst), UOp.maximum, max)
def smin(*lst) -> sint: return _suop(argfix(*lst), UOp.minimum, min)
def srender(x:sint) -> str: return x.render() if isinstance(x, UOp) else str(x)
def _align_left(*shapes:tuple[sint, ...]) -> tuple[tuple[sint, ...], ...]:
  max_dim = max(len(s) for s in shapes)
  return tuple((1,)*(max_dim-len(s))+s for s in shapes)
def _broadcast_shape(*shapes:tuple[sint, ...]) -> tuple[sint, ...]:
  shaped_aligned_left = _align_left(*shapes)
  ret = tuple(0 if 0 in nth_dim_sizes else smax(nth_dim_sizes) for nth_dim_sizes in zip(*shaped_aligned_left))
  if not all(resolve(s == ns) or resolve(s == 1) for shape in shaped_aligned_left for s,ns in zip(shape, ret)):
    raise IndexError(f"shape mismatch: objects cannot be broadcast to a single shape {shapes}")
  return ret

def ssimplify(uop:sint): return uop.ssimplify() if isinstance(uop, UOp) else uop
def sym_infer(uop: UOp|int, var_vals: dict[str, int]) -> int: return uop.sym_infer(var_vals) if isinstance(uop, UOp) else uop

def range_str(u:UOp, color=False) -> str:
  ret = '_'.join([str(x) if x >= 0 else "m"+str(-x) for x in u.arg[0:-1]])
  return colored(ret, axis_colors[u.arg[-1]]) if color else ret

def multirange_str(rngs:Iterable[UOp], color=False, pad=None) -> str:
  ret = ','.join([range_str(x, color=color) for x in sorted(rngs, key=lambda x: x.arg)])
  if pad is not None: ret += " " * (pad-ansilen(ret))
  return ret

def shape_to_shape_arg(arg:tuple[sint, ...]) -> UOp:
  if len(arg) == 0: return UOp(Ops.STACK, dtypes.weakint.vec(0))
  elif all_int(arg): return UOp.const(dtypes.weakint.vec(len(arg)), arg)
  else: return UOp(Ops.STACK, dtypes.weakint.vec(len(arg)), tuple(UOp.const(dtypes.weakint, x) if isinstance(x, int) else x for x in arg))

def consumer_map_from_toposort(lst:Iterable[UOp]):
  ret: dict[UOp, dict[UOp, None]] = {}
  for u in lst:
    ret[u] = {}
    for s in u.src:
      if s in ret: ret[s][u] = None
  return ret

class UOpMetaClass(type):
  ucache:dict[tuple, weakref.ReferenceType[UOp]] = {}
  def __call__(cls, op:Ops, dtype:DType=dtypes.void, src:tuple[UOp,...]=tuple(), arg:Any=None, tag:Any=None,
               metadata:tuple[Metadata,...]|None=None, _buffer:Buffer|None=None):
    if (wret:=UOpMetaClass.ucache.get(key:=(op, dtype, src, arg, tag), None)) is not None and (ret:=wret()) is not None: return ret
    UOpMetaClass.ucache[key] = weakref.ref(created:=super().__call__(*key))
    if metadata is not None: all_metadata[created] = metadata
    # NOTE: this value is set by pickle when pickling a realized tensor
    if _buffer is not None:
      assert op is Ops.BUFFER, f"trying to set Buffer {_buffer} for {op}"
      buffers[created] = _buffer
    if SPEC > 1:
      from tinygrad.uop.spec import spec_full, test_pyrender
      if SPEC > 2:
        # SPEC=3 checks the shape
        _ = created._shape
        if SPEC > 3:
          test_pyrender(created)
      with Context(CHECK_OOB=0): fret = cast(bool|None, spec_full.rewrite(created))
      if fret is not True: raise RuntimeError(f"SPEC ISSUE {fret}: {created}")
    return created

# some uops map to other stuff
buffers:weakref.WeakKeyDictionary[UOp, Buffer|MultiBuffer] = weakref.WeakKeyDictionary() # this maps BUFFER/SLICE uops to their device Buffers
all_metadata:weakref.WeakKeyDictionary[UOp, tuple[Metadata, ...]] = weakref.WeakKeyDictionary() # TODO: should this be here?

# recursive_property replaces functools.cached_property in recursive UOp functions to prevent RecursionError
class recursive_property(property):
  def __init__(self, fxn):
    self.fxn = fxn
    self.nm = "_RECURSIVE_PROPERTY_"+fxn.__name__
    self.__doc__ = fxn.__doc__
  def __get__(self, x:UOp|None, owner=None):
    if x is None: return self
    if self.nm in x.__dict__: return x.__dict__[self.nm]
    for node in x.toposort(gate=lambda node: self.nm not in node.__dict__): node.__dict__[self.nm] = self.fxn(node)
    return x.__dict__[self.nm]

# we import this late so we can use resolve/smax in mixins
from tinygrad.mixin import OpMixin
from tinygrad.mixin.rand import RandMixin

# NOTE: this should be frozen, but frozen is slower
@dataclass(eq=False, slots=True)
class UOp(RandMixin, metaclass=UOpMetaClass):
  op:Ops
  dtype:DType = dtypes.void
  src:tuple[UOp, ...] = tuple()
  arg:Any = None
  tag:Any = None
  def __del__(self):
    if Ops is not None and self.op is Ops.BUFFER and (buffer:=buffers.get(self)) is not None: buffer.ref(-1)
    try: del UOpMetaClass.ucache[(self.op, self.dtype, self.src, self.arg, self.tag)]
    except AttributeError: pass
  def __reduce__(self):
    args = [self.op, self.dtype, self.src, self.arg, self.tag, self.metadata]
    if self.op is Ops.BUFFER and self.realized is not None: args.append(self.realized)
    return UOp, tuple(args)
  def replace(self, **kwargs) -> UOp:
    new_args = (kwargs.pop("op", self.op), kwargs.pop("dtype", self.dtype), kwargs.pop("src", self.src),
                kwargs.pop("arg", self.arg), kwargs.pop("tag", self.tag))
    assert len(kwargs) == 0, f"unused kwargs in replace {list(kwargs)}"
    if (self.op, self.dtype, self.src, self.arg, self.tag) == new_args: return self
    return UOp(*new_args)
  def rtag(self, tag=True): return self.replace(tag=tag)
  @recursive_property
  def key(self) -> bytes:
    return hashlib.sha256(str((self.op, self.dtype, self.arg)).encode() + b"".join([s.key for s in self.src])).digest()
  def __repr__(self):
    from tinygrad.uop.render import pretty_print
    return pretty_print(self)
  def argstr(self):
    if self.op is Ops.REDUCE: return f'({", ".join(map(str, self.arg))})'
    return f"ConstFloat({float.__repr__(self.arg)})" if isinstance(self.arg, ConstFloat) else repr(self.arg)
  def tagstr(self): return f", tag={self.tag}" if self.tag is not None else ""

  def f(self, op, **kwargs): return UOp(op, dtype=kwargs.pop("dtype", self.dtype), src=(self,), **kwargs)

  @functools.cached_property
  def backward_slice(self:UOp) -> dict[UOp, None]:
    res: dict[UOp, None] = self.toposort()
    res.pop(self)
    return res

  @property
  def backward_slice_with_self(self:UOp) -> dict[UOp, None]: return {self:None, **self.backward_slice}
  def op_in_backward_slice_with_self(self, *ops:Ops) -> bool:
    # Check self first, then iterate backward_slice (avoids creating intermediate dict)
    return self.op in ops or any(x.op in ops for x in self.backward_slice)

  def toposort(self, gate:Callable|None=None, enter_calls=True) -> dict[UOp, None]:
    cache: dict[UOp, None] = {}
    stack: list[tuple[UOp, bool]] = [(self, False)] # each stack entry is (node, visited_flag)
    while stack:
      node, visited = stack.pop()
      if node in cache: continue
      if not visited:
        if gate is None or gate(node):
          stack.append((node, True))  # push node back on stack to process after its srcs
          for s in reversed(node.src if enter_calls or node.op not in {Ops.CALL, Ops.FUNCTION} else node.src[1:]):
            stack.append((s, False)) # push srcs on the stack
      else: cache[node] = None # second time i'm seeing this node, add it to returned toposort
    return cache

  def topovisit(self, visitor:Callable[[UOp], T], cache:dict[UOp, T]) -> T:
    # NOTE: this shares a lot of code with toposort
    stack: list[tuple[UOp, bool]] = [(self, False)]
    while stack:
      node, visited = stack.pop()
      if node in cache: continue
      if not visited:
        stack.append((node, True))
        for s in reversed(node.src): stack.append((s, False))
      else: cache[node] = visitor(node)
    return cache[self]

  @functools.cached_property
  def tuplize(self:UOp) -> tuple:
    return (self.op.value, self.arg, self.dtype,)+tuple([x.tuplize for x in self.src])

  @property
  def ptrdtype(self) -> PtrDType:
    if not isinstance(self.dtype, PtrDType): raise RuntimeError(f"ptrdtype called on UOp with type {self.dtype}")
    return self.dtype

  # *** uop shape stuff ***

  @recursive_property
  def _shape(self) -> tuple[sint, ...]|None:
    match self.op:
      # late ops don't have shape
      case Ops.UNIQUE | Ops.LUNIQUE | Ops.DEVICE | Ops.IF | Ops.BARRIER | Ops.CUSTOM | \
           Ops.SINK | Ops.END | Ops.REWRITE_ERROR | Ops.PTRCAT | Ops.ENDIF | \
           Ops.LINEAR | Ops.PROGRAM | Ops.SOURCE | Ops.INS | Ops.TUPLE | Ops.CALL | Ops.FUNCTION:
        return None

      # special (terrible) case for RESHAPE on NOOP
      case Ops.RESHAPE:
        if self.src[0].op is Ops.NOOP: return self.marg

      # hacks for NOOP
      case Ops.NOOP | Ops.MEMORY_SEMANTIC | Ops.SCOPED_VALUE:
        return self.src[0]._shape if len(self.src) >= 1 else None

      case Ops.GETTUPLE:
        # GETTUPLE extracts from a TUPLE (possibly through a FUNCTION)
        in_tuple = self.src[0].src[0] if self.src[0].op is Ops.FUNCTION else self.src[0]
        assert in_tuple.op is Ops.TUPLE
        inner_shape = in_tuple.src[self.arg]._shape
        if inner_shape is None: return None
        # if through a FUNCTION, substitute internal PARAMs in the shape with corresponding args
        if self.src[0].op is Ops.FUNCTION:
          return tuple(graph_rewrite(s, _pm_resolve_params, self.src[0].src[1:], walk=True) if isinstance(s, UOp) else s for s in inner_shape)
        return inner_shape

      case Ops.CAST:
        # if it has a vec dtype, set the shape
        if self.dtype.count > 1: return (self.dtype.count,)
        # when PTX casts from ptr to non ptr, remove the shape of the buffer
        if isinstance(self.src[0].dtype, PtrDType) and not isinstance(self.src[0].dtype, ImageDType) and not isinstance(self.dtype, PtrDType):
          return ()

      case Ops.INDEX:
        shp:list[sint] = []
        for s in self.src[1:]: shp.extend(list(s.shape))
        return tuple(shp) + self.src[0].shape[len(self.src[1:]):]

      case Ops.GEP:
        return (len(self.arg),) if len(self.arg) > 1 else ()
      case Ops.REDUCE_SLOT:
        # A slot is a projection of the composite result and has the ordinary
        # reduced tensor shape. It is not a hardware vector lane.
        if getattr(self.src[0].arg[0], "slot_shapes", ()):
          shape = _normalize_composite_shape(self.src[0].arg[0].slot_shapes[self.arg])
          if shape is not None: return shape
        return self.src[0]._shape
      case Ops.DEFERRED_REDUCE_SLOT:
        owner = self.src[0].arg if self.src[0].op is Ops.DEFERRED_REDUCE_OWNER else \
          (self.src[0].arg[0] if self.src[0].op is Ops.REDUCE and isinstance(self.src[0].arg, tuple) else None)
        if getattr(owner, "slot_shapes", ()):
          shape = _normalize_composite_shape(owner.slot_shapes[self.arg.slot])
          if shape is not None: return shape
        return self.src[0]._shape
      case Ops.DEFERRED_REDUCE_OWNER:
        # The shared owner schedules through its primary accumulator output;
        # scalar state members remain accessible only to deferred projections.
        if getattr(self.arg, "slot_shapes", ()) and len(self.arg.slot_shapes) > 2:
          return _normalize_composite_shape(self.arg.slot_shapes[2])
        return self.src[0]._shape
      case Ops.COMPOSITE_ACCUMULATOR:
        # Backend-neutral tuple carrier: arg is the per-slot logical shape.
        return tuple(self.arg) if isinstance(self.arg, tuple) else self.src[0]._shape
      case Ops.TILE_GATHER:
        # Scheduler-only carrier.  The argument is the explicit fragment
        # shape; no renderer may consume this until ownership lowering exists.
        return self.arg.fragment_shape
      case Ops.ROW_SOFTMAX_REPACK:
        # Scheduler-only QK-C -> PV-A bridge. Backend lowering must consume it
        # before program construction.
        return self.arg.fragment_shape
      case Ops.AMD_ROW_SOFTMAX_REPACK:
        # Physical wave32 QK-C -> PV-A bridge. Its vec16 dtype is the native
        # per-lane PV-A fragment, not a logical tensor dimension.
        return (self.dtype.count,)
      case Ops.AMD_ROW_SOFTMAX_SLOT: return (self.dtype.count,)
      case Ops.AMD_PACKED_FRAGMENT_LOAD: return (self.dtype.count,)
      case Ops.AMD_ATTENTION_LOOP_STATE: return (self.dtype.count,) if self.dtype != dtypes.void and self.dtype.count != 1 else ()
      case Ops.AMD_ATTENTION_OUTPUT_DRAIN | Ops.AMD_ATTENTION_STATS_DRAIN: return self.src[0]._shape
      case Ops.AMD_PV_C_LANE: return ()
      case Ops.SCOPED_REDUCE:
        # The first SCOPED_REDUCE source is its semantically identical
        # fallback.  It is deliberately source-visible so a compiler that
        # cannot lower the scoped form can fail closed without graph surgery.
        return self.src[0]._shape
      case Ops.ATTENTION:
        # src[0] is the ordinary, semantically identical fallback result. The
        # remaining sources are the explicit Q/K/V/(optional mask) inputs.
        return self.src[0]._shape if len(self.src) else None
      case Ops.STACK:
        if len(self.src) == 0: return ()
        if isinstance(self.dtype, PtrDType):
          # TODO: this is broken
          return self.src[0].shape
        else:
          return (len(self.src),) + self.src[0].shape
      # TODO: contract and unroll should be deleted
      case Ops.CONST | Ops.DEFINE_VAR | Ops.CONTRACT | Ops.UNROLL | Ops.VCAT:
        return (self.dtype.count,) if self.dtype.count > 1 else ()

      # some ops init the shape
      case Ops.GETADDR: return ()
      case Ops.GROUP: return ()
      case Ops.BIND | Ops.RANGE | Ops.SPECIAL: return ()
      case Ops.BINARY: return (len(self.arg),)
      case Ops.BUFFER: return self.src[0].as_shape if isinstance(self.arg, ParamArg) else (self.arg,)
      case Ops.SLICE:
        # HACK: SLICE is used inside kernels, so we set the shape to () if it's on an INDEX
        if self.src[0].op is Ops.INDEX: return ()
        return (self.arg,)
      case Ops.CUSTOM_FUNCTION: return None
      case Ops.STAGE:
        # STAGE adds the existing shape to the front, opposite of INDEX
        return tuple([int(r.vmax+1) for r in self.src[1:]])+self.src[0].shape
      case Ops.DEFINE_LOCAL | Ops.DEFINE_REG:
        if len(self.src) >= 1:
          # NOTE: this is the same as PARAM
          return tuple(self.src[0].sgep(i) for i in range(self.src[0].dtype.count))
        if isinstance(self.dtype, PtrDType):
          return (self.ptrdtype.size, self.dtype.count) if self.dtype.count > 1 else (self.ptrdtype.size,)
        return (self.dtype.count,) if self.dtype.count > 1 else ()
      case Ops.PARAM:
        if isinstance(self.dtype, ImageDType): return self.dtype.shape
        if isinstance(self.dtype, PtrDType): return (self.ptrdtype.size,)
        return tuple(self.src[0].sgep(i) for i in range(self.src[0].dtype.count)) if len(self.src) >= 1 else None

      # wmma output shape = accumulator shape (src[2])
      case Ops.WMMA | Ops.SHAPED_WMMA: return self.src[2]._shape

      case Ops.CUSTOMI if isinstance(self.arg, tuple) and self.arg[:1] == ("state_loop_read_v1",): return ()
      case Ops.CUSTOMI: return self.src[0]._shape if len(self.src) else None

      # passthrough ops
      case Ops.WAIT: return None
      case Ops.MSTACK | Ops.MSELECT | Ops.DETACH | Ops.CONTIGUOUS | Ops.CONTIGUOUS_BACKWARD | Ops.AFTER | Ops.LOAD | \
           Ops.COPY | Ops.ALLREDUCE | Ops.STORE:
        return self.src[0]._shape
      # REDUCE with empty axis is passthrough (lowered form)
      case Ops.REDUCE if len(self.arg[1]) == 0:
        # these can mismatch if there's a horizonal reduce
        return (self.dtype.count,) if self.dtype.count > 1 else ()

      # TODO: disallow shape changing bitcast
      case Ops.BITCAST:
        ps = self.src[0]._shape
        if ps is None: return None
        if (output_sz:=self.dtype.itemsize) != (input_sz:=self.src[0].dtype.itemsize):
          return ps[:-1]+(ssimplify((ps[-1]*input_sz) // output_sz),) if len(ps) > 0 else ps
        return ps

      # MULTI marker has no shape
      case Ops.MULTI if len(self.src) == 0: return None

    # movement ops change the shape
    # NOTE: ssimplify is required because the shape needs to be canonical for broadcasting and same shape checking
    if self.op in GroupOp.Movement.union({Ops.MULTI, Ops.REDUCE}):
      ps = self.src[0]._shape
      if ps is None: raise RuntimeError(f"movement op {self.op} requires shape, {self.src[0].op} doesn't have one")
      match self.op:
        case Ops.RESHAPE:
          if not all(x >= 0 for x in self.marg): raise ValueError(f"shape can't contain negative numbers {self.marg}")
          if prod(ps) != prod(self.marg): raise ValueError(f"bad reshape: {ps} -> {self.marg}")
          return self.marg
        case Ops.EXPAND:
          if len(ps) != len(self.marg) or not all(s==ns or (s==1 and ns>=0) for s,ns in zip(ps, self.marg)):
            raise ValueError(f"bad expand: {ps} -> {self.marg}")
          return self.marg
        case Ops.PERMUTE:
          if sorted(self.marg) != list(range(len(ps))): raise ValueError(f"invalid permutation {self.marg} of len {len(ps)}")
          return tuple(ps[i] for i in self.marg)
        case Ops.PAD:
          # TODO: why do i need resolve here?
          if len(ps) != len(self.marg) or not all(resolve(sz>=0) and resolve(0<=o) and resolve(o+s<=sz) for s,(o,sz) in zip(ps, self.marg)):
            raise ValueError(f"invalid pad {self.marg} for {ps}")
          return tuple(ssimplify(sz) for _,sz in self.marg)
        case Ops.SHRINK:
          # TODO: why do i need resolve here?
          if len(ps) != len(self.marg) or not all(resolve(0<=o) and resolve(sz>=0) and resolve(o+sz<=s) for s,(o,sz) in zip(ps, self.marg)):
            raise ValueError(f"invalid shrink {self.marg} for {ps}")
          return tuple(ssimplify(sz) for _,sz in self.marg)
        case Ops.FLIP:
          if len(ps) != len(self.marg) or not all(isinstance(x, bool) for x in self.marg): raise ValueError(f"bad flip on {ps}, {self.marg}")
          return ps
        case Ops.LAYOUT_TRANSFORM:
          if self.marg != "q4k_lane_partition": raise ValueError(f"unknown layout transform {self.marg!r}")
          return ps
        case Ops.MULTI: return tuple(s*len(self.device) if a == self.axis else s for a,s in enumerate(ps))
        case Ops.REDUCE:
          axis_arg = self.arg[1]
          if not isinstance(axis_arg, tuple) or not all(isinstance(x, int) and x>=0 and x<len(ps) for x in axis_arg):
            raise ValueError(f"invalid type for axis: {axis_arg}")
          ret = list(1 if i in axis_arg else s for i,s in enumerate(ps))
          return tuple(ret)

    if self.op in GroupOp.Unary.union({Ops.CAST}):
      assert len(self.src) == 1, "unary ops must have 1 src"
      return self.src[0]._shape

    # elementwise ops keep the shape the same. all inputs with shape must match
    if self.op in GroupOp.Broadcastable:
      input_shapes = [x._shape for x in self.src]
      assert len(self.src) > 0 and all(x is not None for x in input_shapes), f"None input shape not supported for {self.op}"
      if DISALLOW_BROADCAST and not all_same(input_shapes):
        raise RuntimeError(f"shape mismatch at {self.op}: {input_shapes} {[x.op for x in self.src]}")
      # broadcasting lives in _shape property now
      return _broadcast_shape(*input_shapes)

    # all Ops must be explicitly handled
    raise NotImplementedError(f"no shape handling for {self.op} with {self.dtype}")

  @property
  def shape(self) -> tuple[sint, ...]:
    if (ret:=self._shape) is None: raise RuntimeError(f"shape requested, but {self.op} doesn't have a shape")
    return ret

  @property
  def max_shape(self) -> tuple[int, ...]: return to_max_shape(self.shape)
  def max_numel(self) -> int: return prod(self.max_shape)

  @property
  def shard_shape(self) -> tuple[sint, ...]:
    if not isinstance(self.device, tuple) or self.axis is None: return self.shape
    return tuple(x//len(self.device) if i == self.axis else x for i,x in enumerate(self.shape))

  @property
  def max_shard_shape(self) -> tuple[int, ...]:
    if not isinstance(self.device, tuple) or self.axis is None: return self.max_shape
    return tuple(x//len(self.device) if i == self.axis else x for i,x in enumerate(self.max_shape))

  @functools.cached_property
  def ended_ranges(self) -> tuple[UOp, ...]:
    if self.op in range_start: return self.src[range_start[self.op]:]
    if self.op is Ops.AFTER: return tuple(flatten([x.ended_ranges for x in self.src[1:]]))
    if self.op is Ops.CONTRACT:
      contract_rng_ids = {rng_id for rng_id, _ in self.arg}
      return tuple(r for r in self.src[0].ranges if r.op is Ops.RANGE and r.arg[0] in contract_rng_ids)
    return ()

  # determine what ranges this is in
  @recursive_property
  def _ranges(self) -> dict[UOp, None]:
    ret: dict[UOp, None] = {}
    for s in self.src: ret.update(s.ranges)
    for er in self.ended_ranges:
      if er.op is Ops.RANGE:
        # if it's a single RANGE, we don't flow through it.
        ret.pop(er, None)
      else:
        # if it's not a RANGE, we include all ranges in srcs.
        # technically we shouldn't flow through these ranges either, but this is pre pm_add_control_flow so it's the same.
        for s in er.ranges: ret.pop(s, None)
    return ret

  @property
  def ranges(self) -> dict[UOp, None]:
    if self.op is Ops.RANGE: return {self:None} | self._ranges
    return self._ranges

  # *** uop evaluation ***

  def simplify(self, tracked=False):
    if self.op is Ops.CONST: return self
    if self.op is Ops.SINK and all(s.op is Ops.CONST or (s.op is Ops.STACK and len(s.src) == 0) for s in self.src): return self
    # late import!
    from tinygrad.uop.symbolic import symbolic
    with Context(TRACK_MATCH_STATS=0 if not tracked else TRACK_MATCH_STATS.value):
      return graph_rewrite(self, symbolic, name="simplify")
  def ssimplify(self) -> UOp|ConstType: return ret.arg if (ret:=self.simplify()).op is Ops.CONST else ret
  def sintify(self) -> sint: return self.arg if self.op is Ops.CONST else self
  def _eval(self, dtype, expected_type:Type[T]) -> T:
    assert self.dtype in dtype, f"eval with wrong dtype {self}"
    vmin, vmax = (simple_self:=self.simplify())._min_max
    if vmin != vmax: raise ValueError(f"eval failed to be a single number, range is {vmin} to {vmax} in {simple_self.render()}")
    assert isinstance(vmin, expected_type), f"vmin is wrong dtype {type(vmin)} != {expected_type}"
    return vmin
  def __bool__(self): return self._eval((dtypes.bool,), bool)
  def __int__(self): return self._eval(dtypes.ints, int)
  def __float__(self): return float(self._eval(dtypes.floats, float))
  def substitute(self, dvars:dict[UOp, UOp], name:str|None=None, extra_pm:PatternMatcher|None=None, walk:bool=False):
    dvars = {k:v for k,v in dvars.items() if k is not v}
    if len(dvars) == 0: return self
    with Context(TRACK_MATCH_STATS=(0 if name is None else TRACK_MATCH_STATS.value)):
      return graph_rewrite(self, (extra_pm+_substitute) if extra_pm is not None else _substitute, dvars,
                           bottom_up=True, walk=walk, name=name)
  # NOTE: this is not called by Tensor slice (Tensor handles UOps directly), but satisfies SupportsIndex for type checking
  def __index__(self): return self.__int__()

  # *** uop tracing stuff ***

  @recursive_property
  def trace_num(self):
    num = next(ucount)
    uop_fields[num] = (self.op, self.dtype, tuple(s.trace_num for s in self.src), self.arg, self.tag)+((self.metadata,) if TRACEMETA>=2 else ())
    return num

  # *** uop syntactic sugar ***

  def sink(*srcs:UOp|None, **kwargs):  # pylint: disable=no-self-argument
    return UOp(Ops.SINK, dtypes.void, tuple([x for x in srcs if x is not None]), **kwargs)
  def maketuple(*srcs:UOp):  # pylint: disable=no-self-argument
    return UOp(Ops.TUPLE, dtypes.void, srcs)
  def gettuple(self, idx:int) -> UOp:
    in_tuple = self.src[0] if self.op is Ops.FUNCTION else self
    assert in_tuple.op is Ops.TUPLE, f"gettuple requires FUNCTION or TUPLE source, got {self.op}"
    return UOp(Ops.GETTUPLE, in_tuple.src[idx].dtype, (self,), idx)
  def group(*srcs:UOp|None):  # pylint: disable=no-self-argument
    if len(srcs) == 1 and isinstance(srcs[0], UOp): return srcs[0]
    return UOp(Ops.GROUP, dtypes.void, tuple([x for x in srcs if x is not None]))
  def vectorize(self, *srcs):
    return UOp(Ops.STACK, self.dtype.vec(len(srcs)+1), (self,)+srcs)
  def index(self, *srcs:UOp|None, ptr=False, **kwargs):
    # Preserve composite provenance through indexed views, but only for the
    # validated composite tags emitted by composite_reduce_slot()/expansion.
    # Ordinary INDEX nodes remain untagged and therefore cannot masquerade as
    # REDUCE_SLOT projections.
    tag = kwargs.pop("tag", None)
    if tag is None and isinstance(self.tag, tuple) and self.tag and self.tag[0] in ("composite_slot", "composite_reduce", "composite_view"):
      tag = ("composite_view", self.tag)
    return UOp(Ops.INDEX, kwargs.pop("dtype", self.dtype if ptr else self.dtype.base), (self,)+tuple([x for x in srcs if x is not None]), tag=tag, **kwargs)
  def __getitem__(self, idx):
    # pointers index into INDEX UOps (scalar lookup); everything else uses the shared mixin view path
    if not isinstance(self.dtype, PtrDType): return super(UOp, self).__getitem__(idx)
    idx = self._normalize_indices(list(argfix(idx)))
    if len(slice_idx:=[i for i,x in enumerate(idx) if isinstance(x, slice)]):
      # apply SHRINK for slices that aren't the full range
      bounds = tuple((s.start or 0, s.stop if s.stop is not None else self.shape[i]) if isinstance(s, slice) else (0, self.shape[i])
                     for i, s in enumerate(idx))
      src = self if all(b == (0, self.shape[i]) for i, b in enumerate(bounds)) else self.shrink(bounds)
      non_slice_args = [UOp.const(dtypes.weakint, x) if isinstance(x, int) else x for x in idx if not isinstance(x, slice)]
      if not non_slice_args: return src  # all dims are slices, no indexing needed
      perm = src.permute(tuple([i for i in range(src.ndim) if i not in slice_idx] + slice_idx))
      return perm.index(*non_slice_args, ptr=True)
    return self.index(*[UOp.const(dtypes.weakint, x) if isinstance(x, int) else x for x in idx])
  @property
  def _uop(self) -> UOp: return self
  def _wrap_uop(self, u:UOp) -> UOp: return u
  def const_like(self, b:ConstLike, dtype:DType|None=None):
    return UOp.const(dtype or self.dtype.base, b, shape=self._shape)
  def ufix(self, x):
    if isinstance(x, UOp): return x
    return self.const_like(x, None if self._ufix_keep_dtype(x) else dtypes.from_py(x).vec(self.dtype.vcount))
  def broadcast(self, count:int):
    assert self.dtype.vcount == 1
    if count == 1: return self
    return UOp(Ops.STACK, self.dtype.vec(count), (self,)*count)
  def cast(self, dtype:DType):
    # TODO: we shouldn't have to check for dtype.count == 1 here, but CAST is misused in AMD LLVM
    if dtype.count == 1 and dtype.count != self.dtype.count: dtype = dtype.vec(self.dtype.count)
    if self.dtype == dtype: return self
    return UOp(Ops.CAST, dtype, (self,))
  def bitcast(self, dtype:DType): return self if self.dtype == dtype else UOp(Ops.BITCAST, dtype, (self,))
  def gep(self, i:tuple[int, ...]|int):
    if isinstance(i, tuple) and len(i) == 1: return self.gep(i[0])
    if isinstance(i, int):
      # NOTE: these are just shortcuts to not have to create and fold later
      if self.op is Ops.STACK: return self.src[i]
      if self.op is Ops.CONST: return UOp.const(self.dtype.scalar(), self.arg)
      i = (i,)
    return UOp(Ops.GEP, self.dtype.scalar().vec(len(i)) if len(i) > 1 else self.dtype.scalar(), (self,), i)
  def load(self, *src:UOp, **kwargs): return UOp(Ops.LOAD, dtype=kwargs.pop("dtype", self.dtype.base), src=(self,)+src, **kwargs)
  def store(self, src:UOp|ConstType, gate:UOp|None=None, **kwargs):
    srcs = (self, self.const_like(src) if not isinstance(src, UOp) else src) + ((gate,) if gate is not None else ())
    return UOp(Ops.STORE, dtypes.void, srcs, **kwargs)
  def wait(self, src:UOp|ConstType, **kwargs):
    return UOp(Ops.WAIT, dtypes.void, (self, self.const_like(src) if not isinstance(src, UOp) else src), **kwargs)
  def end(self, *src:UOp): return UOp(Ops.END, src=(self,)+src) if len(src) else self
  def after(self, *src:UOp, **kwargs): return UOp(Ops.AFTER, self.dtype, (self,)+src, **kwargs) if len(src) else self
  def barrier(self, *src:UOp): return UOp(Ops.BARRIER, src=(self,)+src)
  def ins(self, arg, **kwargs): return UOp(Ops.INS, kwargs.pop("dtype", self.dtype), kwargs.pop("src", self.src), arg, kwargs.pop("tag", self.tag))
  def contract(self, *rngs:UOp):
    assert all(x.arg[-1] == AxisType.UPCAST for x in rngs), "all contract ranges must be upcast"
    return UOp(Ops.CONTRACT, dtype=self.dtype.vec(prod([x.vmax+1 for x in rngs])), src=(self,), arg=tuple((x.arg[0], x.vmax+1) for x in rngs))
  def alu(self, op, *src:UOp, **kwargs):
    all_srcs = (self, *src)
    # broadcast shaped operands to a common shape (None and () are falsy, so only real shapes participate)
    if (shapes := [s for x in all_srcs if (s:=x._shape)]) and not all_same(shapes):
      out_shape = _broadcast_shape(*shapes)
      all_srcs = tuple(x._broadcast_to(out_shape) if x._shape else x for x in all_srcs)
    out_dtype = all_srcs[-1].dtype
    if op in {Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ}: out_dtype = dtypes.bool.vec(out_dtype.count) if out_dtype.count > 1 else dtypes.bool
    return UOp(op, out_dtype, all_srcs, **kwargs)
  @staticmethod
  def const(dtype:DType, b:ConstLike, shape:tuple[sint, ...]|None=None):
    if isinstance(b, UOp): return b.cast(dtype)
    if isinstance(b, tuple) and all_same(b):
      assert len(b) > 0, "can't create const from empty tuple"
      b = b[0]  # doesn't have to be a STACK if they are all the same
    if isinstance(b, tuple):
      stk = [UOp(Ops.CONST, dtype.scalar(), arg=dtype.const(c), src=()) for c in b]
      ret = UOp.vectorize(*stk)
    else:
      ret = UOp(Ops.CONST, dtype, arg=dtype.const(b), src=())
    return ret.reshape((1,)*len(shape)).expand(shape) if shape is not None and shape != () and ret.shape != shape else ret
  @staticmethod
  def unique_const(fill_value:ConstType, dtype:DTypeLike|None=None, device:str|tuple[str, ...]|None=None,  # type: ignore[override]
                   shape:tuple[sint, ...]|None=None, unique=True):
    # NOTE: fill_value is ConstType, not ConstLike, so UOps and tuples aren't allowed
    assert not isinstance(fill_value, (UOp, tuple)), "unique const only works on numbers"
    dt = to_dtype(dtype) if dtype is not None else dtypes.from_py(fill_value)
    ret = UOp(Ops.CONST, dt, arg=dt.const(fill_value),
              src=(UOp.unique(None if unique is True else unique), UOp(Ops.DEVICE, arg=canonicalize_device(device))))
    return ret.reshape((1,)*len(shape)).expand(shape) if shape is not None and ret.shape != shape else ret
  @staticmethod
  def range(end:sint, axis_id, axis_type=AxisType.LOOP, *arg, dtype=dtypes.weakint, src=(), **kwargs):
    return UOp(Ops.RANGE, dtype=dtype, src=(sint_to_uop(end, dtype),)+src, arg=(axis_id, axis_type)+arg, **kwargs)
  @staticmethod
  def special(end:sint, name:str, dtype=dtypes.weakint): return UOp(Ops.SPECIAL, dtype=dtype, src=(sint_to_uop(end, dtype),), arg=name)
  def _rop(self, op:Ops, axis:tuple[int, ...]):
    axis = tuple(sorted([x for x in axis if resolve(self.shape[x] != 1)]))
    return UOp(Ops.REDUCE, self.dtype, (self,), (op, axis)) if len(axis) else self
  @staticmethod
  def invalid(count=1): return UOp(Ops.CONST, dtypes.weakint.vec(count), src=(), arg=Invalid)
  def valid(self, cond):
    return cond.where(self.cast(dtypes.weakint), UOp.invalid(self.dtype.count))
  def get_idx(self) -> UOp:
    assert self.dtype.scalar() is dtypes.weakint, "Can only call get_idx on index dtype"
    return self.src[1] if self.op is Ops.WHERE and self.src[2].arg is Invalid else self
  def get_valid(self) -> UOp:
    assert self.dtype.scalar() is dtypes.weakint, "Can only call get_valid on index dtype"
    return self.src[0] if self.op is Ops.WHERE and self.src[2].arg is Invalid else UOp.const(dtypes.bool, self.arg is not Invalid)
  def reduce(self, *src:UOp, **kwargs):
    arg = kwargs.pop('arg', None)
    if isinstance(arg, Ops): arg = (arg, ())
    return UOp(Ops.REDUCE, kwargs.pop('dtype', self.dtype), src=(self,)+src, arg=arg, **kwargs)

  def composite_reduce(self, *slots: 'AccumulatorSlot', axis: tuple[int, ...] = (), inputs: tuple['UOp', ...] = (),
                       input_specs: tuple['CompositeInputSpec', ...]|None = None,
                       tile_carrier: 'CompositeTileCarrier|None' = None,
                       slot_shapes: tuple[tuple|None, ...]|None = None,
                       lane_shapes: tuple[tuple|None, ...]|None = None, **kwargs):
    """Create a stateful REDUCE. All logical-element inputs are explicit sources."""
    from tinygrad.uop.ops import CompositeReduce, AccumulatorSlot
    # Compatibility is source-visible: callers formerly passing v_uop now get
    # the same dependency in REDUCE.src rather than invisible metadata.
    legacy_v = kwargs.pop('v_uop', None)
    if legacy_v is not None:
      if inputs: raise ValueError("pass auxiliary composite inputs once, using inputs=")
      inputs = (legacy_v,)
    if input_specs is None:
      input_specs = tuple(CompositeInputSpec("logical", x.arg.axis_map if x.op is Ops.SCOPED_VALUE and hasattr(x.arg, "axis_map") else None)
                          for x in inputs)
    if len(input_specs) != len(inputs): raise ValueError("one CompositeInputSpec is required for each composite input")
    for spec in input_specs:
      if hasattr(spec, "validate_lane_abi"): spec.validate_lane_abi()
    if tile_carrier is not None:
      tile_carrier.validate()
    normalized_lanes = tuple(_normalize_composite_shape(s) for s in (lane_shapes or ()))
    if normalized_lanes and len(normalized_lanes) != len(slots):
      raise ValueError("one physical lane shape is required for each composite slot")
    for lane_shape in normalized_lanes:
      if lane_shape is None or any(not isinstance(x, int) or x < 1 for x in lane_shape):
        raise ValueError("composite physical lane shapes must contain positive integer dimensions")
    composite = CompositeReduce(slots=tuple(slots), combine_fn=kwargs.pop('combine_fn', None),
                                input_specs=tuple(input_specs), tile_carrier=tile_carrier,
                                slot_shapes=tuple(_normalize_composite_shape(s) for s in (slot_shapes or ())),
                                lane_shapes=normalized_lanes,
                                attention_grid=kwargs.pop('attention_grid', None),
                                attention_causal=kwargs.pop('attention_causal', False),
                                attention_context=kwargs.pop('attention_context', None))
    return UOp(Ops.REDUCE, kwargs.pop('dtype', self.dtype), src=(self,)+tuple(inputs), arg=(composite, axis), **kwargs)

  def composite_reduce_slot(self, slot: int, *, dtype=None):
    """Construct a typed projection while preserving composite provenance."""
    if self.op is not Ops.REDUCE or not hasattr(self.arg[0], 'slots'):
      raise ValueError("composite_reduce_slot requires a composite REDUCE")
    composite = self.arg[0]
    if not isinstance(slot, int) or not 0 <= slot < len(composite.slots):
      raise ValueError(f"invalid composite reduction slot {slot}")
    return UOp(Ops.REDUCE_SLOT, dtype or composite.slots[slot].dtype, (self,), slot,
               tag=("composite_slot", composite, self))

  def scoped_reduce(self, producer: 'UOp', *logical_inputs: 'UOp', axis: tuple[int, ...], axis_maps: tuple[tuple[int, ...], ...],
                    scope_owner: int, result_dtypes: tuple[Any, ...]|None=None) -> 'UOp':
    """Create a nested-reduction semantic boundary.

    ``self`` is the ordinary fallback result. ``producer`` is the inner
    reduction body. Every source after the fallback has one explicit map from
    its axes to the outer logical axes (``-1`` marks a value-local axis).
    ``scope_owner`` names the outer reduction loop that owns the producer.
    """
    src = (self, producer) + logical_inputs
    if len(axis_maps) != len(src)-1: raise ValueError("one axis map is required for the producer and each logical input")
    if result_dtypes is None: result_dtypes = (self.dtype,)
    spec = ScopedReduceSpec(tuple(axis), tuple(tuple(x) for x in axis_maps), scope_owner, tuple(result_dtypes))
    spec.validate_producer(producer)
    return UOp(Ops.SCOPED_REDUCE, self.dtype, src=src, arg=spec)

  def scoped_value(self, slot_or_axis_map:int|tuple[int|None, ...]=0) -> 'UOp':
    """Create either a typed scoped result or an owned logical input.

    SCOPED_REDUCE sources use an integer result slot. All other values use an
    axis map and remain unindexed until their owning reduction lowers.
    """
    if self.op is Ops.SCOPED_REDUCE:
      if not isinstance(slot_or_axis_map, int): raise ValueError("SCOPED_REDUCE result requires an integer slot")
      if not 0 <= slot_or_axis_map < len(self.arg.result_dtypes): raise ValueError(f"invalid scoped result slot {slot_or_axis_map}")
      return UOp(Ops.SCOPED_VALUE, self.arg.result_dtypes[slot_or_axis_map], src=(self,), arg=slot_or_axis_map)
    if isinstance(slot_or_axis_map, int): raise ValueError("logical scoped value requires an axis map")
    if len(slot_or_axis_map) != len(self.shape): raise ValueError("scoped value axis map rank mismatch")
    return UOp(Ops.SCOPED_VALUE, self.dtype, (self,), ScopedValueSpec(tuple(slot_or_axis_map)))

  def contiguous(self, *args, **kwargs):
    if self.op is Ops.CONTIGUOUS: return self
    if self.device is None: return self
    if self.has_buffer_identity(): return self
    return UOp(Ops.CONTIGUOUS, dtype=self.dtype, src=(self,)+args, **kwargs)
  def bufferize(self, *args, **kwargs): return UOp(Ops.STAGE, dtype=self.dtype, src=(self,)+args, **kwargs)
  def allreduce(self, op, device:str|tuple[str, ...]|UOp):
    assert isinstance(self.device, tuple), f"allreduce must be on tuple {self.device} isn't"
    return UOp(Ops.ALLREDUCE, self.dtype, (self, UOp(Ops.DEVICE, arg=device) if not isinstance(device, UOp) else device), op)
  def overflows(self, dtype:DType) -> bool: return self.vmin < dtype.min or dtype.max < self.vmax

  def split_uop(self:UOp, sep:Ops) -> Iterator[UOp]:
    if self.op is sep:
      for s in self.src: yield from s.split_uop(sep)
    else: yield self

  @property
  def reg(self:UOp):
    # TODO: add a way to access the nth element in src
    if self.op in (Ops.NOOP, Ops.AFTER) and self.src: return self.src[0].reg
    if isinstance(self.tag, tuple): return self.tag[0]
    return self.tag

  # *** multi-device helpers ***

  def multi(self, axis:int|None):
    assert isinstance(self.device, tuple), f"multi device must be tuple, {self.device} isn't"
    assert axis is not None, "multi None is no longer supported"
    return UOp(Ops.MULTI, self.dtype, (self,), axis)

  @property
  def bounds(self):
    if self.axis is None: raise RuntimeError("bounds is not defined when axis is None")
    return tuple(itertools.pairwise(itertools.accumulate([self.src[0].shape[self.axis] for _ in self.device], initial=0)))

  @functools.cached_property
  def axis(self) -> int|None:
    # COPY removes axis. TODO: add more tests for this, and consider MSELECT/MSTACK
    if self.op is Ops.COPY: return None
    if self.op is Ops.MULTI: return self.arg
    # GETTUPLE: axis comes from the specific TUPLE element, not src[0]
    if self.op is Ops.GETTUPLE:
      in_tuple = self.src[0].src[0] if self.src[0].op is Ops.FUNCTION else self.src[0]
      return in_tuple.src[self.arg].axis if in_tuple.op is Ops.TUPLE else None
    if self.op is Ops.PARAM: return self.arg.axis
    # NOTE: they all have to share an axis, we always choose [-1]
    if self.op in GroupOp.ALU: return axes[-1] if (axes := dedup([x.axis for x in self.src if x.axis is not None])) else None
    if len(self.src) == 0: return None
    src_axis = self.src[0].axis
    if self.op is Ops.SHRINK and src_axis is not None and self.marg[src_axis] != (0, self.src[0].shape[src_axis]):
      return None # SHRINK will remove the sharding if it's on axis
    if self.op is Ops.REDUCE: return None if src_axis is not None and src_axis in self.arg[1] else src_axis
    if self.op is Ops.RESHAPE:
      if src_axis is None: return None
      arg_acc:list[sint] = list(itertools.accumulate(self.marg, operator.mul, initial=1))
      # new_axis is the last one that preserves prod(prior to new_axis) and must not move items between shards
      new_axis = len(arg_acc) - arg_acc[::-1].index(prod(self.src[0].shape[:src_axis])) - 1
      if self.shape[new_axis] % len(self.device) != 0: raise RuntimeError(f"reshape {self.src[0].shape} -> {self.shape} moved items between shards")
      return new_axis
    if self.op is Ops.PERMUTE: return self.marg.index(src_axis) if src_axis is not None else None
    return src_axis

  def _unshard(self, axis:int) -> UOp:
    bsz, dcount = self.shape[axis], len(self.device)
    dnum = UOp.variable("_device_num", 0, dcount-1)
    return self.pad(tuple((0,0) if a != axis else (bsz*dnum, bsz*(dcount-1) - bsz*dnum) for a in range(len(self.shape))))

  def _shard(self, axis:int, dcount:int) -> UOp:
    if len(self.shape) == 0: return self  # scalars broadcast, no sharding needed
    dnum = UOp.variable("_device_num", 0, dcount-1)
    if self.shape[axis] % dcount != 0: raise RuntimeError(f"multi axis uneven: {self.shape[axis]=} {axis=} {dcount=}")
    sz = self.shape[axis] // dcount
    return self.shrink(tuple((0,s) if i != axis else (dnum*sz,dnum*sz+sz) for i,s in enumerate(self.shape)))
  def shard(self, devices:tuple[str, ...], axis:int|None=None) -> UOp:
    copied = self.copy_to_device(devices)
    return copied if axis is None else copied._shard(axis, len(devices)).multi(axis)

  def copy_to_device(self, device:str|tuple[str, ...]|UOp, arg=None):
    assert arg is None or isinstance(self.device, tuple)
    inp = self if arg is None else UOp(Ops.MSELECT, self.dtype, src=(self,), arg=arg)
    return UOp(Ops.COPY, self.dtype, (inp, UOp(Ops.DEVICE, arg=device) if not isinstance(device, UOp) else device))
  def mselect(self, arg:int) -> UOp: return UOp(Ops.MSELECT, self.dtype, (self,), arg)
  def mstack(self, *srcs: UOp) -> UOp: return UOp(Ops.MSTACK, self.dtype, (self,)+srcs)
  @property
  def metadata(self) -> tuple[Metadata, ...]|None: return all_metadata.get(self, None)

  # *** uop movement ops ***

  @property
  def base(self) -> UOp:
    if self.op is Ops.MEMORY_SEMANTIC: return self.src[0].base
    if self.op in GroupOp.Movement: return self.src[0].base
    if self.op is Ops.MULTI: return self.src[0].base  # MULTI is really a VIEW
    if self.op is Ops.DETACH: return self.src[0].base  # DETACH can't change base
    return self

  @property
  def multibase(self) -> UOp:
    if self.op is Ops.MEMORY_SEMANTIC: return self.src[0].multibase
    if self.op in GroupOp.Movement: return self.src[0].base
    if self.op is Ops.DETACH: return self.src[0].base  # DETACH can't change base
    return self

  # like gep, but might return an integer
  def sgep(self, i:int) -> sint:
    match self.op:
      case Ops.CONST: return self.arg
      case Ops.STACK: return self.src[i].sintify()
      case _: raise RuntimeError(f"no sgep on {self.op}")

  # cached property here makes external_uop_gc fail, why?
  @property
  def as_shape(self) -> tuple[sint, ...]:
    if self.op is Ops.CONST: return (self.arg,)*self.dtype.count # NOTE: this will break
    if self.op is not Ops.STACK: return (ssimplify(self),)
    return tuple(ssimplify(self.sgep(i)) for i in range(len(self.src)))

  @functools.cached_property
  def marg(self):
    match self.op:
      case Ops.RESHAPE | Ops.EXPAND: return self.src[1].as_shape
      case Ops.PAD | Ops.SHRINK: return tuple(zip(self.src[1].as_shape, self.src[2].as_shape))
      case Ops.PERMUTE | Ops.FLIP | Ops.LAYOUT_TRANSFORM: return self.arg
      case _: raise RuntimeError(f"{self.op} is not a MovementOp")

  def _mop(self, op:Ops, arg) -> UOp:
    # early NOOP
    if op in {Ops.SHRINK, Ops.PAD, Ops.EXPAND} and len(arg) == 0:
      assert len(self.shape) == 0, "0 len arg only valid on zero length shape"
      return self
    match op:
      case Ops.RESHAPE | Ops.EXPAND: src_args = [arg]
      case Ops.PAD | Ops.SHRINK: src_args = list(zip(*arg))
      case Ops.PERMUTE | Ops.FLIP | Ops.LAYOUT_TRANSFORM: src_args = []
      case _: raise RuntimeError(f"{op} is not a MovementOp")
    usrcs = [shape_to_shape_arg(arg) for arg in src_args]
    if len(usrcs) == 0: return UOp(op, self.dtype, (self,), arg)
    return UOp(op, self.dtype, (self,)+UOp.sink(*usrcs).simplify().src)

  # *** uop UNIQUE ***

  # TODO: use this in Buffer
  unique_num = itertools.count(0)
  @staticmethod
  def unique(arg:int|None=None): return UOp(Ops.UNIQUE, arg=next(UOp.unique_num) if arg is None else arg)

  # *** uop Buffer stuff ***

  @staticmethod
  def new_buffer(device:str|tuple[str, ...], size:int, dtype:DType, num=None):
    return UOp(Ops.BUFFER, dtype, (UOp.unique(num), UOp(Ops.DEVICE, arg=device)), size)
  @staticmethod
  def from_buffer(opaque:Buffer, device:str|tuple[str, ...]|None=None):
    buffers[uop:=UOp.new_buffer(device or opaque.device, opaque.size, opaque.dtype)] = opaque.ref(1)
    return uop
  @staticmethod
  def empty(shape:tuple[sint, ...], dtype:DTypeLike|None=None, device:str|tuple[str, ...]|None=None, axis:int|None=None, num=None) -> UOp:
    dtype, device = to_dtype(dtype) if dtype is not None else dtypes.default_float, canonicalize_device(device)
    max_shape = to_max_shape(shape)
    ret = UOp.new_buffer(device, prod(max_shape), dtype, num).reshape(max_shape).shrink_to(shape)
    return ret.multi(axis) if isinstance(device, tuple) and axis is not None else ret
  def empty_like(self, dtype:DTypeLike|None=None, device:str|tuple[str, ...]|None=None) -> UOp:
    device = canonicalize_device(self.device if device is None else device)
    axis = self.axis if isinstance(device, tuple) else None
    return UOp.empty(self.shard_shape if axis is not None else self.shape, self.dtype if dtype is None else dtype, device, axis)
  @staticmethod
  def _frompy(x:list|tuple|bytes, dtype:DType, device:str|tuple[str, ...]|None=None) -> UOp:
    device = canonicalize_device(device)
    if isinstance(x, bytes): ret, data = UOp.new_buffer("PYTHON", len(x)//dtype.itemsize, dtype), x
    else:
      # bfloat16 and fp8 have no struct format, so pack a float32 buffer and cast
      bdtype = dtypes.float32 if dtype in [dtypes.bfloat16, *dtypes.fp8s] else dtype
      assert bdtype.fmt is not None, f"{bdtype=} has None fmt"
      ret = UOp.empty(shape:=get_shape(x), bdtype, "PYTHON")
      data = struct.pack(f"{prod(shape)}{bdtype.fmt}", *[truncate[bdtype](bdtype.const(xi)) for xi in fully_flatten(x)])
    # fake realize. if target device is PYTHON it needs bytearray to be writable
    ret.buffer.allocate(memoryview(data if device != "PYTHON" else bytearray(data)))
    if ret.dtype != dtype: ret = ret.cast(dtype)
    return ret if ret.device == device else ret.copy_to_device(device)
  def clone(self, device=None) -> UOp:
    device = device or self.device
    ret = self.empty_like(device=device)
    src = self if self.device is None or self.device == device else self.copy_to_device(device)
    return ret.after(ret.store(src))
  @recursive_property
  def device(self) -> str|tuple[str, ...]|None:
    if self.op is Ops.PARAM: return self.arg.device
    if self.op is Ops.DEVICE: return self.arg
    if self.op is Ops.STAGE: return self.arg.device
    if self.op is Ops.AFTER: return self.src[0].device
    if self.op is Ops.MSELECT:
      assert isinstance(self.src[0].device, tuple), f"mselect must be on tuple device, getting {self.src[0].device}"
      return self.src[0].device[self.arg]
    if self.op is Ops.MSTACK: return tuple(cast(str, x.device) for x in self.src)
    if self.op is Ops.BUFFER and isinstance(self.arg, ParamArg): return self.arg.device
    if self.op in {Ops.COPY, Ops.BUFFER, Ops.ALLREDUCE}: return self.src[1].device
    for x in self.src:
      if x.device is not None: return x.device
    return None
  @recursive_property
  def addrspace(self) -> AddrSpace|None:
    if self.op is Ops.PARAM: return self.arg.addrspace
    if self.op is Ops.BUFFER: return self.arg.addrspace if isinstance(self.arg, ParamArg) else AddrSpace.GLOBAL
    if self.op is Ops.DEFINE_LOCAL: return AddrSpace.LOCAL
    if self.op is Ops.DEFINE_REG: return AddrSpace.REG
    if self.op is Ops.LOAD: return AddrSpace.REG # LOAD brings things into registers
    if self.op in {Ops.INDEX, Ops.CAST, Ops.AFTER, Ops.REDUCE, Ops.GEP, Ops.STORE}:
      return self.src[0].addrspace
    if self.op in GroupOp.Movement: return self.src[0].addrspace
    if self.op in {Ops.STACK, Ops.WMMA} or self.op in GroupOp.Elementwise:
      ad = [x.addrspace for x in self.src if x.addrspace is not None]
      if not len(ad) or not all_same(ad): return None
      return ad[0]
    return None
  @property
  def buf_uop(self) -> UOp:
    if self.op in {Ops.BUFFER, Ops.PARAM}: return self
    if self.op is Ops.MSELECT: return self.src[0].buf_uop.mselect(self.arg)
    if self.op is Ops.MSTACK: return UOp(Ops.MSTACK, self.dtype, src=tuple(x.buf_uop for x in self.src))
    if self.base.op is Ops.AFTER: return self.base.src[0].buf_uop.base
    s = self
    while len(s.src) and s.op not in {Ops.BUFFER, Ops.PARAM, Ops.STAGE, Ops.MSTACK}: s = s.src[0]
    return s

  def contiguous_view_offset(self) -> int|None:
    """If movement ops on a BUFFER collapse to a contiguous range, return `offset` in elements. Otherwise None."""
    from tinygrad.schedule.rangeify import pm_mops
    from tinygrad.uop.symbolic import symbolic
    numel = self.numel()
    out = graph_rewrite(self.flatten().index(UOp.range(numel, 0)), pm_mops+symbolic, name="contiguous_view_offset")
    if out.op is not Ops.INDEX: return None
    if out.src[1].op is Ops.CONST and resolve(numel == 1, False):
      if not isinstance(out.src[1].arg, int): return None  # masked/padded regions produce InvalidType
      return out.src[1].arg
    if out.src[1].op is Ops.RANGE: return 0
    if out.src[1].op is Ops.ADD and out.src[1].src[0].op is Ops.RANGE and out.src[1].src[1].op is Ops.CONST:
      if not isinstance(out.src[1].src[1].arg, int): return None  # masked/padded regions produce InvalidType
      return out.src[1].src[1].arg
    return None

  def has_buffer_identity(self):
    """Check if this UOp has a concrete buffer identity in the graph (RESHAPE/MULTI -> BUFFER chain)."""
    if self.op in {Ops.RESHAPE, Ops.MULTI}: return self.src[0].has_buffer_identity()
    if self.op is Ops.GETTUPLE and self.src[0].op is Ops.TUPLE: return self.src[0].src[self.arg].has_buffer_identity()
    return self.op in {Ops.BUFFER, Ops.SLICE, Ops.PARAM}

  def _base_buffer_is_realized(self) -> bool:
    """Walk through AFTER chain to find if the underlying buffer is realized (has allocated memory)."""
    u = self.base
    while u.op is Ops.AFTER: u = u.src[0]
    return u.is_realized

  @property
  def buffer(self) -> Buffer|MultiBuffer:
    if self.op in {Ops.CONTIGUOUS, Ops.RESHAPE, Ops.DETACH, Ops.AFTER, Ops.MEMORY_SEMANTIC}: return self.src[0].buffer
    # this buffer can process disk tensors and simple movement ops
    if self is not self.base:
      buf = self.base.buffer
      assert isinstance(buf, Buffer), "must be a Buffer for movement ops"
      offset = self.contiguous_view_offset()
      if offset is None: raise RuntimeError(f"non-contiguous view is not supported for {buf.device} buffer")
      return buf.view(prod(self.max_shape), self.dtype, offset*self.dtype.itemsize)
    if self.op is Ops.BITCAST:
      buf = self.src[0].buffer
      assert isinstance(buf, Buffer), "must be a Buffer for BITCAST"
      return buf.view(prod(self.max_shape), self.dtype, 0)
    if self.op is Ops.SLICE:
      if (cret:=buffers.get(self)) is not None: return cret
      buf = self.src[0].buffer
      offset = self.src[1].arg
      if isinstance(buf, MultiBuffer):
        mbuf = MultiBuffer.__new__(MultiBuffer)
        mbuf.bufs = [b.view(self.arg, self.dtype, offset * self.src[0].dtype.itemsize) for b in buf.bufs]
        buffers[self] = mbuf
        return mbuf
      assert isinstance(buf, Buffer), "must be a Buffer for SLICE"
      buffers[self] = bv = buf.view(self.arg, self.dtype, offset * self.src[0].dtype.itemsize)
      return bv
    if self.op is Ops.MSELECT:
      ret = self.src[0].buffer
      assert isinstance(ret, MultiBuffer)
      return ret.bufs[self.arg]
    if self.op is Ops.MSTACK:
      ret = MultiBuffer.__new__(MultiBuffer)
      ret.bufs = [cast(Buffer, x.buffer) for x in self.src]
      assert all_same([(x.size, x.dtype) for x in ret.bufs]), "multibuffers mismatch buffers"
      return ret
    assert self.op is Ops.BUFFER, f"must be BUFFER {self.op}"
    assert self.src[0].op is Ops.UNIQUE, f"buffer src[0] must be UNIQUE, not {self.src[0].op}"
    if (cret:=buffers.get(self)) is not None: return cret
    rdtype = self.dtype if isinstance(self.dtype, ImageDType) else self.dtype.base
    if isinstance(self.device, tuple): ret = MultiBuffer(self.device, self.arg, rdtype).ref(1)
    else: ret = Buffer(self.device, self.arg, rdtype).ref(1)
    buffers[self] = ret
    return ret
  @property
  def realized(self) -> Buffer|MultiBuffer|None:
    # only these can be realized
    if self.op not in (Ops.BUFFER, Ops.MSTACK): return None
    # ParamArg is LOCAL/REG and never realized
    if self.op is Ops.BUFFER and isinstance(self.arg, ParamArg): return None
    # LUNIQUEs are never realized
    if self.op_in_backward_slice_with_self(Ops.LUNIQUE): return None
    # NOTE: this is used by the JIT to determine which inputs we capture
    return self.buffer if self.buffer.is_allocated() else None
  @property
  def is_realized(self) -> bool: return self.base.realized is not None

  # *** uop Variable stuff ***

  @staticmethod
  def variable(name:str, min_val:ConstType, max_val:ConstType, dtype:DType=dtypes.weakint) -> UOp:
    assert not isinstance(min_val, UOp) and not isinstance(max_val, UOp), f"can't create Variable {name} with {min_val}/{max_val}"
    return UOp(Ops.DEFINE_VAR, dtype, arg=(name, min_val, max_val))
  @property
  def expr(self) -> str:
    assert self.op is Ops.DEFINE_VAR, f"op is {self.op}, need DEFINE_VAR"
    return self.arg[0]
  def bind(self, val:int|UOp):
    assert self.op is Ops.DEFINE_VAR, f"op is {self.op}, need DEFINE_VAR"
    uval = self.const_like(val) if isinstance(val, int) else val
    assert self.arg[1] <= uval.vmin and uval.vmax <= self.arg[2], f"bind {val} not in range [{self.arg[1]}, {self.arg[2]}]"
    return UOp(Ops.BIND, self.dtype, (self, uval))
  def unbind(self) -> tuple[Variable, int]:
    assert self.op is Ops.BIND and self.src[0].op is Ops.DEFINE_VAR and self.src[1].op is Ops.CONST, f"can't unbind {self}"
    return self.src[0], self.src[1].arg
  def unbind_all(self) -> tuple[UOp, dict[Variable, int]]:
    ret:dict[Variable, int] = {}
    return graph_rewrite(self, pm_unbind, ctx=ret), ret
  @property
  def val(self) -> int: return self.unbind()[1]
  def variables(self) -> list[Variable]:
    return sorted({x for x in self.backward_slice_with_self if x.op is Ops.DEFINE_VAR}, key=lambda v: v.arg)

  # *** uop symbolic stuff ***

  def is_increasing(self:UOp) -> bool:
    # is f a monotonically increasing function regards its input
    if self.op in GroupOp.Irreducible: return True
    if self.op is Ops.ADD: return self.src[0].is_increasing() and self.src[1].is_increasing()
    if self.op in (Ops.MUL, Ops.CDIV, Ops.FLOORDIV) and self.src[1].op is Ops.CONST and self.src[1].arg >= 0: return self.src[0].is_increasing()
    return False  # False if not sure
  def const_factor(self) -> int:
    """largest known int that divides self"""
    # TODO: for negatives it's not the largest
    if self.op is Ops.CONST: return self.arg
    if self.op is Ops.STACK: return math.gcd(*[x.const_factor() for x in self.src])
    if self.op is Ops.ADD: return math.gcd(self.src[0].const_factor(), self.src[1].const_factor())
    if self.op is Ops.MUL: return self.src[0].arg if self.src[0].op is Ops.CONST else self.src[1].arg if self.src[1].op is Ops.CONST else 1
    return 1
  def divides(self, v:int) -> UOp|None:
    if v==1: return self
    if self.op is Ops.CONST: return self.const_like(self.arg//v) if self.arg%v == 0 else None
    if self.op is Ops.STACK:
      srcs = tuple(s.divides(v) for s in self.src)
      return None if any(s is None for s in srcs) else UOp(Ops.STACK, self.dtype, cast(tuple[UOp, ...], srcs))
    if self.op is Ops.ADD: return d0+d1 if (d0:=self.src[0].divides(v)) is not None and (d1:=self.src[1].divides(v)) is not None else None
    if self.op is Ops.MUL:
      if (d0:=self.src[0].divides(v)) is not None: return d0 * self.src[1]
      if (d1:=self.src[1].divides(v)) is not None: return self.src[0] * d1
    return None # generic None if we aren't sure
  def pop_const(self, op=Ops.ADD) -> tuple[UOp, PyConst]:  # NOTE: assume Invalid ALU is resolved
    return (self.src[0], self.src[1].arg) if self.op is op and self.src[1].op is Ops.CONST else (self, identity_element(op, self.dtype))
  @staticmethod
  def gcd(*uops: UOp) -> UOp:
    terms, factors = zip(*[(u.divides(f:=u.const_factor()),f) for u in uops])
    count = functools.reduce(operator.and_, [collections.Counter(term.split_uop(Ops.MUL)) for term in terms])
    return math.prod([*count.elements(), terms[0].const_like(math.gcd(*factors))])  # put the const at the top
  def divide_exact(self, v:UOp) -> UOp|None:
    if self is v: return self.const_like(1)
    if v.op is Ops.CONST: return self.divides(v.arg)
    if self.op is Ops.ADD: return None if (s0:=self.src[0].divide_exact(v)) is None or (s1:=self.src[1].divide_exact(v)) is None else s0+s1
    if self.op is Ops.MUL:
      (fac, const), (div_fac, div_const) = self.pop_const(Ops.MUL), v.pop_const(Ops.MUL)
      new_count = collections.Counter(fac.split_uop(Ops.MUL))
      new_count.subtract(div_fac.split_uop(Ops.MUL))
      if const%div_const==0 and all(v>=0 for v in new_count.values()): return math.prod(new_count.elements(), start=self.const_like(const//div_const))
    return None # generic None if we aren't sure
  @property
  def vmin(self) -> PyConst: return self._min_max[0]
  @property
  def vmax(self) -> PyConst: return self._min_max[1]
  @functools.cached_property
  def _min_max(self) -> tuple[PyConst, PyConst]:
    if self.op in GroupOp.Binary and not dtypes.is_float(self.dtype):
      (s0_vmin, s0_vmax), (s1_vmin, s1_vmax) = self.src[0]._min_max, self.src[1]._min_max
      if self.op is Ops.ADD: return s0_vmin+s1_vmin, s0_vmax+s1_vmax
      if self.op is Ops.SUB: return s0_vmin-s1_vmax, s0_vmax-s1_vmin
      if self.op is Ops.AND and dtypes.is_int(self.dtype) and s1_vmin == s1_vmax >= 0:
        return 0, s1_vmax if s0_vmin < 0 else min(s0_vmax, s1_vmax)
      if self.op is Ops.MUL: return min(vals:=(s0_vmin*s1_vmin, s0_vmin*s1_vmax, s0_vmax*s1_vmin, s0_vmax*s1_vmax)), max(vals)
      # SHL/SHR on consts only
      if self.op is Ops.SHL and s1_vmin == s1_vmax and all_int(t:=(s0_vmin, s0_vmax, s1_vmin)): return t[0] << t[2], t[1] << t[2]
      if self.op is Ops.SHR and s1_vmin == s1_vmax and all_int(t:=(s0_vmin, s0_vmax, s1_vmin)): return t[0] >> t[2], t[1] >> t[2]
      if self.op is Ops.CMOD:
        if (c:=s1_vmin) == s1_vmax > 0:
          return (0 if s0_vmin > 0 else s0_vmin if 0 >= s0_vmin > -c else -(s1_vmax-1), 0 if s0_vmax < 0 else s0_vmax if 0 <= s0_vmax < c else c-1)
        if s1_vmin > 0: return (0, s1_vmax-1) if s0_vmin >= 0 else (-(s1_vmax-1), 0) if s0_vmax <= 0 else (-(s1_vmax-1), s1_vmax-1)
        if s1_vmax < 0: return (0, -s1_vmin-1) if s0_vmin >= 0 else (-(-s1_vmin-1), 0) if s0_vmax <= 0 else (-(-s1_vmin-1), -s1_vmin-1)
      if self.op is Ops.CDIV:
        assert isinstance(s0_vmin, int) and isinstance(s0_vmax, int) and isinstance(s1_vmin, int) and isinstance(s1_vmax, int)
        if s1_vmin*s1_vmax>0:
          return min(vals:=(cdiv(s0_vmin, s1_vmin), cdiv(s0_vmin, s1_vmax), cdiv(s0_vmax, s1_vmin), cdiv(s0_vmax, s1_vmax))), max(vals)
      if self.op is Ops.FLOORDIV:
        assert isinstance(s0_vmin, int) and isinstance(s0_vmax, int) and isinstance(s1_vmin, int) and isinstance(s1_vmax, int)
        if s0_vmin > s0_vmax: return 0, 0  # numerator range is empty (e.g. RANGE with end=0)
        if s1_vmin*s1_vmax>0: return min(vals:=(s0_vmin//s1_vmin, s0_vmin//s1_vmax, s0_vmax//s1_vmin, s0_vmax//s1_vmax)), max(vals)
      if self.op is Ops.FLOORMOD:
        assert isinstance(s0_vmin, int) and isinstance(s0_vmax, int) and isinstance(s1_vmin, int) and isinstance(s1_vmax, int)
        if s0_vmin > s0_vmax: return 0, 0  # numerator range is empty (e.g. RANGE with end=0)
        if (c:=s1_vmin) == s1_vmax > 0: return (s0_vmin%c, s0_vmax%c) if s0_vmin//c == s0_vmax//c else (0, c-1)
        if (c:=s1_vmin) == s1_vmax < 0: return (s0_vmin%c, s0_vmax%c) if s0_vmin//c == s0_vmax//c else (c+1, 0)
        if s1_vmin > 0: return (0, s1_vmax-1)
        if s1_vmax < 0: return (s1_vmin+1, 0)
      if self.op is Ops.XOR and s1_vmin == s1_vmax == -1 and isinstance(s0_vmin, int) and isinstance(s0_vmax, int):
        return ~int(s0_vmax), ~int(s0_vmin)
      if self.op is Ops.MAX: return max(s0_vmin, s1_vmin), max(s0_vmax, s1_vmax)
      if self.op is Ops.CMPLT: return (s0_vmax<s1_vmin, s0_vmin<s1_vmax)
      if self.op is Ops.CMPNE: return ((s0_vmax < s1_vmin) or (s1_vmax < s0_vmin), not (s0_vmin == s0_vmax == s1_vmin == s1_vmax))
      if self.op is Ops.OR and self.dtype == dtypes.bool: return s0_vmin or s1_vmin, s0_vmax or s1_vmax
      if self.op is Ops.AND and self.dtype == dtypes.bool: return s0_vmin and s1_vmin, s0_vmax and s1_vmax
    # float has NAN issue and we use explicit NAN in transcendental
    if self.op is Ops.WHERE and dtypes.is_int(self.dtype): return min(self.src[1].vmin, self.src[2].vmin), max(self.src[1].vmax, self.src[2].vmax)
    # NOTE: returned UOp is assumed to be CONST
    if self.op is Ops.PARAM and self.arg.vmin_vmax is not None: return self.arg.vmin_vmax
    if self.op is Ops.DEFINE_VAR and self.arg: return self.arg[1], self.arg[2]
    if self.op in (Ops.RANGE, Ops.SPECIAL): return 0, (self.src[0]-1).vmax
    if self.op is Ops.BIND: return self.src[0]._min_max # ignore the bound value
    if self.op in {Ops.UNROLL, Ops.STACK}: return min(x.vmin for x in self.src), max(x.vmax for x in self.src)
    if self.op is Ops.CONST and self.arg is not Invalid: return self.arg, self.arg
    if self.op is Ops.GEP: return self.src[0]._min_max
    # TODO: CAST to bool/unsigned is not monotone, still some case can be simplified
    if self.op is Ops.CAST and self.dtype in dtypes.floats+dtypes.sints+(dtypes.weakint,):
      return max(self.dtype.min, self.src[0].vmin), min(self.src[0].vmax, self.dtype.max)
    return self.dtype.min, self.dtype.max

  @functools.cached_property
  def _sym_fxn(self):
    from tinygrad.uop.render import _render_with_splits, renderer_infer
    sself = self.simplify()
    varnames = tuple(x.expr for x in sself.toposort() if x.op is Ops.DEFINE_VAR)
    # TODO: sanitize varnames, or don't use naked eval while staying fast
    ret = _render_with_splits(list(sself.toposort()), renderer_infer, {sself})
    lines = [f"  {k}={v}" for k,v in ret.items() if k != "ast"] + [f"  return {ret['ast']}"]
    ns: dict[str, Any] = {"max": max, "cdiv": cdiv, "cmod": cmod, "floordiv": floordiv, "floormod": floormod, "bitcast": bitcast, "dtypes": dtypes}
    exec(f"def _f({','.join(varnames)}):\n"+'\n'.join(lines), ns)  # pylint: disable=exec-used
    return ns["_f"], varnames

  def sym_infer(self, var_vals:dict[str, int]):
    fxn, varnames = self._sym_fxn
    return fxn(**{k:v for k,v in var_vals.items() if k in varnames})

  def render(self, simplify=True, pm:PatternMatcher|None=None) -> str:
    ctx: dict[UOp, str] = {}
    from tinygrad.uop.render import renderer
    pm = renderer if pm is None else pm
    for u in (s:=self.simplify() if simplify else self).toposort():
      ctx[u] = cast(str, pm.rewrite(u, ctx=ctx))
    return ctx[s]

  def pyrender(self):
    from tinygrad.uop.render import pyrender
    return pyrender(self)

  # *** uop high level syntactic sugar ***

  @staticmethod
  def placeholder(shape:tuple[int, ...], dtype:DType, slot:int, addrspace=AddrSpace.GLOBAL):
    lookup = {AddrSpace.GLOBAL: Ops.PARAM, AddrSpace.LOCAL: Ops.DEFINE_LOCAL, AddrSpace.REG: Ops.DEFINE_REG}
    arg = ParamArg(slot, addrspace=addrspace) if addrspace is AddrSpace.GLOBAL else slot
    ret = UOp(lookup[addrspace], dtype.ptr(prod(shape), addrspace), arg=arg)
    if len(shape) > 1: ret = ret.reshape(shape)
    return ret
  def placeholder_like(self, slot:int):
    assert all_int(self.shape), "no placeholder-like on symbolic shape"
    return UOp.placeholder(self.max_shard_shape, self.dtype, slot)

  # set is store+end+after
  def set(self:UOp, val:UOp|ConstType, end:UOp|tuple[UOp, ...]|list[UOp]=()) -> UOp:
    return self.src[0].after(self.store(val).end(*argfix(end)))

  # TODO: this should replace placeholder
  @staticmethod
  def param(slot:int, dtype:DType, shape:tuple[sint, ...]|None=None, device=None, vmin_vmax:tuple[PyConst, PyConst]|None=None, name=None,
            addrspace=AddrSpace.GLOBAL, axis:int|None=None):
    if shape is not None and axis is not None and isinstance(device, tuple):
      shape = tuple(s*len(device) if i == axis else s for i,s in enumerate(shape))
    src: tuple[UOp, ...] = (UOp(Ops.NOOP) if shape is None else shape_to_shape_arg(shape),)
    return UOp(Ops.PARAM, dtype, src, arg=ParamArg(slot, vmin_vmax, name, addrspace, axis, device))
  def param_like(self, slot:int):
    addrspace = self.addrspace if isinstance(self.dtype, (PtrDType, ImageDType)) else AddrSpace.GLOBAL
    if self.op is Ops.BIND:
      return UOp.param(slot, self.dtype, self._shape, self.device, cast(tuple[int, int], self._min_max), self.src[0].arg[0], addrspace)
    return UOp.param(slot, self.dtype, self.shard_shape if self.axis is not None else self._shape, self.device, addrspace=addrspace, axis=self.axis)

  # opaque bodies stay as Ops.CALL; value-producing bodies become Ops.FUNCTION (wrapped in TUPLE)
  _OPAQUE_CALL_BODIES = {Ops.SINK, Ops.PROGRAM, Ops.LINEAR, Ops.COPY, Ops.SLICE, Ops.CUSTOM_FUNCTION}
  def call(self, *srcs:UOp, grad_fxn:Callable|None=None, metadata:tuple[Metadata, ...]=(),
           name:str|None=None, precompile:bool=False, precompile_backward:bool=False) -> UOp:
    assert len(self.ranges) == 0, f"ranges {self.ranges} are leaking out of the call in {self.pyrender()}"
    if self.op in UOp._OPAQUE_CALL_BODIES:
      return UOp(Ops.CALL, dtypes.void, (self,)+srcs, CallInfo(grad_fxn, metadata, name, precompile, precompile_backward))
    # value-producing bodies are always wrapped in TUPLE so FUNCTION dtype is always void
    body = self if self.op is Ops.TUPLE else UOp.maketuple(self)
    return UOp(Ops.FUNCTION, dtypes.void, (body,)+srcs, CallInfo(grad_fxn, metadata, name, precompile, precompile_backward))
  def custom_kernel(*srcs:UOp, fxn:Callable, grad_fxn:Callable|None=None) -> list[UOp]:
    # MEMORY_SEMANTIC is transparent to physical layout. Preserve an already
    # concrete buffer/view argument so ownership metadata does not force a
    # redundant materialization before an opaque kernel.
    contig_srcs = tuple(x if x.op is Ops.AFTER or (x.op is Ops.MEMORY_SEMANTIC and x.src[0].has_buffer_identity())
                        else x.contiguous() for x in srcs)
    placeholders = [UOp.placeholder_like(s, slot=i) for i,s in enumerate(contig_srcs)]
    kernel = fxn(*placeholders).call(*contig_srcs, grad_fxn=grad_fxn)
    return [s.after(kernel) for s in contig_srcs]

class RegisterResidentAccumulator(NamedTuple): op: Ops

class AccumulatorSlot(NamedTuple):
  op: Ops          # underlying reduce op (ADD, MAX, MUL)
  dtype: Any = None   # dtype of this accumulator slot
  identity: Any = None    # identity element value
  name: str = ""   # debug name

class StateRegionSpec(NamedTuple):
  """Backend-neutral logical storage region for a value crossing phases."""
  name: str
  dtype: DType
  lanes: int = 1

  def validate(self):
    if not isinstance(self.name, str) or not self.name:
      raise ValueError("state region requires a non-empty name")
    if not isinstance(self.dtype, DType) or self.dtype.vcount != 1:
      raise TypeError("state region dtype must be scalar")
    if not isinstance(self.lanes, int) or self.lanes < 1:
      raise ValueError("state region lanes must be positive")
    return self

  @property
  def value_dtype(self) -> DType: return self.dtype.vec(self.lanes)

class PhaseBoundarySpec(NamedTuple):
  """One directed compiler phase boundary; names intentionally remain generic."""
  publish_phase: str
  reload_phase: str
  ordinal: int = 0

  def validate(self):
    if not all(isinstance(x, str) and x for x in (self.publish_phase, self.reload_phase)):
      raise ValueError("phase boundary names must be non-empty strings")
    if self.publish_phase == self.reload_phase or not isinstance(self.ordinal, int) or self.ordinal < 0:
      raise ValueError("phase boundary must be directed with a non-negative ordinal")
    return self

class StateHandle(NamedTuple):
  """Typed identity for generic state publication and reload descriptors."""
  region: StateRegionSpec
  boundary: PhaseBoundarySpec
  generation: int = 0
  storage: UOp|None = None
  lane: UOp|None = None
  lane_stride: int = 0
  element_offset: int = 0

  def validate(self):
    self.region.validate(); self.boundary.validate()
    if not isinstance(self.generation, int) or self.generation < 0:
      raise ValueError("state handle generation must be non-negative")
    if self.storage is None:
      if self.lane is not None or self.lane_stride != 0 or self.element_offset != 0:
        raise ValueError("storage-free state handles cannot carry lane addressing")
      return self
    if self.storage.op is not Ops.DEFINE_LOCAL or not isinstance(self.storage.dtype, PtrDType) or \
       self.storage.dtype.addrspace is not AddrSpace.LOCAL or self.storage.dtype.base != self.region.dtype:
      raise TypeError("state storage must be a local pointer with the region scalar dtype")
    if self.lane is None or not dtypes.is_int(self.lane.dtype) or self.lane.dtype.vcount != 1:
      raise TypeError("state storage requires a scalar integer lane UOp")
    if not isinstance(self.lane_stride, int) or self.lane_stride < self.region.lanes:
      raise ValueError("state lane stride must cover every region lane")
    if not isinstance(self.element_offset, int) or not 0 <= self.element_offset <= self.lane_stride-self.region.lanes:
      raise ValueError("state element offset must fit within one lane stride")
    return self

  @property
  def dtype(self) -> DType: return self.region.value_dtype

  def publish(self, value: UOp) -> UOp:
    self.validate()
    if value.dtype != self.dtype: raise TypeError("state publish dtype does not match its handle")
    src=(value,) if self.storage is None else (value,self.storage,self.lane)
    return UOp(Ops.CUSTOMI, value.dtype, src, ("state_publish_v1", self))

  def reload(self, published: UOp, wait: UOp|None=None) -> UOp:
    self.validate()
    if published.op is not Ops.CUSTOMI or published.dtype != self.dtype or published.arg != ("state_publish_v1", self):
      raise ValueError("state reload requires publication from the same handle")
    if wait is not None and (wait.op is not Ops.WAIT or published not in wait.src):
      raise ValueError("state reload wait must depend on its publication")
    src=(published,) if self.storage is None else (published,self.storage,self.lane)
    if self.storage is not None and self.region.lanes > 1:
      order=wait if wait is not None else published
      base=self.lane*UOp.const(self.lane.dtype, self.lane_stride)+UOp.const(self.lane.dtype, self.element_offset)
      loads=tuple(self.storage.after(order).index(base+UOp.const(self.lane.dtype, i)).load() for i in range(self.region.lanes))
      lanes=UOp(Ops.STACK, self.dtype, loads, tag=("state_reload_lanes_v1", self))
      return UOp(Ops.CUSTOMI, self.dtype, (lanes,), ("state_reload_v1", self))
    return UOp(Ops.CUSTOMI, self.dtype, src if wait is None else (*src,wait), ("state_reload_v1", self))

  def loop_read(self, element:int, after: UOp|None=None) -> UOp:
    """Test-only generic phase-loop lane view backed by this handle's storage."""
    self.validate()
    if self.storage is None or not isinstance(element,int) or not 0 <= element < self.region.lanes:
      raise ValueError("state loop read requires one in-range storage-backed element")
    src=(self.storage,self.lane) if after is None else (self.storage,self.lane,after)
    return UOp(Ops.CUSTOMI, self.region.dtype, src, ("state_loop_read_v1",self,element))

  def loop_write(self, value: UOp, element:int, after: UOp|None=None) -> UOp:
    """Test-only generic phase-loop store with an optional typed ordering edge."""
    self.validate()
    if self.storage is None or value.dtype != self.region.dtype or not isinstance(element,int) or not 0 <= element < self.region.lanes:
      raise ValueError("state loop write requires one in-range storage-backed scalar")
    if after is not None and after.dtype == dtypes.void: raise TypeError("state loop write ordering source must carry a value")
    src=(value,self.storage,self.lane) if after is None else (value,self.storage,self.lane,after)
    return UOp(Ops.CUSTOMI, dtypes.void, src, ("state_loop_write_v1",self,element))

class CompositeReduce(NamedTuple):
  slots: tuple
  combine_fn: Any = None  # UOp sub-graph encoding combine (None = independent slots)
  input_specs: tuple = () # one source-role/axis-map owner per auxiliary REDUCE source
  tile_carrier: Any = None # optional backend-neutral tile/state ABI contract
  slot_shapes: tuple[tuple|None, ...] = () # logical output shape per REDUCE_SLOT
  tile_fragments: tuple = () # optional scheduler-owned score/value/acc TILE_GATHER nodes
  lane_shapes: tuple[tuple|None, ...] = () # physical register lanes per state slot; () is scalar
  # Concrete RANGE ids which own the semantic reduction.  Rangeify fills this
  # while logical axis positions are still authoritative; late lowering must
  # not infer KV ownership from extent or mutable AxisType classification.
  reduce_range_axes: tuple[int, ...] = ()
  # Optional exact launch ownership for a backend-native attention loop. This
  # remains semantic metadata until postrange admits every live PARAM owner.
  attention_grid: Any = None
  attention_causal: bool = False
  attention_context: Any = None

class SharedAttentionCandidateContext(NamedTuple):
  profile: str
  strategy: str
  q_tokens: int
  kv_tokens: int
  start_pos: int
  hq: int
  hkv: int
  hd: int
  causal: bool
  acc_blocks: int = 8

  @property
  def schema_version(self) -> int: return 1

  @property
  def canonical_identity(self) -> str:
    payload = "shared_attention_candidate_v1|" + "|".join(map(str, self))
    return hashlib.sha256(payload.encode()).hexdigest()

  def validate(self):
    if not isinstance(self.profile, str) or not self.profile or self.strategy not in ("FULL_RESIDENT_OVERLAY", "BOUNDED_PACKED_TILES"):
      raise ValueError("invalid shared attention profile/strategy")
    if not all(isinstance(x, int) and x > 0 for x in (self.q_tokens, self.kv_tokens, self.hq, self.hkv, self.hd)) or self.start_pos < 0:
      raise ValueError("invalid shared attention geometry")
    if self.hd != 128 or self.hq % self.hkv: raise ValueError("shared attention requires Hd128 integral GQA")
    if (self.output_block_base,self.acc_blocks) not in {(0,8),(0,4),(4,4)}: raise ValueError("invalid shared attention accumulator slice")
    return self

class NativeAttentionRequest(NamedTuple):
  native_abi: str
  candidate_context: SharedAttentionCandidateContext
  grid: AMDAttentionGridSpec
  input_dtype: DType
  combine_fn: str

  def validate(self):
    self.candidate_context.validate(); self.grid.validate()
    ctx=self.candidate_context
    if self.native_abi != "amd_gfx1100_attention_grid_hd128_v1" or self.grid.native_abi != self.native_abi:
      raise ValueError("native attention request ABI mismatch")
    if self.input_dtype != dtypes.half or self.combine_fn != "online_softmax_state":
      raise ValueError("native attention request dtype/combine mismatch")
    if (ctx.q_tokens,ctx.kv_tokens,ctx.hq,ctx.hkv,ctx.hd) != \
       (self.grid.q_tokens,self.grid.kv_tokens,self.grid.q_heads,self.grid.kv_heads,self.grid.head_dim) or not ctx.causal:
      raise ValueError("native attention request context/grid mismatch")
    return self

class AttentionWMMARole(NamedTuple):
  contraction: str
  tile: int

  def validate(self):
    if self.contraction not in ("QK", "PV") or not isinstance(self.tile, int) or not 0 <= self.tile < 8:
      raise ValueError("invalid attention WMMA role")
    return self

attention_wmma_roles:weakref.WeakKeyDictionary[UOp, AttentionWMMARole] = weakref.WeakKeyDictionary()

def set_attention_wmma_role(u:UOp, role:AttentionWMMARole) -> None:
  role.validate()
  if (old:=attention_wmma_roles.get(u)) is not None and old != role: raise RuntimeError("attention WMMA role tamper detected")
  attention_wmma_roles[u] = role

def get_attention_wmma_role(u:UOp) -> AttentionWMMARole|None: return attention_wmma_roles.get(u)

class WMMARoleLedger(NamedTuple):
  sites: tuple[tuple[int, AttentionWMMARole], ...] = ()
  def validate(self):
    if any(not isinstance(i, int) or i < 0 or not isinstance(r, AttentionWMMARole) for i,r in self.sites):
      raise TypeError("WMMA role ledger requires typed non-negative sites")
    if tuple(i for i,_ in self.sites) != tuple(sorted({i for i,_ in self.sites})):
      raise ValueError("WMMA role ledger sites must be unique and ordered")
    for _,role in self.sites: role.validate()
    return self

class FinalLinearMetadata(NamedTuple):
  regalloc_proof: Any = None
  wmma_roles: WMMARoleLedger = WMMARoleLedger()

class DeferredReduceSlot(NamedTuple):
  slot: int
  # Inner-to-outer view descriptors: (op, arg, number of non-owner sources).
  views: tuple[tuple[Ops, Any, int], ...] = ()
  normalize_by: int|None = None

def _normalize_composite_shape(shape):
  """Normalize compiler shape metadata before it reaches generic shape logic."""
  if shape is None: return None
  if isinstance(shape, UOp): shape = shape.shape
  if shape is None: return None
  if not isinstance(shape, (tuple, list)):
    raise TypeError(f"composite slot shape must be a tuple/list/UOp, got {type(shape).__name__}")
  # Slot metadata is occasionally assembled from symbolic compiler UOps.  Do
  # not let those objects reach generic tuple/shape code, where ``len(UOp)``
  # raises a misleading TypeError.  Normalize only this documented metadata
  # boundary; ordinary graph rewrite and tensor shapes remain untouched.
  return tuple(_normalize_composite_shape(x) if isinstance(x, (UOp, tuple, list)) else x for x in shape)

class CompositeTileCarrier(NamedTuple):
  """Logical tile contract shared by composite attention and WMMA lowering.

  This is metadata only: it does not prescribe a backend fragment layout.  The
  scheduler uses it to prove that score, value, and accumulator lanes share
  the same tile ownership before attempting a hardware lowering.
  """
  score_shape: tuple[int, int, int]       # (M, N, K) score tile
  value_shape: tuple[int, int, int]       # (N, K, Hd) value tile
  output_shape: tuple[int, int, int]      # (M, N, Hd) output tile
  state_slots: tuple[str, ...] = ("m", "l", "acc")
  score_role: str = "score"
  value_role: str = "logical"
  provenance: tuple[str, ...] = ()
  # Fragment ABI metadata.  These are logical shapes, not packed dtypes: the
  # Hd lane remains explicit until a backend proves its register layout.
  score_fragment: tuple[int, int] | None = None
  value_fragment: tuple[int, int] | None = None
  output_fragment: tuple[int, int] | None = None
  lane_axis: int = 2
  lane_group: int = 1
  typed_fragment_abi: str | None = None

  def validate(self):
    if len(self.score_shape) != 3 or len(self.value_shape) != 3 or len(self.output_shape) != 3:
      raise ValueError("composite tile shapes must be rank-3")
    m, n, k = self.score_shape
    vn, vk, hd = self.value_shape
    om, on, ohd = self.output_shape
    if (n, k) != (vn, vk) or (m, hd) != (om, ohd) or on != n:
      raise ValueError("composite tile score/value/output dimensions do not align")
    if self.state_slots != ("m", "l", "acc"):
      raise ValueError("online composite tile requires m/l/acc state slots")
    sf = self.score_fragment or (m, n)
    vf = self.value_fragment or (n, hd)
    of = self.output_fragment or (m, hd)
    if sf != (m, n) or vf != (n, hd) or of != (m, hd):
      raise ValueError("composite fragment roles do not match tile geometry")
    if self.lane_axis != 2 or not isinstance(self.lane_group, int) or self.lane_group < 1:
      raise ValueError("composite lane ABI must use a positive Hd lane group")
    if hd % self.lane_group:
      raise ValueError("lane group must divide output Hd")
    if self.typed_fragment_abi is not None:
      if self.typed_fragment_abi != "online_softmax_qk_pv_v1":
        raise ValueError("unknown composite typed fragment ABI")
      if (m, n, k, hd) != (16, 16, 16, 16) or self.lane_group != 16:
        raise ValueError("online softmax fragment ABI requires exact 16x16x16 ownership")
      if (sf, vf, of) != ((16, 16), (16, 16), (16, 16)):
        raise ValueError("online softmax fragment ABI requires QK/PV/output fragments")
    return self

  def fragment_abi(self) -> dict:
    """Return explicit logical fragment roles for backend lowering."""
    self.validate()
    m, n, _k = self.score_shape
    _vn, _vk, hd = self.value_shape
    return {"qk_a": (m, _k), "qk_b": (_k, n), "score": (m, n),
            "pv_a": (m, n), "pv_b": (n, hd), "acc": (m, hd),
            "state": self.state_slots, "lane_axis": self.lane_axis,
            "lane_group": self.lane_group, "typed_fragment_abi": self.typed_fragment_abi}

class TileGatherSpec(NamedTuple):
  """Explicit ownership contract for a logical attention tile gather."""
  role: str
  fragment_shape: tuple[int, int]
  source_axes: tuple[int, ...]
  tile_axes: tuple[int, ...]
  base_offsets: tuple[int, ...] = ()
  lane_group: int = 1

  def validate(self):
    if self.role not in ("score", "value", "acc"): raise ValueError("invalid tile gather role")
    if len(self.fragment_shape) != 2 or any(not isinstance(x, int) or x <= 0 for x in self.fragment_shape):
      raise ValueError("tile gather fragment shape must be positive rank-2")
    if len(self.source_axes) != len(self.tile_axes): raise ValueError("tile gather axis ownership mismatch")
    if len(set(self.source_axes)) != len(self.source_axes) or len(set(self.tile_axes)) != len(self.tile_axes):
      raise ValueError("tile gather axes must be unique")
    if any(not isinstance(x, int) or x < 0 for x in self.source_axes + self.tile_axes):
      raise ValueError("tile gather axes must be non-negative integers")
    if self.base_offsets and (len(self.base_offsets) != len(self.source_axes) or
                              any(not isinstance(x, int) or x < 0 for x in self.base_offsets)):
      raise ValueError("tile gather base offsets must match source axes")
    if not isinstance(self.lane_group, int) or self.lane_group < 1:
      raise ValueError("tile gather lane_group must be positive")
    return self

class RowSoftmaxRepackSpec(NamedTuple):
  """Typed scheduler contract for the nonlinear QK-C -> PV-A boundary.

  The operation computes rowwise online-softmax weights, stages them through
  one LDS tile, synchronizes the workgroup, and reloads the tile in PV-A
  ownership. This describes required semantics only; it is not a backend
  lowering and deliberately supports only the first proven fragment ABI.
  """
  typed_fragment_abi: str = "online_softmax_qk_pv_v1"
  fragment_shape: tuple[int, int] = (16, 16)
  state_shape: tuple[int, int] = (16, 1)
  lds_shape: tuple[int, int] = (16, 16)
  row_axis: int = 0
  reduction_axis: int = 1
  requires_barrier: bool = True

  def validate(self):
    if self.typed_fragment_abi != "online_softmax_qk_pv_v1":
      raise ValueError("unknown row-softmax repack fragment ABI")
    if self.fragment_shape != (16, 16) or self.state_shape != (16, 1) or self.lds_shape != (16, 16):
      raise ValueError("online softmax repack ABI requires exact 16x16 score/LDS and 16x1 row state")
    if (self.row_axis, self.reduction_axis) != (0, 1):
      raise ValueError("online softmax repack ABI requires rowwise KV reduction")
    if self.requires_barrier is not True:
      raise ValueError("online softmax LDS repack requires a workgroup barrier")
    return self

class AMDRowSoftmaxRepackSpec(NamedTuple):
  """Exact RDNA3 wave32 realization of ``online_softmax_qk_pv_v1``.

  This descriptor is scheduler-owned. It deliberately records every physical
  fact needed by a future instruction selector rather than allowing the
  renderer to infer a lane permutation or a weaker synchronization contract.
  """
  native_abi: str = "amd_gfx1100_online_softmax_qk_pv_v1"
  target: str = "gfx1100"
  wave_size: int = 32
  qk_c_lanes: int = 8
  pv_a_lanes: int = 16
  row_expr: str = "2*e+(lane>>4)"
  col_expr: str = "lane&15"
  xor_masks: tuple[int, ...] = (1, 2, 4, 8)
  lds_dtype: str = "half"
  lds_elements: int = 256
  lds_address: str = "row*16+col"
  requires_barrier: bool = True
  reload_layout: str = "wmma_f32_16x16x16_f16_pv_a_wave32_v1"
  score_scale: float = 1.0
  mode: str = "legacy_normalized"
  validity_mode: str = "all_v1"
  query_start: int = 0
  kv_start: int = 0
  valid_kv: int = 16
  dynamic_kv_v1: bool = False
  grid: Any|None = None

  def validate(self):
    if (self.native_abi, self.target, self.wave_size) != ("amd_gfx1100_online_softmax_qk_pv_v1", "gfx1100", 32):
      raise ValueError("row-softmax native repack requires exact AMD gfx1100 wave32 v1 ABI")
    if (self.qk_c_lanes, self.pv_a_lanes) != (8, 16):
      raise ValueError("row-softmax native repack requires float.vec(8) QK-C and half.vec(16) PV-A")
    if (self.row_expr, self.col_expr) != ("2*e+(lane>>4)", "lane&15") or self.xor_masks != (1, 2, 4, 8):
      raise ValueError("row-softmax native repack has an unsupported lane reduction layout")
    if (self.lds_dtype, self.lds_elements, self.lds_address) != ("half", 256, "row*16+col"):
      raise ValueError("row-softmax native repack requires the exact 256-half LDS identity map")
    if self.requires_barrier is not True or self.reload_layout != "wmma_f32_16x16x16_f16_pv_a_wave32_v1":
      raise ValueError("row-softmax native repack requires a conservative workgroup-barrier fallback")
    if not isinstance(self.score_scale, float) or not math.isfinite(self.score_scale) or self.score_scale <= 0:
      raise ValueError("row-softmax native repack requires one positive finite score scale")
    if self.validity_mode not in {"all_v1", "tail_v1", "causal_v1"}:
      raise ValueError("row-softmax native repack has an unsupported validity mode")
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (self.query_start, self.kv_start, self.valid_kv)):
      raise ValueError("row-softmax native repack validity metadata must be integral")
    if self.dynamic_kv_v1:
      if self.mode != "loop_state_v1" or self.kv_start != -1 or not 0 <= self.valid_kv <= 4096:
        raise ValueError("dynamic row-softmax validity requires loop_state_v1 and a bounded KV loop")
      if self.grid is not None: self.grid.validate()
    elif self.kv_start < 0 or not 0 <= self.valid_kv <= 32 or self.kv_start not in {0, 16}:
      raise ValueError("row-softmax native repack validity requires a supported 16-wide KV tile")
    if self.validity_mode == "all_v1" and (self.query_start, self.kv_start, self.valid_kv) != (0, 0, 16):
      raise ValueError("unmasked row-softmax repack must retain the canonical validity identity")
    if self.mode not in {"legacy_normalized", "initial_state_v1", "stateful_unnormalized_v1", "loop_state_v1"}:
      raise ValueError("row-softmax native repack has unknown normalization mode")
    return self

class AMDPVCLaneSpec(NamedTuple):
  native_abi: str = "amd_gfx1100_pv_c_lane_v1"
  lane_count: int = 8
  row_expr: str = "2*e+(lane>>4)"
  col_expr: str = "lane&15"
  owner_lane_expr: str = "16*(row&1)"
  owner_e_expr: str = "row>>1"
  element: int = 0

  def validate(self):
    if self.native_abi != "amd_gfx1100_pv_c_lane_v1" or self.lane_count != 8:
      raise ValueError("PV-C lane projection requires exact gfx1100 float.vec(8) ABI")
    if (self.row_expr, self.col_expr, self.owner_lane_expr, self.owner_e_expr) != \
       ("2*e+(lane>>4)", "lane&15", "16*(row&1)", "row>>1"):
      raise ValueError("PV-C lane projection has unsupported row ownership")
    if not isinstance(self.element, int) or not 0 <= self.element < 8:
      raise ValueError("PV-C lane projection element must be in [0,8)")
    return self

class AMDRowSoftmaxSlotSpec(NamedTuple):
  native_abi: str = "amd_gfx1100_online_softmax_qk_pv_v1"
  slot: int = 0
  dtypes: tuple[str, ...] = ("half.vec16", "float.vec8", "float.vec8", "float.vec8")

  def validate(self):
    if self.native_abi != "amd_gfx1100_online_softmax_qk_pv_v1" or self.dtypes != \
       ("half.vec16", "float.vec8", "float.vec8", "float.vec8"):
      raise ValueError("native row-softmax slot requires exact gfx1100 repack ABI")
    if not isinstance(self.slot, int) or not 0 <= self.slot < 4:
      raise ValueError("native row-softmax slot must be in [0,4)")
    return self

class AMDAttentionGridSpec(NamedTuple):
  """Compile-time launch ownership for the fixed 16x16x128 attention wave."""
  native_abi: str = "amd_gfx1100_attention_grid_hd128_v1"
  q_tokens: int = 32
  q_heads: int = 4
  kv_heads: int = 2
  group_ratio: int = 2
  kv_tokens: int = 64
  head_dim: int = 128
  group_expr: str = "q_tile=group%q_tiles;q_head=group//q_tiles;kv_head=q_head//group_ratio"
  wave_size: int = 32
  local_size: int = 32

  @property
  def q_tiles(self): return self.q_tokens//16
  @property
  def grid_size(self): return self.q_heads*self.q_tiles
  @property
  def single_wave_workgroup(self): return self.wave_size == 32 and self.local_size == self.wave_size

  def group_coords(self, gid:int) -> tuple[int,int,int]:
    self.validate()
    if not isinstance(gid,int) or isinstance(gid,bool) or not 0 <= gid < self.grid_size: raise ValueError("group id is outside attention grid")
    q_head,q_tile=divmod(gid,self.q_tiles)
    return q_head,q_tile,q_head//self.group_ratio

  def validate(self):
    if self.native_abi != "amd_gfx1100_attention_grid_hd128_v1" or self.group_expr != "q_tile=group%q_tiles;q_head=group//q_tiles;kv_head=q_head//group_ratio": raise ValueError("AMD attention grid has an unsupported ownership ABI")
    if not all(isinstance(x,int) and not isinstance(x,bool) for x in (self.q_tokens,self.q_heads,self.kv_heads,self.group_ratio,self.kv_tokens,self.head_dim,self.wave_size,self.local_size)): raise ValueError("AMD attention grid dimensions must be integral")
    if self.head_dim != 128 or self.q_tokens <= 0 or self.q_tokens % 16 or self.kv_tokens <= 0 or self.kv_tokens % 16 or self.kv_tokens > 4096 or self.q_heads <= 0 or self.kv_heads <= 0 or self.group_ratio <= 0 or self.q_heads != self.kv_heads*self.group_ratio: raise ValueError("AMD attention grid requires Hd128, 16-wide tokens, and grouped heads")
    if self.wave_size != 32 or self.local_size < self.wave_size or self.local_size % self.wave_size:
      raise ValueError("AMD attention grid requires wave32 and a whole-wave workgroup")
    return self

class AMDMultiWaveAttentionGridSpec(NamedTuple):
  """G2 ownership: one workgroup owns (kv_head,q_tile), each wave owns one Q head."""
  native_abi: str = "amd_gfx1100_attention_multiwave_g2_v1"
  q_tokens: int = 512
  q_heads: int = 4
  kv_heads: int = 2
  kv_tokens: int = 512
  head_dim: int = 128
  wave_size: int = 32
  waves_per_group: int = 2

  @property
  def q_tiles(self): return self.q_tokens//16
  @property
  def local_size(self): return self.wave_size*self.waves_per_group
  @property
  def grid_size(self): return self.kv_heads*self.q_tiles
  @property
  def single_wave_workgroup(self): return False
  @property
  def p_wave_slices(self): return tuple((wave*512, 512) for wave in range(self.waves_per_group))
  def group_coords(self,gid:int,wave_id:int)->tuple[int,int,int]:
    self.validate()
    if not 0 <= gid < self.grid_size or not 0 <= wave_id < self.waves_per_group: raise ValueError("multiwave attention coordinate is out of range")
    kv_head,q_tile=divmod(gid,self.q_tiles); return kv_head*self.waves_per_group+wave_id,q_tile,kv_head
  def validate(self):
    if self.native_abi != "amd_gfx1100_attention_multiwave_g2_v1" or self.wave_size != 32 or self.waves_per_group != 2 or self.head_dim != 128:
      raise ValueError("AMD multiwave attention requires exact G2 wave32 Hd128 ABI")
    if self.q_tokens <= 0 or self.q_tokens%16 or self.kv_tokens <= 0 or self.kv_tokens%16 or self.q_heads != self.kv_heads*2:
      raise ValueError("AMD multiwave attention requires 16-wide tokens and G2 heads")
    return self

class AMDAttentionOutputDrainSpec(NamedTuple):
  """Typed final ownership boundary for the native Hd128 attention ABI."""
  native_abi: str = "amd_gfx1100_attention_output_drain_v1"
  head_dim: int = 128
  blocks: int = 8
  lanes_per_fragment: int = 8
  address_expr: str = "e*256+halfwave*128+j*16+col"
  grid: AMDAttentionGridSpec|None = None
  output_block_base: int = 0

  def validate(self):
    full=("amd_gfx1100_attention_output_drain_v1",128,8,8,"e*256+halfwave*128+j*16+col",0)
    slice_ok=self.native_abi == "amd_gfx1100_attention_output_drain_acc_slice_v2" and self.head_dim == 128 and \
      self.blocks in {1,2,4} and self.lanes_per_fragment == 8 and self.address_expr == "e*256+halfwave*128+j*16+col" and \
      0 <= self.output_block_base <= 8-self.blocks and self.output_block_base % self.blocks == 0
    if (self.native_abi,self.head_dim,self.blocks,self.lanes_per_fragment,self.address_expr,self.output_block_base) != full and not slice_ok:
      raise ValueError("AMD attention output drain requires the exact gfx1100 Hd128 v1 ABI")
    if self.grid is not None: self.grid.validate()
    return self

class AMDAttentionOutputDrainPayload(NamedTuple):
  """Post-regalloc sideband for the versioned accumulator-slice drain ABI."""
  native_abi: str
  output_block_base: int
  blocks: int

  def validate(self):
    if self.native_abi != "amd_gfx1100_attention_output_drain_acc_slice_v2" or self.blocks not in {1,2,4} or \
       not 0 <= self.output_block_base <= 8-self.blocks or self.output_block_base % self.blocks:
      raise ValueError("AMD attention output drain payload requires an aligned v2 accumulator slice")
    return self

class AMDAttentionStatsDrainSpec(NamedTuple):
  native_abi: str = "amd_gfx1100_attention_qk_stats_drain_v1"
  def validate(self):
    if self.native_abi != "amd_gfx1100_attention_qk_stats_drain_v1": raise ValueError("AMD attention stats drain ABI mismatch")
    return self

class AMDLoopStateSpec(NamedTuple):
  """Scheduler-visible recurrence ownership for the unlowered KV tile loop."""
  native_abi: str = "amd_gfx1100_attention_loop_state_v1"
  role: str = "m"
  access: str = "read"
  block: int = 0
  lane: int = 0
  owner: int = 9404

  def validate(self):
    if self.native_abi != "amd_gfx1100_attention_loop_state_v1":
      raise ValueError("AMD attention loop state requires the exact gfx1100 v1 ABI")
    if self.role not in {"m", "l", "acc"} or self.access not in {"init", "read", "write", "final_read"}:
      raise ValueError("AMD attention loop state has an unsupported role or access")
    if not isinstance(self.block, int) or isinstance(self.block, bool) or not 0 <= self.block < (8 if self.role == "acc" else 1):
      raise ValueError("AMD attention loop state has an invalid block")
    if not isinstance(self.lane, int) or isinstance(self.lane, bool) or not 0 <= self.lane < 8:
      raise ValueError("AMD attention loop state has an invalid lane")
    if not isinstance(self.owner, int) or isinstance(self.owner, bool) or self.owner < 0:
      raise ValueError("AMD attention loop state has an invalid owner")
    return self

class AMDPackedFragmentLoopSpec(NamedTuple):
  """Exact Hd128 fragment role plus a runtime KV-tile RANGE source."""
  native_abi: str = "amd_gfx1100_packed_fragment_hd128_loop_v1"
  role: str = "Q"
  head_block: int = 0
  grid: AMDAttentionGridSpec|AMDMultiWaveAttentionGridSpec|None = None
  output_block_base: int = 0

  def validate(self):
    if self.native_abi != "amd_gfx1100_packed_fragment_hd128_loop_v1" or self.role not in {"Q", "K", "V"}:
      raise ValueError("AMD loop fragment has an unsupported ABI or role")
    if not isinstance(self.head_block, int) or isinstance(self.head_block, bool) or not 0 <= self.head_block < 8:
      raise ValueError("AMD loop fragment has an invalid head block")
    if self.grid is not None: self.grid.validate()
    return self

class CompositeInputSpec(NamedTuple):
  role: str
  axis_map: tuple[int|None, ...]|None = None
  # Filled by rangeify only. Public axis_map stays in primary logical-dimension
  # coordinates; this maps it to the owning live RANGE ids for late lowering.
  range_axes: tuple[int|None, ...]|None = None
  # Immutable source-rank provenance. Rangeify resolves axis_map to concrete
  # owner RANGE ids without collapsing the source's distinct axis identity.
  source_shape: tuple|None = None
  source_axis_ranges: tuple[int|None, ...]|None = None
  # Semantic provenance: the primary reduction value is expanded by a
  # repeated logical lane (for example score[..., None].expand(..., Hd)).
  # Late lowering may select one representative lane instead of treating the
  # repetition as an independent reduction element.
  primary_repeated: bool = False
  # Optional logical lane carrier for grouped loads.  This is metadata only:
  # lowering must preserve scalar source indexing until a backend consumes it.
  lane_axis: int|None = None
  lane_group: int = 1

  def validate_lane_abi(self):
    if not isinstance(self.lane_group, int) or self.lane_group < 1:
      raise ValueError("composite input lane_group must be a positive integer")
    if self.lane_axis is not None and self.lane_axis < 0:
      raise ValueError("composite input lane_axis must be non-negative")
    return self

class ScopedReduceSpec(NamedTuple):
  """Source-visible contract for a producer nested inside an outer reduction.

  ``source_axis_maps[i]`` maps axes of source ``i+1`` to the fallback's outer
  logical axes. ``-1`` denotes a value-local/SSA axis.  The spec contains no
  UOps: all dependencies remain normal sources and therefore survive graph
  substitution, scheduling, and DCE.
  """
  reduce_axes: tuple[int, ...]
  source_axis_maps: tuple[tuple[int, ...], ...]
  scope_owner: int
  result_dtypes: tuple[Any, ...]

  def validate_producer(self, producer: 'UOp'):
    """Validate the source which is owned by this explicit scope boundary.

    Ownership must never be inferred from a generic ``SCOPED_VALUE`` carrier:
    that node is only an axis-mapped logical input and has no producer
    lifetime.  A scoped boundary may own an ordinary producer (for example an
    inner MATMUL) or a structured REDUCE, but the producer itself must be a
    real computation node with a shape and cannot be another unresolvable
    value projection.
    """
    if producer.op is Ops.SCOPED_VALUE:
      raise ValueError("SCOPED_REDUCE producer must be an explicit computation, not SCOPED_VALUE")
    if producer.op is Ops.SCOPED_REDUCE:
      raise ValueError("nested SCOPED_REDUCE producer requires an outer ownership resolver")
    if producer.shape is None:
      raise ValueError("SCOPED_REDUCE producer must have a concrete logical shape")
    return producer

class ScopedValueSpec(NamedTuple):
  # source axis -> owner primary-input axis, with None for broadcast axes
  axis_map: tuple[int|None, ...]

class AttentionSpec(NamedTuple):
  """Immutable semantics for an attention operation before scheduler lowering.

  Q, K, V, and an optional mask are UOp sources of Ops.ATTENTION.  This arg
  deliberately contains only scalar/layout facts so graph traversal and
  substitution cannot lose a tensor dependency.
  """
  scale: float
  causal: bool
  mask_present: bool
  qk_dtype: Any
  output_dtype: Any
  reduce_axis: int = -1
  kv_block: int = 0
  # Native GQA is opt-in at the shared prefill boundary.  It records the
  # original Q/K/V head ownership instead of making repeat_interleave part of
  # the selected path; unsupported targets keep the ordinary fallback.
  attention_grid: AMDAttentionGridSpec|None = None
  attention_context: SharedAttentionCandidateContext|None = None


@dataclass(frozen=True)
class KernelInfo:
  name: str = "test"            # name of the kernel
  axis_types: tuple[AxisType, ...] = tuple()
  dont_use_locals: bool = False # don't use local indexing
  applied_opts: tuple = tuple()
  opts_to_apply: tuple|None = None
  estimates: Estimates|None = None
  candidate_context: Any|None = None
  required_native_attention: NativeAttentionRequest|None = None
  # Exact CALL parameter slots whose concrete allocations inherit a scheduler
  # memory owner. Values are immutable vocabulary objects, never UOp identities.
  memory_semantic_slots: tuple[tuple[int, Any], ...] = tuple()
  @property
  def function_name(self): return to_function_name(self.name)

@dataclass(frozen=True)
class ScheduleHints:
  """Narrow, per-expression scheduler policy carried by CONTIGUOUS.

  This avoids ambient Context flags for contractions that require nested reductions to stay in one kernel. The
  scheduler consumes ``pcontig``; postrange consumes ``opts_to_apply`` through the resulting KernelInfo.
  """
  pcontig: int = 0
  opts_to_apply: tuple = tuple()
  name: str|None = None

  def __post_init__(self):
    if not isinstance(self.pcontig, int) or isinstance(self.pcontig, bool) or not 0 <= self.pcontig <= 3:
      raise ValueError("ScheduleHints.pcontig must be an int in [0, 3]")
    if not isinstance(self.opts_to_apply, tuple): raise TypeError("ScheduleHints.opts_to_apply must be a tuple")
    if self.name is not None and (not isinstance(self.name, str) or not self.name):
      raise ValueError("ScheduleHints.name must be None or a non-empty string")

@dataclass(frozen=True)
class ProgramInfo:
  name: str = "test"
  global_size: tuple[int|float, ...] = (1, 1, 1)
  local_size: tuple[int, ...]|None = None
  vars: tuple[UOp, ...] = ()
  globals: tuple[int, ...] = ()
  outs: tuple[int, ...] = ()
  ins: tuple[int, ...] = ()
  aux: tuple = ()
  wmma_roles: WMMARoleLedger = WMMARoleLedger()
  wmma_role_expectation: tuple[AttentionWMMARole, ...] = ()
  candidate_context: SharedAttentionCandidateContext|None = None

  @property
  def function_name(self): return to_function_name(self.name)

  @property
  def runtimevars(self) -> dict[str, int]: return {v.expr: i for i, v in enumerate(self.vars) if v.expr == 'core_id'}

  def launch_dims(self, var_vals:dict[str, int]) -> tuple[tuple[int, ...], tuple[int, ...]|None]:
    global_size = tuple([sym_infer(sz, var_vals) for sz in self.global_size])  # type: ignore[arg-type]
    local_size = tuple([sym_infer(sz, var_vals) for sz in self.local_size]) if self.local_size is not None else None
    return global_size, local_size

  def vals(self, var_vals:dict[str, int]): return tuple(var_vals[k.expr] if k.expr not in self.runtimevars else None for k in self.vars)

  @staticmethod
  def from_sink(sink:UOp, aux:tuple=()) -> ProgramInfo:
    _vars: list[UOp] = []
    _globals: list[int] = []
    outs: list[int] = []
    ins: list[int] = []
    global_size: list[int] = [1, 1, 1]
    local_size: list[int]|None = [1, 1, 1]
    for u in sink.toposort():
      if u.op is Ops.DEFINE_VAR: _vars.append(u)
      if u.op is Ops.PARAM: _globals.append(u.arg.slot)
      if u.op in (Ops.STORE, Ops.LOAD):
        if (idx:=u.src[0]).op in (Ops.INDEX, Ops.SHRINK) or (u.src[0].op is Ops.CAST and (idx:=u.src[0].src[0]).op is Ops.INDEX):
          if (buf:=idx.src[0]).op is Ops.PARAM: (outs if u.op is Ops.STORE else ins).append(buf.arg.slot)
      if u.op is Ops.SPECIAL:
        if u.arg[0] == 'i': local_size = None
        special_size = local_size if u.arg[0] == 'l' else global_size
        if special_size is not None: special_size[int(u.arg[-1])] = cast(int, u.src[0].ssimplify())
      if u.op is Ops.DEFINE_VAR and u.arg[0] == 'core_id': global_size[0] = u.arg[2] + 1
    return ProgramInfo(sink.arg.name if isinstance(sink.arg, KernelInfo) else "test", tuple(global_size),
                       tuple(local_size) if local_size is not None else None, tuple(sorted(_vars, key=lambda v: v.arg)),
                       tuple(sorted(dedup(_globals))), tuple(sorted(dedup(outs))), tuple(sorted(dedup(ins))), aux,
                       candidate_context=sink.arg.candidate_context if isinstance(sink.arg, KernelInfo) else None)

@dataclass(frozen=True)
class CallInfo:
  grad_fxn: Callable|None = None
  metadata: tuple[Metadata, ...] = ()
  name: str|None = None
  precompile: bool = False
  precompile_backward: bool = False
  memory_semantic_slots: tuple[tuple[int, Any], ...] = ()
  # grad_fxn can't be pickled, but metadata can
  def __reduce__(self):
    return (CallInfo, (None, self.metadata, self.name, self.precompile, self.precompile_backward, self.memory_semantic_slots))
  def __repr__(self):
    gf = id(self.grad_fxn) if self.grad_fxn else None
    return f"CallInfo({gf}, {self.metadata}, {repr(self.name)}, {self.precompile}, {self.precompile_backward}, {self.memory_semantic_slots})"


DIAGNOSTIC_LAUNCH_AUTHORITY = "tinygrad.research_only.call_global_size.v1"


@dataclass(frozen=True)
class DiagnosticCallInfo(CallInfo):
  """Explicit invocation-only launch reduction; ordinary CallInfo is unchanged."""
  diagnostic_global_size: tuple[int, ...]|None = None
  diagnostic_launch_authority: str|None = None
  def __reduce__(self):
    return (DiagnosticCallInfo, (
      None, self.metadata, self.name, self.precompile, self.precompile_backward,
      self.memory_semantic_slots, self.diagnostic_global_size, self.diagnostic_launch_authority))
  def __repr__(self):
    return (f"DiagnosticCallInfo({super().__repr__()}, {self.diagnostic_global_size}, "
            f"{repr(self.diagnostic_launch_authority)})")

# Allocation ownership is side metadata except while an explicit tensor-graph
# carrier is awaiting scheduler consumption. Keeping the weak binding here
# lets generic compiler code preserve it without importing an application.
_memory_semantic_owners:weakref.WeakKeyDictionary[UOp, MemorySemanticOwner] = weakref.WeakKeyDictionary()

def _semantic_uop(value:Any) -> UOp:
  uop = value if isinstance(value, UOp) else getattr(value, "uop", None)
  if not isinstance(uop, UOp): raise TypeError("memory semantics can only mark a Tensor or UOp result")
  return uop

def bind_memory_semantic_owner(value:Any, owner:MemorySemanticOwner) -> None:
  uop = _semantic_uop(value)
  existing = memory_semantic_owner(uop)
  if existing is not None and existing != owner: raise ValueError(f"allocation already has semantic owner {existing!r}")
  _memory_semantic_owners[uop] = owner

def memory_semantic_owner(value:Any) -> MemorySemanticOwner|None:
  uop, seen = _semantic_uop(value), set()
  while uop not in seen:
    seen.add(uop)
    if (owner := _memory_semantic_owners.get(uop)) is not None: return owner
    if uop.op is Ops.MEMORY_SEMANTIC: return uop.arg if isinstance(uop.arg, MemorySemanticOwner) else None
    if len(uop.src) and uop.op in GroupOp.Movement | {Ops.CAST, Ops.BITCAST, Ops.CONTIGUOUS, Ops.STAGE, Ops.AFTER}: uop = uop.src[0]
    else: break
  return None

def propagate_memory_semantic(source:UOp, target:UOp) -> UOp:
  owner = memory_semantic_owner(source)
  if owner is None: return target
  if (other := memory_semantic_owner(target)) is not None and other != owner:
    raise ValueError(f"conflicting explicit semantic owners: {owner!r} and {other!r}")
  return target if target.op is Ops.MEMORY_SEMANTIC else UOp(Ops.MEMORY_SEMANTIC, target.dtype, (target,), owner)

# ******** ops in python ********

def safe_exp2(x):
  try: return 2 ** x
  except OverflowError: return math.inf

def safe_pow(x, y):
  try: return math.nan if isinstance(p:=pow(x, y), complex) else p
  except ZeroDivisionError: return math.inf
  except ValueError: return math.inf if x > 0 else -math.inf

python_alu: dict[Ops, Callable]  = {
  Ops.LOG2: lambda x: math.log2(x) if x > 0 else -math.inf if x == 0 else math.nan, Ops.EXP2: safe_exp2,
  Ops.SQRT: lambda x: math.sqrt(x) if x >= 0 else math.nan, Ops.RECIPROCAL: lambda x: 1/x if x != 0 else math.copysign(math.inf, x),
  Ops.SIN: lambda x: math.sin(x) if not math.isinf(x) else math.nan, Ops.POW: safe_pow, Ops.TRUNC: math.trunc,
  Ops.NEG: operator.neg, Ops.ADD: operator.add, Ops.SUB: operator.sub, Ops.MUL: operator.mul, Ops.CMPNE: operator.ne, Ops.CMPLT: operator.lt,
  Ops.XOR: operator.xor, Ops.OR: operator.or_, Ops.AND: operator.and_, Ops.SHR: operator.rshift, Ops.SHL: operator.lshift, Ops.MAX: max,
  Ops.CMOD: cmod, Ops.CDIV: cdiv, Ops.FLOORDIV: floordiv, Ops.FLOORMOD: floormod,
  Ops.MULACC: lambda x,y,z: (x*y)+z, Ops.WHERE: lambda x,y,z: y if x else z, Ops.CMPEQ: operator.eq}

def exec_alu(op:Ops, dtype:DType, operands, truncate_output=True):
  if dtype.count > 1:
    return tuple([exec_alu(op, dtype.scalar(), [x[i] if isinstance(x, tuple) else x for x in operands]) for i in range(dtype.count)])
  if dtype==dtypes.weakint and op in GroupOp.Binary and Invalid in operands: return Invalid
  alu = python_alu[op](*operands)
  return truncate.get(dtype, lambda x: x)(alu) if truncate_output else alu

def bitcast(x, in_dtype:DType, out_dtype:DType):
  assert in_dtype.itemsize == out_dtype.itemsize, "bitcast itemsize mismatch"
  in_count, out_count = in_dtype.count, out_dtype.count
  in_vals = (x,) if in_count == 1 else tuple(x)
  assert len(in_vals) == in_count, f"bitcast expected {in_count} values, got {len(in_vals)}"
  packed = struct.pack(f"{in_count}{storage_fmt_for_dtype(in_dtype.scalar())}", *[to_storage_scalar(v, in_dtype.scalar()) for v in in_vals])
  out_vals = struct.unpack(f"{out_count}{storage_fmt_for_dtype(out_dtype.scalar())}", packed)
  ret = tuple(from_storage_scalar(v, out_dtype.scalar()) for v in out_vals)
  return ret[0] if out_count == 1 else ret

# ***** pattern matcher *****

def get_location() -> tuple[str, int]:
  frm = sys._getframe(1)
  # skip over ops.py and anything in mixin
  while frm.f_back is not None and not frm.f_back.f_code.co_filename.startswith("<frozen"):
    fn = frm.f_code.co_filename.replace("\\", "/")
    if not (fn.endswith("/ops.py") or "/mixin/" in fn): break
    frm = frm.f_back
  return frm.f_code.co_filename, frm.f_lineno

class UPat(OpMixin):
  __slots__ = ("op", "match_dtype", "match_tag", "arg", "name", "src", "is_any")
  def __init__(self, op:Ops|tuple[Ops, ...]|set[Ops]|None=None, dtype:DType|tuple[DType, ...]|set[DType]|None=None,
               src:tuple[UPat, ...]|list[UPat]|UPat|None=None, arg:Any=None,
               name:str|None=None, allow_any_len:bool=False, custom_early_reject:set[Ops]|None=None, location=None, is_any:bool=False,
               tag:Any=None):
    assert op is None or isinstance(op, (Ops, tuple, set)), f"op must be Ops or tuple of Ops, not {op!r}"
    self.op: tuple[Ops, ...]|None = (op,) if isinstance(op, Ops) else (tuple(op) if isinstance(op, set) else op)
    self.match_dtype: tuple[DType, ...]|None = (dtype,) if isinstance(dtype, DType) else (tuple(dtype) if isinstance(dtype, set) else dtype)
    self.match_tag: tuple[Any, ...]|None = (tag,) if isinstance(tag, str) else (tuple(tag) if isinstance(tag, set) else tag)
    self.arg, self.name, self._in_src, self.custom_early_reject = arg, name, src, custom_early_reject
    self.src: Any = None
    self.is_any = is_any
    assert self.name != "ctx", "UPat can't be named ctx"
    assert dtype is None or isinstance(dtype, DType) or all(isinstance(x, DType) for x in dtype), f"invalid dtype {dtype}"

    # try all permutations if it's a list
    if isinstance(src, list): self.src = list(itertools.permutations(src)) if not all_same(src) else [tuple(src)]
    # only one if it's a tuple
    elif isinstance(src, tuple): self.src = [src]
    # repeat if it's a UPat
    elif isinstance(src, UPat): self.src = [itertools.repeat(src)]

    self.strict_length = not (allow_any_len or isinstance(src, UPat) or src is None)
    self.required_len: int = 0 if isinstance(src, UPat) or src is None else len(src)
    self.location = location or get_location()

    if custom_early_reject is not None: self.early_reject = custom_early_reject
    else:
      upat_match = [src] if isinstance(src, UPat) else ([] if src is None else self.src[0])
      self.early_reject = {pp.op[0] for pp in upat_match if pp.op is not None and len(pp.op) == 1}

  @property
  def dtype(self) -> DType: return self.match_dtype[0] if self.match_dtype is not None else dtypes.void

  def _check_dtype(self) -> None: pass
  def _ensure_float(self) -> UPat: return self

  def __reduce__(self):
    return UPat, (self.op, self.match_dtype, self._in_src, self.arg, self.name, not self.strict_length, self.custom_early_reject, self.location,
                  self.is_any, self.match_tag)
  def named(self, name:str):
    return UPat(self.op, self.match_dtype, self._in_src, self.arg, name, not self.strict_length, self.custom_early_reject, tag=self.match_tag)

  @staticmethod
  def any(*src): return UPat(src=src, is_any=True)
  def or_casted(self, name:str|None=None): return UPat.any(self if name is None else self.named(name), UPat(Ops.CAST, name=name, src=(self,)))
  def or_after(self, name:str|None=None):
    return UPat.any(self if name is None else self.named(name), UPat(Ops.AFTER, name=name, src=(self,), allow_any_len=True))

  @staticmethod
  @functools.cache
  def var(name:str|None=None, dtype:DType|tuple[DType, ...]|None=None): return UPat(dtype=dtype, name=name)
  @staticmethod
  @functools.cache
  def cvar(name:str|None=None, dtype:DType|tuple[DType, ...]|None=None, arg=None): return UPat(Ops.CONST, dtype, name=name, arg=arg)
  @staticmethod
  def const(dtype:DType|tuple[DType, ...]|None, b:ConstType): return UPat(Ops.CONST, dtype=dtype, arg=b)

  # lil helper
  def f(self, op, **kwargs): return UPat(op, src=(self,), **kwargs)

  # copied from UOp
  def sink(self, *srcs:UPat|None, **kwargs): return UPat(Ops.SINK, dtypes.void, (self,)+tuple([x for x in srcs if x is not None]), **kwargs)
  def index(self, *srcs:UPat|None, **kwargs):
    return UPat(Ops.INDEX, self.match_dtype, (self,)+tuple(x for x in srcs if x is not None), **kwargs)
  def cast(self, dtype=None, **kwargs):
    if dtype is not None and self.match_dtype == (dtype,): return self
    return UPat(Ops.CAST, dtype, (self,), **kwargs)
  def bitcast(self, dtype=None): return UPat(Ops.BITCAST, dtype, (self,))
  def gep(self, i:int|None=None, **kwargs): return UPat(Ops.GEP, None, (self,), (i,) if i is not None else None, **kwargs)
  def load(self, *src:UPat, **kwargs): return UPat(Ops.LOAD, src=(self,)+src, **kwargs)
  def store(self, *src:UPat, **kwargs): return UPat(Ops.STORE, self.match_dtype, (self,)+src, **kwargs)
  def reduce(self, *src:UPat, **kwargs):
    arg = kwargs.pop('arg', None)
    if isinstance(arg, Ops): arg = (arg, ())
    return UPat(Ops.REDUCE, self.match_dtype, src=(self,)+src, arg=arg, **kwargs)
  def broadcast(self, **kwargs): return UPat(Ops.STACK, self.match_dtype, src=self, **kwargs)

  def contiguous(self, *args, **kwargs): return UPat(Ops.CONTIGUOUS, dtype=self.match_dtype, src=(self,)+args, **kwargs)
  def after(self, *src:UPat, **kwargs): return UPat(Ops.AFTER, self.match_dtype, (self,)+src, **kwargs)
  def end(self, *src:UPat, **kwargs): return UPat(Ops.END, self.match_dtype, (self,)+src, **kwargs)

  def const_like(self, b:ConstLike): return UPat.const(self.match_dtype, cast(ConstType, b))
  # UPat patterns are built with `upat + 1`-style operators; don't insert CAST nodes like _broadcasted does
  def _binop(self, op:Ops, x, reverse:bool) -> UPat:
    return self.ufix(x).alu(op, self) if reverse else self.alu(op, self.ufix(x))
  # TODO: need these override due to dtypes.void on broadcast
  def __floordiv__(self, x): return self._binop(Ops.FLOORDIV, x, False)
  def __rfloordiv__(self, x): return self._binop(Ops.FLOORDIV, x, True)
  def mod(self, x, reverse=False): return self._binop(Ops.FLOORMOD, x, reverse)
  def alu(self, op:Ops, *src:UPat):
    asrc = (self,)+src
    return UPat(op, dtypes.bool if op in {Ops.CMPLT, Ops.CMPNE} else asrc[-1].match_dtype, list(asrc) if op in GroupOp.Commutative else asrc)

  def match(self:UPat, uop:UOp, store:dict[str, UOp]) -> list[dict[str, UOp]]:
    if self.is_any:
      matches = [x.match(uop, store.copy()) for x in self.src[0]]
      return flatten([x for x in matches if x is not None])
    if (self.op is not None and uop.op not in self.op) or \
       (self.name is not None and store.setdefault(self.name, uop) is not uop) or \
       (self.match_dtype is not None and uop.dtype not in self.match_dtype and uop.dtype.scalar() not in self.match_dtype) or \
       (self.arg is not None and self.arg != uop.arg) or \
       (self.match_tag is not None and uop.tag not in self.match_tag) or \
       (len(uop.src) < self.required_len) or \
       (self.strict_length and len(uop.src) != self.required_len): return []
    if self.src is None: return [store]
    res: list[dict[str, UOp]] = []
    for vp in self.src:
      stores, new_stores = [store.copy()], []
      for uu, vv in zip(uop.src, vp):
        for s in stores: new_stores.extend(vv.match(uu, s))
        stores, new_stores = new_stores, []
      res.extend(stores)
    return res

def deconstruct_function(fxn:Callable) -> tuple:
  new_globals = {k:v for k,v in fxn.__globals__.items() if k in fxn.__code__.co_names}
  for co in fxn.__code__.co_consts:
    if isinstance(co, types.CodeType): new_globals.update({k:v for k,v in fxn.__globals__.items() if k in co.co_names})
  # NOTE: optional round trip through pickle!
  assert fxn.__closure__ is None, "closures are not supported in pattern matchers"
  ret = fxn.__code__, new_globals, fxn.__name__, fxn.__defaults__
  return pickle.loads(pickle.dumps(ret)) if getenv("TEST_PICKLE") else ret

@functools.cache
def upat_interpret(p:UPat, fxn:Callable) -> Callable:
  real_fxn = types.FunctionType(*deconstruct_function(fxn))
  if 'ctx' in inspect.signature(real_fxn).parameters:
    def universal_match(uop, ctx):
      for match in p.match(uop, {}):
        if (ret:=real_fxn(ctx=ctx, **match)) is not None: return ret  # pylint: disable=not-callable
      return None
  else:
    def universal_match(uop, _):
      for match in p.match(uop, {}):
        if (ret:=real_fxn(**match)) is not None: return ret  # pylint: disable=not-callable
      return None
  return universal_match

def upat_deferred_compile(p:UPat, fxn:Callable, entry:list) -> Callable:
  def lazy_compile(uop, ctx):
    from tinygrad.uop.upat import upat_compile
    entry[1] = upat_compile(p, fxn) or upat_interpret(p, fxn)
    return entry[1](uop, ctx)
  return lazy_compile

class PatternMatcher:
  def __init__(self, patterns:Sequence[tuple[UPat, Callable|tuple]], compiled=bool(getenv("UPAT_COMPILE", 1))):
    # if this comes from a pickle, we reconstruct the lambda functions here
    self.patterns:list[tuple[UPat, Callable]] = [(p,types.FunctionType(*fxn) if isinstance(fxn, tuple) else fxn) for p,fxn in patterns]
    # NOTE: use of DefaultDict here is very dangerous! all keys will live for the lifetime of the PatternMatcher!
    self.pdict: dict[Ops, list[list]] = {}
    # uop is required, arg is optional
    for p,fxn in self.patterns:
      assert p.op is not None
      entry: list = [p, None, p.early_reject]
      entry[1] = upat_deferred_compile(p, fxn, entry) if compiled else upat_interpret(p, fxn)
      for uop in p.op: self.pdict.setdefault(uop, []).append(entry)

  def __reduce__(self): return PatternMatcher, ([(x,deconstruct_function(fxn) if fxn.__name__ == "<lambda>" else fxn) for x,fxn in self.patterns],)

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def __add__(self, more:PatternMatcher) -> PatternMatcher: return PatternMatcher(self.patterns+more.patterns)

  def rewrite(self, uop:UOp, ctx=None):
    if len(pats:=self.pdict.get(uop.op, [])):
      if (ler:=uop.__dict__.get('_src_ops')) is None: uop.__dict__['_src_ops'] = ler = {u.op for u in uop.src}
      for _,match,early_reject in pats:
        if not early_reject.issubset(ler): continue
        if (ret:=match(uop, ctx)) is not None and ret is not uop: return ret
    return None

# *** tracking pattern matcher ***

TRACK_MATCH_STATS = ContextVar("TRACK_MATCH_STATS", 2 if VIZ else 0)
REWRITE_STACK_LIMIT = ContextVar("REWRITE_STACK_LIMIT", 250000)
match_stats:dict[UPat, list[int|float]] = dict()

# TRACK_MATCH_STATS>=2 or VIZ=1 saves all matches
ucount = itertools.count()
uop_fields:dict[int, tuple] = {}

@dataclass(frozen=True)
class TrackedGraphRewrite:
  loc:tuple[str, int]                           # location that called graph_rewrite
  sink:int                                      # the sink input to graph_rewrite
  matches:list[tuple[int, int, tuple, float]]   # before/after UOp, UPat location and time
  name:str                                      # name of the rewrite
  depth:int                                     # depth if it's a subrewrite
  bottom_up:bool

tracked_keys:list[TracingKey] = []
tracked_ctxs:list[list[TrackedGraphRewrite]] = []
_name_cnt:dict[str, itertools.count] = {}

if CAPTURE_PROCESS_REPLAY:
  replay_capture: list[bytes] = []
  import atexit, uuid
  @atexit.register
  def save_to_diskcache():
    uid = uuid.uuid4() # one id per process
    for i,v in enumerate(replay_capture): diskcache_put("process_replay", f"{uid}_{i}", v, prepickled=True)

def add_trace_group(kt:TracingKey) -> None:
  tracked_keys.append(kt)
  tracked_ctxs.append([])

active_group:list[int] = []
def track_rewrites(name:Callable[..., str|TracingKey]|bool=True, replay:bool=False):
  def _decorator(func):
    def __wrapper(*args, **kwargs):
      fn = key = func.__name__
      idx = -1
      if TRACK_MATCH_STATS >= 2:
        add_trace_group(key:=TracingKey(n:=f"{fn} n{next(_name_cnt.setdefault(fn, itertools.count(1)))}", (n,)))
        active_group.append(idx:=len(tracked_keys)-1)
      with cpu_profile(key, "TINY") as e:
        ret = func(*args, **kwargs)
      if TRACK_MATCH_STATS >= 2: active_group.pop()
      if TRACK_MATCH_STATS >= 2 and callable(name):
        name_ret = name(*args, **kwargs, ret=ret)
        assert isinstance(name_ret, (TracingKey, str)), f"name function returned {type(name_ret)}"
        tracked_keys[idx] = k = TracingKey(n:=tracked_keys[idx].display_name.replace(fn, name_ret), (n,)) if isinstance(name_ret, str) else name_ret
        e.name = TracingKey(k.display_name if isinstance(name_ret, str) else f"{fn} for {k.display_name}", k.keys)
      if CAPTURE_PROCESS_REPLAY and replay:
        # find the unittest frame we're capturing in
        frm = sys._getframe(1)
        while (f_back:=frm.f_back) is not None and "unittest" not in f_back.f_code.co_filename: frm = f_back
        loc = f"{frm.f_code.co_filename.split('/')[-1]}:{frm.f_lineno} {frm.f_code.co_name}"
        # capture global context vars and all the args passed in
        inputs = (fn, args, kwargs, ContextVar._cache)
        replay_capture.append(pickle.dumps(inputs+(loc, ret)))
      return ret
    return __wrapper
  return _decorator

active_rewrites:list[TrackedGraphRewrite] = []
def profile_matches(fxn:Callable):
  def wrap_profile_matches(*args, **kwargs):
    if TRACK_MATCH_STATS >= 2:
      name = str(kwargs.get("name", None) or fxn.__name__)
      assert args and isinstance(args[0], UOp), f"invalid match tracing inputs for {name} with {args}"
      loc = ((frm:=sys._getframe(1)).f_code.co_filename, frm.f_lineno)
      depth = len(active_rewrites)
      if not tracked_ctxs: add_trace_group(TracingKey(f"default {fxn.__name__}"))
      dest_group = active_group[-1] if active_group else len(tracked_ctxs)-1
      tracked_ctxs[dest_group].append(ctx:=TrackedGraphRewrite(loc, args[0].trace_num, [], name, depth, kwargs.get("bottom_up", False)))
      active_rewrites.append(ctx)
      with cpu_profile(name, "TINY"):
        ret = fxn(*args, **kwargs)
      active_rewrites.pop()
      return ret
    # without tracking, we just call the function
    return fxn(*args, **kwargs)
  return wrap_profile_matches

class TrackedPatternMatcher(PatternMatcher):
  def rewrite(self, uop:UOp, ctx=None):
    if len(pats:=self.pdict.get(uop.op, [])):
      ret = None
      ler = {u.op for u in uop.src}
      for p,match,early_reject in pats:
        if p not in match_stats: match_stats[p] = [0,0,0.0,0.0]
        st = time.perf_counter()
        if not early_reject.issubset(ler):
          match_stats[p][2] += time.perf_counter()-st
          continue
        match_stats[p][1] += 1
        try: ret = match(uop, ctx)
        except Exception:
          if TRACK_MATCH_STATS >= 2 and active_rewrites:
            active_rewrites[-1].matches.append((uop.trace_num, UOp(Ops.REWRITE_ERROR,src=uop.src,arg=str(sys.exc_info()[1])).trace_num,p.location,0))
          raise
        if ret is not None and ret is not uop:
          match_stats[p][0] += 1
          match_stats[p][3] += (et:=time.perf_counter()-st)
          if TRACK_MATCH_STATS >= 3: print(f"{et*1e6:7.2f} us -- ", printable(p.location))
          if TRACK_MATCH_STATS >= 2 and isinstance(ret, UOp) and active_rewrites:
            active_rewrites[-1].matches.append((uop.trace_num, ret.trace_num, p.location, et))
          return ret
        match_stats[p][2] += time.perf_counter()-st
    return None

@dataclass(frozen=True)
class RewriteTrace: keys:list[TracingKey]; rewrites:list[list[TrackedGraphRewrite]]; uop_fields:dict[int, tuple] # noqa: E702

if TRACK_MATCH_STATS or PROFILE:
  PatternMatcher = TrackedPatternMatcher  # type: ignore
  import atexit
  @atexit.register
  def print_match_stats():
    if TRACK_MATCH_STATS >= 2:
      with open(fn:=temp("rewrites.pkl", append_user=True), "wb") as f:
        print(f"rewrote {len(tracked_ctxs)} graphs and matched {sum(len(r.matches) for x in tracked_ctxs for r in x)} times, saved to {fn}")
        pickle.dump(RewriteTrace(tracked_keys, tracked_ctxs, uop_fields), f)
    TRACK_MATCH_STATS.value = 0
    launch_viz("REWRITE_DATA", temp("rewrites.pkl", append_user=True))
    if getenv("PRINT_MATCH_STATS", TRACK_MATCH_STATS.value and not VIZ):
      ret = [0,0,0.0,0.0]
      for k,v in sorted(list(match_stats.items()), key=lambda x: x[1][2]+x[1][3]):
        loc_str = f"{k.location[0].split('/')[-1]}:{k.location[1]}"
        if v[1] != 0: print(f"{v[0]:6d} / {v[1]:7d} -- {v[3]*1000.:9.2f} / {(v[2]+v[3])*1000.:9.2f} ms -- {loc_str:20s}", printable(k.location))
        ret = [x+y for x,y in zip(ret, v)]
      print(f"{ret[0]:6d} / {ret[1]:7d} -- {ret[3]*1000.:9.2f} / {(ret[2]+ret[3])*1000.:9.2f} ms -- TOTAL")
      print(f"{len(match_stats)} rules, {sum(v[0] > 0 for v in match_stats.values())} matched once")

  def launch_viz(env_str:str, data:str):
    os.environ[f"{env_str}_DATA"] = data
    if not TRACK_MATCH_STATS and not PROFILE:
      os.environ["VIZ"], os.environ["PROFILE"], os.environ["TRACK_MATCH_STATS"] = "0", "0", "0"
      args = ['--rewrites-path', os.getenv("REWRITE_DATA", "")] if os.getenv("REWRITE_DATA", "") else []
      args += ['--profile-path', os.getenv("PROFILE_DATA", "")] if os.getenv("PROFILE_DATA", "") else []
      viz_path = pathlib.Path(__file__).resolve().parent.parent / "viz" / "serve.py"
      if VIZ > 0 and sys.stdout.isatty(): os.execv(sys.executable, [sys.executable, viz_path.as_posix()] + args)
      if VIZ: print("saved viz files, view using: python -m tinygrad.viz.cli")
      VIZ.value = 0

# *** simple graph rewrite engine ***

# A pure Python sentinel, but *typed* as UOp so it fits all the dict annotations
SENTINEL: Final[UOp] = cast(UOp, object())
class BottomUpGate(Exception): pass
class RewriteContext:
  def __init__(self, pm, bpm, ctx=None, enter_calls=False):
    self.pm: PatternMatcher|None = pm
    self.bpm: PatternMatcher|None = bpm
    self.bpm_cache: dict[UOp, UOp|None] = {}
    self.ctx = ctx
    self.replace: dict[UOp, UOp] = {}
    self.enter_calls = enter_calls

  # no cache needed: pm_rewrite is called at most once per UOp due to the replace dict check in unified_rewrite
  def pm_rewrite(self, x:UOp) -> UOp|None: return unwrap(self.pm).rewrite(x, self.ctx)

  def cached_bpm_rewrite(self, x:UOp) -> UOp|None:
    if (ret:=self.bpm_cache.get(x,SENTINEL)) is not SENTINEL: return ret
    ret = self.bpm_cache[x] = unwrap(self.bpm).rewrite(x, self.ctx)
    return ret

  def walk_rewrite(self, root:UOp) -> UOp:
    """MLIR-style Walk Pattern Rewrite Driver: single-pass, no re-traversal into rewritten subtrees."""
    stack: list[tuple[UOp, bool]] = [(root, False)]
    while stack:
      n, processed = stack.pop()
      if n in self.replace: continue
      if not processed:
        # bottom-up: try bpm on original node first, if it rewrites, use result as-is (no traversal into replacement)
        if self.bpm is not None and (rewritten:=self.cached_bpm_rewrite(n)) is not None:
          self.replace[n] = rewritten
          continue
        # no rewrite, process children then come back to rebuild
        stack.append((n, True))
        if not self.enter_calls and n.op in {Ops.CALL, Ops.FUNCTION}: self.replace[n.src[0]] = n.src[0]
        for x in reversed(n.src):
          if x not in self.replace: stack.append((x, False))
      else:
        # rebuild node with rewritten srcs
        new_src = tuple(self.replace.get(x, x) for x in n.src)
        new_n = UOp(n.op, n.dtype, new_src, n.arg, n.tag) if new_src != n.src else n
        # top-down: try pm on rebuilt node, use result as-is (no re-traversal)
        if self.pm is not None and (rewritten:=self.pm_rewrite(new_n)) is not None: new_n = rewritten
        self.replace[n] = new_n
    return self.replace.get(root, root)

  def unified_rewrite(self, root:UOp) -> UOp:
    stack: collections.deque[tuple[UOp, int, UOp]] = collections.deque([(root, 0, root)])
    on_stack = {root}  # all UOps either on the stack or in self.replace, i.e. dont have to be placed again
    waitlist: dict[UOp, list[tuple[UOp, int, UOp]]] = {}  # UOps waiting on a dependency to be in self.replace
    while stack:
      if len(stack) > REWRITE_STACK_LIMIT: raise RuntimeError("infinite loop in graph_rewrite (stack too big)")
      n, stage, new_n = stack.pop()
      if n in self.replace: continue  # skip any nodes we have seen
      if stage == 0:
        # if bottom up, we rewrite this node early. in both cases, we add its srcs to the stack
        if self.bpm is not None:
          # apply rewrite rules until a fixed point is reached. may return `uop` itself if PatternMatcher doesn't match
          test_n: UOp|None = n
          seen = set()
          try:
            while test_n is not None:
              if test_n in seen: raise RuntimeError("infinite loop in fixed_point_rewrite")
              seen.add(test_n)
              new_n, test_n = test_n, self.cached_bpm_rewrite(test_n)
          except BottomUpGate:
            # if the bpm matching raised a gate, we are done with this node and dont continue down the srcs
            self.replace[n] = unwrap(test_n)
            if n in waitlist: stack.extend(waitlist.pop(n))
            continue
        stack.append((n, 1, new_n))
        # NOTE: CALL/FUNCTION are handled as a special case.
        # The function that is called is not included in the graph_rewrite.
        # If you want to graph_rewrite a call, you can
        if not self.enter_calls and new_n.op in {Ops.CALL, Ops.FUNCTION}: self.replace[new_n.src[0]] = new_n.src[0]
        for x in reversed(new_n.src):
          if x in on_stack: continue
          stack.append((x, 0, x))
          on_stack.add(x)
      elif stage == 1:
        tmp = []
        for x in new_n.src:
          if (rx:=self.replace.get(x, SENTINEL)) is SENTINEL:
            # source not ready: register in waitlist instead of spinning
            waitlist.setdefault(x, []).append((n, 1, new_n))
            break
          tmp.append(rx)
        else:
          # in stage 1, once all srcs are rewritten, rebuild (if changed) or run top-down rewrite
          if (new_src:=tuple(tmp)) == new_n.src:
            # if top down, do the rewrite. if no rewrite or bottom up, we are done rewriting this node so we add it to the dict
            if self.pm is None or (new_src_n:=self.pm_rewrite(new_n)) is None:
              self.replace[n] = new_n
              if n in waitlist: stack.extend(waitlist.pop(n))
              continue
          else:
            # if srcs changed from rewrites, construct a new UOp with the new srcs
            new_src_n = UOp(new_n.op, new_n.dtype, new_src, new_n.arg, new_n.tag)
          # trigger a rewrite of new_src_n, then after that rewrite is done, link it back to n
          stack.append((n, 2, new_src_n))
          stack.append((new_src_n, 0, new_src_n))
      else:
        # in stage 2, we link the result of new_n to the result of n
        if (replaced_new_n:=self.replace.get(new_n, SENTINEL)) is SENTINEL:
          # not ready: register in waitlist instead of spinning
          waitlist.setdefault(new_n, []).append((n, 2, new_n))
        else:
          # otherwise we are done
          self.replace[n] = replaced_new_n
          if n in waitlist: stack.extend(waitlist.pop(n))
    return self.replace[root]

@profile_matches
def graph_rewrite(sink:UOp, pm:PatternMatcher, ctx=None, bottom_up=False, name=None, bpm=None, walk=False, enter_calls=False) -> UOp:
  rewrite_ctx = RewriteContext(pm if not bottom_up else None, pm if bottom_up else bpm, ctx, enter_calls)
  return rewrite_ctx.walk_rewrite(sink) if walk else rewrite_ctx.unified_rewrite(sink)

def sint_to_uop(x:sint, dtype=dtypes.weakint) -> UOp: return UOp.const(dtype, x) if isinstance(x, int) else x.cast(dtype)
def to_max_shape(shape:tuple[sint, ...]) -> tuple[int, ...]: return tuple(int(x.vmax) if isinstance(x, UOp) else x for x in shape)

def select_dtype(u): return (dtypes.long if u.overflows(dtypes.int32) else dtypes.int).vec(u.dtype.count)
def _flatten_concatenated_index(x:UOp, buf:UOp) -> UOp|None:
  # Consecutive tensor INDEX operations concatenate additive offsets. Images
  # are the sole two-coordinate pointer form and retain their coordinates.
  if len(x.src) <= 2 or isinstance(buf.dtype, ImageDType): return None
  return x.replace(src=(buf, functools.reduce(operator.add, x.src[1:])))
pm_lower_index_dtype = PatternMatcher([
  # There are no Unary ops at this point in symbolic, those are introduced later
  (UPat(GroupOp.Binary, name="u", src=(UPat.var("x").cast(dtypes.weakint), UPat.var("y").cast(dtypes.weakint))), lambda u,x,y:
    x.cast(dt:=least_upper_dtype(select_dtype(u), x.dtype, y.dtype)).alu(u.op, y.cast(dt)).cast(u.dtype)),
  (UPat(Ops.CONST, dtype=dtypes.weakint, name="u"), lambda u: u.replace(dtype=select_dtype(u)).cast(u.dtype) if u.arg!=Invalid else None),
  (UPat(Ops.WHERE, dtypes.weakint, src=(UPat.var("cond"), UPat.var("x").cast(dtypes.weakint), UPat.var("y").cast(dtypes.weakint))), lambda cond,x,y:
    cond.where(x.cast(dt:=least_upper_dtype(x.dtype, y.dtype)), y.cast(dt)).cast(dtypes.weakint)),
  (UPat(Ops.RANGE, src=(UPat.var("end").cast(dtypes.weakint)), name="r"), lambda r,end: r.replace(dtype=end.dtype, src=(end,)).cast(dtypes.weakint)),
  (UPat(Ops.STACK, src=UPat().cast(dtypes.weakint), name="v"),
    lambda v: v.replace(dtype=(dt:=select_dtype(v)), src=tuple(s.src[0].cast(dt.scalar()) for s in v.src)).cast(dtypes.weakint)),
  # special can only be int32
  (UPat(Ops.SPECIAL, src=(UPat.var("var").cast(dtypes.weakint),), name="u"),
    lambda u,var: u.replace(dtype=dtypes.int, src=(var,)).cast(dtypes.weakint)),
  (UPat(Ops.DEFINE_VAR, dtype=dtypes.weakint, name="u"), lambda u: u.replace(dtype=dtypes.int).cast(dtypes.weakint)),
  (UPat(Ops.BIND, src=(UPat.var("var").cast(dtypes.weakint), UPat.cvar("val").cast(dtypes.weakint))),
    lambda var,val: var.bind(val).cast(dtypes.weakint)),
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat(), UPat()), allow_any_len=True, name="x"), _flatten_concatenated_index),
  # remove hanging casts
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("idx", dtypes.ints).cast()),), lambda buf,idx: buf.index(idx, ptr=True)),
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("gate").where(UPat.var("idx", dtypes.ints).cast(), UPat(Ops.CONST, arg=Invalid)))),
   lambda buf,idx,gate: buf.index(gate.where(idx, idx.const_like(Invalid)), ptr=True)),
  # remove hanging casts for images
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("idx_y", dtypes.ints).cast(), UPat.var("idx_x", dtypes.ints).cast()),),
   lambda buf,idx_x,idx_y: buf.index(idx_y, idx_x, ptr=True)),
  (UPat(Ops.INDEX, src=(UPat.var("buf"),
                        UPat.var("gate").where(UPat.var("idx_y", dtypes.ints).cast(), UPat(Ops.CONST, arg=Invalid)),
                        UPat.var("gate").where(UPat.var("idx_x", dtypes.ints).cast(), UPat(Ops.CONST, arg=Invalid)))),
   lambda buf,idx_x,idx_y,gate: buf.index(gate.where(idx_y, idx_y.const_like(Invalid)),
                                          gate.where(idx_x, idx_x.const_like(Invalid)), ptr=True)),
  # Normalize any remaining direct weak-index casts after concatenated offsets
  # have been flattened to one concrete address expression.
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat()), allow_any_len=True, name="x"),
   lambda x,buf: x.replace(src=(buf, *(s.src[0] if s.op is Ops.CAST and s.dtype == dtypes.weakint and
                                       s.src[0].dtype in dtypes.ints else s for s in x.src[1:])))
   if any(s.op is Ops.CAST and s.dtype == dtypes.weakint and s.src[0].dtype in dtypes.ints for s in x.src[1:]) else None),
  (UPat((Ops.SINK, Ops.NOOP, Ops.END), name="n"),
   lambda n: n.replace(src=tuple(s.src[0] if s.op is Ops.CAST and s.dtype == dtypes.weakint else s for s in n.src))),
])
def _index_to_concrete_int(u:UOp) -> UOp: return graph_rewrite(u.sink(), pm_lower_index_dtype).src[0]

_substitute = PatternMatcher([(UPat(tuple(Ops), name="x"), lambda ctx,x: ctx.get(x,None))])
_pm_resolve_params = PatternMatcher([(UPat(Ops.PARAM, name="p"), lambda ctx,p: ctx[p.arg.slot])])
def _clear_non_composite_tag(x: UOp):
  # Backend scheduling may clear ordinary register/optimization tags, but
  # validated composite provenance is part of the REDUCE_SLOT type contract.
  if x.tag is None: return None
  if isinstance(x.tag, tuple) and len(x.tag) == 2:
    kind, payload = x.tag
    valid = (kind in ("composite_reduce", "composite_slot") and hasattr(payload, "slots")) or \
            (kind == "composite_view" and isinstance(payload, tuple) and len(payload) >= 2 and
             payload[0] in ("composite_reduce", "composite_slot", "composite_view"))
    if valid: return None
  return x.replace(tag=None)
remove_all_tags = PatternMatcher([(UPat(GroupOp.All, name="x"), _clear_non_composite_tag)])

def gate_kernel_sink(x:UOp) -> bool:
  if x.op is Ops.LINEAR: return False
  if x.op is Ops.SINK and isinstance(x.arg, KernelInfo): return False
  return True

def do_unbind(ctx:dict[Variable, int], x:UOp):
  v,i = x.unbind()
  ctx[v] = i
  return v
pm_unbind = PatternMatcher([(UPat(Ops.BIND, name="x"), do_unbind)])

# *** what was symbolic.py ***

sint = int|UOp
Variable = UOp

ConstLike = ConstType|Variable|tuple[ConstType, ...]
