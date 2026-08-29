import functools, itertools, time
from typing import Generic, TypeVar, Callable, cast, overload
from tinygrad.helpers import Context, dedup, getenv, DEBUG
from tinygrad.uop.ops import UOp, Ops, ProgramInfo, memory_semantic_owner, graph_rewrite, PatternMatcher, UPat
from tinygrad.uop import MemorySemanticClass
from tinygrad.tensor import Tensor
from tinygrad.nn.state import get_state_dict

def add_to_ctx(ctx, x:UOp):
  ret = x.param_like(len(ctx[0]))
  ctx[0].append(x)
  return ret

def _ancestor_kinds(x:UOp, *ops:Ops) -> tuple[bool, ...]:
  """Whether each listed op appears in x's backward slice, computed without caching.

  op_in_backward_slice_with_self stores a full toposort dict on every node it touches.
  Precompile bodies grow per composite (the resadd fold admits the previous block output),
  so that cached_property retention is O(body) per matched AFTER/CONTIGUOUS and wedges
  host RSS at flash-decode scale.  This scan is semantically identical and retains nothing.
  """
  found = [False]*len(ops)
  stack, seen = [x], set()
  while stack:
    n = stack.pop()
    for i, op in enumerate(ops):
      if n.op is op: found[i] = True
    if all(found) or n in seen: continue
    seen.add(n)
    stack.extend(n.src)
  return tuple(found)

def _maybe_add_to_ctx(ctx, x:UOp):
  # A PROGRAM output AFTER is its dependency edge, not an already-computed
  # implicit input.  Follow the declared write ABI rather than source/binary
  # spelling: compiler-owned and finalized PROGRAMs have the same contract.
  for after in x.toposort():
    if after.op is not Ops.AFTER or not after.src: continue
    try: output = after.src[0].buf_uop
    except RuntimeError: continue
    for call in after.src[1:]:
      if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM or not isinstance(call.src[0].arg, ProgramInfo): continue
      for slot in call.src[0].arg.outs:
        if slot >= len(call.src)-1: continue
        try:
          if call.src[slot+1].buf_uop is output: return None
        except RuntimeError: pass
  has_param, has_buffer = _ancestor_kinds(x, Ops.PARAM, Ops.BUFFER)
  return add_to_ctx(ctx, x) if not has_param and has_buffer else None

def _computed_program_inputs(uret:UOp, explicit:tuple[UOp, ...]) -> list[UOp]:
  """Inputs which must cross the enclosing FUNCTION boundary as values.

  A PROGRAM has an opaque body, so callification cannot reconstruct a buffer
  for an arbitrary expression left in one of its declared input slots.  Hoist
  only read-only slots from ProgramInfo.ins (never outs/read-write slots), and
  only when the argument is a computed value rather than an existing buffer.
  The invocation argument retains the producer graph and its AFTER edge; the
  normalized body receives the matching PARAM.  This is ordinary closure
  conversion, keyed entirely by the executable ABI.
  """
  # Imported lazily to keep Tensor/function module initialization acyclic.
  # This helper treats each FUNCTION body as its own PARAM namespace and
  # proves forwarding transitively from exact ProgramInfo input-only slots.
  from tinygrad.callify import _readonly_program_input_param_slots

  found:list[UOp] = []
  def admit(arg:UOp):
    if arg in explicit or arg.op in {Ops.PARAM, Ops.BIND} or arg.has_buffer_identity(): return
    if arg not in found: found.append(arg)

  # Do not descend through opaque executable or nested FUNCTION bodies: their
  # PARAM slot numbers are local.  Map a proven nested read slot back to that
  # invocation's concrete argument, then continue through invocation args.
  seen:set[UOp] = set()
  stack = [uret]
  while stack:
    node = stack.pop()
    if node in seen: continue
    seen.add(node)
    if node.op is Ops.CALL and node.src and node.src[0].op is Ops.PROGRAM and isinstance(node.src[0].arg, ProgramInfo):
      info = node.src[0].arg
      for slot in info.ins:
        if slot not in info.outs and slot < len(node.src)-1: admit(node.src[slot+1])
      stack.extend(node.src[1:])
      continue
    if node.op is Ops.FUNCTION and node.src and node.src[0].op is Ops.TUPLE:
      for slot in _readonly_program_input_param_slots(node.src[0].src):
        if slot < len(node.src)-1: admit(node.src[slot+1])
      stack.extend(node.src[1:])
      continue
    stack.extend(node.src)
  return found

def _readonly_program_input_explicit_aliases(uret:UOp, explicit:tuple[UOp, ...]) -> dict[UOp, UOp]:
  """Collapse a redundant CONTIGUOUS around an exact explicit model input.

  Tensor.uop_program conservatively requests contiguous inputs before the
  enclosing FUNCTION has an ABI.  Once ProgramInfo proves that this slot is
  read-only, a full canonical model allocation already satisfies that request.
  Record the exact substitution here so it does not become a second closure
  PARAM (and leave an unused, materialized copy of the explicit PARAM behind).
  """
  aliases:dict[UOp, UOp] = {}
  explicit_set = set(explicit)
  for call in uret.toposort():
    if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM or not isinstance(call.src[0].arg, ProgramInfo): continue
    info = call.src[0].arg
    for slot in info.ins:
      if slot in info.outs or slot >= len(call.src)-1: continue
      arg, direct = call.src[slot+1], call.src[slot+1]
      while direct.op is Ops.CONTIGUOUS and len(direct.src) == 1 and direct.dtype == arg.dtype and direct.numel() == arg.numel():
        direct = direct.src[0]
      if direct not in explicit_set or direct is arg: continue
      owner = memory_semantic_owner(direct)
      if owner is None or owner.semantic_class is not MemorySemanticClass.MODEL_PARAMETER: continue
      try:
        if direct.numel() * direct.dtype.itemsize != direct.buf_uop.numel() * direct.buf_uop.dtype.itemsize: continue
      except RuntimeError: continue
      aliases[arg] = direct
  return aliases

pm_transform_unique_const = PatternMatcher([
  # transform unique consts to LUNIQUE
  (UPat(Ops.CONST, src=(UPat(Ops.UNIQUE), UPat(Ops.DEVICE)), name="x"),
   lambda ctx,x: x.replace(src=(UOp(Ops.LUNIQUE, arg=next(ctx[1])), x.src[1]))),
])

pm_ctx = PatternMatcher([
  (UPat((Ops.BUFFER, Ops.BIND), name="x"), add_to_ctx),
  (UPat((Ops.AFTER, Ops.CONTIGUOUS), name="x"),
   _maybe_add_to_ctx),
])+pm_transform_unique_const

ReturnType = TypeVar('ReturnType')
class _function(Generic[ReturnType]):
  depth = 0
  def __init__(self, fxn:Callable[..., ReturnType], *, precompile:bool, precompile_backward:bool, allow_implicit:bool, grad_fxn:Callable|None):
    self.fxn = fxn
    self.precompile = precompile
    self.precompile_backward = precompile_backward
    self.allow_implicit = allow_implicit
    self.grad_fxn = grad_fxn

  def __get__(self, obj, objtype=None): return functools.partial(self.__call__, obj) if obj is not None else self

  def __call__(self, *args, **kwargs) -> ReturnType:
    st = time.perf_counter()

    params = get_state_dict((args, kwargs), tensor_type=(Tensor, UOp)).values()

    # deduplicate input_uops, keeping the first occurrence index for each unique uop
    call_uops: list[UOp] = dedup([u for t in params if (u:=t._uop).device is not None])

    # disable realize/schedule while this is running
    # run it and do surgery later
    with Context(ALLOW_DEVICE_USAGE=getenv("DEVICE_IN_FUNCTION_BUG", 0)):
      _function.depth += 1
      ret = self.fxn(*args, **kwargs)
      _function.depth -= 1
    if isinstance(ret, Tensor):
      uret = ret.uop
    elif isinstance(ret, tuple) and all(isinstance(x, Tensor) for x in ret):
      uret = UOp.maketuple(*[x.uop for x in ret])
    else:
      raise RuntimeError(f"function return type {type(ret)} not supported")

    # Opaque PROGRAM consumers need computed inputs to be explicit closure
    # values.  Otherwise the enclosing precompiled FUNCTION can allocate one
    # buffer for the producer and bind a different, unwritten buffer to the
    # PROGRAM.  Preserve the producer outside the body and pass its exact
    # result (including AFTER dependency) through a positional PARAM.
    explicit_uops = tuple(call_uops)
    readonly_aliases = _readonly_program_input_explicit_aliases(uret, explicit_uops)
    call_uops.extend(x for x in _computed_program_inputs(uret, explicit_uops) if x not in readonly_aliases)

    # replace the known inputs with params (using deduplicated slots)
    subs = {}
    for i,x in enumerate(call_uops): subs[x] = x.param_like(i)
    subs.update({alias:subs[source] for alias,source in readonly_aliases.items()})
    uret = uret.substitute(subs)

    # add contiguous to call_uops
    #call_uops = [x.contiguous() for x in call_uops]

    # the BUFFERs that are left are the implicit inputs
    num_explicit = len(call_uops)
    uret = graph_rewrite(uret, pm_ctx, (call_uops, itertools.count(0)), bottom_up=True, name="get_implicit_inputs")
    name = getattr(self.fxn, '__qualname__', None) or type(self.fxn).__qualname__
    if not self.allow_implicit:
      implicit_buffers = [x for x in call_uops[num_explicit:] if x.op is Ops.BUFFER]
      if implicit_buffers:
        buf_strs = '\n  '.join(f"{i}: dtype={b.dtype}, size={b.arg}, device={b.device}" for i,b in enumerate(implicit_buffers))
        raise RuntimeError(f"function {name} has {len(implicit_buffers)} implicit buffer(s), but allow_implicit=False\n  {buf_strs}")

    fret = uret.call(*call_uops, grad_fxn=self.grad_fxn, name=name, precompile=self.precompile,
                     precompile_backward=self.precompile_backward)

    if DEBUG >= 2:
      #signature = [(x._shape, x.dtype, x.device) for x in call_uops]
      print("  "*_function.depth+f"function {uret.key.hex()[:8]} in {(time.perf_counter()-st)*1000:8.2f} ms: {name}") # with sig {signature}")

    if isinstance(ret, tuple):
      return cast(ReturnType, tuple(Tensor(fret.gettuple(i)) for i in range(len(ret))))
    else:
      return cast(ReturnType, Tensor(fret.gettuple(0)))

# overload signatures support both @function and @function(precompile=True) syntax
@overload
def function(fxn:Callable[..., ReturnType], *, precompile:bool=False, precompile_backward:bool=False,
             allow_implicit:bool=False, grad_fxn:Callable|None=None) -> _function[ReturnType]: ...
@overload
def function(fxn:None=None, *, precompile:bool=False, precompile_backward:bool=False,
             allow_implicit:bool=False, grad_fxn:Callable|None=None) -> Callable[[Callable[..., ReturnType]], _function[ReturnType]]: ...
def function(fxn=None, *, precompile:bool=False, precompile_backward:bool=False,
             allow_implicit:bool=False, grad_fxn:Callable|None=None):
  if fxn is None:
    return lambda f: _function(f, precompile=precompile, precompile_backward=precompile_backward,
                               allow_implicit=allow_implicit, grad_fxn=grad_fxn)
  return _function(fxn, precompile=precompile, precompile_backward=precompile_backward,
                   allow_implicit=allow_implicit, grad_fxn=grad_fxn)
