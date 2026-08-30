from __future__ import annotations
import contextlib, math, itertools, os
from dataclasses import replace
from typing import cast, Callable
from tinygrad.uop.ops import Ops, UOp, KernelInfo, NativeAttentionRequest, graph_rewrite, AxisType, ssimplify, GroupOp, remove_all_tags
from tinygrad.uop.ops import axis_letters, axis_colors, axis_to_pos
from tinygrad.device import Buffer
from tinygrad.dtype import dtypes, PtrDType
from tinygrad.helpers import colored, DEBUG, NOOPT, argsort, round_up, prod, merge_dicts, get_single_element, flatten
from tinygrad.helpers import ALLOW_TF32, count
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError, check
from tinygrad.codegen.opt.kernel_pipeline import validate_scheduler_tile_loop_pressure
from tinygrad.codegen.simplify import pm_flatten_range
from tinygrad.renderer import Renderer

def _split_range_axis(u:UOp) -> UOp|None:
  """Recover the RANGE that `pm_split_ranges` (tinygrad/codegen/simplify.py:72-75) leaves behind in place
  of a reduce-owning RANGE `k` that is referenced somewhere under `RANGE % CONST` (Q4_K's 256-element
  block addressing does this to the K axis; that pass runs unconditionally, before apply_opts, at
  codegen/__init__.py:130). Its SINK rule substitutes every occurrence of `k` -- including inside the
  reduce's own `src[1:]` range list -- with:

    k.replace(src=(k.src[0]//v,), arg=k.arg[:-1]+(0,k.arg[-1])) * v + k.replace(src=(v,), arg=k.arg[:-1]+(1,k.arg[-1]))

  i.e. `ADD(MUL(RANGE_lo, CONST_v), RANGE_hi)`, where RANGE_lo is `k//v` (second-to-last arg field tagged
  0) and RANGE_hi is `k%v` (tagged 1); both retain everything else about the original range's arg tuple
  (id and any other fields in `arg[:-1]`) unchanged, and both keep its AxisType in `arg[-1]`. Note the
  original `k.arg` is not necessarily a bare `(id, AxisType)` pair -- it can carry extra fields before the
  trailing AxisType -- so RANGE_lo/RANGE_hi's arg tuples are one element LONGER than `k.arg`, with the new
  0/1 tag inserted second-to-last. `_apply_generic_tensor_core_opt`'s all-RANGE reduce-axis path (below)
  has no way to read that shape -- it indexes `.arg[0]` on every `reduceop.src[1:]` entry assuming a bare
  RANGE, which crashes with `TypeError: 'NoneType' object is not subscriptable` on the ADD (whose `.arg`
  is None).

  This recovers RANGE_hi (the `k%v` block-local half) as the axis this split range contributes to TC
  selection: it is the finer-grained half pm_split_ranges produced, the one whose extent a TC contraction
  actually wants to fold, and its `arg[0]` is the same as the original range's `arg[0]` (pm_split_ranges
  never touches `arg[:-1]` beyond appending the tag), so sorting it alongside never-split RANGE axes by
  `.arg[0]` (as `_apply_generic_tensor_core_opt` already does) orders it exactly where the pre-split RANGE
  would have sorted.

  Fails closed (returns None) for anything that is not precisely this shape -- this recovers one exact
  rewrite, it does not generally simplify ADD/MUL.
  """
  if u.op is not Ops.ADD or len(u.src) != 2: return None
  for mul, hi in (u.src, tuple(reversed(u.src))):
    if mul.op is not Ops.MUL or len(mul.src) != 2: continue
    lo, v = mul.src
    if lo.op is not Ops.RANGE or hi.op is not Ops.RANGE or v.op is not Ops.CONST: continue
    if hi.src[0] is not v: continue
    if not (isinstance(lo.arg, tuple) and isinstance(hi.arg, tuple)): continue
    if len(lo.arg) < 2 or len(lo.arg) != len(hi.arg): continue
    # everything but the trailing (tag, AxisType) pair must match, and the AxisType itself must match
    if lo.arg[:-2] != hi.arg[:-2] or lo.arg[-1] != hi.arg[-1]: continue
    if lo.arg[-2] != 0 or hi.arg[-2] != 1: continue
    return hi
  return None

class Scheduler:
  def __init__(self, ast:UOp, ren:Renderer):
    self.ast, self.ren = ast, ren
    self.dont_use_locals = self.ast.arg.dont_use_locals if self.ast.arg is not None else False
    self.applied_opts = list(self.ast.arg.applied_opts) if self.ast.arg is not None else []
    self.planned_opts: tuple[Opt, ...] = ()
    self.opt_range = count(start=max([x.arg[0] for x in self.rngs], default=0)+1)

  @property
  def rngs(self):
    # always in order by axistype
    return sorted([u for u in self.ast.backward_slice if u.op is Ops.RANGE and u.vmax > 0], key=lambda x: (axis_to_pos[x.arg[-1]],) + x.arg[0:-1])
  @property
  def shape_len(self) -> int: return len(self.rngs)
  @property
  def full_shape(self): return [ssimplify(x.src[0]) for x in self.rngs]
  @property
  def axis_types(self) -> list[AxisType]: return [x.arg[-1] for x in self.rngs]

  @property
  def composite_state_ranges(self) -> frozenset[UOp]:
    """Ranges owned by a stateful composite outside its reduction axis.

    Composite accumulators have a scalar state contract today.  Keep their
    logical output lanes as LOOP ranges until a backend explicitly advertises
    a lane ABI; otherwise generic UPCAST selection turns the state tuple into
    an accidental vector and changes the reduction contract.  This is
    provenance-based (not an op-name carve-out): ordinary REDUCEs are
    unaffected, and future composites can opt into lanes by carrying their own
    lowering rather than bypassing the scheduler.
    """
    ret = set()
    for red in self.ast.backward_slice:
      if red.op is Ops.REDUCE and getattr(red.arg[0], "combine_fn", None) is not None:
        ret.update(r for r in red.src[1:] if r.op is Ops.RANGE and r.arg[-1] is not AxisType.REDUCE)
    return frozenset(ret)

  # strings like ['g0', 'g1', 'l0', 'l1', 'l2', 'l3', 'l4', 'l5', 'R0', 'r0', 'r1', 'r2', 'u0', 'u1', 'u2']
  def shape_str(self) -> list[str]:
    ret: list[str] = []
    cnt: dict[AxisType, int] = {}
    for x in self.axis_types:
      cnt[x] = (cnt[x] + 1) if x in cnt else 0
      ret.append(f"{axis_letters[x]}{cnt[x]}")
    return ret
  def shape_str_to_axis(self, nms:list[str]) -> tuple[int, ...]: return tuple([self.shape_str().index(x) for x in nms])

  def copy(self) -> Scheduler:
    ret = Scheduler(self.ast, self.ren)
    ret.dont_use_locals = self.dont_use_locals
    ret.applied_opts = self.applied_opts[:]
    ret.planned_opts = self.planned_opts
    if hasattr(self, 'tensor_core'): ret.tensor_core = self.tensor_core
    return ret

  def get_optimized_ast(self, name_override:str|None=None) -> UOp:
    if name_override is not None: name = name_override
    else:
      k_type = "r" if self.reduceop is not None else "E"
      special_uops = sorted([x for x in self.ast.toposort() if x.op is Ops.SPECIAL], key=lambda x: x.arg)
      special_ops = [colored(str(x.vmax+1), "blue" if x.arg[0] == "g" else "cyan") for x in special_uops]
      name = k_type + colored('_', 'BLACK').join(['']+special_ops+[colored(x.src[0].render(), color) for x,color in zip(self.rngs, self.colors())])
      # UOp.key covers operation arguments and graph topology while intentionally excluding tags and metadata. Drop the
      # root KernelInfo too: its display name and candidate context are provenance, not generated-kernel structure.
      # The full digest makes equal optimized graphs process-independently equal and avoids counter/order collisions.
      name += colored("_" + self.ast.replace(arg=None).key.hex(), 'BLACK')
    self.ast = graph_rewrite(self.ast, pm_flatten_range, name="flatten range")
    return self.ast.replace(arg=KernelInfo(name=name, applied_opts=tuple(self.applied_opts), dont_use_locals=self.dont_use_locals,
                                           candidate_context=self.ast.arg.candidate_context), tag=1)

  def _output_rngs(self) -> list[UOp]:
    return flatten([[r for r in UOp.sink(*s.src[1:]).ranges if r.arg[-1] != AxisType.REDUCE] for s in self.ast.src if s.op is Ops.END])
  def _globalizable_rngs(self) -> list[UOp]:
    ret = [r for r in self._output_rngs() if r.arg[-1] == AxisType.LOOP]
    # exclude any output ranges from global that don't appear in all BUFFERIZE
    for x in self.ast.toposort():
      if x.op is Ops.STAGE:
        ret = [r for r in ret if r in x.ranges]
    return ret

  def convert_loop_to_global(self) -> None:
    if not self.ren.has_local: return

    globalizible_rngs = self._globalizable_rngs()
    rng = [x.replace(arg=x.arg[0:-1]+(AxisType.GLOBAL,)) if x in globalizible_rngs else x for x in self.rngs]

    self.ast = self.ast.substitute(dict(zip(self.rngs, rng)))

  def colors(self) -> list[str]:
    output_rngs = self._output_rngs()
    globalizible_rngs = self._globalizable_rngs()
    ret = []
    for x,r in zip(self.axis_types, self.rngs):
      if self.dont_use_locals and x == AxisType.GLOBAL: ret.append("BLUE")
      elif r not in output_rngs and x == AxisType.LOOP: ret.append("BLACK")
      elif r not in globalizible_rngs and x == AxisType.LOOP: ret.append("white")
      else: ret.append(axis_colors[x])
    return ret
  def colored_shape(self) -> str: return ' '.join([colored(f'{x.src[0].render():>4s}', color) for x,color in zip(self.rngs, self.colors())])

  def shift_to(self, rng:UOp, amount:int, new_type:AxisType, top:bool=False, input_new_rng:UOp|None=None):
    if (old_sz:=rng.src[0].divides(amount)) is None:
      raise KernelOptError(f"{amount} can't divide {rng.src[0]} in {self.colored_shape()}")
    new_rng = UOp.range(amount, next(self.opt_range), new_type) if input_new_rng is None else input_new_rng
    replaced_rng = rng.replace(src=(old_sz,))
    sub_axis = (new_rng * old_sz + replaced_rng) if top else (replaced_rng * amount + new_rng)
    self.ast = self.ast.substitute({rng:sub_axis}, name=f"shift {rng.arg[:-1]} {amount} {str(new_type).split('.')[1].lower()}")
    return replaced_rng, new_rng

  def ranges_of(self, *axis_type:AxisType) -> list[UOp]: return [r for r in self.rngs if r.arg[-1] in axis_type]
  def axes_of(self, *axis_type:AxisType) -> list[int]: return [i for i,t in enumerate(self.axis_types) if t in axis_type]

  def upcast_size(self): return prod(self.full_shape[a] for a in self.axes_of(AxisType.UPCAST, AxisType.UNROLL))

  def bound_expanded_reduction_pressure(self, transient_vgpr_reserve:int=128) -> None:
    """Retain loops when output/reduction expansion widens a complex reduction body.

    This is the final admission seam for schedules already encoded in the AST
    (for example rangeify or warm-start schedules), which bypass the heuristic
    candidate checks.  Pressure is expressed in target-independent carrier
    units: each independently indexed reduction stream is live for every
    expanded output/reduction lane.  A bounded output tile loop is preferred
    over shrinking reduction ILP: the loop reuses one admitted carrier set
    while its inner output pair retains contiguous writeback vectorization.
    """
    # Tensor-core schedules already pass the dedicated fragment/epilogue
    # admission in the heuristic.  Their UNROLL axes encode intrinsic carrier
    # topology and cannot be retagged as scalar reduction loops here.
    if self.reduceop is None or hasattr(self, "tensor_core"): return
    reduction_ranges = set(self.ranges_of(AxisType.REDUCE, AxisType.UNROLL, AxisType.GROUP_REDUCE))
    # Multi-axis contractions already carry explicit nested loop/staging
    # structure.  This pass owns flat reductions whose one loop plus one
    # expanded inner axis otherwise become one wide straight-line body.
    if len(reduction_ranges) > 2: return
    streams = sum(u.op is Ops.INDEX and len(u.src) >= 2 and bool(u.src[1].ranges.keys() & reduction_ranges)
                  for u in self.reduceop.backward_slice)
    if streams <= 0: return

    def admitted() -> bool:
      output_lanes = prod(int(r.vmax+1) for r in self.ranges_of(AxisType.UPCAST))
      reduction_lanes = prod(int(r.vmax+1) for r in self.ranges_of(AxisType.UNROLL))
      try:
        validate_scheduler_tile_loop_pressure(resident_accumulator_vgprs=output_lanes*reduction_lanes*streams,
          resident_fragment_vgprs=0, transient_vgpr_reserve=transient_vgpr_reserve)
        return True
      except ValueError:
        return False

    while not admitted():
      if DEBUG >= 4:
        print(f"SCHEDULER PRESSURE: streams={streams} shape={self.shape_str()} sizes={self.full_shape}")
      owned_loop = None
      if upcast := self.ranges_of(AxisType.UPCAST):
        axis = max(upcast, key=lambda x: int(x.vmax+1))
        if int(axis.vmax+1) > 2 and int(axis.vmax+1) % 2 == 0:
          inner = UOp.range(2, next(self.opt_range), AxisType.UPCAST)
          owned_loop = axis.replace(src=(axis.src[0]//2,), arg=axis.arg[:-1]+(AxisType.LOOP,))
          replacement = owned_loop*2 + inner
        else:
          replacement = owned_loop = axis.replace(arg=axis.arg[:-1]+(AxisType.LOOP,))
      elif unrolled := self.ranges_of(AxisType.UNROLL):
        axis = max(unrolled, key=lambda x: int(x.vmax+1))
        if int(axis.vmax+1) > 2 and int(axis.vmax+1) % 2 == 0:
          # Keep a bounded inner pair for ILP and carry the remaining factor
          # with a real reduction loop.  This is the inverse of a partial
          # UNROLL and avoids needlessly scalarizing the entire contraction.
          inner = UOp.range(2, next(self.opt_range), AxisType.UNROLL)
          outer = axis.replace(src=(axis.src[0]//2,), arg=axis.arg[:-1]+(AxisType.REDUCE,))
          replacement = outer*2 + inner
        else:
          replacement = axis.replace(arg=axis.arg[:-1]+(AxisType.REDUCE,))
      else: break
      self.ast = self.ast.substitute({axis:replacement}, name="bound expanded reduction pressure")
      if owned_loop is not None:
        # UPCAST ranges have no structural END because they were previously
        # expanded.  A scheduler-owned output loop must close every affected
        # root explicitly or the renderer emits an unterminated loop nest. If
        # the root already closes output ranges, add this owner to the same END
        # group so the standard END splitter establishes their canonical nest.
        def close_output_loop(root:UOp) -> UOp:
          if owned_loop not in root.backward_slice_with_self or owned_loop in root.ended_ranges: return root
          return root.replace(src=root.src+(owned_loop,)) if root.op is Ops.END else root.end(owned_loop)
        self.ast = self.ast.replace(src=tuple(close_output_loop(x) for x in self.ast.src))

  # copied from kernel.py
  @property
  def upcastable_dims(self) -> list[int]: return [i for i in self.axes_of(AxisType.GLOBAL, AxisType.LOCAL, AxisType.LOOP) \
                                                  if isinstance(s:=self.full_shape[i], int) and s > 1]
  @property
  def unrollable_dims(self) -> list[int]: return [i for i in self.axes_of(AxisType.GROUP_REDUCE, AxisType.REDUCE) \
                                                  if isinstance(s:=self.full_shape[i], int) and s > 1]

  def real_axis(self, op:OptOps, axis:int|None) -> int:
    try:
      if axis is None or op is OptOps.TC: return -1
      if op is OptOps.UNROLL: return self.unrollable_dims[axis]
      if op in {OptOps.GROUP, OptOps.GROUPTOP}: return self.axes_of(AxisType.REDUCE)[axis]
      check(axis < self.shape_len, f"invalid axis on {axis=} {op=} {self.shape_len=}")
      return axis
    except IndexError as e: raise KernelOptError from e

  def apply_opt(self, opt:Opt, append_opt:bool=True):
    if opt.op is OptOps.NOLOCALS:
      check(all(x not in {AxisType.WARP, AxisType.LOCAL, AxisType.GROUP_REDUCE} for x in self.axis_types), "no locals can't have locals")
      if append_opt: self.applied_opts.append(opt)
      self.dont_use_locals = True
      return

    if opt.op in {OptOps.LOCAL, OptOps.GROUP, OptOps.GROUPTOP}:
      check(self.ren.has_local, "locals needed for opt")

    rng = self.rngs[real_axis] if (real_axis:=self.real_axis(opt.op, opt.axis)) >= 0 else UOp(Ops.NOOP)

    opt_to_at = {
      OptOps.LOCAL: AxisType.LOCAL, OptOps.UPCAST: AxisType.UPCAST,
      OptOps.UNROLL: AxisType.UNROLL, OptOps.GROUP: AxisType.GROUP_REDUCE,
      OptOps.GROUPTOP: AxisType.GROUP_REDUCE, OptOps.THREAD: AxisType.THREAD}

    ret = None
    if opt.op in opt_to_at:
      amt:int = int(rng.vmax+1) if opt.arg == 0 else cast(int, opt.arg)

      # copied from kernel.py. prevents METAL compiler hangs
      if self.reduceop is not None and (opt.op in {OptOps.GROUP, OptOps.GROUPTOP} or \
                                        (self.group_for_reduces and opt.op not in {OptOps.NOLOCALS, OptOps.PADTO})):
        upcast_local_sz = prod([self.full_shape[a] for a in self.axes_of(AxisType.UPCAST, AxisType.WARP, AxisType.LOCAL, AxisType.GROUP_REDUCE)])
        smem_sz = amt*upcast_local_sz*self.reduceop.dtype.itemsize
        check(smem_sz <= self.ren.shared_max, f"exceeds maximum shared memory size: needs {smem_sz}, max {self.ren.shared_max}")
      if self.reduceop is not None and (opt.op in {OptOps.GROUP, OptOps.GROUPTOP}):
        # We currently dont support a group within another rudece, TODO: fix if-contexts
        reduce = [u for u in self.ast.backward_slice if u.op is Ops.REDUCE and rng in merge_dicts([r.ranges for r in u.src[1:]])][0]
        check(not any(u.arg[-1] in (AxisType.REDUCE, AxisType.UNROLL, AxisType.GROUP_REDUCE) for u in reduce.ranges),
          "cannot have a GROUP_REDUCE inside another reduce")

      if opt.op is OptOps.UNROLL:
        check(amt <= 32, "don't unroll more than 32")
        check(rng.arg[-1] in {AxisType.GROUP_REDUCE, AxisType.REDUCE}, "unroll is for GROUP_REDUCE/REDUCE")
      if opt.op is OptOps.UPCAST:
        check((self.ren is not None and self.ren.target.device == "DSP") or amt <= 16, "don't upcast more than 16")
        check(rng.arg[-1] in {AxisType.GLOBAL, AxisType.LOCAL, AxisType.LOOP}, f"upcast is for GLOBAL/LOCAL/LOOP, not {rng.arg[-1]}")
        check(rng not in self.composite_state_ranges, "composite state lanes must remain LOOP")
      if opt.op is OptOps.LOCAL:
        check(not self.dont_use_locals, "can't use locals")
        check(rng.arg[-1] in {AxisType.GLOBAL, AxisType.LOOP}, "local is for globals")
      if opt.op is OptOps.THREAD:
        check(self.ren is not None and self.ren.has_threads, "target does not support threads")
        check(self.ren is not None and self.ren.global_max is not None and amt <= self.ren.global_max[0], "too many threads")
        check(all(x is not AxisType.THREAD for x in self.axis_types), "already threaded")
        check(rng in self._globalizable_rngs(), "can't apply range to this dim")
      if opt.op in {OptOps.GROUP, OptOps.GROUPTOP}:
        check(all(x.op is not OptOps.TC for x in self.applied_opts), "no grouping with tensor cores")  # TODO: why is this wrong?
        check(not self.dont_use_locals, "can't use locals")
        check(rng.arg[-1] == AxisType.REDUCE, "group is for reduce")
      ret = self.shift_to(rng, amt, opt_to_at[opt.op], top=opt.op in {OptOps.GROUPTOP, OptOps.THREAD})
    elif opt.op is OptOps.COALESCE:
      # P3 marker-only first slice. The static q4k coalesce scorer chooses the lane-partition candidate before timing;
      # generic AST rewrites remain gated until LAYOUT_TRANSFORM/add_gpudims semantics are proved safe.
      check(opt.arg is None or isinstance(opt.arg, (int, tuple)), "coalesce arg must be an int/tuple marker")
      ret = []
    elif opt.op is OptOps.TC:
      check(len(self.applied_opts) == 0, "tensor core opts must be first") # TODO: remove the need for this by having warps
      check(opt.axis is not None, "tensor core opts must have an axis")
      check(opt.arg is not None and isinstance(opt.arg, tuple) and len(opt.arg) == 3, "tensor core opts must have valid arg")
      check(-1 <= (tc_select:=cast(tuple, opt.arg)[0]) < len(self.ren.tensor_cores), "tensor core opts must have valid tc_select")
      check(0 <= (tc_opt:=cast(tuple, opt.arg)[1]) <= 2, "tensor core opts must have valid tc_opt")
      check(0 < (use_tensor_cores:=cast(tuple, opt.arg)[2]) <= 2, "use_tensor_cores value is not valid")
      try: ret = self._apply_tc_opt(use_tensor_cores, cast(int, opt.axis), tc_select, tc_opt)
      except ValueError as e: raise KernelOptError(str(e))
      check(ret is not None, "no tensor core available")
    elif opt.op is OptOps.PADTO:
      check(rng.src[0].op is Ops.CONST, "only pad const axes")
      check(rng.arg[-1] not in {AxisType.UPCAST, AxisType.UNROLL}, "cannot pad upcasted") # TODO: why is this wrong?
      check(rng.arg[-1] is not AxisType.THREAD, "cannot pad thread")
      # ok to pad SUM if all parent ALU ops have f(0) = 0
      if (r:=self.reduceop) is not None and rng.arg[-1] in (AxisType.GROUP_REDUCE, AxisType.REDUCE):
        check(r.arg[0] is Ops.ADD and not r.op_in_backward_slice_with_self(*GroupOp.UnsafePad), f"cannot pad {r}")
      new_sz = round_up(int(rng.vmax+1), cast(int, opt.arg))
      check(rng.vmax+1 > new_sz//4, "pad adds more than quadruple the work")
      replaced_rng = UOp.range(new_sz, *rng.arg)
      replaces = {rng:replaced_rng}
      valid = replaced_rng < rng.vmax+1
      for b in self.bufs:
        if rng in (i:=b.src[1].get_idx()).backward_slice_with_self:
          replaces[b] = b.replace(src=(b.src[0],(valid&b.src[1].get_valid()).where(i, UOp.invalid())))
      self.ast = self.ast.substitute(replaces, f"padto {rng.arg[:-1]} {opt.arg}")
    elif opt.op is OptOps.SWAP:
      try:
        altrng:UOp = self.rngs[opt.arg]
      except IndexError:
        raise KernelOptError
      check(rng.arg[-1] == AxisType.GLOBAL and altrng.arg[-1] == AxisType.GLOBAL, "swap only for globals")
      self.ast = self.ast.substitute({rng:rng.replace(arg=(*altrng.arg[0:-1], rng.arg[-1]), tag=1),
                                      altrng:altrng.replace(arg=(*rng.arg[0:-1], altrng.arg[-1]), tag=1)},
                                      name=f"swap {rng.arg[:-1]} {altrng.arg[:-1]}")
      self.ast = graph_rewrite(self.ast, remove_all_tags, name="swap remove tags")
    else:
      raise KernelOptError(f"unsupported opt {opt.op}")

    if append_opt: self.applied_opts.append(opt)
    return ret

  def _apply_tc_opt(self, use_tensor_cores:int, axis:int, tc_select:int, opt_level:int) -> None|list[UOp]:
    """Orchestrator only: find the (at most one) composite reduce, offer it to any backend-specific
    kernel substitution registered for this target, then fall back to admission policy + the generic
    tensor-core search. The three concerns -- substitution, admission policy, and generic TC search --
    are each owned by a dedicated helper; see _COMPOSITE_KERNEL_SUBSTITUTIONS, _composite_tc_admission_blocked,
    and _apply_generic_tensor_core_opt."""
    if not (reduceops := self.reduceops): raise KernelOptError("no reduce ops for TensorCore")
    # One exact descriptor-owned composite route. Consume only live PARAM and
    # INDEX ownership from the canonical scalar graph; tile_fragments are
    # detached metadata and are never consulted. Any mismatch falls through
    # to the existing blanket composite guard below.
    composites = [r for r in reduceops if getattr(_composite_descriptor(r), "combine_fn", None) == "online_softmax_state"]
    if len(composites) == 1:
      red, comp = composites[0], composites[0].arg[0]
      # Backend kernel SUBSTITUTION: a target-keyed seam (matching _PREFILL_EMITTERS in
      # tinygrad/llm/fused_attention.py), not inline device/arch branches. Each substitution owns its
      # own exact admission request and either replaces self.ast wholesale or declines
      # (_TC_COMPOSITE_NO_MATCH) so the next one -- or the fail-closed admission policy below -- gets a turn.
      target = f"{self.ren.target.device.lower()}_{self.ren.target.arch}"
      for substitute in _COMPOSITE_KERNEL_SUBSTITUTIONS.get(target, ()):
        result = substitute(self, red, comp)
        if result is not _TC_COMPOSITE_NO_MATCH: return result
    # Composite admission POLICY: fail closed until the composite lane ABI is explicitly owned by a
    # backend. See _composite_tc_admission_blocked for the rule this enforces.
    if _composite_tc_admission_blocked(self): return None
    return self._apply_generic_tensor_core_opt(use_tensor_cores, axis, tc_select, opt_level, reduceops)

  def _apply_generic_tensor_core_opt(self, use_tensor_cores:int, axis:int, tc_select:int, opt_level:int,
                                      reduceops:list[UOp]) -> None|list[UOp]:
    """Generic tensor-core optimization: pick a TC-compatible dot-product reduce and lower it to WMMA.
    No composite/backend-substitution knowledge lives here -- callers have already ruled that out."""
    # Research-only Stream-K admission seam.  This is intentionally a no-op in phase one:
    # the exact packed Q4/Q6 body below remains the implementation, while a flag-on compile
    # proves the candidate context can reach this point without changing the ordinary AST.
    # Keep this branch before any candidate-local state is consumed and fail closed for malformed
    # contexts; the flag is never consulted on the production/default path.
    if os.environ.get("TINYGRAD_STREAMK_RESEARCH") == "1":
      streamk = getattr(getattr(self.ast.arg, "candidate_context", None), "streamk", None)
      if streamk is not None:
        validate = getattr(streamk, "validate", None)
        if not callable(validate): raise KernelOptError("malformed Stream-K research context")
        try: validate()
        except (TypeError, ValueError) as exc: raise KernelOptError(str(exc)) from exc
    try:
      tensor_cores = self.ren.tensor_cores if tc_select == -1 else [self.ren.tensor_cores[tc_select]]
    except IndexError:
      raise KernelOptError(f"invalid tensor core choice {tc_select}")

    # A grouped contraction can carry an outer epilogue reduction around the actual dot-product reduction. Select
    # the first reduce whose body is a tensor-core-compatible MUL instead of assuming reduceops[0] is the dot. This
    # keeps group/tile ownership in the scheduler while allowing the inner K contraction to become WMMA.
    reduceop, mul = None, None
    for candidate in reduceops:
      if candidate.arg[0] is not Ops.ADD: continue
      candidate_mul = candidate.src[0] if candidate.src[0].op is not Ops.CAST else candidate.src[0].src[0]
      if candidate_mul.op is not Ops.MUL: continue
      in0, in1 = candidate_mul.src
      compatible = any(not (self.ren.target.device in ("CUDA", "NV") and tc.dtype_in == dtypes.float and not ALLOW_TF32) and
        tc.dtype_in == in0.dtype.scalar() and tc.dtype_in == in1.dtype.scalar() and tc.dtype_out == candidate.dtype.scalar()
        for tc in tensor_cores)
      if compatible: reduceop, mul = candidate, candidate_mul; break
    if reduceop is None or mul is None: return None
    if use_tensor_cores and reduceop.arg[0] is Ops.ADD:
      in0, in1 = mul.src
      for tc in tensor_cores:
        if self.ren.target.device in ("CUDA", "NV") and tc.dtype_in == dtypes.float and not ALLOW_TF32: continue
        if tc.dtype_in == in0.dtype.scalar() and tc.dtype_in == in1.dtype.scalar() and tc.dtype_out == reduceop.dtype.scalar():
          # Fail closed: a descriptor whose operand dtype the renderer cannot express natively (e.g.
          # fp8 on gfx942, where only gfx950 has fp8_ocp) must not be emitted. The dtype decomposer
          # would emulate that dtype (fp8 -> half) after the TC opt, but WMMA render paths require
          # native operands; emulated operands either crash the renderer (K=128 fp8_index) or, worse,
          # silently reach an fp8 builtin with half-typed values (K=32). Refuse the descriptor here
          # with a message naming the capability, instead of either failure mode.
          if tc.dtype_in.scalar() not in self.ren.supported_dtypes():
            raise KernelOptError(
              f"tensor core {tc.dims} {tc.dtype_in.name}:{tc.dtype_out.name} cannot be emitted: "
              f"{type(self.ren).__name__} (target {self.ren.target.arch}) does not support "
              f"{tc.dtype_in.name} natively, and dtype emulation (fp8->half) is incompatible with "
              "WMMA descriptors")
          # tensor cores have three ranges. X, Y, and REDUCE
          in0_ranges = sorted([u for u in in0.ranges if u not in in1.ranges], key=lambda x: x.arg[0], reverse=True)
          in1_ranges = sorted([u for u in in1.ranges if u not in in0.ranges], key=lambda x: x.arg[0], reverse=True)
          # owner: the ordinary case, every reduce src is a bare RANGE.
          if all(x.op is Ops.RANGE for x in reduceop.src[1:]):
            red_ranges = sorted(reduceop.src[1:], key=lambda x: x.arg[0], reverse=True)
          else:
            # sibling: some reduce axes survived pm_split_ranges as `(k//v)*v + (k%v)` instead of a bare
            # RANGE (see _split_range_axis). Recover a representative RANGE for each such entry rather than
            # crashing on `.arg[0]` of the ADD it left behind; decline this tc (continue) if any entry is
            # neither a RANGE nor that exact split shape.
            red_axes: list[UOp]|None = []
            for x in reduceop.src[1:]:
              if x.op is Ops.RANGE: red_axes.append(x)
              elif (recovered:=_split_range_axis(x)) is not None: red_axes.append(recovered)
              else: red_axes = None; break
            if red_axes is None: continue
            red_ranges = sorted(red_axes, key=lambda x: x.arg[0], reverse=True)
          if DEBUG >= 3:
            print(f"TC({axis}): {[(x.arg[0],x.vmax+1) for x in in0_ranges]}",
                              f"{[(x.arg[0],x.vmax+1) for x in in1_ranges]} {[(x.arg[0],x.vmax+1) for x in red_ranges]}")
          if not len(in0_ranges) or not len(in1_ranges) or not len(red_ranges): continue

          # pick ranges
          # NOTE: why are in1 and in0 switched?
          axis_choices = list(itertools.product(in1_ranges, in0_ranges, red_ranges))
          if not (axis < len(axis_choices)): continue
          axes = list(axis_choices[axis])
          original_axes = tuple(axes)
          candidate_geometry = getattr(getattr(self.ast.arg, "candidate_context", None), "geometry", None)
          if _Q6_TC_AXIS_HOOK is not None and candidate_geometry is not None:
            _Q6_TC_AXIS_HOOK(self, tuple(axes), candidate_geometry)

          # tag the reduceop
          self.ast = self.ast.substitute({reduceop: reduceop.replace(tag="TC")})

          # do optimizations and save the ranges
          try:
            for i,a in enumerate(axes):
              idx = self.rngs.index(a)
              if (a.vmax+1) % tc.dims[i] != 0:
                if opt_level < 2: raise KernelOptError("tc padding requires opt_level >= 2")
                # apply_opt should return the updated range?
                self.apply_opt(Opt(OptOps.PADTO, idx, tc.dims[i]), append_opt=False) # PADTO might fail
                axes[i] = self.rngs[idx]
          except KernelOptError as exc:
            if os.environ.get("TINYGRAD_Q6_CANDIDATE_DIAGNOSTICS"):
              print(f"Q6 candidate rejected before warp lowering: {type(exc).__name__}: {exc}", flush=True)
            continue

          # we create the warp as a whole thing, in case some of these ranges are moved/removed later
          warp = UOp.range(tc.threads, -1, AxisType.WARP)
          warp_full = warp
          ne: list[UOp] = []
          for opt in tc.opts:
            if opt[0] == "l":
              axes[int(opt[1])], new_range = self.shift_to(axes[int(opt[1])], 2, AxisType.LOCAL, input_new_rng=warp%2)
              warp //= 2
            elif opt[0] == "u":
              axes[int(opt[1])], new_range = self.shift_to(axes[int(opt[1])], 2, AxisType.UPCAST)
            else: raise RuntimeError(f"unsupported opt {opt[0]} in tensor cores")
            ne.append(new_range)

          for _, amt in tc.get_reduce_axes():
            axes[2], new_range = self.shift_to(axes[2], amt, AxisType.UNROLL)
            ne.append(new_range)

          candidate_axes = candidate_contract = None
          if candidate_geometry is not None:
            # Consume the complete exact candidate while the original scalar A/B templates are still available.
            from tinygrad.codegen.opt.kernel_lds import PrecontractCandidateContract
            try: candidate_contract = PrecontractCandidateContract.create(self.ast.arg.candidate_context, tc)
            except ValueError as exc: raise KernelOptError(str(exc)) from exc
            factors = candidate_contract.factors
            axes[0], subtile_n = self.shift_to(axes[0], factors.subtiles_n, AxisType.UPCAST)
            axes[1], subtile_m = self.shift_to(axes[1], factors.subtiles_m, AxisType.UPCAST)
            # A constant gives wave-private schedules cross-wave ownership without an unsupported size-one RANGE.
            if factors.waves_m == 1: wave_m = UOp.const(dtypes.weakint, 0)
            else: axes[1], wave_m = self.shift_to(axes[1], factors.waves_m, AxisType.LOCAL)
            if factors.waves_n == 1: wave_n = UOp.const(dtypes.weakint, 0)
            else: axes[0], wave_n = self.shift_to(axes[0], factors.waves_n, AxisType.LOCAL)
            axes[2], k_substep = self.shift_to(axes[2], factors.k_substeps, AxisType.UNROLL)
            candidate_axes = (subtile_m, subtile_n, wave_m, wave_n, k_substep, axes[0], axes[1], axes[2], warp_full)

          if use_tensor_cores != 2:
            # fix the srcs
            reduceop = get_single_element([x for x in self.ast.toposort() if x.op is Ops.REDUCE and x.tag == "TC"])
            tne = [x.replace(tag=1) for x in ne]
            ret = reduceop.substitute(dict(zip(ne, tne)))
            srcs = list((ret.src[0] if ret.src[0].op is not Ops.CAST else ret.src[0].src[0]).src)
            srcs = [x.substitute(dict(zip(tne, [ne[i] for i in argsort(p)]))) for x,p in zip(srcs, tc.permutes_for_shape_str(tc.base_shape_str()))]

            # get reduce/upcast axes for the tensor cores
            tc_reduce_axes = self.shape_str_to_axis([f"r{i}" for i in range(len(tc.get_reduce_axes()))])
            base_upcast_axes = tuple([(s,2) for s in self.shape_str_to_axis(tc.base_upcast_axes())])
            tc_upcast_axes = tuple([base_upcast_axes[:int(math.log2(tc.elements_per_thread[i]))] for i in range(3)])

            # axes to range number (was done in lowerer)
            tc_upcast_axes = tuple([tuple([(self.rngs[a].arg[0], sz) for a,sz in v]) for v in tc_upcast_axes])
            tc_reduce_axes = tuple([self.rngs[a].arg[0] for a in tc_reduce_axes])

            # TODO: remove tc_upcast_axes from the WMMA arg once reduce axes are always consumed.
            wmma_arg = (str(tc), tc.dims, tc.dtype_in, tc.dtype_out, self.ren.target.device, tc.threads, tc_upcast_axes, ()) #, tc_reduce_axes)
            if candidate_axes is not None:
              from tinygrad.codegen.opt.kernel_lds import PrecontractKAxis, build_precontract_lds_stage
              subtile_m, subtile_n, wave_m, wave_n, k_substep, outer_n, outer_m, outer_k, lane = candidate_axes
              range_by_id = {r.arg[0]:r for r in self.rngs}
              try: operands, thread_axes, contracts, allocation = candidate_contract.assemble(
                in0=in0, in1=in1, original_axes=original_axes, outer_n=outer_n, outer_m=outer_m, wave_m=wave_m, wave_n=wave_n, lane=lane,
                tc_upcast_axes=tc_upcast_axes, range_by_id=range_by_id, allocation_id=None if candidate_contract.register_mode else lambda: _candidate_lds_buffer_id(self))
              except (TypeError, ValueError) as exc: raise KernelOptError(str(exc)) from exc
              factors, candidate_pipeline, register_mode = candidate_contract.factors, candidate_contract.pipeline, candidate_contract.register_mode
              pipeline_tc_uop = None
              if register_mode:
                # Direct global/L2 -> WMMA keeps ordinary CONTRACT inputs: the staged register graph flattens output-subtile
                # ownership and models row fragments as K substeps, while these inputs retain exact lane/subtile mapping.
                wmma_srcs = [
                  UOp(Ops.CONTRACT, dtype=srcs[0].dtype.vec(tc.elements_per_thread[0]), src=(srcs[0],), arg=tc_upcast_axes[0], tag=1),
                  UOp(Ops.CONTRACT, dtype=srcs[1].dtype.vec(tc.elements_per_thread[1]), src=(srcs[1],), arg=tc_upcast_axes[1], tag=1),
                ]
              if candidate_pipeline is not None and not register_mode:
                from tinygrad.codegen.opt.kernel_lds import PrecontractPipelineTemplate
                from tinygrad.codegen.opt.kernel_pipeline import (KernelStage1FragmentStage, KernelStage1ProducerStage,
                  Stage1StorageAdapter, build_stage1_uop_graph_with_storage, validate_stage1_uop_graph,
                  storage_policy_from_stage1)
                from tinygrad.codegen.opt.kernel_lds import binary_axis_count, fold_binary_axes
                template=PrecontractPipelineTemplate(candidate_geometry,tc,allocation,operands,thread_axes,
                  subtile_m,subtile_n,tuple(contracts),candidate_pipeline)
                factors=template.factors
                def _produce(epoch,slot,reuse):
                  p=template.producer(epoch,slot)
                  ready=UOp.barrier(UOp.group(*p.role_nodes) if reuse is None else UOp.group(*p.role_nodes,reuse))
                  return KernelStage1ProducerStage(epoch,slot,p.role_nodes,ready)
                def _fragments(epoch,slot,ready):
                  substeps=[]
                  for substep in range(factors.k_substeps):
                    f=template.fragments(epoch,slot,ready,substep)
                    substeps.extend(f.fragments)
                  return KernelStage1FragmentStage(epoch,slot,ready,tuple(substeps))
                def _wmma(stage,acc,_subtile):
                  chain=acc
                  for substep in range(factors.k_substeps):
                    chain=UOp(Ops.WMMA,tc.dtype_out.vec(tc.elements_per_thread[2]),
                      (stage.fragments[2*substep],stage.fragments[2*substep+1],chain),wmma_arg,tag=("pipeline_k_substep",substep))
                  return chain
                c_axes=tuple(range_by_id[a] for a,sz in tc_upcast_axes[2] if sz == 2)
                # NOTE: the binary-axis count and accumulator-elements-per-subtile-pair are both
                # tc.elements_per_thread[2] (log2 of it, for the axis count), the descriptor's own
                # per-lane accumulator width -- not a per-target constant. RDNA3's (16,16,8) gives
                # log2=3 (three binary axes, 8 elements); Metal's (2,2,2) gives log2=1 (one binary
                # axis, 2 elements). Derived exactly as PG0/PG1a derived the sibling A/B-operand
                # binary-axis folds, never branched on backend. The total accumulator element count
                # is subtiles_m*subtiles_n*tc.elements_per_thread[2] and need not equal 64: an
                # earlier version compared against a hardcoded 64, which silently forced sm*sn==8 on
                # RDNA3. Compare against the actual derived total instead so any sm*sn is admissible.
                accumulator_lane_width = tc.elements_per_thread[2]
                if len(c_axes) != binary_axis_count(tc, 2):
                  raise KernelOptError(f"buffer2 accumulator contract does not have {binary_axis_count(tc, 2)} binary axes")
                c_elem = fold_binary_axes(c_axes)
                accumulator_total = factors.subtiles_m*factors.subtiles_n*accumulator_lane_width
                accumulator_owners=[(sm*factors.subtiles_n+sn)*accumulator_lane_width+elem for sm in range(factors.subtiles_m)
                  for sn in range(factors.subtiles_n) for elem in range(accumulator_lane_width)]
                if len(accumulator_owners) != accumulator_total or set(accumulator_owners) != set(range(accumulator_total)):
                  raise KernelOptError("buffer2 accumulator ownership must be an exact unique cover of "
                                       "[0, subtiles_m*subtiles_n*elements_per_thread[2])")
                class _StageCallbacks:
                  producer = staticmethod(_produce)
                  fragments = staticmethod(_fragments)
                storage_adapter = Stage1StorageAdapter(_StageCallbacks(), storage_policy_from_stage1(candidate_pipeline))
                pipeline_plan = candidate_pipeline
                graph=build_stage1_uop_graph_with_storage(storage_adapter, pipeline_plan, outer_k.vmax+1, _wmma, subtile_count=1,
                  accumulator_elements=accumulator_total,
                  accumulator_offset=(subtile_m*factors.subtiles_n+subtile_n)*accumulator_lane_width,
                  accumulator_contract=(c_elem,tc_upcast_axes[2]),body_range_id=next(self.opt_range),accumulator_id=next(self.opt_range),
                  accumulator_dtype=tc.dtype_out, accumulator_lane_width=accumulator_lane_width)
                if errors := validate_stage1_uop_graph(graph, tc=tc):
                  raise KernelOptError("buffer2 lifecycle UOp validation failed: "+"; ".join(errors))
                pipeline_tc_uop=UOp(Ops.UNROLL,tc.dtype_out,(graph.drain[0],),arg=tc_upcast_axes[2],tag=1)
              elif not register_mode:
                # Thread the renderer's own declared bank and ordering facts (Renderer.lds_bank_dwords/
                # lds_bank_cycle_lanes, PG1; Renderer.lds_read_before_next_write_ordered, MB2) into the
                # cooperative store -- never a per-target snapshot living here or in kernel_lds.py.
                stage = build_precontract_lds_stage(candidate_geometry, tc=tc, allocation=allocation, operands=operands,
                  threads=thread_axes,k_axis=PrecontractKAxis(outer_k,k_substep,outer_k*candidate_geometry.tile[2],k_substep),
                  subtile_m=subtile_m,subtile_n=subtile_n,contracts=tuple(contracts),pipeline_plan=None,
                  lds_bank_dwords=self.ren.lds_bank_dwords,lds_bank_cycle_lanes=self.ren.lds_bank_cycle_lanes,
                  lds_read_before_next_write_ordered=self.ren.lds_read_before_next_write_ordered)
                wmma_srcs = [stage.fragment_a, stage.fragment_b]
            else:
              wmma_srcs = [
                UOp(Ops.CONTRACT, dtype=srcs[0].dtype.vec(tc.elements_per_thread[0]), src=(srcs[0],), arg=tc_upcast_axes[0], tag=1),
                UOp(Ops.CONTRACT, dtype=srcs[1].dtype.vec(tc.elements_per_thread[1]), src=(srcs[1],), arg=tc_upcast_axes[1], tag=1),
              ]
            group_accumulator = getattr(getattr(self.ast.arg, "candidate_context", None), "group_accumulator", None)
            q6_half_tc_uops = None
            if candidate_axes is not None and pipeline_tc_uop is not None: tc_uop = pipeline_tc_uop
            else:
              from tinygrad.codegen.opt.packed_weight import Q6KQ8SubgroupAccumulatorContract
              if isinstance(group_accumulator, Q6KQ8SubgroupAccumulatorContract):
                # Mask before CONTRACT, while logical K is still explicit.
                # Masking carrier GEPs here is invalid: later output-axis
                # UNROLL substitution can permute those vector lanes.
                if candidate_axes is None or candidate_pipeline is not None or register_mode or stage.fragment_b_k16 is None:
                  raise KernelOptError("Q6_K paired K16 IMMA requires logical-K-masked LDS B fragments")
                low_b,high_b=stage.fragment_b_k16
                def _half_imma(b:UOp, tag:str) -> UOp:
                  wmma = UOp(Ops.WMMA, dtype=tc.dtype_out.vec(tc.elements_per_thread[2]), src=(wmma_srcs[0], b,
                    UOp.const(tc.dtype_out.vec(tc.elements_per_thread[2]), 0.0)), arg=wmma_arg, tag=("q6_k16",tag))
                  return UOp(Ops.UNROLL, tc.dtype_out, (wmma,), arg=tc_upcast_axes[2], tag=("q6_k16",tag))
                q6_half_tc_uops = (_half_imma(low_b, "low"), _half_imma(high_b, "high"))
                tc_uop = q6_half_tc_uops[0]
              else:
                wmma = UOp(Ops.WMMA, dtype=tc.dtype_out.vec(tc.elements_per_thread[2]), src=(
                  wmma_srcs[0], wmma_srcs[1], UOp.const(tc.dtype_out.vec(tc.elements_per_thread[2]), 0.0)), arg=wmma_arg, tag=1)
                tc_uop = UOp(Ops.UNROLL, tc.dtype_out, (wmma,), arg=tc_upcast_axes[2], tag=1)

            # The direct Q4_K/Q8_1 path deliberately makes one candidate K tile
            # exactly one IMMA K32 group. Apply its affine metadata correction
            # here, while every accumulator scalar still owns its precise
            # logical (M,N,K32) coordinate, and only then reduce outer K. This
            # avoids both a global [groups,M,N] materialization and the invalid
            # alternative of first combining differently-scaled int32 groups.
            if candidate_axes is not None and group_accumulator is not None:
              from tinygrad.codegen.opt.kernel_lds import PackedPrecontractOperandTemplate, binary_axis_count, fold_binary_axes
              from tinygrad.codegen.opt.packed_weight import Q4KQ8GroupAccumulatorContract, Q6KQ8SubgroupAccumulatorContract
              if not isinstance(group_accumulator, (Q4KQ8GroupAccumulatorContract, Q6KQ8SubgroupAccumulatorContract)):
                raise KernelOptError("candidate group accumulator must use a typed K-quant/Q8_1 ABI")
              is_q6 = isinstance(group_accumulator, Q6KQ8SubgroupAccumulatorContract)
              if (self.ren.target.device not in ("NV", "CUDA") or tc.dims != (8, 16, 32) or tc.dtype_in != dtypes.char or
                  tc.dtype_out != dtypes.int or tc.elements_per_thread != (16, 8, 4) or tc.threads != 32):
                raise KernelOptError("K-quant/Q8_1 group accumulator requires NVIDIA m16n8k32 s8 IMMA")
              if (candidate_geometry.tile[2] % 32 or factors.k_substeps != candidate_geometry.tile[2]//32 or
                  candidate_pipeline is not None or register_mode):
                raise KernelOptError("K-quant/Q8_1 correction requires a non-pipelined LDS tile made of exact K32 substeps")
              if not (isinstance(operands[0], PackedPrecontractOperandTemplate) and
                      isinstance(operands[1], PackedPrecontractOperandTemplate)):
                raise KernelOptError("K-quant/Q8_1 correction requires typed packed A and B operands")
              if (operands[0].fragment_provider != group_accumulator.activation or
                  operands[1].fragment_provider != group_accumulator.weight):
                raise KernelOptError("K-quant/Q8_1 correction providers do not match staged operands")
              if is_q6 and q6_half_tc_uops is None:
                raise KernelOptError("Q6_K/Q8_1 correction requires paired masked K16 IMMA subtotals")
              c_axes = tuple(range_by_id[a] for a, size in tc_upcast_axes[2] if size == 2)
              if len(c_axes) != binary_axis_count(tc, 2):
                raise KernelOptError("K-quant/Q8_1 accumulator does not retain the descriptor output axes")
              c_elem = fold_binary_axes(c_axes)
              local_m = wave_m*(factors.subtiles_m*tc.dims[1]) + subtile_m*tc.dims[1] + lane//4 + 8*(c_elem//2)
              local_n = wave_n*(factors.subtiles_n*tc.dims[0]) + subtile_n*tc.dims[0] + 2*(lane%4) + c_elem%2
              logical_m, logical_n = outer_m*candidate_geometry.tile[0]+local_m, outer_n*candidate_geometry.tile[1]+local_n
              metadata_bytes_per_row = factors.vectors_per_row*4
              windows = {window.role:window for window in candidate_geometry.lds_windows}
              if all(windows[role].stride_bytes >= candidate_geometry.tile[2]+metadata_bytes_per_row for role in ("A", "B")):
                ordered_metadata = stage.allocation.after(stage.barrier)
                def _metadata(role, local_row):
                  window = windows[role]
                  byte_index = window.base + local_row*window.stride_bytes + candidate_geometry.tile[2] + k_substep*8
                  byte_values = tuple(ordered_metadata.index(byte_index+i).load().bitcast(dtypes.uint8).cast(dtypes.uint16) for i in range(4))
                  halves = tuple(byte_values[2*part].bitwise_or(byte_values[2*part+1].lshift(8)).bitcast(dtypes.half) for part in range(2))
                  return UOp(Ops.STACK, dtypes.half.vec(2), halves)
                integer_result = q6_half_tc_uops if is_q6 else tc_uop
                tc_uop = group_accumulator.combine_staged(integer_result, _metadata("B", local_n), _metadata("A", local_m))
              else:
                k_base = outer_k*candidate_geometry.tile[2]+k_substep*32
                if is_q6:
                  tc_uop = group_accumulator.correct(operands[1].source, operands[0].source, row=logical_m, column=logical_n,
                                                     k_base=k_base, integer_dots=q6_half_tc_uops)
                else:
                  tc_uop = group_accumulator.correct(operands[1].source, operands[0].source, row=logical_m, column=logical_n,
                                                     k_base=k_base, integer_dot=tc_uop)

            # preserve extra reduces
            reduce_ranges = [x for x in UOp.sink(*reduceop.src[1:]).toposort() if x.op is Ops.RANGE and x.arg[0] not in tc_reduce_axes]
            if candidate_axes is not None and pipeline_tc_uop is not None:
              pipeline_reduce_ranges = [x for x in reduce_ranges if x.arg[-1] is AxisType.REDUCE]
              if len(pipeline_reduce_ranges) != 1 or pipeline_reduce_ranges[0].arg != outer_k.arg:
                raise KernelOptError(f"buffer2 must consume exactly the original outer-K reduce: "
                                     f"expected {outer_k.arg!r}, got {[x.arg for x in pipeline_reduce_ranges]!r}")
              if outer_k in pipeline_tc_uop.backward_slice or k_substep in pipeline_tc_uop.backward_slice:
                raise KernelOptError("buffer2 replacement retained an original K ownership range")
              reduce_ranges = []
            if len(reduce_ranges): tc_uop = UOp(Ops.REDUCE, tc_uop.dtype, (tc_uop,)+tuple(reduce_ranges), (Ops.ADD, ()))
            self.ast = self.ast.substitute({reduceop: tc_uop})
          self.tensor_core = tc
          return axes
    return None

  # helpers for hand_coded_optimizations
  @property
  def reduceops(self) -> list[UOp]: return [x for x in self.ast.backward_slice if x.op is Ops.REDUCE]
  @property
  def reduceop(self) -> UOp|None:
    if not (red := self.reduceops): return None
    return UOp(Ops.REDUCE, red[0].dtype, red[0].src, red[0].arg)
  @property
  def bufs(self) -> list[UOp]: return [x for x in self.ast.toposort() if x.op is Ops.INDEX][::-1]
  @property
  def output_shape(self):
    return [s if at not in {AxisType.REDUCE, AxisType.UNROLL, AxisType.GROUP_REDUCE} else 1 for s,at in zip(self.full_shape, self.axis_types)]
  @property
  def upcasted(self) -> int: return len(self.axes_of(AxisType.UPCAST, AxisType.UNROLL))
  @property
  def group_for_reduces(self) -> int: return len(self.axes_of(AxisType.GROUP_REDUCE))

def _composite_descriptor(reduceop:UOp):
  """The composite descriptor payload carried in a REDUCE op's arg tuple, if any."""
  return reduceop.arg[0] if isinstance(reduceop.arg, tuple) and reduceop.arg else None

# Sentinel: a composite kernel substitution's precondition wasn't met for this composite (e.g. no
# attention_grid, or the final admission check failed) -- try the next substitution registered for
# this target, or fall through to the generic admission policy if none matched. Distinct from `None`,
# which a substitution returns to mean "matched, but hard-reject the whole tensor-core opt here".
_TC_COMPOSITE_NO_MATCH = object()

def _composite_native_grid_attention_swap(k:Scheduler, red:UOp, comp) -> None|list[UOp]:
  """Exact semantic GQA native-grid attention swap (gfx1100). The descriptor is only an admission
  request: all four live PARAM slots and their physical sizes must match before the generic grid
  builder can replace the scalar semantic graph."""
  grid = getattr(comp, "attention_grid", None)
  if grid is None: return _TC_COMPOSITE_NO_MATCH
  try: grid.validate()
  except ValueError: return None
  request = k.ast.arg.required_native_attention
  if request is not None and (request.grid != grid or request.candidate_context != getattr(comp,"attention_context",None) or
                              request.combine_fn != comp.combine_fn):
    raise KernelOptError("native attention request does not match composite owner")
  params = sorted({u for u in k.ast.toposort() if u.op is Ops.PARAM}, key=lambda u:u.arg.slot)
  sizes = (grid.q_heads*grid.q_tokens*128, grid.q_heads*grid.q_tokens*128,
           grid.kv_heads*grid.kv_tokens*128, grid.kv_heads*grid.kv_tokens*128)
  scale = next((float(s.arg) for s in red.src[0].src if s.op is Ops.CONST and s.dtype.scalar() in dtypes.floats), None) \
    if red.src[0].op is Ops.MUL else None
  if not ([p.arg.slot for p in params] == [0,1,2,3] and tuple(p.ptrdtype.size for p in params) == sizes and
          all(p.ptrdtype.base == dtypes.half for p in params) and scale is not None and math.isfinite(scale) and scale > 0):
    return _TC_COMPOSITE_NO_MATCH
  from tinygrad.schedule.wmma.flash_prefill import FlashPrefillAttentionSpec
  # Lazily imported (matching how this file imports the builder today) to sidestep any
  # tinygrad.llm <-> tinygrad.codegen.opt import-order coupling.
  from tinygrad.llm.fused_attention import _PREFILL_EMITTERS
  context = getattr(comp, "attention_context", None)
  existing_context = k.ast.arg.candidate_context
  if context is not None:
    context.validate()
    if existing_context is not None and existing_context != context:
      raise KernelOptError("conflicting shared attention candidate context at native handoff")
  # The SPEC owns the topology params here too (source-of-truth for q_tokens/heads/kv_tokens/
  # scale/causal/valid_kv/query_start/acc slicing), and this now routes through the SAME
  # target-keyed _PREFILL_EMITTERS seam fused_attention.py's custom_kernel_attention uses --
  # no site calls the gfx1100 builder directly. The only difference from that call site is
  # this one injects its carried-forward kernel_info (self.ast.arg's existing fields via
  # replace()) through spec.emit()'s optional kernel_info override.
  # Same fix as fused_attention.py:custom_kernel_attention (P4a follow-up): the literal `8`
  # fallback (and a context whose own acc_blocks==8 default was never overridden) is the
  # Hd=128 full-accumulator value; forwarding it unconditionally would break the full-drain
  # case at Hd!=head_dim==128. Pass None in the full-default case so
  # FlashPrefillAttentionSpec.__post_init__ resolves via head_dim//16 -- byte-identical at
  # head_dim=128 (None -> 128//16==8, the exact value previously forwarded).
  ctx_full_default = context is None or (context.output_block_base == 0 and context.acc_blocks == 8)
  spec = FlashPrefillAttentionSpec(Hq=grid.q_heads, Hkv=grid.kv_heads, Hd=grid.head_dim,
    q_tokens=grid.q_tokens, kv_tokens=grid.kv_tokens, causal=comp.attention_causal, scale=scale,
    output_block_base=context.output_block_base if context is not None else 0,
    acc_blocks=None if ctx_full_default else context.acc_blocks)
  try: spec.validate()
  except ValueError: return None
  fxn = _PREFILL_EMITTERS[spec.target](spec,
    kernel_info=replace(k.ast.arg, candidate_context=context if context is not None else existing_context))
  k.ast = fxn(params[0], params[1], params[2], params[3])
  k.tensor_core = next((tc for tc in k.ren.tensor_cores if tc.dims == (16,16,16) and
                       tc.dtype_in == dtypes.half and tc.dtype_out == dtypes.float), None)
  if k.tensor_core is None: raise KernelOptError("gfx1100 grid attention requires fp16 16x16x16 tensor core")
  return []

def _composite_q16_tile_carrier_swap(k:Scheduler, red:UOp, comp) -> None|list[UOp]:
  """Exact-KV16 typed-fragment ABI swap (gfx1100, online_softmax_qk_pv_v1 carrier). Requires the
  renderer's native_repack_matcher and an exact score/value/output (16,16,16) carrier shape."""
  if getattr(k.ren, "native_repack_matcher", None) is None: return _TC_COMPOSITE_NO_MATCH
  carrier = comp.tile_carrier
  try: carrier.validate()
  except (AttributeError, ValueError): carrier = None
  all_params = sorted({u for u in k.ast.toposort() if u.op is Ops.PARAM}, key=lambda u:u.arg.slot)
  score_params = sorted({u for u in red.src[0].toposort() if u.op is Ops.PARAM}, key=lambda u:u.arg.slot)
  aux = tuple(s for s in red.src[1:] if s.op is not Ops.RANGE)
  value_params = sorted({u for s in aux for u in s.toposort() if u.op is Ops.PARAM}, key=lambda u:u.arg.slot)
  stores = [u for u in k.ast.toposort() if u.op is Ops.STORE]
  output_params = sorted({p for st in stores for p in st.src[0].toposort() if p.op is Ops.PARAM}, key=lambda u:u.arg.slot)
  scale = next((float(s.arg) for s in red.src[0].src if s.op is Ops.CONST and s.dtype.scalar() in dtypes.floats), None) \
    if red.src[0].op is Ops.MUL else None
  exact = carrier is not None and carrier.typed_fragment_abi == "online_softmax_qk_pv_v1" and \
    carrier.score_shape == carrier.value_shape == carrier.output_shape == (16,16,16) and \
    comp.slot_shapes == ((1,1,16),(1,1,16),(1,1,16,16)) and comp.lane_shapes == ((),(),(16,)) and \
    len(aux) == 1 and [u.arg.slot for u in all_params] == [0,1,2,3] and \
    [u.arg.slot for u in score_params] == [1,2] and [u.arg.slot for u in value_params] == [3] and \
    [u.arg.slot for u in output_params] == [0] and scale is not None and math.isfinite(scale) and scale > 0
  if not exact: return _TC_COMPOSITE_NO_MATCH
  from tinygrad.schedule.wmma import amd_gfx1100_q16_attention
  k.ast = amd_gfx1100_q16_attention(score_params[0], score_params[1], value_params[0], output_params[0],
                                        scale=scale, kernel_info=k.ast.arg)
  k.tensor_core = next((tc for tc in k.ren.tensor_cores if tc.dims == (16,16,16) and
                       tc.dtype_in == dtypes.half and tc.dtype_out == dtypes.float), None)
  if k.tensor_core is None: raise KernelOptError("gfx1100 q16 attention requires fp16 16x16x16 tensor core")
  return []

# Target-keyed backend kernel substitution seam for exact-descriptor composite reduces, mirroring
# _PREFILL_EMITTERS in tinygrad/llm/fused_attention.py (same "amd_gfx1100"-shaped key). A new backend
# or a new exact route is a new dict entry, not a new inline device/arch branch in _apply_tc_opt.
# Entries for one target are tried in order; each declines via _TC_COMPOSITE_NO_MATCH.
_COMPOSITE_KERNEL_SUBSTITUTIONS: dict[str, tuple[Callable[[Scheduler,UOp,object], "None|list[UOp]|object"], ...]] = {
  "amd_gfx1100": (_composite_native_grid_attention_swap, _composite_q16_tile_carrier_swap),
}

def _composite_tc_admission_blocked(k:Scheduler) -> bool:
  """Composite reductions currently consume scalar score/state values.  A tensor-core rewrite would
  pack the QK contraction into fragment lanes before the online combine can read it, violating that
  ABI (and producing either a shape error or numerically corrupted softmax state).  Keep this boundary
  fail-closed until the composite lane ABI is explicitly owned by a backend.  Ordinary REDUCE matmuls
  have no composite state ranges and retain the existing WMMA path unchanged."""
  composite_carriers = []
  for reduceop in k.reduceops:
    carrier = getattr(_composite_descriptor(reduceop), "tile_carrier", None)
    if carrier is not None:
      try: carrier.validate()
      except (AttributeError, ValueError): return True
      composite_carriers.append(carrier)
  # Carrier metadata is now part of WMMA candidate validation.  It proves the score/value/output tile
  # geometry, but does not by itself authorize fragment lowering; the typed online-softmax ABI remains
  # fail-closed.
  return bool(k.composite_state_ranges or composite_carriers)

def bufs_from_ast(ast:UOp, dname:str) -> list[Buffer]:
  glbls = sorted([x for x in ast.backward_slice if x.op is Ops.PARAM], key=lambda x: x.arg.slot)
  return [Buffer(dname, x.max_numel(), x.dtype.base) for x in glbls]

# Step 3 warm-start: force a loop-found schedule on matmuls of a known shape signature.
# Map key = (frozenset(output dims), product(reduce dims)); value = tuple[Opt]. Default None = no-op.
_WARMSTART_OPTS = None
_WARMSTART_CANDIDATE_CONTEXTS = None
_warmstart_stats = {"match": 0, "apply": 0, "error": 0}

@contextlib.contextmanager
def warmstart_candidate_state(opts, candidate_contexts=None):
  """Install all candidate-sensitive warmstart state for one compile/capture scope."""
  global _WARMSTART_OPTS, _WARMSTART_CANDIDATE_CONTEXTS
  installed_opts = None if opts is None else dict(opts)
  installed_contexts = None if candidate_contexts is None else dict(candidate_contexts)
  if installed_contexts:
    missing = installed_contexts.keys() - (installed_opts or {}).keys()
    if missing: raise RuntimeError(f"warmstart candidate contexts lack schedule opts for keys: {sorted(map(repr, missing))}")
    active_contexts = _WARMSTART_CANDIDATE_CONTEXTS or {}
    collisions = {key for key, context in installed_contexts.items()
                  if key in active_contexts and active_contexts[key] != context}
    if collisions: raise RuntimeError(f"warmstart candidate context collision for keys: {sorted(map(repr, collisions))}")
  saved = (_WARMSTART_OPTS, _WARMSTART_CANDIDATE_CONTEXTS)
  _WARMSTART_OPTS, _WARMSTART_CANDIDATE_CONTEXTS = installed_opts, installed_contexts
  try: yield
  finally: _WARMSTART_OPTS, _WARMSTART_CANDIDATE_CONTEXTS = saved

def _candidate_lds_buffer_id(k:Scheduler) -> int:
  buffer_id = next(k.opt_range)
  if any(r.arg[0] == buffer_id for r in k.rngs): raise KernelOptError("candidate LDS allocation ID collides with a live range")
  return buffer_id

# Packed-weight quant formats (packed_weight.py: PackedWeightTransform.storage_dtype) carry their weight
# bytes as a narrow uint PARAM instead of the dense fp16 activation/weight dtype. Two different-quant
# linears (e.g. Q4_K and Q6_K ffn_down) can share an identical (m, n, k) real shape -- their raw packed
# PARAM dtype (uint32 for Q4_K, uint16 for Q6_K) is the one thing that already differs on the AST at
# apply_opts time, before any candidate_context is attached, so it's what the key uses to tell them apart.
_PACKED_STORAGE_DTYPES = (dtypes.uint16, dtypes.uint32)

# Research-only seam: callers may inspect the packed-fragment scheduler AST before
# final lowering.  No callback is installed by default and the AST is never mutated.
_Q6_PRE_SCHEDULER_HOOK = None
_Q6_PRE_GLOBAL_HOOK = None
_Q6_TC_AXIS_HOOK = None
def set_q6_pre_scheduler_hook(callback):
  global _Q6_PRE_SCHEDULER_HOOK
  _Q6_PRE_SCHEDULER_HOOK = callback
def set_q6_pre_global_hook(callback):
  global _Q6_PRE_GLOBAL_HOOK
  _Q6_PRE_GLOBAL_HOOK = callback
def set_q6_tc_axis_hook(callback):
  global _Q6_TC_AXIS_HOOK
  _Q6_TC_AXIS_HOOK = callback

def warmstart_key(out_dims, reduce, packed_dtype=None):
  """Public key builder mirroring `_warmstart_key`, for callers (e.g. model-init warmstart-table
  precomputation) that don't have a live Scheduler/AST to derive the discriminator from directly.
  `packed_dtype` is the packed-weight PARAM's storage dtype (e.g. dtypes.uint16/uint32) for a
  packed-weight candidate, or None for the plain dense (non-packed) path."""
  packed_dtypes = frozenset(packed_dtype) if isinstance(packed_dtype, (tuple, list, set, frozenset)) else \
    frozenset((packed_dtype,)) if packed_dtype is not None else frozenset()
  if not packed_dtypes.issubset(_PACKED_STORAGE_DTYPES):
    raise ValueError(f"packed warmstart dtype discriminator must use canonical packed storage dtypes, got {packed_dtypes}")
  return (frozenset(out_dims), reduce, packed_dtypes)

def _warmstart_key(k):
  # match on CONCRETE dims only (the forward's batch dim is a symbolic JIT variable); key = (out-dims, reduce,
  # packed-weight-dtype-discriminator). The discriminator is empty for kernels with no packed-weight PARAM
  # (the plain dense fp16 path), so it doesn't change that path's keying.
  red, out = 1, []
  for s, t in zip(k.full_shape, k.axis_types):
    if t in (AxisType.REDUCE, AxisType.UNROLL, AxisType.GROUP_REDUCE):
      if isinstance(s, int): red *= s
    elif isinstance(s, int): out.append(s)
  packed_dtypes = frozenset(u.dtype.base for u in k.ast.backward_slice
                             if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType) and u.dtype.base in _PACKED_STORAGE_DTYPES)
  return (frozenset(out), red, packed_dtypes)

def _warmstart_match(k):
  return _WARMSTART_OPTS.get(_warmstart_key(k))

def apply_opts(ast:UOp, ren:Renderer) -> UOp:
  if ast.tag is not None: return ast
  k = Scheduler(ast, ren)
  if _Q6_PRE_GLOBAL_HOOK is not None:
    _Q6_PRE_GLOBAL_HOOK(ast, k)
  k.convert_loop_to_global()
  required_native = ast.arg.required_native_attention if isinstance(ast.arg,KernelInfo) else None
  if required_native is not None:
    if not isinstance(required_native,NativeAttentionRequest): raise KernelOptError("malformed native attention request")
    required_native.validate()
    if ren.target.device != "AMD" or ren.target.arch != "gfx1100": raise KernelOptError("native attention request target mismatch")
    if ast.arg.candidate_context != required_native.candidate_context: raise KernelOptError("native attention request context mismatch")
    native_opt=Opt(OptOps.TC,0,(0,0,1)); k.planned_opts=(native_opt,); k.apply_opt(native_opt)
  elif ast.arg is not None and ast.arg.opts_to_apply is not None:
    k.planned_opts = tuple(ast.arg.opts_to_apply)
    for opt in ast.arg.opts_to_apply: k.apply_opt(opt)
  elif _WARMSTART_OPTS is not None and (forced := _warmstart_match(k)) is not None:
    _warmstart_stats["match"] += 1
    warm_key = _warmstart_key(k)
    if _WARMSTART_CANDIDATE_CONTEXTS is not None and (candidate_context := _WARMSTART_CANDIDATE_CONTEXTS.get(warm_key)) is not None:
      k.ast = k.ast.replace(arg=replace(k.ast.arg, candidate_context=candidate_context))
    try:
      k.planned_opts = tuple(forced)
      for o in forced: k.apply_opt(o)
      _warmstart_stats["apply"] += 1
    except KernelOptError as _e:  # axis/fusion mismatch -> safe fallback to the heuristic on a fresh kernel
      if getattr(k.ast.arg, "candidate_context", None) is not None: raise
      _warmstart_stats.setdefault("errs", []).append(f"{[str(o) for o in forced]} -> {str(_e)[:90]}")
      _warmstart_stats["error"] += 1
      k = Scheduler(ast, ren); k.convert_loop_to_global()
      if not NOOPT and not any(u.op is Ops.STAGE for u in ast.backward_slice):
        from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
        k = hand_coded_optimizations(k)
  elif not NOOPT and (ast.arg is None or ast.arg.applied_opts == ()):
    from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
    # NOTE: hand_coded_optimizations doesn't support multiblock opts yet
    if not any(u.op is Ops.STAGE for u in ast.backward_slice):
      k = hand_coded_optimizations(k)
  k.bound_expanded_reduction_pressure()
  optimized = k.get_optimized_ast(name_override=ast.arg.name if ast.arg is not None and ast.arg.name != "test" else None)
  if _Q6_PRE_SCHEDULER_HOOK is not None and getattr(k.ast.arg, "candidate_context", None) is not None:
    replacement = _Q6_PRE_SCHEDULER_HOOK(optimized)
    if replacement is not None: optimized = replacement
  return optimized
