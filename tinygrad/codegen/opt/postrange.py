from __future__ import annotations
import json, math, itertools
from collections import defaultdict
from typing import cast, Final
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, KernelInfo, graph_rewrite, AxisType, ssimplify, GroupOp, remove_all_tags
from tinygrad.uop.ops import axis_letters, axis_colors, axis_to_pos
from tinygrad.device import Buffer
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.helpers import colored, getenv, DEBUG, to_function_name, NOOPT, argsort, round_up, prod, merge_dicts, get_single_element, flatten
from tinygrad.helpers import ALLOW_TF32, count
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError, check
from tinygrad.codegen.simplify import pm_flatten_range
from tinygrad.renderer import Renderer
from tinygrad.schedule.indexing import BufferizeOpts
from tinygrad.schedule.rangeify import PREFILL_DBUF, PREFILL_DBUF_NBUF, prefill_dbuf_reduce_range

_TC_LOCAL_STAGE_DF_PACK_START = 238

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

  kernel_cnt: Final[defaultdict[str, int]] = defaultdict(int)
  def get_optimized_ast(self, name_override:str|None=None) -> UOp:
    if name_override is not None: name = name_override
    else:
      k_type = "r" if self.reduceop is not None else "E"
      special_uops = sorted([x for x in self.ast.toposort() if x.op is Ops.SPECIAL], key=lambda x: x.arg)
      special_ops = [colored(str(x.vmax+1), "blue" if x.arg[0] == "g" else "cyan") for x in special_uops]
      name = k_type + colored('_', 'BLACK').join(['']+special_ops+[colored(x.src[0].render(), color) for x,color in zip(self.rngs, self.colors())])
      Scheduler.kernel_cnt[(function_name := to_function_name(name))] += 1
      num = f"n{Scheduler.kernel_cnt[function_name]-1}" if Scheduler.kernel_cnt[function_name] > 1 else ""
      name += colored(num, 'BLACK')
    self.ast = graph_rewrite(self.ast, pm_flatten_range, name="flatten range")
    return self.ast.replace(arg=KernelInfo(name=name, applied_opts=tuple(self.applied_opts), dont_use_locals=self.dont_use_locals), tag=1)

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
    if not (reduceops := self.reduceops): raise KernelOptError("no reduce ops for TensorCore")
    reduceop = reduceops[0]
    if use_tensor_cores and reduceop.arg[0] is Ops.ADD:
      mul = reduceop.src[0] if reduceop.src[0].op is not Ops.CAST else reduceop.src[0].src[0]
      if mul.op is not Ops.MUL: return None
      in0, in1 = mul.src
      try:
        tensor_cores = self.ren.tensor_cores if tc_select == -1 else [self.ren.tensor_cores[tc_select]]
      except IndexError:
        raise KernelOptError(f"invalid tensor core choice {tc_select}")
      for tc in tensor_cores:
        if self.ren.target.device in ("CUDA", "NV") and tc.dtype_in == dtypes.float and not ALLOW_TF32: continue
        if tc.dtype_in == in0.dtype.scalar() and tc.dtype_in == in1.dtype.scalar() and tc.dtype_out == reduceop.dtype.scalar():
          # tensor cores have three ranges. X, Y, and REDUCE
          in0_ranges = sorted([u for u in in0.ranges if u not in in1.ranges], key=lambda x: x.arg[0], reverse=True)
          in1_ranges = sorted([u for u in in1.ranges if u not in in0.ranges], key=lambda x: x.arg[0], reverse=True)
          red_ranges = sorted(reduceop.src[1:], key=lambda x: x.arg[0], reverse=True)
          if DEBUG >= 3:
            print(f"TC({axis}): {[(x.arg[0],x.vmax+1) for x in in0_ranges]}",
                              f"{[(x.arg[0],x.vmax+1) for x in in1_ranges]} {[(x.arg[0],x.vmax+1) for x in red_ranges]}")
          if not len(in0_ranges) or not len(in1_ranges) or not len(red_ranges): continue

          # pick ranges
          # NOTE: why are in1 and in0 switched?
          axis_choices = list(itertools.product(in1_ranges, in0_ranges, red_ranges))
          if not (axis < len(axis_choices)): continue
          axes = list(axis_choices[axis])

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
          except KernelOptError: continue

          # we create the warp as a whole thing, in case some of these ranges are moved/removed later
          warp = UOp.range(tc.threads, -1, AxisType.WARP)
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

            # construct the op
            # TODO: remove tc_upcast_axes from the arg
            # do the reduce_axes always disappear? i think they don't
            # they need to be moved into the WMMA srcs
            wmma_arg = (str(tc), tc.dims, tc.dtype_in, tc.dtype_out, self.ren.target.device, tc.threads, tc_upcast_axes, ()) #, tc_reduce_axes)
            wmma_srcs = [
              UOp(Ops.CONTRACT, dtype=srcs[0].dtype.vec(tc.elements_per_thread[0]), src=(srcs[0],), arg=tc_upcast_axes[0], tag=1),
              UOp(Ops.CONTRACT, dtype=srcs[1].dtype.vec(tc.elements_per_thread[1]), src=(srcs[1],), arg=tc_upcast_axes[1], tag=1),
            ]
            wmma_srcs = _tc_local_stage_wmma_sources(wmma_srcs, _tc_local_stage_ranges((wmma_srcs[0], wmma_srcs[1])),
                                                     enabled=not (_tc_local_stage_coop_b_post_opt() or _tc_local_stage_coop_post_opt()) and
                                                     (_tc_local_stage_with_planned_local() or
                                                     not any(o.op is OptOps.LOCAL for o in (*self.planned_opts, *self.applied_opts)))
                                                     )
            wmma = UOp(Ops.WMMA, dtype=tc.dtype_out.vec(tc.elements_per_thread[2]), src=(
              wmma_srcs[0], wmma_srcs[1], UOp.const(tc.dtype_out.vec(tc.elements_per_thread[2]), 0.0)), arg=wmma_arg, tag=1)
            tc_uop = UOp(Ops.UNROLL, tc.dtype_out, (wmma,), arg=tc_upcast_axes[2], tag=1)

            # preserve extra reduces
            reduce_ranges = [x for x in UOp.sink(*reduceop.src[1:]).toposort() if x.op is Ops.RANGE and x.arg[0] not in tc_reduce_axes]
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

def bufs_from_ast(ast:UOp, dname:str) -> list[Buffer]:
  glbls = sorted([x for x in ast.backward_slice if x.op is Ops.PARAM], key=lambda x: x.arg.slot)
  return [Buffer(dname, x.max_numel(), x.dtype.base) for x in glbls]

# Step 3 warm-start: force a loop-found schedule on matmuls of a known shape signature.
# Map key = (frozenset(output dims), product(reduce dims)); value = tuple[Opt]. Default None = no-op.
_WARMSTART_OPTS = None
_WARMSTART_LOCAL_STAGE_KEYS = None
_WARMSTART_LOCAL_STAGE_DENY_KEYS = set()
_warmstart_stats = {"match": 0, "apply": 0, "error": 0}

def _tc_local_stage_mode() -> str:
  return str(getenv("PREFILL_TC_LOCAL_STAGE", "")).strip().lower()

def _tc_local_stage_with_planned_local() -> bool:
  return bool(getenv("PREFILL_TC_LOCAL_STAGE_WITH_LOCAL", 0))

def _tc_local_stage_post_opt() -> bool:
  return bool(getenv("PREFILL_TC_LOCAL_STAGE_POST", 0))

def _tc_local_stage_scalar_post_opt() -> bool:
  return bool(getenv("PREFILL_TC_LOCAL_STAGE_SCALAR_POST", 0))

def _tc_local_stage_tile_only() -> bool:
  return bool(getenv("PREFILL_TC_LOCAL_STAGE_TILE_ONLY", 0))

def _tc_local_stage_coop_b_post_opt() -> bool:
  return bool(getenv("PREFILL_TC_LOCAL_STAGE_COOP_B_POST", 0))

def _tc_local_stage_coop_post_opt() -> bool:
  return bool(getenv("PREFILL_TC_LOCAL_STAGE_COOP_POST", 0))

def _prefill_lds_pack_post_expand() -> bool:
  return bool(getenv("PREFILL_LDS_PACK_POST_EXPAND", 0))

def _prefill_lds_pack_carrier() -> bool:
  return bool(getenv("PREFILL_LDS_PACK_CARRIER", 0))

_tc_local_stage_coop_b_stats = {"seen": 0, "rewritten": 0, "dumped": 0, "skipped": 0}


def _tc_local_stage_contract_axes(x:UOp) -> tuple[int, ...]:
  if x.op is not Ops.CONTRACT: return tuple()
  if not isinstance(x.arg, tuple): return tuple()
  axes: list[int] = []
  for item in x.arg:
    if not isinstance(item, tuple): continue
    if len(item) < 1 or not isinstance(item[0], int): continue
    axes.append(item[0])
  return tuple(axes)


def _tc_local_stage_coop_b_ranges(src:UOp) -> tuple[tuple[UOp, ...], tuple[UOp, ...]]:
  ranges = sorted(src.ranges, key=lambda r: r.arg)
  # Fragment identity carries warp rows plus explicit CONTRACT fragment axes.
  contract_axes = set(_tc_local_stage_contract_axes(src))
  fragment = tuple(r for r in ranges if r.arg[-1] is AxisType.WARP or r.arg[0] in contract_axes)
  tile = tuple(r for r in ranges if r.arg[-1] is not AxisType.REDUCE and r not in fragment)
  return fragment, tile


def _tc_local_stage_src(src:UOp, ranges:tuple[UOp, ...]) -> UOp:
  return src.bufferize(*ranges, arg=BufferizeOpts(None, AddrSpace.LOCAL, removable=False)).index(*ranges)

def _tc_local_stage_wmma_sources(srcs:list[UOp], stage_ranges:tuple[UOp, ...], *, enabled:bool=True) -> list[UOp]:
  mode = _tc_local_stage_mode()
  if mode in ("", "0", "false", "off", "no") or not enabled: return srcs
  if mode not in ("a", "b", "both", "1", "true", "yes"):
    raise KernelOptError(f"PREFILL_TC_LOCAL_STAGE supports a/b/both/off, got {mode!r}")
  if getenv("PREFILL_TC_LOCAL_STAGE_DUMP"):
    print("TC_LOCAL_STAGE", {"ranges": [(r.arg, r.vmax+1) for r in stage_ranges],
                             "src0_ranges": [(r.arg, r.vmax+1) for r in sorted(srcs[0].ranges, key=lambda r: r.arg)],
                             "src1_ranges": [(r.arg, r.vmax+1) for r in sorted(srcs[1].ranges, key=lambda r: r.arg)]})
  if mode in ("a", "1", "true", "yes", "both"): srcs[0] = _tc_local_stage_src(srcs[0], stage_ranges)
  if mode in ("b", "both"): srcs[1] = _tc_local_stage_src(srcs[1], stage_ranges)
  return srcs

def _tc_local_stage_ranges(srcs:tuple[UOp, ...]) -> tuple[UOp, ...]:
  rngs = sorted(UOp.sink(*srcs).ranges, key=lambda r: r.arg)
  if _tc_local_stage_tile_only():
    return tuple(r for r in rngs if r.arg[-1] is AxisType.WARP)
  lane_rngs = tuple(r for r in rngs if r.arg[-1] in (AxisType.LOCAL, AxisType.WARP))
  return lane_rngs or tuple(r for r in rngs if r.arg[-1] is not AxisType.UPCAST)

def _tc_local_stage_wmma_post(wmma:UOp) -> UOp|None:
  if wmma.src[0].op_in_backward_slice_with_self(Ops.STAGE): return None
  srcs = _tc_local_stage_wmma_sources([wmma.src[0], wmma.src[1]], _tc_local_stage_ranges((wmma.src[0],)))
  return wmma.replace(src=(srcs[0], wmma.src[1], wmma.src[2])) if srcs[0] is not wmma.src[0] else None

pm_tc_local_stage_post = PatternMatcher([
  (UPat(Ops.WMMA, name="wmma"), _tc_local_stage_wmma_post),
])

def _tc_local_stage_coop_operand(wmma:UOp, operand_idx:int) -> UOp|None:
  mode = _tc_local_stage_mode()
  if (operand_idx == 0 and mode not in ("a", "1", "true", "yes", "both")) or (operand_idx == 1 and mode not in ("b", "both")): return None
  _tc_local_stage_coop_b_stats["seen"] += 1
  if (limit := getenv("PREFILL_TC_LOCAL_STAGE_COOP_B_LIMIT", 0)) and _tc_local_stage_coop_b_stats["rewritten"] >= limit:
    return None
  # Keep this diagnostic cheap. A full backward-slice scan on every WMMA in a
  # medium GEMM is quadratic enough to hide the actual compile result.
  src = wmma.src[operand_idx]
  if src.op in {Ops.STAGE, Ops.AFTER, Ops.BARRIER} or src.op_in_backward_slice_with_self(Ops.BARRIER): return None
  # Cooperative map is only safe today for lane+fragment operand shaping. Keep
  # fragment carriers (warp + explicit CONTRACT axes) in the staged identity and
  # close the store group over supported tile loops, including route-shaped GLOBAL
  # tile carriers.
  warp_ranges = [r for r in sorted(src.ranges, key=lambda r: r.arg) if r.arg[-1] is AxisType.WARP]
  if len(warp_ranges) != 1 or src.dtype.count != 16: return None
  lane = warp_ranges[0]
  fragment_ranges, tile_ranges = _tc_local_stage_coop_b_ranges(src)
  if lane not in fragment_ranges:
    _tc_local_stage_coop_b_stats["skipped"] += 1
    if getenv("PREFILL_TC_LOCAL_STAGE_DUMP") and _tc_local_stage_coop_b_stats["dumped"] < getenv("PREFILL_TC_LOCAL_STAGE_DUMP_LIMIT", 8):
      _tc_local_stage_coop_b_stats["dumped"] += 1
      print("TC_LOCAL_STAGE_COOP_B_SKIP", json.dumps({
        "reason": "missing_fragment_warp_axis",
        "operand": operand_idx,
        "warp": {"arg": str(lane.arg), "size": lane.vmax+1},
        "all_ranges": [{"arg": str(r.arg), "size": r.vmax+1} for r in sorted(src.ranges, key=lambda r: r.arg)],
      }))
    return None
  allowed_tile_types = {AxisType.UPCAST, AxisType.UNROLL}
  if getenv("PREFILL_TC_LOCAL_STAGE_COOP_GLOBAL", 0): allowed_tile_types.add(AxisType.GLOBAL)
  unsupported_ranges = tuple(r for r in tile_ranges if r.arg[-1] not in allowed_tile_types)
  if unsupported_ranges:
    _tc_local_stage_coop_b_stats["skipped"] += 1
    if getenv("PREFILL_TC_LOCAL_STAGE_DUMP") and _tc_local_stage_coop_b_stats["dumped"] < getenv("PREFILL_TC_LOCAL_STAGE_DUMP_LIMIT", 8):
      _tc_local_stage_coop_b_stats["dumped"] += 1
      print("TC_LOCAL_STAGE_COOP_B_SKIP", json.dumps({
        "reason": "unsupported_tile_loop_range_type",
        "operand": operand_idx,
        "warp": {"arg": str(lane.arg), "size": lane.vmax+1},
        "unsupported_ranges": [{"arg": str(r.arg), "size": r.vmax+1} for r in unsupported_ranges],
      }))
    return None
  row = lane & 15
  # 1b (paired double-buffer): PREFILL_DBUF gives the cooperative LDS tile NBUF slots and offsets BOTH the
  # store and the read by (k&1)*256, built from the same reduce carrier -> self-consistent (co-located store
  # and read share `slot`), so this is bit-exact for the still-rolled loop (one live slot per K instant) and
  # becomes the true double-buffer once 1c peels the K-loop so both slots coexist in one body. Fail-closed to
  # the single 256-wide tile when there is no const-even reduce carrier in the fragment ranges.
  nbuf = PREFILL_DBUF_NBUF() if PREFILL_DBUF() else 1
  kr = prefill_dbuf_reduce_range(src.ranges) if nbuf > 1 else None
  tile_count = prod(r.vmax+1 for r in tile_ranges)
  base = 256 * tile_count * nbuf if kr is not None else 256 * tile_count
  tile_idx = UOp.const(dtypes.weakint, 0)
  tile_mul = 1
  for r in tile_ranges[::-1]:
    tile_idx = tile_idx + r * tile_mul
    tile_mul *= r.vmax+1
  slot = ((kr % nbuf) * tile_count + tile_idx) * 256 if kr is not None else tile_idx * 256
  bsh = UOp.placeholder((base,), src.dtype.scalar(), 990 + operand_idx, addrspace=AddrSpace.LOCAL)

  def _slot_idx(i:int|UOp) -> UOp:
    return slot + row*16 + i

  stores: list[UOp] = []
  if _prefill_lds_pack_post_expand():
    # E3 diagnostic branch. This intentionally remains flag-only: placing V_PACK here is currently too early and is
    # expected to fail verifier on route-shaped 4x4. Keep it isolated so E1/E2 can be tested without interference.
    from tinygrad.renderer.isa.amd import AMDOps, Register
    pack_tag_base = _TC_LOCAL_STAGE_DF_PACK_START
    pack_tags = tuple((Register(f"v{pack_tag_base + i}", pack_tag_base + i),) for i in range(4))
    store_groups = ((((0, 1), (2, 3), (4, 5), (6, 7)), 0), (((8, 9), (10, 11), (12, 13), (14, 15)), 8))
    for group_pairs, store_slot in store_groups:
      packed_words = tuple(UOp(Ops.INS, dtypes.int32, src=(src.gep(i0), src.gep(i1)), arg=AMDOps.V_PACK, tag=pack_tags[i])
                           for i, (i0, i1) in enumerate(group_pairs))
      carry = UOp(Ops.NOOP, dtypes.int32.vec(4), src=packed_words)
      stores.append(bsh.index(_slot_idx(store_slot), dtype=bsh.dtype).store(carry, lane < 16).end())
  elif _prefill_lds_pack_carrier():
    store_groups = ((tuple(range(0, 8)), 0), (tuple(range(8, 16)), 8))
    for elems, store_slot in store_groups:
      carry = UOp(Ops.NOOP, src.dtype.scalar().vec(8), tuple(src.gep(i) for i in elems))
      stores.append(bsh.index(_slot_idx(store_slot), dtype=bsh.dtype).store(carry, lane < 16).end())
  else:
    stores = [bsh.index(_slot_idx(i), dtype=bsh.dtype).gep(0).store(src.gep(i), lane < 16).end() for i in range(16)]

  stage = UOp.group(*stores)
  stage_ranges = tile_ranges
  if stage_ranges: stage = stage.end(*stage_ranges)
  bar = UOp.barrier(stage)
  range_by_axis = {r.arg[0]: r for r in src.src[0].ranges if r.op is Ops.RANGE}
  frag_idx = UOp.const(dtypes.weakint, 0)
  mul = 1
  assert isinstance(src.arg, tuple), f"WMMA cooperative LDS staging expects CONTRACT arg, got {src.arg}"
  for axis, size in src.arg[::-1]:
    frag_idx = frag_idx + range_by_axis[axis] * mul
    mul *= size
  assert mul == src.dtype.count, f"WMMA cooperative LDS staging expected {src.dtype.count} fragment lanes, got {mul}"
  scalar = bsh.after(bar).index(_slot_idx(frag_idx)).load()
  bvec = UOp(Ops.CONTRACT, src.dtype, (scalar,), src.arg, tag=1)
  _tc_local_stage_coop_b_stats["rewritten"] += 1
  if getenv("PREFILL_TC_LOCAL_STAGE_DUMP") and _tc_local_stage_coop_b_stats["dumped"] < getenv("PREFILL_TC_LOCAL_STAGE_DUMP_LIMIT", 8):
    _tc_local_stage_coop_b_stats["dumped"] += 1
    print("TC_LOCAL_STAGE_COOP_B", json.dumps({
      "operand": operand_idx,
      "warp": {"arg": str(lane.arg), "size": lane.vmax+1},
      "src_ranges": [{"arg": str(r.arg), "size": r.vmax+1} for r in sorted(src.ranges, key=lambda r: r.arg)],
    }))
  return wmma.replace(src=wmma.src[:operand_idx] + (bvec,) + wmma.src[operand_idx+1:])

def _tc_local_stage_coop_wmma_post(wmma:UOp) -> UOp|None:
  orig = wmma
  out = _tc_local_stage_coop_operand(wmma, 0)
  if out is not None: wmma = out
  out = _tc_local_stage_coop_operand(wmma, 1)
  if out is not None: wmma = out
  return wmma if wmma is not orig else None

def _tc_local_stage_coop_b_wmma_post(wmma:UOp) -> UOp|None:
  mode = _tc_local_stage_mode()
  if mode not in ("b", "both"): return None
  return _tc_local_stage_coop_operand(wmma, 1)

pm_tc_local_stage_coop_b_post = PatternMatcher([
  (UPat(Ops.WMMA, name="wmma"), _tc_local_stage_coop_b_wmma_post),
])

pm_tc_local_stage_coop_post = PatternMatcher([
  (UPat(Ops.WMMA, name="wmma"), _tc_local_stage_coop_wmma_post),
])

def _tc_local_stage_contract_src_post(wmma:UOp) -> UOp|None:
  if wmma.src[0].op is not Ops.CONTRACT: return None
  if wmma.src[0].src[0].op_in_backward_slice_with_self(Ops.STAGE): return None
  contracted = wmma.src[0]
  stage_ranges = _tc_local_stage_ranges((contracted.src[0], contracted))
  if getenv("PREFILL_TC_LOCAL_STAGE_DUMP"):
    print("TC_LOCAL_STAGE_SCALAR", {"ranges": [(r.arg, r.vmax+1) for r in stage_ranges],
                                    "scalar_ranges": [(r.arg, r.vmax+1) for r in sorted(contracted.src[0].ranges, key=lambda r: r.arg)],
                                    "contract_ranges": [(r.arg, r.vmax+1) for r in sorted(contracted.ranges, key=lambda r: r.arg)]})
  staged = _tc_local_stage_src(contracted.src[0], stage_ranges)
  return wmma.replace(src=(contracted.replace(src=(staged,)), wmma.src[1], wmma.src[2]))

pm_tc_local_stage_scalar_post = PatternMatcher([
  (UPat(Ops.WMMA, name="wmma"), _tc_local_stage_contract_src_post),
])

def _warmstart_key(k):
  # match on CONCRETE dims only (the forward's batch dim is a symbolic JIT variable); key = (out-dims, reduce)
  red, out = 1, []
  for s, t in zip(k.full_shape, k.axis_types):
    if t in (AxisType.REDUCE, AxisType.UNROLL, AxisType.GROUP_REDUCE):
      if isinstance(s, int): red *= s
    elif isinstance(s, int): out.append(s)
  return (frozenset(out), red)

def _warmstart_match(k):
  return _WARMSTART_OPTS.get(_warmstart_key(k))

def _warmstart_local_stage_allowed(k:Scheduler) -> bool:
  # None preserves the historical global env behavior used by standalone probes. Primitive graph-GEMM routes set this
  # to a concrete key set so LDS/DBUF rewrites only touch the intended role, not every warmstarted GEMM in the model.
  key = _warmstart_key(k)
  if (key == (frozenset({512, 1024}), 4096) and getenv("PREFILL_WMMA_PIPE_PRIMITIVE", 0) and
      str(getenv("PREFILL_WMMA_PIPE_ATTN_KV_NO_LOCAL_STAGE", "1")).strip().lower() not in ("", "0", "false", "off", "no")):
    return False
  return key not in _WARMSTART_LOCAL_STAGE_DENY_KEYS and (_WARMSTART_LOCAL_STAGE_KEYS is None or key in _WARMSTART_LOCAL_STAGE_KEYS)

def _prefill_dbuf_peel(k:Scheduler) -> None:
  # 1c: peel the K reduce by 2 by applying ONE extra UNROLL(=2) on a const-even REDUCE axis. UNROLL becomes an
  # AxisType.UNROLL axis that the expander expands straight-line -- it adds NO second END over the K range, so
  # CFGContext (linearizer.py:162) never sees two ENDs over the same range and the loop-carried WAR turns into
  # ordinary intra-body forward AFTER edges. This mirrors the hand kernel's unroll-by-2 without any new UOp.
  # ROLE GUARD: this peel is a WMMA/GEMM-only primitive. Without this guard it would fire on ANY const-even REDUCE
  # (softmax, RMSNorm, ...), which is out of scope and could perturb unrelated kernels. Only proceed when this
  # kernel actually carries a tensor-core role -- a TC opt already applied, or an Ops.WMMA node in the AST.
  if not (any(o.op is OptOps.TC for o in (*k.applied_opts, *k.planned_opts))
          or any(u.op is Ops.WMMA for u in k.ast.backward_slice)): return
  # Fail-closed: no plain const-even REDUCE axis (e.g. already TC-consumed to UNROLL, or symbolic/odd) -> no-op.
  for ui, ax in enumerate(k.unrollable_dims):
    r = k.rngs[ax]
    if r.arg[-1] is AxisType.REDUCE and isinstance(s:=k.full_shape[ax], int) and s % 2 == 0 and s > 2:
      try:
        k.apply_opt(Opt(OptOps.UNROLL, ui, 2))
        return
      except KernelOptError:
        continue

def apply_opts(ast:UOp, ren:Renderer) -> UOp:
  if ast.tag is not None: return ast
  _tc_local_stage_coop_b_stats.update({"seen": 0, "rewritten": 0, "dumped": 0, "skipped": 0})
  k = Scheduler(ast, ren)
  k.convert_loop_to_global()
  if ast.arg is not None and ast.arg.opts_to_apply is not None:
    k.planned_opts = tuple(ast.arg.opts_to_apply)
    for opt in ast.arg.opts_to_apply: k.apply_opt(opt)
  elif _WARMSTART_OPTS is not None and (forced := _warmstart_match(k)) is not None:
    _warmstart_stats["match"] += 1
    if getenv("WARMSTART_DUMP") and len(_warmstart_stats.setdefault("dumps", [])) < 4:
      rs = k.reduceops
      if rs:
        r = rs[0]; s0 = r.src[0]
        s0b = s0.src[0] if s0.op is Ops.CAST else s0
        _warmstart_stats["dumps"].append(f"reduce.arg={r.arg[0]} dtype={r.dtype} src0={s0.op} "
          f"(after-cast={s0b.op} dtypes={[x.dtype for x in s0b.src][:2] if s0b.op is Ops.MUL else '?'})")
      else: _warmstart_stats["dumps"].append("NO reduceops")
    try:
      k.planned_opts = tuple(forced)
      for o in forced: k.apply_opt(o)
      _warmstart_stats["apply"] += 1
    except KernelOptError as _e:  # axis/fusion mismatch -> safe fallback to the heuristic on a fresh kernel
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
  local_stage_allowed = _warmstart_local_stage_allowed(k)
  if local_stage_allowed and PREFILL_DBUF():
    _prefill_dbuf_peel(k)
  if local_stage_allowed and _tc_local_stage_coop_post_opt() and _tc_local_stage_mode() not in ("", "0", "false", "off", "no"):
    k.ast = graph_rewrite(k.ast, pm_tc_local_stage_coop_post, name="tc local stage coop post")
  elif local_stage_allowed and _tc_local_stage_coop_b_post_opt() and _tc_local_stage_mode() not in ("", "0", "false", "off", "no"):
    k.ast = graph_rewrite(k.ast, pm_tc_local_stage_coop_b_post, name="tc local stage coop b post")
  elif local_stage_allowed and _tc_local_stage_scalar_post_opt() and _tc_local_stage_mode() not in ("", "0", "false", "off", "no"):
    k.ast = graph_rewrite(k.ast, pm_tc_local_stage_scalar_post, name="tc local stage scalar post")
  elif local_stage_allowed and _tc_local_stage_post_opt() and _tc_local_stage_mode() not in ("", "0", "false", "off", "no"):
    k.ast = graph_rewrite(k.ast, pm_tc_local_stage_post, name="tc local stage post")
  if getenv("PREFILL_TC_LOCAL_STAGE_DUMP"):
    print("TC_LOCAL_STAGE_COOP_B_STATS", json.dumps(_tc_local_stage_coop_b_stats))
  return k.get_optimized_ast(name_override=ast.arg.name if ast.arg is not None and ast.arg.name != "test" else None)
