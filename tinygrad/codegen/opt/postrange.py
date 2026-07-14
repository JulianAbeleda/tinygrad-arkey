from __future__ import annotations
import contextlib, math, itertools
from dataclasses import replace
from collections import defaultdict
from typing import cast, Final
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, KernelInfo, graph_rewrite, AxisType, ssimplify, GroupOp, remove_all_tags
from tinygrad.uop.ops import axis_letters, axis_colors, axis_to_pos
from tinygrad.device import Buffer
from tinygrad.dtype import dtypes, AddrSpace, PtrDType
from tinygrad.helpers import colored, getenv, DEBUG, to_function_name, NOOPT, argsort, round_up, prod, merge_dicts, get_single_element, flatten
from tinygrad.helpers import ALLOW_TF32, count
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError, check
from tinygrad.codegen.simplify import pm_flatten_range
from tinygrad.renderer import Renderer

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
          original_axes = tuple(axes)
          candidate_geometry = getattr(getattr(self.ast.arg, "candidate_context", None), "geometry", None)

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

          candidate_axes = None
          if candidate_geometry is not None:
            # Consume the complete exact candidate while the original scalar A/B templates are still available.
            from tinygrad.codegen.opt.kernel_lds import derive_precontract_factors, derive_precontract_shape_factors
            candidate_pipeline = getattr(self.ast.arg.candidate_context, "pipeline", None)
            register_candidate = getattr(getattr(candidate_pipeline, "storage", None), "kind", None) == "global_register_resident"
            try: factors = (derive_precontract_shape_factors(candidate_geometry, tc) if register_candidate else
                            derive_precontract_factors(candidate_geometry, tc))
            except ValueError as exc: raise KernelOptError(str(exc)) from exc
            axes[0], subtile_n = self.shift_to(axes[0], factors.subtiles_n, AxisType.UPCAST)
            axes[1], subtile_m = self.shift_to(axes[1], factors.subtiles_m, AxisType.UPCAST)
            # Wave-private register schedules have no cross-wave axis.  A
            # constant encodes that ownership without creating a size-one
            # RANGE, which shift_to intentionally does not support.
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

            # construct the op
            # TODO: remove tc_upcast_axes from the arg
            # do the reduce_axes always disappear? i think they don't
            # they need to be moved into the WMMA srcs
            wmma_arg = (str(tc), tc.dims, tc.dtype_in, tc.dtype_out, self.ren.target.device, tc.threads, tc_upcast_axes, ()) #, tc_reduce_axes)
            if candidate_axes is not None:
              from tinygrad.codegen.opt.kernel_lds import (PackedPrecontractOperandTemplate, PrecontractContractSpec,
                PrecontractKAxis, PrecontractOperandTemplate, PrecontractThreadAxes, build_precontract_lds_stage)
              subtile_m, subtile_n, wave_m, wave_n, k_substep, outer_n, outer_m, outer_k, lane = candidate_axes
              range_by_id = {r.arg[0]:r for r in self.rngs}
              contracts = []
              for operand_idx, role in enumerate(("A", "B")):
                contract_axes = tuple(range_by_id[a] for a,sz in tc_upcast_axes[operand_idx] if sz == 2)
                if len(contract_axes) != 4: raise KernelOptError(f"candidate {role} contract does not have four binary axes")
                element = ((contract_axes[0]*2+contract_axes[1])*2+contract_axes[2])*2+contract_axes[3]
                contracts.append(PrecontractContractSpec(role, contract_axes, tc_upcast_axes[operand_idx], element,
                  tuple(tc.lane_map.remaps()[operand_idx].items())))
              candidate_pipeline = getattr(self.ast.arg.candidate_context, "pipeline", None)
              candidate_policy = None
              if candidate_pipeline is not None:
                from tinygrad.codegen.opt.kernel_pipeline import pipeline_policy_from_candidate
                try: candidate_policy = pipeline_policy_from_candidate(candidate_pipeline)
                except (TypeError, ValueError) as exc: raise KernelOptError(str(exc)) from exc
                if candidate_policy.storage_kind != "lds":
                  coverage = getattr(candidate_pipeline, "wait_coverage", None)
                  if coverage is None or not coverage.passed:
                    raise KernelOptError("register-resident candidate lacks proven wait dependency coverage")
              register_mode = candidate_policy is not None and candidate_policy.storage_kind == "global_register_resident"
              candidate_lds_id = _candidate_lds_buffer_id(self) if not register_mode else None
              allocation = None
              if not register_mode:
                allocation = UOp.placeholder((candidate_geometry.lds_windows[-1].end//2,), dtypes.half, candidate_lds_id,
                                               addrspace=AddrSpace.LOCAL).replace(tag=("kernel_tile_lds", candidate_geometry))
              if candidate_pipeline is not None and not register_mode:
                allocation = UOp.placeholder((candidate_pipeline.active_lds_bytes//2,), dtypes.half, candidate_lds_id,
                                               addrspace=AddrSpace.LOCAL).replace(tag=("kernel_tile_lds", candidate_geometry, candidate_pipeline))
              operand_a=PrecontractOperandTemplate("A",in0,original_axes[1],original_axes[2],outer_m*candidate_geometry.tile[0])
              packed_weight=getattr(self.ast.arg.candidate_context,"packed_weight",None)
              if packed_weight is None:
                operand_b=PrecontractOperandTemplate("B",in1,original_axes[0],original_axes[2],outer_n*candidate_geometry.tile[1])
              else:
                if register_mode: raise KernelOptError("packed-weight candidate requires LDS tile storage")
                if (original_axes[0].vmax+1,original_axes[2].vmax+1) != (packed_weight.rows,packed_weight.k):
                  raise KernelOptError("packed-weight candidate row/K ownership does not match admitted transform")
                packed_params=[u for u in in1.toposort() if u.op is Ops.PARAM and isinstance(u.dtype,PtrDType) and
                               u.ptrdtype.base == packed_weight.storage_dtype]
                if len(packed_params) != 1:
                  raise KernelOptError(f"packed-weight B carrier must reach exactly one canonical packed PARAM, found {len(packed_params)}")
                if getattr(packed_params[0].arg, "slot", packed_params[0].arg) != 2:
                  raise KernelOptError(f"packed-weight B carrier must own ABI slot 2, got PARAM {packed_params[0].arg!r}")
                if any(u.op is Ops.PARAM and isinstance(u.dtype,PtrDType) and u.ptrdtype.base == dtypes.half
                       for u in in1.toposort()):
                  raise KernelOptError("packed-weight B carrier unexpectedly reaches a dense fp16 PARAM")
                operand_b=PackedPrecontractOperandTemplate("B",packed_params[0],packed_weight,original_axes[0],
                  original_axes[2],outer_n*candidate_geometry.tile[1])
              operands=(operand_a,operand_b)
              thread_axes=PrecontractThreadAxes(wave_m,wave_n,lane)
              pipeline_tc_uop = None
              if register_mode:
                # A register-resident candidate is a direct global/L2 -> WMMA lowering. Preserve the ordinary tensor-core
                # operand and output-axis contracts here: the staged register graph stores all eight output subtiles in a
                # flat vec64 accumulator and consequently loses their M/N ownership before the epilogue. It also models
                # A and B row fragments as K substeps, which feeds A/A and B/B pairs to WMMA. The ordinary CONTRACT inputs
                # already lower to VGPR fragments without LDS and retain the exact TC lane/subtile mapping.
                wmma_srcs = [
                  UOp(Ops.CONTRACT, dtype=srcs[0].dtype.vec(tc.elements_per_thread[0]), src=(srcs[0],), arg=tc_upcast_axes[0], tag=1),
                  UOp(Ops.CONTRACT, dtype=srcs[1].dtype.vec(tc.elements_per_thread[1]), src=(srcs[1],), arg=tc_upcast_axes[1], tag=1),
                ]
              if candidate_pipeline is not None and not register_mode:
                from tinygrad.codegen.opt.kernel_lds import PrecontractPipelineTemplate
                from tinygrad.codegen.opt.kernel_pipeline import (KernelStage1FragmentStage, KernelStage1ProducerStage,
                  Stage1StorageAdapter, build_stage1_uop_graph_with_storage, prove_stage1_uop_graph,
                  storage_policy_from_stage1)
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
                if len(c_axes) != 3: raise KernelOptError("buffer2 accumulator contract does not have three binary axes")
                c_elem=(c_axes[0]*2+c_axes[1])*2+c_axes[2]
                accumulator_owners=[(sm*factors.subtiles_n+sn)*8+elem for sm in range(factors.subtiles_m)
                  for sn in range(factors.subtiles_n) for elem in range(8)]
                if len(accumulator_owners) != 64 or set(accumulator_owners) != set(range(64)):
                  raise KernelOptError("buffer2 accumulator ownership must be an exact unique cover of [0, 64)")
                class _StageCallbacks:
                  producer = staticmethod(_produce)
                  fragments = staticmethod(_fragments)
                storage_adapter = Stage1StorageAdapter(_StageCallbacks(), storage_policy_from_stage1(candidate_pipeline))
                pipeline_plan = candidate_pipeline
                graph=build_stage1_uop_graph_with_storage(storage_adapter, pipeline_plan, outer_k.vmax+1, _wmma, subtile_count=1,
                  accumulator_elements=factors.subtiles_m*factors.subtiles_n*8,
                  accumulator_offset=(subtile_m*factors.subtiles_n+subtile_n)*8,
                  accumulator_contract=(c_elem,tc_upcast_axes[2]),body_range_id=next(self.opt_range),accumulator_id=next(self.opt_range))
                proof=prove_stage1_uop_graph(graph)
                if not proof.passed: raise KernelOptError("buffer2 lifecycle UOp proof failed: "+"; ".join(proof.errors))
                pipeline_tc_uop=UOp(Ops.UNROLL,tc.dtype_out,(graph.drain[0],),arg=tc_upcast_axes[2],tag=1)
              elif not register_mode:
                stage = build_precontract_lds_stage(candidate_geometry, tc=tc, allocation=allocation, operands=operands,
                  threads=thread_axes,k_axis=PrecontractKAxis(outer_k,k_substep,outer_k*candidate_geometry.tile[2],k_substep),
                  subtile_m=subtile_m,subtile_n=subtile_n,contracts=tuple(contracts),pipeline_plan=None)
                wmma_srcs = [stage.fragment_a, stage.fragment_b]
            else:
              wmma_srcs = [
                UOp(Ops.CONTRACT, dtype=srcs[0].dtype.vec(tc.elements_per_thread[0]), src=(srcs[0],), arg=tc_upcast_axes[0], tag=1),
                UOp(Ops.CONTRACT, dtype=srcs[1].dtype.vec(tc.elements_per_thread[1]), src=(srcs[1],), arg=tc_upcast_axes[1], tag=1),
              ]
            if candidate_axes is not None and pipeline_tc_uop is not None: tc_uop = pipeline_tc_uop
            else:
              wmma = UOp(Ops.WMMA, dtype=tc.dtype_out.vec(tc.elements_per_thread[2]), src=(
                wmma_srcs[0], wmma_srcs[1], UOp.const(tc.dtype_out.vec(tc.elements_per_thread[2]), 0.0)), arg=wmma_arg, tag=1)
              tc_uop = UOp(Ops.UNROLL, tc.dtype_out, (wmma,), arg=tc_upcast_axes[2], tag=1)

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

def apply_opts(ast:UOp, ren:Renderer) -> UOp:
  if ast.tag is not None: return ast
  k = Scheduler(ast, ren)
  k.convert_loop_to_global()
  if ast.arg is not None and ast.arg.opts_to_apply is not None:
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
  return k.get_optimized_ast(name_override=ast.arg.name if ast.arg is not None and ast.arg.name != "test" else None)
