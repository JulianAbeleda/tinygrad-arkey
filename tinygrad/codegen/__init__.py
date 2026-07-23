from typing import cast
import inspect
from dataclasses import replace
import itertools
from tinygrad.helpers import DISABLE_FAST_IDIV, TRANSCENDENTAL, SPEC, DEBUG, VIZ, IMAGE, NOOPT, EMULATED_DTYPES, NOLOCALS, USE_TC, getenv
from tinygrad.helpers import ALLOW_TF32, TracingKey, Context, panic
from tinygrad.uop.ops import PatternMatcher, graph_rewrite, UOp, pm_lower_index_dtype, Ops, UPat, track_rewrites, KernelInfo, ProgramInfo, GroupOp
from tinygrad.uop.ops import AttentionWMMARole, WMMARoleLedger, FinalLinearMetadata, get_attention_wmma_role, set_attention_wmma_role
from tinygrad.uop.ops import ParamArg
from tinygrad.uop.render import pyrender
from tinygrad.uop.spec import type_verify, spec_tensor, spec_program
from tinygrad.renderer import Renderer, Estimates
from tinygrad.renderer.isa import CompilerCaptureProof, ISARenderer, IselContext, PreRegAllocContext
from tinygrad.dtype import dtypes, PtrDType, ImageDType, AddrSpace

# import all pattern matchers here
from tinygrad.codegen.gpudims import pm_add_gpudims
from tinygrad.uop.symbolic import sym, symbolic_simple, gep_pushing, symbolic, pm_move_where_on_load, pm_clean_up_group_sink
from tinygrad.uop.decompositions import get_late_rewrite_patterns, get_transcendental_patterns, pm_dtype_decomps
from tinygrad.codegen.late.expander import expander, pm_pre_expander, pm_group_for_reduce
from tinygrad.codegen.late.devectorizer import load_store_folding, load_store_indexing, devectorize_buf_and_index, devectorize_alu, pm_reduce, \
  ReduceContext, correct_load_store, pm_render, pm_add_loads, pm_make_images, pm_reduce_acc_upcast_fix, pm_distinct_reg_store_devec, pm_group_wmma_reg_store
from tinygrad.codegen.opt.postrange import apply_opts
from tinygrad.codegen import experimental as cg_extras
from tinygrad.codegen.late.gater import pm_move_gates_from_index
from tinygrad.codegen.simplify import pm_simplify_ranges, pm_flatten_range, pm_split_ranges, pm_load_collapse
from tinygrad.schedule.rangeify import pm_add_buffers_local, rangeify_codegen, pm_mops, pm_syntactic_sugar, pm_store_ranges, pm_native_row_softmax_repack
from tinygrad.codegen.late.linearizer import CFGContext, pm_split_ends, pm_add_control_flow, linearize
from tinygrad.codegen.late.regalloc import LinearScanRegallocContext, pm_regalloc_rewrite, pressure_schedule

# Register carriers must not re-run the generic GEP-pushing fixed point after
# register-index expansion: side-effecting store GEPs become GROUPs under a
# void UNROLL, and vector carriers can become nested STACKs. Keep scalar folds
# and flatten only scalar VCAT children for this route.
def _register_pipe_vcat(x):
  return UOp(Ops.STACK, x.dtype, x.src) if len(x.src) == x.dtype.count and all(y.dtype.count == 1 for y in x.src) else None

register_pipe_symbolic = symbolic_simple + PatternMatcher([
  (UPat(Ops.VCAT, name="x"), _register_pipe_vcat),
  # The generic symbolic pass normally removes these late weak-index casts;
  # register carriers skip GEP pushing, so normalize concrete loop and launch
  # index sources here without widening this to arbitrary integer values.
  (UPat(Ops.CAST, dtype=dtypes.weakint, src=(UPat((Ops.RANGE, Ops.SPECIAL), dtype=dtypes.ints),), name="x"), lambda x: x.src[0]),
  # A fully contracted side-effect group carries the same ordering as its
  # enclosing void UNROLL; retain the GROUP directly for program legality.
  (UPat(Ops.UNROLL, dtype=dtypes.void, src=(UPat(Ops.GROUP, name="g"),)), lambda g: g),
])

pm_index_is_shrink = PatternMatcher([
  # rewrite non-image INDEX to SHRINK
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("idx"))).cast(name="x"), lambda buf,idx,x:
    UOp(Ops.SHRINK, dtype=buf.dtype.base, src=(buf, idx, UOp.const(dtypes.int, x.dtype.count))) if isinstance(buf.dtype, PtrDType) else None),
  # rewrite GEP to INDEX
  (UPat(Ops.GEP, name="x"), lambda x: x.replace(op=Ops.INDEX, src=x.src+(UOp.const(dtypes.int, x.arg),), arg=None)),
])

pm_remove_vec_dtypes = PatternMatcher([
  # rewrite PARAM to non pointer
  (UPat((Ops.PARAM, Ops.BUFFER, Ops.DEFINE_LOCAL, Ops.DEFINE_REG), name="buf"), lambda buf:
   buf.replace(dtype=buf.dtype.base, src=(UOp.const(dtypes.int, buf.ptrdtype.size),)) \
    if isinstance(buf.dtype, PtrDType) and not isinstance(buf.dtype, ImageDType) else None),
  # no LOADs on register dtypes
  (UPat(Ops.LOAD, name="x"), lambda x: x.src[0] if x.src[0].addrspace == AddrSpace.REG else None),
  # remove all vec dtypes
  (UPat(GroupOp.All-{Ops.PARAM, Ops.BUFFER, Ops.DEFINE_LOCAL, Ops.DEFINE_REG}, name="x"),
   lambda x: x.replace(dtype=x.dtype.base.scalar().base)),
  # replace DEFINE_LOCAL/DEFINE_REG with BUFFER
  (UPat((Ops.DEFINE_LOCAL, Ops.DEFINE_REG), name="x"), lambda x:
   x.replace(op=Ops.BUFFER, arg=ParamArg(x.arg, addrspace=AddrSpace.LOCAL if x.op == Ops.DEFINE_LOCAL else AddrSpace.REG))),
])+pm_clean_up_group_sink

def full_rewrite_to_sink(ast:UOp, ren:Renderer, optimize:bool=True) -> UOp:
  if VIZ: graph_rewrite(ast, PatternMatcher([]), name="View Base AST")
  if DEBUG >= 5: print(pyrender(ast))
  if (_u:=getenv("SCHED_UNROLL")) > 1 and ren.target.device == "AMD":
    # recurrence-aware loop-unroll primitive (default-off codegen scheduling capability)
    ast = cg_extras.unroll_recurrence(ast, _u)
  # Preserve the exact native QK C fragment for its nonlinear consumer before
  # bottom-up movement lowering can install the ordinary logical wrapper.
  ast = graph_rewrite(ast, pm_native_row_softmax_repack, ctx=itertools.count(900),
                      bottom_up=False, name="native row softmax repack before spec")
  if (native_repack_pm:=getattr(ren, "native_repack_matcher", None)) is not None:
    ast = graph_rewrite(ast, native_repack_pm, ctx=itertools.count(800), bottom_up=True,
                        name="expand native row softmax repack")
  # SPEC validation runs before the late devectorizer.  Resolve only the
  # provenance-tagged INDEX views that the expander can place around a
  # composite REDUCE_SLOT; ordinary REDUCE_SLOT/INDEX nodes remain subject to
  # the existing fail-closed spec rules.
  from tinygrad.codegen.late.composite_combines import resolve_composite_reduce_slot_prebufferize
  ast = graph_rewrite(ast, PatternMatcher([
    (UPat(Ops.REDUCE_SLOT, src=(UPat(),), name="slot"), resolve_composite_reduce_slot_prebufferize),
  ]), name="resolve_composite_slots_before_spec")
  if SPEC: type_verify(ast, spec_tensor)
  if (native_state_pm:=getattr(ren, "native_state_lane_matcher", None)) is not None:
    ast = graph_rewrite(ast, native_state_pm, name="lower native state lanes after tensor spec")

  # preprocess
  sink = graph_rewrite(ast, pm_mops+pm_syntactic_sugar+pm_store_ranges, ctx=itertools.count(1000), name="early movement ops", bottom_up=True)

  # first we optimize
  if optimize:
    # collapse loads reduce (indexing by a tensor)
    sink = graph_rewrite(sink, pm_load_collapse, name="load collapse")

    # split ranges
    sink = graph_rewrite(sink, pm_split_ranges+pm_flatten_range, ctx={}, name="split ranges")

    # symbolic (NOTE: this is a requirement for pm_simplify_ranges to be correct)
    sink = graph_rewrite(sink, sym+pm_flatten_range, name="initial symbolic")

    # optimize (schedule) the AST
    sink = graph_rewrite(sink, pm_flatten_range+pm_simplify_ranges, ctx={}, name="simplify ranges")

    # do postrange optimization: explicit opts_to_apply, warm-start, or hand_coded_optimizations
    sink = apply_opts(sink, ren)
    # Scheduler TC integration may introduce a descriptor-owned native
    # repack during apply_opts, after the initial AST hook above.
    if (native_repack_pm:=getattr(ren, "native_repack_matcher", None)) is not None:
      sink = graph_rewrite(sink, native_repack_pm, ctx=itertools.count(700), bottom_up=True,
                           name="expand optimizer native row softmax repack")
    if (native_state_pm:=getattr(ren, "native_state_lane_matcher", None)) is not None:
      sink = graph_rewrite(sink, native_state_pm, name="lower optimizer native state lanes")

  # ** expander (expand_rewrite) **
  sink = graph_rewrite(sink, sym+pm_move_where_on_load, name="postopt symbolic")

  # opt-in (COALESCED_LOAD_LOWERING): predicate-driven promotion of unit-stride load axes to UPCAST so the
  # existing expander+devectorizer vectorize the load (codegen realization of the layout-IR OptOps.COALESCE).
  # The shared register-store lowering keeps accumulator stores scalar. See
  # extra/qk/coalesced_load_lowering.py + docs/decode-coalesced-load-primitive-scope-20260626.md.
  if getenv("COALESCED_LOAD_LOWERING") and ren.target.device == "AMD":
    sink = cg_extras.coalesce_loads(sink)

  # expand
  # opt-in (WARP_REDUCE_LOWERING): auto-lower a full-warp REDUCE to the AMD ds_bpermute cross-lane ladder BEFORE
  # pm_group_for_reduce claims it for the LDS tree. Milestone 5 of the generic-low-level-search goal -- makes the
  # cross-lane reduce primitive scheduler-emittable (today only the hand kernels emit it). See
  # extra/qk/warp_reduce_lowering.py + bench/qk-search-spaces/decode_ffn_gemv_gfx1100_v1.json.
  _expander_pm = sym+pm_pre_expander+pm_group_for_reduce+expander
  if getenv("WARP_REDUCE_LOWERING") and ren.target.device == "AMD":
    _expander_pm = sym+pm_pre_expander+cg_extras.warp_reduce_pm()+pm_group_for_reduce+expander
  sink = graph_rewrite(sink, _expander_pm, name="expander")

  # add locals
  sink = graph_rewrite(sink, pm_add_buffers_local+rangeify_codegen, ctx=itertools.count(0), name="add local buffers")

  # ** devectorizer (full_graph_rewrite) **
  # Handle composite REDUCEs with no ranges (post-expander STACK form)
  from tinygrad.codegen.late.composite_combines import _lower_composite_no_range_pm
  sink = graph_rewrite(sink, PatternMatcher([(UPat(Ops.REDUCE, name="red"), _lower_composite_no_range_pm)]),
                       name="lower_composite_no_range")

  # remove reduce
  had_deferred_reduce_projection = any(u.op is Ops.DEFERRED_REDUCE_SLOT for u in sink.toposort())
  sink = graph_rewrite(sink, pm_reduce+gep_pushing, ctx=ReduceContext(), name="remove_reduce")
  if had_deferred_reduce_projection:
    lane_range = None
    store_subs = {}
    for store in (u for u in sink.toposort() if u.op is Ops.STORE and u.src[0].op is Ops.INDEX):
      unrolls = [u for u in (store.src[-1], *store.src[-1].backward_slice) if u.op is Ops.UNROLL and len(u.arg)]
      if len(unrolls) != 1: continue
      unroll = unrolls[0]
      lanes = unroll.src[0].dtype.count
      idx = store.src[0]
      candidates = [r for r in idx.src[-1].backward_slice if r.op is Ops.RANGE and r.vmin == 0 and r.vmax == lanes-1]
      for r in candidates:
        z, o = UOp.const(r.dtype, 0), UOp.const(r.dtype, 1)
        delta = (idx.src[-1].substitute({r:o}) - idx.src[-1].substitute({r:z})).simplify()
        if delta.vmin == delta.vmax == 1: lane_range = r; break
      if lane_range is None: continue
      lane_indices = UOp.vectorize(*(idx.substitute({lane_range: UOp.const(lane_range.dtype, lane)}) for lane in range(lanes)))
      indexed_unroll = UOp(Ops.UNROLL, idx.dtype, (lane_indices,), unroll.arg)
      store_subs[store] = store.replace(src=(indexed_unroll, *store.src[1:]))
    if store_subs:
      sink = sink.substitute(store_subs)
      sink = graph_rewrite(sink, expander, name="expand deferred reduce output")
      topo = sink.toposort()
      ended_ranges = {r for u in topo if u.op is Ops.END for r in u.src[1:] if r.op is Ops.RANGE}
      users = {r: [] for r in topo if r.op is Ops.RANGE}
      for u in topo:
        for s in u.src:
          if s in users: users[s].append(u)
      stale = {r for r, us in users.items() if r not in ended_ranges and all(u.op is Ops.AFTER for u in us)}
      if stale:
        sink = sink.substitute({u: u.replace(src=tuple(s for s in u.src if s not in stale)) for u in topo if u.op is Ops.AFTER})
  # A composite REDUCE lowered during the pass above can be shared by several
  # REDUCE_SLOT users. Resolve those projections after the structured TUPLE is
  # present, then let dead graph nodes disappear naturally.
  from tinygrad.codegen.late.composite_combines import resolve_reduce_slot_tensor
  slot_subs = {u: resolved for u in sink.toposort() if u.op is Ops.REDUCE_SLOT
               if (resolved:=resolve_reduce_slot_tensor(u)) is not None}
  if slot_subs: sink = sink.substitute(slot_subs)
  if ren.target.device == "AMD":
    sink = graph_rewrite(sink, pm_group_wmma_reg_store, name="group wmma reg ownership")

  # GPU dimension assignment requires scalar global STORE addresses. A
  # deferred projection can already own its Hd lanes as STACK(INDEX...), so
  # lower that backend-neutral store ABI before gpudims inspects destinations.
  if had_deferred_reduce_projection:
    output_store_subs = {}
    for store in (u for u in sink.toposort() if u.op is Ops.STORE and u.src[0].op is Ops.STACK):
      targets, value = store.src[0].src, store.src[1]
      if not targets or not all(x.op is Ops.INDEX for x in targets) or value.dtype.count != len(targets): continue
      output_store_subs[store] = UOp.group(*(target.store(value.gep(i)) for i,target in enumerate(targets)))
    if output_store_subs: sink = sink.substitute(output_store_subs)

  # add gpu dims (late). this works after devectorize, but it's faster here
  sink = graph_rewrite(sink, pm_add_gpudims, ctx=ren, name="add gpudims")

  # AMD baseline: give manual END/AFTER scalar-REG accumulators the same widen+horizontal-reduce treatment Ops.REDUCE
  # gets, so an UPCAST/UNROLL'd reduce body no longer broadcasts the scalar slot into an unassignable
  # make_floatN(acc,...) store. Exact + fail-closed; runs before add_loads to match reduce_to_acc's form.
  # See tinygrad/codegen/late/devectorizer.py reduce_acc_upcast_fix + docs/tg-p12-*.
  if ren.target.device == "AMD":
    sink = graph_rewrite(sink, pm_reduce_acc_upcast_fix, name="reduce acc upcast fix")

  # **** optimizations are done, now we lower to actual code ****

  # add loads
  sink = graph_rewrite(sink, pm_add_loads, name="** add loads (code)")

  # create image buffers
  if IMAGE and ren.target.device in {"QCOM", "CL", "PYTHON", "NULL"}:
    sink = graph_rewrite(sink, pm_make_images, name="create image buffers", bottom_up=True, ctx=ren.target.arch)

  # devectorize
  has_register_pipe = any(isinstance(u.tag, tuple) and u.tag[:1] == ("register_pipe_stage_buffer",)
                          for u in sink.toposort())
  devectorize_symbolic = register_pipe_symbolic if has_register_pipe else sym
  devectorize_folding = PatternMatcher([]) if has_register_pipe else load_store_folding
  sink = graph_rewrite(sink, devectorize_symbolic+devectorize_alu+devectorize_buf_and_index+devectorize_folding+correct_load_store+load_store_indexing,
                       ctx=ren, name="devectorize")
  if had_deferred_reduce_projection:
    # add_loads may interpret the expanded output INDEX lanes as values and
    # produce STORE(STACK(LOAD(INDEX)...), values). Restore the original sink
    # ownership: projection lanes are values, output lanes remain addresses.
    output_store_subs = {}
    for store in (u for u in sink.toposort() if u.op is Ops.STORE and u.src[0].op is Ops.STACK):
      if not store.src[0].src or not all(x.op is Ops.LOAD and x.src[0].op is Ops.INDEX for x in store.src[0].src): continue
      addresses = tuple(x.src[0] for x in store.src[0].src)
      output_store_subs[store] = store.replace(src=(store.src[0].replace(src=addresses), *store.src[1:]))
    if output_store_subs: sink = sink.substitute(output_store_subs)
  if ren.target.device == "AMD":
    sink = graph_rewrite(sink, pm_distinct_reg_store_devec, name="distinct reg store devec")
  if getenv("COALESCED_LOAD_LOWERING") and ren.target.device == "AMD":
    sink = graph_rewrite(sink, cg_extras.reg_store_devec_pm(), name="reg store devec")
  if getenv("V_DOT2_LOWERING") and ren.target.device == "AMD":
    sink = graph_rewrite(sink, cg_extras.fdot2_pm(), name="fdot2 lowering")

  # lower the index dtype to a concrete int
  sink = graph_rewrite(sink, pm_lower_index_dtype+load_store_indexing+(PatternMatcher([]) if has_register_pipe else gep_pushing), name="lower all index dtypes")
  sink = graph_rewrite(sink, register_pipe_symbolic if has_register_pipe else symbolic, name="post index symbolic")
  if (loop_fragment_pm:=getattr(ren,"native_loop_fragment_matcher",None)) is not None:
    sink=graph_rewrite(sink,loop_fragment_pm,name="expand native loop fragments after index lowering")

  # optional pre matcher
  if ren.pre_matcher is not None: sink = graph_rewrite(sink, ren.pre_matcher, name="pre_matcher")

  # decompositions
  supported_ops = tuple(ren.code_for_op.keys())
  pm_decomp = symbolic_simple+get_late_rewrite_patterns(supported_ops, bool(DISABLE_FAST_IDIV))
  pm_transcendental = symbolic_simple+get_transcendental_patterns(supported_ops, TRANSCENDENTAL>=2)
  sink = graph_rewrite(sink, pm_decomp, ctx=ren, name="decompositions")
  sink = graph_rewrite(sink, pm_dtype_decomps, ctx=(set(), ren), name="decomp dtypes")
  sink = graph_rewrite(sink, pm_transcendental, name="transcendental")

  # move gates from unrenderable INVALID where
  sink = graph_rewrite(sink, pm_move_gates_from_index, name="move gates from index")

  # final rules for the renderer (without sym)
  extra_matcher = ren.extra_matcher if ren.extra_matcher is not None else PatternMatcher([])
  pm_final_rewrite = pm_decomp+pm_render+extra_matcher+pm_split_ends
  sink = graph_rewrite(sink, pm_final_rewrite, ctx=ren, name="final rewrite")
  if getenv("V_DOT2_LOWERING") and ren.target.device == "AMD":
    sink = graph_rewrite(sink, cg_extras.fdot2_pm(), name="fdot2 final lowering")

  if ren.new_style:
    sink = graph_rewrite(sink, pm_index_is_shrink, name="index is shrink")
    sink = graph_rewrite(sink, pm_remove_vec_dtypes, name="transform to new style")

  # this was the linearizer
  sink = graph_rewrite(sink, pm_add_control_flow, ctx=CFGContext(sink), name="add control flow", bottom_up=True)
  if ren.target.device == "AMD":
    sink = graph_rewrite(sink, pm_distinct_reg_store_devec, name="post control-flow stack store devec")

  if VIZ: graph_rewrite(sink, PatternMatcher([]), name="View Output AST")
  if SPEC: type_verify(sink, spec_program)

  # return the rewritten sink
  return sink

# inject IF/ENDIF. only needed if device doesn't support gated stores
pm_linearize_cleanups = PatternMatcher([
  # if statements are not allowed in the graph
  (UPat((Ops.IF, Ops.ENDIF)), lambda: panic(RuntimeError, "if not allowed in graph")),
  # gated STORE becomes IF-STORE-ENDIF. this is the only use of IF-ENDIF
  (UPat(Ops.STORE, name="u", src=(UPat((Ops.INDEX, Ops.SHRINK)).or_casted(), UPat(), UPat(name="gate", dtype=dtypes.bool))),
   lambda u, gate: ((st:=u.replace(src=u.src[0:2])), [mif:=UOp(Ops.IF, src=(gate, u.src[0])), st, UOp(Ops.ENDIF, src=(mif,))]))
])

# requires lst be toposorted. like graph rewrite, but for lines
def line_rewrite(lst:list[UOp], pm:PatternMatcher, ctx=None) -> list[UOp]:
  newlst = []
  replaced: dict[UOp, UOp] = {}
  for u in lst:
    nu = u.replace(src=tuple([replaced.get(x, x) for x in u.src]))
    if (role:=get_attention_wmma_role(u)) is not None: set_attention_wmma_role(nu, role)
    ret: tuple[UOp, list[UOp]] = cast(tuple[UOp, list[UOp]]|None, pm.rewrite(nu, ctx)) or (nu, [nu])
    if role is not None:
      if len(ret[1]) == 1:
        for node in tuple(dict.fromkeys((ret[0], ret[1][0]))): set_attention_wmma_role(node, role)
      else:
        machine = [node for node in ret[1] if "wmma" in type(node.arg).__name__.lower() or
                   "wmma" in str(getattr(node.arg, "op", "")).lower()]
        if len(machine) != 1: raise RuntimeError("attention WMMA role lost across one-to-many line replacement")
        set_attention_wmma_role(machine[0], role)
    replaced[u] = ret[0]
    newlst.extend(ret[1])
  return newlst

# The final native rewrite may intern equal physical instructions.  Preserve roles by
# LINEAR occurrence here instead of trying to attach distinct roles to one UOp object.
def line_rewrite_wmma_ledger(lst:list[UOp], pm:PatternMatcher, ctx=None) -> tuple[list[UOp], WMMARoleLedger]:
  newlst: list[UOp] = []
  replaced: dict[UOp, UOp] = {}
  sites: list[tuple[int, AttentionWMMARole]] = []
  for u in lst:
    nu = u.replace(src=tuple(replaced.get(x, x) for x in u.src))
    ret: tuple[UOp, list[UOp]] = cast(tuple[UOp, list[UOp]]|None, pm.rewrite(nu, ctx)) or (nu, [nu])
    if (role:=get_attention_wmma_role(u)) is not None:
      machine = [i for i,node in enumerate(ret[1]) if "wmma" in type(node.arg).__name__.lower() or
                 "wmma" in str(getattr(node.arg, "op", "")).lower()]
      if len(machine) != 1: raise RuntimeError("attention WMMA role lost across final line replacement")
      sites.append((len(newlst)+machine[0], role))
    replaced[u] = ret[0]
    newlst.extend(ret[1])
  return newlst, WMMARoleLedger(tuple(sites)).validate()

def do_linearize(ctx:Renderer, prg:UOp, sink:UOp) -> UOp:
  if DEBUG >= 3 and sink.arg.applied_opts: print(f"{sink.arg.function_name:<25} opts: {sink.arg.applied_opts}")
  lst = linearize(sink)
  expected_roles = prg.arg.wmma_role_expectation if isinstance(prg.arg, ProgramInfo) else ()
  if getenv("V_DOT2_LOWERING") and ctx.target.device == "AMD":
    lst = cg_extras.line_lower_fdot2(lst)
  lst = line_rewrite(lst, pm_linearize_cleanups)
  # isa renderers need to allocate registers
  selection_proof = sink.tag if isinstance(sink.tag, CompilerCaptureProof) else None
  final_regalloc_proof = None
  native_ledger = None
  if isinstance(ctx, ISARenderer):
    # Order compiler-owned reusable register leases while their structural
    # dependencies are still present.  Backend pre-allocation cleanup may then
    # erase zero-code order operands without erasing the resulting lifetimes.
    lst = pressure_schedule(lst)
    if ctx.pre_regalloc_matcher is not None: lst = line_rewrite(lst, ctx.pre_regalloc_matcher, PreRegAllocContext())
    regalloc_ctx = LinearScanRegallocContext(lst, ctx)
    lst = line_rewrite(lst, pm_regalloc_rewrite, regalloc_ctx)
    lst, native_ledger = line_rewrite_wmma_ledger(lst, ctx.post_regalloc_matcher, regalloc_ctx)
    if selection_proof is not None and not regalloc_ctx.spills and regalloc_ctx.stack_size == 0:
      final_regalloc_proof = selection_proof.finalize_zero_spill()
    if DEBUG >= 4: print(ctx.asm_str(lst, sink.arg.function_name))
  ledger = native_ledger if native_ledger is not None else \
    WMMARoleLedger(tuple((i, role) for i,u in enumerate(lst) if (role:=get_attention_wmma_role(u)) is not None)).validate()
  sites = ledger.sites
  if sorted(expected_roles) != sorted(role for _,role in sites):
    raise RuntimeError(f"attention WMMA role ledger loss/tamper: expected {expected_roles}, final {tuple(r for _,r in sites)}")
  linear_arg = FinalLinearMetadata(final_regalloc_proof, ledger) if sites else final_regalloc_proof
  new_arg = replace(prg.arg, wmma_roles=ledger) if isinstance(prg.arg, ProgramInfo) else prg.arg
  return prg.replace(src=prg.src + (UOp(Ops.LINEAR, src=tuple(lst), arg=linear_arg),), arg=new_arg)

def do_estimates(prg:UOp, sink:UOp, lin:UOp) -> UOp|None:
  if sink.arg.estimates is not None: return None
  return prg.replace(src=(sink.replace(arg=replace(sink.arg, estimates=Estimates.from_uops(lin.src, ignore_indexing=True))),)+prg.src[1:])

class _CompileCaptureAttachment:
  """Hashable carrier for a backend-owned, opaque compile-only record."""
  __slots__ = ("record",)
  def __init__(self, record): self.record = record

def do_assemble(ctx:Renderer, prg:UOp, lin:UOp) -> UOp:
  src = "\n".join(str(u.arg) for u in lin.src)
  if DEBUG >= 4: print(src)
  binary = ctx.asm(prg, lin)
  # ISA renderers may expose a pure, compile-only final-artifact hook.  Keep
  # this duck-typed: the generic codegen path must not import a backend's
  # capture schema, create a runtime program, or make capture mandatory.
  # The hook runs after asm(), so its record can join the exact source/binary
  # with renderer-owned final descriptor/regalloc/disassembly facts.
  capture = getattr(ctx, "compile_capture", None) if isinstance(ctx, ISARenderer) else None
  if callable(capture):
    # New capture hooks may receive the proof directly.  Inspecting the bound
    # signature preserves the established three-argument hook ABI, including
    # hooks which intentionally reject extra backend metadata.
    try:
      accepts_proof = len(inspect.signature(capture).parameters) >= 4
    except (TypeError, ValueError):
      accepts_proof = False
    proof = lin.arg.regalloc_proof if isinstance(lin.arg, FinalLinearMetadata) else lin.arg
    record = capture(prg, lin, binary, proof) if accepts_proof else capture(prg, lin, binary)
  else: record = None
  new_arg = prg.arg
  if record is not None and hasattr(new_arg, "aux"):
    # ProgramInfo participates in UOp interning and therefore must remain
    # hashable even when the backend record is a dict-like payload.
    new_arg = replace(new_arg, aux=new_arg.aux + (_CompileCaptureAttachment(record),))
  return prg.replace(src=prg.src[:3]+(UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)), arg=new_arg)

def do_render(ctx:Renderer, prg:UOp, lin:UOp) -> UOp:
  src = ctx.render(list(lin.src))
  new_arg = replace(prg.arg, aux=tuple(ctx.aux(list(lin.src)))) if ctx.has_aux else prg.arg
  return prg.replace(src=prg.src + (UOp(Ops.SOURCE, arg=src),), arg=new_arg)

def do_compile(ctx:Renderer, prg:UOp, source:UOp) -> UOp|None:
  if DEBUG >= 4: print(source.arg)
  candidate = prg.src[0].arg.candidate_context
  cache_context = None if candidate is None else (candidate.schema_version, candidate.canonical_identity)
  lib = ctx.compiler.compile_cached(source.arg, cache_context=cache_context)
  if DEBUG >= 7: ctx.compiler.disassemble(lib)
  return prg.replace(src=prg.src + (UOp(Ops.BINARY, arg=lib),))

pm_to_program = PatternMatcher([
  (UPat(Ops.PROGRAM, src=(UPat(Ops.SINK, name="sink"), UPat(Ops.DEVICE)), name="prg"), do_linearize),
  (UPat(Ops.PROGRAM, src=(UPat(Ops.SINK, name="sink"), UPat(Ops.DEVICE), UPat(Ops.LINEAR, name="lin")), name="prg"), do_estimates),
  (UPat(Ops.PROGRAM, src=(UPat(), UPat(Ops.DEVICE), UPat(Ops.LINEAR, src=UPat(Ops.INS), name="lin")), name="prg"), do_assemble),
  (UPat(Ops.PROGRAM, src=(UPat(), UPat(Ops.DEVICE), UPat(Ops.LINEAR, name="lin")), name="prg"), do_render),
  (UPat(Ops.PROGRAM, src=(UPat(), UPat(Ops.DEVICE), UPat(Ops.LINEAR), UPat(Ops.SOURCE, name="source")), name="prg"), do_compile),
])

@track_rewrites(name=lambda ast,renderer,ret,**kwargs: TracingKey(ret.src[0].arg.name,(ret.src[0].arg.function_name, ast), ret=renderer), replay=True)
@Context(ALLOW_DEVICE_USAGE=0)
def do_to_program(ast:UOp, renderer:Renderer) -> UOp:
  """
  Transform an AST into a compiled PROGRAM.

  Args:
    ast: The Ops.SINK/Ops.PROGRAM rooted AST
    renderer: The renderer used to generate the code

  Returns:
    The Ops.PROGRAM with SINK/DEVICE/LINEAR/SOURCE/BINARY.
  """
  if ast.op is Ops.PROGRAM: prg = ast
  elif ast.op is Ops.SINK:
    assert isinstance(ast.arg, KernelInfo), "requires KernelInfo on arg to to_program"
    full_sink = full_rewrite_to_sink(ast, renderer, optimize=ast.tag is None)
    declared_roles = []
    for u in full_sink.toposort():
      if u.op is not Ops.WMMA: continue
      role = u.tag if isinstance(u.tag, AttentionWMMARole) else \
        AttentionWMMARole(u.tag[1], u.tag[2]) if isinstance(u.tag, tuple) and len(u.tag) == 3 and u.tag[0] == "attention_wmma" else None
      if role is not None: role.validate(); set_attention_wmma_role(u, role); declared_roles.append(role)
    prog_info = replace(ProgramInfo.from_sink(full_sink), wmma_role_expectation=tuple(declared_roles))
    # instruction selection
    if isinstance(renderer, ISARenderer):
      if (loop_state_pm:=getattr(renderer,"native_loop_state_matcher",None)) is not None:
        full_sink=graph_rewrite(full_sink,loop_state_pm,name="lower native attention loop state",bottom_up=True)
      full_sink = graph_rewrite(full_sink, renderer.pre_isel_matcher, ctx=itertools.count(-1, -1), name="pre instruction selection", bottom_up=True)
      if (opaque_pm:=getattr(renderer,"native_fragment_opaque_matcher",None)) is not None:
        full_sink=graph_rewrite(full_sink,opaque_pm,name="opaque native fragments",bottom_up=True)
      isel_ctx = IselContext(full_sink)
      full_sink = graph_rewrite(full_sink, renderer.isel_matcher, ctx=isel_ctx, name="instruction selection", bottom_up=True)
      if renderer.post_isel_matcher is not None:
        full_sink = graph_rewrite(full_sink, renderer.post_isel_matcher, ctx=isel_ctx, name="post instruction selection", bottom_up=True)
      if declared_roles and (bind_roles:=getattr(renderer, "bind_attention_wmma_roles", None)) is not None:
        bind_roles(isel_ctx, full_sink, tuple(declared_roles))
      if (capture_proof := renderer.capture_selection_proof(isel_ctx)) is not None:
        full_sink = full_sink.replace(tag=capture_proof)
    prg = UOp(Ops.PROGRAM, src=(full_sink, UOp(Ops.DEVICE, arg=renderer.target.device)), arg=prog_info)
  else: raise RuntimeError(f"can't call to_program on {ast.op}")
  if not isinstance(prg.arg, ProgramInfo): prg = prg.replace(arg=ProgramInfo.from_sink(prg.src[0]))
  prg = graph_rewrite(prg, pm_to_program, ctx=renderer, name="linearize/render")
  if VIZ: graph_rewrite(prg, PatternMatcher([]), name="View Program")
  return prg

# LOWER_DISK_CACHE: persist the lowered/rendered Program (to_program's output) across processes. Measurement showed the
# lowering pipeline -- not kernel compilation -- is the startup tax (~10s of a ~13s first token, and it re-runs on WARM
# restart because to_program_cache is per-process). The compiled-binary disk cache (device.py) sits one level below and
# only saves the comgr subprocess (~0.5s). The lowered Program pickles round-trip identically (UOp.__reduce__); the
# lowering is cross-process deterministic (proven: 43/43 rendered-source compile-cache hits warm). Fails SAFE: on any
# unpickle error or table-fingerprint mismatch (codegen source changed) it recomputes. Correctness authority stays the
# generated output (token-identity) -- this cache only skips re-deriving a byte-identical Program.
LOWER_DISK_CACHE = getenv("LOWER_DISK_CACHE", 0)
_LOWER_CACHE_TABLE: str|None = None
def _lower_cache_table() -> str:
  global _LOWER_CACHE_TABLE
  if _LOWER_CACHE_TABLE is None:
    import hashlib, pathlib
    h, here = hashlib.sha256(), pathlib.Path(__file__).resolve().parent
    for rel in ("__init__.py", "../uop/ops.py", "../uop/render.py", "../renderer/cstyle.py", "../renderer/__init__.py"):
      try: h.update((here / rel).read_bytes())
      except Exception: pass
    _LOWER_CACHE_TABLE = "to_program_" + h.hexdigest()[:12]   # codegen fingerprint -> a source change invalidates
  return _LOWER_CACHE_TABLE

to_program_cache: dict[tuple, UOp] = {}
def to_program(ast:UOp, renderer:Renderer) -> UOp:
  config = (NOOPT, EMULATED_DTYPES, NOLOCALS, USE_TC, IMAGE, DISABLE_FAST_IDIV, TRANSCENDENTAL, ALLOW_TF32)
  key = (ast.key, type(renderer), renderer.target, *[x.value for x in config], getenv("WARP_REDUCE_LOWERING"), getenv("V_DOT2_LOWERING"), getenv("SCHED_UNROLL"), getenv("SCHED_LIST"), getenv("COALESCED_LOAD_LOWERING"), getenv("DECODE_FAST_EXP2"))
  if (prg:=to_program_cache.get(key)) is not None: return prg
  _dk = None
  if LOWER_DISK_CACHE:
    import hashlib, pickle
    from tinygrad.helpers import diskcache_get
    _dk = hashlib.sha256(ast.key + repr(key[1:]).encode()).hexdigest()
    if (blob:=diskcache_get(_lower_cache_table(), _dk)) is not None:
      try:
        to_program_cache[key] = prg = pickle.loads(blob); return prg
      except Exception: pass   # fail safe -> recompute below
  to_program_cache[key] = prg = do_to_program(ast, renderer)
  if LOWER_DISK_CACHE and _dk is not None:
    import pickle
    from tinygrad.helpers import diskcache_put
    try: diskcache_put(_lower_cache_table(), _dk, pickle.dumps(prg))
    except Exception: pass
  return prg
