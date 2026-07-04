from typing import cast
from dataclasses import replace
import itertools
from tinygrad.helpers import DISABLE_FAST_IDIV, TRANSCENDENTAL, SPEC, DEBUG, VIZ, IMAGE, NOOPT, EMULATED_DTYPES, NOLOCALS, USE_TC, getenv
from tinygrad.helpers import ALLOW_TF32, TracingKey, Context, panic
from tinygrad.uop.ops import PatternMatcher, graph_rewrite, UOp, pm_lower_index_dtype, pm_unbind, Ops, UPat, track_rewrites, KernelInfo, ProgramInfo, GroupOp
from tinygrad.uop.ops import ParamArg
from tinygrad.uop.render import pyrender
from tinygrad.uop.spec import type_verify, spec_tensor, spec_program
from tinygrad.renderer import Renderer, Estimates
from tinygrad.renderer.isa import ISARenderer, IselContext, PreRegAllocContext
from tinygrad.dtype import dtypes, PtrDType, ImageDType, AddrSpace

# import all pattern matchers here
from tinygrad.codegen.gpudims import pm_add_gpudims
from tinygrad.uop.symbolic import sym, symbolic_simple, gep_pushing, symbolic, pm_move_where_on_load, pm_clean_up_group_sink
from tinygrad.uop.decompositions import get_late_rewrite_patterns, get_transcendental_patterns, pm_dtype_decomps
from tinygrad.codegen.late.expander import expander, pm_pre_expander, pm_group_for_reduce
from tinygrad.codegen.late.devectorizer import load_store_folding, load_store_indexing, devectorize_buf_and_index, devectorize_alu, pm_reduce, \
  ReduceContext, correct_load_store, pm_render, pm_add_loads, pm_make_images, pm_reduce_acc_upcast_fix, pm_distinct_reg_store_devec
from tinygrad.codegen.opt.postrange import apply_opts
from tinygrad.codegen import experimental as cg_extras
from tinygrad.codegen.late.gater import pm_move_gates_from_index
from tinygrad.codegen.simplify import pm_simplify_ranges, pm_flatten_range, pm_split_ranges, pm_load_collapse
from tinygrad.schedule.rangeify import pm_add_buffers_local, rangeify_codegen, pm_mops, pm_syntactic_sugar, pm_store_ranges
from tinygrad.codegen.late.linearizer import CFGContext, pm_split_ends, pm_add_control_flow, linearize
from tinygrad.codegen.late.regalloc import LinearScanRegallocContext, pm_regalloc_rewrite

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
  if (_kb:=getenv("DECODE_OUTER_B_SPLIT")) > 1 and ren.target.device == "AMD":
    # outer-b independent split-combine primitive: split the serial block loop into K independent LDS-staged
    # online-softmax partitions + flash combine (default-off, declines unrecognized structure). See
    # extra/qk/codegen_outer_b_lds_split.py + docs/decode-attention-outer-b-lds-split-combine-scope-20260627.md.
    ast = cg_extras.outer_b_split(ast, _kb)
  if SPEC: type_verify(ast, spec_tensor)

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

    # do postrange optimization: FutureSight/BubbleBeam opts_to_apply, warm-start, or hand_coded_optimizations
    sink = apply_opts(sink, ren)

  # ** expander (expand_rewrite) **
  sink = graph_rewrite(sink, sym+pm_move_where_on_load, name="postopt symbolic")

  # opt-in (COALESCED_LOAD_LOWERING): predicate-driven promotion of unit-stride load axes to UPCAST so the
  # existing expander+devectorizer vectorize the load (codegen realization of the layout-IR OptOps.COALESCE).
  # Pairs with REG_STORE_DEVEC (fired below) to keep accumulator stores scalar. See
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
  # remove reduce
  sink = graph_rewrite(sink, pm_reduce+gep_pushing, ctx=ReduceContext(), name="remove_reduce")

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
  sink = graph_rewrite(sink, sym+devectorize_alu+devectorize_buf_and_index+load_store_folding+correct_load_store+load_store_indexing,
                       ctx=ren, name="devectorize")
  if ren.target.device == "AMD":
    sink = graph_rewrite(sink, pm_distinct_reg_store_devec, name="distinct reg store devec")
  if (getenv("REG_STORE_DEVEC") or getenv("COALESCED_LOAD_LOWERING")) and ren.target.device == "AMD":
    sink = graph_rewrite(sink, cg_extras.reg_store_devec_pm(), name="reg store devec")
  if getenv("V_DOT2_LOWERING") and ren.target.device == "AMD":
    sink = graph_rewrite(sink, cg_extras.fdot2_pm(), name="fdot2 lowering")

  # lower the index dtype to a concrete int
  sink = graph_rewrite(sink, pm_lower_index_dtype+load_store_indexing+gep_pushing, name="lower all index dtypes")
  sink = graph_rewrite(sink, symbolic, name="post index symbolic")

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

  if getenv("CODEGEN_UNBIND_BEFORE_RENDER", 0):
    sink = graph_rewrite(sink, pm_unbind, ctx={}, name="unbind runtime vars before program render")

  # this was the linearizer
  sink = graph_rewrite(sink, pm_add_control_flow, ctx=CFGContext(sink), name="add control flow", bottom_up=True)

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
    ret: tuple[UOp, list[UOp]] = cast(tuple[UOp, list[UOp]]|None, pm.rewrite(nu, ctx)) or (nu, [nu])
    replaced[u] = ret[0]
    newlst.extend(ret[1])
  return newlst

def do_linearize(ctx:Renderer, prg:UOp, sink:UOp) -> UOp:
  if DEBUG >= 3 and sink.arg.applied_opts: print(f"{sink.arg.function_name:<25} opts: {sink.arg.applied_opts}")
  lst = linearize(sink)
  if getenv("V_DOT2_LOWERING") and ctx.target.device == "AMD":
    lst = cg_extras.line_lower_fdot2(lst)
  lst = line_rewrite(lst, pm_linearize_cleanups)
  # isa renderers need to allocate registers
  if isinstance(ctx, ISARenderer):
    if ctx.pre_regalloc_matcher is not None: lst = line_rewrite(lst, ctx.pre_regalloc_matcher, PreRegAllocContext())
    regalloc_ctx = LinearScanRegallocContext(lst, ctx)
    lst = line_rewrite(lst, pm_regalloc_rewrite, regalloc_ctx)
    lst = line_rewrite(lst, ctx.post_regalloc_matcher, regalloc_ctx)
    if DEBUG >= 4: print(ctx.asm_str(lst, sink.arg.function_name))
  return prg.replace(src=prg.src + (UOp(Ops.LINEAR, src=tuple(lst)),))

def do_estimates(prg:UOp, sink:UOp, lin:UOp) -> UOp|None:
  if sink.arg.estimates is not None: return None
  return prg.replace(src=(sink.replace(arg=replace(sink.arg, estimates=Estimates.from_uops(lin.src, ignore_indexing=True))),)+prg.src[1:])

def do_assemble(ctx:Renderer, prg:UOp, lin:UOp) -> UOp:
  src = "\n".join(str(u.arg) for u in lin.src)
  if DEBUG >= 4: print(src)
  binary = ctx.asm(prg, lin)
  return prg.replace(src=prg.src[:3]+(UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)))

def do_render(ctx:Renderer, prg:UOp, lin:UOp) -> UOp:
  src = ctx.render(list(lin.src))
  new_arg = replace(prg.arg, aux=tuple(ctx.aux(list(lin.src)))) if ctx.has_aux else prg.arg
  return prg.replace(src=prg.src + (UOp(Ops.SOURCE, arg=src),), arg=new_arg)

def do_compile(ctx:Renderer, prg:UOp, source:UOp) -> UOp|None:
  if DEBUG >= 4: print(source.arg)
  lib = ctx.compiler.compile_cached(source.arg)
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
    prog_info = ProgramInfo.from_sink(full_sink)
    # instruction selection
    if isinstance(renderer, ISARenderer):
      full_sink = graph_rewrite(full_sink, renderer.pre_isel_matcher, ctx=itertools.count(-1, -1), name="pre instruction selection", bottom_up=True)
      full_sink = graph_rewrite(full_sink, renderer.isel_matcher, ctx=IselContext(full_sink), name="instruction selection", bottom_up=True)
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
  key = (ast.key, type(renderer), renderer.target, *[x.value for x in config], getenv("WARP_REDUCE_LOWERING"), getenv("V_DOT2_LOWERING"), getenv("REG_STORE_DEVEC"), getenv("SCHED_UNROLL"), getenv("SCHED_LIST"), getenv("COALESCED_LOAD_LOWERING"), getenv("DECODE_FAST_EXP2"), getenv("DECODE_OUTER_B_SPLIT"))
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
