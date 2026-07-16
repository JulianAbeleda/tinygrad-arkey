import codecs, difflib
from dataclasses import dataclass, field
from typing import Any, Generator, TypedDict
from tinygrad.helpers import ansistrip, NO_COLOR, printable, tqdm, TRACEMETA, word_wrap
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import TrackedGraphRewrite, RewriteTrace, UOp, Ops, GroupOp, KernelInfo, srender, sint, sym_infer, range_str, range_start, multirange_str
from tinygrad.uop.render import print_uops, pyrender
from tinygrad.device import ProfileDeviceEvent, ProfileGraphEvent, ProfileGraphEntry, ProfileProgramEvent

uops_colors = {Ops.LOAD: "#ffc0c0", Ops.STORE: "#87CEEB", Ops.CONST: "#e0e0e0", Ops.REDUCE: "#FF5B5B",
               **{x:"#f2cb91" for x in {Ops.DEFINE_LOCAL, Ops.DEFINE_REG}}, Ops.SHAPED_WMMA: "#FF5B5B",
               Ops.RANGE: "#c8a0e0", Ops.BARRIER: "#ff8080", Ops.IF: "#c8b0c0", Ops.SPECIAL: "#c0c0ff",
               Ops.INDEX: "#D8F9E4", Ops.STACK: "#D8F9E4",
               Ops.WMMA: "#efefc0", Ops.MULTI: "#f6ccff", Ops.INS: "#eec4ff",
               **{x:"#D8F9E4" for x in GroupOp.Movement}, **{x:"#ffffc0" for x in GroupOp.ALU}, Ops.THREEFRY:"#ffff80",
               Ops.SLICE: "#E5EAFF", Ops.BUFFER: "#B0BDFF", Ops.GETADDR: "#9DB1F0", Ops.COPY: "#a040a0", Ops.CUSTOM_FUNCTION: "#bf71b6",
               Ops.CALL: "#00B7C8", Ops.FUNCTION: "#C07788", Ops.PARAM: "#14686F", Ops.SOURCE: "#c0c0c0", Ops.BINARY: "#404040",
               Ops.LINEAR: "#7DF4FF",
               Ops.ALLREDUCE: "#ff40a0", Ops.MSELECT: "#d040a0", Ops.MSTACK: "#d040a0", Ops.CONTIGUOUS: "#FFC14D",
               Ops.STAGE: "#AC640D", Ops.REWRITE_ERROR: "#ff2e2e", Ops.AFTER: "#8A7866", Ops.END: "#524C46"}

addrspace_colors = {AddrSpace.REG:"#e68181", AddrSpace.LOCAL:"#e7c86a", AddrSpace.GLOBAL:"#75bd7b"}

# VIZ API

# A step is a lightweight descriptor for a trace entry
# Includes a name, metadata and a URL path for fetching the full data

def create_step(name:str, query:tuple[str, int, int], data=None, depth:int=0, **kwargs) -> dict:
  return {"name":name, "query":f"{query[0]}?ctx={query[1]}&step={query[2]}", "data":data, "depth":depth, **kwargs}

@dataclass(frozen=True)
class VizData:
  trace:RewriteTrace = field(default_factory=lambda: RewriteTrace([], [], {}))
  ctxs:list[dict] = field(default_factory=list)
  ref_map:dict[Any, int] = field(default_factory=dict)
  all_uops:dict[int, UOp] = field(default_factory=dict)

# ** load all saved rewrites

def load_rewrites(data:VizData) -> None:
  assert not data.ctxs and not data.ref_map, "load_rewrites called multiple times"
  for i,k in enumerate(data.trace.keys):
    steps:list[dict] = []
    p:UOp|None = None
    for j,s in enumerate(data.trace.rewrites[i]):
      steps.append(create_step(s.name, ("/graph-rewrites", i, j), loc=s.loc, match_count=len(s.matches), code_line=printable(s.loc),
                               trace=k.tb if j==0 else None, depth=s.depth))
      # get source and binary from Ops.PROGRAM
      if s.name == "View Program":
        p = _reconstruct(data, s.sink, depth=1)
        steps.append(create_step("View UOp List", ("/uops", i, len(steps))))
        steps.append(create_step("View Source", ("/code", i, len(steps)), p.src[3].arg))
        steps.append(create_step("View Disassembly", ("/asm", i, len(steps)), (k.ret, p.src[4].arg)))
    for key in k.keys: data.ref_map[canonicalize_ast(key) if isinstance(key, UOp) else key] = i
    data.ctxs.append({"name":k.display_name, "steps":steps, "prg":p})

# ** get the complete UOp graphs for one rewrite

class GraphRewriteDetails(TypedDict):
  graph: dict                            # JSON serialized UOp for this rewrite step
  uop: str                               # strigified UOp for this rewrite step
  diff: list[str]|None                   # diff of the single UOp that changed
  change: list[int]|None                 # the new UOp id + all its parents ids
  upat: tuple[tuple[str, int], str]|None # [loc, source_code] of the matched UPat

def shape_to_str(s:tuple[sint, ...]): return "(" + ','.join(srender(x) for x in s) + ")"
def pystr(u:UOp) -> str:
   # pyrender may check for shape mismatch
  try: return pyrender(u)
  except Exception: return str(u)

def fmt_colored(s:str) -> str: return ansistrip(s) if NO_COLOR else s

def canonicalize_ast(u:UOp) -> UOp: return u.replace(arg=KernelInfo()) if u.op is Ops.SINK and isinstance(u.arg, KernelInfo) else u

def uop_to_json(data:VizData, x:UOp) -> dict[int, dict]:
  assert isinstance(x, UOp)
  graph: dict[int, dict] = {}
  excluded: set[UOp] = set()
  for u in (toposort:=x.toposort()):
    # always exclude DEVICE/CONST/UNIQUE
    if u.op in {Ops.DEVICE, Ops.CONST, Ops.UNIQUE, Ops.LUNIQUE} and u is not x: excluded.add(u)
    if u.op is Ops.CONST and len(u.src) and u.src[0].op in {Ops.UNIQUE, Ops.LUNIQUE}: excluded.remove(u)
    if u.op is Ops.STACK and len(u.src) == 0: excluded.add(u)
    # exclude RESHAPE/EXPAND that only serve to broadcast a CONST
    if u.op in {Ops.RESHAPE, Ops.EXPAND} and len(u.src) >= 1 and u.src[0] in excluded and u is not x: excluded.add(u)
    if u.op in {*GroupOp.Movement, Ops.PARAM}: excluded.update(s for s in u.src if s.op is Ops.STACK and all(x.op is Ops.CONST for x in s.src))
  for u in toposort:
    argst = codecs.decode(str(u.arg), "unicode_escape")
    if u.op in GroupOp.Movement: argst = "("+','.join(shape_to_str(x) for x in u.marg)+")" if u.op in {Ops.SHRINK, Ops.PAD} else shape_to_str(u.marg)
    if u.op is Ops.BINARY: argst = f"<{len(u.arg)} bytes>"
    if u.op is Ops.CONST and dtypes.is_float(u.dtype): argst = f"{u.arg:g}"
    wrap_len = 200 if u.op is Ops.SOURCE else 80
    label = f"{str(u.op).split('.')[1]}{(chr(10)+word_wrap(argst.replace(':', ''), wrap=wrap_len)) if u.arg is not None else ''}"
    if u.dtype != dtypes.void: label += f"\n{u.dtype}"
    for idx,x in enumerate(u.src[:1] if u.op in {Ops.STAGE, Ops.INDEX} else (u.src if u.op is not Ops.END else [])):
      if x in excluded:
        # walk through excluded movement ops to find the underlying CONST
        cx = x
        while cx.op in GroupOp.Movement and len(cx.src) >= 1 and cx.src[0] in excluded: cx = cx.src[0]
        arg = f"{cx.arg:g}" if cx.op is Ops.CONST and dtypes.is_float(cx.dtype) else f"{cx.arg}"
        label += f"\n{cx.op.name}{idx} {arg}" + (f" {cx.src[0].op}" if len(cx.src) else "")
    try:
      if len(rngs:=u.ranges):
        label += f"\n({multirange_str(rngs, color=True)})"
      if u._shape is not None:
        label += f"\n{shape_to_str(u.shape)}"
      if u.op in {Ops.CALL, Ops.FUNCTION}:
        label += f"\n{u.src[0].key.hex()[:8]}"
      if u.op in {Ops.INDEX, Ops.STAGE}:
        if len(u.toposort()) < 30: label += f"\n{u.render()}"
        ranges: list[UOp] = []
        for us in u.src[1:]: ranges += [s for s in us.toposort() if s.op in {Ops.RANGE, Ops.SPECIAL}]
        if ranges: label += "\n"+' '.join([f"{s.render()}={s.vmax+1}" for s in ranges])
      if u.op in {Ops.END, Ops.REDUCE} and len(trngs:=list(UOp.sink(*u.src[range_start[u.op]:]).ranges)):
        label += "\n"+' '.join([f"{range_str(s, color=True)}({s.vmax+1})" for s in trngs])
    except Exception:
      label += "\n<ISSUE GETTING LABEL>"
    ref = data.ref_map.get(canonicalize_ast(u.src[0])) if u.op in {Ops.CALL, Ops.FUNCTION} else None
    if ref is not None: label += f"\ncodegen@{fmt_colored(data.ctxs[ref]['name'])}"
    # NOTE: kernel already has metadata in arg
    if TRACEMETA >= 2 and u.metadata is not None and u.op not in {Ops.CALL, Ops.FUNCTION}: label += "\n"+str(u.metadata)
    # limit SOURCE labels line count
    if u.op is Ops.SOURCE and len(lines:=label.split("\n")) > 40:
      label = "\n".join(lines[:30]) + "\n..."
    graph[id(u)] = {"label":label, "src":[(i,id(x)) for i,x in enumerate(u.src)], "exclude":u in excluded, "color":uops_colors.get(u.op, "#ffffff"),
                    "ref":ref, "tag":repr(u.tag) if u.tag is not None else None,
                    "addrspace":addrspace_colors.get(u.addrspace, None) if u.addrspace is not None else None}
  return graph

def _reconstruct(data:VizData, a:int, depth:int|None=None):
  if depth is None and a in data.all_uops: return data.all_uops[a]
  op, dtype, src, arg, *rest = data.trace.uop_fields[a]
  if depth is not None and depth <= 0: return UOp(op, dtype, (), arg, *rest)
  ret = UOp(op, dtype, tuple(_reconstruct(data, s, None if depth is None else depth-1) for s in src), arg, *rest)
  if depth is None: data.all_uops[a] = ret
  return ret

def get_full_rewrite(data:VizData, ctx:TrackedGraphRewrite) -> Generator[GraphRewriteDetails, None, None]:
  next_sink = _reconstruct(data, ctx.sink)
  yield {"graph":uop_to_json(data, next_sink), "uop":pystr(next_sink), "change":None, "diff":None, "upat":None}
  replaces: dict[UOp, UOp] = {}
  for u0_num,u1_num,upat_loc,dur in tqdm(ctx.matches, disable=not ctx.matches):
    replaces[u0:=_reconstruct(data, u0_num)] = u1 = _reconstruct(data, u1_num)
    try: new_sink = next_sink.substitute(replaces)
    except RuntimeError as e: new_sink = UOp(Ops.NOOP, arg=str(e))
    match_repr = f"# {dur*1e6:.2f} us\n"+printable(upat_loc)
    yield {"graph":(sink_json:=uop_to_json(data, new_sink)), "uop":pystr(new_sink), "change":[id(x) for x in u1.toposort() if id(x) in sink_json],
           "diff":list(difflib.unified_diff(pystr(u0).splitlines(), pystr(u1).splitlines())), "upat":(upat_loc, match_repr)}
    if not ctx.bottom_up: next_sink = new_sink
