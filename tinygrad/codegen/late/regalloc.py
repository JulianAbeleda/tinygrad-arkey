import itertools
from tinygrad.helpers import dedup, getenv
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.renderer.isa import ISARenderer, Register
from tinygrad.dtype import dtypes, PtrDType

PSEUDO_OPS = {Ops.CONST, Ops.NOOP, Ops.AFTER, Ops.BARRIER, Ops.WAIT, Ops.GROUP}

class LinearScanRegallocContext:
  # returns the uop that defines the virtual register
  def vdef(self, v:Register) -> UOp: return self.uops[self.live_range[v][0]]
  def __init__(self, uops:list[UOp], ren:ISARenderer):
    self.uops = uops
    self.ren = ren
    self.idx = itertools.count()
    # the label associated with each loop NOTE: this is only used post regalloc and should be removed
    self.loop_label: dict[UOp, str] = {}

    # compute live ranges
    self.live_range: dict[Register, list[int]] = {}
    lr = self.live_range
    ranges: list[Register] = []
    for i,u in enumerate(reversed(uops)):
      if u.op in PSEUDO_OPS: continue
      defs = u.tag if isinstance(u.tag, tuple) else ()
      src_regs = tuple(s.reg for s in dedup(u.src) if not (u.op is Ops.END and getenv("REGALLOC_END_NO_SOURCE_LIVE", 0) and s.op is not Ops.RANGE))
      for v in defs + src_regs:
        if isinstance(v, Register): lr.setdefault(v, []).insert(0, len(uops) - 1 - i)
      for v in defs:
        if getenv("REGALLOC_NO_LOOP_EXTEND_ADDR", 0) and u.op is Ops.INS and str(u.arg).split(".", 1)[-1] in {"V_OFFSET", "V_IADD"}:
          continue
        if v in lr and (n:=max((lr[rng][-1] for rng in ranges if lr[rng][0] <= lr[v][-1] < lr[rng][-1]), default=None)): lr[v].append(n)
      if u.op is Ops.RANGE: ranges.append(u.reg)

    # REGALLOC_DEBUG: the "no spills" failure is opaque -- it says nothing about WHAT is over the register pool. This
    # traces the peak simultaneously-live virtual count and its composition (categorized by each vreg's defining op),
    # so a spill is diagnosable: too many of one op (e.g. un-serialized fragment packs / store-offsets) points at a
    # SCHEDULING/lifetime issue; a broad mix points at genuine capacity. Reproducible on DEV=PYTHON (no GPU needed).
    if getenv("REGALLOC_DEBUG"):
      import sys
      def _cat(v:Register) -> str:
        d = uops[lr[v][0]]
        if not isinstance(d.arg, tuple): return str(d.arg).split("(",1)[0].split(".")[-1] or str(d.op)
        return str(d.op).split(".")[-1]
      live_at = [0]*len(uops)
      for v,rng in lr.items():
        for i in range(rng[0], rng[-1]+1): live_at[i]+=1
      peak_i = max(range(len(uops)), key=lambda i: live_at[i]) if uops else 0
      comp:dict[str,int] = {}
      for v,rng in lr.items():
        if rng[0] <= peak_i <= rng[-1]:
          c = _cat(v); comp[c] = comp.get(c, 0) + 1
      pool = len(ren._vpool(self)) if hasattr(ren, "_vpool") else "?"
      sys.stderr.write(f"REGALLOC_DEBUG: {len(uops)} uops, PEAK {live_at[peak_i] if uops else 0} live vregs @ uop {peak_i}, pool={pool}\n")
      for k,n in sorted(comp.items(), key=lambda kv:-kv[1]): sys.stderr.write(f"  {n:5d}  {k}\n")
      if getenv("REGALLOC_DEBUG_DETAIL"):
        for v,rng in sorted(lr.items(), key=lambda kv:(kv[1][0], kv[0].index if hasattr(kv[0], "index") else 0)):
          if not (rng[0] <= peak_i <= rng[-1]): continue
          d, e = uops[rng[0]], uops[rng[-1]]
          sys.stderr.write(f"  LIVE {v}: {rng[0]}..{rng[-1]} {d.op} {d.arg} -> {e.op} {e.arg} "
                           f"src={[getattr(s, 'arg', None) for s in d.src]}\n")
      if (win := getenv("REGALLOC_DEBUG_WINDOW", 0)):
        center = getenv("REGALLOC_DEBUG_WINDOW_CENTER", peak_i)
        lo, hi = max(0, center-int(win)), min(len(uops), center+int(win)+1)
        for j in range(lo, hi):
          u = uops[j]
          sys.stderr.write(f"  UOP {j}: {u.op} {u.arg} tag={u.tag} src={[getattr(s, 'arg', None) for s in u.src]}\n")
      if getenv("REGALLOC_DEBUG_END_DETAIL"):
        for i,u in enumerate(uops):
          if u.op is Ops.END:
            sys.stderr.write(f"  END {i}: src={[(s.op, getattr(s, 'arg', None), getattr(s.reg, 'index', None)) for s in u.src]}\n")

    # allocate registers
    self.stack_size: int = 0
    self.locals: dict[UOp, UOp] = {}
    self.spills: dict[Register, UOp] = {} # mapping from virtual to stack slot
    self.remats: dict[tuple[int, Register], Register] = {} # rematerialized virtual -> real at program point
    self.remat_before: dict[int, list[Register]] = {} # rematerializations to insert before a program point
    self._remat_uops: dict[tuple[int, Register], tuple[UOp, list[UOp]]] = {}
    self.reals: dict[int, dict[Register, Register]] = {} # mapping from virtual to real at each program point
    self.insert_before: dict[int, list[tuple[Register, Register]]] = {} # fills to be inserted at each program point
    live: dict[Register, Register] = {} # mapping from virtual to real that's currently assigned to it
    live_ins: list[dict[Register, Register]] = [] # mapping from virtual to real at loop entry
    remat_pinned: set[Register] = set() # remat dependencies that must not be evicted while materializing a parent

    def alloc(cons:tuple[Register, ...], i:int) -> Register:
      live_inv = {v:k for k,v in live.items()}
      # allocate the best register. Registers not in live or not used again are free and have priority,
      # otherwise pick the one with the furthest next use. Regs that appear first in cons have priority in case of a tie
      reg,vreg = max(((r,live_inv.get(r)) for r in cons),
                    key=lambda rv: -1 if rv[1] in remat_pinned else next((j-i for j in ([] if rv[1] is None else lr[rv[1]]) if j >= i), len(uops)))
      return live.pop(vreg) if vreg is not None else reg

    def can_remat(v:Register, i:int) -> bool:
      if not getenv("REGALLOC_ADDR_REMAT", 0): return False
      d, u = self.vdef(v), uops[i]
      dop, uop = str(d.arg).split(".", 1)[-1], str(u.arg).split(".", 1)[-1]
      pure_addr = {"V_AND", "V_IMUL", "V_IADD", "V_OFFSET", "V_LSHR"}
      pure_addr_roots = pure_addr | {"WG_ID", "WI_ID", "MOV_S2V"}
      addr_users = pure_addr | {"DS_LOAD", "DS_STORE", "DS_LOAD_B128", "DS_STORE_B128", "DS_STORE_B64",
                                "GATED_STORE", "GATED_STORE_B128", "GATED_STORE_B64",
                                "GLOBAL_LOAD", "GLOBAL_LOAD_B128", "GLOBAL_STORE"}
      pure_def = (d.op is Ops.INS and dop in pure_addr_roots and d.dtype in dtypes.ints) or d.op is Ops.SPECIAL
      return pure_def and ((u.op is Ops.INS and uop in addr_users) or (u.op is Ops.END and not getenv("REGALLOC_ADDR_REMAT_NO_END", 0)))

    def remat_addr_def(v:Register) -> bool:
      if not getenv("REGALLOC_ADDR_REMAT", 0): return False
      d = self.vdef(v)
      dop = str(d.arg).split(".", 1)[-1]
      return d.op is Ops.SPECIAL or (d.dtype in dtypes.ints and d.op is Ops.INS and dop in {"V_AND", "V_IMUL", "V_IADD", "V_OFFSET", "V_LSHR", "WG_ID", "WI_ID", "MOV_S2V"})

    # assign register to spilled virtual and record load to be emitted before current uop, also assign it a stack slot
    def fill(v:Register, i:int, cons:tuple[Register, ...]|None=None, emit_remat_before=False) -> Register:
      if can_remat(v, i) or (getenv("REGALLOC_ADDR_REMAT", 0) and self.vdef(v).op is Ops.SPECIAL):
        pinned:list[Register] = []
        for s in self.vdef(v).src:
          if not isinstance(sv:=s.reg, Register): continue
          if sv not in live: live[sv] = fill(sv, i)
          self.reals.setdefault(i, {})[sv] = live[sv]
          remat_pinned.add(sv)
          pinned.append(sv)
        r = alloc(cons if cons is not None else v.cons, i)
        for sv in pinned: remat_pinned.discard(sv)
        self.remats[(i, v)] = r
        if emit_remat_before: self.remat_before.setdefault(i, []).append(v)
        return r
      if v not in self.spills:
        dt = self.vdef(v).dtype
        sz = dt.scalar().itemsize * dt.count if not isinstance(dt, PtrDType) else 8
        offset = self.stack_size + (sz - self.stack_size % sz) % sz
        self.spills[v] = UOp.const(dtypes.int32, offset)
        self.stack_size = offset + sz
      r = alloc(cons if cons is not None else v.cons, i)
      self.insert_before.setdefault(i, []).append((v, r))
      return r

    for i,u in enumerate(uops):
      if u.op in PSEUDO_OPS: continue
      # allocate uses
      for s in u.src:
        # HACK: cause of later hacks to lower range
        if u.op is Ops.END: continue
        if not isinstance(v:=s.reg, Register): continue
        if v not in live: live[v] = fill(v, i)
        self.reals.setdefault(i, {})[v] = live[v]

      # allocate defs
      if isinstance(u.tag, tuple):
        for j,v in enumerate(u.tag):
          # register should only be defined once
          assert isinstance(v, Register) and lr[v][0] == i
          cons = v.cons
          # two address instructions (src is reused by def) can only coalesce reused src. reused src goes first to get priority in case of a tiebreak
          if ren.is_two_address(u) and j == 0:
            uses = tuple(live.get(s.reg) for s in u.src)
            cons = ((uses[0],) if uses[0] in cons else ()) + tuple(r for r in cons if r not in uses)
          # HACK: cause the range is missing the comparison
          live[v] = alloc(cons, i+1 if u.op is not Ops.RANGE else i)
          self.reals.setdefault(i, {})[v] = live[v]

      # allocate stack array
      if u.op is Ops.DEFINE_LOCAL:
        self.locals[u] = UOp.const(dtypes.int32, self.stack_size)
        self.stack_size += u.dtype.nbytes()

      # loop prologue, avoid loading inside the loop
      if u.op is Ops.RANGE:
        # we move to registers vars used in the loop sorted by next use, vars not used in the loop will not be reloaded in the epilogue
        used_in_loop = [v for v in live.keys() | self.spills.keys() if any(i <= l < lr[u.reg][-1] for l in lr[v])]
        sorted_uses = sorted(used_in_loop, key=lambda k: (next(l-i for l in lr[k] if l >= i), lr[k][0], k.name, k.index))
        live_in: dict[Register, Register] = {}
        for v in sorted_uses:
          # if all the possible registers are already in live_in there's no space for this var
          if set(v.cons).issubset(live_in.values()): continue
          if v not in live: live[v] = fill(v, i)
          live_in[v] = live[v]
        if getenv("REGALLOC_DEBUG_LOOP_LIVE"):
          import sys
          sys.stderr.write(f"REGALLOC_LOOP_RANGE {i}: loop_end={lr[u.reg][-1]} live_in={[(v.index, r.index, lr[v], str(self.vdef(v).arg).split('.',1)[-1]) for v,r in live_in.items()]}\n")
        live_ins.append(live_in)

      # loop epilogue, reload registers that were live at loop entry
      if u.op is Ops.END:
        # TODO: if a uop is in a different reg in live out vs live in move between registers instead of loading
        # TODO: don't reload if first use in loop is a load
        if getenv("REGALLOC_DEBUG_LOOP_LIVE"):
          import sys
          sys.stderr.write(f"REGALLOC_LOOP_END {i}: restoring={[(v.index, r.index, v in live, None if v not in live else live[v].index) for v,r in live_ins[-1].items()]}\n")
        for v,r in live_ins.pop().items():
          if remat_addr_def(v) or v not in live or live[v] != r:
            live[v] = fill(v, i, (r,), emit_remat_before=not getenv("REGALLOC_ADDR_REMAT_END_NO_EMIT", 0))
        if getenv("REGALLOC_DEBUG_LOOP_LIVE"):
          import sys
          sys.stderr.write(f"REGALLOC_LOOP_END {i}: remat_before={[(v.index, self.remats[(i, v)].index) for v in self.remat_before.get(i, [])]}\n")

    if getenv("REGALLOC_DEBUG_SPILLS"):
      import sys
      sys.stderr.write(f"REGALLOC_SPILLS: count={len(self.spills)} stack_size={self.stack_size}\n")
      for v,slot in sorted(self.spills.items(), key=lambda kv: kv[0].index):
        d = self.vdef(v)
        sys.stderr.write(f"  SPILL {v} slot={slot.arg} def={d.op} {d.arg} range={self.live_range[v]}\n")
    if getenv("REGALLOC_DEBUG_REMAT"):
      import sys
      def _aname(u:UOp) -> str: return str(u.arg).split(".", 1)[-1] if u.op is Ops.INS else str(u.op).split(".")[-1]
      rows = []
      for (i, v), r in sorted(self.remats.items(), key=lambda kv:(kv[0][0], kv[0][1].index)):
        d, u = self.vdef(v), uops[i]
        rows.append((_aname(u), _aname(d), i, v.index, r.index, self.live_range[v]))
      sys.stderr.write(f"REGALLOC_REMAT: count={len(rows)}\n")
      cats:dict[tuple[str, str], int] = {}
      for user, define, *_ in rows: cats[(user, define)] = cats.get((user, define), 0) + 1
      for (user, define), n in sorted(cats.items(), key=lambda kv:(kv[0][0], kv[0][1])):
        sys.stderr.write(f"  REMAT_CAT user={user} def={define} count={n}\n")
      for user, define, i, vi, ri, rng in rows[:getenv("REGALLOC_DEBUG_REMAT_LIMIT", 80)]:
        sys.stderr.write(f"  REMAT i={i} v={vi}->r{ri} user={user} def={define} range={rng}\n")

  def remat(self, v:Register, i:int) -> tuple[UOp, list[UOp]]:
    if (cached := self._remat_uops.get((i, v))) is not None: return cached
    d = self.vdef(v)
    before:list[UOp] = []
    srcs:list[UOp] = []
    for s in d.src:
      sv = s.reg
      if not isinstance(sv, Register):
        srcs.append(s)
      elif (i, sv) in self.remats:
        rs, rb = self.remat(sv, i)
        before += rb
        srcs.append(rs)
      else:
        srcs.append(s.replace(tag=(self.reals[i][sv],)) if i in self.reals and sv in self.reals[i] and isinstance(s.tag, tuple) else s)
    nx = d.replace(src=tuple(srcs), tag=(self.remats[(i, v)],))
    ret = (nx, before + [nx])
    self._remat_uops[(i, v)] = ret
    return ret

def regalloc_rewrite(ctx:LinearScanRegallocContext, x:UOp):
  i = next(ctx.idx)
  if x.op in PSEUDO_OPS: return None
  nsrc = []
  before:list[UOp] = []
  for j,s in enumerate(x.src):
    # v here is the virtual defined by the original s as s is the rewritten version
    v = ctx.uops[i].src[j].reg
    if i in ctx.reals and v in ctx.spills: nsrc.append(ctx.ren.fill(ctx.spills[v], ctx.vdef(v), ctx.reals[i][v]))
    elif isinstance(v, Register) and (i, v) in ctx.remats:
      rs, rb = ctx.remat(v, i)
      before += rb
      nsrc.append(rs)
    else: nsrc.append(s)
  ndefs = tuple(ctx.reals[i][v] for v in x.tag) if isinstance(x.tag, tuple) else x.tag
  if x.op is Ops.DEFINE_LOCAL: nx = ctx.ren.isel_matcher.rewrite(ctx.ren.stack_pointer().index(ctx.locals[x], dtype=x.dtype, tag=ndefs))
  else: nx = x.replace(src=tuple(nsrc), tag=ndefs)

  emitted_remats:set[Register] = set()
  for v in ctx.remat_before.get(i, []):
    if v in emitted_remats: continue
    emitted_remats.add(v)
    _rs, rb = ctx.remat(v, i)
    before += rb
  before = [ctx.ren.fill(ctx.spills[v], ctx.vdef(v), r) for v,r in ctx.insert_before.get(i, [])] + before
  after = [ctx.ren.spill(ctx.spills[v], nx) for v in x.tag if v in ctx.spills] if isinstance(x.tag, tuple) else []

  # alloc/dealloc stack
  if ctx.stack_size > 0:
    sp = ctx.ren.stack_pointer()
    offset = UOp(Ops.CONST, sp.dtype, arg=ctx.stack_size)
    if i == 0: before = [ctx.ren.isel_matcher.rewrite(UOp(Ops.SUB, sp.dtype, (sp, offset), tag=sp.tag))] + before
    elif i == len(ctx.uops) - 2: before += [ctx.ren.isel_matcher.rewrite(UOp(Ops.ADD, sp.dtype, (sp, offset), tag=sp.tag))]

  return nx, before + [nx] + after

pm_regalloc_rewrite = PatternMatcher([
  (UPat({Ops.INS, Ops.RANGE, Ops.END, Ops.DEFINE_REG, Ops.DEFINE_LOCAL, Ops.PARAM, Ops.DEFINE_VAR, Ops.SPECIAL} | PSEUDO_OPS, name="x"),
   regalloc_rewrite),
])
