#!/usr/bin/env python3
"""T5: instruction-count comparison, fused Q4_K vs dense fp16, at the same (512,12288,4096)
generic-TC shape established by T4 (scratchpad/t4_fused_generic_tc_execute.py: fused 544.27 GFLOPS
vs dense fp16 2733.15 GFLOPS on real Metal hardware, same session). Compile-only, no GPU dispatch.

Structural template (per docs/task_workflow/input/metal-prefill-loop-body-decomposition-scope-20260730.md
section 4.2): scratchpad/kv_tile_amortization_probe.py's shape -- compile, locate the loop body,
classify every instruction against a table, emit the AMD-shaped table with counts/shares/coverage.
That probe's own Metal section (written earlier in the campaign) recorded that `xcrun metal` failed
outright on this machine at the time ("missing Metal Toolchain"). It is present now (Apple metal
version 32023.883 / metal-objdump for air64 targets), so this probe goes one rung better than pure
MSL-text classification: it feeds the SAME rendered MSL that T4 executed through the real offline
`xcrun metal -c ... -std=metal4.0` frontend and disassembles the resulting .air via
`xcrun metal-objdump --disassemble` -- real compiled LLVM IR (AIR), not source text. This is still
NOT final GPU ISA: AIR is upstream of the driver-side backend that does register allocation and
instruction scheduling (the Metal-side analogue of AMD's s_waitcnt/s_delay_alu instructions is
inserted THERE, inside the Metal driver's JIT, not visible from AIR). That backend's output is only
inspectable via the `applegpu` disassembler (github.com/dougallj/applegpu), which
extra/disassemblers/applegpu/ does NOT contain in this checkout -- confirmed absent, not assumed.
So exactly as the scope doc requires: every conclusion below is stated as "of the AIR instructions we
can see", never as a claim about the final scheduled ISA, and register/occupancy (group c) is
reported as NOT OBSERVABLE by this instrument, not inferred.

Kernels reused verbatim from T4's harness (same ASTs, same generic TC opt, same shape), not
rebuilt: T1B._experiment_a_naive_dodge(DEVICE, WIDTH) (fused Q4_K) and T1._dense_gemm_ast(DEVICE)
(dense fp16 ceiling), both forced through T1._force_generic_tc exactly as T4 measured them.

Pipeline per kernel:
  1. to_program(ast_forced, MetalRenderer(...)) -> extract Ops.SOURCE MSL text (identical technique
     to T1/T1b/T4; this call also runs MetalCompiler.compile, the offline MTLCodeGenService path
     documented as device-independent in T4's docstring -- no Device[...] is ever opened here).
  2. Write MSL to a temp .metal file; `xcrun metal -c file.metal -o file.air -std=metal4.0`
     (metal4.0 chosen to match ops_metal.py's own macos_major>=26 branch on this machine's
     platform.mac_ver()=='26.5' -- read directly off ops_metal.py:88, not guessed).
  3. `xcrun metal-objdump --disassemble file.air` -> real LLVM IR text.
  4. Parse basic blocks, locate the K-reduction loop via natural-loop detection (header = the block
     whose phi nodes take one operand from the function's entry block; loop body = blocks that are
     both reachable from the header and can reach back to it -- a real SCC computation on the parsed
     CFG, not a hardcoded label number, since AIR's SSA numbering is compiler-assigned and differs
     between the two kernels).
  5. Classify every instruction in the loop body's block set into the 9 groups this task specifies,
     using a forward dataflow tag (PACKED_DERIVED) seeded at every `load i32 ... addrspace(1)`
     (the only i32-typed buffer in the fused kernel is the packed Q4_K source -- confirmed by checking
     which PARAM dtype is uint32 in T4/T1B, not assumed) and propagated through every consuming
     instruction. This distinguishes the two structurally-identical-looking uses of shl/lshr/and/or in
     this code: index/address math (present in BOTH kernels, computing which byte/half to fetch) vs
     the Q4_K bit-unpacking tax (present ONLY in the fused kernel, manipulating already-loaded packed
     words). Full classification rules are in `classify_block_instrs` below, each with a one-line
     justification; nothing is eyeballed from source text.
  6. Separately (not instruction counting): a lane-redundancy audit on the RENDERED MSL address
     expressions for every packed-word load, checking which of the 5 lidx0 bits (local_size=(32,1,1),
     confirmed from T4's printed prog.arg.local_size) each address depends on. This is the direct,
     algebraic check of the standing "redundant per-lane scalar loads" suspicion -- not inferred, not
     estimated, computed from the literal index expressions already in fused_q4k.metal.
"""
from __future__ import annotations
import re, subprocess, sys, json, tempfile, os
from collections import defaultdict, Counter
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp")
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp/scratchpad")

from tinygrad.helpers import Target
from tinygrad.codegen import to_program
from tinygrad.uop.ops import Ops

import t1_generic_tc_dequant_probe as T1
import t1b_generic_tc_dequant_vectorized_probe as T1B

DEVICE = "METAL"
WIDTH = T1B.WIDTH_FOR_BACKEND[DEVICE]
WORKDIR = "/tmp/t5_work"
os.makedirs(WORKDIR, exist_ok=True)

# ------------------------------------------------------------------ step 1-3: render, compile, disassemble
def render_source(ast_builder, label: str) -> tuple[str, dict]:
  ast = ast_builder()
  ast_forced = T1._force_generic_tc(ast)
  target_str, make_renderer = T1.TARGETS[DEVICE]
  renderer = make_renderer(Target.parse(target_str))
  prog = to_program(ast_forced, renderer)
  source = next((u.arg for u in prog.src if u.op is Ops.SOURCE and isinstance(u.arg, str)), None)
  assert source, f"{label}: no rendered MSL source"
  meta = {"globals": prog.arg.globals, "global_size": prog.arg.global_size, "local_size": prog.arg.local_size}
  path = os.path.join(WORKDIR, f"{label}.metal")
  with open(path, "w") as f: f.write(source)
  return path, meta


def compile_and_disassemble(metal_path: str, label: str) -> str:
  air_path = os.path.join(WORKDIR, f"{label}.air")
  r = subprocess.run(["xcrun", "metal", "-c", metal_path, "-o", air_path, "-std=metal4.0"],
                      capture_output=True, text=True)
  assert r.returncode == 0, f"{label}: xcrun metal -c failed: {r.stdout} {r.stderr}"
  r2 = subprocess.run(["xcrun", "metal-objdump", "--disassemble", air_path],
                       capture_output=True, text=True)
  assert r2.returncode == 0 and r2.stdout, f"{label}: metal-objdump failed: {r2.stdout} {r2.stderr}"
  ll_path = os.path.join(WORKDIR, f"{label}.ll")
  with open(ll_path, "w") as f: f.write(r2.stdout)
  return r2.stdout


# ------------------------------------------------------------------ step 4: parse blocks / CFG / find loop
_BLOCK_HEADER = re.compile(r"^(\d+):\s*(?:;.*)?$")
_INSTR = re.compile(r"^\s*(?:%(\d+)\s*=\s*)?(.+)$")
_BR_UNCOND = re.compile(r"^br label %(\d+)$")
_BR_COND = re.compile(r"^br i1 %\d+, label %(\d+), label %(\d+)")
_RET = re.compile(r"^ret\b")
_OPCODE = re.compile(r"^([a-zA-Z][a-zA-Z0-9_.]*)")
_DEF_FUNC = re.compile(r"^define .*@(\w+)\(([^)]*)\)")


def parse_kernel_function(ll_text: str, func_name_substr: str):
  """Extract the named kernel function's body (skip the __WMMA helper and declares), split into
  basic blocks in TEXT order (not necessarily execution order -- LLVM prints blocks in creation
  order, which is why loop detection below is graph-based, not position-based)."""
  lines = ll_text.splitlines()
  start = next(i for i, l in enumerate(lines) if l.startswith("define") and func_name_substr in l)
  # function body ends at the first line that is just "}"
  end = next(i for i in range(start, len(lines)) if lines[i].rstrip() == "}")
  body = lines[start:end + 1]
  # Extract the balanced-paren argument list (types like `addrspace(1)` contain their own parens,
  # so a naive "up to the first )" regex undercounts -- balance parens by hand instead).
  sig = body[0]
  paren_start = sig.index("(")
  depth, i = 0, paren_start
  while True:
    if sig[i] == "(": depth += 1
    elif sig[i] == ")": depth -= 1
    if depth == 0: break
    i += 1
  argstr = sig[paren_start + 1:i]
  nargs = len(re.findall(r"%\d+\b", argstr))  # count actual %N parameter placeholders, not commas
  entry_label = nargs  # LLVM's implicit numbering: unnamed entry block gets the next free number

  blocks: dict[int, dict] = {}
  cur_label = entry_label
  cur_instrs: list[tuple[int | None, str]] = []
  for line in body[1:-1]:
    line = line.rstrip()
    if not line.strip():
      continue
    m = _BLOCK_HEADER.match(line.strip())
    if m:
      blocks[cur_label] = {"instrs": cur_instrs}
      cur_label = int(m.group(1))
      cur_instrs = []
      continue
    m2 = _INSTR.match(line)
    if not m2:
      continue
    reg = int(m2.group(1)) if m2.group(1) is not None else None
    text = m2.group(2).strip()
    cur_instrs.append((reg, text))
  blocks[cur_label] = {"instrs": cur_instrs}

  # successors from the terminator (last instruction) of each block
  for lbl, b in blocks.items():
    term = b["instrs"][-1][1]
    if (m := _BR_UNCOND.match(term)):
      b["succ"] = [int(m.group(1))]
    elif (m := _BR_COND.match(term)):
      b["succ"] = [int(m.group(1)), int(m.group(2))]
    elif _RET.match(term):
      b["succ"] = []
    else:
      raise RuntimeError(f"block {lbl}: unrecognized terminator: {term!r}")
  return blocks, entry_label


def find_loop_blocks(blocks: dict, entry_label: int) -> tuple[int, set[int]]:
  """Header = block with a phi whose entry-predecessor operand is the function's entry label.
  Loop body = blocks that are both forward-reachable from the header and can themselves reach the
  header (a direct SCC check on this small CFG) -- standard natural-loop membership, not a guess."""
  header = None
  for lbl, b in blocks.items():
    for reg, text in b["instrs"]:
      if text.startswith("phi") and f"%{entry_label}]" in text.replace(" ", ""):
        header = lbl
        break
    if header is not None:
      break
  assert header is not None, "no loop header found (no phi references the entry block)"

  def forward_reach(start):
    seen, stack = {start}, [start]
    while stack:
      u = stack.pop()
      for v in blocks[u]["succ"]:
        if v not in seen:
          seen.add(v); stack.append(v)
    return seen

  reach_from_header = forward_reach(header)
  loop_blocks = set()
  for b in reach_from_header:
    if header in forward_reach(b):
      loop_blocks.add(b)
  return header, loop_blocks


# ------------------------------------------------------------------ step 5: classify
GROUPS = ["global_load", "threadgroup_ldst", "barrier", "simdgroup_matrix",
          "shift_mask_bitwise", "other_arith", "index_addr_math", "control", "unclassified"]

FP_OPS = {"fmul", "fsub", "fadd", "fdiv", "fptrunc", "fpext", "sitofp", "uitofp", "fptosi", "fptoui"}
BIT_OPS = {"shl", "lshr", "ashr", "and", "or", "xor", "trunc", "zext", "sext", "select", "bitcast"}


def classify_kernel(blocks: dict, loop_blocks: set[int], is_fused: bool):
  """Returns (Counter of group->count, list of (block,reg,text,group,reason) for audit,
  set of unclassified instruction texts)."""
  # ---- pass 1: global PACKED_DERIVED dataflow tag, seeded at `load i32 ... addrspace(1)` results,
  # propagated forward through every consuming instruction (operand-by-register-number match) within
  # the loop-block set only (prologue/entry constants are invariant, never packed-derived, and are
  # not classified anyway since they're outside the loop body).
  packed_derived: set[int] = set()
  all_loop_instrs: list[tuple[int, int | None, str]] = []
  for lbl in loop_blocks:
    for reg, text in blocks[lbl]["instrs"]:
      all_loop_instrs.append((lbl, reg, text))

  def opcode_of(text: str) -> str:
    # LLVM prefixes call-site qualifiers ("tail", "musttail", "notail") before the real "call"
    # opcode -- strip them so `op == "call"` matches instead of silently falling through to
    # unclassified (caught by inspecting the fallback dump below during development).
    t = text
    for prefix in ("tail ", "musttail ", "notail "):
      if t.startswith(prefix):
        t = t[len(prefix):]
    m = _OPCODE.match(t)
    return m.group(1) if m else ""

  def is_packed_load(text: str) -> bool:
    # the only i32-typed global buffer in the fused kernel is the packed Q4_K source (confirmed via
    # T1B/T1's PARAM dtype check in t4_fused_generic_tc_execute.py); a `load i32, i32 addrspace(1)*`
    # is unambiguously a packed-word fetch. Dense kernel has no i32 buffer at all -> never matches.
    return text.startswith("load i32, i32 addrspace(1)*")

  def referenced_regs(text: str) -> list[int]:
    return [int(r) for r in re.findall(r"%(\d+)\b", text)]

  changed = True
  while changed:
    changed = False
    for lbl, reg, text in all_loop_instrs:
      if reg is None or reg in packed_derived:
        continue
      op = opcode_of(text)
      if op == "getelementptr":
        continue  # address computation never propagates PACKED_DERIVED forward past itself
      if is_packed_load(text):
        packed_derived.add(reg); changed = True; continue
      refs = referenced_regs(text)
      if any(r in packed_derived for r in refs):
        packed_derived.add(reg); changed = True

  results: list[tuple[int, int | None, str, str, str]] = []
  counts = Counter()
  for lbl, reg, text in all_loop_instrs:
    op = opcode_of(text)
    refs = referenced_regs(text)
    group, reason = None, ""
    if op in ("br", "ret"):
      group, reason = "control", "terminator"
    elif op == "phi":
      group, reason = "control", "loop-carried merge"
    elif op == "icmp":
      group, reason = "control", "branch predicate compute"
    elif op == "call" and "air.simdgroup_matrix" in text:
      group, reason = "simdgroup_matrix", "the WMMA call itself"
    elif op in ("shufflevector", "insertelement", "extractelement"):
      group, reason = "simdgroup_matrix", "WMMA operand/result marshalling (pad 2-wide <-> 64-wide)"
    elif op == "load":
      group, reason = "global_load", "device addrspace(1) load"
    elif op == "getelementptr":
      group, reason = "index_addr_math", "address computation"
    elif op == "call" and "air.convert.f.f32.u.i32" in text:
      group, reason = "other_arith", "int->float conversion feeding scale reconstruction"
    elif op == "call" and "llvm.usub.sat" in text:
      group, reason = "shift_mask_bitwise", "saturating index bookkeeping for byte-straddling field select"
    elif op in FP_OPS:
      group, reason = "other_arith", "floating-point arithmetic/conversion"
    elif op in BIT_OPS:
      if any(r in packed_derived for r in refs) or reg in packed_derived:
        group, reason = "shift_mask_bitwise", "operand traces to a packed-word load (dequant bit-unpack)"
      else:
        group, reason = "index_addr_math", "operand traces only to gid/lid/loop-index (address bit-packing)"
    elif op in ("add", "sub", "mul"):
      if any(r in packed_derived for r in refs) or reg in packed_derived:
        group, reason = "shift_mask_bitwise", "integer arithmetic on packed-derived operand"
      else:
        group, reason = "index_addr_math", "integer arithmetic building an address offset"
    else:
      group, reason = "unclassified", f"no rule for opcode {op!r}"
    counts[group] += 1
    results.append((lbl, reg, text, group, reason))
  return counts, results, packed_derived


# ------------------------------------------------------------------ step 6: lane-redundancy audit (MSL-level)
LIDX_BITS = {"alu0": "lidx0>>4 (bit4)", "alu1": "(lidx0>>2)&1 (bit2)", "alu2": "(lidx0>>1)&1 (bit1)",
             "alu3": "(lidx0>>3)&1 (bit3)", "alu4": "lidx0&1 (bit0)"}


def lane_redundancy_audit(metal_path: str) -> dict:
  """For every `data3_...` (packed buffer) load's index expression in the rendered MSL, determine
  which of the 5 lidx0-derived aluN terms (bit4,bit3,bit2,bit1,bit0) appear in it. A packed load
  whose index omits some bits means every lane sharing the OTHER bits' values fetches the identical
  address -- 2**(#omitted bits) lanes redundantly issue the same load. Read directly off the actual
  index-expression text in fused_q4k.metal; no simulation, no estimate."""
  with open(metal_path) as f:
    src = f.read()
  loop_lines = [l for l in src.splitlines() if "data3_" in l and ("val" in l or "alu9 " in l or "alu10 " in l
                or "alu11 " in l or "alu18 " in l)]
  # collect the alu9/alu10/alu11/alu18 address-base definitions (may reference each other)
  defs = {}
  for l in src.splitlines():
    m = re.match(r"\s*int (alu\d+) = (.*);", l.strip())
    if m:
      defs[m.group(1)] = m.group(2)

  def bits_referenced(expr: str, seen=None) -> set[str]:
    seen = seen or set()
    found = set()
    for bit_var in LIDX_BITS:
      if re.search(rf"\b{bit_var}\b", expr):
        found.add(bit_var)
    for name, subexpr in defs.items():
      if name in seen:
        continue
      if re.search(rf"\b{name}\b", expr):
        found |= bits_referenced(subexpr, seen | {name})
    return found

  val_lines = [l.strip() for l in src.splitlines() if re.match(r"\s*uint val\d+ = ", l)]
  audit = []
  for l in val_lines:
    # `uint valN = (*(data3_NNNN+<expr>));` -- <expr> is either `(...)` or a bare identifier
    # (val12/val13 have no inner parens: `data3_7077888+alu9`). Strip exactly the trailing `));`
    # and the leading `(*(data3_NNNN+`, then peel one redundant wrapping `(...)` if present.
    m = re.match(r"uint (val\d+) = \(\*\(data3_\d+\+(.*)\)\);", l)
    name, idxexpr = m.group(1), m.group(2)
    if idxexpr.startswith("(") and idxexpr.endswith(")"):
      idxexpr = idxexpr[1:-1]
    bits = bits_referenced(idxexpr)
    audit.append({"value": name, "index_expr": idxexpr, "lidx0_bits_used": sorted(bits),
                  "n_bits_used": len(bits), "redundant_lanes_per_group": 2 ** (5 - len(bits))})
  # activation load for comparison
  act_line = next(l.strip() for l in src.splitlines() if "data1_" in l and "half2" in l and "val14" in l)
  act_bits = bits_referenced(act_line)
  audit.append({"value": "val14(activation, control)", "index_expr": act_line, "lidx0_bits_used": sorted(act_bits),
                "n_bits_used": len(act_bits), "redundant_lanes_per_group": 2 ** (5 - len(act_bits))})
  return {"local_size_assumed": 32, "per_load": audit}


def dynamic_per_iteration_estimate(blocks: dict, loop_blocks: set[int], results) -> tuple[int, dict, Counter]:
  """The fused kernel's loop body contains real `br i1` diamonds (e.g. 135 vs 143), unlike the AMD
  loop body or the dense fp16 kernel, which are straight-line. Each diamond's predicate (`alu14 =
  ((Ridx0>>2)&7)<4` and friends) depends ONLY on the loop induction variable Ridx0 -- never on
  lidx0/gid -- so it is UNIFORM across the whole SIMD group: every lane takes the same side on a
  given iteration, and only one side's instructions actually execute that iteration (no per-lane
  divergence cost, but the static block-sum overcounts what runs on any single pass). This is not
  an estimate of an unknown quantity: which side runs on which iteration is fully determined by
  Ridx0's value, so the exact dynamic count per iteration is computable from the parsed CFG alone --
  find each "diamond" (two blocks sharing one predecessor and one successor) and take the OTHER
  blocks (shared, non-branching) at face value plus one representative side of each diamond. Since
  both diamond sides are taken equally often over the full 512-iteration sweep (period-32 pattern,
  16 vs 16 per 32), the AVERAGE dynamic per-iteration count is the exact expectation, not a guess."""
  # find diamonds: blocks A,B such that some header H has succ=[A,B] and A,B share the same single
  # successor C (a linear predecessor->{A,B}->successor CFG shape, exactly this kernel's pattern).
  diamonds = []
  handled = set()
  for lbl, b in blocks.items():
    if lbl not in loop_blocks or len(b.get("succ", [])) != 2:
      continue
    a_lbl, b_lbl = b["succ"]
    if a_lbl in handled or b_lbl in handled:
      continue
    a_succ, b_succ = blocks[a_lbl]["succ"], blocks[b_lbl]["succ"]
    if a_succ == b_succ and len(a_succ) == 1:
      diamonds.append((lbl, a_lbl, b_lbl))
      handled.add(a_lbl); handled.add(b_lbl)
  diamond_blocks = {x for tri in diamonds for x in tri[1:]}
  straight_blocks = loop_blocks - diamond_blocks
  straight_total = sum(len(blocks[l]["instrs"]) for l in straight_blocks)
  diamond_avg_total = sum((len(blocks[a]["instrs"]) + len(blocks[b]["instrs"])) / 2 for _, a, b in diamonds)
  dynamic_total = straight_total + diamond_avg_total

  # per-group dynamic average: straight blocks counted fully, each diamond pair counted as the
  # average of its two sides' group counts.
  by_block_group = defaultdict(Counter)
  for lbl, reg, text, group, reason in results:
    by_block_group[lbl][group] += 1
  dyn_counts = Counter()
  for l in straight_blocks:
    dyn_counts += by_block_group[l]
  for _, a, b in diamonds:
    for g in GROUPS:
      dyn_counts[g] += (by_block_group[a][g] + by_block_group[b][g]) / 2
  return dynamic_total, {"diamonds": diamonds, "straight_blocks": sorted(straight_blocks)}, dyn_counts


# ------------------------------------------------------------------ report
def emit_table(label: str, counts: Counter, total_instrs: int):
  print(f"\n=== {label}: loop-body instruction classification (AIR/LLVM IR) ===")
  print(f"{'group':<20}{'count':>8}{'share':>10}")
  accounted = 0
  for g in GROUPS:
    c = counts.get(g, 0)
    accounted += c
    print(f"{g:<20}{c:>8}{c/total_instrs*100:>9.1f}%")
  print(f"{'total':<20}{total_instrs:>8}{100.0:>9.1f}%")
  print(f"accounted fraction = {accounted}/{total_instrs} = {accounted/total_instrs*100:.2f}%")


def main():
  fused_metal, fused_meta = render_source(lambda: T1B._experiment_a_naive_dodge(DEVICE, WIDTH), "fused_q4k")
  dense_metal, dense_meta = render_source(lambda: T1._dense_gemm_ast(DEVICE), "dense_fp16")
  print("fused meta:", fused_meta)
  print("dense meta:", dense_meta)

  fused_ll = compile_and_disassemble(fused_metal, "fused_q4k")
  dense_ll = compile_and_disassemble(dense_metal, "dense_fp16")

  fused_blocks, fused_entry = parse_kernel_function(fused_ll, "r_64_1536_32_2_512_b410fbbf")
  dense_blocks, dense_entry = parse_kernel_function(dense_ll, "r_64_1536_32_2_512_d97deabd")

  fused_header, fused_loop = find_loop_blocks(fused_blocks, fused_entry)
  dense_header, dense_loop = find_loop_blocks(dense_blocks, dense_entry)
  print(f"\nfused: entry_label={fused_entry} header={fused_header} loop_blocks={sorted(fused_loop)} "
        f"(n_blocks={len(fused_loop)})")
  print(f"dense: entry_label={dense_entry} header={dense_header} loop_blocks={sorted(dense_loop)} "
        f"(n_blocks={len(dense_loop)})")

  fused_counts, fused_results, fused_packed = classify_kernel(fused_blocks, fused_loop, is_fused=True)
  dense_counts, dense_results, dense_packed = classify_kernel(dense_blocks, dense_loop, is_fused=False)
  fused_total = sum(fused_counts.values())
  dense_total = sum(dense_counts.values())

  emit_table("fused Q4_K (per K-loop iteration, static instr count across all loop-body blocks)",
             fused_counts, fused_total)
  emit_table("dense fp16 (per K-loop iteration, static instr count, single block, no branching)",
             dense_counts, dense_total)

  print(f"\nfused: n_packed_derived_values(within loop)={len(fused_packed)}")
  print(f"dense: n_packed_derived_values(within loop)={len(dense_packed)}  (expect 0: no packed buffer)")

  print("\n=== side-by-side ===")
  print(f"{'group':<20}{'fused':>10}{'fused%':>9}{'dense':>10}{'dense%':>9}")
  for g in GROUPS:
    fc, dc = fused_counts.get(g, 0), dense_counts.get(g, 0)
    print(f"{g:<20}{fc:>10}{fc/fused_total*100:>8.1f}%{dc:>10}{dc/dense_total*100:>8.1f}%")
  print(f"{'TOTAL':<20}{fused_total:>10}{100.0:>8.1f}%{dense_total:>10}{100.0:>8.1f}%")
  print(f"ratio fused_total/dense_total = {fused_total/dense_total:.2f}x")

  # unclassified dump for honesty
  fused_unclass = [(l, r, t) for l, r, t, g, _ in fused_results if g == "unclassified"]
  dense_unclass = [(l, r, t) for l, r, t, g, _ in dense_results if g == "unclassified"]
  print(f"\nfused unclassified ({len(fused_unclass)}): {fused_unclass}")
  print(f"dense unclassified ({len(dense_unclass)}): {dense_unclass}")

  fused_dyn_total, fused_dyn_meta, fused_dyn_counts = dynamic_per_iteration_estimate(
      fused_blocks, fused_loop, fused_results)
  print(f"\n=== dynamic per-iteration expectation (fused only; dense has no branches, dynamic==static) ===")
  print(f"diamonds found: {fused_dyn_meta['diamonds']}  straight_blocks: {fused_dyn_meta['straight_blocks']}")
  print(f"fused dynamic-average per-iteration instruction count = {fused_dyn_total:.1f}  "
        f"(static was {fused_total})")
  print(f"fused_dynamic / dense_static ratio = {fused_dyn_total/dense_total:.2f}x   "
        f"(measured GFLOPS ratio this session: 2733.15/544.27 = {2733.15/544.27:.2f}x)")
  print(f"{'group':<20}{'fused(dyn avg)':>16}{'share':>9}")
  for g in GROUPS:
    c = fused_dyn_counts.get(g, 0.0)
    print(f"{g:<20}{c:>16.1f}{c/fused_dyn_total*100:>8.1f}%")

  # step 6: lane redundancy audit
  audit = lane_redundancy_audit(fused_metal)
  print("\n=== lane-redundancy audit (fused kernel, packed-word loads, local_size assumed 32) ===")
  for row in audit["per_load"]:
    print(f"  {row['value']:<28} bits_used={row['lidx0_bits_used']} "
          f"n_bits={row['n_bits_used']}  redundant_lanes_per_distinct_address={row['redundant_lanes_per_group']}")

  out = {
    "fused_meta": fused_meta, "dense_meta": dense_meta,
    "fused_loop_blocks": sorted(fused_loop), "dense_loop_blocks": sorted(dense_loop),
    "fused_counts": dict(fused_counts), "dense_counts": dict(dense_counts),
    "fused_total": fused_total, "dense_total": dense_total,
    "fused_unclassified": fused_unclass, "dense_unclassified": dense_unclass,
    "fused_dynamic_avg_total": fused_dyn_total, "fused_dynamic_avg_counts": dict(fused_dyn_counts),
    "fused_diamonds": fused_dyn_meta["diamonds"],
    "lane_redundancy_audit": audit,
  }
  with open("/tmp/t5_result.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
  print("\nwrote /tmp/t5_result.json")


if __name__ == "__main__":
  main()
