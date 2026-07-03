#!/usr/bin/env python3
"""Split-aware hot-loop schedule-diff audit tool for decode attention.

This is the measure-first instrument for pure-search delta closure. It compares owned + generated decode tile
ISA, enumerates backward-branch loop candidates, classifies inner/outer loops, and reports:
  - waitcnt drains,
  - latency-shadow fill after long-latency ops,
  - compute/memory/cross-lane mix,
  - loop-candidate metadata so split experiments can tell whether they changed the inner tt loop or outer b loop.

Default input = disasm files emitted by the ISA gates; pass paths to compare any two.

Run: PYTHONPATH=. python3 extra/qk/decode_hotloop_schedule_diff.py [owned.txt] [generated.txt]
"""
from __future__ import annotations
import re, sys, json, pathlib, statistics
from dataclasses import dataclass, asdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
OWNED = ROOT / "bench/qk-decode-attention-isa-diff/disasm_owned_flash_tile_gqa_whole.txt"
GEN = ROOT / "bench/qk-decode-isa-vectorization/disasm_flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128.txt"

_INSN = re.compile(r"^\s*([a-z][a-z0-9_]+)\b(.*?)\s*//\s*([0-9A-Fa-f]+):")
_TARGET = re.compile(r"<[^+>]+\+0x([0-9A-Fa-f]+)>")
_BRANCH_IMM = re.compile(r"\b(\d+)\b")

@dataclass
class Insn:
  line: int
  op: str
  operands: str
  addr: int
  target_off: int | None = None
  off: int | None = None

LATENCY_KEYS = ("ds_bpermute", "global_load", "ds_read")

def _signed16(x: int) -> int:
  return x - 65536 if x >= 32768 else x

def _parse_insns(asm: str) -> list[Insn]:
  out: list[Insn] = []
  for line_no, ln in enumerate(asm.splitlines(), 1):
    m = _INSN.match(ln)
    if not m: continue
    op, operands, addr_s = m.group(1), m.group(2).strip(), m.group(3)
    tm = _TARGET.search(ln)
    out.append(Insn(line_no, op, operands, int(addr_s, 16), int(tm.group(1), 16) if tm else None))
  _infer_offsets(out)
  return out

def _infer_offsets(insns: list[Insn]) -> None:
  """Infer function-relative offsets from branch comments and immediates.

  AMD branch comments include the target as <kernel+0x...> but each instruction line only has the absolute code
  address. For a branch with immediate N, target = current + 4 + signed16(N)*4. That gives us current offset and
  therefore the code-object base. Once one base is known, every instruction gets a relative offset.
  """
  bases: list[int] = []
  for ins in insns:
    if ins.target_off is None or not (ins.op.startswith("s_cbranch") or ins.op == "s_branch"): continue
    im = _BRANCH_IMM.search(ins.operands)
    if not im: continue
    cur_off = ins.target_off - 4 - _signed16(int(im.group(1))) * 4
    bases.append(ins.addr - cur_off)
  if not bases: return
  base = int(statistics.median(bases))
  for ins in insns: ins.off = ins.addr - base

def _target_index(insns: list[Insn], target_off: int) -> int | None:
  best_i, best_d = None, 1 << 60
  for i, ins in enumerate(insns):
    if ins.off is None: continue
    d = abs(ins.off - target_off)
    if d < best_d:
      best_i, best_d = i, d
  return best_i if best_d <= 8 else None

def _loop_candidates(insns: list[Insn]) -> list[dict]:
  cands = []
  for j, ins in enumerate(insns):
    if not (ins.op.startswith("s_cbranch") or ins.op == "s_branch") or ins.target_off is None: continue
    if ins.off is not None and ins.target_off >= ins.off: continue
    i = _target_index(insns, ins.target_off)
    if i is None or i >= j: continue
    body = insns[i:j]
    metrics = _metrics(body)
    score = (metrics["mix"]["ds_bpermute"] * 20 + metrics["mix"]["global_load"] * 8 +
             metrics["mix"]["ds_read"] * 8 + metrics["mix"]["s_waitcnt"] * 4 + len(body) * 0.1)
    cands.append({
      "start_index": i, "end_index": j, "start_line": insns[i].line, "end_line": ins.line,
      "start_off": hex(insns[i].off) if insns[i].off is not None else None,
      "branch_off": hex(ins.off) if ins.off is not None else None,
      "target_off": hex(ins.target_off), "branch_op": ins.op,
      "loop_body_insns": len(body), "score": round(score, 2), "metrics": metrics,
    })
  cands.sort(key=lambda x: x["score"], reverse=True)
  # annotate nesting after sort-independent spans
  spans = [(c["start_index"], c["end_index"]) for c in cands]
  for c in cands:
    containing = sum(1 for a, b in spans if a <= c["start_index"] and c["end_index"] <= b and (a, b) != (c["start_index"], c["end_index"]))
    c["nesting_depth"] = containing
    c["loop_class"] = _classify_loop(c, cands)
  return cands

def _classify_loop(c: dict, cands: list[dict]) -> str:
  mix = c["metrics"]["mix"]
  if c["nesting_depth"] == 0 and c["loop_body_insns"] > 250: return "outer_b_or_main_ctx_loop"
  if mix["ds_bpermute"] >= 4 and c["loop_body_insns"] < 250: return "inner_reduce_or_tt_loop"
  if mix["global_load"] >= 4 and c["loop_body_insns"] >= 80: return "load_stage_loop"
  return "control_or_tail_loop"

def _latency_key(op: str) -> str | None:
  if op.startswith("ds_bpermute"): return "ds_bpermute"
  if op.startswith("global_load"): return "global_load"
  if op.startswith("ds_read") or op.startswith("ds_load"): return "ds_read"
  return None

def _is_work(op: str) -> bool:
  return not (op.startswith("s_") or op.startswith(".p2align"))

def _shadow_fill(body: list[Insn]) -> dict[str, list[int]]:
  fills: dict[str, list[int]] = {k: [] for k in LATENCY_KEYS}
  for i, ins in enumerate(body):
    key = _latency_key(ins.op)
    if key is None: continue
    fill = 0
    for nxt in body[i+1:]:
      if nxt.op.startswith("s_waitcnt"): break
      if _is_work(nxt.op): fill += 1
    fills[key].append(fill)
  return fills

def _metrics(body: list[Insn]) -> dict:
  def c(pred): return sum(1 for ins in body if pred(ins.op))
  fills = _shadow_fill(body)
  def avg(xs): return round(sum(xs) / len(xs), 2) if xs else None
  return {
    "drains": {
      "lgkmcnt0": sum(1 for ins in body if ins.op == "s_waitcnt" and "lgkmcnt(0)" in ins.operands),
      "lgkmcnt_any": sum(1 for ins in body if ins.op == "s_waitcnt" and "lgkmcnt" in ins.operands),
      "vmcnt": sum(1 for ins in body if ins.op == "s_waitcnt" and "vmcnt" in ins.operands),
    },
    "mix": {
      "ds_bpermute": c(lambda o: o.startswith("ds_bpermute")),
      "global_load": c(lambda o: o.startswith("global_load")),
      "ds_read": c(lambda o: o.startswith("ds_read") or o.startswith("ds_load")),
      "ds_write": c(lambda o: o.startswith("ds_write") or o.startswith("ds_store")),
      "v_fma": c(lambda o: o.startswith("v_fma") or "dot2" in o),
      "v_alu": c(lambda o: o.startswith("v_") and not o.startswith("v_fma")),
      "s_waitcnt": c(lambda o: o.startswith("s_waitcnt")),
      "s_barrier": c(lambda o: o == "s_barrier"),
      "branch": c(lambda o: o.startswith("s_cbranch") or o == "s_branch"),
    },
    "latency_shadow_fill_avg": {k: avg(v) for k, v in fills.items()},
    "latency_shadow_fill_raw": fills,
  }

def _pick_hot(cands: list[dict]) -> dict | None:
  return cands[0] if cands else None

def analyze(path: pathlib.Path) -> dict:
  insns = _parse_insns(path.read_text())
  cands = _loop_candidates(insns)
  hot = _pick_hot(cands)
  return {
    "file": path.name,
    "instruction_count": len(insns),
    "loop_candidates_count": len(cands),
    "selected_loop": hot,
    "loop_candidates_top": cands[:8],
    "split_aware": True,
  }

def _first_metric(x: dict, path: list[str], default=None):
  for p in path:
    if not isinstance(x, dict): return default
    x = x.get(p)
  return x if x is not None else default

def main() -> int:
  owned_p = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else OWNED
  gen_p = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else GEN
  out = {"owned": analyze(owned_p), "generated": analyze(gen_p)}
  o, g = out["owned"].get("selected_loop") or {}, out["generated"].get("selected_loop") or {}
  om, gm = o.get("metrics", {}), g.get("metrics", {})
  ob = _first_metric(om, ["mix", "ds_bpermute"], 0); gb = _first_metric(gm, ["mix", "ds_bpermute"], 0)
  og = _first_metric(om, ["mix", "global_load"], 0); gg = _first_metric(gm, ["mix", "global_load"], 0)
  osw = _first_metric(om, ["mix", "s_waitcnt"], 0); gsw = _first_metric(gm, ["mix", "s_waitcnt"], 0)
  ofill = _first_metric(om, ["latency_shadow_fill_avg", "ds_bpermute"])
  gfill = _first_metric(gm, ["latency_shadow_fill_avg", "ds_bpermute"])
  selected_ok = gb > 0 or gg > 0
  if selected_ok and gb > max(ob * 2, ob + 8): verdict = "HOTLOOP_SCHEDULE_DIFF__GENERATED_CROSSLANE_OVERHEAD_BOUND"
  elif selected_ok and gsw > max(osw * 2, osw + 8): verdict = "HOTLOOP_SCHEDULE_DIFF__GENERATED_WAITCNT_BOUND"
  elif selected_ok and gfill is not None and ofill is not None and gfill < ofill - 0.5: verdict = "HOTLOOP_SCHEDULE_DIFF__SCHEDULING_BOUND"
  elif selected_ok: verdict = "HOTLOOP_SCHEDULE_DIFF__SPLIT_AWARE_PARITY_OR_STRUCTURAL"
  else: verdict = "HOTLOOP_SCHEDULE_DIFF__INCONCLUSIVE_NO_LONG_LATENCY_LOOP_SELECTED"
  out["comparison"] = {
    "selected_loop_valid": selected_ok,
    "owned_selected_class": o.get("loop_class"),
    "generated_selected_class": g.get("loop_class"),
    "ds_bpermute_owned_vs_generated": [ob, gb],
    "global_load_owned_vs_generated": [og, gg],
    "s_waitcnt_owned_vs_generated": [osw, gsw],
    "ds_bpermute_shadow_fill_owned_vs_generated": [ofill, gfill],
  }
  out["verdict"] = verdict
  OUTDIR = ROOT / "bench/qk-decode-hotloop-schedule-diff"; OUTDIR.mkdir(parents=True, exist_ok=True)
  (OUTDIR / "latest.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
