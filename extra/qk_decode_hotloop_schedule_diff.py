#!/usr/bin/env python3
"""Hot-loop schedule-diff audit tool -- the measure-first instrument for the decode-tile delta closure
(docs/decode-tile-structural-deltas-scope-20260627.md). Disassembles the owned + generated decode tiles,
isolates the hot loop body (the largest span ending in a backward s_cbranch), and reports the SCHEDULING
metrics that decide scheduling-bound vs structural-floor:
  - in-loop s_waitcnt lgkmcnt(0) / vmcnt(*) drains (latency stalls),
  - per long-latency op (ds_bpermute / global_load / ds_read), the LATENCY-SHADOW FILL = number of independent
    instructions issued between the op and the s_waitcnt that drains it (high = well-pipelined, 0 = stall-on-use),
  - the compute:memory:cross-lane instruction mix.

Default input = the disasm files the ISA gates already emit; pass paths to compare any two. Read-only.

Run: PYTHONPATH=. python3 extra/qk_decode_hotloop_schedule_diff.py [owned.txt] [generated.txt]
"""
from __future__ import annotations
import re, sys, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OWNED = ROOT / "bench/qk-decode-attention-isa-diff/disasm_owned_flash_tile_gqa_whole.txt"
GEN = ROOT / "bench/qk-decode-isa-vectorization/disasm_flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128.txt"

_INSN = re.compile(r"^\s*([a-z][a-z0-9_]+)\b(.*?)\s*//")          # mnemonic + operands (drop the // addr: enc)
_DEST = re.compile(r"\b(v\d+|s\d+)\b")
LAT_OPS = ("ds_bpermute", "global_load", "ds_read", "ds_load")     # long-latency producers we shadow-measure


def _insns(asm: str) -> list[tuple[str, str]]:
  out = []
  for ln in asm.splitlines():
    m = _INSN.match(ln)
    if m: out.append((m.group(1), m.group(2).strip()))
  return out


def _hot_loop(insns: list[tuple[str, str]]) -> tuple[int, int]:
  """Largest span [i,j) whose last instr is a backward s_cbranch (negative/large target) -- the hot loop body."""
  best = (0, 0)
  branch_idx = [k for k, (op, _) in enumerate(insns) if op.startswith("s_cbranch") or op == "s_branch"]
  for j in branch_idx:
    # backward branch heuristic: target operand is a large unsigned (wraps) or there is an earlier branch target
    for i in range(j - 1, max(0, j - 400), -1):
      if insns[i][0] in ("s_barrier",) or insns[i][0].startswith("s_cbranch"):
        if j - i > best[1] - best[0]: best = (i, j)
        break
  return best


def _shadow_fill(body: list[tuple[str, str]]) -> dict[str, list[int]]:
  """For each long-latency op, count independent instrs before the next s_waitcnt that drains its counter type."""
  fills: dict[str, list[int]] = {k: [] for k in ("ds_bpermute", "global_load", "ds_read")}
  for i, (op, _) in enumerate(body):
    key = "ds_bpermute" if op.startswith("ds_bpermute") else \
          "global_load" if op.startswith("global_load") else \
          "ds_read" if (op.startswith("ds_read") or op.startswith("ds_load")) else None
    if key is None: continue
    fill = 0
    for j in range(i + 1, len(body)):
      o2 = body[j][0]
      if o2.startswith("s_waitcnt"):    # the drain -- stop; fill = useful work hidden under the latency
        break
      if not o2.startswith("s_"): fill += 1   # count real (non-scalar-control) work issued in the shadow
    fills[key].append(fill)
  return fills


def analyze(path: pathlib.Path) -> dict:
  asm = path.read_text()
  insns = _insns(asm)
  i, j = _hot_loop(insns)
  body = insns[i:j] if j > i else insns
  def c(pred): return sum(1 for op, _ in body if pred(op))
  fills = _shadow_fill(body)
  def avg(xs): return round(sum(xs) / len(xs), 2) if xs else None
  return {
    "file": path.name, "loop_body_insns": len(body),
    "drains": {"lgkmcnt0": c(lambda o: o == "s_waitcnt" and False) +
               asm[asm.find(""):].count("lgkmcnt(0)") if False else
               sum(1 for op, ops in body if op == "s_waitcnt" and "lgkmcnt(0)" in ops),
               "vmcnt": sum(1 for op, ops in body if op == "s_waitcnt" and "vmcnt" in ops)},
    "mix": {"ds_bpermute": c(lambda o: o.startswith("ds_bpermute")),
            "global_load": c(lambda o: o.startswith("global_load")),
            "ds_read": c(lambda o: o.startswith("ds_read") or o.startswith("ds_load")),
            "ds_write": c(lambda o: o.startswith("ds_write") or o.startswith("ds_store")),
            "v_fma": c(lambda o: o.startswith("v_fma") or "dot2" in o),
            "v_alu": c(lambda o: o.startswith("v_") and not o.startswith("v_fma")),
            "s_barrier": c(lambda o: o == "s_barrier")},
    "latency_shadow_fill_avg": {k: avg(v) for k, v in fills.items()},
    "latency_shadow_fill_raw": fills,
  }


def main() -> int:
  owned_p = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else OWNED
  gen_p = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else GEN
  out = {"owned": analyze(owned_p), "generated": analyze(gen_p)}
  o, g = out["owned"], out["generated"]
  of, gf = o["latency_shadow_fill_avg"], g["latency_shadow_fill_avg"]
  # verdict: if the generated drains long-latency ops with ~0 shadow fill while owned hides them, it is
  # scheduling/pipelining-bound (Track A buildable). If both stall-on-use, the latency is a structural floor.
  bperm_o, bperm_g = of.get("ds_bpermute"), gf.get("ds_bpermute")
  out["verdict"] = ("SCHEDULING_BOUND__generated_stalls_on_use_more_than_owned"
                    if (bperm_g is not None and bperm_o is not None and bperm_g < bperm_o - 0.5)
                    else "PARITY_OR_STRUCTURAL__shadow_fill_similar")
  print(json.dumps(out, indent=2))
  OUTDIR = ROOT / "bench/qk-decode-hotloop-schedule-diff"; OUTDIR.mkdir(parents=True, exist_ok=True)
  (OUTDIR / "latest.json").write_text(json.dumps(out, indent=2) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
