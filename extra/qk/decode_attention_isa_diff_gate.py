#!/usr/bin/env python3
"""Owned-vs-generated ISA diff gate: pin where the generated decode tile bleeds vs the owned AMDGCN tile.

Builds on extra/qk/isa_helpers.py (reuses _disasm/_hist/_parse_desc)
and mirrors bench/qk-prefill-schedule-diff-oracle/static_isa_diff.json's key_diff format.

Captures owned_flash_tile_gqa_whole (baseline) and flash_fused_xlane_score_pv_tile_whole_cache_32_128
(DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE) via the runtime hook, disassembles both, and reports a normalized
instruction/resource diff + a key_diff narrative. Static ISA is ctx-independent.

Run: DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk/decode_attention_isa_diff_gate.py
Scope: docs/decode-isa-diff-gate-scope.md
"""
from __future__ import annotations
from extra.qk.isa_helpers import CROSS_LANE_RE
import json, os, pathlib, re, subprocess, sys, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-decode-attention-isa-diff"
MAXC = 4608
CTX = int(os.environ.get("QK_ISA_DIFF_CTX", "1024"))

TILES = {
  "owned": {"tile": "owned_flash_tile_gqa_whole", "env": {}},
  "xlane": {"tile": "flash_fused_xlane_score_pv_tile_whole_cache_32_128",
            "env": {"DECODE_ATTN_GENERATED_WHOLECACHE": "1", "DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE": "1",
                    "V_DOT2_LOWERING": "1"}},
}
_ZERO = ("DECODE_ATTN_GENERATED_SKELETON", "DECODE_ATTN_GENERATED_WHOLECACHE", "DECODE_ATTN_SCORE_VDOT2",
         "DECODE_ATTN_SCORE_XLANE", "DECODE_ATTN_FUSED_PV_TILE", "DECODE_ATTN_FUSED_SCORE_STATE_PV_TILE",
         "DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE", "DECODE_ATTN_PHYSICAL_TILE_PALL_LIFECYCLE",
         "DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE", "DECODE_ATTN_ONLINE_STATE_PV_TILE_XLANE")


def _markers(asm: str) -> dict[str, int]:
  def c(pat: str) -> int: return len(re.findall(pat, asm))
  return {
    "global_load_dwordx4": c(r"\bglobal_load_dwordx4\b"), "global_load_dwordx2": c(r"\bglobal_load_dwordx2\b"),
    "global_load_dword": c(r"\bglobal_load_dword\b"), "global_load_d16": c(r"\bglobal_load_(u?short|d16)"),
    "ds_read": c(r"\bds_(read|load)"), "ds_write": c(r"\bds_(write|store)"), "s_barrier": c(r"\bs_barrier\b"),
    "s_waitcnt": c(r"\bs_waitcnt\b"), "v_dot2": c(r"\bv_dot2"),
    "cross_lane": c(CROSS_LANE_RE), "scratch": c(r"\bscratch_(load|store)"),
    "v_fma": c(r"\bv_fma"), "v_exp": c(r"\bv_exp"), "v_mul": c(r"\bv_mul"), "v_add": c(r"\bv_add"),
  }


# attention kernels to time per arm (tile is also disassembled)
_ROUTE_KERNELS = {
  "owned": ("owned_flash_tile_gqa_whole", "owned_flash_combine"),
  "xlane": ("flash_fused_xlane_score_pv_tile_whole_cache_32_128", "flash_state_gmax_32_128", "flash_state_combine_32_128"),
}


def _capture(arm: str) -> dict[str, Any]:
  import contextlib, io
  from collections import Counter
  from tinygrad import Tensor, UOp, TinyJit, Context, GlobalCounters, Device
  from extra.llm.generate import load_model_and_tokenizer
  from extra.qk.harness_contract import DEFAULT_MODEL
  from extra.qk.isa_helpers import _disasm, _hist, _parse_desc, _parse_debug, ANSI
  tile = TILES[arm]["tile"]
  dev = Device[Device.DEFAULT]
  captured: dict[str, bytes] = {}
  orig = dev.runtime
  def hook(name, lib, **kw):
    if name.startswith(tile) and name not in captured: captured[name] = lib
    return orig(name, lib, **kw)
  dev.runtime = hook
  m, _tok = load_model_and_tokenizer(os.environ.get("QK_MODEL", DEFAULT_MODEL), MAXC, seed=20260617)
  q4k = getattr(m, "_q4k_linears", None)
  for lin in (q4k.linears if q4k else []): lin.decode_enabled = True
  for b in m.blk: b._use_flash, b._prefill_v2 = True, False
  v = UOp.variable("start_pos", 0, MAXC - 1); temp = Tensor([0.0]); step = TinyJit(m.forward)
  tk = Tensor([[100]], dtype="int32").contiguous()
  for _ in range(8): step(tk, v.bind(CTX), temp).realize().item()
  # one DEBUG=2 eager step for per-program GPU ms
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), Context(DEBUG=2):
    GlobalCounters.reset(); step(tk, v.bind(CTX + 1), temp).realize()
  lines = [ANSI.sub("", l) for l in buf.getvalue().splitlines() if "***" in l]
  prog_ms = _parse_debug(lines).get("program_ms", {})
  attn_ms = {k: round(sum(vv for kk, vv in prog_ms.items() if kk.startswith(k)), 4) for k in _ROUTE_KERNELS.get(arm, ())}
  res: dict[str, Any] = {"arm": arm, "tile_name_prefix": tile, "captured": list(captured.keys()),
                         "attn_program_ms": attn_ms, "attn_total_ms": round(sum(attn_ms.values()), 4)}
  for name, lib in captured.items():
    asm = _disasm(lib)
    try: desc = _parse_desc(lib)
    except Exception as e: desc = {"error": str(e)}
    res["tile"] = {"name": name, "resources": desc, "hist": _hist(asm), "markers": _markers(asm),
                   "asm_lines": len(asm.splitlines())}
    (OUT / f"disasm_{name}.txt").write_text(asm)
    break
  return res


def _run_child(arm: str) -> dict[str, Any]:
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_ISA_DIFF_CHILD": "1", "QK_ISA_DIFF_ARM": arm}
  for k in _ZERO: env[k] = "0"
  env.update(TILES[arm]["env"])
  p = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve())], cwd=ROOT, env=env, text=True,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  if p.returncode != 0: return {"arm": arm, "failed": True, "returncode": p.returncode, "output_tail": (p.stdout or "")[-8000:]}
  for line in reversed((p.stdout or "").splitlines()):
    try: return json.loads(line)
    except Exception: pass
  return {"arm": arm, "failed": True, "error": "no json", "output_tail": (p.stdout or "")[-8000:]}


def _diff(owned: dict[str, Any], xlane: dict[str, Any]) -> dict[str, Any]:
  oh, xh = owned["hist"], xlane["hist"]
  om, xm = owned["markers"], xlane["markers"]
  def row(o, x): return {"owned": o, "xlane": x, "delta": x - o, "ratio": round(x / o, 2) if o else (None if x == 0 else float("inf"))}
  hist = {k: row(oh.get(k, 0), xh.get(k, 0)) for k in sorted(set(oh) | set(xh))}
  markers = {k: row(om.get(k, 0), xm.get(k, 0)) for k in sorted(set(om) | set(xm))}
  ores, xres = owned.get("resources", {}), xlane.get("resources", {})
  olds, xlds = ores.get("lds", 0) or 0, xres.get("lds", 0) or 0
  bleeders = []
  # owned block-stages a K tile in LDS and processes a block; generated stages per-token
  if olds >= 4 * max(1, xlds) and olds >= 1024:
    bleeders.append(f"per-token (not block) processing: owned LDS {olds}B (K-block staged) vs xlane {xlds}B (~1 token) -> owned amortizes loads/reduction over a token block, xlane is per-token latency-bound")
  # owned vectorizes K/V loads (fp16 d16/wide); generated loads scalar fp32
  if om.get("global_load_d16", 0) >= 4 * max(1, xm.get("global_load_d16", 0)) and xm.get("global_load_d16", 0) == 0:
    bleeders.append(f"uncoalesced loads: owned global_load_d16 {om['global_load_d16']} (fp16 vectorized) vs xlane 0 (V cast to fp32, scalar)")
  # xlane emits cross-lane in the hot path that owned avoids
  if xm.get("cross_lane", 0) > max(2, 2 * om.get("cross_lane", 0)):
    bleeders.append(f"cross-lane reduction in hot path: xlane {xm['cross_lane']} vs owned {om.get('cross_lane',0)} (per-token warp_reduce not amortized)")
  if xm.get("s_barrier", 0) > max(2, 2 * om.get("s_barrier", 0)):
    bleeders.append(f"LDS barriers: xlane {xm['s_barrier']} vs owned {om.get('s_barrier',0)} (per-token K staging barrier)")
  if xm.get("global_load_dwordx4", 0) == 0 and (xm.get("global_load_dword", 0) > 0 or xm.get("global_load_d16", 0) > 0) and om.get("global_load_dwordx4", 0) > 0:
    bleeders.append(f"uncoalesced loads: owned uses dwordx4 ({om['global_load_dwordx4']}), xlane uses scalar/d16 (dword={xm.get('global_load_dword',0)}, d16={xm.get('global_load_d16',0)})")
  if xm.get("scratch", 0) > 0 and om.get("scratch", 0) == 0:
    bleeders.append(f"register spill: xlane scratch_ {xm['scratch']} vs owned 0 (acc/recurrence spilled)")
  if xh.get("valu", 0) > max(50, 2 * oh.get("valu", 0)):
    bleeders.append(f"valu bloat: xlane {xh['valu']} vs owned {oh.get('valu',0)} ({row(oh.get('valu',0),xh.get('valu',0))['ratio']}x)")
  if xm.get("s_waitcnt", 0) > max(4, 2 * om.get("s_waitcnt", 0)):
    bleeders.append(f"memory stalls: xlane s_waitcnt {xm['s_waitcnt']} vs owned {om.get('s_waitcnt',0)} (poor latency hiding)")
  return {"hist": hist, "markers": markers, "bleeders": bleeders}


def build() -> dict[str, Any]:
  owned = _run_child("owned")
  xlane = _run_child("xlane")
  if owned.get("failed") or xlane.get("failed") or "tile" not in owned or "tile" not in xlane:
    return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"), "verdict": "ISA_DIFF_FAIL__CAPTURE",
            "owned": owned, "xlane": xlane}
  ot, xt = owned["tile"], xlane["tile"]
  diff = _diff(ot, xt)
  timing = {"owned_attn_program_ms": owned.get("attn_program_ms"), "owned_attn_total_ms": owned.get("attn_total_ms"),
            "xlane_attn_program_ms": xlane.get("attn_program_ms"), "xlane_attn_total_ms": xlane.get("attn_total_ms"),
            "xlane_vs_owned_attn_ratio": (round(xlane.get("attn_total_ms", 0) / owned["attn_total_ms"], 1)
                                          if owned.get("attn_total_ms") else None)}
  # which xlane kernel dominates the attention time
  xms = xlane.get("attn_program_ms") or {}
  timing["xlane_dominant_kernel"] = max(xms, key=xms.get) if xms else None
  verdict = "ISA_DIFF_PINNED" if diff["bleeders"] else "ISA_DIFF_INCONCLUSIVE"
  key_diff = (" | ".join(diff["bleeders"]) if diff["bleeders"] else
              "no dominant static-ISA bleeder; the gap may be dynamic (occupancy/scheduling) -> SQTT/profile next")
  return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"), "verdict": verdict, "ctx": CTX,
          "owned_tile": {"name": ot["name"], "resources": ot["resources"], "hist": ot["hist"], "markers": ot["markers"], "asm_lines": ot["asm_lines"]},
          "xlane_tile": {"name": xt["name"], "resources": xt["resources"], "hist": xt["hist"], "markers": xt["markers"], "asm_lines": xt["asm_lines"]},
          "diff": diff, "timing": timing, "key_diff": key_diff,
          "decision": "Target the top bleeder with a renderer/lowering change, then re-diff here. Do NOT write another attention layout."}


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  if os.environ.get("QK_ISA_DIFF_CHILD") == "1":
    print(json.dumps(_capture(os.environ.get("QK_ISA_DIFF_ARM", "owned"))))
    return 0
  out = build()
  (OUT / "latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"isa-diff-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] == "ISA_DIFF_PINNED" else 1


if __name__ == "__main__":
  raise SystemExit(main())
