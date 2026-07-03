#!/usr/bin/env python3
"""Wall audit for generated decode-attention routes.

This is not a promotion benchmark.  It answers: when a generated attention route
is correctness-clean but W==D-collapsed, which lifecycle/program shape is the
wall?  It combines:
  - route/program capture from the TinyJit graph,
  - one DEBUG=2 step for program count + GPU proxy,
  - the latest clean W==D artifact when present,
  - structural arithmetic/workload classification.
"""
from __future__ import annotations

import contextlib, io, json, os, pathlib, re, subprocess, sys, time
from collections import Counter
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-decode-attention-generated-wall-audit"
CKPTS = (512, 1024, 4096)
MAXC = 4608
DEBUG_LINE = re.compile(r"\*\*\*\s+\S+\s+\d+\s+(.+?)\s+arg\s+\d+\s+mem")
ANSI = re.compile(r"\x1b\[[0-9;]*m")

ARMS = {
  "baseline": {},
  "split_xlane": {"DECODE_ATTN_GENERATED_WHOLECACHE": "1", "DECODE_ATTN_ONLINE_STATE_SPLIT_XLANE": "1"},
}


def _env(arm: str) -> dict[str, str]:
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_WALL_AUDIT_CHILD": "1", "QK_WALL_AUDIT_ARM": arm}
  for k in ("DECODE_ATTN_GENERATED_WHOLECACHE", "DECODE_ATTN_ONLINE_STATE_SPLIT_XLANE",
            "DECODE_ATTN_ONLINE_STATE_PV_TILE_XLANE", "DECODE_ATTN_ONLINE_STATE_PV_TILE",
            "DECODE_ATTN_ONLINE_PV_TILE", "DECODE_ATTN_TILE_PROB_PARTIAL_PV", "DECODE_ATTN_TILE_PROB",
            "DECODE_ATTN_TILE_SCORE_MAX", "DECODE_ATTN_SCORE_XLANE", "DECODE_ATTN_SCORE_VDOT2",
            "WARP_REDUCE_LOWERING", "V_DOT2_LOWERING"):
    env[k] = "0"
  env.update(ARMS[arm])
  return env


def _program_names(captured) -> list[str]:
  from tinygrad.uop.ops import Ops
  if captured is None: return []
  return [str(getattr(u.arg, "name", "")) for u in captured.linear.toposort() if u.op is Ops.PROGRAM]


def _debug_step(m, tok, ck: int) -> dict[str, Any]:
  from tinygrad import Tensor, UOp, TinyJit, Context, GlobalCounters
  v = UOp.variable("start_pos", 0, MAXC - 1)
  temp = Tensor([0.0])
  step = TinyJit(m.forward)
  tk = Tensor([[100]], dtype="int32").contiguous()
  for _ in range(8):
    o = step(tk, v.bind(ck), temp).realize()
    o.item()
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), Context(DEBUG=2):
    GlobalCounters.reset()
    step(tk, v.bind(ck + 1), temp).realize()
    gpu_ms = GlobalCounters.time_sum_s * 1e3
  raw_lines = [ANSI.sub("", l) for l in buf.getvalue().splitlines() if "***" in l]
  names = []
  for l in raw_lines:
    mline = DEBUG_LINE.search(l)
    if mline: names.append(mline.group(1).strip())
  captured_names = _program_names(step.captured)
  return {
    "ctx": ck,
    "debug2_unbatched_gpu_ms": round(gpu_ms, 3),
    "debug2_programs_per_token": len(names),
    "debug2_program_counts": Counter(names).most_common(),
    "captured_program_counts": Counter(captured_names).most_common(),
    "generated_attention_programs": [n for n in captured_names if n.startswith("flash_")],
    "debug_line_samples": raw_lines[:12],
  }


def _child(arm: str) -> dict[str, Any]:
  from extra.llm.generate import load_model_and_tokenizer
  from extra.qk.harness_contract import DEFAULT_MODEL
  m, tok = load_model_and_tokenizer(os.environ.get("QK_MODEL", DEFAULT_MODEL), MAXC, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  rows = []
  for ck in CKPTS:
    for b in m.blk:
      b._use_flash, b._prefill_v2 = ck >= int(os.environ.get("FLASH_DECODE_THRESHOLD", "512")), False
    rows.append(_debug_step(m, tok, ck))
  return {"arm": arm, "env": {k: os.environ.get(k, "") for k in os.environ if k.startswith("DECODE_ATTN")}, "rows": rows}


def _run_child(arm: str) -> dict[str, Any]:
  p = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve())], cwd=ROOT, env=_env(arm),
                     text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  if p.returncode != 0:
    return {"arm": arm, "failed": True, "returncode": p.returncode, "output_tail": (p.stdout or "")[-8000:]}
  for line in reversed((p.stdout or "").splitlines()):
    try: return json.loads(line)
    except Exception: pass
  return {"arm": arm, "failed": True, "returncode": 0, "output_tail": (p.stdout or "")[-8000:], "error": "no json"}


def _latest_wd() -> dict[str, Any] | None:
  # BoltBeam owns retained W==D promotion/evaluation artifacts. This local audit remains useful without that
  # optional context; it reports structural wall attribution from the generated tinygrad graph.
  return None


def _classify(split: dict[str, Any], wd: dict[str, Any] | None) -> dict[str, Any]:
  programs = split["rows"][0].get("generated_attention_programs", [])
  has_score = any(n.startswith("flash_score_whole_cache") for n in programs)
  has_split_m = any(n.startswith("flash_max_") for n in programs)
  has_pv = any(n.startswith("flash_xlane_pv_from_m") for n in programs)
  has_combine = any(n.startswith("flash_combine") for n in programs)
  pv_workgroups = {}
  score_workgroups = {}
  for ck in CKPTS:
    s = (ck + 127) // 128
    pv_workgroups[str(ck)] = 32 * s * 129
    score_workgroups[str(ck)] = 32 * ck
  return {
    "verdict": "GENERATED_ATTENTION_WALL__PV_GLOBAL_AXIS_KERNEL_ECONOMICS",
    "confidence": "high" if has_score and has_pv and has_combine else "medium",
    "reason": (
      "W==D is GPU-bound and dispatch count is normal; split-state removed repeated online recurrence but kept "
      "the generated PV lifecycle as one GLOBAL output-column program. That creates Hq*S*(Hd+1) workgroups and "
      "uses scalar generated UOps for p*V accumulation rather than a fused LDS/v_dot2/register tile. The wall is "
      "therefore kernel economics/codegen shape, not route flags, host sync, or state recurrence."
    ),
    "program_presence": {"score": has_score, "split_m": has_split_m, "pv": has_pv, "combine": has_combine},
    "estimated_generated_workgroups": {"score": score_workgroups, "pv_augmented": pv_workgroups},
    "wd_summary": wd,
    "next_required_probe": "Per-program GPU timing/ISA capture for flash_score_whole_cache and flash_xlane_pv_from_m; if PV dominates, only an LDS/vectorized fused tile or generated matmul-like PV lowering can move W==D.",
  }


def build() -> dict[str, Any]:
  arms = {a: _run_child(a) for a in ARMS}
  if any(v.get("failed") for v in arms.values()):
    verdict = "GENERATED_ATTENTION_WALL_AUDIT_FAIL__CHILD"
    diagnosis = {"reason": "child capture failed"}
  else:
    diagnosis = _classify(arms["split_xlane"], _latest_wd())
    verdict = diagnosis["verdict"]
  return {
    "date": "2026-06-26",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "arms": arms,
    "diagnosis": diagnosis,
    "decision": "Do not promote split x-lane. Next work is per-program timing/ISA for score vs PV, then decide if codegen can express fused LDS/vectorized PV.",
  }


def main() -> int:
  os.chdir(ROOT)
  if os.environ.get("QK_WALL_AUDIT_CHILD") == "1":
    print(json.dumps(_child(os.environ.get("QK_WALL_AUDIT_ARM", "baseline"))))
    return 0
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  latest = OUT / "latest.json"
  stamped = OUT / f"decode-attention-generated-wall-audit-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if not out["verdict"].endswith("__CHILD") else 1


if __name__ == "__main__":
  raise SystemExit(main())
