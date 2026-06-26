#!/usr/bin/env python3
"""Decode attention purity capture.

Captures the current decode-attention lifecycle route and classifies whether the
attention path is generated/search-owned or still backed by owned AMDGCN tile +
combine programs.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-attention-purity"
A1_OUT = ROOT / "bench/qk-decode-attention-generated-skeleton"


def _program_names(captured) -> list[str]:
  from tinygrad.uop.ops import Ops
  if captured is None: return []
  return [str(getattr(u.arg, "name", "")) for u in captured.linear.toposort() if u.op is Ops.PROGRAM]


def capture(mode: str="baseline") -> dict[str, Any]:
  from extra.qk_decode_search_gate import _setup_model, capture_decode, check_route_fire, check_materialization

  os.environ.setdefault("DECODE_ATTN_KV_IDENTITY", "1")
  os.environ.setdefault("DECODE_ATTN_AMDGCN_TILE", "1")
  if mode == "a1":
    os.environ["DECODE_ATTN_GENERATED_SKELETON"] = "1"
    os.environ.setdefault("DECODE_ATTN_GENERATED_SKELETON_VARIANT", "gqa_coop_vec")
  else:
    os.environ["DECODE_ATTN_GENERATED_SKELETON"] = "0"
  m, tok = _setup_model()
  toks, captured, _step, _v, _temp = capture_decode(m, tok)
  names = _program_names(captured)
  counts = Counter(names)
  owned_tile = sum(n == "owned_flash_tile_gqa_whole" or n.startswith("owned_flash_tile_gqa_whole") for n in names)
  owned_combine = sum(n.startswith("owned_flash_combine") for n in names)
  generated_attention = sum(n.startswith("flash_") for n in names)
  generated_skeleton_selected = mode == "a1" and os.environ.get("DECODE_ATTN_GENERATED_SKELETON") == "1"
  materialization = check_materialization(captured)
  route_fire = check_route_fire(captured, "owned_flash_tile_gqa_whole")
  selected_route_buffer_identity = not materialization["E_49152_present"] and len(materialization.get("full_maxc_copy_kernels", [])) == 0
  if mode == "a1":
    if materialization["E_49152_present"]:
      verdict = "DECODE_ATTENTION_A1_FAIL__E_49152_REINTRODUCED"
    elif owned_tile > 0:
      verdict = "DECODE_ATTENTION_A1_FAIL__OWNED_TILE_STILL_FIRES"
    elif owned_combine > 0:
      verdict = "DECODE_ATTENTION_A1_PARTIAL__OWNED_COMBINE_REMAINS"
    elif generated_attention > 0:
      verdict = "DECODE_ATTENTION_A1_GENERATED_SKELETON_ROUTE_CLEAN"
    else:
      verdict = "DECODE_ATTENTION_A1_FAIL__GENERATED_ROUTE_NOT_CAPTURED"
  elif owned_tile > 0 and owned_combine > 0 and not materialization["E_49152_present"]:
    verdict = "DECODE_ATTENTION_NOT_PURE__OWNED_TILE_COMBINE"
  elif generated_attention > 0 and owned_tile == 0 and owned_combine == 0 and not materialization["E_49152_present"]:
    verdict = "DECODE_ATTENTION_PURE_SEARCH_GENERATED"
  else:
    verdict = "DECODE_ATTENTION_PURITY_CAPTURE_FAIL"
  return {
    "date": "2026-06-25",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "mode": mode,
    "verdict": verdict,
    "selected_candidate": "decode_attention_generated_skeleton" if generated_skeleton_selected else "owned_flash_tile_gqa_whole",
    "tokens_sample": toks,
    "route_counts": {
      "owned_flash_tile_gqa_whole": owned_tile,
      "owned_flash_combine": owned_combine,
      "generated_attention_programs": generated_attention,
    },
    "route_fire": route_fire,
    "materialization": {**materialization, "selected_route_buffer_identity": selected_route_buffer_identity},
    "top_program_counts": counts.most_common(40),
    "classification": ("A1 generated skeleton route capture." if mode == "a1" else
      "Decode attention is still backed by owned AMDGCN tile + combine programs; GEMV purity is complete, attention purity is next."),
  }


def _run_child(mode: str) -> dict[str, Any]:
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_DECODE_ATTENTION_CAPTURE_CHILD": "1",
         "QK_DECODE_ATTENTION_CAPTURE_MODE": mode}
  if mode == "a1":
    env["DECODE_ATTN_GENERATED_SKELETON"] = "1"
    env.setdefault("DECODE_ATTN_GENERATED_SKELETON_VARIANT", "gqa_coop_vec")
  else:
    env["DECODE_ATTN_GENERATED_SKELETON"] = "0"
  r = subprocess.run([sys.executable, str(Path(__file__).resolve())], cwd=ROOT, env=env,
                     capture_output=True, text=True)
  if r.returncode != 0:
    raise RuntimeError(f"{mode} capture failed rc={r.returncode}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
  for line in reversed(r.stdout.strip().splitlines()):
    try: return json.loads(line)
    except Exception: pass
  raise RuntimeError(f"{mode} capture did not emit JSON\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")


def capture_a1_gate() -> dict[str, Any]:
  baseline = _run_child("baseline")
  a1 = _run_child("a1")
  token_match = baseline["tokens_sample"] == a1["tokens_sample"]
  if not token_match:
    verdict = "DECODE_ATTENTION_A1_FAIL__TOKEN_MISMATCH"
  else:
    verdict = a1["verdict"]
  return {
    "date": "2026-06-25",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "search_space_id": "decode_attention_gfx1100_v1",
    "candidate": {
      "id": "decode_attention_generated_skeleton",
      "search_generation_status": "generated_skeleton",
      "promotion_status": "attribution_only",
      "blocked_primitives": ["v_dot2", "cross_lane_reduction", "lds_staged_tile_layout", "tile_combine_lifecycle"],
    },
    "token_byte_identical_to_baseline": token_match,
    "baseline": baseline,
    "a1": a1,
    "decision": ("A1 route-clean generated skeleton captured; speed work remains blocked on missing primitives."
                 if verdict == "DECODE_ATTENTION_A1_GENERATED_SKELETON_ROUTE_CLEAN" else
                 "A1 did not reach route-clean generated skeleton; use verdict to classify the next blocker."),
  }


def main() -> int:
  if os.environ.get("QK_DECODE_ATTENTION_CAPTURE_CHILD") == "1":
    print(json.dumps(capture(os.environ.get("QK_DECODE_ATTENTION_CAPTURE_MODE", "baseline"))))
    return 0
  ap = argparse.ArgumentParser()
  ap.add_argument("--a1", action="store_true", help="run baseline + generated-skeleton A1 gate")
  args = ap.parse_args()
  os.chdir(ROOT)
  if args.a1:
    A1_OUT.mkdir(parents=True, exist_ok=True)
    out = capture_a1_gate()
    latest = A1_OUT / "latest.json"
    stamped = A1_OUT / f"decode-attention-generated-skeleton-{out['timestamp']}.json"
  else:
    OUT.mkdir(parents=True, exist_ok=True)
    out = capture()
    latest = OUT / "latest.json"
    stamped = OUT / f"decode-attention-purity-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  fail_verdicts = ("DECODE_ATTENTION_PURITY_CAPTURE_FAIL", "DECODE_ATTENTION_A1_FAIL__TOKEN_MISMATCH",
                   "DECODE_ATTENTION_A1_FAIL__E_49152_REINTRODUCED", "DECODE_ATTENTION_A1_FAIL__OWNED_TILE_STILL_FIRES",
                   "DECODE_ATTENTION_A1_FAIL__GENERATED_ROUTE_NOT_CAPTURED")
  return 0 if out["verdict"] not in fail_verdicts else 1


if __name__ == "__main__":
  raise SystemExit(main())
