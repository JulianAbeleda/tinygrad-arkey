#!/usr/bin/env python3
"""Gate for the lower-level fused tile lifecycle lowering blocker.

This gate is intentionally below decode attention. It records whether the repo
can lower a nested-reduce + recurrence-state + local-output/metadata-store UOp
shape, and links that to the fused score/state/PV attention blocker.
"""
from __future__ import annotations

import json, pathlib, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-fused-tile-lifecycle-lowering"
ATTN_BLOCKER = ROOT / "bench/qk-decode-attention-fused-score-state-pv-tile/latest.json"


def _read_json(path: pathlib.Path) -> dict[str, Any]:
  if not path.exists(): return {"available": False, "path": str(path.relative_to(ROOT))}
  d = json.loads(path.read_text())
  return {"available": True, "path": str(path.relative_to(ROOT)), "verdict": d.get("verdict"), "standalone_numeric": d.get("standalone_numeric", {})}


def _minimal_repro() -> dict[str, Any]:
  """For now, use the attention blocker as the authoritative repro.

  A smaller synthetic repro is the next implementation step. The current
  attention builder is already minimal enough to classify the failing shape:
  two REDUCE scopes plus local d plus recurrence-state stores fail before run.
  """
  attn = _read_json(ATTN_BLOCKER)
  numeric = attn.get("standalone_numeric", {}) if attn.get("available") else {}
  tb = numeric.get("traceback_tail", "")
  if numeric.get("verdict") == "FUSED_SCORE_STATE_PV_TILE_BLOCKED__MULTI_REDUCTION_STORE_SHAPE" and "Estimates.from_uops" in tb and "pop from empty list" in tb:
    verdict = "FUSED_TILE_LIFECYCLE_BLOCKED__ESTIMATE_SCOPE_STACK"
    classified = True
  elif attn.get("available"):
    verdict = "FUSED_TILE_LIFECYCLE_BLOCKED__UNCLASSIFIED_FROM_ATTENTION_ARTIFACT"
    classified = False
  else:
    verdict = "FUSED_TILE_LIFECYCLE_REPRO_MISSING"
    classified = False
  return {
    "checked": True,
    "verdict": verdict,
    "classified": classified,
    "source": "attention_builder_blocker_artifact",
    "known_failure_signature": {
      "exception": numeric.get("exception"),
      "exception_type": numeric.get("exception_type"),
      "has_estimates_from_uops": "Estimates.from_uops" in tb,
      "has_pop_from_empty_list": "pop from empty list" in tb,
    },
  }


def build() -> dict[str, Any]:
  repro = _minimal_repro()
  return {
    "date": "2026-06-26",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": repro["verdict"],
    "minimal_repro": repro,
    "attention_blocker_artifact": _read_json(ATTN_BLOCKER),
    "required_lowering_capability": {
      "generic_shape": "nested reduce + recurrence tuple + local output axis + compact metadata store",
      "attention_shape": "q.k score reduce inside token recurrence with local-d PV and l/m metadata columns",
      "first_fix_target": "scope-balanced lowering/estimation for nested END scopes",
      "not_yet": ["LDS tuning", "v_dot2 lowering", "W==D promotion"],
    },
    "next_step": "Build a smaller synthetic repro that fails without attention-specific code, then fix or encapsulate the lowering pattern.",
  }


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  latest = OUT / "latest.json"
  stamped = OUT / f"fused-tile-lifecycle-lowering-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
