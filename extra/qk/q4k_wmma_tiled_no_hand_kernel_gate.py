#!/usr/bin/env python3
"""Hard no-hand-kernel gate for the Q4_K/Q8_1 tiled WMMA prefill route."""
from __future__ import annotations

import json, pathlib
from dataclasses import dataclass
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]

IMPLEMENTATION_FILES = (
  "extra/qk/prefill_int8_wmma_spec.py",
  "tinygrad/llm/route_ops.py",
  "tinygrad/llm/generated_candidates.py",
)

OPTIONAL_IMPLEMENTATION_FILES = (
  "extra/qk/q4k_wmma_tile_lowering.py",
)

FORBIDDEN_TOKENS = (
  ("source.inline_asm", "asm volatile"),
  ("source.route_local_wmma_builtin", "__builtin_amdgcn_wmma"),
  ("binding.route_local_ops_wmma", "Ops.WMMA"),
  ("binding.route_local_custom_kernel", ".custom_kernel("),
  ("binding.route_local_tensor_custom_kernel", "Tensor.custom_kernel"),
  ("import.handwritten_prefill_wmma", "extra.qk.prefill.wmma"),
)


@dataclass(frozen=True)
class Finding:
  path: str
  line: int
  kind: str
  detail: str

  def to_json(self) -> dict[str, Any]:
    return {"path": self.path, "line": self.line, "kind": self.kind, "detail": self.detail}


def _line_no(src:str, offset:int) -> int:
  return src.count("\n", 0, offset) + 1


def _scan_text(path:str, src:str) -> list[Finding]:
  findings: list[Finding] = []
  for kind, token in FORBIDDEN_TOKENS:
    start = 0
    while True:
      idx = src.find(token, start)
      if idx < 0: break
      findings.append(Finding(path, _line_no(src, idx), kind, token))
      start = idx + len(token)
  return findings


def _wmma_tiled_route_block() -> tuple[str, str]:
  path = "tinygrad/llm/prefill_routes.py"
  src = (ROOT / path).read_text()
  marker = 'if q8_mode == "wmma_tiled":'
  start = src.find(marker)
  if start < 0:
    return path, ""
  end_marker = "\n      out = partials.custom_kernel"
  end = src.find(end_marker, start)
  if end < 0:
    end = len(src)
  return path, src[start:end]


def build() -> dict[str, Any]:
  scanned: list[str] = []
  findings: list[Finding] = []
  for path in IMPLEMENTATION_FILES + OPTIONAL_IMPLEMENTATION_FILES:
    p = ROOT / path
    if not p.exists():
      continue
    scanned.append(path)
    findings.extend(_scan_text(path, p.read_text()))
  route_path, route_block = _wmma_tiled_route_block()
  scanned.append(f"{route_path}::q8_mode==wmma_tiled")
  if not route_block:
    findings.append(Finding(route_path, 0, "route.missing_wmma_tiled_branch", "if q8_mode == \"wmma_tiled\""))
  else:
    findings.extend(_scan_text(f"{route_path}::q8_mode==wmma_tiled", route_block))

  ok = not findings
  return {"schema": "q4k_wmma_tiled_no_hand_kernel_gate.v1",
          "scope": "Q4_K/Q8_1 wmma_tiled route implementation must remain generated/tinygrad-owned",
          "verdict": "Q4K_WMMA_TILED_NO_HAND_KERNEL_PASS" if ok else "Q4K_WMMA_TILED_NO_HAND_KERNEL_FAIL",
          "route_id": "prefill_q4k_int8_wmma_tiled_research",
          "scanned": scanned,
          "forbidden": [{"kind": k, "token": t} for k, t in FORBIDDEN_TOKENS],
          "allowed_external_owners": ["tinygrad/renderer/cstyle.py", "tinygrad/codegen/opt/tc.py",
                                      "tinygrad/schedule/rangeify.py"],
          "findings": [f.to_json() for f in sorted(findings, key=lambda x: (x.path, x.line, x.kind))]}


if __name__ == "__main__":
  out = build()
  print(json.dumps(out, indent=2))
  raise SystemExit(0 if out["verdict"] == "Q4K_WMMA_TILED_NO_HAND_KERNEL_PASS" else 1)
