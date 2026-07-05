#!/usr/bin/env python3
"""Generated quant runtime binding audit.

Phase 1 scaffold for docs/generated-quant-runtime-execution-map-20260705.md. This is intentionally conservative:
route-manifest rows are authoritative for known route provenance, while static source findings identify custom-kernel
and source-string surfaces that need descriptor/candidate ownership before promotion.
"""
from __future__ import annotations

import ast, json, pathlib
from dataclasses import dataclass
from typing import Any

from extra.qk import route_manifest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCAN_FILES = (
  "tinygrad/llm/prefill_routes.py",
  "tinygrad/llm/decode_routes.py",
  "tinygrad/llm/route_ops.py",
  "extra/qk/prefill_graph_gemm_route.py",
  "extra/qk/prefill_int8_wmma_spec.py",
  "extra/qk/prefill_packed_tile_spec.py",
  "extra/qk/prefill_schedule_spec.py",
  "extra/qk/quant/q4_k_gemv_primitive.py",
  "extra/qk/quant/q6_k_gemv_primitive.py",
  "extra/qk/q6k_route_spec.py",
  "extra/qk/gemv_g3_codegen_lowering.py",
  "extra/qk/flash_decode.py",
  "extra/qk/live_split_geometry.py",
  "extra/qk/flash_decode_fused_combine.py",
)

ALLOWED_PROVENANCE = {"machine_authored_generated", "tinygrad_scheduler_generated"}
TRANSITIONAL_PROVENANCE = {"hand_authored_uop_template"}
BANNED_PROVENANCE = {"external_handwritten_kernel", "rollback_oracle"}


@dataclass(frozen=True)
class Finding:
  path: str
  line: int
  kind: str
  detail: str

  def to_json(self) -> dict[str, Any]:
    return {"path": self.path, "line": self.line, "kind": self.kind, "detail": self.detail}


def _source(path:str) -> str:
  return (ROOT / path).read_text()


def _call_name(node:ast.AST) -> str:
  if isinstance(node, ast.Name): return node.id
  if isinstance(node, ast.Attribute): return f"{_call_name(node.value)}.{node.attr}"
  if isinstance(node, ast.Call): return _call_name(node.func)
  return type(node).__name__


def _literal_contains(node:ast.AST, needle:str) -> bool:
  if isinstance(node, ast.Constant) and isinstance(node.value, str): return needle in node.value
  return any(_literal_contains(ch, needle) for ch in ast.iter_child_nodes(node))


def scan_file(path:str) -> list[Finding]:
  src = _source(path)
  try:
    tree = ast.parse(src, filename=path)
  except SyntaxError as e:
    return [Finding(path, e.lineno or 0, "unknown.syntax_error", str(e))]
  findings: list[Finding] = []
  for node in ast.walk(tree):
    if isinstance(node, ast.Call):
      name = _call_name(node.func)
      if name.endswith(".custom_kernel") or name == "Tensor.custom_kernel":
        findings.append(Finding(path, node.lineno, "binding.custom_kernel", name))
      if name == "UOp":
        text = ast.get_source_segment(src, node) or ""
        if "Ops.CUSTOMI" in text: findings.append(Finding(path, node.lineno, "binding.ops_customi", "UOp(Ops.CUSTOMI)"))
        if "Ops.CUSTOM" in text: findings.append(Finding(path, node.lineno, "binding.ops_custom", "UOp(Ops.CUSTOM)"))
        if "Ops.PROGRAM" in text: findings.append(Finding(path, node.lineno, "binding.ops_program", "UOp(Ops.PROGRAM)"))
        if "Ops.INS" in text: findings.append(Finding(path, node.lineno, "binding.ops_ins", "UOp(Ops.INS)"))
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
      val = node.value
      if "asm volatile" in val: findings.append(Finding(path, node.lineno, "source.inline_asm", "asm volatile"))
      if "__builtin_amdgcn" in val: findings.append(Finding(path, node.lineno, "source.renderer_builtin", "__builtin_amdgcn"))
  # Catch source builders where the string is assembled across constants.
  for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and _literal_contains(node, "asm volatile"):
      findings.append(Finding(path, node.lineno, "source_builder.inline_asm", node.name))
    if isinstance(node, ast.FunctionDef) and _literal_contains(node, "typedef"):
      findings.append(Finding(path, node.lineno, "source_builder.custom_source", node.name))
  return sorted(findings, key=lambda f: (f.path, f.line, f.kind))


def scan_bindings(paths:tuple[str, ...]=SCAN_FILES) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for p in paths:
    if not (ROOT / p).exists():
      rows.append(Finding(p, 0, "unknown.missing_file", "scan target missing").to_json())
      continue
    rows.extend(f.to_json() for f in scan_file(p))
  return rows


def _classification_for_route(route_id:str, row:dict[str, Any]) -> str:
  prov = str(row.get("provenance", "unknown"))
  status = str(row.get("status", "unknown"))
  if prov in ALLOWED_PROVENANCE: return "allowed.generated"
  if prov in TRANSITIONAL_PROVENANCE: return "transitional.hand_authored_uop"
  if prov in BANNED_PROVENANCE:
    if status in ("removed", "superseded_rollback", "rollback_reference"): return "banned.not_default_but_reachable_or_ledgered"
    return "banned.default_or_research"
  return "unknown.investigate"


def route_rows() -> list[dict[str, Any]]:
  rows = []
  for route_id, r in sorted(route_manifest.ROUTES.items()):
    rows.append({
      "route_id": route_id,
      "workload": r.get("workload"),
      "status": r.get("status"),
      "provenance": r.get("provenance"),
      "classification": _classification_for_route(route_id, r),
      "purity_status": r.get("purity_status"),
      "replacement_scope": r.get("replacement_scope", ""),
      "authority_gate": r.get("authority_gate", ""),
      "route_attribution": r.get("route_attribution", ""),
    })
  return rows


def summarize(routes:list[dict[str, Any]], bindings:list[dict[str, Any]]) -> dict[str, Any]:
  by_class: dict[str, int] = {}
  for r in routes: by_class[r["classification"]] = by_class.get(r["classification"], 0) + 1
  by_kind: dict[str, int] = {}
  for b in bindings: by_kind[b["kind"]] = by_kind.get(b["kind"], 0) + 1
  default_debt = [r["route_id"] for r in routes if r["status"] in ("promoted_default", "default_shipped") and not r["classification"].startswith("allowed.")]
  return {"routes_by_classification": by_class, "bindings_by_kind": by_kind, "default_debt": default_debt,
          "unknown_bindings": [b for b in bindings if str(b["kind"]).startswith("unknown.")]}


def build() -> dict[str, Any]:
  routes = route_rows()
  bindings = scan_bindings()
  summary = summarize(routes, bindings)
  verdict = "GENERATED_QUANT_BINDING_AUDIT_READY"
  return {"schema": "generated_quant_binding_audit.v1",
          "scope": "Phase 1 route/binding provenance inventory for generated quant runtime",
          "verdict": verdict,
          "scan_files": list(SCAN_FILES),
          "summary": summary,
          "routes": routes,
          "bindings": bindings,
          "next": ["convert this inventory into RuntimeOpSpec/GeneratedCandidate descriptors",
                   "resolve unknown bindings before promoting any new optimized route",
                   "do not add new Q4_K-specific route branches outside the candidate model"]}


if __name__ == "__main__":
  print(json.dumps(build(), indent=2))
