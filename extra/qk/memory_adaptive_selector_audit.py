#!/usr/bin/env python3
"""Static audit for hard-coded selectors on the selected-model prefill path.

The report is deliberately source based: it does not import production modules,
touch a GPU, or depend on the current environment.  Evidence/harness files are
reported as non-gating fixtures.  Exact candidate target constraints are legal
only when expressed as ``target_requirements`` consumed by the autoscan catalog.
"""
from __future__ import annotations

import argparse, ast, json, re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

SCHEMA = "tinygrad.memory_adaptive_selector_audit.v1"

# This is the integration boundary, not a repository-wide naming lint.  Keep it
# explicit so additions to the runtime path require a reviewable audit update.
PRODUCTION_MODULES = (
  "tinygrad/llm/admission.py", "tinygrad/llm/decode_routes.py", "tinygrad/llm/device_facts.py", "tinygrad/llm/model.py",
  "tinygrad/llm/gguf_memory_scan.py", "tinygrad/llm/memory_adaptive_authority.py", "tinygrad/llm/memory_ledger.py",
  "tinygrad/llm/memory_semantics.py", "tinygrad/llm/model_facts.py", "tinygrad/llm/model_route_plan.py",
  "tinygrad/llm/prefill_memory_plan.py", "tinygrad/llm/prefill_route_census.py",
  "tinygrad/llm/prefill_policy.py", "tinygrad/llm/prefill_routes.py", "tinygrad/llm/prefill_workload_plan.py",
  "tinygrad/llm/qk_primitives.py", "tinygrad/llm/route_policy.py", "tinygrad/llm/cooperative_mmq_gate.py",
  "tinygrad/llm/physical_memory_ledger.py", "tinygrad/llm/schedule_memory_evidence.py",
  "extra/qk/memory_adaptive_allocation_observer.py", "extra/qk/memory_adaptive_autoscan.py",
  "extra/qk/memory_adaptive_boundary_gate.py", "extra/qk/memory_adaptive_candidate_catalog.py",
  "extra/qk/memory_adaptive_evidence_runner.py", "extra/qk/memory_adaptive_policy.py",
  "extra/qk/memory_adaptive_runtime_collector.py", "extra/qk/memory_adaptive_search_controller.py",
  "extra/qk/memory_adaptive_tinygrad_seam.py", "extra/qk/prefill/workload_inventory.py",
  "extra/qk/prefill_graph_gemm_route.py", "extra/qk/route_manifest.py",
)

_FIXTURE_PARTS = re.compile(r"(?:^|/)(?:test|tests|bench|docs)(?:/|$)|(?:evidence|benchmark|harness|fixture|probe|validation|canary|gate)\.py$")
_NAME_SELECTOR = re.compile(r"(?:^|[.\['_])(?:model_name|filename|file_name|model_path|profile|profile_id|size_label|model_size(?:_label)?|parameter_size|param_count)(?:$|[.\]'_])", re.I)
_SIZE_LABEL = re.compile(r"(?<![A-Za-z0-9])(?:\d+(?:\.\d+)?[bB])(?:[/_-]\d+[bB])?(?![A-Za-z0-9])")
_TARGET_LITERAL = re.compile(r"^(?:AMD|CUDA|METAL|NV|gfx\d+[a-z]*|sm_?\d+)$", re.I)
_MEMORY_NAME = re.compile(r"(?:vram|memory|mem_bytes|free_bytes|total_bytes|budget|reserve)", re.I)
_SELECTOR_CALL = re.compile(r"(?:^|_)(?:select|admit|route|candidate|policy|strategy|support|lookup)(?:_|s?$|ed$)", re.I)
_MATCHER_CALL = re.compile(r"(?:^|_)(?:requirements?_met|match(?:es|ed)?|compatible|supports?)(?:_|$)", re.I)
_SCANNED_FACT_NAME = re.compile(r"(?:target_capabilities|device_facts|gpu_facts|scanned|observed|actual)", re.I)
_PROVENANCE_FIELDS = frozenset(("profile", "profile_id", "profiles", "note", "promotion_artifacts", "authority_gate",
                                "route_attribution", "disposition", "benchmark", "evidence", "evidence_path",
                                "citation", "provenance_profiles", "candidate_profiles", "compatibility_aliases",
                                "legacy_route_id"))
_GLOBAL_CONTROL = re.compile(r"(?:route|admit|admission|policy|strategy|quant|primitive|generated|graph.?gemm|tc.?attn|concrete.?kv|kv.?quant|reserve|vram|prefill|decode)", re.I)
_ENV_READ = re.compile(r"(?:^|\.)(?:getenv|environ)(?:$|\.)", re.I)


def _module_global_assignments(tree: ast.AST) -> dict[str, ast.AST]:
  """Names initialized at module scope (including annotated/augmented names)."""
  result: dict[str, ast.AST] = {}
  for node in getattr(tree, "body", ()):
    targets = ()
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
      targets = tuple(node.targets) if isinstance(node, ast.Assign) else (node.target,)
    for target in targets:
      if isinstance(target, ast.Name): result[target.id] = node
  return result


def _is_global_control_name(name: str) -> bool:
  # Route-kind labels, caches, manifests, and debug switches describe or
  # cache policy; they are not independent admission/route inputs.
  if re.search(r"(?:ROUTE_KIND|CACHE|MANIFEST|PROVENANCE|^SCHEMA$|DEBUG)", name, re.I): return False
  return bool(_GLOBAL_CONTROL.search(name))

def _is_context_local(declaration: ast.AST) -> bool:
  value = declaration.value if isinstance(declaration, (ast.Assign, ast.AnnAssign)) else None
  return isinstance(value, ast.Call) and _text_name(value.func).split(".")[-1] == "ContextVar"


@dataclass(frozen=True, order=True)
class Finding:
  path: str
  line: int
  column: int
  rule_id: str
  symbol: str
  classification: str
  gating: bool
  message: str
  source: str


def _symbol(parents: dict[ast.AST, ast.AST], node: ast.AST) -> str:
  cur = node
  while cur in parents:
    cur = parents[cur]
    if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): return cur.name
  return "<module>"


def _text_name(node: ast.AST) -> str:
  if isinstance(node, ast.Name): return node.id
  if isinstance(node, ast.Attribute): return f"{_text_name(node.value)}.{node.attr}".strip(".")
  if isinstance(node, ast.Subscript):
    key = node.slice.value if isinstance(node.slice, ast.Constant) else None
    return f"{_text_name(node.value)}[{key!r}]"
  return ""


def _in_semantic_control(parents: dict[ast.AST, ast.AST], node: ast.AST) -> bool:
  cur = node
  while cur in parents:
    cur = parents[cur]
    if isinstance(cur, (ast.If, ast.IfExp, ast.While, ast.Assert)):
      # Input/provenance validation is not route selection.
      if isinstance(cur, ast.If) and cur.body and all(isinstance(x, (ast.Raise, ast.Assert)) for x in cur.body): return False
      return node in ast.walk(cur.test)
    if isinstance(cur, ast.comprehension): return any(node in ast.walk(x) for x in cur.ifs)
    if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)): break
  return False


def _candidate_constraint(parents: dict[ast.AST, ast.AST], node: ast.AST) -> bool:
  """True only for a CandidateSpec target_requirements payload.

  A random dict named target is intentionally insufficient: production must
  flow through CandidateSpec -> build_candidate_catalog -> _requirements_met.
  """
  cur = node
  while cur in parents:
    cur = parents[cur]
    if isinstance(cur, ast.Call):
      callee = _text_name(cur.func).split(".")[-1]
      return callee == "CandidateSpec" and any(k.arg == "target_requirements" and node in ast.walk(k.value) for k in cur.keywords)
    if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)): break
  return False


def _enclosing_dict_field(parents: dict[ast.AST, ast.AST], node: ast.AST) -> str | None:
  cur = node
  while cur in parents:
    parent = parents[cur]
    if isinstance(parent, ast.Dict):
      for key, value in zip(parent.keys, parent.values):
        if value is cur or node in ast.walk(value):
          return key.value if isinstance(key, ast.Constant) and isinstance(key.value, str) else None
    cur = parent
  return None


def _resolved_dicts(tree: ast.AST) -> dict[str, list[ast.Dict]]:
  assigned: dict[str, list[ast.Dict]] = {}
  for node in ast.walk(tree):
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
      value = node.value
      targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
      if isinstance(value, ast.Dict):
        for target in targets:
          if isinstance(target, ast.Name): assigned.setdefault(target.id, []).append(value)
  return assigned


def _dicts_in(node: ast.AST, assigned: dict[str, list[ast.Dict]]) -> list[ast.Dict]:
  out = [x for x in ast.walk(node) if isinstance(x, ast.Dict)]
  for name in (x.id for x in ast.walk(node) if isinstance(x, ast.Name)): out.extend(assigned.get(name, ()))
  return out


def _matched_target_literals(tree: ast.AST, assigned: dict[str, list[ast.Dict]]) -> set[ast.Constant]:
  """Find requirement mappings explicitly matched against scanned/observed facts."""
  matched: set[ast.Constant] = set()
  for node in ast.walk(tree):
    requirement_nodes: list[ast.AST] = []
    if isinstance(node, ast.Call) and _MATCHER_CALL.search(_text_name(node.func).split(".")[-1]):
      args = [*node.args, *(x.value for x in node.keywords)]
      if any(_SCANNED_FACT_NAME.search(_text_name(x)) for arg in args for x in ast.walk(arg) if isinstance(x, (ast.Name, ast.Attribute))):
        requirement_nodes.extend(args)
    elif isinstance(node, ast.Compare):
      sides = [node.left, *node.comparators]
      if any(_SCANNED_FACT_NAME.search(_text_name(x)) for side in sides for x in ast.walk(side) if isinstance(x, (ast.Name, ast.Attribute))):
        requirement_nodes.extend(sides)
    for requirement in requirement_nodes:
      for mapping in _dicts_in(requirement, assigned):
        matched.update(x for x in ast.walk(mapping) if isinstance(x, ast.Constant) and isinstance(x.value, str) and _TARGET_LITERAL.match(x.value))
  return matched


def _line(lines: list[str], lineno: int) -> str:
  return lines[lineno-1].strip() if 0 < lineno <= len(lines) else ""


def _numeric_constant(node: ast.AST) -> int | float | None:
  """Evaluate a side-effect-free numeric literal expression."""
  try: value = ast.literal_eval(node)
  except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add, ast.Sub, ast.Div, ast.FloorDiv)):
      left, right = _numeric_constant(node.left), _numeric_constant(node.right)
      if left is None or right is None: return None
      if isinstance(node.op, ast.Mult): return left * right
      if isinstance(node.op, ast.Add): return left + right
      if isinstance(node.op, ast.Sub): return left - right
      if right == 0: return None
      return left / right if isinstance(node.op, ast.Div) else left // right
    return None
  return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _audit_file(root: Path, path: Path) -> list[Finding]:
  rel = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()
  fixture = bool(_FIXTURE_PARTS.search(rel))
  text = path.read_text(encoding="utf-8")
  lines = text.splitlines()
  try: tree = ast.parse(text, filename=rel)
  except SyntaxError as exc:
    return [Finding(rel, exc.lineno or 0, exc.offset or 0, "AUDIT_PARSE_ERROR", "<module>", "invalid_source", True,
                    exc.msg, _line(lines, exc.lineno or 0))]
  parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
  assigned_dicts = _resolved_dicts(tree)
  module_globals = _module_global_assignments(tree)
  matched_target_literals = _matched_target_literals(tree, assigned_dicts)
  out: list[Finding] = []
  seen: set[tuple[str, int, int, str]] = set()

  def add(node: ast.AST, rule: str, classification: str, message: str, *, gating: bool = True) -> None:
    line, col = getattr(node, "lineno", 0), getattr(node, "col_offset", 0)+1
    key = (rule, line, col, message)
    if key in seen: return
    seen.add(key)
    out.append(Finding(rel, line, col, rule, _symbol(parents, node), classification,
                       gating and not fixture, message, _line(lines, line)))

  def provenance(node: ast.AST) -> bool:
    return rel == "extra/qk/route_manifest.py" and _enclosing_dict_field(parents, node) in _PROVENANCE_FIELDS

  # Environment and module-global route/admission switches are deliberately
  # separate from identity lint: they can bypass an immutable selected policy
  # even when the caller supplied no model name at all.
  for node in ast.walk(tree):
    if isinstance(node, ast.Call) and _ENV_READ.search(_text_name(node.func)):
      global_binding = any(value is node or node in ast.walk(value) for value in module_globals.values())
      if _in_semantic_control(parents, node) or global_binding:
        add(node, "FORBIDDEN_ENVIRONMENT_CONTROL", "evidence_fixture" if fixture else "forbidden_selector",
            f"environment read {_text_name(node.func)!r} controls runtime route/admission policy")
    elif isinstance(node, (ast.Subscript, ast.Attribute)) and _ENV_READ.search(_text_name(node)) \
        and not (isinstance(parents.get(node), ast.Attribute) and _ENV_READ.search(_text_name(parents[node]))):
      if _in_semantic_control(parents, node):
        add(node, "FORBIDDEN_ENVIRONMENT_CONTROL", "evidence_fixture" if fixture else "forbidden_selector",
            f"environment value {_text_name(node)!r} controls runtime route/admission policy")

  for name, declaration in module_globals.items():
    if not _is_global_control_name(name) or _is_context_local(declaration): continue
    for use in ast.walk(tree):
      if isinstance(use, ast.Name) and use.id == name and use is not next((x for x in ast.walk(declaration) if x is use), None):
        if _in_semantic_control(parents, use):
          add(use, "FORBIDDEN_MODULE_GLOBAL_CONTROL", "evidence_fixture" if fixture else "forbidden_selector",
              f"module-global control {name!r} influences runtime route/admission execution")

  # Semantic reads of explicitly non-semantic identity fields.
  for node in ast.walk(tree):
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
      name = _text_name(node)
      if name and _NAME_SELECTOR.search(name) and _in_semantic_control(parents, node):
        is_provenance = provenance(node)
        add(node, "FORBIDDEN_IDENTITY_SELECTOR", "evidence_metadata" if is_provenance else "evidence_fixture" if fixture else "forbidden_selector",
            f"non-semantic identity {name!r} influences runtime control flow", gating=not is_provenance)

  # Calls into selector/admission surfaces can hide identity use outside an if.
  for node in ast.walk(tree):
    if not isinstance(node, ast.Call): continue
    callee = _text_name(node.func).split(".")[-1]
    if not _SELECTOR_CALL.search(callee): continue
    for keyword in node.keywords:
      if keyword.arg and _NAME_SELECTOR.search(keyword.arg):
        add(keyword.value, "FORBIDDEN_IDENTITY_SELECTOR", "evidence_fixture" if fixture else "forbidden_selector",
            f"non-semantic identity keyword {keyword.arg!r} is passed to {callee}")
    for arg in node.args:
      names = sorted({_text_name(x) for x in ast.walk(arg) if isinstance(x, (ast.Name, ast.Attribute, ast.Subscript))})
      bad = next((x for x in names if _NAME_SELECTOR.search(x)), None)
      if bad is None:
        bad = next((x.value for x in ast.walk(arg) if isinstance(x, ast.Constant) and isinstance(x.value, str)
                    and _NAME_SELECTOR.search(x.value)), None)
      if bad: add(arg, "FORBIDDEN_IDENTITY_SELECTOR", "evidence_fixture" if fixture else "forbidden_selector",
                  f"non-semantic identity {bad!r} is passed to {callee}")

  for node in ast.walk(tree):
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str): continue
    value = node.value
    if _SIZE_LABEL.search(value) and (_in_semantic_control(parents, node) or isinstance(parents.get(node), (ast.Dict, ast.Set, ast.List, ast.Tuple))):
      is_provenance = provenance(node)
      add(node, "FORBIDDEN_PARAMETER_SIZE_LABEL", "evidence_metadata" if is_provenance else "evidence_fixture" if fixture else "forbidden_selector",
          f"parameter-size label {value!r} participates in executable selector data", gating=not is_provenance)
    if _TARGET_LITERAL.match(value) and (_in_semantic_control(parents, node) or isinstance(parents.get(node), ast.Dict)):
      if rel == "tinygrad/llm/device_facts.py":
        add(node, "ALLOWED_AUTOSCAN_SOURCE_FACT", "autoscan_source_fact",
            f"target probe backend fact {value!r} is an autoscan input, not a route selector", gating=False)
      elif _candidate_constraint(parents, node) or node in matched_target_literals:
        add(node, "ALLOWED_EXACT_CANDIDATE_CONSTRAINT", "candidate_capability_constraint",
            f"exact target fact {value!r} is structurally matched against scanned target capabilities", gating=False)
      else:
        add(node, "UNMATCHED_TARGET_FACT", "evidence_fixture" if fixture else "forbidden_selector",
            f"hard-coded target fact {value!r} is not an autoscan-matched candidate constraint")

  # A literal memory cutoff is a tier. Arithmetic formulas and percentages are
  # handled by the planner/reserve policy and are not fixed-tier selectors.
  for node in ast.walk(tree):
    if not isinstance(node, ast.Compare): continue
    names = " ".join(_text_name(x) for x in ast.walk(node) if isinstance(x, (ast.Name, ast.Attribute)))
    if _MEMORY_NAME.search(names):
      for expression in (node.left, *node.comparators):
        value = _numeric_constant(expression)
        if value is not None and abs(float(value)) >= 1024:
          add(expression, "FIXED_MEMORY_TIER", "evidence_fixture" if fixture else "forbidden_selector",
              f"literal memory cutoff {value!r} controls admission/selection")
  return out


def audit_repository(root: str | Path, paths: Iterable[str | Path] | None = None) -> dict:
  root = Path(root).resolve()
  selected = tuple(Path(x) for x in (PRODUCTION_MODULES if paths is None else paths))
  files = sorted({(x if x.is_absolute() else root/x).resolve() for x in selected if (x if x.is_absolute() else root/x).is_file()})
  findings = sorted((f for path in files for f in _audit_file(root, path)),
                    key=lambda x: (x.path, x.line, x.column, x.rule_id, x.message))

  # Report duplicated literal memory tiers across distinct source locations.
  tiers: dict[str, list[Finding]] = {}
  for finding in findings:
    if finding.rule_id == "FIXED_MEMORY_TIER":
      match = re.search(r"cutoff (.+?) controls", finding.message)
      if match: tiers.setdefault(match.group(1), []).append(finding)
  duplicates: list[Finding] = []
  for literal, occurrences in tiers.items():
    if len(occurrences) < 2: continue
    for old in occurrences:
      duplicates.append(Finding(old.path, old.line, old.column, "DUPLICATED_MEMORY_THRESHOLD", old.symbol,
        old.classification, old.gating, f"memory cutoff {literal} is duplicated at {len(occurrences)} audited locations", old.source))
  findings = sorted((*findings, *duplicates), key=lambda x: (x.path, x.line, x.column, x.rule_id, x.message))
  gating = [x for x in findings if x.gating]
  counts: dict[str, int] = {}
  for item in findings: counts[item.rule_id] = counts.get(item.rule_id, 0)+1
  return {"schema": SCHEMA, "status": "FAIL" if gating else "PASS", "root": root.as_posix(),
          "audited_files": [x.relative_to(root).as_posix() if x.is_relative_to(root) else x.as_posix() for x in files],
          "counts": dict(sorted(counts.items())), "gating_finding_count": len(gating),
          "findings": [asdict(x) for x in findings]}


def stable_json(report: dict) -> str:
  return json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
  parser.add_argument("paths", nargs="*", help="optional repository-relative files (defaults to production boundary)")
  parser.add_argument("--pretty", action="store_true")
  args = parser.parse_args(argv)
  report = audit_repository(args.root, args.paths or None)
  print(json.dumps(report, sort_keys=True, indent=2 if args.pretty else None, separators=None if args.pretty else (",", ":")))
  return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__": raise SystemExit(main())

__all__ = ["SCHEMA", "PRODUCTION_MODULES", "Finding", "audit_repository", "stable_json", "main"]
