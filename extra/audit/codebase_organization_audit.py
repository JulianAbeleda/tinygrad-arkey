#!/usr/bin/env python3
"""Codebase organization census: manifest-declared organizational intent vs. objectively derived drift.

The MANIFEST (extra/audit/codebase_organization_manifest.json) is the human-authored source of organizational intent:
what each authored file is for, which domain owns it, what authority it holds, and what should happen to it. This
script owns only the objective half -- discovery, sz.py token-bearing LOC, manifest coverage/schema, an AST import
graph, reference scanning, boundary checks, and deterministic reporting. It makes NO semantic judgment: it never
decides that organization is good or bad, never scores abstraction quality, and never infers purpose from filenames.

Run:  PYTHONPATH=. python3 extra/audit/codebase_organization_audit.py
      PYTHONPATH=. python3 extra/audit/codebase_organization_audit.py --check         # hard errors only, no writes
      PYTHONPATH=. python3 extra/audit/codebase_organization_audit.py --scope extra/qk
      PYTHONPATH=. python3 extra/audit/codebase_organization_audit.py --top 30
Outputs (bench/codebase-organization-audit/):
  latest.json            -- full census (facts + manifest-declared findings + hard errors + warnings)
  summary.md             -- human report
  workflow_table.json    -- workflow inventory with repeated phases
  action_candidates.json -- decouple/centralize/modularize/reuse/promote/prune candidates with LOC arithmetic

This tool reads source files and writes JSON/MD only. It imports no tinygrad runtime module and executes no project
code: the import graph is built with `ast`, and LOC reuses sz.py's tokenizer rules rather than a second definition.
"""
from __future__ import annotations
import ast, io, json, os, pathlib, re, subprocess, sys, token, tokenize

ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "extra/audit/codebase_organization_manifest.json"
OUT = ROOT / "bench/codebase-organization-audit"

# sz.py is the single owner of the authored/generated boundary and of token-bearing LOC. Import its rules; do not
# restate them. (sz.py imports nothing from tinygrad, so this keeps the auditor runtime-independent.)
sys.path.insert(0, str(ROOT))
import sz  # noqa: E402

SOURCE_ROOTS = ("tinygrad", "extra", "bench", "test")   # plus root-level *.py
SOURCE_EXT = (".py", ".js")
EXCLUDE_PREFIX = ("tinygrad/runtime/autogen/", "docs/artifacts/", "tinygrad/viz/assets/")
REFERENCE_EXT = (".md", ".json", ".sh", ".yml", ".yaml", ".toml", ".txt", ".cfg")
PLACEHOLDER_PURPOSE = ("utilities", "utility", "helpers", "helper", "miscellaneous", "misc", "various", "stuff")
RECORD_FIELDS = ("path", "domain", "purpose", "role", "status", "default_path", "authority_keys", "public_surface",
                 "allowed_dependency_domains", "disposition", "evidence", "workflow_ids", "responsibilities",
                 "reusable_assets", "duplication_group", "promotion_target", "retention_criterion")
REQUIRED_NONEMPTY = ("path", "domain", "purpose", "role", "status", "disposition")
RESEARCH_ROLES = ("research", "evidence", "benchmark")
DEAD_STATUSES = ("refuted", "superseded", "deletion_candidate")
WORKFLOW_FIELDS = ("workflow_id", "domain", "purpose", "entry_points", "phases", "inputs", "outputs",
                   "authority_files", "execution_files", "evidence_files", "shared_assets", "default_path",
                   "retention_criterion")

# ---------------------------------------------------------------------------------------------------- discovery ----
def _tracked(root: pathlib.Path) -> list[str]:
  """Tracked files, git-first. The os.walk fallback exists so temp-tree tests need no git repository."""
  try:
    r = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True, check=True)
    files = [x for x in r.stdout.splitlines() if x]
    if files: return files
  except (OSError, subprocess.CalledProcessError): pass
  files = []
  for path, dirs, names in os.walk(root):
    dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "node_modules", ".venv", "env")]
    for n in names: files.append(os.path.relpath(os.path.join(path, n), root).replace("\\", "/"))
  return files

def in_source_scope(rel: str) -> bool:
  if not rel.endswith(SOURCE_EXT): return False
  if any(rel.startswith(p) for p in EXCLUDE_PREFIX): return False
  return rel.startswith(tuple(s + "/" for s in SOURCE_ROOTS)) or "/" not in rel

def commit_identity(root: pathlib.Path) -> dict:
  def _git(*a):
    try: return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError): return "unknown"
  return {"commit": _git("rev-parse", "HEAD"), "dirty": _git("status", "--porcelain") != "", "described": _git("describe", "--always", "--dirty")}

# --------------------------------------------------------------------------------------------------- accounting ----
def loc_of(path: pathlib.Path) -> int:
  """Token-bearing lines, using sz.py's rules verbatim (whitelisted tokens, docstrings excluded)."""
  if path.suffix == ".js":
    lines = [l.strip() for l in path.read_text(errors="ignore").splitlines()]
    return sum(1 for l in lines if sz.is_js_token(l))
  try:
    with tokenize.open(str(path)) as f:
      toks = [t for t in tokenize.generate_tokens(f.readline) if t.type in sz.TOKEN_WHITELIST and not sz.is_docstring(t)]
  except (tokenize.TokenizeError, SyntaxError, UnicodeDecodeError, OSError): return 0
  return len({x for t in toks for x in range(t.start[0], t.end[0] + 1)})

# ------------------------------------------------------------------------------------------------- import graph ----
def module_index(files: list[str]) -> dict[str, str]:
  """dotted module name -> repo-relative path. Packages resolve through __init__.py."""
  idx = {}
  for rel in files:
    if not rel.endswith(".py"): continue
    stem = rel[:-3]
    if stem.endswith("/__init__"): idx[stem[:-len("/__init__")].replace("/", ".")] = rel
    else: idx[stem.replace("/", ".")] = rel
  return idx

def _resolve(mod: str, idx: dict[str, str]) -> str | None:
  """Longest dotted prefix that names a tracked module; `from a.b import c` may name module a.b.c or symbol c in a.b."""
  parts = mod.split(".")
  for n in range(len(parts), 0, -1):
    hit = idx.get(".".join(parts[:n]))
    if hit: return hit
  return None

def imports_of(root: pathlib.Path, rel: str, idx: dict[str, str]) -> list[str]:
  if not rel.endswith(".py"): return []
  try: tree = ast.parse((root / rel).read_text(errors="ignore"))
  except (SyntaxError, ValueError, OSError): return []
  pkg = rel[:-len("/__init__.py")] if rel.endswith("/__init__.py") else os.path.dirname(rel)
  out = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      for a in node.names:
        if (hit := _resolve(a.name, idx)) and hit != rel: out.add(hit)
    elif isinstance(node, ast.ImportFrom):
      if node.level:                                    # relative: walk up `level-1` packages from this file's package
        base = pkg.split("/") if pkg else []
        base = base[:len(base) - (node.level - 1)] if node.level > 1 else base
        prefix = ".".join(base)
        mods = [f"{prefix}.{node.module}" if node.module else prefix]
      else: mods = [node.module] if node.module else []
      for m in mods:
        if not m: continue
        for cand in [f"{m}.{a.name}" for a in node.names] + [m]:
          if (hit := _resolve(cand, idx)) and hit != rel:
            out.add(hit); break
  return sorted(out)

def dynamic_imports_of(root: pathlib.Path, rel: str, idx: dict[str, str]) -> list[str]:
  """Edges the import graph cannot see: `importlib.import_module("extra.qk.x")` and the `_attr("extra.qk.x", "f")`
  lazy-boundary wrappers that tinygrad/llm/route_ops.py and tinygrad/codegen/experimental.py are built from.

  Without this, the entire tinygrad -> extra/qk seam is invisible and every extra/qk module looks unreachable from
  production. Only constant string arguments are resolved; a computed module name is not a machine fact.
  """
  if not rel.endswith(".py"): return []
  try: tree = ast.parse((root / rel).read_text(errors="ignore"))
  except (SyntaxError, ValueError, OSError): return []
  out = set()
  for node in ast.walk(tree):
    if not isinstance(node, ast.Call): continue
    name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
    if name not in ("import_module", "_attr", "getattr_module"): continue
    for arg in node.args[:1]:
      if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and (hit := _resolve(arg.value, idx)) and hit != rel:
        out.add(hit)
  return sorted(out)

def strongly_connected(edges: dict[str, list[str]]) -> list[list[str]]:
  """Tarjan, iterative. Returns components of size>1 (self-loops are not cycles here), sorted for determinism."""
  index, low, onstack, stack, comps, counter = {}, {}, set(), [], [], [0]
  for root_node in sorted(edges):
    if root_node in index: continue
    work = [(root_node, iter(edges.get(root_node, [])))]
    index[root_node] = low[root_node] = counter[0]; counter[0] += 1
    stack.append(root_node); onstack.add(root_node)
    while work:
      v, it = work[-1]
      for w in it:
        if w not in index:
          index[w] = low[w] = counter[0]; counter[0] += 1
          stack.append(w); onstack.add(w)
          work.append((w, iter(edges.get(w, [])))); break
        if w in onstack: low[v] = min(low[v], index[w])
      else:
        work.pop()
        if work: low[work[-1][0]] = min(low[work[-1][0]], low[v])
        if low[v] == index[v]:
          comp = []
          while True:
            w = stack.pop(); onstack.discard(w); comp.append(w)
            if w == v: break
          if len(comp) > 1: comps.append(sorted(comp))
  return sorted(comps)

# ---------------------------------------------------------------------------------------------- non-import refs ----
def reference_index(root: pathlib.Path, files: list[str], sources: list[str], ignore: set[str]) -> dict[str, list[str]]:
  """Who NAMES a source file without importing it: docs, manifests, shell, CI, subprocess argv, dynamic collection.

  Absence of a Python import is only one fact; standalone scripts are routinely invoked by command. Matching is on the
  full relative path and on the dotted module path -- never on the bare basename, which would produce false hits.
  """
  # Both forms need a right boundary: without it `extra.qk.layout` matches every mention of
  # `extra.qk.layout_coalesce_check`, and `extra/qk/bench` would match `extra/qk/benchmark_shared_attention.py`.
  names = {}
  for rel in sources:
    forms = [rel] + ([rel[:-3].replace("/", ".")] if rel.endswith(".py") else [])
    names[rel] = [re.compile(re.escape(f) + r"(?![\w.])") for f in forms]
  refs = {rel: set() for rel in sources}
  # the manifest and this audit's own reports name every file by construction; counting them would make every file
  # look referenced and would let the audit vouch for itself.
  scan = [f for f in files if f.endswith(REFERENCE_EXT + (".py",)) and not any(f.startswith(p) for p in EXCLUDE_PREFIX)
          and f not in ignore and not f.startswith("bench/codebase-organization-audit/")]
  for f in scan:
    try: text = (root / f).read_text(errors="ignore")
    except OSError: continue
    for rel, keys in names.items():
      if rel == f: continue
      if any(k.search(text) for k in keys): refs[rel].add(f)
  return {k: sorted(v) for k, v in refs.items()}

# -------------------------------------------------------------------------------------------------- manifest -------
def load_manifest(path: pathlib.Path) -> dict:
  try: m = json.loads(path.read_text())
  except FileNotFoundError: return {"files": [], "groups": [], "workflows": [], "domains": {}, "_missing": str(path)}
  m.setdefault("files", []); m.setdefault("groups", []); m.setdefault("workflows", [])
  m.setdefault("domains", {}); m.setdefault("actions", []); m.setdefault("config", {})
  return m

def _workflow_retained(wf: dict) -> bool:
  """A workflow whose retention criterion has not expired still owns its files, imports or not."""
  return bool(wf.get("default_path")) or bool(wf.get("retention_criterion"))

def group_match(rule: dict, rel: str) -> bool:
  if not any(rel.startswith(p) for p in rule.get("path_prefixes", [])): return False
  if (pat := rule.get("path_regex")) and not re.search(pat, rel): return False
  if any(re.search(x, rel) for x in rule.get("exclude_regex", [])): return False
  return True

# ----------------------------------------------------------------------------------------------------- audit -------
def audit(root: pathlib.Path = ROOT, manifest_path: pathlib.Path | None = None, scope: str | None = None,
          top: int = 20) -> dict:
  manifest_path = manifest_path or (root / "extra/audit/codebase_organization_manifest.json")
  man = load_manifest(manifest_path)
  cfg = man.get("config", {})
  loc_threshold = int(cfg.get("large_file_loc_threshold", 400))
  fan_threshold = int(cfg.get("fan_threshold", 12))
  manifest_scope = tuple(cfg.get("manifest_scope", ["extra/qk/"]))

  files = sorted(_tracked(root))
  sources = [f for f in files if in_source_scope(f)]
  loc = {f: loc_of(root / f) for f in sources}
  # sz.py drops 0-LOC files from its table; match that on both sides so an unparseable file cannot inflate either count
  generated = [f for f in sources if loc[f] > 0 and sz.is_generated(str(root / f))]
  authored = [f for f in sources if loc[f] > 0 and f not in set(generated)]

  idx = module_index(files)
  edges = {f: imports_of(root, f, idx) for f in authored}
  dyn = {f: [d for d in dynamic_imports_of(root, f, idx) if d not in edges.get(f, [])] for f in authored}
  dyn = {f: v for f, v in dyn.items() if v}
  reach = {f: sorted(set(edges.get(f, [])) | set(dyn.get(f, []))) for f in authored}
  inbound = {f: [] for f in authored}
  for src, dsts in reach.items():
    for d in dsts:
      if d in inbound: inbound[d].append(src)
  inbound = {k: sorted(v) for k, v in inbound.items()}
  try: manifest_rel = str(manifest_path.resolve().relative_to(root.resolve())).replace("\\", "/")
  except ValueError: manifest_rel = ""
  refs = reference_index(root, files, authored, {manifest_rel})

  # ---- manifest coverage ------------------------------------------------------------------------------------
  records = {r["path"]: r for r in man["files"]}
  dup_records = sorted({r["path"] for r in man["files"] if [x["path"] for x in man["files"]].count(r["path"]) > 1})
  groups = man["groups"]
  in_manifest_scope = [f for f in authored if f.startswith(manifest_scope)]
  covered_by_group = {}
  for f in in_manifest_scope:
    hits = [g["group_id"] for g in groups if group_match(g, f)]
    if hits: covered_by_group[f] = hits

  hard, warn = [], []
  def E(kind, msg, **ev): hard.append({"error": kind, "message": msg, **ev})
  def W(kind, msg, principle, **ev): warn.append({"warning": kind, "message": msg, "principle": principle, **ev})

  for p in dup_records: E("duplicate_manifest_entry", f"{p} appears more than once in manifest.files", path=p)
  for p in sorted(records):
    if not (root / p).exists(): E("stale_manifest_path", f"manifest names {p}, which does not exist", path=p)
    elif p in generated: E("generated_counted_as_authored", f"{p} carries a generated marker but has an authored record", path=p)
  for f in in_manifest_scope:
    if f not in records and f not in covered_by_group:
      E("unmanifested_file", f"authored file {f} has no explicit record and no group rule", path=f)
    if f in records and f in covered_by_group:
      E("ambiguous_coverage", f"{f} is covered by an explicit record AND group rule(s) {covered_by_group[f]}", path=f)
    if len(covered_by_group.get(f, [])) > 1:
      E("ambiguous_coverage", f"{f} matches multiple group rules {covered_by_group[f]}", path=f)

  for g in groups:
    for f in [x for x in in_manifest_scope if group_match(g, x)]:
      pass  # group-hidden checks need per-file declarations; the hiding check below uses machine facts only
  # a group rule must not hide a file that machine facts show is production-reachable or holds declared authority
  prod_reachable = set()
  frontier = [f for f in authored if f.startswith("tinygrad/")]
  seen = set(frontier)
  while frontier:
    cur = frontier.pop()
    for d in reach.get(cur, []):
      if d not in seen: seen.add(d); frontier.append(d)
  prod_reachable = {f for f in seen if not f.startswith("tinygrad/")}
  for f, gids in covered_by_group.items():
    if f in prod_reachable:
      E("group_hides_production_file", f"{f} is reachable from tinygrad/ but is only covered by group rule(s) {gids}",
        path=f, groups=gids)

  # ---- record schema ----------------------------------------------------------------------------------------
  for p, r in sorted(records.items()):
    for k in RECORD_FIELDS:
      if k not in r: E("missing_field", f"{p}: record is missing required field '{k}'", path=p, field=k)
    for k in REQUIRED_NONEMPTY:
      if not r.get(k): E("missing_field", f"{p}: required field '{k}' is empty", path=p, field=k)
    purpose = (r.get("purpose") or "").strip()
    if purpose and purpose.lower().rstrip(".") in PLACEHOLDER_PURPOSE:
      E("placeholder_purpose", f"{p}: purpose '{purpose}' is a placeholder", path=p)
    if r.get("default_path") and r.get("status") in DEAD_STATUSES:
      E("default_path_dead_status", f"{p} is default_path but status is '{r['status']}'", path=p, status=r["status"])
    if len(r.get("authority_keys") or []) > int(cfg.get("authority_key_warn_threshold", 3)):
      W("multiple_authority_keys", f"{p} declares {len(r['authority_keys'])} authority keys: {r['authority_keys']}",
        "centralize authoritative knowledge under one owner", path=p)
    if loc.get(p, 0) > loc_threshold:
      W("large_file", f"{p} is {loc[p]} LOC (threshold {loc_threshold}); responsibilities declared: "
        f"{len(r.get('responsibilities') or [])}", "minimize what a reader must hold in their head", path=p, loc=loc[p])
    if r.get("status") == "fallback" and not r.get("retention_criterion"):
      W("fallback_without_retention", f"{p} is a fallback path with no stated retention criterion",
        "every rollback path needs an expiry owner", path=p)
    if r.get("role") in RESEARCH_ROLES and not refs.get(p) and not inbound.get(p):
      W("research_without_owner", f"{p} is role={r['role']} with no import and no document/manifest reference",
        "experimental machinery needs a live question or a verdict", path=p)
    if r.get("disposition") in ("delete", "archive") and (inbound.get(p) or refs.get(p)):
      W("deletion_candidate_referenced", f"{p} is disposition={r['disposition']} but is still referenced by "
        f"{len(inbound.get(p, [])) + len(refs.get(p, []))} file(s)", "do not delete referenced code", path=p)

  # ---- authority ---------------------------------------------------------------------------------------------
  claims: dict[str, list[str]] = {}
  for p, r in records.items():
    for k in r.get("authority_keys") or []:
      if not (r.get("authority_exclusive") is False): claims.setdefault(k, []).append(p)
  for k, owners in sorted(claims.items()):
    if len(owners) > 1:
      E("duplicate_authority", f"authority key '{k}' is claimed exclusively by {sorted(owners)}", key=k, owners=sorted(owners))

  # ---- declared dependency boundaries --------------------------------------------------------------------------
  domain_of = {p: r.get("domain") for p, r in records.items()}
  for g in groups:
    for f in in_manifest_scope:
      if group_match(g, f): domain_of.setdefault(f, g.get("domain"))
  allowed = {p: set(r.get("allowed_dependency_domains") or []) for p, r in records.items()}
  for g in groups:
    for f in in_manifest_scope:
      if group_match(g, f): allowed.setdefault(f, set(g.get("allowed_dependency_domains") or []))
  boundary_violations = []
  for src, dsts in sorted(edges.items()):
    if src not in allowed: continue
    for d in dsts:
      dd = domain_of.get(d)
      if dd is None: continue
      if allowed[src] and dd not in allowed[src] and dd != domain_of.get(src):
        boundary_violations.append({"from": src, "to": d, "from_domain": domain_of.get(src), "to_domain": dd})
        E("forbidden_dependency", f"{src} ({domain_of.get(src)}) imports {d} ({dd}); allowed: {sorted(allowed[src])}",
          **{"from": src, "to": d})

  # production tinygrad/ must not depend on audit or research-role modules. A *static* import is a hard error. A
  # *lazy seam* wrapper is only a warning: the wrapper's existence is not proof that production ever calls it, and the
  # seam (route_ops.py / experimental.py) deliberately declares more than the default path uses.
  for src in sorted(f for f in authored if f.startswith("tinygrad/")):
    for d in edges.get(src, []):
      if d.startswith("extra/audit/"):
        E("production_imports_audit", f"production {src} imports {d}", **{"from": src, "to": d})
      elif (rec := records.get(d)) and rec.get("role") in RESEARCH_ROLES:
        E("production_imports_research", f"production {src} imports {d} (role={rec['role']})", **{"from": src, "to": d})
    for d in dyn.get(src, []):
      if (rec := records.get(d)) and rec.get("role") in RESEARCH_ROLES:
        W("production_seam_to_research", f"lazy seam {src} declares a wrapper onto {d} (role={rec['role']}); "
          "the wrapper is not evidence that the default path calls it",
          "production must not depend on research surfaces", **{"from": src, "to": d})

  # ---- cycles ---------------------------------------------------------------------------------------------------
  comps = strongly_connected(edges)
  cross_domain_cycles = []
  for c in comps:
    doms = sorted({domain_of.get(f) for f in c if domain_of.get(f)})
    if len(doms) > 1:
      cross_domain_cycles.append({"component": c, "domains": doms})
      W("cross_domain_cycle", f"import cycle crosses domains {doms}: {c}", "cycles across domains destroy locality",
        component=c)

  # ---- fan in/out ------------------------------------------------------------------------------------------------
  for p in sorted(records):
    if len(edges.get(p, [])) > fan_threshold:
      W("high_fan_out", f"{p} imports {len(edges[p])} internal modules", "prefer deep modules with small interfaces",
        path=p, count=len(edges[p]))
    if len(inbound.get(p, [])) > fan_threshold:
      W("high_fan_in", f"{p} is imported by {len(inbound[p])} modules", "a widely-imported module is a de-facto authority",
        path=p, count=len(inbound[p]))

  # ---- declared actions: promotion / prune / test-role gates ----------------------------------------------------
  actions, workflows = man.get("actions", []), man["workflows"]
  for a in actions:
    p = a.get("path")
    if p and p not in records and not any(group_match(g, p) for g in groups):
      E("action_without_record", f"action candidate {p} has no manifest record", path=p)
    if a.get("action") == "promote" and a.get("promotion_readiness") == "ready":
      for d in edges.get(p, []):
        if (rec := records.get(d)) and rec.get("role") in RESEARCH_ROLES:
          E("ready_promotion_research_dep", f"promotion-ready {p} depends on research module {d} (role={rec['role']})",
            path=p, dep=d)
    if a.get("action") == "prune":
      live = [c for c in inbound.get(p, []) if (rec := records.get(c)) and (rec.get("default_path") or rec.get("status") in ("production", "promoted_default"))]
      if live: E("prune_candidate_has_live_consumer", f"prune candidate {p} is imported by live consumer(s) {live}", path=p, consumers=live)
      # a retained workflow is a consumer too, and it names its files by path rather than by import
      wf_live = sorted({w["workflow_id"] for w in workflows if _workflow_retained(w)
                        and p in (w.get("entry_points", []) + w.get("authority_files", []) + w.get("execution_files", []))})
      if wf_live:
        E("prune_candidate_has_retained_workflow_consumer",
          f"prune candidate {p} is named by retained workflow(s) {wf_live}", path=p, workflows=wf_live)
      if a.get("deletion_confidence") == "delete_ready":
        collected = [c for c in refs.get(p, []) if c.startswith(("test/", ".github/")) or c.endswith((".sh", ".yml", ".yaml"))]
        if collected:
          E("delete_ready_still_wired", f"delete_ready {p} is still named by {collected}", path=p, refs=collected)
        for field in ("former_purpose", "recovery", "loc_removed"):
          if not a.get(field): E("delete_ready_missing_recovery", f"delete_ready {p} lacks '{field}'", path=p, field=field)
    # the reproducer signal lives in BOTH fields; gating on only one lets a record be pruned by contradicting itself
    if (rec := records.get(p)) and "unresolved_reproducer" in (rec.get("test_role"), rec.get("status")):
      if a.get("action") == "prune" or a.get("deletion_confidence") in ("delete_ready", "delete_after_verdict_capture"):
        E("reproducer_marked_for_deletion", f"{p} is the unique reproducer for an unresolved issue but is marked "
          f"{a.get('deletion_confidence') or a.get('action')}", path=p)
    if a.get("action") == "promote" and a.get("promotion_readiness") == "ready" and not a.get("loc_deleted"):
      W("promotion_moves_without_deleting", f"promotion of {p} declares no LOC deleted through consolidation",
        "moving a file is not a LOC improvement", path=p)

  # ---- workflows -------------------------------------------------------------------------------------------------
  for wf in workflows:
    for k in WORKFLOW_FIELDS:
      if k not in wf: E("workflow_missing_field", f"workflow {wf.get('workflow_id')} missing '{k}'",
                        workflow=wf.get("workflow_id"), field=k)
    for f in wf.get("authority_files", []) + wf.get("execution_files", []):
      if not (root / f).exists(): E("workflow_stale_path", f"workflow {wf.get('workflow_id')} names missing {f}",
                                    workflow=wf.get("workflow_id"), path=f)
  authority_owner: dict[str, list[str]] = {}
  for wf in workflows:
    for k in wf.get("authority_rules", []) or []:
      authority_owner.setdefault(k, []).append(wf["workflow_id"])
  for k, wfs in sorted(authority_owner.items()):
    if len(wfs) > 1:
      reason = any(w.get("duplicate_authority_reason") for w in workflows if w["workflow_id"] in wfs)
      if not reason:
        E("workflows_duplicate_authority", f"authority rule '{k}' is encoded by workflows {sorted(wfs)} with no stated reason",
          key=k, workflows=sorted(wfs))
  phase_counts: dict[str, list[str]] = {}
  for wf in workflows:
    for ph in wf.get("phases", []): phase_counts.setdefault(ph, []).append(wf["workflow_id"])
  repeated_phases = {k: sorted(v) for k, v in sorted(phase_counts.items()) if len(v) > 1}
  for k, v in repeated_phases.items():
    if len(v) > 2:
      W("repeated_workflow_phase", f"phase '{k}' is re-implemented by {len(v)} workflows: {v}",
        "modularize execution without scattering authority", phase=k, workflows=v)

  # ---- role/location conflicts -----------------------------------------------------------------------------------
  for p, r in sorted(records.items()):
    if p.startswith("extra/") and r.get("default_path") and not r.get("promotion_decision"):
      W("extra_on_default_path", f"{p} is on the default production path from extra/ with no promotion or retention decision",
        "production behavior should live with its domain owner", path=p)
    if r.get("role") == "test" and not p.startswith(("test/", "bench/")) and "test" not in os.path.basename(p):
      W("role_location_conflict", f"{p} declares role=test but does not live in test/ or name itself a test",
        "organize around domain meaning, not accident of location", path=p)

  # ---- reporting facts -------------------------------------------------------------------------------------------
  def _scoped(seq): return [f for f in seq if scope is None or f.startswith(scope)]
  scope_files = _scoped(authored)
  by = lambda key: _agg(scope_files, records, covered_by_group, groups, loc, key)
  crossing = sorted({(s, d) for s, ds in edges.items() for d in ds
                     if scope and ((s.startswith(scope)) != (d.startswith(scope)))})

  census = {
    "audited_commit": commit_identity(root),
    "scope": scope or "<repository>",
    "manifest_scope": list(manifest_scope),
    "coverage": {
      "authored_files": len(authored), "authored_loc": sum(loc[f] for f in authored),
      "generated_files": len(generated), "generated_loc": sum(loc[f] for f in generated),
      "manifest_scope_files": len(in_manifest_scope),
      "manifest_scope_loc": sum(loc[f] for f in in_manifest_scope),
      "explicit_records": len([f for f in in_manifest_scope if f in records]),
      "group_covered": len(covered_by_group),
      "uncovered": sorted(f for f in in_manifest_scope if f not in records and f not in covered_by_group),
    },
    "loc_by": {"domain": by("domain"), "role": by("role"), "status": by("status"), "disposition": by("disposition")},
    "largest_authored_files": [{"path": f, "loc": loc[f], "domain": domain_of.get(f)}
                               for f in sorted(scope_files, key=lambda x: (-loc[x], x))[:top]],
    "default_path_footprint": {
      "files": sorted(p for p, r in records.items() if r.get("default_path")),
      "loc": sum(loc.get(p, 0) for p, r in records.items() if r.get("default_path")),
      "production_reachable_from_tinygrad": sorted(f for f in prod_reachable if scope is None or f.startswith(scope)),
    },
    "graph": {
      "edges": {f: edges[f] for f in sorted(scope_files) if edges.get(f)},
      "dynamic_seam_edges": {f: dyn[f] for f in sorted(dyn)},
      "inbound_counts": {f: len(inbound.get(f, [])) for f in sorted(scope_files)},
      "outbound_counts": {f: len(edges.get(f, [])) for f in sorted(scope_files)},
      "strongly_connected_components": comps,
      "cross_domain_cycles": cross_domain_cycles,
      "boundary_violations": boundary_violations,
      "scope_crossing_edges": [{"from": s, "to": d} for s, d in crossing],
      "no_inbound_import": sorted(f for f in scope_files if not inbound.get(f)),
      "no_reference_at_all": sorted(f for f in scope_files if not inbound.get(f) and not refs.get(f)),
    },
    "non_import_references": {f: refs[f] for f in sorted(scope_files) if refs.get(f)},
    "authority_keys": {k: sorted(v) for k, v in sorted(claims.items())},
    "workflows": workflows,
    "repeated_workflow_phases": repeated_phases,
    "reusable_asset_candidates": man.get("reusable_assets", []),
    "actions": actions,
    "investigation_backlog": man.get("investigation_backlog", []),
    "test_inventory": _test_inventory(records, covered_by_group, groups),
    "hard_errors": sorted(hard, key=lambda x: (x["error"], x.get("path", ""), x["message"])),
    "warnings": sorted(warn, key=lambda x: (x["warning"], x.get("path", ""), x["message"])),
    "loc_impact": _loc_impact(actions),
    "human_findings": man.get("human_findings", {}),
    "coverage_limitations": man.get("coverage_limitations", []),
  }
  census["verdict"] = "ORG_R1_PASS_CENSUS_PINNED" if not hard else "ORG_R1_BLOCKED_ORGANIZATION_DRIFT"
  return census

def _agg(files, records, covered_by_group, groups, loc, key):
  out: dict[str, dict] = {}
  gmap = {g["group_id"]: g for g in groups}
  for f in files:
    r = records.get(f)
    if r is None:
      gid = (covered_by_group.get(f) or [None])[0]
      r = gmap.get(gid, {})
    v = str(r.get(key, "unclassified"))
    e = out.setdefault(v, {"files": 0, "loc": 0})
    e["files"] += 1; e["loc"] += loc.get(f, 0)
  return dict(sorted(out.items()))

def _test_inventory(records, covered_by_group, groups):
  gmap = {g["group_id"]: g for g in groups}
  inv: dict[str, list[str]] = {}
  for p, r in records.items():
    if (tr := r.get("test_role")) and tr != "not_test_like": inv.setdefault(tr, []).append(p)
  for f, gids in covered_by_group.items():
    if (tr := gmap.get(gids[0], {}).get("test_role")) and tr != "not_test_like": inv.setdefault(tr, []).append(f)
  return {k: sorted(v) for k, v in sorted(inv.items())}

def _loc_impact(actions):
  gross = sum(int(a.get("loc_removed") or 0) for a in actions)
  added = sum(int(a.get("loc_added") or 0) for a in actions)
  moved = sum(int(a.get("loc_moved") or 0) for a in actions)
  return {"gross_loc_removed": gross, "replacement_loc_added": added, "net_authored_loc_reduction": gross - added,
          "loc_moved_not_reduced": moved}

# ------------------------------------------------------------------------------------------------------ report -----
def _md(c: dict) -> str:
  cov, g = c["coverage"], c["graph"]
  L = ["# Codebase organization audit", "",
       f"Audited commit: `{c['audited_commit']['described']}` (dirty: {c['audited_commit']['dirty']})",
       f"Scope: `{c['scope']}` | manifest coverage required for: {', '.join(c['manifest_scope'])}",
       f"Verdict: **{c['verdict']}** ({len(c['hard_errors'])} hard errors, {len(c['warnings'])} warnings)", "",
       "> Machine-derived facts (discovery, LOC, imports, references, coverage, boundary checks) are produced by this",
       "> script. Every purpose, role, status, disposition, action, and promotion/prune judgment below is human-authored",
       "> in `extra/audit/codebase_organization_manifest.json` and is reproduced here, not computed.", "",
       "## Coverage", "",
       f"- Authored: {cov['authored_files']} files / {cov['authored_loc']} token-bearing LOC (sz.py rules)",
       f"- Generated (reported, never manifested): {cov['generated_files']} files / {cov['generated_loc']} LOC",
       f"- Manifest scope: {cov['manifest_scope_files']} files / {cov['manifest_scope_loc']} LOC "
       f"({cov['explicit_records']} explicit records, {cov['group_covered']} covered by group rule, {len(cov['uncovered'])} uncovered)", ""]
  for label, key in (("domain", "domain"), ("role", "role"), ("status", "status"), ("disposition", "disposition")):
    L += [f"## LOC by {label}", "", f"| {label} | files | loc |", "|---|---|---|"]
    L += [f"| {k} | {v['files']} | {v['loc']} |" for k, v in c["loc_by"][key].items()] + [""]
  dpf = c["default_path_footprint"]
  L += ["## Default-path source footprint", "", f"{len(dpf['files'])} declared default-path files / {dpf['loc']} LOC.", ""]
  L += [f"- `{p}`" for p in dpf["files"]] + [""]
  L += ["## Largest authored files", "", "| loc | path | domain |", "|---|---|---|"]
  L += [f"| {r['loc']} | `{r['path']}` | {r['domain']} |" for r in c["largest_authored_files"]] + [""]
  L += ["## Duplicate authority", ""]
  dups = {k: v for k, v in c["authority_keys"].items() if len(v) > 1}
  L += ([f"- `{k}`: {v}" for k, v in dups.items()] if dups else ["None: every declared authority key has one owner."]) + [""]
  L += ["## Cross-domain dependency violations", ""]
  L += ([f"- `{v['from']}` ({v['from_domain']}) -> `{v['to']}` ({v['to_domain']})" for v in g["boundary_violations"]]
        or ["None."]) + [""]
  L += ["## Import cycles", ""]
  L += ([f"- {comp}" for comp in g["strongly_connected_components"]] or ["None."]) + [""]
  L += ["## Files with no inbound import and no reference anywhere", "",
        "An entry point, gate, probe, or operator tool may legitimately have no inbound import; this list is evidence",
        "for human classification, not a death sentence.", ""]
  L += [f"- `{p}`" for p in g["no_reference_at_all"]] + [""]
  L += ["## Test and one-off script inventory", "", "| test_role | count | files |", "|---|---|---|"]
  L += [f"| {k} | {len(v)} | {', '.join('`'+x+'`' for x in v)} |" for k, v in c["test_inventory"].items()] + [""]
  for title, pred in (("Decouple candidates", "decouple"), ("Centralize candidates", "centralize"),
                      ("Modularize / reusable-asset candidates", "modularize"), ("Reuse candidates", "reuse"),
                      ("Prune candidates", "prune")):
    rows = [a for a in c["actions"] if a.get("action") == pred]
    L += [f"## {title}", ""]
    L += ([f"- **`{a.get('path') or a.get('target')}`** -- {a.get('finding')} "
           f"(gross -{a.get('loc_removed', 0)}, +{a.get('loc_added', 0)}, net -{int(a.get('loc_removed') or 0) - int(a.get('loc_added') or 0)}) "
           f"| evidence: {'; '.join(a.get('evidence', []))} | tests: {'; '.join(a.get('tests_required', []))}"
           for a in rows] or ["None proposed at this evidence level."]) + [""]
  L += ["## Promotion candidates", ""]
  for readiness in ("ready", "blocked", "not justified"):
    rows = [a for a in c["actions"] if a.get("action") == "promote" and a.get("promotion_readiness") == readiness]
    L += [f"### {readiness}", ""]
    L += ([f"- `{a.get('path')}` -> `{a.get('promotion_target')}`: {a.get('finding')} | invariant: {a.get('invariant')} "
           f"| moved {a.get('loc_moved', 0)}, deleted {a.get('loc_deleted', 0)} | missing evidence: {a.get('evidence_missing', 'none')}"
           for a in rows] or ["None."]) + [""]
  dr = [a for a in c["actions"] if a.get("deletion_confidence") in ("delete_ready", "delete_after_verdict_capture")]
  L += ["## Deletion candidates with recovery information", "",
        "| path | class | former purpose | last campaign | replacement | commit | recovery | loc |", "|---|---|---|---|---|---|---|---|"]
  L += [f"| `{a['path']}` | {a['deletion_confidence']} | {a.get('former_purpose','')} | {a.get('last_campaign','')} | "
        f"{a.get('replacement','')} | `{c['audited_commit']['commit'][:12]}` | {a.get('recovery','')} | {a.get('loc_removed',0)} |"
        for a in dr] + [""]
  L += ["## Workflow inventory", "", "| workflow_id | domain | default path | entry points | phases |", "|---|---|---|---|---|"]
  L += [f"| {w['workflow_id']} | {w['domain']} | {w['default_path']} | {', '.join('`'+e+'`' for e in w['entry_points'])} "
        f"| {' -> '.join(w['phases'])} |" for w in c["workflows"]] + [""]
  L += ["## Repeated workflow phases", ""]
  L += [f"- `{k}`: {v}" for k, v in c["repeated_workflow_phases"].items()] + [""]
  imp = c["loc_impact"]
  L += ["## LOC impact of proposed actions", "",
        f"- Gross authored LOC removed: {imp['gross_loc_removed']}",
        f"- Replacement LOC added: {imp['replacement_loc_added']}",
        f"- **Net authored LOC reduction: {imp['net_authored_loc_reduction']}**",
        f"- LOC merely moved between directories (NOT a reduction): {imp['loc_moved_not_reduced']}", ""]
  if seq := c["human_findings"].get("recommended_sequence"):
    L += ["## Recommended sequence", ""] + [f"{i+1}. {s}" for i, s in enumerate(seq)] + [""]
  if hc := c["human_findings"].get("requires_human_classification"):
    L += ["## Files requiring human classification", ""] + [f"- {x}" for x in hc] + [""]
  if bl := c["investigation_backlog"]:
    L += ["## Investigation backlog", ""] + [f"- `{b.get('path','')}`: {b.get('question','')}" for b in bl] + [""]
  L += ["## Hard errors", ""] + ([f"- **{e['error']}**: {e['message']}" for e in c["hard_errors"]] or ["None."]) + [""]
  L += ["## Warnings", ""] + ([f"- **{w['warning']}** ({w['principle']}): {w['message']}" for w in c["warnings"]] or ["None."]) + [""]
  L += ["## Coverage limitations", ""] + [f"- {x}" for x in c["coverage_limitations"]] + [""]
  return "\n".join(L)

def main(argv=None):
  import argparse
  ap = argparse.ArgumentParser(description="Census authored-source organization against declared organizational intent.")
  ap.add_argument("--check", action="store_true", help="validate only; write nothing; exit 1 on hard errors")
  ap.add_argument("--scope", default=None, help="restrict reporting to a path prefix (dependency edges crossing it are still reported)")
  ap.add_argument("--top", type=int, default=20, help="how many largest authored files to list")
  args = ap.parse_args(argv)
  c = audit(ROOT, MANIFEST, scope=args.scope, top=args.top)
  if not args.check:
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(c, open(OUT / "latest.json", "w"), indent=2, sort_keys=True)
    json.dump({"workflows": c["workflows"], "repeated_phases": c["repeated_workflow_phases"]},
              open(OUT / "workflow_table.json", "w"), indent=2, sort_keys=True)
    json.dump({"actions": c["actions"], "reusable_asset_candidates": c["reusable_asset_candidates"],
               "investigation_backlog": c["investigation_backlog"], "loc_impact": c["loc_impact"]},
              open(OUT / "action_candidates.json", "w"), indent=2, sort_keys=True)
    open(OUT / "summary.md", "w").write(_md(c))
    print(f"wrote organization census to {OUT}")
  print("verdict:", c["verdict"])
  print(f"hard errors: {len(c['hard_errors'])}  warnings: {len(c['warnings'])}  "
        f"manifest coverage: {c['coverage']['explicit_records']}+{c['coverage']['group_covered']}"
        f"/{c['coverage']['manifest_scope_files']}")
  for e in c["hard_errors"]: print("  HARD:", e["message"])
  if c["hard_errors"]: raise SystemExit(1)

if __name__ == "__main__":
  main()
