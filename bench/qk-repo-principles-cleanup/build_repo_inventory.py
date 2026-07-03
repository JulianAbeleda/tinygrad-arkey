#!/usr/bin/env python3
"""Whole-repo principles cleanup audit — reference-graph inventory builder (2026-06-21).

Extends the proven engines `bench/qk-active-surface-reduction/build_{inventory,docs_index}.py` from the
extra/*.py + docs/*.md surface to the WHOLE project surface, mapping every tracked file to a 10-way
recommendation taxonomy with a real import/doc/test/ledger reference graph.

Method (unchanged from the prior round, generalized):
  - import-closure of live evaluator/search roots  -> live
  - canonical-doc / ledger / test references        -> provenance kept
  - import-safety fixpoint + external-importer       -> never strand a kept import
  - all import styles (`from extra import X`, import_module, path strings)

Read-only. No deletes, no moves. Emits inventory.json + inventory.md.
Run: PYTHONPATH=. .venv/bin/python bench/qk-repo-principles-cleanup/build_repo_inventory.py
"""
import json, pathlib, re, subprocess
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-repo-principles-cleanup"

# ---------------------------------------------------------------- file universe
tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
tracked = sorted(tracked)  # splitlines() (not split()) so paths with spaces ("System Guide/") stay intact

# ---------------------------------------------------------------- vendor boundary (user: bulk-exclude at dir granularity)
VENDOR_DIR_PREFIXES = [
    "tinygrad/",                       # framework core (EXCEPT tinygrad/llm/, handled below)
    "examples/", ".github/",
    "test/backend/", "test/null/", "test/mockgpu/", "test/models/", "test/opt/",
    "test/device/", "test/speed/",
    "extra/thunder/", "extra/sqtt/", "extra/hcqfuzz/", "extra/torch_backend/", "extra/mlx_driver/",
    "extra/huggingface_onnx/", "extra/hip_gpu_driver/", "extra/remote/", "extra/tinyfs/",
    "extra/hcq/", "extra/hcq2/", "extra/fp8/", "extra/llama_kernels/", "extra/amdpci/", "extra/gemm/",
]
VENDOR_FILES = {
    "extra/bench_log.py", "extra/export_model.py", "extra/f16_decompress.py", "extra/gradcheck.py",
    "extra/lr_scheduler.py", "extra/onnx_helpers.py", "extra/training.py",
    "mkdocs.yml", "serve_docs.sh", "LICENSE", "uv.lock", "sz.py", "opencode.json",
    ".pylintrc", ".coveragerc", ".pre-commit-config.yaml",
}
# osx/shim setup scripts + mi350 shim are upstream
VENDOR_GLOBS = ("extra/setup_", "extra/test_mi350")
# upstream tinygrad docs + docs-site assets living in the flat docs/ tree
UPSTREAM_DOCS = {"docs/index.md", "docs/mnist.md", "docs/nn.md", "docs/dtypes.md", "docs/quickstart.md",
                 "docs/showcase.md", "docs/abstractions2.md", "docs/abstractions3.md", "docs/env_vars.md",
                 "docs/function.md", "docs/developer/developer.md",
                 "docs/CNAME", "docs/abstractions3.py", "docs/abstractions4.py", "docs/favicon.svg",
                 "docs/logo_tiny_dark.svg", "docs/logo_tiny_light.svg", "docs/tinygrad_intro.pdf",
                 "docs/tinygrad_vs_others.png"}

def is_vendor(p: str) -> bool:
    if p.startswith("tinygrad/llm/"): return False           # project core runtime
    if any(p.startswith(v) for v in VENDOR_DIR_PREFIXES): return True
    if p in VENDOR_FILES or p in UPSTREAM_DOCS: return True
    if any(p.startswith(g) for g in VENDOR_GLOBS): return True
    if p.startswith(("docs/showcase/", "docs/developer/")): return True   # upstream docs-site asset subtrees
    return False

# the builder's own report outputs are the audit ARTIFACT, not a subject of it; excluding them keeps the run
# idempotent (inventory.json records LOC, so classifying itself would never reach a fixed point).
SELF_OUTPUTS = {"bench/qk-repo-principles-cleanup/inventory.json", "bench/qk-repo-principles-cleanup/inventory.md"}
# Legacy FILE_INDEX.md outputs were generated navigation artifacts, not inventory subjects. Keep excluding any
# leftover/restored copy so the inventory stays about source/docs/tests, not generated navigation.
project = [p for p in tracked if not is_vendor(p) and p not in SELF_OUTPUTS and not p.endswith("FILE_INDEX.md")]
vendor = [p for p in tracked if is_vendor(p)]

# ---------------------------------------------------------------- helpers
def loc(p):
    try: return (ROOT / p).read_text(errors="ignore").count("\n") + 1
    except Exception: return 0
def text(p):
    try: return (ROOT / p).read_text(errors="ignore")
    except Exception: return ""

project_py = [p for p in project if p.endswith(".py")]
# the import-reference graph + per-file script rows cover ONLY the fork's perf surface (extra top-level + core
# runtime). test/*.py and bench/*.py get their own rows (KEEP_TEST / bench loop) and must NOT be run through the
# extra-script delete graph (they are importERS, not delete-eligible probes).
graph_py = [p for p in project_py if p.startswith("extra/") or p.startswith("tinygrad/llm/")]
docs_md = [p for p in project if p.startswith("docs/") and p.endswith(".md")]
test_py = [p for p in project if p.startswith("test/") and p.endswith(".py")]
bench_files = [p for p in project if p.startswith("bench/")]
structure_files = [p for p in project if p.startswith("structure/")]

# ---------------------------------------------------------------- python reference graph (reuse prior engine, generalized)
# stem -> path for the graph surface so we resolve `extra.X`, `tinygrad.llm.X`, bare imports
stem_to_path = {}
for p in graph_py:
    stem_to_path.setdefault(pathlib.Path(p).stem, p)

imp_re = re.compile(r"from extra\.([a-z0-9_]+) import|import extra\.([a-z0-9_]+)\b|"
                    r"import_module\([\"']extra\.([a-z0-9_]+)[\"']\)|"
                    r"from tinygrad\.llm\.([a-z0-9_]+) import|import tinygrad\.llm\.([a-z0-9_]+)\b")
fromextra_re = re.compile(r"from extra import\s+\(?([a-z0-9_,\s]+)\)?")
ptext = {p: text(p) for p in graph_py}
imports = {}
for p in graph_py:
    t = ptext[p]
    mods = {g for m in imp_re.finditer(t) for g in m.groups() if g}
    for fe in fromextra_re.finditer(t):
        for name in re.split(r"[,\s]+", fe.group(1)):
            name = name.split(" as ")[0].strip()
            if name and not name[0].isdigit(): mods.add(name)
    imports[p] = {m for m in mods if m in stem_to_path}

imported_by = defaultdict(list)
for p in graph_py:
    for m in imports[p]:
        imported_by[stem_to_path[m]].append(p)
# path-string (subprocess) edges among graph python
for p in graph_py:
    for other in graph_py:
        if other != p and other in ptext[p] and p not in imported_by[other]:
            imported_by[other].append(p)

# ---------------------------------------------------------------- live closure (evaluator/search roots + ledger-cited)
LEDGERS = ["bench/qk-decode-eval/candidates.json", "bench/qk-decode-eval/binding_templates.json",
           "bench/qk-lifecycle-search/search_candidates.json", "bench/qk-lifecycle-search/refutations.json",
           "bench/qk-lifecycle-search/generated_candidates.json", "bench/qk-decode-eval/schema.json",
           "bench/qk-lifecycle-search/search_policy.json", "bench/qk-lifecycle-search/evaluator_contract.json",
           "bench/qk-lifecycle-search/runner_bindings.json", "bench/qk-lifecycle-search/templates.json"]
ledger_text = {l: text(l) for l in LEDGERS if (ROOT / l).exists()}
roots = {"extra/qk_decode_eval.py", "extra/qk_lifecycle_search_loop.py", "extra/qk_candidate_template_gen.py",
         "extra/qk_policy_consistency_check.py", "extra/qk_decode_runtime_overhead.py", "extra/qk_flash_decode.py",
         "extra/qk_clock_pin.py", "extra/qk_harness_contract.py", "extra/qk_nll_eval.py"}
for tt in ledger_text.values():
    for m in re.findall(r"extra/qk_[a-z0-9_]+\.py", tt): roots.add(m)
roots = {r for r in roots if r in graph_py}
live, stack = set(), list(roots)
while stack:
    s = stack.pop()
    if s in live: continue
    live.add(s)
    for m in imports.get(s, ()):
        if stem_to_path[m] not in live: stack.append(stem_to_path[m])

# ---------------------------------------------------------------- reference corpora for basename search
doc_text = {d: text(d) for d in docs_md}
doc_text["bench/README.md"] = text("bench/README.md")
doc_text["structure/Development/performance-primitive-research-principles.md"] = text(
    "structure/Development/performance-primitive-research-principles.md")
test_text = {t: text(t) for t in test_py}
CANON_DOCS = {"docs/current-project-state-handoff-20260624.md", "docs/README.md", "bench/README.md",
              "structure/Development/performance-primitive-research-principles.md",
              "structure/Development/coding-principles.md",
              "docs/decode-campaign-final-synthesis-20260623.md",
              "docs/prefill-decode-next-workstreams-codex-scope-20260624.md"}

def refs(corpus, p, stem, want_pyname=True):
    needle_py = stem + ".py"
    return [k for k, t in corpus.items() if k != p and (p in t or (want_pyname and needle_py in t))]

# shared-lib heuristic: imported by >=3 kept project py
SHARED_LIB_MIN = 3

# ---------------------------------------------------------------- per-file classification
rows = []

def add(path, subsystem, typ, recommendation, reason, score, rd, rk, extra=None):
    r = {"path": path, "type": typ, "loc": loc(path), "subsystem": subsystem,
         "recommendation": recommendation, "reason": reason, "principle_score": score,
         "risk_if_deleted": rd, "risk_if_kept": rk}
    if extra: r.update(extra)
    rows.append(r)

# --- vendor (dir-granularity summary rows) ---
vendor_by_dir = defaultdict(list)
for p in vendor:
    if p.startswith("tinygrad/"): key = "tinygrad/ (framework core)"
    elif p.startswith("examples/"): key = "examples/ (upstream)"
    elif p.startswith("test/"): key = "test/" + p.split("/")[1] + "/ (upstream)"
    elif p.startswith("extra/") and "/" in p[6:]: key = "extra/" + p.split("/")[1] + "/ (upstream)"
    elif p.startswith("docs/"): key = "docs/ upstream tinygrad pages"
    else: key = "root upstream config/util"
    vendor_by_dir[key].append(p)
for key, members in sorted(vendor_by_dir.items()):
    add(key, "vendor", "vendor_dir", "IGNORE_EXTERNAL_VENDOR",
        f"{len(members)} upstream tinygrad files; not fork work. Acceptance gate forbids tinygrad/ edits.",
        5, "would diverge from upstream / break framework", "none — out of audit scope",
        {"member_count": len(members), "members_sample": members[:6]})

# --- project python: extra top-level + tinygrad/llm core (test/bench .py handled in their own loops) ---
prelim = {}
for p in graph_py:
    stem = pathlib.Path(p).stem
    drefs = refs(doc_text, p, stem)
    trefs = refs(test_text, p, stem)
    lrefs = [l for l in ledger_text if (p in ledger_text[l] or (stem + ".py") in ledger_text[l])]
    canon = [d for d in drefs if d in CANON_DOCS]
    in_live = p in live
    is_root = p in roots
    n_importers = len(set(imported_by[p]) - {p})
    if p.startswith("tinygrad/llm/"):
        rec, reason, score = "KEEP_CORE", "core runtime / model path (decode hot path, CLI, gguf loader)", 10
    elif is_root:
        rec, reason, score = "KEEP_LIVE_TOOLING", "live evaluator/search/runner ROOT entrypoint", 9
    elif n_importers >= SHARED_LIB_MIN:
        rec, reason, score = "KEEP_LIBRARY_HELPER", f"shared lib imported by {n_importers} project scripts", 9
    elif in_live:
        rec, reason, score = "KEEP_LIVE_TOOLING", "in import-closure of live evaluator/search/runner roots", 9
    elif canon or lrefs or trefs:
        rec, reason, score = "ARCHIVE_PROVENANCE", (
            f"cited by canonical/ledger/test ({len(canon)} canon, {len(lrefs)} ledger, {len(trefs)} test) — kept as provenance/asset"), 7
    elif drefs:
        rec, reason, score = "ARCHIVE_PROVENANCE", f"cited only by {len(drefs)} dated result doc(s) — historical provenance, verdict captured there", 6
    else:
        rec, reason, score = "DELETE", "no reference in ANY doc/test/ledger and not imported (proven zero-reference scratch)", 2
    prelim[p] = rec
    rd = "none — conclusion captured in doc/ledger; git history preserves" if rec in ("DELETE", "ARCHIVE_PROVENANCE") else \
         ("BREAKS live evaluator/search route" if rec in ("KEEP_LIVE_TOOLING", "KEEP_CORE") else "breaks importers")
    rk = "false authority / sprawl if mistaken for current" if rec == "ARCHIVE_PROVENANCE" else \
         ("misleads future search as stale scratch" if rec == "DELETE" else "none — live/authority")
    add(p, "extra_qk_tooling" if p.startswith("extra/") else "core_runtime", "script", rec, reason, score, rd, rk,
        {"stem": stem, "imports": sorted(imports[p]), "imported_by": sorted(set(imported_by[p]) - {p}),
         "in_live_closure": in_live, "canonical_doc_refs": canon,
         "dated_doc_refs": [d for d in drefs if d not in CANON_DOCS][:8],
         "test_refs": trefs, "ledger_refs": lrefs})

# IMPORT-SAFETY FIXPOINT + external-importer protection (no kept script left with a broken import)
row_by_path = {r["path"]: r for r in rows}
ext_protect = {}
for p in graph_py:
    if prelim.get(p) != "DELETE": continue
    stem = pathlib.Path(p).stem
    pat = (f"from extra\\.{stem} import|import extra\\.{stem}\\b|from extra import .*\\b{stem}\\b|"
           f"import_module\\([\"']extra\\.{stem}[\"']\\)|extra/{stem}\\.py|{stem}\\.py")
    hits = subprocess.run(["grep", "-rlE", pat, "--include=*.py", "--exclude-dir=.git", "."],
                          cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    hits = [h.lstrip("./") for h in hits if h.lstrip("./") not in (p,) and "qk-repo-principles-cleanup" not in h
            and "qk-active-surface-reduction" not in h]
    if hits: ext_protect[p] = hits[:5]
for p, hits in ext_protect.items():
    prelim[p] = "ARCHIVE_PROVENANCE"
    r = row_by_path[p]; r["recommendation"] = "ARCHIVE_PROVENANCE"
    r["reason"] = f"external importer (test/model/non-qk): {hits} — keep to avoid broken import"
    r["risk_if_deleted"] = "breaks an external importer"
changed = True
while changed:
    changed = False
    for p in graph_py:
        if prelim.get(p) != "DELETE": continue
        kept = [i for i in set(imported_by[p]) - {p} if prelim.get(i) != "DELETE"]
        if kept:
            prelim[p] = "ARCHIVE_PROVENANCE"
            r = row_by_path[p]; r["recommendation"] = "ARCHIVE_PROVENANCE"
            r["reason"] = f"shared utility imported by {len(kept)} kept script(s) {kept[:3]} — keep to avoid broken imports"
            r["risk_if_deleted"] = "breaks importers"; changed = True

# --- docs (reuse canonical/current/provenance classification) ---
def date_of(n):
    m = re.search(r"20\d{6}|20\d{2}-\d{2}-\d{2}", n)
    return m.group(0) if m else ""
canon_corpus = "\n".join(text(c) for c in CANON_DOCS)
for d in docs_md:
    name = pathlib.Path(d).name
    cited = (d in canon_corpus) or (name in canon_corpus)
    dt = date_of(name); day = re.sub(r"\D", "", dt)[:8]
    stale = "STALE" in doc_text.get(d, "")[:400]
    if d in CANON_DOCS:
        rec, reason, score, status = "KEEP_DOC_AUTHORITY", "canonical authority doc", 10, "canonical"
    elif stale:
        rec, reason, score, status = "ARCHIVE_PROVENANCE", "self-marked STALE — historical, must not be read as current", 4, "stale"
    elif cited or day == "20260621" or not dt:
        rec, reason, score, status = "KEEP_DOC_AUTHORITY", "current (cited by canonical / dated today / evergreen)", 8, "current"
    else:
        rec, reason, score, status = "ARCHIVE_PROVENANCE", f"dated {dt}, not cited by canonical — chronological probe log provenance", 6, "provenance"
    add(d, "docs", "doc", rec, reason, score,
        "loses historical verdict (mitigated by git + provenance-index)" if status in ("provenance", "stale") else "loses navigation/authority",
        "false authority if mistaken for current — needs index/banner" if status in ("provenance", "stale") else "none",
        {"doc_status": status, "date": dt, "cited_by_canonical": cited})

# --- bench (HARNESS_GUIDE classification) ---
LIVE_BENCH_PREFIX = ("bench/qk-decode-eval/", "bench/qk-lifecycle-search/")
for p in bench_files:
    if p.endswith(".py"): continue  # the 2 builder .py handled as project_py? no — they live under bench; classify here
    if any(p.startswith(x) for x in LIVE_BENCH_PREFIX):
        rec, reason, score, sub = "KEEP_LIVE_TOOLING", "live evaluator/search ledger or contract (HARNESS_GUIDE authority surface)", 9, "evaluator_search_ledger"
    elif p.endswith(("README.md", "matrix-summary.md", "profile-report.md", "summary.md", "decision.json")):
        rec, reason, score, sub = "ARCHIVE_PROVENANCE", "durable verdict artifact (force-added; rest of bench/** gitignored)", 6, "bench_artifact"
    else:
        rec, reason, score, sub = "ARCHIVE_PROVENANCE", "tracked durable bench artifact (gitignore force-add)", 5, "bench_artifact"
    abspath = "/home/ubuntu" in text(p)
    add(p, sub, "bench_artifact", rec, reason + (" — ⚠ contains absolute /home/ubuntu path" if abspath else ""),
        score - (2 if abspath else 0), "loses verdict provenance (git preserves)", "non-portable" if abspath else "false authority if stale",
        {"absolute_path": abspath})
# the 2 bench/*.py builders
for p in bench_files:
    if not p.endswith(".py"): continue
    add(p, "audit_tooling", "script", "KEEP_LIVE_TOOLING", "audit inventory/index builder (read-only, re-runnable)", 8,
        "loses re-runnable audit tooling", "none", {"stem": pathlib.Path(p).stem})

# --- tests (project) ---
for p in test_py:
    add(p, "test", "test", "KEEP_TEST", "fork test suite (boundary/byte-proof/SSOT)", 8,
        "loses regression/boundary coverage", "none", {"stem": pathlib.Path(p).stem})
# project non-py test files (README/data)
for p in project:
    if p.startswith("test/") and not p.endswith(".py"):
        add(p, "test", "test_data", "KEEP_TEST", "fork test fixture/readme", 7, "loses test data", "none")

# --- structure ---
for p in structure_files:
    stale = "STALE" in text(p)[:400]
    if p.endswith(("coding-principles.md", "performance-primitive-research-principles.md")):
        rec, reason, score = "KEEP_DOC_AUTHORITY", "canonical principles authority", 10
    elif "cache/repo-" in p and stale:
        rec, reason, score = "ARCHIVE_PROVENANCE", "STALE cache (predates active-surface reduction) — regenerate or banner", 4
    elif re.search(r"20\d{2}-?\d{2}-?\d{2}", p) or "audit" in p or "manifest" in p or "round" in p:
        rec, reason, score = "ARCHIVE_PROVENANCE", "dated sub-arc handoff/audit — provenance", 6
    else:
        rec, reason, score = "KEEP_DOC_AUTHORITY", "structure/role/convention layer (current)", 8
    add(p, "structure", "doc", rec, reason, score,
        "loses principle/role authority" if rec == "KEEP_DOC_AUTHORITY" else "loses historical handoff",
        "none" if rec == "KEEP_DOC_AUTHORITY" else "false authority if mistaken for current", {"stale": stale})

# --- catch-all: every remaining project file gets exactly one row (full-coverage acceptance gate) ---
covered = {r["path"] for r in rows}
for p in project:
    if p in covered: continue
    if p.startswith("docs/"):                                  # fork doc artifacts (e.g. docs/artifacts/*.json)
        add(p, "docs", "artifact", "ARCHIVE_PROVENANCE", "fork doc artifact (data backing a result doc)", 6,
            "loses artifact provenance (git preserves)", "false authority if stale", {})
    elif p.startswith(("bench/", "test/", "structure/")):
        add(p, p.split("/")[0], "artifact", "ARCHIVE_PROVENANCE", "tracked project artifact", 5,
            "loses provenance (git preserves)", "false authority if stale", {})
    else:
        add(p, "root_config", "config", "KEEP_CORE", "project root config / entry doc", 8,
            "breaks build/orientation", "none")

# ---------------------------------------------------------------- summary
counts = Counter(r["recommendation"] for r in rows)
by_sub = defaultdict(Counter)
for r in rows: by_sub[r["subsystem"]][r["recommendation"]] += 1
delete_set = sorted(r["path"] for r in rows if r["recommendation"] == "DELETE")
summary = {
    "generated": "2026-06-24", "git_head": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                                           capture_output=True, text=True).stdout.strip(),
    "total_tracked": len(tracked), "project_rows": len([r for r in rows if r["subsystem"] != "vendor"]),
    "vendor_dirs": len(vendor_by_dir), "vendor_files": len(vendor),
    "recommendation_counts": dict(counts),
    "by_subsystem": {k: dict(v) for k, v in sorted(by_sub.items())},
    "live_roots": sorted(roots), "canonical_docs": sorted(CANON_DOCS),
    "delete_candidates": delete_set,
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "inventory.json").write_text(json.dumps({"summary": summary, "files": rows}, indent=2))

# markdown index
L = ["# Whole-Repo Principles Cleanup — Inventory", "",
     f"HEAD `{summary['git_head']}` · {summary['total_tracked']} tracked · "
     f"{summary['project_rows']} project rows · {summary['vendor_files']} vendor files in {summary['vendor_dirs']} dirs", "",
     "## By recommendation", ""]
for k, v in sorted(counts.items(), key=lambda x: -x[1]): L.append(f"- **{k}**: {v}")
L += ["", "## By subsystem", "", "| subsystem | " + " | ".join(sorted({r['recommendation'] for r in rows})) + " |",
      "|---|" + "---|" * len(sorted({r['recommendation'] for r in rows}))]
allrec = sorted({r["recommendation"] for r in rows})
for sub, c in sorted(by_sub.items()):
    L.append("| " + sub + " | " + " | ".join(str(c.get(rr, "")) for rr in allrec) + " |")
L += ["", f"## DELETE candidates ({len(delete_set)}) — proof: no importer/doc/test/ledger ref", ""]
for d in delete_set: L.append(f"- `{d}`")
(OUT / "inventory.md").write_text("\n".join(L) + "\n")

print(json.dumps(summary, indent=2))
