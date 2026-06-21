#!/usr/bin/env python3
"""Per-folder FILE_INDEX.md generator — so each folder shows which files are LIVE vs provenance + a description.

Reads the whole-repo inventory (bench/qk-repo-principles-cleanup/inventory.json) and writes one FILE_INDEX.md
per section-root (extra/, tinygrad/llm/, test/, structure/, docs/, and each bench/ project subdir). Each lists
its subtree's project files with a live flag, recommendation, LOC, and a one-line description extracted from the
file's OWN docstring/heading (so it cannot drift from the file). Read-only w.r.t. source; writes only FILE_INDEX.md.

Run: PYTHONPATH=. .venv/bin/python bench/qk-repo-principles-cleanup/build_folder_indexes.py
"""
from __future__ import annotations
import ast, json, pathlib, re, subprocess
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
INV = json.load(open(ROOT / "bench/qk-repo-principles-cleanup/inventory.json"))
rows = [r for r in INV["files"] if r["type"] != "vendor_dir"]

LIVE_RECS = {"KEEP_CORE", "KEEP_LIVE_TOOLING", "KEEP_LIBRARY_HELPER"}

CODE_LABEL = {".hip": "HIP kernel source", ".cpp": "C++ source", ".h": "C/C++ header", ".sh": "shell script",
              ".tex": "LaTeX source", ".html": "HTML", ".ver": "linker version script", ".pdf": "PDF",
              ".svg": "SVG asset", ".csv": "CSV data", ".txt": "text", ".jsonl": "JSONL records",
              ".toml": "TOML config", ".lock": "lockfile", ".cfg": "config", ".yml": "YAML config"}

def describe(path: str) -> str:
  """One-line description from the file's OWN header (docstring / heading / json shape / first real comment)."""
  p = ROOT / path
  ext = pathlib.PurePosixPath(path).suffix
  try:
    if ext == ".py":
      txt = p.read_text(errors="ignore")
      doc = ast.get_docstring(ast.parse(txt))
      if doc: return doc.strip().splitlines()[0]
      for ln in txt.splitlines()[:12]:                       # fallback: first REAL # comment (skip embedded C)
        s = ln.strip()
        if (s.startswith("# ") and not s.startswith("#!") and "coding:" not in s
            and "include" not in s and not set(s) <= set("# -=")):
          return s.lstrip("# ").strip()
      defs = [(m.group(1), m.group(2)) for ln in txt.splitlines()
              if (m := re.match(r"(class|def)\s+(\w+)", ln))]
      pub = [d for d in defs if not d[1].startswith("_")]    # prefer a public symbol over a private helper
      if pub or defs:
        kind, name = (pub or defs)[0]
        extra = f" (+{len(pub or defs) - 1} more)" if len(pub or defs) > 1 else ""
        return f"(no docstring; {kind} {name}{extra})"
      return "(no docstring)"
    if ext == ".md":
      for ln in p.read_text(errors="ignore").splitlines():
        s = ln.strip().lstrip("#").strip()
        if s and not s.startswith(">"): return s
      return "(empty md)"
    if ext == ".json":
      d = json.loads(p.read_text(errors="ignore"))
      if isinstance(d, dict):
        if isinstance(d.get("description"), str): return d["description"]
        return "JSON: " + ", ".join(list(d.keys())[:6])
      if isinstance(d, list): return f"JSON list ({len(d)} items)"
      return "(json)"
    if ext in CODE_LABEL:                                     # code/asset: a clean // or # comment, else a label
      if ext in (".pdf", ".svg", ".lock"): return CODE_LABEL[ext]
      for ln in p.read_text(errors="ignore").splitlines()[:20]:
        s = ln.strip()
        if s.startswith(("//", "/*", "%", "<!--")) or (s.startswith("# ") and "include" not in s):
          c = re.sub(r"^[/#*%<!-]+\s*", "", s).strip()
          if c and "include" not in c: return c[:120]
      return CODE_LABEL[ext]
  except Exception as e:
    return f"(unreadable: {type(e).__name__})"
  return CODE_LABEL.get(ext, f"({ext[1:] or 'file'})")

def section_root(path: str) -> str:
  if path.startswith("tinygrad/llm/"): return "tinygrad/llm"
  if path.startswith("extra/"): return "extra"
  if path.startswith("test/"): return "test"
  if path.startswith("structure/"): return "structure"
  if path.startswith("docs/"): return "docs"
  if path.startswith("bench/"):
    parts = path.split("/")
    return "/".join(parts[:2]) if len(parts) > 2 else "bench"
  return "."   # repo root

groups: dict[str, list] = defaultdict(list)
for r in rows:
  groups[section_root(r["path"])].append(r)

# always index the main section roots; for bench project subdirs only index multi-file folders (a 1-file folder's
# status is self-evident + already in inventory.json) to avoid one-row index sprawl.
SECTION_ROOTS = {"extra", "tinygrad/llm", "test", "structure", "docs", "bench"}

# remove stale FILE_INDEX.md from a previous run so dropped/renamed folders don't keep an orphan index
for old in subprocess.run(["git", "ls-files", "*FILE_INDEX.md"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines():
  (ROOT / old).unlink(missing_ok=True)

written = []
for root, members in sorted(groups.items()):
  if root == ".": continue                                   # root config files are covered by repo-map
  if len(members) < 2 and root not in SECTION_ROOTS: continue
  CURRENT_RECS = ("KEEP_DOC_AUTHORITY", "KEEP_TEST")
  def marker(r): return "●" if r["recommendation"] in LIVE_RECS else ("◐" if r["recommendation"] in CURRENT_RECS else "○")
  members = sorted(members, key=lambda r: ({"●": 0, "◐": 1, "○": 2}[marker(r)], r["path"]))
  n_live = sum(1 for r in members if marker(r) == "●")
  n_cur = sum(1 for r in members if marker(r) == "◐")
  n_prov = sum(1 for r in members if marker(r) == "○")
  rootdir = ROOT / root
  rel = lambda p: p[len(root) + 1:] if p.startswith(root + "/") else p
  lead = (" This is also the high-level directory of the whole `bench/` tree — every subfolder is rolled up below "
          "(single-file ones inline, multi-file ones pointing to their own `FILE_INDEX.md`)." if root == "bench" else "")
  L = [f"# {root}/ — file index",
       "",
       f"Mechanized index of {len(members)} project files: **{n_live} ● live** (runtime/tooling/library), "
       f"**{n_cur} ◐ current** (authority doc / test), **{n_prov} ○ provenance** (historical, kept as a refutation "
       "asset). Generated by `bench/qk-repo-principles-cleanup/build_folder_indexes.py` from `inventory.json` (the "
       "SSOT) — do not hand-edit. Vendor/upstream files are omitted (see repo-map.md)." + lead,
       "",
       "| file | | recommendation | LOC | description |",
       "|---|---|---|---:|---|"]
  for r in members:
    desc = describe(r["path"]).replace("|", "\\|")[:140]
    L.append(f"| `{rel(r['path'])}` | {marker(r)} | {r['recommendation']} | {r.get('loc','')} | {desc} |")
  L.append("")

  # bench root doubles as the high-level directory of the whole bench tree: roll up the single-file subfolders
  # (which get no index of their own) and point to the multi-file ones' indexes.
  if root == "bench":
    subs = {k: v for k, v in groups.items() if k.startswith("bench/")}
    singles = sorted((k, v[0]) for k, v in subs.items() if len(v) == 1)
    multis = sorted((k, v) for k, v in subs.items() if len(v) >= 2)
    if singles:
      L += [f"## Single-file subfolders ({len(singles)}) — documented here (no separate index)", "",
            "| subfolder | | recommendation | description |", "|---|---|---|---|"]
      for k, r in singles:
        desc = describe(r["path"]).replace("|", "\\|")[:130]
        L.append(f"| `{k[len('bench/'):]}/{pathlib.PurePosixPath(r['path']).name}` | {marker(r)} | {r['recommendation']} | {desc} |")
      L.append("")
    if multis:
      L += [f"## Multi-file subfolders ({len(multis)}) — see each folder's own `FILE_INDEX.md`", "",
            "| subfolder | files | ● live |", "|---|---:|---:|"]
      for k, v in multis:
        L.append(f"| `{k[len('bench/'):]}/` | {len(v)} | {sum(1 for r in v if marker(r) == '●')} |")
      L.append("")

  rootdir.mkdir(parents=True, exist_ok=True)
  (rootdir / "FILE_INDEX.md").write_text("\n".join(L))
  written.append((f"{root}/FILE_INDEX.md", len(members), n_live))

# force-add the gitignored bench ones so they are durable like the other bench artifacts
bench_idx = [f"{w[0]}" for w in written if w[0].startswith("bench/")]
if bench_idx:
  subprocess.run(["git", "add", "-f", *bench_idx], cwd=ROOT)

print(f"wrote {len(written)} FILE_INDEX.md files")
for path, n, live in written:
  print(f"  {path}: {n} files ({live} live)")
