#!/usr/bin/env python3
"""2026-06-24 docs declutter: move all non-current docs/*.md into docs/archive/ and rewrite live-surface
references so nothing dangles. History preserved (git mv, never deleted).

KEEP = curated current/canonical + upstream tinygrad docs (stay in docs/ root). Everything else moves.
Live-surface (refs rewritten to point at docs/archive/...): docs/README.md, the KEEP docs, tracked bench
*.md, and the user's MEMORY.md + memory/*.md. Archived docs are NOT individually rewritten (sibling refs
among them stay bare-filename and still resolve within docs/archive/).

Run dry first:  python3 bench/qk-docs-archive/run_archive.py --dry-run
Execute:        python3 bench/qk-docs-archive/run_archive.py
The policy-check's CANONICAL paths are repointed separately (extra/qk_policy_consistency_check.py)."""
from __future__ import annotations
import pathlib, re, subprocess, sys

ROOT = pathlib.Path("/home/ubuntu/tinygrad-arkey")
MEMORY_DIR = pathlib.Path("/home/ubuntu/.claude/projects/-home-ubuntu/memory")
DRY = "--dry-run" in sys.argv

# upstream tinygrad docs living in docs/ root — never touch
UPSTREAM = {"index.md", "mnist.md", "nn.md", "dtypes.md", "quickstart.md", "showcase.md",
            "abstractions2.md", "abstractions3.md", "env_vars.md", "function.md", "developer.md"}
# curated current/canonical fork docs that stay in docs/ root
KEEP_CURATED = {
    "README.md",
    "current-project-state-handoff-20260624.md",
    "provenance-index-20260624.md",            # generated below; stays in root
    "prefill-baseline-confirmed-aggressive-bound-handoff-20260624.md",
    "decode-q4k-gemv-warp-promotion-result-20260624.md",
    "decode-campaign-final-synthesis-20260623.md",
    "gpu-lifecycle-primitive-coverage-tracker-20260624.md",
    "decode-parity-no-regression-audit-result-20260623.md",
    "prefill-long-context-no-regression-audit-result-20260623.md",
    "prefill-eightwave-promotion-result-20260624.md",
    "post-owned-attention-promotion-synthesis-20260623.md",
    "three-lane-completion-result-20260623.md",
    "prefill-decode-next-workstreams-codex-scope-20260624.md",
    "decode-aggressive-target-proof-scope-20260624.md",
    "prefill-aggressive-target-proof-scope-20260624.md",
}
KEEP = KEEP_CURATED | UPSTREAM

docs = sorted(p.name for p in ROOT.glob("docs/*.md"))
MOVE = [d for d in docs if d not in KEEP]
MOVESET = set(MOVE)


def rewrite(text: str) -> tuple[str, int]:
  """Repoint references to any moved doc into docs/archive/. Idempotent."""
  n = 0
  for f in MOVESET:
    esc = re.escape(f)
    # path-style refs: docs/F and ../docs/F  ->  docs/archive/F  (skip already-archived)
    text, c1 = re.subn(r"docs/(?!archive/)" + esc, "docs/archive/" + f, text)
    # bare filename token (e.g. `F`): not preceded by a path/word char  ->  archive/F
    text, c2 = re.subn(r"(?<![\w./-])" + esc, "archive/" + f, text)
    n += c1 + c2
  return text, n


def live_files() -> list[pathlib.Path]:
  out = [ROOT / "docs/README.md"]
  out += [ROOT / "docs" / f for f in KEEP_CURATED if f != "provenance-index-20260624.md"
          and (ROOT / "docs" / f).exists()]
  tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "bench"],
                           capture_output=True, text=True).stdout.split()
  out += [ROOT / t for t in tracked if t.endswith(".md")]
  if MEMORY_DIR.exists():
    out += sorted(MEMORY_DIR.glob("*.md"))
  seen, uniq = set(), []
  for p in out:
    if p.exists() and p not in seen:
      seen.add(p)
      uniq.append(p)
  return uniq


PROVENANCE = """# Docs Provenance Index (2026-06-24)

After the 2026-06-24 declutter, the fork's historical `docs/*.md` (the full 06-16→06-22 probe log,
superseded results/handoffs, and completed scopes) were **moved into `docs/archive/`** (git mv, nothing
deleted) and references were rewritten. `docs/` root now holds only current/canonical + active docs.

## Authority order (read these; ignore docs/archive/ unless tracing provenance)
1. `docs/current-project-state-handoff-20260624.md` — canonical current state
2. `docs/README.md` — curated navigation map
3. `bench/README.md` — bench/evaluator map
4. `docs/decode-campaign-final-synthesis-20260623.md` — how decode reached llama parity
5. `docs/prefill-decode-next-workstreams-codex-scope-20260624.md` — next-work map

## Provenance
`docs/archive/` holds {n_arch} superseded docs — the chronological probe log; their verdicts are folded into
the canonical docs above. Kept for history, **not authority**. (Refs among archived docs stay bare-filename;
they resolve as siblings within `docs/archive/`.)
"""


def main() -> int:
  live = live_files()
  print(f"docs/*.md total: {len(docs)} | KEEP: {len(docs) - len(MOVE)} | MOVE: {len(MOVE)}")
  # per-live-file rewrite counts (dry preview)
  total_refs = 0
  preview = []
  for p in live:
    _, c = rewrite(p.read_text(errors="ignore"))
    if c:
      total_refs += c
      preview.append((c, str(p.relative_to(ROOT) if ROOT in p.parents else p)))
  preview.sort(reverse=True)
  print(f"live-surface files: {len(live)} | files with refs to rewrite: {len(preview)} | total refs: {total_refs}")
  for c, name in preview[:25]:
    print(f"   {c:5d}  {name}")
  if len(preview) > 25:
    print(f"   ... +{len(preview) - 25} more files")

  if DRY:
    print("\n[DRY RUN] no changes made. MOVE list (first 20):")
    for d in MOVE[:20]:
      print("   docs/ ->  docs/archive/ ", d)
    print(f"   ... {len(MOVE)} total")
    return 0

  # 1) git mv
  (ROOT / "docs/archive").mkdir(exist_ok=True)
  for d in MOVE:
    subprocess.run(["git", "-C", str(ROOT), "mv", f"docs/{d}", f"docs/archive/{d}"], check=True)
  print(f"moved {len(MOVE)} docs -> docs/archive/")

  # 2) rewrite live surface
  changed = 0
  for p in live:
    new, c = rewrite(p.read_text(errors="ignore"))
    if c:
      p.write_text(new)
      changed += 1
  print(f"rewrote refs in {changed} live-surface files ({total_refs} refs)")

  # 3) regenerate provenance index
  (ROOT / "docs/provenance-index-20260624.md").write_text(PROVENANCE.format(n_arch=len(MOVE)))
  print("wrote docs/provenance-index-20260624.md")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
