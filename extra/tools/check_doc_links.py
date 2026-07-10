#!/usr/bin/env python3
"""Dead-link linter for the docs corpus (machine-enforces the "no dead links" rule).

Scans docs/*.md and structure/**/*.md for cross-references to other markdown
files and FAILS (exit 1) if any referenced file does not exist on disk. This
turns the "no dead links" rule in structure/Development/*-principles.md into a
machine-enforced check instead of a decorative one.

Two kinds of references are flagged:
  1. Root-anchored paths:  docs/<...>.md  or  structure/<...>.md
     (in prose, backticks, or markdown links) -> resolved from the repo root.
  2. Markdown links [text](target) whose target ends in .md and is not a URL
     -> resolved relative to the file's own directory (or the repo root if the
     target is itself root-anchored).

External URLs (http://, https://, mailto:) and in-page anchors (#...) are
ignored. Template/placeholder filenames (containing YYYYMMDD / YYYY / MMDD /
2026MMDD) are ignored -- they are documented naming patterns, not real files.

Pre-existing orphaned targets (docs deleted/archived before this check existed)
are recorded in `doc_link_baseline.txt` next to this script. The linter passes
if the only dead links are baselined, and FAILS on any NEW dead link. That keeps
the rule enforced going forward without resurrecting intentionally-archived
history. Run:  python3 extra/tools/check_doc_links.py
Regenerate the baseline (after an intentional corpus change):
                python3 extra/tools/check_doc_links.py --update-baseline
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BASELINE = HERE / "doc_link_baseline.txt"

ROOT_REF = re.compile(r"(?<![\w./-])((?:docs|structure)/[\w./-]+?\.md)")
MD_LINK = re.compile(r"\]\(([^)]+?\.md)(?:#[^)]*)?\)")
PLACEHOLDER = re.compile(r"YYYYMMDD|YYYY|MMDD|2026MMDD")


def iter_refs(path: Path):
    """Yield (lineno, raw_ref, resolved_target) for every md reference in path."""
    text = path.read_text(encoding="utf-8", errors="replace")
    for lineno, line in enumerate(text.splitlines(), start=1):
        for m in ROOT_REF.finditer(line):
            ref = m.group(1)
            yield lineno, ref, REPO_ROOT / ref
        for m in MD_LINK.finditer(line):
            ref = m.group(1)
            if ref.startswith(("http://", "https://", "mailto:", "#")):
                continue
            if ref.startswith(("docs/", "structure/")):
                target = REPO_ROOT / ref
            else:
                target = (path.parent / ref).resolve()
            yield lineno, ref, target


def corpus_files():
    return sorted(
        list((REPO_ROOT / "docs").glob("*.md"))
        + list((REPO_ROOT / "structure").rglob("*.md"))
    )


def collect_dead():
    """Return dict: normalized-missing-target -> list of 'file:line' sites."""
    dead: dict[str, list[str]] = {}
    scanned = 0
    for f in corpus_files():
        scanned += 1
        for lineno, ref, target in iter_refs(f):
            if PLACEHOLDER.search(ref):
                continue
            if target.exists():
                continue
            try:
                key = str(target.relative_to(REPO_ROOT))
            except ValueError:
                key = str(target)
            dead.setdefault(key, []).append(f"{f.relative_to(REPO_ROOT)}:{lineno}")
    return dead, scanned


def load_baseline() -> set[str]:
    if not BASELINE.exists():
        return set()
    out = set()
    for line in BASELINE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(line)
    return out


def main(argv: list[str]) -> int:
    dead, scanned = collect_dead()

    if "--update-baseline" in argv:
        header = (
            "# Known-missing doc link targets (pre-existing orphaned/archived docs).\n"
            "# The dead-link linter treats these as accepted debt and FAILS only on NEW\n"
            "# dead links. Regenerate with: python3 extra/tools/check_doc_links.py --update-baseline\n"
            "# Remove an entry when its target is restored; do not add current authority here.\n"
        )
        BASELINE.write_text(header + "\n".join(sorted(dead)) + ("\n" if dead else ""))
        print(f"wrote baseline with {len(dead)} entries -> {BASELINE.relative_to(REPO_ROOT)}")
        return 0

    baseline = load_baseline()
    new_dead = {k: v for k, v in dead.items() if k not in baseline}

    for key in sorted(new_dead):
        for site in new_dead[key]:
            print(f"{site}: dead link -> {key}  (missing)")

    print(f"\nscanned {scanned} markdown files under docs/ and structure/")
    baselined = len(dead) - len(new_dead)
    if baselined:
        print(f"note: {baselined} pre-existing dead target(s) accepted via doc_link_baseline.txt")
    if new_dead:
        print(f"FAIL: {len(new_dead)} NEW dead doc link target(s) found")
        return 1
    print("OK: no new dead doc links")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
