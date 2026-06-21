#!/usr/bin/env python3
"""Phase 0 reference-graph inventory for the perf-probe active-surface reduction.
Classifies every perf/probe script as live / provenance / manual_review / delete by a real import + doc + ledger
reference graph, with an IMPORT-SAFETY FIXPOINT (a script is delete-eligible only if every importer is also delete,
so no kept script is left with a broken import). Emits inventory.json + inventory.md. Read-only (no deletes)."""
import json, pathlib, re
from collections import Counter
ROOT = pathlib.Path("/home/ubuntu/tinygrad-arkey")
OUT = ROOT / "bench/qk-active-surface-reduction"

scripts = sorted({p.relative_to(ROOT).as_posix() for p in ROOT.glob("extra/qk_*.py")})
for s in ("extra/lds_attention_tile.py", "extra/llm_generate.py"):
    if (ROOT / s).exists(): scripts.append(s)
scripts = sorted(set(scripts))
basenames = {s: pathlib.Path(s).stem for s in scripts}

import_re = re.compile(r"from extra\.([a-z0-9_]+) import|import extra\.([a-z0-9_]+)\b")
imports, text = {}, {}
for s in scripts:
    t = (ROOT / s).read_text(errors="ignore"); text[s] = t
    imports[s] = {(m.group(1) or m.group(2)) for m in import_re.finditer(t)}
stem_to_path = {pathlib.Path(s).stem: s for s in scripts}

LEDGERS = ["bench/qk-decode-eval/candidates.json", "bench/qk-decode-eval/binding_templates.json",
           "bench/qk-lifecycle-search/search_candidates.json", "bench/qk-lifecycle-search/refutations.json",
           "bench/qk-lifecycle-search/generated_candidates.json", "bench/qk-decode-eval/schema.json",
           "bench/qk-lifecycle-search/search_policy.json", "bench/qk-lifecycle-search/evaluator_contract.json"]
ledger_text = {l: (ROOT / l).read_text(errors="ignore") for l in LEDGERS if (ROOT / l).exists()}
roots = {"extra/qk_decode_eval.py", "extra/qk_lifecycle_search_loop.py", "extra/qk_candidate_template_gen.py",
         "extra/qk_policy_consistency_check.py", "extra/qk_decode_runtime_overhead.py", "extra/qk_flash_decode.py",
         "extra/qk_clock_pin.py", "extra/qk_harness_contract.py", "extra/qk_nll_eval.py"}
for tt in ledger_text.values():
    for m in re.findall(r"extra/qk_[a-z0-9_]+\.py", tt): roots.add(m)
roots = {r for r in roots if r in scripts}

live, stack = set(), list(roots)
while stack:
    s = stack.pop()
    if s in live: continue
    live.add(s)
    for mod in imports.get(s, ()):
        p = stem_to_path.get(mod)
        if p and p not in live: stack.append(p)

imported_by = {s: [] for s in scripts}
for s in scripts:
    for mod in imports[s]:
        p = stem_to_path.get(mod)
        if p: imported_by[p].append(s)
# ALSO add path-string reference edges (subprocess calls like _run([..., "extra/qk_X.py", ...])) so the
# import-safety fixpoint keeps any script a kept script invokes by path, not only `from extra.X import`.
for s in scripts:
    for other in scripts:
        if other == s: continue
        if other in text[s] and s not in imported_by[other]:
            imported_by[other].append(s)

CANON = {"docs/current-project-state-handoff-20260621.md", "docs/README.md", "bench/README.md",
         "structure/Development/performance-primitive-research-principles.md",
         "docs/project-north-star-llama-and-lifecycle-search-20260620.md",
         "docs/decode-prefill-headline-reconciliation-result-20260621.md"}
all_docs = sorted({p.relative_to(ROOT).as_posix() for p in ROOT.glob("docs/*.md")}
                  | {"bench/README.md", "structure/Development/performance-primitive-research-principles.md"})
doc_text = {d: (ROOT / d).read_text(errors="ignore") for d in all_docs if (ROOT / d).exists()}

def refs_in(tm, path, stem):
    return [d for d, t in tm.items() if path in t or (stem + ".py") in t]

prelim, meta = {}, {}
for s in scripts:
    stem = basenames[s]
    drefs = refs_in(doc_text, s, stem)
    canon_refs = [d for d in drefs if d in CANON]
    lrefs = [l for l in ledger_text if s in ledger_text[l]]
    in_live = s in live
    if in_live: status, reason = "live", "in import-closure of live evaluator/search roots"
    elif canon_refs or lrefs: status, reason = "provenance", f"cited by canonical/ledger ({len(canon_refs)} canon-doc, {len(lrefs)} ledger)"
    elif drefs: status, reason = "manual_review", f"cited only by dated result doc(s): {len(drefs)} (conclusion captured; not live/canonical)"
    else: status, reason = "delete", "no reference in ANY doc or ledger and not imported by a kept script (no_canonical_reference)"
    prelim[s] = status
    meta[s] = {"path": s, "type": "script", "stem": stem, "imports": sorted(imports[s]), "imported_by": sorted(imported_by[s]),
               "in_live_closure": in_live, "canonical_doc_refs": canon_refs, "dated_doc_refs": [d for d in drefs if d not in CANON],
               "ledger_refs": lrefs, "status": status, "reason": reason}

# EXTERNAL-IMPORTER PROTECTION: scan ALL repo .py outside the inventory (tests, model, non-qk extra) for imports
# or path-references of our scripts; such a script is a dependency of external/live code -> never delete.
import subprocess as _sp
ext_protect = set()
for s in scripts:
    stem = basenames[s]
    hits = _sp.run(["grep","-rlE", f"from extra\\.{stem} import|import extra\\.{stem}\\b|extra/{stem}\\.py",
                    "--include=*.py","--exclude-dir=.git","test/","extra/","examples/",".",], capture_output=True, text=True).stdout.split()
    hits = [h.lstrip("./") for h in hits if h.lstrip("./") not in scripts and "qk-active-surface-reduction" not in h]
    if hits:
        ext_protect.add(s); meta[s]["external_importers"] = hits[:5]

# IMPORT-SAFETY FIXPOINT
for s in list(ext_protect):
    if prelim[s] == "delete":
        prelim[s] = "provenance"; meta[s]["status"] = "provenance"
        meta[s]["reason"] = f"imported/invoked by external code (tests/model): {meta[s].get('external_importers')} -- keep"
changed = True
while changed:
    changed = False
    for s in scripts:
        if prelim[s] != "delete": continue
        kept = [i for i in imported_by[s] if prelim[i] != "delete"]
        if kept:
            prelim[s] = "provenance"; meta[s]["status"] = "provenance"
            meta[s]["reason"] = f"shared utility imported by {len(kept)} kept script(s) {kept[:3]} -- keep to avoid broken imports"
            changed = True
inv = [meta[s] for s in scripts]

def cat(n):
    if "scope" in n or "roadmap" in n or "plan" in n: return "stale_scope_helper"
    if "prefill" in n: return "superseded_prefill_probe"
    if "mmvq" in n or "q8" in n or "dp4a" in n: return "superseded_mmvq_probe"
    if "tensile" in n or "gemm" in n or "bb5a" in n or "wmma" in n: return "superseded_tensile_probe"
    if "flywheel" in n or "loop" in n or "ansor" in n: return "generated_scratch"
    if "decode" in n or "flash" in n or "att" in n or "gqa" in n: return "superseded_decode_probe"
    return "no_canonical_reference"
for x in inv:
    x["category"] = cat(x["path"]) if x["status"] in ("delete", "manual_review") else ""

counts = Counter(x["status"] for x in inv)
summary = {"total_scripts": len(scripts), "counts": dict(counts), "live_roots": sorted(roots), "canonical_docs": sorted(CANON),
           "delete_categories": dict(Counter(x["category"] for x in inv if x["status"]=="delete"))}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "inventory.json").write_text(json.dumps({"summary": summary, "files": inv}, indent=2))

# markdown
lines = ["# Perf-Probe Active-Surface Inventory (Phase 0)", "",
         f"Total perf scripts: **{len(scripts)}** | " + " | ".join(f"{k}: **{v}**" for k,v in sorted(counts.items())), "",
         "Reference graph: import-closure (live) + canonical-doc refs + ledger refs, with an import-safety fixpoint",
         "(a script is DELETE only if every importer is also DELETE). Generated by `build_inventory.py`.", ""]
for st in ("live","provenance","manual_review","delete"):
    rows=[x for x in inv if x["status"]==st]
    lines.append(f"## {st} ({len(rows)})")
    if st=="delete":
        for x in rows: lines.append(f"- `{x['path']}` — {x['category']}")
    elif st=="manual_review":
        bycat=Counter(x["category"] for x in rows)
        lines.append("By category: " + ", ".join(f"{k} {v}" for k,v in sorted(bycat.items())))
    else:
        for x in rows: lines.append(f"- `{x['path']}`")
    lines.append("")
(OUT / "inventory.md").write_text("\n".join(lines))
print(json.dumps(summary, indent=2))
print("\nDELETE set (final, import-safe):", counts["delete"], "scripts")
