#!/usr/bin/env python3
"""Roadmap #5 — mechanized docs supersession index (consolidation, NO deletion).
Classifies every fork doc as canonical / current / provenance by a reference graph (cited-by-canonical) + date +
topic cluster, so future agents navigate 650+ docs from one map. Emits docs_index.json + docs/provenance-index-*.md.
Does NOT move/delete any doc (that would break the ~251 canonical->dated-doc pointers)."""
import json, pathlib, re, subprocess
from collections import Counter, defaultdict
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-active-surface-reduction"

# upstream tinygrad docs (leave alone) -- skip the tinygrad library doc tree, only classify the fork's flat docs/*.md
UPSTREAM = {"index.md", "mnist.md", "nn.md", "dtypes.md", "quickstart.md", "showcase.md", "abstractions2.md",
            "abstractions3.md", "env_vars.md", "function.md"}
docs = sorted(p.name for p in ROOT.glob("docs/*.md") if p.name not in UPSTREAM)

CANON = {"current-project-state-handoff-20260621.md", "README.md",
         "project-north-star-llama-and-lifecycle-search-20260620.md",
         "decode-prefill-headline-reconciliation-result-20260621.md"}
canon_text = "\n".join((ROOT / "docs" / c).read_text(errors="ignore") for c in CANON if (ROOT / "docs" / c).exists())
canon_text += (ROOT / "bench/README.md").read_text(errors="ignore")
canon_text += (ROOT / "structure/Development/performance-primitive-research-principles.md").read_text(errors="ignore")

CURRENT_POINTERS = {
    "decode": "handoff §4 + docs/post-matmul-pv-decode-strategic-scope-20260621.md (REST_DECODE) + docs/fused-flash-concrete-gate-result-20260621.md",
    "prefill": "handoff §2-3 + docs/prefill-policy-integration-result-20260620.md",
    "q8": "handoff §2 (q8 opt-in, default-off) + docs/q8-mmvq-lifecycle-deep-result-20260619.md",
    "mmvq": "docs/decode-gap-is-attention-not-weight-gemv... (closed) + handoff §3",
    "tensile": "docs/amd-prefill-lds-gemm... + handoff (prefill kernels closed)",
    "flash": "docs/fused-flash-concrete-gate-result-20260621.md (FAIL_LOCAL_AB -> REST)",
    "attention": "docs/llama-flash-attn-tile-oracle-result-20260621.md (oracle) + decode pointers",
    "harness": "docs/harness-contract-audit-20260621.md",
    "lifecycle": "docs/lifecycle-search-loop-v0-result-20260621.md + bench/qk-lifecycle-search/",
    "active-surface": "docs/perf-probe-active-surface-reduction-result-20260621.md",
    "wmma": "handoff (WMMA decode not pursued; prefill Tensile path)",
    "gemm": "docs/amd-prefill-lds-gemm-not-refuted... (handoff)",
    "spec": "handoff (speculative decode rested)", "flywheel": "bench/amd-decode-flywheel-proof-20260614/",
    "quant": "handoff §2", "headline": "docs/decode-prefill-headline-reconciliation-result-20260621.md",
    "north-star": "docs/project-north-star-llama-and-lifecycle-search-20260620.md", "other": "docs/README.md map",
}

def topic(n):
    for key in ("prefill", "mmvq", "q8", "tensile", "decode", "flash", "attention", "harness", "lifecycle",
                "spec", "flywheel", "wmma", "gemm", "quant", "prefill", "headline", "north-star", "active-surface"):
        if key in n: return key
    return "other"

def date_of(n):
    m = re.search(r"20(\d{6})|20(\d{2}-\d{2}-\d{2})", n)
    return (m.group(0) if m else "undated")

rows = []
for d in docs:
    cited = (f"docs/{d}" in canon_text) or (d in canon_text)
    dt = date_of(d)
    daynum = re.sub(r"\D", "", dt)[:8] if dt != "undated" else ""
    if d in CANON: status = "canonical"
    elif cited or daynum == "20260621": status = "current"
    elif dt == "undated": status = "current"          # evergreen (no date) -> treat as current unless proven stale
    else: status = "provenance"                        # dated 0616-0620, not cited by canonical
    rows.append({"doc": f"docs/{d}", "status": status, "topic": topic(d), "date": dt, "cited_by_canonical": cited})

counts = Counter(r["status"] for r in rows)
by_topic_prov = defaultdict(list)
for r in rows:
    if r["status"] == "provenance": by_topic_prov[r["topic"]].append(r["doc"])

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "docs_index.json").write_text(json.dumps({"counts": dict(counts), "total": len(rows),
    "canonical": sorted(f"docs/{c}" for c in CANON), "rows": rows}, indent=2))

# concise human index
L = ["# Docs Supersession Index (2026-06-21)", "",
     "Mechanized map of the fork's `docs/*.md` (roadmap #5 consolidation). **No docs deleted/moved** — this only",
     "classifies, so the ~251 canonical->dated-doc pointers stay intact. Regenerate: "
     "`PYTHONPATH=. .venv/bin/python bench/qk-active-surface-reduction/build_docs_index.py`. Backing data: "
     "`bench/qk-active-surface-reduction/docs_index.json`.", "",
     f"**Totals:** {len(rows)} fork docs — " + " · ".join(f"{k} **{v}**" for k, v in sorted(counts.items())), "",
     "## Authority order (read these; ignore the rest unless tracing provenance)",
     "1. `docs/current-project-state-handoff-20260621.md` — canonical current state",
     "2. `docs/README.md` — curated navigation map", "3. `bench/README.md` — bench/evaluator map",
     "4. `structure/Development/performance-primitive-research-principles.md` — method authority",
     "5. `docs/project-north-star-llama-and-lifecycle-search-20260620.md` — completion definition", "",
     "## Provenance (historical; superseded by the canonical syntheses) — by topic",
     "These dated `*-result/-scope/-probe/-audit.md` are the chronological probe log; their verdicts are folded into",
     "the canonical docs above. Kept for history, **not authority**.", ""]
L.append("| topic | provenance docs | CURRENT authority (read this instead) |")
L.append("|---|---:|---|")
for t in sorted(by_topic_prov, key=lambda k: -len(by_topic_prov[k])):
    L.append(f"| {t} | {len(by_topic_prov[t])} | {CURRENT_POINTERS.get(t, 'docs/README.md map')} |")
(ROOT / "docs/provenance-index-20260621.md").write_text("\n".join(L) + "\n")
print(json.dumps({"counts": dict(counts), "total": len(rows)}, indent=2))
print("provenance by topic:", {k: len(v) for k, v in sorted(by_topic_prov.items(), key=lambda x: -len(x[1]))})
