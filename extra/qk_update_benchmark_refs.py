#!/usr/bin/env python3
"""Rewrite current benchmark references from bench/canonical-benchmarks.json.

This keeps the structure session handoff synchronized without editing dated
handoff/provenance docs. The manifest is the hand-edited source of truth.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "bench/canonical-benchmarks.json"


def _load(path: pathlib.Path) -> dict[str, Any]:
  return json.loads(path.read_text())


def _vals(d: dict[str, Any], ctxs: list[int], decimals: int | None = None) -> str:
  xs = []
  for ctx in ctxs:
    v = d[str(ctx)]
    if decimals is None:
      xs.append(str(int(v)) if float(v).is_integer() else str(v))
    else:
      xs.append(f"{float(v):.{decimals}f}")
  return " / ".join(xs)


def _pct_vs(a: dict[str, Any], b: dict[str, Any], ctxs: list[int]) -> tuple[float, float]:
  ps = [(float(a[str(c)]) / float(b[str(c)])) * 100.0 for c in ctxs]
  return min(ps), max(ps)


def _fmt_pct_range(lo: float, hi: float) -> str:
  return f"{lo:.1f}-{hi:.1f}%"


def _replace(path: pathlib.Path, old: str, new: str) -> bool:
  txt = path.read_text()
  if txt == new:
    return False
  path.write_text(new)
  return True


def _sub_once(txt: str, pattern: str, repl: str, path: pathlib.Path) -> str:
  new, n = re.subn(pattern, repl, txt, count=1, flags=re.MULTILINE)
  if n != 1:
    raise RuntimeError(f"{path}: expected exactly one match for {pattern!r}, got {n}")
  return new


def _decode_lines(m: dict[str, Any]) -> dict[str, str]:
  d = m["decode"]
  ctx = d["contexts"]
  base = _vals(d["baseline"]["tok_s"], ctx, 1)
  probe = _vals(d["aggressive_probe"]["tok_s"], ctx, 1)
  target = _vals(d["aggressive_target"]["tok_s"], ctx, 1)
  llama = _vals(d["llama_reference"]["tok_s"], ctx, 2)
  bubblebeam = _vals(d.get("bubblebeam_futuresight", d["aggressive_probe"])["tok_s"], ctx, 1)
  lo, hi = _pct_vs(d["baseline"]["tok_s"], d["llama_reference"]["tok_s"], ctx)
  return {
    "base": base,
    "probe": probe,
    "target": target,
    "llama": llama,
    "bubblebeam": bubblebeam,
    "pct_llama": _fmt_pct_range(lo, hi),
    "ctx": "/".join(str(x) for x in ctx),
    "latest_run": d["baseline"]["latest_run"],
  }


def _prefill_lines(m: dict[str, Any]) -> dict[str, str]:
  p = m["prefill"]
  ctx = p["contexts"]
  base = _vals(p["baseline"]["tok_s"], ctx, None)
  target = _vals(p["aggressive_target"]["tok_s"], ctx, 2)
  return {
    "base": base,
    "target": target,
    "ctx": "/".join(str(x) for x in ctx),
  }


def update_docs_readme(m: dict[str, Any]) -> pathlib.Path:
  path = ROOT / "docs/README.md"
  txt = path.read_text()
  d, p = _decode_lines(m), _prefill_lines(m)
  repl = (
    f"- **`current-project-state-handoff-20260624.md`** — ⭐⭐ CANONICAL CURRENT STATE (read first). Current numbers\n"
    f"  (decode **{d['base']} tok/s** @ctx512/1024/2048/4096 ≈ **{d['pct_llama']} of llama** on the `Q4K_GEMV_WARP*`\n"
    f"  default stack; prefill **{p['base']}**), decided policies, and the parity win (owned attention tile +"
  )
  txt = _sub_once(
    txt,
    r"- \*\*`current-project-state-handoff-20260624\.md`\*\* — ⭐⭐ CANONICAL CURRENT STATE \(read first\)\. Current numbers\n  \(decode \*\*.*? tok/s\*\* @ctx512/1024/2048/4096 ≈ \*\*.*? of llama\*\* on the `Q4K_GEMV_WARP\*`\n  default stack; prefill \*\*.*?\*\*\), decided policies, and the parity win \(owned attention tile \+",
    repl,
    path,
  )
  _replace(path, "", txt)
  return path


def update_bench_readme(m: dict[str, Any]) -> pathlib.Path:
  path = ROOT / "bench/README.md"
  txt = path.read_text()
  d, p = _decode_lines(m), _prefill_lines(m)
  txt = _sub_once(
    txt,
    r"\| \*\*Decode 8B\*\* \(default\) \| \*\*.*? tok/s\*\* @ctx 512/1024/2048/4096 \(~.*? of llama\) \| `extra/qk_decode_runtime_overhead\.py` \|",
    f"| **Decode 8B** (default) | **{d['base']} tok/s** @ctx 512/1024/2048/4096 (~{d['pct_llama']} of llama) | `extra/qk_decode_runtime_overhead.py` |",
    path,
  )
  txt = _sub_once(
    txt,
    r"\| \*\*Prefill 8B\*\* \(default, `eightwave`\) \| \*\*.*? tok/s\*\* @ctx 512/1024/2048/4096/8192 \| `extra/qk_prefill_emit_search\.py` \|",
    f"| **Prefill 8B** (default, `eightwave`) | **{p['base']} tok/s** @ctx 512/1024/2048/4096/8192 | `extra/qk_prefill_emit_search.py` |",
    path,
  )
  _replace(path, "", txt)
  return path


def update_current_handoff(m: dict[str, Any]) -> pathlib.Path:
  path = ROOT / "docs/current-project-state-handoff-20260624.md"
  txt = path.read_text()
  d, p = _decode_lines(m), _prefill_lines(m)
  txt = _sub_once(
    txt,
    r"\| decode @ctx 512 / 1024 / 2048 / 4096 \| \*\*.*? tok/s\*\* \(≈ \*\*.*? of llama\*\* — at/above parity\) \| `.*?` \|",
    f"| decode @ctx 512 / 1024 / 2048 / 4096 | **{d['base']} tok/s** (≈ **{d['pct_llama']} of llama** — at/above parity) | `bench/canonical-benchmarks.json` |",
    path,
  )
  txt = _sub_once(
    txt,
    r"\| decode full-stack envelope \(non-search\) \| .*? tok/s \| `.*?` \|",
    f"| decode full-stack envelope (non-search) | {d['target']} tok/s | `bench/canonical-benchmarks.json` |",
    path,
  )
  txt = _sub_once(
    txt,
    r"\| llama reference \(same ctx\) \| .*? tok/s \| `.*?` \|",
    f"| llama reference (same ctx) | {d['llama']} tok/s | `bench/canonical-benchmarks.json` |",
    path,
  )
  txt = _sub_once(
    txt,
    r"\| prefill @ctx 512 / 1024 / 2048 / 4096 / 8192 \| \*\*.*? tok/s\*\* .*? \| `.*?` \|",
    f"| prefill @ctx 512 / 1024 / 2048 / 4096 / 8192 | **{p['base']} tok/s** (`eightwave` promoted; long-context stable) | `bench/canonical-benchmarks.json` |",
    path,
  )
  _replace(path, "", txt)
  return path


def _table_prefill(m: dict[str, Any]) -> str:
  p = m["prefill"]
  lines = [
    "| ctx | Baseline (current default) | Confirmed `eightwave` | Confirmed Δ | Aggressive non-search bound* |",
    "|---:|---:|---:|---:|---:|",
  ]
  for c in p["contexts"]:
    b = float(p["baseline"]["tok_s"][str(c)])
    conf = float(p["confirmed"]["tok_s"][str(c)])
    tgt = float(p["aggressive_target"]["tok_s"][str(c)])
    lines.append(f"| {c} | {b:.0f} | {conf:.0f} | {(conf - b) / b * 100.0:+.2f}% | {tgt:.2f} |")
  return "\n".join(lines)


def _table_decode(m: dict[str, Any]) -> str:
  d = m["decode"]
  lines = [
    "| ctx | Baseline (current default) | Confirmed (probe; not promoted) | Confirmed Δ | Aggressive non-search target |",
    "|---:|---:|---:|---:|---:|",
  ]
  for c in d["contexts"]:
    b = float(d["baseline"]["tok_s"][str(c)])
    probe = float(d["aggressive_probe"]["tok_s"][str(c)])
    tgt = float(d["aggressive_target"]["tok_s"][str(c)])
    lines.append(f"| {c} | {b:.1f} | {probe:.1f} | {(probe - b) / b * 100.0:+.2f}% | {tgt:.1f} |")
  return "\n".join(lines)


def update_baseline_handoff(m: dict[str, Any]) -> pathlib.Path:
  path = ROOT / "docs/prefill-baseline-confirmed-aggressive-bound-handoff-20260624.md"
  txt = path.read_text()
  d, p = _decode_lines(m), _prefill_lines(m)
  txt = re.sub(
    r"- \[latest baseline\]\(/home/ubuntu/tinygrad-arkey/bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-[^)]*\)",
    f"- [latest baseline](/home/ubuntu/tinygrad-arkey/bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-{d['latest_run']}/periodic_diff.json)",
    txt,
    count=1,
  )
  txt = _sub_once(
    txt,
    r"- Decode: canonical W==D baseline is `.*?` tok/s @ctx512/1024/2048/4096\.",
    f"- Decode: canonical W==D baseline is `{d['base']}` tok/s @ctx512/1024/2048/4096.",
    path,
  )
  txt = _sub_once(
    txt,
    r"- Latest aggressive decode probe \(unpromoted\) is `.*?` tok/s, and remains below the non-search envelope \(`.*?`\)\.",
    f"- Latest aggressive decode probe (unpromoted) is `{d['probe']}` tok/s, and remains below the non-search envelope (`{d['target']}`).",
    path,
  )
  txt = _sub_once(
    txt,
    r"\| ctx \| Baseline \(current default\) \| Confirmed `eightwave` \| Confirmed Δ \| Aggressive non-search bound\* \|\n\|---:.*?\|\n(?:\|.*\|\n){5}",
    _table_prefill(m) + "\n",
    path,
  )
  txt = _sub_once(
    txt,
    r"\| ctx \| Baseline \(current default\) \| Confirmed \(probe; not promoted\) \| Confirmed Δ \| Aggressive non-search target \|\n\|---:.*?\|\n(?:\|.*\|\n){4}",
    _table_decode(m) + "\n",
    path,
  )
  txt = _sub_once(
    txt,
    r"\| ctx \| Prefill base \| Prefill `pipe_tm2_tn2` \| Prefill `pipe_tm4_tn2` \| Decode baseline \| Decode aggressive \|\n\|---:.*?\|\n(?:\|.*\|\n){4}",
    "\n".join([
      "| ctx | Prefill base | Prefill `pipe_tm2_tn2` | Prefill `pipe_tm4_tn2` | Decode baseline | Decode aggressive |",
      "|---:|---:|---:|---:|---:|---:|",
      "| 512 | 3572 | 4253 | 2332 | 101.6 | 103.4 |",
      "| 1024 | 3483 | 4037 | 2263 | 99.8 | 101.6 |",
      "| 2048 | 3226 | 3659 | 2139 | 97.3 | 99.1 |",
      "| 4096 | 2789 | 3110 | 1937 | 92.7 | 94.4 |",
      "",
    ]),
    path,
  )
  _replace(path, "", txt)
  return path


def update_next_workstreams(m: dict[str, Any]) -> pathlib.Path:
  path = ROOT / "docs/prefill-decode-next-workstreams-codex-scope-20260624.md"
  txt = path.read_text()
  d = _decode_lines(m)
  diff = m["decode"]["comparator"]["delta_pct"]
  txt = _sub_once(
    txt,
    r"- latest baseline: `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-[^`]+`",
    f"- latest baseline: `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-{d['latest_run']}`",
    path,
  )
  txt = _sub_once(
    txt,
    r"- current A/B vs old internal route: \+.*? across ctx 512\.\.4096",
    f"- current A/B vs old internal route: +{diff['4096']:.2f}% to +{diff['512']:.2f}% across ctx 512..4096",
    path,
  )
  _replace(path, "", txt)
  return path


def update_aggressive_probe_system(m: dict[str, Any]) -> pathlib.Path:
  path = ROOT / "docs/decode-aggressive-probe-promotion-system-20260624.md"
  txt = path.read_text()
  txt = _sub_once(
    txt,
    r"\| ctx \| current baseline \| aggressive probe \| aggressive target \| probe vs baseline \| probe vs target \|\n\|---:.*?\|\n(?:\|.*\|\n){4}",
    _table_aggressive_probe(m) + "\n",
    path,
  )
  _replace(path, "", txt)
  return path


def _table_aggressive_probe(m: dict[str, Any]) -> str:
  d = m["decode"]
  lines = [
    "| ctx | current baseline | aggressive probe | aggressive target | probe vs baseline | probe vs target |",
    "|---:|---:|---:|---:|---:|---:|",
  ]
  for c in d["contexts"]:
    b = float(d["baseline"]["tok_s"][str(c)])
    probe = float(d["aggressive_probe"]["tok_s"][str(c)])
    target = float(d["aggressive_target"]["tok_s"][str(c)])
    lines.append(f"| {c} | {b:.1f} | {probe:.1f} | {target:.1f} | {(probe - b) / b * 100.0:+.2f}% | {(probe - target) / target * 100.0:+.2f}% |")
  return "\n".join(lines)


def update_decode_aggressive_scope(m: dict[str, Any]) -> pathlib.Path:
  path = ROOT / "docs/decode-aggressive-target-proof-scope-20260624.md"
  txt = path.read_text()
  d = _decode_lines(m)
  txt = _sub_once(
    txt,
    r"- Baseline \(current default, measured\): `.*?` tok/s @ `512,1024,2048,4096`",
    f"- Baseline (current default, measured): `{d['base']}` tok/s @ `512,1024,2048,4096`",
    path,
  )
  txt = _sub_once(
    txt,
    r"- Confirmed target \(measured\): `.*?` tok/s",
    f"- Confirmed target (measured): `{d['probe']}` tok/s",
    path,
  )
  txt = _sub_once(
    txt,
    r"- Aggressive-theoretical target \(non-search stack envelope\): `.*?` tok/s",
    f"- Aggressive-theoretical target (non-search stack envelope): `{d['target']}` tok/s",
    path,
  )
  _replace(path, "", txt)
  return path


def update_structure_session_handoff(m: dict[str, Any]) -> pathlib.Path:
  path = ROOT / "structure/Development/session-handoff.md"
  txt = path.read_text()
  d, p = _decode_lines(m), _prefill_lines(m)
  block = "\n".join([
    "<!-- CANONICAL_BENCHMARKS:START -->",
    "## Current Benchmark Authority",
    "",
    "Source of truth:",
    "",
    "- `bench/canonical-benchmarks.json`",
    "- Update derived docs with `PYTHONPATH=. .venv/bin/python extra/qk_update_benchmark_refs.py`.",
    "- Check derived docs with `PYTHONPATH=. .venv/bin/python extra/qk_update_benchmark_refs.py --check`.",
    "",
    "Current baseline snapshot:",
    "",
    f"- Decode baseline @ctx512/1024/2048/4096: `{d['base']}` tok/s.",
    f"- Decode BubbleBeam FutureSight @ctx512/1024/2048/4096: `{d['bubblebeam']}` tok/s (`BUBBLEBEAM_FUTURESIGHT=1`, default-off selector).",
    f"- Decode aggressive probe, measured but not promoted: `{d['probe']}` tok/s.",
    f"- Decode aggressive target envelope: `{d['target']}` tok/s.",
    f"- Prefill baseline @ctx512/1024/2048/4096/8192: `{p['base']}` tok/s.",
    f"- Latest decode lifecycle run: `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-{d['latest_run']}`.",
    "- Latest BubbleBeam artifact: `bench/qk-scheduler-gemv-vs-owned/coalesced_dequant_mE_20260625-162422.json`.",
    "",
    "Do not hand-edit benchmark numbers in derived docs; change the manifest and rerun the updater.",
    "<!-- CANONICAL_BENCHMARKS:END -->",
    "",
  ])
  if "<!-- CANONICAL_BENCHMARKS:START -->" in txt:
    txt = re.sub(
      r"<!-- CANONICAL_BENCHMARKS:START -->.*?<!-- CANONICAL_BENCHMARKS:END -->\n?",
      block,
      txt,
      count=1,
      flags=re.DOTALL,
    )
  else:
    txt = txt.replace("# Session Handoff\n\n", "# Session Handoff\n\n" + block, 1)
  _replace(path, "", txt)
  return path


UPDATERS = [update_structure_session_handoff, update_docs_readme, update_bench_readme, update_current_handoff]


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--manifest", default=str(MANIFEST))
  ap.add_argument("--check", action="store_true", help="fail if any target would change")
  args = ap.parse_args()
  m = _load(pathlib.Path(args.manifest))
  targets = [ROOT / t for t in m["update_targets"]]
  missing = [str(p.relative_to(ROOT)) for p in targets if not p.exists()]
  if missing:
    raise RuntimeError(f"manifest update_targets missing: {missing}")
  before = {p: p.read_text() for p in targets}
  touched = [fn(m) for fn in UPDATERS]
  unexpected = sorted(str(p.relative_to(ROOT)) for p in touched if p not in before)
  if unexpected:
    raise RuntimeError(f"updater touched paths not listed in manifest update_targets: {unexpected}")
  changed = [str(p.relative_to(ROOT)) for p in touched if p.read_text() != before[p]]
  if args.check and changed:
    for p, txt in before.items():
      p.write_text(txt)
    print("benchmark refs out of date:")
    for p in changed:
      print(f"  {p}")
    return 1
  print("checked benchmark refs:" if args.check else "updated benchmark refs:")
  for p in sorted(set(str(x.relative_to(ROOT)) for x in touched)):
    print(f"  {p}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
