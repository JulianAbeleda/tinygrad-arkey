#!/usr/bin/env python3
"""Render the per-model bench markdown from the JSON artifacts (the artifacts are the source of truth).

Two artifact kinds per model under data/<backend>/:
  <id>.json            -- model_e2e_bench: end-to-end generate (DIAGNOSTIC: growing-context decode median + default
                          universal-path prefill TTFT). Kept as a secondary, clearly-labeled diagnostic.
  <id>.authority.json  -- model_authority_bench: clean W==D fixed-context decode (qk_decode_runtime_overhead) vs
                          llama.cpp tg128 at MATCHED depth, plus an optional prefill_authority block. HEADLINE.

Re-run after adding/updating any artifact.

Usage:
  python extra/tools/gen_model_bench_doc.py --data bench/models/qwen/data/amd-gfx1100 \
      --out bench/models/qwen/amd-rx7900xtx-gfx1100.md --gpu "AMD Radeon RX 7900 XTX (gfx1100, 24GB)" --family Qwen3
"""
from __future__ import annotations
import json, argparse, pathlib

def _params(n): return f"{n/1e9:.2f}B" if n and n >= 1e9 else (f"{n/1e6:.0f}M" if n else "?")

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--data", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--gpu", required=True)
  ap.add_argument("--family", default="Qwen")
  args = ap.parse_args()

  data_dir = pathlib.Path(args.data)
  e2e = {p.stem: json.loads(p.read_text()) for p in sorted(data_dir.glob("*.json")) if not p.name.endswith(".authority.json")}
  auth = {p.name[:-len(".authority.json")]: json.loads(p.read_text()) for p in sorted(data_dir.glob("*.authority.json"))}
  def params(i): return e2e.get(i, {}).get("params") or auth.get(i, {}).get("params")
  def quant(i): return e2e.get(i, {}).get("quant") or auth.get(i, {}).get("quant") or "?"
  def family(i): return auth.get(i, {}).get("family") or args.family
  ids = sorted(set(e2e) | set(auth), key=lambda i: (params(i) or 0))
  cross_family = sorted({i for i in ids if family(i) != args.family})

  L = []
  L.append(f"# {args.family} benchmarks — {args.gpu}")
  L.append("")
  L.append(f"Backend: **AMD** · GPU: **{args.gpu}** · family: **{args.family}**")
  L.append("")
  L.append("**Quant matters** — decode re-reads the weights every token, so bytes-per-weight (the quant) is the "
           "dominant decode cost. Read tok/s next to its quant, not parameter count alone.")
  L.append("")

  # ---------- HEADLINE: authority decode vs llama.cpp at matched context ----------
  if auth:
    L.append("## Decode vs llama.cpp — authority (matched context)")
    L.append("")
    L.append("tinygrad: clean **W==D** decode (`qk_decode_runtime_overhead.py` — `TinyJit`, device-synced, NMEAS=40, "
             "**fixed** context, shipped `FLASH_DECODE_THRESHOLD=512` so the owned flash-attention route fires at "
             "ctx≥512). llama.cpp: `llama-bench tg128` at the **matched depth** (`-d ctx`). Comparing at the same "
             "context is essential — tinygrad switches to the owned flash route at ctx≥512, and llama is ~flat across "
             "context, so a single number hides the crossover.")
    L.append("")
    L.append("| Model | Quant | ctx | route | tinygrad W==D tok/s | llama tg@depth tok/s | ratio | host-sync |")
    L.append("|---|---|---|---|---|---|---|---|")
    for i in ids:
      comp = auth.get(i, {}).get("decode_matched_comparison")
      if not comp:
        L.append(f"| {i} | {quant(i)} | — | — | _pending authority rerun_ | — | — | — |")
        continue
      for c in comp:
        route = "flash" if c.get("flash_route") else "non-flash"
        ratio = f"**{c['ratio_pct']}%**" if c.get("ratio_pct") else "—"
        L.append(f"| {i} | {quant(i)} | {c['ctx']} | {route} | {c['tinygrad_tok_s_W']} | {c.get('llama_tok_s') or '—'} "
                 f"| {ratio} | {c.get('host_sync_pct')}% |")
    L.append("")
    L.append("Low **host-sync %** means the measurement is GPU-bound (not host-loop noise). At ctx≥512 the owned "
             "flash route fires; below it the non-flash path runs and is the weaker regime.")
    L.append("")
    L.append("**Reading the ratios:** 8B (the size the decode kernels were tuned for) is at/above llama in the flash "
             "regime (~105% @ctx512) and ~82% on the sub-512 non-flash path. 14B/32B sit near ~40% — the larger "
             "shapes were never decode-optimized (a known, separate gap), not a measurement error. 0.6B is "
             "launch/dispatch-bound (tiny per-token GPU work), where tinygrad's per-kernel overhead costs the most.")
    if cross_family:
      L.append("")
      L.append(f"> ⚠️ **Different architecture:** {', '.join(cross_family)} is **Qwen3.5** (hybrid SSM/attention "
               "layers), not Qwen3. tinygrad has no tuned path for that architecture, so its ~7% is an unsupported-"
               "performance result, not a like-for-like Qwen3 comparison. Listed for completeness only.")
    L.append("")

  # ---------- Prefill: tuned authority where measured, else llama pp512 + note ----------
  if auth:
    L.append("## Prefill (pp512)")
    L.append("")
    L.append("Prefill is compute-bound (a different regime from decode). tinygrad's tuned path is `PREFILL_V2` "
             "graph-gemm (needs ~+14GB VRAM, so it only fits the smaller models); where it wasn't measured the "
             "cell says so rather than showing the slow universal-path number.")
    L.append("")
    L.append("| Model | Quant | tinygrad pp512 (tuned authority) | llama.cpp pp512 | ratio | route |")
    L.append("|---|---|---|---|---|---|")
    for i in ids:
      a = auth.get(i, {})
      pa = a.get("prefill_authority")
      lc_pp = (a.get("llama_cpp", {}).get("prefill_pp512") or {}).get("tok_s")
      if pa and pa.get("pp512_tok_s"):
        tg = pa["pp512_tok_s"]; ratio = f"**{round(tg/lc_pp*100)}%**" if lc_pp else "—"
        L.append(f"| {i} | {quant(i)} | {tg} | {lc_pp or '—'} | {ratio} | {pa.get('route','graph-gemm')} |")
      else:
        L.append(f"| {i} | {quant(i)} | _not measured (VRAM / pending_) | {lc_pp or '—'} | — | — |")
    L.append("")

  # ---------- Secondary: end-to-end generate diagnostic ----------
  if e2e:
    L.append("## End-to-end `generate` (diagnostic, not parity)")
    L.append("")
    L.append("> These are first-pass end-to-end numbers: decode is a **median over a growing-context** "
             "`model.generate` window (context-mixed + host jitter), and prefill is the **default universal path** "
             "(`PREFILL_V2=false`) via `generate` TTFT. Useful as a rough end-to-end feel; **not** a parity number — "
             "use the authority tables above for tinygrad-vs-llama.")
    L.append("")
    L.append("| Model | Quant | Params | Ctx | Decode tok/s (median) | Spread | Decode GB/s | Prefill TTFT (default path) | VRAM | Load s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for i in ids:
      r = e2e.get(i)
      if not r:
        continue
      d = r.get("decode", {}); ts = d.get("tok_s", {}); pf = r.get("prefill") or {}
      spread = f"{ts.get('spread_pct')}%" if ts.get("spread_pct") is not None else "—"
      L.append(f"| {i} | {r.get('quant') or '?'} | {_params(r.get('params'))} | {r.get('max_context')} "
               f"| {ts.get('median') or '—'} | {spread} | {d.get('gb_s') or '—'} "
               f"| {pf.get('prefill_tok_s') or '—'} | {r.get('vram_used_gb')} GB | {r.get('load_s')} |")
    L.append("")

  # provenance
  commit = next((r.get("provenance", {}).get("git_commit") for r in e2e.values() if r.get("provenance", {}).get("git_commit")), None)
  L.append(f"Provenance: tinygrad commit `{commit or '?'}`. Regenerate with `python extra/tools/gen_model_bench_doc.py` "
           f"from the JSON artifacts in `{args.data}/` (artifacts are local per the bench policy; this table is the "
           f"committed durable record).")
  L.append("")
  pathlib.Path(args.out).write_text("\n".join(L))
  print(f"wrote {args.out} ({len(ids)} models; {len(auth)} with authority data)")

if __name__ == "__main__":
  main()
