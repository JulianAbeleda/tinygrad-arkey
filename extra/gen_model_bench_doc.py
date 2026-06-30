#!/usr/bin/env python3
"""Render a model-bench markdown table from the per-model JSON artifacts produced by extra/model_e2e_bench.py.

The markdown doc is a *view*; the JSON artifacts under data/<backend>/ are the source of truth. Re-run this after
adding/updating any artifact so the table stays in sync.

Usage:
  python extra/gen_model_bench_doc.py \
      --data bench/models/qwen/data/amd-gfx1100 \
      --out  bench/models/qwen/amd-rx7900xtx-gfx1100.md \
      --gpu  "AMD Radeon RX 7900 XTX (gfx1100, 24GB)" \
      --family Qwen3
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
  rows = [json.loads(p.read_text()) for p in sorted(data_dir.glob("*.json"))]
  rows.sort(key=lambda r: r.get("params") or 0)

  commit = next((r["provenance"]["git_commit"] for r in rows if r.get("provenance", {}).get("git_commit")), None)
  any_dirty = any(r.get("provenance", {}).get("git_dirty") for r in rows)

  L = []
  L.append(f"# {args.family} benchmarks — {args.gpu}")
  L.append("")
  L.append(f"Backend: **AMD** · GPU: **{args.gpu}** · family: **{args.family}**")
  L.append("")
  L.append("Decode tok/s is the headline (decode is HBM-bandwidth bound). Numbers come from clean whole-decode "
           "`model.generate` (W==D), `PROFILE=0`, auto clock, warmed JIT, with a median over a steady-state window "
           "and the observed spread. **Quant matters** — it sets the bytes-per-weight moved each decode step, which "
           "is the dominant decode cost; compare sizes with quant in mind, not just parameter count.")
  L.append("")
  L.append("| Model | Quant | Params | Ctx | Decode tok/s (median) | Decode band [min–max] | Spread | Decode GB/s | Prefill pp512 tok/s | VRAM | Load s |")
  L.append("|---|---|---|---|---|---|---|---|---|---|---|")
  for r in rows:
    d = r.get("decode", {}); ts = d.get("tok_s", {}); pf = r.get("prefill") or {}
    band = f"{ts.get('min')}–{ts.get('max')}" if ts.get("min") is not None else "—"
    spread = f"{ts.get('spread_pct')}%" if ts.get("spread_pct") is not None else "—"
    pp = pf.get("prefill_tok_s")
    L.append(f"| {r.get('id')} | {r.get('quant') or '?'} | {_params(r.get('params'))} | {r.get('max_context')} "
             f"| {ts.get('median') or '—'} | {band} | {spread} | {d.get('gb_s') or '—'} "
             f"| {pp if pp is not None else '—'} | {r.get('vram_used_gb')} GB | {r.get('load_s')} |")
  L.append("")
  L.append("## Notes")
  L.append("")
  L.append("- **Decode tok/s** is the steady-state median (clock-ramp/first tokens dropped). High **spread** on the "
           "smallest models is expected: they are launch/dispatch-bound (tiny per-token GPU work), so wall-clock "
           "decode is noisy — the band shows it honestly rather than hiding it behind a single number.")
  L.append("- **Decode GB/s** is the HBM-bandwidth proxy (bytes moved per token ÷ median token time). For a fixed "
           "quant it should rise with model size until it saturates the GPU's memory bandwidth.")
  L.append("- **Prefill pp512 tok/s** is time-to-first-token for a 512-token prompt on the default prefill path "
           "(prefill is compute-bound, not memory-bound — a different regime from decode).")
  L.append("- **VRAM** is `GlobalCounters.mem_used` after load+warmup at the listed context. Larger contexts grow "
           "the KV cache and raise this.")
  L.append("")
  L.append(f"Provenance: tinygrad commit `{commit or '?'}`"
           f"{' (dirty tree)' if any_dirty else ''}. Regenerate with "
           f"`python extra/gen_model_bench_doc.py` from the JSON artifacts in `{args.data}/`.")
  L.append("")
  pathlib.Path(args.out).write_text("\n".join(L))
  print(f"wrote {args.out} ({len(rows)} model rows)")

if __name__ == "__main__":
  main()
