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
  L.append("> ⚠️ **DIAGNOSTIC, NOT PARITY AUTHORITY.** These numbers come from a first-pass end-to-end harness with "
           "two known methodology gaps, pending a synced-authority rerun: (1) **decode** is a median over a "
           "*growing-context* `model.generate` window (mixes contexts + host jitter; the repo authority is synced "
           "`TinyJit` min-of-K bursts at fixed context), and (2) **prefill** is measured on the *default universal* "
           "path (`PREFILL_V2=false`) via `generate` TTFT, **not** the tuned/server prefill profile — so the prefill "
           "column is apples-to-oranges vs llama.cpp `pp512` and understates the shipped tuned path (8B authority "
           "is ~3500 tok/s @512, not the value shown here). Do not cite these for tinygrad-vs-llama parity yet.")
  L.append("")
  L.append("Decode tok/s is the headline (decode is HBM-bandwidth bound). Numbers come from clean whole-decode "
           "`model.generate` (W==D), `PROFILE=0`, auto clock, warmed JIT, with a median over a steady-state window "
           "and the observed spread. **Quant matters** — it sets the bytes-per-weight moved each decode step, which "
           "is the dominant decode cost; compare sizes with quant in mind, not just parameter count.")
  L.append("")
  L.append("| Model | Quant | Params | Ctx | Decode tok/s (median) | Decode band [min–max] | Spread | Decode GB/s | Prefill pp512 (default path, diag) | VRAM | Load s |")
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
  # llama.cpp comparison (same GGUF, same GPU) -- only if any artifact has llama numbers
  if any(r.get("llama_cpp") for r in rows):
    L.append("## vs llama.cpp (same GGUF, same GPU)")
    L.append("")
    L.append("Reference: `llama-bench` (ROCm/HIP build) on the identical GGUF file and GPU. `tg128` = decode, "
             "`pp512` = prefill. **Decode ratio** is tinygrad median ÷ llama.cpp — the headline parity number.")
    L.append("")
    L.append("| Model | Quant | tinygrad decode | llama.cpp decode | decode ratio | tinygrad pp512 (default,diag) | llama.cpp pp512 | prefill ratio |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
      lc = r.get("llama_cpp")
      if not lc:
        L.append(f"| {r.get('id')} | {r.get('quant') or '?'} | {r.get('decode',{}).get('tok_s',{}).get('median') or '—'} "
                 f"| — | — | {(r.get('prefill') or {}).get('prefill_tok_s') or '—'} | — | — |")
        continue
      tg_t = r.get("decode", {}).get("tok_s", {}).get("median")
      lc_t = lc.get("decode_tg_tok_s")
      ratio = r.get("decode_ratio_tinygrad_over_llama")
      tg_pp = (r.get("prefill") or {}).get("prefill_tok_s")
      lc_pp = lc.get("prefill_pp_tok_s")
      pp_ratio = round(tg_pp / lc_pp, 3) if (tg_pp and lc_pp) else None
      L.append(f"| {r.get('id')} | {r.get('quant') or '?'} | {tg_t or '—'} | {lc_t} ±{lc.get('decode_tg_stddev')} "
               f"| **{round(ratio*100)}%** | {tg_pp or '—'} | {lc_pp} | {round(pp_ratio*100)}% |"
               if ratio is not None else
               f"| {r.get('id')} | {r.get('quant') or '?'} | {tg_t or '—'} | {lc_t} | — | {tg_pp or '—'} | {lc_pp} | — |")
    L.append("")
    lc_build = next((r["llama_cpp"].get("build_commit") for r in rows if r.get("llama_cpp")), None)
    L.append(f"llama.cpp build `{lc_build or '?'}`, `llama-bench` defaults (warmup + repeats). Decode is the fair "
             "comparison; tinygrad's default prefill path is the universal (long-prompt-slow) one unless "
             "`PREFILL_V2`/server profile is enabled, so the prefill ratio understates a tuned-prefill config.")
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
