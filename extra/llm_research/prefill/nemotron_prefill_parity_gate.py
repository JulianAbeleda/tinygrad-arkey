#!/usr/bin/env python3
"""Real-model gate: padded-piece NemotronHPrefill vs model.prefix on the Nemotron-H 4B BF16 GGUF (off the production path).

Per prompt length n: (1) pad-token invariance, bit for bit (hidden + every cache); (2) rows past the prompt stay zero;
(3) relative max error of the last hidden row, attention K/V and Mamba conv/state against model.prefix, each held to a
declared bound.  The two paths differ only in reduction order (piece-batched hi/lo routes, SSD vs the prefix scan), so
the bounds are error-budget bounds, not noise floors:

  * hidden 2e-3: measured 1.1e-4..8.4e-4 over 8 lengths 255..3000 on 2026-09-26 with both the chunked 512-row and the
    2048-row route tables (the error grows with prompt length: SSD chunk scan vs the prefix scan); 2e-3 is 2.4x the
    largest observed and 2**-9, far below a bf16 ulp (2**-8) at the logits' input.
  * kv 1.5e-2: K/V are bf16 caches (ulp 2**-8 = 3.9e-3 relative); measured max 7.5e-3 (two ulps); bound 2x = 4 ulps.
  * mamba 1e-2: fp32 conv/state; measured max 3.0e-3; bound 3.3x.
  A length above 3000 is outside the measured range: re-derive before widening --lengths.

  DEV=NV python3 extra/llm_research/prefill/nemotron_prefill_parity_gate.py [--lengths 255,256,...] [--out gate.json]
Exits 1 on any breach.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
GGUF = "/home/ubuntu/storage/models/NVIDIA-Nemotron3-Nano-4B-BF16.gguf"
BOUNDS = {"hidden": 2e-3, "kv": 1.5e-2, "mamba": 1e-2}
LENGTHS = (255, 256, 600, 1000, 1024, 1500, 2049, 3000)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--lengths", default=",".join(map(str, LENGTHS)))
  parser.add_argument("--out")
  args = parser.parse_args()
  from tinygrad.llm.nemotron_h import load
  from tinygrad.llm.nemotron_h_prefill import NemotronHPrefill
  lengths = [int(x) for x in args.lengths.split(",")]
  model, _ = load(GGUF, max_context=max(lengths) + 1024)
  pf = NemotronHPrefill(model, capacity=max(lengths) + 64)
  rng = np.random.default_rng(0)
  rel = lambda a, b: float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-6))
  rows, passed = [], True
  for n in lengths:
    prompt = [int(t) for t in rng.integers(100, 30000, n)]
    h = pf(prompt).numpy()
    kept = [{k: v.numpy() for k, v in b.items()} if b else None for b in pf.buffers]
    pf.pad_token = 777
    h2 = pf(prompt).numpy(); pf.pad_token = 0
    invariant = bool((h.view(np.uint32) == h2.view(np.uint32)).all() and all(
      (b[k].numpy().view(np.uint32) == kb[k].view(np.uint32)).all() for b, kb in zip(pf.buffers, kept) if b for k in b))
    ref, caches = model.prefix(prompt, through=len(model.blk) - 1)
    attn = [i for i, b in enumerate(model.blk) if b.block_type == "attention"]
    mamba = [i for i, b in enumerate(model.blk) if b.block_type == "mamba"]
    err = {"hidden": rel(h, ref.numpy()[:, -1:]),
           "kv": max(rel(kept[i][k][:, :, :n], caches[i][k].numpy()) for i in attn for k in ("k", "v")),
           "mamba": max(rel(kept[i][k], caches[i][k].numpy()) for i in mamba for k in ("conv", "state"))}
    tail = max(float(np.abs(kept[i][k][:, :, n:]).max()) if kept[i][k].shape[2] > n else 0.0 for i in attn for k in ("k", "v"))
    ok = invariant and tail == 0.0 and all(err[k] <= BOUNDS[k] for k in BOUNDS)
    passed &= ok
    rows.append(row := {"n": n, "spans": pf.spans(n), "pad_invariant": invariant, "rows_past_prompt_max": tail, **err, "passed": ok})
    print(json.dumps(row), flush=True)
  result = {"schema": "nemotron-prefill-parity-gate.v1", "bounds": BOUNDS, "passed": passed, "rows": rows}
  if args.out: Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps({"passed": passed, "rows": len(rows)}))
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
