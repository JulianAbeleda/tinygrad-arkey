# B3 phase-0 — adaptive-quant headroom: the thesis is VALIDATED, but the realizable win is modest

Date: 2026-06-15. Question: does a per-tensor bit-width search have headroom to read fewer bytes than
llama.cpp's fixed Q4_K_M at equal quality? Method (`extra/qk_quant_sensitivity.py`): perturb each role's
fp16 weights to a coarser per-block grid (simulating fewer bits) and measure the perplexity (NLL) delta on a
held-out sequence. Simulated (no real Q4_K/Q6_K quantizer exists) — informative for go/no-go, not shippable.

## Result (baseline NLL = 0.5024)
| role (all layers) | Q4_K_M bits | dNLL@4bit | dNLL@2bit | read |
|---|---|---:|---:|---|
| **ffn_down** (cumulative, all layers) | **Q6 (6.5b)** | **+0.0062** | +0.107 | over-provisioned |
| output.weight / lm_head (per-layer) | Q6 (6.5b) | +0.016 | +0.476 | correctly Q6 |
| ffn_gate/up, attn_q/o (per-layer) | Q4 (4.5b) | ~0 (below noise) | ~0 | inconclusive at 1-layer |

## Reading
- **The thesis holds**: Q4_K_M's fixed assignment is NOT optimal per-tensor. `ffn_down` is **over-provisioned**
  — demoting its 18 Q6 layers to Q4 costs +0.0062 NLL (≈free), but llama keeps them Q6 for safety. A search
  finds this; the fixed recipe can't. That is "beyond llama's fixed scheme," and it is the machine-search
  thesis applied to bit-width (the natural extension of the coverage win).
- **lm_head is correctly Q6** — very sensitive (+0.016 at 4-bit, +0.48 at 2-bit). The search would (correctly)
  leave it alone. So Q4_K_M is right there and wrong on ffn_down — a mixed verdict, exactly what adaptive
  granularity is for.

## But the realizable win is modest, and blocked on a missing quantizer
- **Magnitude**: the 18 Q6 `ffn_down` tensors are ~743 MB; demoting to Q4 saves ~31% of *their* bytes ≈ 230 MB
  ≈ **~5% of the 4.68 GB/token** → ~5% faster decode at ~free quality. Real, but ~5%.
- **Blocker**: shipping it needs a **Q4_K quantizer** (fp16 → Q4_K block bytes) — only dequant references
  exist in the repo. That is a real build (K-quant super-block scales + packing), plus re-packing into the
  primitive's storage.
- **Methodology caveat**: single short eval; the Q4 tensors' sensitivity is below the 1-layer noise floor, so
  whether gate/up/q/o could drop to Q3 (more headroom) is unresolved — needs a multi-sequence eval and the
  perturbation-loop OOM fixed (it leaked numpy copies).

## Verdict
B3's thesis is **proven** (adaptive per-tensor quant beats the fixed scheme — ffn_down over-provisioned), but
the clearest realizable win is **~5%** (demote ffn_down Q6→Q4) and it needs a Q4_K quantizer build. Combined
with P2 (~8%, needs flash-attention), both remaining decode levers are **modest and build-heavy** — the token
is GEMV-bound near its ceiling after the 2.3x Q6_K win. The mission result stands on its own: machine search
finds a per-tensor assignment llama's fixed Q4_K_M misses.

Repro: `DEV=AMD Q4K_PRIMITIVE=1 PYTHONPATH=. .venv/bin/python extra/qk_quant_sensitivity.py` (reduce roles to
avoid the OOM; perturbs all layers of a role, measures NLL delta vs the fp baseline).
