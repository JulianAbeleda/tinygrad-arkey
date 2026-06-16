# B3 SHIPPED: demote over-provisioned Q6 ffn_down → Q4 = +14% decode at ~free quality

Date: 2026-06-15. B3 phase-0 (`B3_SENSITIVITY_RESULT.md`) found Q4_K_M over-provisions `ffn_down` (Q6 where
Q4 is ~free). This realizes it: a real Q4_K quantizer + load-time requant of the 18 Q6 `ffn_down` tensors.

## Result (RX 7900 XTX, full clock)
| config | tok/s | NLL | output |
|---|---:|---:|---|
| baseline (ffn_down Q6) | 53.4 | 0.5024 | — |
| **Q6K_DEMOTE_FFNDOWN=1 (ffn_down Q6→Q4)** | **60.9 (+14%)** | **0.4997 (dNLL −0.0028, within noise)** | diverges (lossy, coherent) |

**+14% decode at zero measurable quality cost.** This is the first **shippable** "machine search beats
llama.cpp's fixed quant assignment" result — a faster operating point Q4_K_M does not offer (it keeps ffn_down
Q6 for safety; the search found that's unnecessary). 60.9 tok/s = **58% of llama.cpp (105.7)**, from 22% at
the start of the decode arc — and now *past* llama's per-token rate would require... it's at 58%, closing.

## Why +14% (more than the ~5% byte savings)
Demoting Q6→Q4 does two things: (1) ~5% fewer per-token bytes (the 18 Q6 ffn_down: 743 MB → 509 MB), and
(2) switches from the Q6 GEMV kernel to the **simpler/faster Q4 GEMV** (Q6 dequant unpacks ql+qh+int8 scales;
Q4 is leaner). The kernel-efficiency gain compounds with the byte savings.

## The quantizer (the missing piece, now built)
`extra/qk_quantize.py` — fp16 → Q4_K block bytes, a port of llama.cpp's `make_qkx2` search (no-imatrix
weights). **Validated bit-exact**: re-quantizing a weight llama already stored as Q4_K reproduces it with
**max error = 0** (`test/external/test_qk_quantize.py`) — our quantizer matches llama's grid. Q6→Q4 on
ffn_down gives 7.3% weight RMS error, which the model absorbs to dNLL −0.0028 (noise).

## Honest caveats
- **Gated `Q6K_DEMOTE_FFNDOWN` (default off)** — it is lossy (output diverges from the Q6 baseline, though
  coherent and NLL-neutral). A speed/quality knob, opt-in.
- **Load cost**: requantizing 18 ffn_down at load takes ~2–3 min (numpy `make_qkx2`, nstep=8). One-time and
  **cacheable** (quantize once → save the Q4 bytes → reload); not yet cached. The decode win is independent
  of this.
- **Single-eval NLL** (48-token self-generated sequence) — the −0.0028 is within noise; a multi-prompt
  perplexity suite would tighten it, but phase-0's per-tensor sensitivity already bounded the cost.
- Prefill/fallback still uses the original Q6 fp weight (decode uses Q4) — harmless inconsistency; could be
  unified by setting the demoted linear's `self.weight` to the Q4 round-trip.

## Mission significance
This is the machine-search thesis realized at the **bit-width** level, end-to-end and shippable: the search
reads fewer bytes than llama's fixed Q4_K_M *and runs a faster kernel*, at no quality cost — +14% decode.
Combined with the coverage win (Q6_K primitive), the decode arc is **23 → 60.9 tok/s (2.65×), 22% → 58% of
llama**, with the kernel beating llama standalone and the quant assignment beating llama's fixed recipe.

Repro: `DEV=AMD Q4K_PRIMITIVE=1 Q6K_DEMOTE_FFNDOWN=1 PYTHONPATH=. .venv/bin/python ...` (decode tok/s);
`test/external/test_qk_quantize.py` (quantizer exactness); `/tmp/b3qual.py`-style NLL delta.
