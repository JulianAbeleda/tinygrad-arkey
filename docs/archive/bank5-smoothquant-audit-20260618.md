# Bank 5 — SmoothQuant / model transform (audit) 2026-06-18

Hypothesis: transform the model so activation outliers shrink → W4A8 becomes viable.

## Decisive premise check
SmoothQuant's value is reducing activation outliers that make activation quantization inaccurate. But we already
measured the q8_1 activation quant for the Qwen3-8B FFN input: **rel err 0.006** (rmsnorm output → q8_1). That's
small — **activation outliers are NOT a major q8 accuracy problem for this model/role.**

Crucially, **the W4A8 blocker is not q8 accuracy — it is q8 PACK COST** (Bank 2 wall: reuse ceiling 2 + ~7µs
per-kernel floor > the 4.8µs break-even). SmoothQuant reduces outliers; it does **not** make the q8 pack cheaper.
So even a perfect SmoothQuant transform leaves the actual blocker (pack cost) untouched.

## Cost / risk
Large separate arc: calibration data, per-channel activation-scale collection, weight rescaling, GGUF rewrite,
new artifact/policy format, dNLL gates. Changes the model premise (no longer a stock GGUF).

## Verdict: NOT WORTH IT for the stated goal (decode speed)
- It targets a problem we don't have (q8 accuracy is already fine, rel 0.006).
- It doesn't touch the real blocker (q8 pack cost).
- High effort + model-format surface for no speed mechanism.
- **Rank last.** Only revisit if the goal becomes accuracy-at-lower-bit (e.g. W4A4 / sub-4-bit, which sub4 already
  refuted on dNLL) rather than decode speed. Defer indefinitely.

## Files
`[docs]` this. No code/model changes.
