# Phase X0 RESULT (2026-06-15): weak home -- int8 is broadly viable; lossy-quant SEARCH adds little.

Captured per-layer int8 (q8_1) activation tolerance + outlier stats during real decode (4 tokens, 162
distinct Q4_K linears) via a QK_CAPTURE hook in `Q4KPrimitiveLinear` (reverted).

## The three pre-registered conditions
- (a) ROOM: YES, broadly -- int8-activation relative L2 error is 0.51% - 1.07% across ALL layers.
  int8 doesn't tank accuracy anywhere; it is broadly viable.
- (b) HETEROGENEITY: WEAK -- only 2.1x spread (vs the schedule space's 111-223x in N0b). Real but
  modest structure: by TYPE ffn_down (0.93%) & attn_output (0.82%) worst, ffn_gate/up (0.65%) best;
  by DEPTH nearly flat (early 0.72% -> late 0.79%). Activation outliers present (max/mean per block up
  to 8.2 on blk.20.attn_output) -- the known LLM-quant outlier signal -- but the error impact is small.
- (c) LEARNABILITY: YES -- corr(outlier_ratio, int8_err) = 0.757; a cheap feature predicts tolerance.

## Verdict: the "constant" outcome -> ship uniform int8 (Q), the lossy SEARCH is not justified at int8
int8 activation is broadly safe (<=1.1% per-layer error, weakly heterogeneous). No layer strongly
needs fp, so per-layer mixed-precision SEARCH has little to give beyond "use int8 everywhere." The
pre-registered ROOM=all branch applies: ship uniform int8 (Phase Q, ~81 tok/s); do NOT build the
multi-objective search for int8.

## Where the lossy-quant search WOULD have a home (the honest next step if pursued)
The search needs LARGE, exploitable heterogeneity. At int8 there isn't (int8 is too safe). It would
appear at MORE AGGRESSIVE precision -- int4 activation / mixed bit-widths -- where outlier layers
(attn_output, ffn_down; the corr=0.757 predicts them) break down hard while others stay fine. That is
exactly the regime AWQ / mixed-precision methods target. So Phase X's real form is "search mixed
bit-widths (int4/int8/fp per layer) under an accuracy budget", and X0 says: at int8 it's a constant,
at int4 it likely becomes a real search. Caveat: this used the cheap input-error proxy; an end-to-end
perplexity gate (X1) could still surface a few int8-sensitive layers, but the proxy says the effect is
small.

## Net
X0 did its job: the cheap probe shows the cross-layer lossy-quant SEARCH does not have a strong home
at int8 on this model -- uniform int8 (Q) captures the win, and the search only earns its keep at more
aggressive precision. One more "search doesn't help here, the simple thing does" -- the recurring,
honest pattern.
