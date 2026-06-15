# Phase X — lossy-quantization-aware search (the first cross-layer rung)

Date opened: 2026-06-15
Goal: extend tinygrad's machine search from the SCHEDULE layer (semantics-preserving) to the
ALGORITHM layer's lossy-quantization choices (accuracy-trading), searched WITH accuracy as a
co-objective. This is the one rung between schedule-only search and the decode win that search cannot
currently propose -- because the win (int8 activation quant) is LOSSY and tinygrad's search preserves
semantics. It assembles three things that exist separately into the one place they're needed.

## Why this is the right frontier (grounded)

- Our whole arc localized the gap to layers tinygrad doesn't search: schedule is searched; ALGORITHM
  (lossy quant), LAYOUT, and INSTRUCTION-SELECTION are fixed/delegated. The decode win lives there.
- The research frontier shows going beyond schedule-only WORKS: **PET (OSDI'21)** searches *partially
  equivalent* transforms + automated correction (semantics-changing, +2.5x); **Mirage (OSDI'25)**
  does *joint* algebraic+schedule+custom-kernel multi-level search. Lossy quantization is a DIFFERENT
  kind of non-equivalence (accuracy-trading, not algebraic) -- the least-charted corner, NAS /
  approximate-computing-adjacent. This phase builds that corner on the decode problem.
- We already have the loop machinery (N1/N2: `qk_loop_learnability.py`, `qk_loop_search.py`) and the
  fast int8 kernel (Q-phase fused int-dot). X reuses both, adding the accuracy objective.

## HONEST ceiling (pre-registered up front)

Lossy-quant search does NOT exceed the int8 ceiling (~81 tok/s, ~78% of llama.cpp). Mixed precision is
a weighted average between fp-slow and int8-fast; it cannot beat uniform-int8 on pure speed (int8 is
the fast end). Its UNIQUE value is **accuracy-constrained speed, automatically**: capture the int8 win
(58->~81) on the layers that tolerate it, and find the MINIMAL fp fallback on the layers that don't --
WITHOUT manual per-layer tuning. The remaining 81->104 is the OTHER rungs (algebraic reformulation a la
Mirage, layout co-design, instruction selection / hand-asm), out of scope here. So X's prize is "the
int8 win, captured safely and automatically across a heterogeneous model," not parity.

## The new machinery vs the schedule loop (N1/N2)

1. **Lossy-transform vocabulary (the new search axis):** per-linear precision choice -- e.g.
   `{fp16 (current), q8_1-int8-activation (Q int-dot), maybe int4-activation}`. Expressed as a graph/
   dispatch choice in `Q4KPrimitiveLinear`.
2. **Accuracy evaluator + gate (the new objective):** per-layer output relative error (cheap proxy)
   for ranking; an end-to-end accuracy check (perplexity on a calibration set, or a task metric) for
   the budget gate. This is the genuinely new, expensive piece.
3. **Multi-objective joint search:** the cost model predicts (speed, accuracy); the search maximizes
   speed s.t. accuracy >= budget. Per-layer precision assignment -- the cross-layer co-design.

## Phases (cheap make-or-break FIRST; same discipline as N0b/D0/L0)

**X0 -- "is there room + structure?" probe (do FIRST; the whole idea's make-or-break).** Build a small
dataset over the model's ~7 linear TYPES x layers x precision choices {fp16, q8_1-int8}: measure
(per-layer output relative error, decode speed contribution). Pre-register the THREE conditions the
search needs a home (exactly the N0b test, on the precision axis):
- (a) ROOM: does int8-activation stay within an acceptable end-to-end accuracy budget (perplexity
  drift / output divergence on a calibration set)? If int8-everywhere TANKS accuracy -> no room, the
  lossy search has nothing safe to give. If int8-everywhere is FINE -> it's a constant ("just use
  int8"), no search needed -- and the answer is simply to ship uniform int8 (the Q phase).
- (b) HETEROGENEITY: is the accuracy tolerance DIFFERENT across layers (some int8-safe, some not --
  e.g. attn vs ffn, early vs late, down-proj sensitivity)? Uniform -> no search value; heterogeneous
  -> search has a real job (the mixed-precision assignment).
- (c) LEARNABILITY: is per-layer tolerance predictable from layer features (depth, type, shape,
  weight stats)? If yes -> the loop can learn it and transfer.
Gate: ROOM (some but not all layers tolerate int8) AND HETEROGENEITY AND LEARNABILITY -> the lossy
search has a home; proceed. ROOM=all -> just ship uniform int8 (Q), no X needed. ROOM=none -> int8
isn't accuracy-viable on this model, record and stop. Cheap: uses per-layer error proxies + the Q
kernel; no full search yet.

**X0 -- RESULT (2026-06-15): WEAK home -- int8 is broadly viable; the lossy SEARCH is not justified at
int8 -> ship uniform int8 (Q).** `lossy-X0/RESULT.md`. Captured per-layer int8 (q8_1) activation
tolerance + outlier stats during real decode (162 linears). (a) ROOM: int8 error 0.51-1.07% across ALL
layers -- broadly viable. (b) HETEROGENEITY: WEAK -- only 2.1x spread (vs the schedule space's 111-223x
in N0b); by type ffn_down/attn_output worst (~1%), ffn_gate/up best (~0.5%); depth ~flat. (c)
LEARNABILITY: YES -- `corr(outlier, int8_err)=0.757`. Verdict = the pre-registered "constant" branch:
no layer strongly needs fp, so per-layer mixed-precision search has little to give beyond "use int8
everywhere" -> ship uniform int8 (Phase Q). The search's real home is at MORE AGGRESSIVE precision
(int4-activation / mixed bit-widths), where outlier layers (the corr=0.757 predicts them) break hard
while others stay fine -- the AWQ/mixed-precision regime. Honest: this used the cheap input-error
proxy; an end-to-end perplexity gate could still surface a few int8-sensitive layers, but the proxy
says the effect is small. Net: another "search doesn't help here at int8; the simple thing (uniform
int8) does."

**X1 -- the lossy vocabulary + accuracy evaluator.** Wire per-linear precision dispatch into
`Q4KPrimitiveLinear` (fp16 vs Q int-dot). Build the accuracy evaluator: per-layer rel-error (fast) +
an end-to-end perplexity/calibration harness for the budget gate. Correctness here means accuracy
within budget, not bit-exactness.

**X2 -- the cross-layer loop (extend N1/N2 to multi-objective).** Reuse `qk_loop_learnability.py` /
`qk_loop_search.py`, but the cost model now predicts (speed, accuracy) per (layer, precision) and the
search assigns per-layer precision to maximize speed s.t. accuracy >= budget. This is the loop
co-designing the LOSSY ALGORITHM choice with the (already-searched) schedule -- the cross-layer move,
scoped to the quantization layer. Pre-register: the learned per-layer assignment beats both uniform-fp
(too slow) and a naive accuracy-blind uniform-int8 (if that violates budget), AND transfers across
layers (the N1b transfer test on the precision axis).

**X3 -- end-to-end measure.** The searched mixed-precision config's decode tok/s AND accuracy vs:
uniform-fp (58, full accuracy), uniform-int8 (~81, Q-phase, accuracy=?), llama.cpp (104). The win:
the highest tok/s that stays within the accuracy budget, found automatically. Pre-registered: if
uniform-int8 already meets budget, X reduces to "ship Q" (honest null -- the search confirmed a
constant); if not, X's mixed config recovers accuracy at minimal speed cost and that is the result.

## Pre-registered honesty + boundary

- This is OPEN frontier (searched lossy-quant co-design is unbuilt); X0 TESTS whether it has a home on
  this model, it does not assume it. Three honest null outcomes are possible and acceptable: no room
  (int8 not viable), constant (int8-everywhere fine -> just ship Q), unlearnable (no transferable
  structure).
- The accuracy evaluator is the hard/expensive part and the metric choice matters (per-layer proxy
  vs end-to-end perplexity vs task accuracy); X0 uses cheap proxies, X1+ needs a real harness.
- Ceiling restated: X tops out at the int8 ceiling (~81); it AUTOMATES capturing it safely, it does
  not exceed it. Going past 81 is the Mirage-style rungs (algebraic + layout + instruction), a
  separate, larger frontier.
- This is the natural extension of the result the whole investigation actually produced (the loop
  works on the schedule axis); X tests whether the SAME loop machinery extends to the lossy-quant
  axis -- i.e. whether "machine search" climbs from schedule to cross-layer on a real decode model.
- Touch points: `extra/qk_loop_{learnability,search}.py` (multi-objective), `extra/q4_k_gemv_primitive.py`
  (Q int-dot), `tinygrad/llm/model.py` (`Q4KPrimitiveLinear` per-layer precision dispatch), a new
  accuracy harness (`extra/qk_accuracy_eval.py`), calibration data.
