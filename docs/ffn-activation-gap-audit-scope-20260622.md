# FFN Activation Gap Audit — Scope (2026-06-22)

**Phase 1 of 8B decode-gap exhaustion.** Audit only; no kernel/default change.

## Question
The tinygrad-vs-llama time-tax diff put ~1.3–1.6 ms/token in an **"FFN activation"** bucket at a
**10–20× ratio** to llama (`docs/tinygrad-vs-llama-decode-time-tax-diff-result-20260622.md`). Before
proposing any activation-fusion primitive, determine whether that gap is:
1. **Real** — is there genuinely an expensive activation op?
2. **Mapped correctly** — are the kernels in this bucket actually the silu activation?
3. **Critical-path** — does the cost translate to wall token_ms (transfer), or is it overlapped?
4. **Bounded** — is there a tractable lever?

## Method
- Render the decode kernels and read ground truth, not the name heuristic. Capture per-kernel
  AST op-histograms + source-derived flags (`exp`/`sin`/`sqrt`/`uchar`/`start_pos`) via
  `extra/qk_decode_audit_common.py` → `bench/qk-decode-kernel-probe/latest.json`.
- The diff's bucket came from `classify()`'s `E_49152|E_1536 → ffn_activation` heuristic. Test it:
  does `E_49152` contain an exponential (silu = x·σ(x))? Where does silu actually live?
- ctx-scaling: is the bucket flat (MAXC-bound) or ctx-dependent?
- Transfer: MAXC-shrink test — if the kernel scales with `max_context`, shrink it and measure the
  wall token_ms delta (does it transfer?).

## Mapping rule under test
`classify()` (`extra/qk_decode_time_tax_audit.py:28`): `n.startswith("E_49152") or n.startswith("E_1536") → ffn_activation`.
This is a **name heuristic with no verification** — the audit validates or refutes it from the rendered source.

## Deliverables
`extra/qk_ffn_activation_gap_audit.py`, `bench/qk-ffn-activation-gap-audit/latest.json`, this scope + the result doc.

## Stop condition
If `E_49152`/`E_1536` are not the activation (mapping artifact), report it and hand the reclassified
bytes to the small-ops sub-audit / decision doc — do **not** propose an activation primitive.
