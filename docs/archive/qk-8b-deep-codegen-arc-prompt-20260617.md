# NEXT-SESSION PROMPT — 8B deep-codegen decode arc (flash-decode tile/reduce tuning)

Paste the block below to resume. It is self-contained; it points at the banked state so the new session starts
from solid ground without re-deriving.

---

Attack the 8B deep-codegen decode lever: flash-decode tile/split + attention-reduce tuning.

Context:
- Repo: /home/ubuntu/tinygrad-arkey, branch master, clean + pushed.
- Model: Qwen3-8B Q4_K_M on RX 7900 XTX (gfx1100). llama.cpp ≈ 101–106 tok/s.
- READ FIRST: `docs/qk-8b-decode-banked-20260617.md` (current bank) and `docs/qk-8b-attention-fusion-result-20260617.md`
  (the just-shipped flash-threshold win). Memory `amd-decode-next-step.md` points at the current bank.
- All bounded/local 8B levers are shipped, refuted, or necessary — see the bank's "closed/refuted" table
  (do NOT re-explore: GEMV final-mile, sub4, small-op fusion, big-copy, ring2-decode, spec-decode integration,
  host/runtime overhead). The 8B gap is GPU-kernel-structural (~780 progs/token vs llama ~260), not host/runtime
  (normal decode is GPU-bound, W==D proven).
- Just shipped: FLASH_DECODE_THRESHOLD 1024→512 (+12.8% real ctx520 decode, byte-identical). flash-decode now
  covers ctx≥512.

Goal:
Find whether a codegen-level change to the decode attention path (flash-decode kernel internals, or the SDPA
softmax-reduce sequence) improves base decode beyond the shipped threshold win — without a broad rewrite.

Hard rules:
- No broad fusion framework, no decode-block rewrite as a first move.
- Measure with the carried-forward discipline: warm device-token-feed (W vs D, see
  `extra/qk_decode_runtime_overhead.py`) for decode timing — NOT eager DEBUG=2 (unbatches→inflates), NOT
  per-step Tensor creation (~2× host artifact). Verify every isolated win in-model (full model.generate),
  byte-identical greedy, before changing a default. Multi-window dNLL for any quality claim.
- Do not change defaults unless full ctx512+ decode improves and greedy output is byte-identical.
- Stop at kill gates. If the lever is the known linearizer wall (coupled multi-accumulator reduce rejection —
  why flash-decode splits softmax across 5 kernels), document the exact blocker and STOP; do not attempt a
  linearizer rewrite without an isolated proof of value.

Success gate:
- Full ctx512 decode +5% beyond the current (post-threshold) baseline; strong ≥10%.
- Or program-count reduction with a measured tok/s win (no count-without-speed).
- If isolated win doesn't carry to full decode, unwire and record refutation.

Phases:
0. [test] Characterize the flash-decode kernel as actually emitted at ctx512–1024: its split (how many kernels,
   the tile/partition over KV, the single-accumulator decomposition). `extra/qk_flash_decode.py` is the builder;
   `extra/qk_attention_kernel_map.py` maps the SDPA path (note: eager uses concrete start_pos → SDPA; the flash
   path needs symbolic start_pos in the JIT, so capture it via a JIT decode, not eager m.logits). Quantify where
   flash-decode's time goes at KV~512–1024.
1. [test] Identify the smallest codegen/tuning candidate: (A) flash-decode KV tile/split size for KV~512–1024;
   (B) reduce the flash kernel count if the single-accumulator constraint allows; (C) a ctx<512 SDPA softmax
   collapse (low value — ctx<512 is GEMV-dominated and a small slice of generation). Score each: kernels touched,
   expected tok/s, risk, isolated gate, full-model gate. Pick the smallest with plausible ≥5%.
2. [test] Isolated repro of the chosen candidate (exact vs SDPA/flash reference, the tiny-shape harness pattern
   from the prior flash-prefill work). Gate: exact + faster isolated.
3. [nn/codegen] Implement only if the isolated gate passes; env-gated; keep fallback.
4. [test] Before/after at ctx 128/512/1024/4096: decode tok/s, argmax/greedy identity, no regression.
5. [docs] Verdict: shipped / refuted / blocked-by-linearizer / not-worth-local. Next: deeper codegen, or stop →
   14B.

Commit plan: [test] flash-decode internals map; [test] candidate + isolated repro; [docs] candidate selection;
[nn]/[codegen] prototype only if earned; [test] before/after; [docs] verdict.

Stop condition: if Phase 1 finds no candidate with plausible ≥5% full-decode upside, or Phase 2's isolated win
doesn't materialize, STOP and report — deep codegen is not worth local work, move to 14B.

---

State at handoff (2026-06-17): clean tree, pushed through `8269ecf9e`. Last shipped: `d05973e2b` (flash-threshold
512). No background work running; GPU idle.
