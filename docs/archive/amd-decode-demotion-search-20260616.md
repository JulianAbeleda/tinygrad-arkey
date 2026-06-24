# B3 per-tensor demotion search — 2026-06-16

First end-to-end use of the machine-search scaffold (`extra/qk_search_spec.py`). The loop the
direction prescribes, run for real:

    search spec (SearchRow) -> candidate Q6->Q4 demotion sets -> isolated runner
    (tok/s via tinygrad.llm.cli, dNLL via extra.qk_nll_eval) -> quality gate -> AcceptedPolicy

Pure orchestration (`extra/qk_demote_search.py`) over the two existing measurement CLIs — no new
measurement code. The `[nn]` change that made it possible: the Q6->Q4 demotion target set is now
data (`QK_DEMOTE_TENSORS`, a searched per-tensor list) instead of the hardcoded ffn_down flag.

## Frontier (Qwen3-8B-Q4_K_M, RX 7900 XTX; epsilon = dNLL budget 0.01; llama.cpp = 101.2 tok/s)

| demotion set | tok/s | % llama | dNLL | faster | quality | accepted |
|---|---:|---:|---:|:-:|:-:|:-:|
| baseline (none) | 56.0 | 55.3 | 0 | — | — | — |
| ffn_down | 64.3 | 63.5 | +0.0005 | Y | pass | **ACCEPT** |
| ffn_down + attn_v | 63.7 | 63.0 | −0.0103 | Y | pass | **ACCEPT** |
| ffn_down + attn_v + **output** | **75.0** | **74.1** | **+0.0509** | Y | **FAIL** | — |

Accepted-policy artifacts: `bench/qk-demote-search/accepted-{ffn_down, ffn_down+attn_v}.json`.

## Verdict — the quality gate works; the in-pattern lever is tapped at the budget

- **The gate did its job.** Demoting `output` (lm_head, 622 MB Q6→Q4) is by far the *fastest* —
  75 tok/s, **74% of llama, +34%** — and a naive "fewer bytes = faster" policy would have shipped
  it. But it raises dNLL by **+0.051** (a real ~2% quality loss, far above the ~0.01 calib noise),
  so the search **rejected** it. This is the machine-search thesis validated: search gains the
  per-tensor bit-width degree of freedom, and the quality gate keeps it honest.
- **The lever is essentially tapped within the quality budget.** ffn_down was the real win
  (already shipped); attn_v is quality-neutral but too small to add throughput (ffn_down+attn_v
  ≈ ffn_down ≈ 64 tok/s, the difference is run-to-run noise); lm_head is the only big byte payoff
  and it fails quality. So **~64 tok/s (63% of llama) is the Q6→Q4 demotion frontier** — the
  current shipped operating point is at it.
- **The bigger bytes require sub-4-bit** (Q3/Q2 on the Q4 bulk = 72% of weight bytes), which needs
  a new quantizer + new GEMV kernel = dangerous-power surface — deliberately deferred (the same
  reason overlap's 2nd-ring build is gated).

## Honest caveats

- **dNLL noise ~0.01 over 128 calib tokens.** attn_v's −0.0103 is within that noise (read as
  "neutral", not a real improvement); lm_head's +0.0509 is clearly real (5× the noise). The
  *conclusions* are robust to the noise; a stricter gate would use more calib tokens.
- The evaluator reproduces the **direction** of the capstone ffn_down result ("≈ free") but not its
  exact −0.0028 — dNLL is calibration-data-dependent and our fixed passage differs from the
  capstone's set. It is *sensitive* (detects every demotion) and *discriminating* (lm_head's +0.05
  ≫ ffn_down's +0.0005), which is what the gate needs.

## What this delivered (beyond tok/s)

The machine-search pipeline shipped this session now runs **end-to-end on a real lever**: typed
search rows, an isolated runner, a quality-gated scorer, and durable `AcceptedPolicy` artifacts —
with the gate demonstrably refusing a tempting-but-degrading config. That working, reusable loop
(and the documented per-tensor byte/quality frontier) is the deliverable; the throughput was
honestly expected to be modest and is.

Anchors: `amd-decode-capstone.md` (ffn_down +14% / dNLL −0.0028), `qk_search_spec.py` (scaffold),
`amd-decode-beyond-llama-roadmap.md` (B3), `machine-search-decode-context-plan-2026-06-16.md`.
