# Current Project State — Handoff (2026-06-30 refresh)

Canonical, high-signal snapshot. If anything elsewhere contradicts this file, this file wins.
Machine: gfx1100 RX 7900 XTX 24GB, Qwen3-8B-Q4_K_M. Supersedes the 2026-06-21 handoff (available from
git history), whose `~67% llama` decode baseline and "bounded decode RESTED / capped at backend
ceiling" frontier were overturned by the 06-22→06-30 campaign (owned attention tile + buffer-identity fix,
generated G3 Q4_K route parity, Q6_K direct-route refutation, and prefill pipe promotion).

## 1. Canonical numbers (clean-wall, PROFILE=0, auto clock, W==D)

| metric | value | source |
|---|---|---|
| decode @ctx 512 / 1024 / 2048 / 4096 | **103.9 / 102.0 / 99.7 / 94.4 tok/s** (G3 speed-equivalent to owned Q4_K; Q6_K direct route refuted/default-off) | `bench/amd-isa-backend-g3-weight-promotion/latest.json`, `bench/amd-isa-backend-q6k-direct-speed/latest.json` |
| decode practical ceiling | **~110-130 tok/s** for this model/GPU/quant stack; further weight-kernel tuning is closed absent representation/primitive changes | system-residual and Q6K-3 audits |
| llama reference (same ctx) | 97.71 / 97.39 / 95.00 / 92.37 tok/s | `bench/canonical-benchmarks.json` |
| prefill @ctx 512 / 1024 / 2048 / 4096 / 8192 | **4434 / 4236 / 3846 / 3192 / 2532 tok/s** (**role-selective** pipe promoted default — pipe on attn_qo/attn_kv/ffn_down, gate/up kept on its faster path; rollback `PREFILL_PIPE_ROLE_SELECTIVE=0` → global pipe, then `PREFILL_GEMM_PIPELINE=0` → old lds2) | `bench/qk-prefill-pipe-role-selective/latest.json` |
| q8 FFN opt-in | ~+7% decode, **default-OFF, dNLL-gated** | `Q8_FFN_HANDWRITTEN=1` |
| VRAM | default ~5–6 GB; **`PREFILL_V2` adds ~+14 GB fp16** (≈19–21 GB), resident through decode | handoff history |

**Decode caveat (do not mis-state the current result).** Generated G3 is promoted/hardened as the Q4_K
speed-equivalent route; it replaces the owned warp route where eligible but does not exceed it. The Q6_K direct
half-warp/lane-map route was wired correctly and token-correct, then refuted by W==D (-6.1/-5.8/-5.1/-4.8% across
ctx512→4096), so it remains default-off. Current decode max-out is practical-ceiling/documentation work, not another
Q4_K/Q6_K route search.

## 2. Decided policies (do not re-open)

- **Global `PREFILL_V2` default stays OFF** (decided 2026-06-21, unchanged). Not flipped to `auto` — the +14 GB
  fp16 prefill state stays resident during decode for zero decode benefit; the common decode/short-prompt user
  must not pay it.
- **`PREFILL_V2=auto`**: opt-in (VRAM-gated; 24GB+ on, ≤16GB / unknown off).
- **`PREFILL_SERVER_PROFILE=1`**: opt-in (⇒ `PREFILL_V2=auto` + concrete-KV precompile; server/long-prompt profile).
- **`PREFILL_REMAINDER_FIX`**: default-ON but only active under `PREFILL_V2`; byte-identical (kills the 32-token trap).
- **q8 FFN (`Q8_FFN_HANDWRITTEN=1`)**: opt-in, default-off.
- **Q4_K G3 LaneMap**: now the **default** Q4_K decode GEMV route (`BUBBLEBEAM_FUTURESIGHT` default-on, model.py:255); speed-equivalent to the owned warp kernel, which is the rollback (`BUBBLEBEAM_FUTURESIGHT=0`).
- **Q6_K direct route**: refuted by W==D, default-off; current coop/default route remains.
- **Prefill role-selective pipe**: promoted default (`PREFILL_GEMM_PIPELINE=1` + `PREFILL_PIPE_ROLE_SELECTIVE=1`, pipe on attn_qo/attn_kv/ffn_down, gate/up on its faster lds2 path); rollback `PREFILL_PIPE_ROLE_SELECTIVE=0` → global pipe, then `PREFILL_GEMM_PIPELINE=0` → old lds2. (Global `pipe_tm2_tn2` is now the A/B rollback comparator.)

## 3. What changed since the 06-21 handoff (the parity win)

The 06-21 frontier called bounded decode RESTED and decode "capped at tinygrad's backend ceiling (~67% llama),
closable only by a deep, separately-funded codegen capability." The 06-22→06-24 campaign refuted that:

- **Attention was not exhausted.** The owned hand-AMDGCN decode-attention tile (north-star lane, not a bounded
  patch) added +12–22% and entered the decode path — see `decode-campaign-final-synthesis-20260623.md` and
  `post-owned-attention-promotion-synthesis-20260623.md`.
- **Buffer identity was the actual wall**, not a runtime-KV core block; resolving it unblocked W==D promotion.
- **Weight-GEMV** reached at/below llama, then generated G3 LaneMap matched the owned Q4_K route under BubbleBeam/FutureSight.
- **Q6_K direct routing** was tested after the system-residual audit and refuted; the apparent lm_head reduce win was gumbel/sampling attribution, not removable GEMV work.
- **Prefill** moved from `eightwave` → global `pipe_tm2_tn2` (+19.3/+16.8/+14.2/+11.3/+8.5% across ctx512→8192) → then the **role-selective** pipe (excludes the saturated gate/up where the pipe regressed ~17%): a further +2.9..3.7% over global pipe (+11.7..23.4% over old lds2), output-equivalent, rollback chain available. Role-selective is the current promoted default; global pipe is the rollback comparator.

Net: decode is effectively closed under the current representation/primitive set; prefill now carries the live promoted
TIER_A win and is the more promising frontier for further role-selective pipeline/search work.

## 4. Where to start

1. `docs/README.md` — curated navigation map.
2. `docs/current-project-state-handoff-20260624.md` — this file (canonical current state).
3. `bench/README.md` — bench/evaluator map.
4. `docs/decode-campaign-final-synthesis-20260623.md` — how decode reached parity.
5. `docs/prefill-decode-next-workstreams-codex-scope-20260624.md` — next-work map.

Historical provenance from the full 06-16→06-22 probe log, superseded results, and completed scopes was removed
from the tracked tree. Recover it from git history when needed; it is history, **not authority**.

## Consistency guardrail
Run `DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_policy_consistency_check.py` — it scans the canonical
START-HERE docs and fails if one re-opens a closed question (bare `87.6` with no context, an open
`PREFILL_V2=auto` owner call, a "flip global PREFILL_V2=auto" proposal, or bounded decode fusion presented as
current work).
