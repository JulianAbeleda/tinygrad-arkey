# Decode / Prefill Headline Reconciliation Result

Date: 2026-06-21 (revised — fresh clean-wall reruns + exact-command provenance).

Verdict: **`87.6` is a genuine NUMERIC COINCIDENCE — it is BOTH a real ctx≈0 decode `tok/s` AND, separately, a real
ctx4096 decode `ms/token` (= 11.4 tok/s).** The reported "87.6 tok/s (matches banked ~86)" was the **real ctx≈0
`tok/s`** (verified by re-running the exact `--benchmark` command today → 85–86 tok/s @ 11.6 ms), NOT the ms-value
mislabeled. **Either reading lands at the same headline decision: ctx≈0 ~86 is real but is NOT comparable to the
canonical ctx512–4096 table → the `~67% llama` decode headline STANDS.** No decode-ROI rerank; decode default not
changed; and (new) a recommendation to NOT flip the global `PREFILL_V2=auto` default. Artifact:
`bench/qk-headline-reconciliation/result.json`.

> Supersedes the earlier "stale unit mixup" framing: the ms-collision artifact is real (below), but the *reported*
> number was the genuine ctx≈0 tok/s, not that artifact. The actionable conclusion is unchanged.

## 1. `87.6` provenance — TWO real, opposite-unit sources (that is the trap)

| number | unit | source | ctx | other axis |
|---|---|---|---:|---|
| `87.06`–`87.57` | **tok/s** | CLI `--warmup --benchmark`, `bench/qk-ctx-sweep-20260618/cli_benchmark.out` | **≈0** | 11.4 ms/token |
| `85`–`86` | **tok/s** | CLI `--benchmark` rerun 2026-06-21 (default + `PREFILL_V2=auto`) | **≈0** | 11.6 ms/token |
| `87.62` | **ms/token** | `bench/qk-long-context-20260617/result.json` (`decode_ms_per_token`) | **4096** | 11.4 tok/s |
| `87.942` | **ms/token** | `bench/qk-flash-decode-auto-20260617/result.json` (`decode_ms`) | **4096** | 11.4 tok/s |

The number `87.6` is maximally confusing because ctx≈0 decode is **~87 tok/s @ ~11.4 ms** while ctx4096 decode is
**~87.6 ms @ 11.4 tok/s** — the same ~11.4 appears on the *opposite* axis. The "87.6 tok/s" report came from the
CLI `--benchmark` path (which literally prints `tok/s` first, `ms` second) at ctx≈0 → it is the **real ctx≈0 rate**,
reproduced today at 85–86, NOT the ms artifact. But because the bare number collides with a ctx4096 ms-value, it
must **never** be quoted as "the decode headline" without its context. Mark a bare "87.6" **ambiguous / non-headline**.

## 2. Clean-wall decode table (authority: PROFILE=0, no DEBUG, auto clock, this HEAD)

| ctx | tok/s | harness | llama ref | % llama |
|---:|---:|---|---:|---:|
| ≈0 | **~85–86** | CLI `--benchmark` (empty KV; decays over the run) | ~100 (tg) | ~86% |
| 128 | 71.5 | `qk_decode_runtime_overhead.py` W path | — | — |
| 512 | 68.1 | W path | ~98.6 | ~69% |
| 1024 | 66.4 | W path | ~97.6 | ~68% |
| 4096 | 60.7 | W path | ~92.2 | ~66% |

W-path 512/1024/4096 = **68.1 / 66.4 / 60.7** — **exactly reproduces the banked canonical table**. Decode decays
~86 → ~61 as KV/attention grows; ctx≈0 is fastest because attention is near-free at empty KV. Gap to llama **grows
with context** (~86% @ctx≈0 → ~67% @ctx512–4096) — the known attention slope.

## 3. Prefill policy / default — decode impact (clean reruns)

`PREFILL_V2` only swaps the **prefill** JIT; decode (T==1) always runs `_prefill_v2=False`. Clean-wall decode under
each mode (tok/s @ ctx 128/512/1024/4096):

| mode | ctx≈0 | 128 | 512 | 1024 | 4096 | decode VRAM pool |
|---|---:|---:|---:|---:|---:|---:|
| default | ~85 | 71.5 | 68.1 | 66.4 | 60.7 | 6.3 GB |
| `PREFILL_V2=auto` | ~86 | 72.1 | 68.1 | 66.5 | 60.8 | **20.2 GB** |
| `PREFILL_SERVER_PROFILE=1` | — | 71.9 | 68.3 | 66.7 | 61.0 | **20.2 GB** |
| `Q8_FFN_HANDWRITTEN=1` (opt-in) | — | 76.6 | 72.8 | 70.9 | 64.3 | 6.3 GB |

**No decode regression:** auto/server are within **<1%** of default at every context, output **identical** (same
T==1 code). q8 opt-in is +~7% (dNLL-gated). **New caveat:** `PREFILL_V2=auto`/server keep **+14 GB fp16 prefill
weights resident for the whole session, including decode** (20.2 GB pool vs 6.3 GB), unused by decode.

## 4. Updated project headline

| area | default | opt-in / server | llama-relative status | next action |
|---|---|---|---|---|
| prefill | universal path (slow long prompts) | `PREFILL_V2=auto` (~5–15× faster, +14GB) · `PREFILL_SERVER_PROFILE=1` (warm prefill 0.17–1.6s) | concrete-KV **73–111% of llama pp512**; `PREFILL_REMAINDER_FIX` kills the 32-tok trap | kernel-solved; **owner: keep `auto` opt-in (do NOT flip global default — §5)** |
| decode | **~85–87 @ctx≈0; 68/66/61 @ctx 512/1024/4096** | q8 opt-in 76.6/72.8/70.9/64.3 (+~7%) | **~67% llama @ctx (≈86% @ctx≈0)** | decode is the real frontier — attention/elementwise fusion |

Prefill and decode are separated; prefill policy does **not** change decode speed. The honest decode headline is the
**curve** (~86 @ctx≈0 → ~61 @ctx4096); quoting only ctx≈0 would cherry-pick → **keep "~67% llama" as the
steady-state characterization** (ctx≈0 ~86 as the peak). 87.6 was a real ctx≈0 number, so the decode frontier is
**unchanged**.

## 5. Owner recommendation on global `PREFILL_V2=auto`

**Do NOT flip the global default to `auto`.** Keep it opt-in (with the shipped CLI hint + `PREFILL_SERVER_PROFILE`).
Grounded in this reconciliation: `auto` realizes **+14 GB fp16 weights resident for the whole session including
decode** (20.2 GB pool, +~11 s load) but benefits **only long prefills**. A decode-only / short-prompt user — the
common case — pays the full VRAM+load cost for zero gain, and the 20 GB resident pool also eats long-context KV
headroom on a 24 GB card. The fast path is one flag away and the CLI already recommends it on large GPUs. Revisit
only if the Phase-5 VRAM reduction lands, or if auto-enable is gated on an actual long-prompt/`--serve` signal
rather than at load.

## Commands
```bash
# ctx≈0 clean-wall (the real ~86 tok/s):
DEV=AMD PYTHONPATH=. .venv/bin/python -m tinygrad.llm -m <Qwen3-8B-Q4_K_M.gguf> --warmup --benchmark 30
# ctx128-4096 clean W path (default / PREFILL_V2=auto / PREFILL_SERVER_PROFILE=1 / Q8_FFN_HANDWRITTEN=1):
DEV=AMD JIT=1 [env] PYTHONPATH=. .venv/bin/python extra/qk_decode_runtime_overhead.py
```

## Stop-condition check (all clear)
87.6 traced to exact artifacts+commands (both unit forms) ✓; decode reruns clean-wall + labeled ✓; llama bar kept
historical/labeled ✓; prefill rows do not change decode output, regress <1% ✓; 16GB/unknown verified via the
auto-decision unit tests ✓. No defaults changed.
