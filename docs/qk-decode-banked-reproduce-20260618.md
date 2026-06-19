# 8B decode banked line — REPRODUCED on HEAD (2026-06-18)

Health-check re-measurement of the banked 8B decode line on the current tree (`2b81dff42`), prompted by a
ctx-sweep request. **Verdict: reproduced dead-on; nothing regressed; the whole stack is default-on.**
Qwen3-8B-Q4_K_M, RX 7900 XTX / gfx1100, llama.cpp ≈ 101–106 tok/s.

## Result — in-model W==D (authoritative method)

`extra/qk_decode_runtime_overhead.py` — `TinyJit(m.forward)` + device-token feedback (`out = step(out, …)`) +
`.item()` readback per token (the real `model.generate` sync path), NMEAS=40 median, perf level `auto`.

| ctx | W (real decode) | D (dispatch ceiling) | host-sync | banked | match |
|---|---|---|---|---|---|
| 128  | 71.5 | 70.2 | 0.0% | — | — |
| 512  | **68.2** | 67.4 | 0.0% | 68.3 | ✓ |
| 1024F | **66.4** | 65.5 | 0.0% | 66.3 | ✓ |
| 4096F | **60.7** | 59.9 | 0.0% | 60.9 | ✓ |

Within 0.2 tok/s of banked at every ctx → **~67% of llama, slope −11% across ctx 512→4096.** `host-sync 0.0%`
(W≈D) at every depth re-confirms **decode is GPU-bound**, not host/runtime-overhead-bound (the measured
refutation of the old "55% host" theory; see `qk-runtime-overhead-arc-result-20260617.md`). Artifact:
`bench/qk-decode-runtime-overhead/result.json`.

## The stack is the AMD default (answering "why isn't this default?")

It **is** default on this machine. `model.py:1046-1051`: `Q4K_PRIMITIVE` auto-defaults ON when
`Device.DEFAULT == "AMD"` **and** the gguf is a file *path* (not a pre-loaded Tensor); `Q6K_PRIMITIVE` follows.
The coop kernels (`Q6K_LM_HEAD_COOP` / `Q6K_FFN_DOWN_COOP` / `Q4K_ATTN_QO_COOP`) and `FLASH_VARIANT=gqa_coop_vec`
/ `FLASH_DECODE=auto` (threshold 512) are all `getenv(…, 1)` default-on. Plain `model.generate` / the `llm` CLI
gets 68.2/66.4/60.7 out of the box. Silent off-switches: a non-AMD device, or passing the gguf as a `Tensor`
(then `q4k_auto` is False). Genuinely opt-in (by design): `PREFILL_V2`, `AMD_COMPUTE_RINGS=2`, spec decode,
`QK_GENERATED_POLICY` (14B/32B).

## Measurement lesson (banked)

The flash auto-bench (`extra/qk_flash_decode_auto_bench.py`) reads **~54–56 flat** — it is a flash-**policy**
gate, not a production tok/s measurement. **Why it reads low: it creates a fresh host
`Tensor([[tokid]]).contiguous()` inside the timed loop every step** — the per-step Tensor-creation artifact that
the runtime-overhead arc already proved *halves* the rate (it serializes host work against the GPU). The W==D
harness feeds the device output token back (`out = step(out, …)`) with no per-step host tensor, exactly as
`model.generate` does → the clean GPU-bound 68.2. **For the decode tok/s headline always use W==D; the
auto-bench's absolute number is contaminated by design** (it exists to test policy/selection, not throughput). `rocm-smi --setperflevel high` was a **red herring** (did not lift numbers, depressed
SDPA-off); auto-boost is correct. No kernel/model/default changes.
