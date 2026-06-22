# Owned AMDGCN Decode Attention — Default Flip (2026-06-23)

## Verdict: **OWNED_ATTENTION_DEFAULT_FLIP_CONFIRMED** — `default_on=true`

The owned AMDGCN decode-attention route (native fp16 cache) is now the **default** decode attention for the
validated shape/device, after a final clean confirmation on the **shipped code** with the **canonical W==D
harness**. Byte-identical to gqa across the whole decode range; **+12.7/+15.4/+18.7/+22.4% @ctx512/1024/2048/4096**.
Every other shape/device stays on gqa+fp32; `DECODE_ATTN_AMDGCN_TILE=0` fully restores the old path.

## 1. Final confirmation (canonical harness, shipped code)
`extra/qk_decode_runtime_overhead.py` (`tok_s_W` = real decode wall/token, mirrors `model.generate` with a
per-token `.item()` readback). The harness was aligned to the shipped real-generate path (ctx list env-overridable;
flash capability tracks `FLASH_DECODE_THRESHOLD`, default 512). Baseline = `DECODE_ATTN_AMDGCN_TILE=0` (gqa fp32);
candidate = default (owned route, fp16 cache):

| ctx | gqa tok/s | owned tok/s | Δ |
|---|---|---|---|
| 512 | 76.1 | **85.8** | **+12.7%** |
| 1024 | 74.0 | **85.4** | **+15.4%** |
| 2048 | 71.0 | **84.3** | **+18.7%** |
| 4096 | 67.1 | **82.1** | **+22.4%** |

Clears every gate (≥+5%@1024, ≥+7%@4096, no regression). Consistent with the FO2 interleaved repeats
(+13.1/+16.0/+18.8/+23.2%). (`bench/qk-owned-attention-default-flip/confirmation.json`.)

## 2. Correctness (byte-identical to gqa, shipped code)
- **Short-ctx decode** (ctx≈8–13, **SDPA path** over the fp16 cache): `[279, 1156, 22148, 18495, 1033, 5798]` —
  byte-identical to the fp32 baseline. (Critical: flipping makes the cache fp16 for the *whole* decode range, not
  just where the owned tile fires; SDPA over fp16 is byte-identical.)
- **ctx1024** 64-token, **ctx512/2048**: byte-identical to gqa (real chunked-prefill, token authority).
- No `151936` collapse; no NaN.

## 3. Route / fallback (shipped code)
- **Default (no flags)**: owned route **fires** on the supported shape (ctx1024 `owned_flash` nodes = 2), cache
  `dtypes.half`.
- **`DECODE_ATTN_AMDGCN_TILE=0`**: gqa, cache `dtypes.float` (fp32) — old behavior restored, 0 owned nodes.
- **ctx<512** (below `MIN_CTX`) and **unsupported shape/device**: route does not fire → gqa fallback.

## 4. What changed (the flip)
`tinygrad/llm/model.py`, both gated to **gfx1100 / Qwen3-8B / B=1 / T=1 / Hq=32 / Hkv=8 / Hd=128 / ctx≥512**:
- route condition: `getenv("DECODE_ATTN_AMDGCN_TILE", 0)` → **`, 1)`** (default-on).
- `_init_state`: fp16 cache default-on **gated on the supported shape** (`_owned_supported`) — other models keep
  fp32. `DECODE_ATTN_AMDGCN_TILE=0` (or unsupported shape) → fp32 cache + gqa.
- env overrides preserved: `DECODE_ATTN_AMDGCN_TILE=0` disable, `DECODE_ATTN_AMDGCN_FP16CACHE`,
  `DECODE_ATTN_AMDGCN_S`, `DECODE_ATTN_AMDGCN_MIN_CTX`, `DECODE_ATTN_AMDGCN_COMBINE`.

## 5. Registry
`candidates.json` `decode_attention_llama_flash_tile_owned_amdgcn_b4`: **`default_on=true`**, `default_eligible=true`,
`default_flip` note with the confirmation numbers.

## 6. Decode tok/s picture (now default)
Default decode @ctx1024 moves **~74 → ~85 tok/s** (~76% → **~88% of llama.cpp** ~97 @ctx1024), and the gap closes
further at long context (+22% @ctx4096). Runtime-KV deferred (incremental); FO2 fp16 cache shipped.

## 7. Remaining blockers / follow-ons
None for the flip. Optional: runtime-KV residual-copy elimination (incremental, open append-NaN); cheaper/fused
split-KV combine; broaden the supported-shape guard (other head configs) if desired.

## 8. Artifacts and commands
- `bench/qk-owned-attention-default-flip/confirmation.json`.
- Confirm: `DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 QK_CKPTS=512,1024,2048,4096 PYTHONPATH=. .venv/bin/python extra/qk_decode_runtime_overhead.py` (default) vs `+ DECODE_ATTN_AMDGCN_TILE=0` (baseline).

## 9. Files changed
- `tinygrad/llm/model.py` (default flip: route default-on + fp16 cache default-on, both shape-gated).
- `bench/qk-decode-eval/candidates.json` (`default_on=true` + `default_flip`).
- `extra/qk_decode_runtime_overhead.py` (env-overridable ctx + flash threshold tracks `FLASH_DECODE_THRESHOLD`).
- New: this doc + `bench/qk-owned-attention-default-flip/confirmation.json`. Synthesis + session-handoff updated.

## 10. Working tree status
Default decode now uses the owned route (fp16 cache) on the validated shape; every other shape/device and
`DECODE_ATTN_AMDGCN_TILE=0` keep the byte-identical gqa+fp32 path. No 14B/32B; no runtime-KV; no new attention
tile; no codegen/renderer; no paged KV.
