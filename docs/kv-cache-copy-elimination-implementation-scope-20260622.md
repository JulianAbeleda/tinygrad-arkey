# KV-Cache Copy Elimination — Implementation Scope / Claude Prompt (2026-06-22)

## Mission

Implement and measure the bounded 8B decode lever identified by the decode-gap exhaustion audit:

**Eliminate the full-`max_context` KV-cache rematerialization copy in `tinygrad/llm/model.py:952`.**

This is the current `NEXT_IMPL_NORM_ROPE_KV` item. The name is historical from the coarse bucket, but the
actual target is not RMSNorm, RoPE, or FFN activation. The target is the full-buffer `cache_kv` copy caused
by taking `.after()` on the entire KV cache after storing only the new decode slice.

The work must start as a tractability probe, then graduate to a default-off route only if it is correct,
JIT-stable, and transfers to whole-decode W==D.

## Corrected Remaining-Gap Scope

This implementation scope depends on the corrected 8B decode-gap audit. Do not treat this as an isolated
`model.py` cleanup. It is the next step because the full post-`Q4K_GEMV_WARP` remaining-gap table was re-audited
with rendered kernel sources / AST fingerprints and the prior name-based buckets were corrected.

| Previously believed gap | New finding | Decision for this task |
|---|---|---|
| FFN activation is 10-20x slower | Misbucketed. `silu` is already fused into the FFN gate/up GEMV. The kernels labeled as activation (`E_49152`, `E_1536`) are mostly full KV-cache copy/rematerialization. | Reclassify as KV-cache copy tax. Do **not** build an activation kernel. |
| Norm/rope/small ops are large | Mostly mislabeled KV projection / q8 quant. Genuine RMSNorm/qk-norm is near parity or faster than llama. | Close the norm/rope lane. Do **not** build RMSNorm/RoPE kernels. |
| Attention is largest raw gap | Real ctx-growing gap, but bounded Route B/B5 attention levers already saturated below the W==D promotion gate; deeper fused-flash is codegen-level, not this bounded task. | Do **not** reopen attention unless a separate codegen-level fused-flash route is explicitly scoped. |
| KV-cache copy | Newly identified transferable O(MAXC) redundant copy, about `1.4-1.5 ms/token`, hidden inside the old activation/small-op buckets. | Highest-priority next lane. Implement a tractability probe for copy elimination. |

The phrase `NEXT_IMPL_NORM_ROPE_KV` therefore means: **act on the KV-cache lifecycle/copy tax that was hidden in
the old norm/rope/activation buckets**, not that norm or RoPE themselves are slow.

### Required lane outcomes

By the end of this task, the result doc must restate the status of all four lanes above:

- **FFN activation:** closed as mapping artifact unless the implementation disproves the audit.
- **Norm/rope:** closed as near-parity unless the implementation disproves the audit.
- **Attention:** not reopened; cite B5 saturation / bounded-lever exhaustion.
- **KV-cache copy:** probed and classified with one of the allowed verdicts.

If the KV-cache probe fails, do **not** automatically fall back to attention, norm/rope, or activation. The next
step would be a new decision doc, not opportunistic scope expansion.

## Required Reading Before Editing

Read these in order:

1. `docs/decode-gap-audit-consolidated-20260622.md`
   - One-page authority for the corrected 8B remaining-gap picture.
   - Key verdict: after `Q4K_GEMV_WARP`, weight-GEMV is closed; FFN activation and norm/rope were misbucketed;
     the actionable remaining bounded lever is KV-cache rematerialization copy elimination.

2. `docs/8b-exhaustion-next-implementation-decision-20260622.md`
   - The implementation decision doc.
   - It names the failing layer, expected transfer, risk, and required tractability gate.

3. `docs/ffn-activation-gap-audit-result-20260622.md`
   - Explains why the "FFN activation" gap is a mapping artifact.
   - `silu` is fused into gate/up GEMV; `E_49152` / `E_1536` are pure buffer copies.

4. `docs/small-ops-time-tax-sub-audit-result-20260622.md`
   - Explains why "norm/rope/small ops" is mostly mislabeled.
   - Genuine norm/rope is near parity; do not scope a norm/rope kernel.

5. `docs/attention-tail-after-b5-audit-result-20260622.md`
   - Explains why attention is not reopened here.
   - B5 cheaper combine proved attention's bounded lever saturates below promotion gate.

6. `docs/decode-ffn-gemv-warp-result-20260622.md`
   - Current default-off, lossless W==D winner.
   - Use `Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1` as the current 8B "post-warp" baseline when comparing to llama
     and when measuring this new lever.

7. `docs/q4k-gemv-warp-promotion-hardening-result-20260622.md`
   - Promotion hardening state for the warp route.
   - Confirms the weight-GEMV lane is default-eligible and no further same-lever expansion should distract this task.

8. `structure/Development/performance-primitive-research-principles.md`
   - Follow the local-A/B -> W==D -> correctness -> artifact discipline.
   - Do not claim success from isolated kernel changes or name-based buckets.

9. `structure/Development/session-handoff.md`
   - Current project state and defaults.

Required artifacts:

- `bench/qk-decode-kernel-probe/latest.json`
- `bench/qk-ffn-activation-gap-audit/latest.json`
- `bench/qk-small-ops-time-tax-audit/latest.json`
- `bench/qk-attention-tail-after-b5-audit/latest.json`
- `bench/qk-tinygrad-vs-llama-time-tax/latest.json`

## Current Failing Layer

The relevant code is in `tinygrad/llm/model.py`, inside `TransformerBlock._attention`:

```python
# NOTE: we don't want to change self.cache_kv, the function API doesn't support this well
assigned_kv = Tensor(self.cache_kv.uop.after(self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).uop)))
k = assigned_kv[0, :, :, 0:start_pos+T, :]
v = assigned_kv[1, :, :, 0:start_pos+T, :]

#self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
#k = self.cache_kv[0, :, :, 0:start_pos+T, :]
#v = self.cache_kv[1, :, :, 0:start_pos+T, :]
```

The store targets only `[start_pos:start_pos+T]`, but `.after()` is taken on the **full** `cache_kv`. The
rendered-kernel audit shows this materializes/copies the whole static KV buffer every decode step:

- `E_49152` at MAXC 4608, about `1.4 ms/token`.
- Shrinking MAXC to 1152 shrinks the copy to `E_12288`, about `0.375 ms/token`.
- The wall transfer test showed about `+1.5 ms` / `+8 tok/s`.

This is O(MAXC) where a decode append should be O(T) with T=1.

## Non-Goals

- Do not implement FFN activation fusion. It is already fused.
- Do not implement RMSNorm/RoPE kernels. Genuine norm/rope is at parity.
- Do not reopen attention Route B. B5 showed bounded attention saturates below the promotion gate.
- Do not move to 14B/32B.
- Do not change defaults.
- Do not broaden into generic JIT mutation semantics unless the tractability probe proves a tiny local change is impossible.

## Required Phase Plan

### Phase 0 — Baseline Reproduction

Confirm the checked-in audit state before editing:

- `git status --short`
- Confirm current commit includes `a9d29014a` and `a25bcd7eb`.
- Run or inspect the existing artifacts to confirm:
  - `E_49152` / `E_1536` are present under the post-warp baseline.
  - `E_49152` is a pure copy and MAXC-bound.
  - Current post-warp W==D baseline is about `74 tok/s @ctx1024` with `Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1`.

Write a tiny pre-edit note in the result doc draft with:

- commit hash,
- GPU/arch,
- model path,
- enabled env flags,
- baseline tok/s at ctx512/1024/2048/4096,
- copy kernel names and microseconds.

### Phase 1 — Tractability Probe Variants

Try the smallest local route first, behind a default-off environment flag. Recommended flag:

- `KV_CACHE_INPLACE=1`

Probe variants, in this order:

1. **Re-enable slice `.assign()`**
   - Use the already-commented intended form:
     ```python
     self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
     k = self.cache_kv[0, :, :, 0:start_pos+T, :]
     v = self.cache_kv[1, :, :, 0:start_pos+T, :]
     ```
   - Gate this only for decode first: `B == 1`, `T == 1`, symbolic `start_pos`, Qwen3-8B shape if needed.
   - Keep fallback to current `.after()` path on exception or unsupported shape.

2. **Slice-scoped `.after()`**
   - If direct `.assign()` violates JIT/function purity, try to take `.after()` on the needed slice/range instead of
     the full `cache_kv`.
   - The intended shape is: write the new slice, then expose only `0:start_pos+T` to attention without materializing
     the whole MAXC buffer.

3. **Minimal helper boundary**
   - If the above needs an isolated helper to preserve trace semantics, keep it local to `model.py` or a small
     `extra/` probe. Do not change core tinygrad unless the blocker is explicitly proven and documented.

For every variant, report one of:

- `KV_CACHE_INPLACE_TRACTABLE`
- `KV_CACHE_SLICE_AFTER_TRACTABLE`
- `KV_CACHE_JIT_PURITY_BLOCKED`
- `KV_CACHE_CORRECTNESS_BLOCKED`
- `KV_CACHE_NO_WD_TRANSFER`

### Phase 2 — Correctness Gate

Correctness must pass before timing claims:

- Greedy byte-identical tokens vs baseline for a deterministic prompt.
- At least 40 decode steps at ctx1024.
- Prefer also a 64-token natural-prompt generation, matching the Q4K warp hardening quality style.
- No stale cache leakage across repeated generation calls.
- No divergence between eager/capture/replay if a small standalone probe is possible.

If tokens differ, stop and classify `KV_CACHE_CORRECTNESS_BLOCKED`.

### Phase 3 — Kernel Identity / Copy Shrink Gate

Use rendered kernel identity, not kernel names alone.

Required checks:

- The full-MAXC copy kernel (`E_49152` for MAXC 4608) disappears or shrinks to O(T) / O(`start_pos+T`) bounded behavior.
- No replacement copy of equivalent bytes appears under a different name.
- If a smaller copy remains, quantify:
  - bytes moved,
  - µs/token,
  - whether it scales with MAXC or actual ctx.
- Preserve existing `Q4K_GEMV_WARP` kernels and attention route behavior.

Update or add a probe artifact under:

- `bench/qk-kv-cache-copy-elimination/latest.json`

Minimum fields:

```json
{
  "verdict": "...",
  "commit": "...",
  "model": "...",
  "gpu": "...",
  "env": {},
  "baseline": {
    "tok_s": {},
    "copy_kernels": []
  },
  "candidate": {
    "tok_s": {},
    "copy_kernels": [],
    "tokens_match": true,
    "jit_stable": true
  },
  "delta": {
    "tok_s_pct": {},
    "copy_us_saved": {},
    "wall_ms_saved": {}
  }
}
```

### Phase 4 — W==D Transfer Gate

Run in-process or otherwise tightly controlled A/B, with `.item()` inside the timed window. Use the current post-warp
baseline as the comparator:

```bash
DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 PYTHONPATH=. .venv/bin/python <new_or_existing_wd_harness>
DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 KV_CACHE_INPLACE=1 PYTHONPATH=. .venv/bin/python <new_or_existing_wd_harness>
```

Required ctx points:

- 512
- 1024
- 2048
- 4096

Promotion threshold for this probe:

- `>= +5% @ctx1024`
- no regression at ctx512/2048/4096 beyond measurement noise
- tokens match
- spread tight enough to trust the result, ideally `<0.5%` per repeated run

Expected rough signal if successful:

- `~+1.2-1.5 ms/token` wall recovery.
- Around `+8 tok/s` near the 68-74 tok/s band, depending on whether measuring default or post-warp baseline.

If the copy shrinks but W==D does not move, classify `KV_CACHE_NO_WD_TRANSFER` and explain the overlap/critical-path
reason. Do not keep optimizing blindly.

### Phase 5 — Integration Decision

If all gates pass:

- Keep route default-off unless explicitly asked to flip defaults.
- Register a candidate in `bench/qk-decode-eval/candidates.json`.
- Candidate name suggestion:
  - `kv_cache_inplace_append_decode_8b`
- Mark:
  - `default_eligible=true` only if byte-identical and guard/fallback safe.
  - `default_on=false`.
- Add binding metadata if the registry requires it.

If gates fail:

- Do not leave a half-enabled route.
- Write the exact blocker and preserve the audit artifact.

## Required Result Doc

Write:

- `docs/kv-cache-copy-elimination-result-20260622.md`

Required sections:

1. Verdict.
2. What changed.
3. Why the original copy existed.
4. Correctness result.
5. Kernel identity / copy shrink result.
6. W==D result table.
7. Default decision.
8. Corrected remaining-gap table: FFN activation, norm/rope, attention, KV-cache copy.
9. Remaining 8B gap after this result.
10. Artifacts and commands.

Use one of these final verdicts:

- `KV_CACHE_COPY_ELIMINATION_WD_PASS`
- `KV_CACHE_COPY_ELIMINATION_LOCAL_PASS_WD_FAIL`
- `KV_CACHE_COPY_ELIMINATION_JIT_BLOCKED`
- `KV_CACHE_COPY_ELIMINATION_CORRECTNESS_FAIL`
- `KV_CACHE_COPY_ELIMINATION_NOOP`

## Acceptance Checklist

- [ ] Read all required docs.
- [ ] Baseline reproduced from current post-warp state.
- [ ] Implementation is env-gated and default-off.
- [ ] Correctness is byte-identical.
- [ ] JIT capture/replay is stable.
- [ ] Rendered-kernel audit proves the full-MAXC copy disappears or shrinks.
- [ ] W==D measured at ctx512/1024/2048/4096.
- [ ] Result doc written.
- [ ] Artifact JSON written.
- [ ] README/session handoff updated only if the verdict changes the project state.
- [ ] No 14B/32B work.
- [ ] No attention/norm/activation scope creep.
- [ ] Result doc preserves the corrected-gap table and explicitly states why the other lanes remain closed.
- [ ] Working tree status reported.

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch `qk-prefill-flag-leak-resolution`.

Read `docs/kv-cache-copy-elimination-implementation-scope-20260622.md` completely, then execute it exactly.

The goal is not to optimize FFN activation, norm/rope, or attention. The goal is to probe and, if tractable,
implement a default-off elimination of the full-`max_context` KV-cache rematerialization copy at
`tinygrad/llm/model.py:952`.

Use the post-`Q4K_GEMV_WARP` route as the baseline:

```bash
DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1
```

Start with the smallest local change behind `KV_CACHE_INPLACE=1`. First try the commented `.assign()` path;
if JIT/function purity blocks that, try a slice-scoped `.after()` that avoids taking `.after()` on the full
`cache_kv`. Keep fallback to the current path on unsupported shapes or exceptions.

Success requires byte-identical greedy output, stable JIT capture/replay, proof that the `E_49152` full-MAXC
copy disappears or shrinks to bounded behavior, and W==D `>= +5% @ctx1024` with no ctx regression. Write
`bench/qk-kv-cache-copy-elimination/latest.json` and
`docs/kv-cache-copy-elimination-result-20260622.md`.

Stop and classify honestly if blocked:

- `KV_CACHE_COPY_ELIMINATION_JIT_BLOCKED`
- `KV_CACHE_COPY_ELIMINATION_CORRECTNESS_FAIL`
- `KV_CACHE_COPY_ELIMINATION_LOCAL_PASS_WD_FAIL`
- `KV_CACHE_COPY_ELIMINATION_NOOP`

Do not change defaults. Do not move to 14B/32B. Do not reopen attention. Do not build norm/rope or activation
kernels. Preserve the corrected-gap table in the result narrative so the KV-cache copy is not mistaken for an
activation/norm/attention task later. Report the final verdict, commands run, artifacts written, source files
changed, default status, and working tree status.
