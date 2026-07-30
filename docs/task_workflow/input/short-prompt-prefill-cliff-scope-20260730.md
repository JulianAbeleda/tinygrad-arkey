# Short-prompt prefill cliff scope

Date: 2026-07-30

Status: scoped, not implemented. Branch boundary: tinygrad `exp`. Does not authorize promotion to `dev`/`master`.

Third scope of the 2026-07-30 campaign, after `target-capability-policy-decoupling-scope-20260730.md` (decode,
complete) and `prefill-codegen-recovery-scope-20260730.md` (prefill codegen, PR0/PR1 complete).

## 1. End goal

A prompt shorter than `PREFILL_UBATCH` (512) never reaches prefill-v2 and falls into what the code itself calls
"the fallback trap": symbolic 32-token chunks with no tensor cores. **A 256-token prompt costs 5.5x the wall
clock of a 512-token prompt while doing half the work.** Most real prompts are shorter than 512 tokens, so most
prompts pay this.

The goal is that prefill cost is monotonic in prompt length. No prompt should be cheaper to serve by padding it.

## 2. Pinned evidence

Apple M4 10-core / Metal, Qwen3-8B-Q4_K_M (`d98cdcbd...5785`), commit `329053647`, `METAL_HYBRID_REPLAY=1`,
one measured decode token after two warmups. Batched (JIT on), wall clock per whole run including model load.

| depth | wall | route | decode tok/s | s/token |
| ---: | ---: | --- | ---: | ---: |
| 128 | 98 s | sdpa | 16.85 | 0.77 |
| 256 | **232 s** | sdpa | 15.86 | **0.91** |
| 512 | **42 s** | flash | 17.15 | **0.08** |

Three facts:

1. **Non-monotonic in prompt length.** 512 tokens is 5.5x faster in absolute wall clock than 256 tokens, and 2.3x
   faster than 128 tokens, despite 2x and 4x the work.
2. **The slow path degrades with depth rather than amortising** — 0.77 -> 0.91 s/token from 128 to 256, both on
   the symbolic path. Attention is quadratic and compounds within the trap.
3. **Decode is unaffected** — 16.85 / 15.86 / 17.15 tok/s across all three. The tuned Q4_K/Q6_K kernels serve
   decode identically regardless of which prefill path ran. The entire difference is prefill.

Subtracting a nominal ~40 s model load, implied prefill throughput is roughly **2 tok/s on the slow path versus
~250 tok/s on the fast path**. Treat that ~100x loosely — one run per point, sensitive to load-time attribution.
The 5.5x absolute comparison between 256 and 512 requires no such adjustment.

### 2.1 Per-kernel corroboration

Unbatched `JIT=0 DEBUG=2` profiles on the same commit:

| depth | prefill launches | symbolic-`toks` share of prefill time |
| ---: | ---: | ---: |
| 128 | 11177 | **99%** |
| 512 | 5258 | **0%** |

Different kernel families entirely: `r_toks_*` (symbolic token dim) below the threshold, `r_16_256_8_*`
(concrete dims) at and above it. More than twice the launches doing a quarter of the work.

### 2.2 The selector

`tinygrad/llm/model.py`:

```text
:285   PREFILL_UBATCH = 512
:393   prefill_ubatch: int = 512          # candidate-local physical M
:1427  if self.config.prefill_v2 and (prompt_len - start_pos) >= ubatch:      -> prefill-v2, concrete T, TC
:1439  elif self.config.prefill_v2 and start_pos < prompt_len and prompt_len >= ubatch:   -> remainder rescue
:1457  else: v_toks.bind(min(chunk_size, ...))                                -> symbolic, 32-tok chunks, no TC
```

The `:1439` branch exists specifically to rescue a sub-ubatch **remainder** — its comment names the failure mode
— but it still requires `prompt_len >= ubatch`, so it cannot help a prompt that is short overall. Both
prefill-v2 branches fail and control reaches `:1457`.

**Not to be confused with the flash-decode threshold**, which is also 512 (`model.py:1422`) and is unrelated
code. Both thresholds firing at the same depth caused a misattribution earlier in this campaign; the flash route
affects decode only and was measured to have zero effect on prefill (61164.3 ms sdpa vs 61154.4 ms forced-flash
at depth 128, ratio 1.000x).

### 2.3 The complication

`PREFILL_UBATCH` is not only a loop bound. `model.py:928` documents the kernel signature as
`(frozenset({out_features, PREFILL_UBATCH}), in_features)` — it participates in kernel selection and in the
packed-WMMA warmstart table. Lowering the constant globally would change which kernels are looked up and could
miss warmstart entries entirely. A fix must not assume the constant is free to change.

## 3. Architectural boundaries

| Concern | Authority |
| --- | --- |
| prefill chunk/branch selection | `tinygrad/llm/model.py::generate` |
| prefill route admission | `tinygrad/llm/prefill_routes.py` |
| kernel signature / warmstart lookup | existing packed-WMMA warmstart owners |
| measurement | `extra/llm_research/decode/kernel_log_diff.py` |

Reuse the existing prefill-v2 machinery. Do not add a third prefill implementation; the goal is to make the
existing fast path reachable, not to write another one.

## 4. Evidence contract

1. **Correctness first.** Generated token identity must be unchanged **at each depth tested** (note the token
   differs between depths because the pinned prompt is a repeating cycle — compare like depth to like depth, not
   across depths), plus `prompt_evidence` sha256.
2. **Monotonicity.** Wall clock must be non-decreasing in prompt length across at least 128 / 256 / 512 / 1024.
   That property, not a single ratio, is the acceptance criterion.
3. Per-kernel before/after via the TG0 parser, reporting the symbolic-`toks` share at each depth.
4. 3 reps per depth with spread. Single-run figures in section 2 are indicative only and must be re-measured.
5. Decode non-regression: decode tok/s unchanged at 16.9-17.2 with tokens `13876`/`38835` at depth 128.

## 5. Work packages

### SP0 — Confirm and characterise

Prerequisite: none.

- Re-measure section 2 with 3 reps per depth and explicit model-load subtraction, so prefill cost is separated
  from load. The ~100x implied figure needs to become a measured one.
- Add 1024 and 2048 to establish that the fast path stays monotonic above the threshold.

Stop condition: if the cliff does not reproduce with repetition, stop and report — the single-run figures in
section 2 would then be an artifact.

### SP1 — Make a short prompt reach prefill-v2

Prerequisite: SP0.

The comment at `:1428` states prefill-v2 needs *"only the token dim must be concrete for tensor cores"* — 512 is
a tile-size choice, not a correctness requirement. Add a branch for `prompt_len < ubatch` that runs one
prefill-v2 chunk with concrete `T`.

Open design questions the packet must answer, not assume:

- Does a concrete `T` smaller than `PREFILL_UBATCH` find a warmstart entry, or does it compile fresh? Measure the
  first-call compile cost; a 5 s compile on a 2 s prefill is not a win.
- Should short prompts be **padded up** to a supported tile instead, trading wasted compute for an existing
  kernel? Padding 128 to 512 wastes 4x the FLOPs but the measurements suggest that is still ~2.3x faster than the
  trap. Quantify both.
- Does `2.3` above hold for the shortest prompts, where padding waste is largest?

### SP2 — Verify monotonicity and decode non-regression

Prerequisite: SP1. Full section 4 contract.

## 6. Non-goals

- Changing `PREFILL_UBATCH` globally (see 2.3).
- Rewriting prefill-v2 or adding a third prefill path.
- The devectorizer reduction-strategy work (`prefill-codegen-recovery-scope-20260730.md` PR2) — independent, and
  sequenced separately.
- Prefill attention re-tiling (that scope's PR3).
- Promotion to `dev`/`master`.

## 7. Known limitations

- **No AMD hardware.** This cliff was found on Metal; whether AMD hits it is unmeasured. AMD non-regression can
  only be shown structurally.
- Section 2 is one run per depth. SP0 exists to fix that before anything is built.
- `test/unit` carries ~114 pre-existing failures. Diff failing-test-id **sets**, not counts.
