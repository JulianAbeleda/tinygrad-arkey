# Decode Attention + Elementwise — Final Report

Date: 2026-06-20

Executor: Claude

Scope: `docs/decode-attention-elementwise-solution-scope-20260620.md`

Outcome: **minimum-useful-success** per the scope. Both costs are split with timed, clock-pinned attribution and
the exact build targets are pinned. **All cheap (env-only) candidates were measured and rejected** — the default
flash-decode path is already tuned, and the `.contiguous()` removal does not touch the target. The real wins are
**custom-kernel fusions** (specced below), deferred as bounded multi-day builds, not cheap candidates. **No decode
default behavior changed.**

## 1. Current W/D baseline (clock-pinned, W = authority; host-sync 0% → GPU-bound)

| mode | ctx512 | ctx1024 | ctx2048 | ctx4096 |
|---|---:|---:|---:|---:|
| baseline | 68.0 | 66.5 | 63.5 | 60.8 tok/s |
| q8 opt-in | 72.8 | ~71.0 | — | 64.5 tok/s |
| llama.cpp | 98.6 | 97.6 | 95.4 | 92.2 tok/s |

## 2. Attention cost split (Deliverable 1 — `PASS`, 100% classified)

`docs/decode-attention-cost-split-result-20260620.md`. Dominant bucket = **`reduce_fixup`** (flash-decode
partial-reduction/fixup); `softmax_stats` grows fastest with ctx; `partial_compute` (QK·V) is small/flat.

| ctx | attention ms (%wall) | reduce_fixup | softmax_stats | partial_compute |
|---:|---:|---:|---:|---:|
| 512 | 3.22 (22%) | 1.66 | 0.79 | 0.77 |
| 1024 | 3.51 (23%) | **1.79** | 0.94 | 0.78 |
| 2048 | 4.17 (27%) | 2.08 | 1.26 | 0.83 |
| 4096 | 5.18 (32%) | **2.43** | 1.85 | 0.90 |

reduce_fixup + softmax_stats = 2.73 ms = 78% of attention @1024. The cost and its ctx-slope are owned by the
flash-decode **reduction / online-softmax machinery, not the compute**.

## 3. Elementwise cost split (Deliverable 3 — `PASS`, 99.2% classified)

`docs/decode-elementwise-cost-split-result-20260620.md`. Dominant family = **`E_49152_32_3` = FFN `silu(gate)*up`
= 1.24 ms/token** (56% of elementwise, flat across ctx, present in both modes). It is launch-overhead-bound
(~33 µs/call for ~0.15 µs of HBM work). Remainder is small: residual_add 0.32, rope 0.28, casts_copies 0.34,
unclassified 0.018 ms.

## 4. Candidate A/B results

| candidate | lever | result | gate |
|---|---|---|---|
| Attention C1 | `FLASH_L` 128→256/512 | **regress** (partial_compute parallelism collapses; ctx4096 53.9 vs 60.8 tok/s) | FAIL |
| Attention C2 | `FLASH_DECODE=0` (SDPA) | **catastrophic** (10.4 tok/s @1024 vs 66.5; batch-1 SDPA ~1% occupancy) | FAIL |
| FFN C1 | remove `.silu().contiguous()` (`no_contig`) | **no win** — `E_49152` unchanged 1.22 vs 1.24 ms (custom GEMV forces materialization) | FAIL |

Details: `docs/decode-attention-candidate-ab-result-20260620.md`,
`docs/decode-ffn-activation-fusion-result-20260620.md`. The default flash-decode policy (`gqa_coop_vec`,
`FLASH_L=128`, threshold 512) is already the tuned optimum.

## 5. Full W==D timing for candidates

No candidate cleared its local gate, so none was promoted to a full W==D ctx512/1024/4096 build. The W==D wall
authority (clock-pinned) was used for every A/B above; all regressed or were neutral.

## 6. Correctness / dNLL / greedy status

No default changed → decode greedy output is byte-identical to baseline. The deferred real candidates must be
gated on exact-greedy/dNLL within the existing decode policy (flash-decode is already exact-vs-SDPA up to fp
reassociation; the FFN fusion must preserve `silu(gate)*up` numerics).

## 7. Stacked result — PROJECTION (no candidate built; not measured)

Since no candidate passed cheaply, stacking is projected from the timed splits, not measured:

| action | recoverable ms/tok (target) | realistic recovery | projected ctx1024 tok/s |
|---|---:|---:|---:|
| Attention reduce/stat fusion (8→~3-4 kernels) | 2.73 | ~1.3-1.9 | ~73-77 |
| FFN activation fusion (eliminate `E_49152`) | 1.24 | ~1.0-1.2 | ~73 |
| stacked | ~3.4 | ~2.3-3.0 | **~83-88** (approaching llama 97.6) |

These remain projections; each requires its custom-kernel build + same-process A/B + W==D promotion before any
tok/s is claimed.

## 8. Lanes dropped / closed (unchanged from Deliverable 0; confirmed by these splits)

Q6 `ffn_down`/`lm_head`, full MMVQ family, Q4 `ffn_gate/up`, q8 lifecycle, host/persistent runtime, prefill —
all dropped/closed. Weight-GEMV is at/above llama parity in-model (Del 0). Nothing here reopens them.

## 9. Build recommendations (ranked, deferred custom-kernel work)

1. **Attention reduction / online-softmax-stat fusion** (largest, ctx-growing): collapse the ~8 tinygrad
   flash-decode stat/reduce kernels (`flash_max/den/prob/gmax/combine` + `r_*`) toward llama's ~3, in
   `extra/qk_flash_decode.py`'s `gqa_coop_vec` generators. Target ~2.7 ms @1024.
2. **FFN activation fusion** (flat ~1.24 ms): fuse `silu(gate)*up` into the gate/up producer epilogue
   (`extra/q4_k_gemv_primitive.py`) or the `ffn_down` GEMV prologue (`extra/q6_k_gemv_primitive.py`), eliminating
   the `E_49152` launches.

Both are bounded (~1-2 days each) custom-kernel-generator builds — beyond this session's cheap-candidate scope.
The decode gap is structurally the "many tiny kernels vs llama's fused" problem (1074 vs ~260 progs/token);
fusion is the lever.

## 10. Exact commands

```bash
# Deliverable 1
PYTHONPATH=. python3 extra/qk_decode_attention_cost_split.py --modes baseline,q8 --ckpts 512 1024 2048 4096 \
  --nmeas 20 --warmups 8 --out bench/qk-decode-attention-elementwise/attention_cost_split.json
# Deliverable 3
PYTHONPATH=. python3 extra/qk_decode_elementwise_cost_split.py --modes baseline,q8 --ckpts 512 1024 4096 \
  --nmeas 20 --warmups 8 --out bench/qk-decode-attention-elementwise/elementwise_cost_split.json
# Candidate A/Bs
FLASH_L=512 PYTHONPATH=. python3 extra/qk_decode_attention_cost_split.py --modes baseline --ckpts 1024 4096 ...
FLASH_DECODE=0 PYTHONPATH=. python3 extra/qk_decode_attention_cost_split.py --modes baseline --ckpts 512 1024 2048 4096 ...
FFN_ACT_VARIANT=no_contig PYTHONPATH=. python3 extra/qk_decode_elementwise_cost_split.py --mode baseline --ckpts 512 1024 ...
```

## 11. Artifacts and whether default changed

Tools: `extra/qk_decode_attention_cost_split.py`, `extra/qk_decode_elementwise_cost_split.py`,
`extra/qk_clock_pin.py` (reusable peak-clock pin). Data:
`bench/qk-decode-attention-elementwise/{attention,elementwise}_cost_split.json`. Docs: the four result docs above
+ this report.

**Default decode behavior NOT changed.** The one core edit (a default-off `FFN_ACT_VARIANT` test flag) was
reverted; `tinygrad/llm/model.py` is unmodified. GPU perf-state pinned only for measurement windows and restored
to `auto` (verified).

## Success-criteria assessment

- **Minimum useful success: ACHIEVED** — the exact attention bucket (`reduce_fixup` + `softmax_stats`, flash
  reduction/online-softmax machinery) and the exact elementwise family (`E_49152` FFN `silu(gate)*up`) to build
  next are identified, with timed evidence and the precise kernel-generator files.
- **Strong success: NOT achieved cheaply** — no env-only candidate recovered ≥0.5 ms / ≥3%; the real wins require
  custom-kernel fusion (correctly deferred rather than force a low-confidence build, per the scope's
  "no implementation if not enough evidence").
