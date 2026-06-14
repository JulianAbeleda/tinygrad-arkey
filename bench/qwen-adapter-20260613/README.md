# Qwen Adapter Artifacts

This directory contains the first Qwen adapter-training gates for the Track 1
practical loop.

## V0 Self-Distilled Plumbing Gate

The first adapter is intentionally narrow:

- model: Qwen3-8B Q4_K_M
- base inference path: generated QK policy with `QK_PRIMITIVE_STORAGE=shared`
- adapter: output-head LoRA only, rank `4`, alpha `8`
- training data: `bench/qwen-rollout-20260612/training-data-v1/sft.jsonl`

Artifacts:

- `8b-output-lora-r4/README.md`: adapter training summary and saved adapter.
- `8b-output-lora-r4/adapter.json`: adapter config.
- `8b-output-lora-r4/adapter.npz`: adapter tensors.
- `8b-output-lora-r4-rollout/summary.md`: 75-prompt rollout with adapter loaded.
- `compare-8b-base-vs-output-lora/report.md`: deterministic base-vs-adapter
  comparison.

Current result:

- adapter weights changed: L2 delta `0.003541`
- train/eval loss was already saturated on self-generated rows and stayed
  saturated
- adapter rollout quality: `75/75`
- base vs adapter: `0` regressions, `0/75` text changes, `0/75` token changes
- eval-loop speed ratio: `0.5925` vs base generated rollout

This proves adapter install, save/load, rollout, comparison, and contract
plumbing. It does not prove Qwen quality improvement. The training data is
self-distilled from the base model, so it has almost no supervised loss signal
for this output-head adapter.

## V2 Sentinel Behavior-Change Gate

The second adapter uses a non-self-generated supervised target:

- dataset: `training-data-v2/sft.jsonl`
- held-out eval prompts: `training-data-v2/eval-prompts.jsonl`
- target behavior: answer ordinary questions with exactly the one-token
  sentinel `OK`
- adapter: output-head LoRA only, rank `8`, alpha `16`

Artifacts:

- `training-data-v2/README.md`: generated SFT/eval dataset and exact-match
  held-out prompt set.
- `8b-sentinel-base-rollout/summary.md`: base model probe on held-out prompts.
- `8b-output-lora-r8-v2/README.md`: adapter training summary and saved adapter.
- `8b-output-lora-r8-v2-rollout/summary.md`: held-out rollout with adapter
  loaded.
- `compare-8b-base-vs-output-lora-r8-v2/report.md`: deterministic base-vs-v2
  comparison.

Current result:

- base held-out rollout: `0/12` exact-match (`<think>` first token)
- adapter held-out rollout: `12/12` exact-match (`OK`)
- compare improvement: `+12` passes, `0` regressions
- teacher-forced held-out accuracy: `0.0 -> 1.0`
- eval-loop speed ratio: `0.9685` vs the base sentinel rollout

This proves the adapter loop can produce a contract-gated behavior change on
held-out prompts. It is still a deliberately synthetic sentinel override, not a
general capability or preference-training result.

## V3 Strict JSON Answer Gate

The third gate replaces the sentinel with a human-authored strict JSON-answer
dataset:

- dataset: `training-data-v3/sft.jsonl`
- held-out eval prompts: `training-data-v3/eval-prompts.jsonl`
- target behavior: return only compact JSON with exactly one key, `answer`
- adapter: output-head LoRA only, rank `16`, alpha `16`, trained with EOS
  targets

Artifacts:

- `training-data-v3/README.md`: SFT/eval dataset and strict JSON schema.
- `8b-json-base-rollout/summary.md`: base model probe on held-out prompts.
- `8b-output-lora-r16-v3/README.md`: adapter training summary and saved
  adapter.
- `8b-output-lora-r16-v3-rollout/summary.md`: held-out rollout with adapter
  loaded.
- `compare-8b-json-base-vs-output-lora-r16-v3/report.md`: deterministic
  base-vs-v3 comparison.

Current result:

- base held-out rollout: `0/12`
- adapter held-out rollout: `3/12`
- compare improvement: `+3` passes, `0` regressions
- teacher-forced held-out token accuracy: `0.5000 -> 0.8542`
- eval-loop speed ratio: `0.3989` vs the base JSON rollout

Verdict: failed promotion. Output-only LoRA learned to suppress `<think>` and
emit JSON-shaped text, but it did not reliably produce the correct held-out
answer values and sometimes collapsed numeric answers into repeated `1`s. Do
not lower the gate or keep LR-tuning this path as a win. The next adapter
experiment should target more conditional capacity than the output head alone,
for example a small allowlisted set of FFN/attention projection adapters, using
the same strict JSON rollout/compare gate.

## V4 Internal Adapter Diagnostic

The fourth gate attempted that next step with allowlisted internal adapters.

Artifacts:

- `internal-adapter-v4-diagnostic/README.md`: target-policy and training-path
  diagnostic.

Current result:

- `lastN_ffn` target expansion works and fails loudly on invalid groups.
- Internal LoRA now preserves activation gradients; output-only LoRA keeps the
  original exact-preserving detached behavior.
- Generated-QK-path internal training fails on unsupported quant bit-op
  gradients (`Ops.OR`).
- Baseline training with `REALIZE=1` OOMs on 8B at `23.78 GB` used.
- Baseline/no-REALIZE can pass a one-step `last4_ffn` smoke, but full
  `last4_ffn` / `last1_ffn` runs are too slow through the plain-block path to
  be a practical gate.

Verdict: blocked, not promoted. The next step is not more target sweeps. It is
building a practical 8B internal-adapter training path that avoids generated QK
bit-op gradients, avoids full fp16 realization, and avoids the current
plain-block runtime cost.

## V5 Suffix-Cache Internal Adapter Gate

The fifth gate adds that dedicated internal-adapter training path for suffix
targets:

- trainer: `extra/llm_adapter_suffix_train.py`
- target: `last1_ffn` (`blk.35.ffn_gate`, `blk.35.ffn_up`, `blk.35.ffn_down`)
- adapter: LoRA rank `4`, alpha `8`
- training mode: baseline QK path, cached hidden state at block `35`, suffix-only
  backprop
- inference mode: generated QK policy with shared storage

Artifacts:

- `8b-last1-ffn-suffix-lora-r4-v5/README.md`: suffix training summary and saved
  adapter.
- `8b-last1-ffn-suffix-lora-r4-v5-rollout/summary.md`: held-out strict JSON
  rollout with adapter loaded.
- `compare-8b-json-base-vs-last1-ffn-suffix-lora-r4-v5/report.md`: base-vs-v5
  comparison.
- `compare-8b-output-lora-r16-v3-vs-last1-ffn-suffix-lora-r4-v5/report.md`:
  output-LoRA-vs-v5 comparison.

Current result:

- suffix parity: `pass`, `max_abs=0.0`
- cache entries: `48` train rows and `12` eval rows, expanded to `319` train and
  `80` eval token targets
- train loss: `7.1041 -> 0.2817`
- eval loss: `7.4458 -> 0.2680`
- teacher-forced eval token accuracy: `0.5000 -> 0.9167`
- base rollout comparison: `0/12 -> 4/12`, `+4` improvements, `0` regressions
- V3 output-LoRA comparison: `3/12 -> 4/12`, with `2` improvements and `1`
  regression. At `N=12`, this is not a meaningful generation-quality win over
  V3; treat both as the same failed `3-4/12` band.

Verdict: training-path positive, quality-gate negative. The suffix-cache trainer
solves the V4 practical blocker: internal `lastN_ffn` adapters can now train
without generated-QK bit-op gradients, without full fp16 realization, and without
recomputing the full prefix every step. It does not solve strict JSON answering.
The held-out rollout is only `4/12`, so this adapter is a diagnostic artifact,
not a promoted behavior gate. The large teacher-forced versus free-generation
gap (`0.9167` token accuracy versus `0.3333` strict generation pass rate)
points at exposure bias / objective mismatch more than adapter capacity. The
next practical gate should enlarge the held-out generation set and train on
filtered own-generations or another generation-matched objective before adding
more adapter targets.

## V4 Strict JSON Evaluation Dataset

The next gate is an eval/objective foundation, not another adapter-capacity
sweep:

- dataset: `training-data-v4/sft.jsonl`
- held-out eval prompts: `training-data-v4/eval-prompts.jsonl`
- held-out eval size: `204` prompts
- categories: arithmetic, fact, code, compiler, string, categorization
- scorer: deterministic multi-axis strict JSON scorer
- disjointness: train/eval prompts, answers, and template instances are
  mechanically checked

This replaces the previous `N=12` held-out set as the promotion ruler. Phase 3
should re-run base, V3, and V5 against this dataset before any more training is
interpreted as a quality result.

## V4 Re-Baseline Result

Artifacts:

- `8b-v4-json-base-rollout/summary.md`
- `8b-output-lora-r16-v3-v4-rollout/summary.md`
- `8b-last1-ffn-suffix-lora-r4-v5-v4-rollout/summary.md`
- `compare-8b-v4-json-base-vs-output-lora-r16-v3/report.md`
- `compare-8b-v4-json-base-vs-last1-ffn-suffix-lora-r4-v5/report.md`
- `compare-8b-v4-output-lora-r16-v3-vs-last1-ffn-suffix-lora-r4-v5/report.md`
- `v4-rebaseline-verdict.json`

Results on the `204`-prompt V4 held-out eval:

| model path | strict pass | Wilson 95% CI | parse/schema | value correct |
|---|---:|---:|---:|---:|
| base generated | `0/204` | `[0.000, 0.018]` | `0/204` | `0/204` |
| V3 output-LoRA | `69/204` | `[0.277, 0.406]` | `164/204` | `69/204` |
| V5 suffix-cache `last1_ffn` | `105/204` | `[0.446, 0.582]` | `190/204` | `107/204` |

V5 is a real generation-quality improvement over V3 on this larger gate:
`105/204` versus `69/204`, `+36` strict passes, `0` regressions, and
non-overlapping Wilson intervals. The old `4/12` versus `3/12` result was too
small to trust; this re-baseline shows the suffix-cache internal adapter did
matter.

The result is still not a solved strict-JSON behavior gate. The main residual
is content/value correctness, not JSON form: V5 reaches `190/204` parse/schema
but only `107/204` value-correct. Category deltas versus V3 are concentrated in
arithmetic (`+26`), code (`+7`), and categorization (`+3`); fact and string are
tied, and compiler remains `0/34`.

Verdict: promote V5 to current-best adapter for the next objective experiment,
not to production behavior. The next step should be Phase 4 rejection-sampling
SFT using the deterministic scorer as the filter. Do not add more adapter
capacity until that objective loop has been tested.

## Phase 4 Rejection-Sampling SFT Status

Phase 4 started with the required gold-data control:

- artifact: `8b-last1-ffn-suffix-lora-r4-v6-gold-v4/README.md`
- input: `training-data-v4/sft.jsonl`
- architecture: same as V5, `last1_ffn` suffix LoRA rank `4`, alpha `8`
- rows: `408` train, `204` eval
- status: `pass`
- teacher-forced eval accuracy: `0.6563 -> 0.9219`
- elapsed: `1720.8s`
- cache cost: `3.83GB` train hidden copies plus `1.97GB` eval hidden copies

The rejection-sampling generation step was attempted with V5 as the generator
and `K=4` temperatures `[0.0, 0.2, 0.5, 0.8]`, but the AMD runtime hit a
synchronize wait timeout before writing the RS artifact. A bounded `DEV=AMD`
smoke test also timed out afterward, so further GPU work is blocked until the
AMD runtime/device path is reset.

The sampler was hardened after the failure:

- `samples.jsonl` is now written incrementally;
- `--resume` skips completed sample IDs and appends missing attempts;
- downstream `accepted.jsonl`, `near-miss.jsonl`, `sft.jsonl`, and
  `summary.json` derive from the persisted samples file.

Next after AMD reset:

```bash
PYTHONPATH=. .venv/bin/python extra/llm_json_rejection_sample.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --policy bench/qk-shared-storage-20260612/8b/policy.json \
  --adapter bench/qwen-adapter-20260613/8b-last1-ffn-suffix-lora-r4-v5 \
  --input bench/qwen-adapter-20260613/training-data-v4/sft.jsonl \
  --out bench/qwen-adapter-20260613/training-data-v4-rs-v5-k4 \
  --device AMD --storage shared --prompt-format chat \
  --seed 20260614 --k 4 --temperatures 0.0 0.2 0.5 0.8 \
  --max-accepted-per-source 1 --resume
```

Only train the RS adapter if the resulting `summary.json` has enough accepted
rows and acceptable category coverage. If accepted rows are sparse or missing
hard categories entirely, increase `K` or stratify before training.
