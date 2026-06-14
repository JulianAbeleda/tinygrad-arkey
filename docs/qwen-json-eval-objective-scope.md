# Qwen Strict JSON Eval And Objective Scope

Date: 2026-06-14

Status: research-backed plan of record for the next adapter/eval loop.

## Purpose

The V5 suffix-cache adapter fixed the practical internal-adapter training
blocker, but did not solve strict JSON generation. The current strict JSON eval
is too small (`N=12`) and too coarse (single exact-match pass/fail) to guide the
next decision. This document grounds the next course of action in existing eval
and post-training practice instead of inventing a bespoke objective from
scratch.

The immediate goal is not another adapter-capacity sweep. The goal is a
generation-based eval and scorer that can gate decisions, then reuse the same
scorer as the filter/reward for rejection-sampling SFT.

## Current Evidence

Committed artifacts in `bench/qwen-adapter-20260613/` establish:

- Base generated strict JSON rollout: `0/12`.
- V3 output-head LoRA: teacher-forced token accuracy improves to `0.8542`, but
  generation reaches only `3/12`.
- V5 suffix-cache `last1_ffn` LoRA: teacher-forced token accuracy improves to
  `0.9167`, but generation reaches only `4/12`.
- V5 versus V3 is one net prompt at `N=12` with one regression, so it is not a
  meaningful generation-quality result.

The useful signal is the teacher-forced versus free-generation gap. V5 fits the
supervised targets under teacher forcing, but exact generated answers still fail
on most prompts. That points to objective/eval mismatch and exposure bias more
than to adapter capacity.

## Research

### Eval Harness Shape

Inspect AI is the best design reference for the harness abstraction. Inspect
defines evaluations around composable tasks, datasets, solvers, and scorers, and
the official task docs describe tasks as the integration point for datasets,
solvers, and scorers. Inspect also has local model-provider support, but not a
native tinygrad model interface today, so the right move is to mirror the shape
inside the existing harness rather than pay the integration tax first.

Sources:

- Inspect AI overview and component model: https://inspect.aisi.org.uk/
- Inspect task docs: https://inspect.aisi.org.uk/tasks.html
- Practical walkthrough / composition note: https://hamel.dev/notes/llm/evals/inspect.html

### Deterministic, Verifiable Scoring

IFEval is the key philosophical reference: evaluate instruction-following with
automatically verifiable constraints, avoiding slow human evaluation and
avoiding biased LLM-as-judge scoring when a programmatic check exists. Our JSON
task is exactly this class: parseability, schema shape, no extra prose, and
known answer value are all verifiable.

Sources:

- IFEval paper: https://arxiv.org/abs/2311.07911
- lm-evaluation-harness IFEval task README:
  https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/ifeval/README.md
- Inspect-evals IFEval implementation:
  https://ukgovernmentbeis.github.io/inspect_evals/evals/reasoning/ifeval/

### Structured JSON Metrics

JSONSchemaBench and structured-output benchmarks show that "valid JSON" is not
enough. Structured-output evaluation should separate syntax/schema compliance
from value correctness and task accuracy. The SOB paper is especially relevant:
it explicitly reports schema compliance and value accuracy separately, and
highlights that structured outputs can be structurally valid while semantically
wrong.

Use these as metric-design references, not as dependencies. The tinygrad eval
only needs a small deterministic scorer now.

Sources:

- JSONSchemaBench repo:
  https://github.com/guidance-ai/jsonschemabench
- JSONSchemaBench paper:
  https://arxiv.org/html/2501.10868v1
- Structured Output Benchmark:
  https://arxiv.org/abs/2604.25359
- STED / consistency scoring for structured outputs:
  https://arxiv.org/abs/2512.23712
- StructEval:
  https://tiger-ai-lab.github.io/StructEval/

### Objective: Rejection-Sampling SFT Before RL

STaR is the closest objective reference for the next loop: generate candidate
outputs, keep the ones that satisfy the verifier, fine-tune on the successful
model-produced traces, and repeat. For this project, the "rationale" part is
not needed initially; the useful pattern is generate K strict-JSON completions,
score them deterministically, keep passers, and train the adapter on accepted
own-generations.

TRL is the right SFT loss/API reference, especially prompt-completion data,
completion-only loss, and PEFT adapter training. It is not a drop-in because
our training path is tinygrad-specific and suffix-cache based.

verl and OpenRLHF are the right references for later RLVR/GRPO/PPO-scale work,
but they are too heavy for the immediate consumer-AMD loop. The next step should
be rejection-sampling SFT; RLVR can come only after the deterministic scorer and
larger eval are stable.

Sources:

- STaR: https://arxiv.org/abs/2203.14465
- TRL docs: https://huggingface.co/docs/trl/index
- TRL SFTTrainer: https://huggingface.co/docs/trl/en/sft_trainer
- verl: https://github.com/verl-project/verl
- OpenRLHF docs: https://openrlhf.readthedocs.io/

### Structured Decoding Is A Later Generation Lever

Constrained decoding can make JSON form cheaper or free by masking invalid
tokens during generation. XGrammar is the strongest reference here, and vLLM now
supports structured outputs with `xgrammar` or `guidance` backends. This matters
for later because it can separate form failures from value failures.

Do not use constrained decoding to replace the eval. The eval must first measure
unconstrained generation, then later quantify how much constrained decoding
helps form and whether it hurts value accuracy.

Sources:

- XGrammar engineering post:
  https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar
- XGrammar paper:
  https://arxiv.org/pdf/2411.15100
- vLLM structured outputs:
  https://docs.vllm.ai/en/latest/features/structured_outputs/

### Why Not LLM-As-Judge First

LLM-as-judge is useful for subjective tasks, but this task is programmatically
verifiable. Using a judge model here adds cost, latency, nondeterminism, and
possible bias without improving the core signal. Keep LLM-as-judge out of the
promotion gate. It can be used later only for optional dataset authoring review
or semantic answer equivalence when deterministic normalization is insufficient.

Sources:

- Evidently LLM-as-judge guide:
  https://www.evidentlyai.com/llm-guide/llm-as-a-judge
- Limitations of LLM-as-a-judge without human grounding:
  https://arxiv.org/html/2503.05061v1

### Statistical Gate

The current `N=12` eval cannot support one-example comparisons. Future
generation gates should report pass-rate confidence intervals. Use Wilson score
intervals for binomial pass rates; this is standard for proportions and behaves
better than the naive normal approximation, especially for small or skewed
samples.

Source:

- NIST proportion confidence intervals / Wilson method:
  https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm

## Thesis

The next bottleneck is not adapter installation, suffix training, or
teacher-forced fit. Those are solved enough for the strict JSON task. The
remaining failure is that the current objective trains on ground-truth prefixes
while deployment samples from the model's own prefixes, and the current eval is
too small and too coarse to separate form, content, and noise.

Therefore:

1. Promotion must be based on free-generation accuracy, never teacher-forced
   token accuracy.
2. The eval must be large enough and statistically reported enough to distinguish
   real changes from one-prompt noise.
3. The scorer must split form from content:
   - `parse_valid`: output parses as JSON.
   - `schema_ok`: exactly the expected shape; for the first task, one key
     `answer`, no extra keys, no prose.
   - `value_correct`: normalized answer equals the expected value.
   - `strict_pass`: all required axes pass.
4. The same scorer should become the rejection-sampling SFT filter and later the
   RLVR reward. Eval-first is not a detour; it is the prerequisite objective.
5. Adapter-capacity experiments (`last2_ffn`, attention projections,
   output+suffix) should wait until this eval/objective loop exists. Otherwise
   they will optimize the same teacher-forced proxy that already misled V3/V5.

## Architecture

Implement an Inspect-shaped local harness layer inside `extra/`, not a hard
Inspect dependency:

- `Dataset`: JSONL prompt rows with ground-truth answer metadata and category.
- `Solver`: existing tinygrad generated rollout path, with optional adapter and
  optional future constrained-decoding mode.
- `Scorer`: deterministic multi-axis JSON scorer.
- `Report`: per-sample JSONL, aggregate JSON/Markdown, Wilson confidence
  intervals, and compare artifacts.

The current `extra/llm_rollout.py` already has most of the solver/report
plumbing. The new work should centralize strict JSON scoring and dataset
generation rather than fork a parallel system.

## Dataset Scope

Create `training-data-v4` for strict JSON evaluation:

- Held-out generation eval: target `~200` prompts.
- Train split: separate prompts used only for SFT/rejection-sampling data.
- Content-disjointness: no identical question, answer, or template instance may
  appear in both train and held-out eval.
- Balanced categories:
  - arithmetic;
  - facts;
  - code identifiers;
  - compiler/GPU/tinygrad facts already known locally;
  - string/pattern manipulation;
  - controlled categorization / boolean answers.
- Each row must include:
  - `id`;
  - `prompt`;
  - `expected_json`;
  - normalized answer value;
  - category tags;
  - split;
  - generation max tokens.

Avoid broad open-ended semantic answers in V4. They force judge-like scoring and
make the first gate less deterministic. If semantic equivalence is needed later,
add it as a separate axis with STED or a bounded synonym map, not as the initial
promotion criterion.

## Scorer Contract

For each generated text:

1. `parse_valid`: `json.loads(text.strip())` succeeds.
2. `no_extra_text`: stripped output round-trips to JSON only. No markdown,
   `<think>`, duplicate JSON blocks, or trailing prose.
3. `schema_ok`: parsed object is a dict with exactly one key, `answer`, unless
   a row declares a different schema.
4. `type_ok`: value type matches row expectation.
5. `value_correct`: normalized parsed value equals normalized expected value.
6. `strict_pass`: all above axes pass.

Normalize values conservatively:

- strings: strip whitespace, case-fold only when row marks `case_insensitive`;
- numbers: compare exact integer value or decimal string normalized by the row
  type;
- booleans: exact boolean;
- lists/objects: defer until a later schema; do not add in V4 unless needed.

Reports must aggregate every axis independently, plus `strict_pass`.

## Statistical Gate

For every rollout summary:

- report `n`, `passed`, `pass_rate`, and Wilson 95% CI for each axis;
- compare candidates with:
  - absolute pass-rate delta;
  - CI overlap / non-overlap;
  - regression count;
  - category deltas.

Pre-register promotion for V4:

- candidate strict pass-rate lower Wilson bound must exceed base upper Wilson
  bound;
- candidate strict pass-rate lower Wilson bound must exceed V3/V5 upper Wilson
  bound if claiming improvement over prior adapters;
- zero high-severity regressions in parse/schema axes relative to the previous
  accepted adapter, unless the candidate's strict-pass CI improvement is large
  enough to justify review;
- teacher-forced token accuracy may be reported, but cannot promote a result.

## Course Of Action

### Phase 1: Build The Eval Ruler

Deliverables:

- `extra/llm_json_scorer.py` with deterministic multi-axis scoring and Wilson
  CI helpers.
- Tests for parse failures, schema failures, extra text, type mismatch,
  normalized value match, and Wilson interval bounds.
- Extend or wrap `extra/llm_rollout.py` so strict JSON rows record all score
  axes, not only pass/fail.

Stop rule:

- Do not run more adapter training until this scorer is tested and produces
  stable report artifacts on existing V3/V5 rollouts.

### Phase 2: Author V4 Dataset

Deliverables:

- `bench/qwen-adapter-20260613/training-data-v4/`.
- `sft.jsonl` and `eval-prompts.jsonl` with `~200` held-out eval prompts.
- Dataset summary reporting category balance and train/eval disjointness checks.
- Tests for duplicate IDs, leaked answers/templates, malformed expected JSON,
  and scorer compatibility.

Stop rule:

- If disjointness cannot be mechanically checked, do not use the dataset as a
  promotion gate.

### Phase 3: Re-Baseline Existing Models

Run the new eval on:

- base generated Qwen3-8B;
- V3 output-LoRA;
- V5 suffix-cache `last1_ffn` LoRA.

Deliverables:

- rollout artifacts for all three;
- compare report with axis-level pass rates and Wilson CIs;
- updated `bench/qwen-adapter-20260613/README.md` verdict.

Decision:

- If V3/V5 are statistically indistinguishable from base, adapter capacity is
  not yet justified.
- If V5 is materially better on value but not form, consider constrained
  decoding as a separate form lever.
- If V5 is materially better on form but not value, use rejection-sampling SFT
  before adding more capacity.

### Phase 4: Rejection-Sampling SFT

Use the scorer as the filter:

1. For each training prompt, generate `K` completions with the current best
   adapter and controlled seeds/temperatures.
2. Score each completion.
3. Keep strict passes, plus optionally keep near-miss rows in separate diagnostic
   buckets.
4. Train suffix-cache adapter on accepted own-generations.
5. Re-run Phase 3 eval.

Initial defaults:

- `K=4` or `K=8`;
- deterministic scorer only;
- no LLM judge;
- suffix-cache trainer first, same adapter target as V5 for comparability;
- only then test output+suffix or `last2_ffn` if the objective loop shows a real
  generation gain.

### Phase 5: Structured Decoding A/B

Do not implement this first. Once Phase 3/4 exist, add an optional generation
mode inspired by XGrammar/vLLM:

- constrained JSON form only;
- same value scorer;
- report form-axis improvements separately from value-axis changes;
- compare unconstrained versus constrained speed and pass rate.

This tells us whether strict JSON failure is mostly form, value, or both.

## Non-Goals

- Do not integrate full Inspect AI before the local tinygrad scorer exists.
- Do not switch to TRL/verl/OpenRLHF for this consumer-AMD loop.
- Do not use LLM-as-judge as the primary promotion gate.
- Do not train another adapter-capacity sweep using `N=12` evals.
- Do not promote on teacher-forced loss or token accuracy.
- Do not make constrained decoding the default before measuring unconstrained
  behavior.

## Acceptance Criteria For This Plan

The next implementation phase is complete when:

- the strict JSON scorer is tested and reusable;
- the held-out eval has at least `~200` mechanically checked prompts;
- base, V3, and V5 are re-run through the same generation gate;
- reports include per-axis pass rates and Wilson CIs;
- the next training decision is based on generation pass-rate evidence, not
  teacher-forced fit.

Only after that should the project choose between rejection-sampling SFT,
structured decoding, or more adapter capacity.
