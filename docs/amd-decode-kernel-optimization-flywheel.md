# AMD Decode Kernel Optimization Flywheel

Date: 2026-06-14

Status: consolidated hypothesis note. This ties together the kernel
optimization loop and the structured model/eval loop, but explicitly treats the
model-to-kernel closing link as unproven. It is not a new promotion gate by
itself.

Update (2026-06-14): the model-to-kernel link is now **falsified at the current
feature set** — the learned triage model adds no value over a cheap deterministic
rule once kernel outcomes are measured on device (not wall-clock) throughput. The
real, correctly-measured optimization lever is batching / a fused Q4_K GEMM. See
`docs/amd-decode-flywheel-postmortem.md` for the honest arc; the deterministic
end-to-end inference wins in `docs/amd-decode-current-verdicts.md` are a separate
result and stand.

Proof plan: `docs/amd-decode-flywheel-proof-plan.md`.

Superseding note (2026-06-23): the "model-to-kernel closing link" is reframed. The learned model's job is **not** to
judge kernel speed or triage kernels — that direction stays falsified (see the 2026-06-14 note above). Its job is to be
a **primitive-space proposer**: emit a bounded search spec (lane / primitive / hypothesis / knobs / required-evidence /
stop-rules) that the **deterministic** machine-search runner expands and the harness / ISA / correctness / W==D gates
decide. LoRA/SFT first; RLVR deferred until schema + reward + shadow-mode utility are proven. The closing link is thus
"learned proposal under deterministic gates," not "learned kernel triage." See
`docs/primitive-space-learning-loop-lora-first-result-20260623.md` and
`docs/oracle-guided-gpu-primitive-explorer-result-20260623.md`.

## Purpose

There are two separate loops in this repo:

1. The kernel optimization loop, which directly improves tinygrad decode speed.
2. The structured model/eval loop, which makes model outputs reliable enough to
   help organize, triage, and generate compiler-search work.

One direction is real today: faster kernels make rollouts, evals, and
rejection-sampling cheaper, which produces more structured artifacts. The
reverse direction is only a hypothesis: a better structured model might improve
kernel candidate triage or proposal quality. That closing link must be tested
before this should be called a true flywheel.

Kernel speed is decided only by correctness, microbench, and full-decode gates.
The model can propose and classify; it is not the performance judge.

## Kernel Loop

The ideal kernel loop is:

```text
profile decode
-> freeze one hot real tensor shape
-> build a semantic descriptor
-> propose one mechanism
-> static gate
-> compile/source/disassembly gate
-> numeric correctness gate
-> repeated device-time microbench
-> full decode A/B
-> promote or record rejection
```

The loop starts from real decode, not from abstract kernels. For the current AMD
decode work, QK GEMV remains the active bottleneck. A candidate should name the
exact role and shape it targets, for example Q4_K `ffn_gate` on a specific
Qwen3 tensor.

Each candidate should change one real mechanism at a time:

- `parts` / `LOCAL` policy selection
- direct output versus partial output
- row grouping or row-axis mapping
- packed-weight load shape
- lane-to-weight mapping
- block-local load/decode/dot semantic lowering
- renderer or assembly-quality memory-access lowering

The candidate is rejected early if it violates the quant layout, hides too much
of the GEMV body, needs unsupported UOps, or cannot preserve scheduler-visible
row/K/split axes.

## Promotion Gates

Promotion is intentionally conservative:

- Static gates catch invalid shapes, unsupported layouts, and wrong scope.
- Compile gates prove the intended kernel exists, including source or
  disassembly evidence when the mechanism is about load width or instruction
  shape.
- Correctness gates compare numeric output against a reference on the exact
  target tensor.
- Microbench gates use repeated device-time measurements against the current
  best implementation.
- Full decode gates install the candidate into the generated policy and run the
  actual model path with correctness A/B and token-speed measurement.

A microbench win is only a raw accept. It is not a promoted optimization until a
matching full-decode confirmation also accepts. Negative results are artifacts,
not waste: they stop repeated dead branches.

## Structured Model Loop

The model/eval loop is separate:

```text
strict JSON dataset
-> rollout
-> deterministic scorer
-> rejection-sampling filter
-> adapter training
-> structured outputs
-> possible compiler triage/proposals
```

This loop improves whether the model can emit usable structured records, for
example:

```json
{"answer":"qk_wide_load"}
```

or later:

```json
{"mechanism":"packed_load","role":"ffn_gate","expected_evidence":"global_load_b128"}
```

The scorer checks form and value. It does not decide whether a kernel is fast.
That remains the job of the kernel gates.

## Proven One-Way Benefit

The easy half already works:

```text
faster kernels
-> cheaper rollouts, rejection sampling, and evals
-> more structured artifacts
-> better compiler training data
```

This is valuable even if the reverse direction never works. It makes the model
loop cheaper and makes the project history more machine-readable.

## Unproven Closing Link

The hard half is load-bearing and currently unproven:

```text
kernel experiment artifacts
-> structured accept/reject examples
-> model learns repo-specific compiler vocabulary and failure modes
-> better candidate triage or proposal quality
-> fewer wasted kernel experiments
```

This is not established by the current artifacts. The model has shown compiler
weakness on the original V4 compiler slice: V5 had zero compiler accepts under
rejection sampling, and V6 gold-control reached only `14/34` compiler eval
passes on the original row-specific task. The V4.1 stable-key fix proves the
model can copy and learn simpler compiler labels; it does not prove kernel
reasoning.

The current high-quality kernel direction changes also came from human analysis
grounded in profiles, rooflines, prior art, and generated-source inspection.
Structured history can prevent repeated dead branches, but it has not yet shown
that it can generate the next mechanism.

For example, a rejected kernel experiment can become a training row:

```json
{
  "mechanism": "wide_load_only",
  "result": "rejected",
  "reason": "microbench_regression",
  "evidence": "vector_load -8.58% versus v1"
}
```

This is why stable compiler vocabulary matters. The Phase 4.2 V4.1 compiler
data changed brittle row-specific answers such as `train_qk_gemv_005` into
stable concept keys such as `qk_gemv`, making compiler examples usable for
rejection-sampling SFT.

## Closing-Link Test

Do not take the reverse link on faith. The cheap test is a historical
triage/ranking benchmark over existing kernel artifacts.

The phased plan for this proof track is
`docs/amd-decode-flywheel-proof-plan.md`.

Input:

- profile or descriptor context
- a candidate list or candidate summary
- generated-source / disassembly evidence when available
- prior verdict labels hidden from the model

Tasks:

- Predict accept, reject, tie, needs-rerun, or construction-blocked.
- Rank candidates by expected value before microbench/full-decode evidence.
- Identify which rejected mechanism should not be retried.
- Optionally propose the next mechanism, but score proposal novelty separately
  from triage accuracy.

Baselines:

- reject-all / majority-class baseline
- random ranking
- simple hand heuristic based on mechanism and previous family result
- human-selected next step when that decision is available

Scoring:

- held-out accuracy or macro-F1 for verdict prediction
- precision@k / NDCG for ranking useful candidates
- false-positive rate on known dead branches
- calibration: whether high-confidence accepts actually clear deterministic
  gates

Holdout:

- Prefer time-split or family-split holdout, not random row split. Random split
  would overfit the repo's repeated near-duplicate artifacts.

Pass condition:

- The model must beat reject-all/random and a simple heuristic on held-out
  kernel families. If it does not, the two loops remain separate: faster kernels
  help the model loop, but the model loop is not a kernel accelerant.

## Current Repo Mapping

Kernel loop source of truth:

- `docs/amd-decode-current-verdicts.md`
- `docs/amd-decode-ansor-direction.md`
- `docs/amd-decode-bandwidth-roofline.md`
- `docs/amd-decode-packed-load-lowering.md`
- `docs/amd-decode-packed-qk-semantic-op.md`
- `bench/qk-ansor-transition-20260612/`

Kernel loop tools:

- `extra/qk_gap_profile.py`
- `extra/qk_semantic_descriptor.py`
- `extra/qk_descriptor_policy.py`
- `extra/qk_candidate_generator.py`
- `extra/qk_candidate_static_gate.py`
- `extra/qk_ansor_transition_loop.py`
- `extra/qk_semantic_schedule_bench.py`
- `extra/qk_block_dot_microbench.py`
- `extra/qk_threeway_load_microbench.py`

Structured model/eval loop source of truth:

- `docs/qwen-json-eval-objective-scope.md`
- `bench/qwen-adapter-20260613/README.md`
- `structure/Development/session-handoff.md`

Structured model/eval tools:

- `extra/llm_rollout.py`
- `extra/llm_rollout_compare.py`
- `extra/llm_json_scorer.py`
- `extra/llm_json_rejection_sample.py`
- `extra/llm_json_rs_coverage_gate.py`
- `extra/llm_adapter_suffix_train.py`
- `extra/llm_adapter_json_data_v4_1_compiler.py`

## Current State

Kernel side:

- Generated QK policy with shared storage is the current accepted local
  inference path for 8B, 14B, and 32B.
- Descriptor-level `parts` / `LOCAL` search is exhausted.
- Semantic schedule/codegen v0/v1/v2/v3/v4 surfaces are rejected or blocked.
- Raw wide-load-only paths are rejected by device-time gates.
- The remaining kernel direction is diagnostic or lower-level: instruction
  mix, occupancy, memory transactions, or renderer-quality packed QK lowering.

Model/eval side:

- V5 suffix-cache internal adapter is the current best non-gold behavior
  artifact on the V4 strict JSON gate.
- V6 gold-control proves the suffix adapter/objective setup can improve
  free-generation behavior when data quality is available.
- Original V4 V5 rejection sampling failed compiler coverage.
- V4.1 compiler stable-key data fixes that slice: V5 reaches `30/34` on the
  V4.1 compiler eval, and V5 RS selects `68/68` compiler train rows.

## Next Practical Bridge

The next practical bridge between the loops is not to let the model rewrite
kernels directly. It is to keep collecting structured compiler artifacts and
train the model to handle the repo's actual compiler vocabulary and verdicts.

Concrete next step:

1. Build a combined RS-SFT artifact that keeps usable non-compiler V4 RS rows
   and replaces the compiler slice with V4.1 stable-key compiler rows.
2. Train a V7 candidate with the existing suffix-cache architecture.
3. Evaluate V7 on both the original V4 strict JSON gate and the V4.1 compiler
   gate.
4. Run the closing-link triage/ranking benchmark before treating the model as a
   kernel accelerant.
5. Use it only for structured proposal/triage assistance unless deterministic
   kernel gates accept the resulting candidates.

If the closing-link benchmark fails, the strategy should be stated plainly:
there are two useful loops with a one-way benefit, not a compounding flywheel.
The kernel loop then stands on its own correctness/performance merits, and the
model loop is a structured-output capability rather than a kernel optimization
accelerant.

If the benchmark passes, the long-term target is a machine-assisted kernel
search loop where every proposal, rejection, and promotion is structured enough
to become future training data.
