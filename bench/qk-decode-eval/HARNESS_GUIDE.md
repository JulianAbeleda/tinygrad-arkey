# Decode Harness Best Practices

This guide is the operational baseline for new decode performance harnesses in this fork. It turns the project rule
"harnesses are performance primitives too" into a checklist a new candidate can follow.

Authority:

- `structure/Development/performance-primitive-research-principles.md`, especially "Harnesses Are Performance
  Primitives Too" and "Machine Search Is Generate, Evaluate, Prune, And Remember".
- `extra/qk_harness_contract.py`, the local helper that stamps provenance, comparator metadata, timing authority,
  reproducibility bands, and the 13-field contract audit.
- `extra/qk_decode_eval.py`, the live decode evaluator and promotion/refutation authority.

## External Baseline

These references explain why the local rules exist:

- **MLPerf Inference**: benchmark results need representative workloads, comparable scenarios, and quality targets,
  not just speed numbers. See Reddi et al., *MLPerf Inference Benchmark*:
  https://arxiv.org/abs/1911.02549.
- **SPEC RG reproducible evaluation methodology**: complex performance results need preserved artifacts,
  environment details, and enough methodology to reproduce the result. See the SPEC Research Group report:
  https://research.spec.org/fileadmin/user_upload/documents/rg_cloud/endorsed_publications/SPEC_RG_2019_Methodological_Principles_for_Reproducible_Performance_Evaluation_in_Cloud_Computing.pdf.
- **Google Benchmark**: warmup, repeated measurement, and reporting controls are first-class benchmark behavior.
  See the user guide:
  https://google.github.io/benchmark/user_guide.html.
- **AMD rocprofv3 / thread trace** and **NVIDIA Nsight Compute**: profiling tools separate kernel/runtime
  attribution from end-to-end promotion. See:
  https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html,
  https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.1.1/how-to/using-thread-trace.html, and
  https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html.
- **FlashAttention**: attention performance is an IO/dataflow property, so the harness must preserve the memory
  hierarchy and lifecycle being claimed. See Dao et al., *FlashAttention*:
  https://arxiv.org/abs/2205.14135.
- **vLLM / PagedAttention**: serving throughput depends on KV-cache lifecycle and request scheduling, so local
  kernels and in-model routes are different primitives. See Kwon et al., *PagedAttention*:
  https://arxiv.org/abs/2309.06180.

## Measurement Authority

Do not mix these layers.

| authority | use | promotion? |
|---|---|---|
| clean W==D, `PROFILE=0`, auto clock | whole-decode route and tok/s promotion authority | yes |
| clock-pinned local A/B | standalone diagnostic gate against current winner | no |
| PROFILE / GPU timestamps | attribution only | no |
| DEBUG / stdout timing | debugging only | no |
| raw-dispatch throughput | diagnostic only unless the real route is raw dispatch | no |

Promotion is reported, never applied by a harness. A harness may recommend `PASS_PROMOTE`, but model defaults and
policy changes are owner decisions.

## New Harness Checklist

1. Register live candidates in `bench/qk-decode-eval/candidates.json`.
2. Compare against the current winner, normally `gqa_coop_vec`, and state why that comparator is current.
3. Run correctness or quality before speed. For approximate fp reorderings, state the tolerance and why it is valid.
4. Separate local A/B from W==D. Local pass can unlock W==D; it cannot promote by itself.
5. Use repeated timings and record a spread/noise band. A bare median is not enough.
6. Stamp the artifact with `extra.qk_harness_contract.stamp()`.
7. Validate the emitted artifact through `decode_eval` when it is a lifecycle candidate.
8. Link the artifact to the ledger/refutation row or state why no ledger write is expected.
9. Stop on the first failed gate. Do not tune blindly after a local or W==D miss.
10. Do not reopen closed lanes without new evidence and an explicit scope.

## Required Artifact Contract

A performance-claiming artifact must capture:

1. workload shape and context;
2. candidate id and primitive class;
3. comparator id and why it is the current winner;
4. exact command and env;
5. git commit and dirty status;
6. hardware and clock/perf state;
7. warmup and compile handling;
8. repeats, median, spread, and noise/reproducibility band;
9. correctness or quality gate;
10. local diagnostic timing vs in-model W==D authority;
11. pass/fail threshold;
12. final verdict and stop reason;
13. ledger/refutation links.

Use:

```python
from extra.qk_harness_contract import repro_band, stamp

band = repro_band(samples_us)
artifact = {
  "candidate_id": "my_candidate",
  "family": "attention",
  "ctx_fixed": 1024,
  "warmups": 8,
  "repro_band": {"1024": band},
  "correctness_rel_rmse": 5e-4,
  "first_gate_pass": False,
  "threshold": {"local_speedup_vs_comparator": 1.05},
  "verdict": "FAIL_LOCAL_AB",
  "stop_reason": "local A/B missed the 1.05x gate",
}
artifact = stamp(
  artifact,
  comparator_id="gqa_coop_vec",
  comparator_why="shipped default decode-attention primitive and current local A/B winner",
  timing_authority="clock-pinned local throughput proxy; diagnostic only, not W==D promotion authority",
  ledger_links=["bench/qk-lifecycle-search/refutations.json#my_candidate"],
)
```

## Verdict Discipline

Use the `Verdict` enum in `extra/qk_modes.py`; do not invent string verdicts in a child harness. If a harness needs a
new verdict, add it to the enum, schema, search policy, and evaluator contract in the same change. (NOTE: the enforcing
test `test/unit/test_verdict_ssot.py` was removed in a cleanup and is not currently present; restoring it is scoped in
`docs/qk-consolidate-r1-config-code-decoupling-scope-20260630.md`, Phase 2.)

Use loop-level prune decisions for policy or closed-lane filtering. Do not encode loop pruning as decode-eval verdicts.

## Reproducibility Bands

Default local A/B guidance:

- at least 5 measured samples for a local diagnostic gate;
- report median, min, max, mean, spread percentage, and MAD;
- a claimed local win must exceed the threshold and the observed spread;
- a W==D promotion claim must exceed the promotion threshold and the whole-decode reproducibility band.

Near-threshold movement is learning, not promotion. Record it as `LOCAL_PASS_WD_FAIL`, `FAIL_LOCAL_AB`, or `REST` as
appropriate.

## What Not To Do

- Do not promote from PROFILE-on timing, DEBUG logs, stdout-only scripts, or raw-dispatch timing.
- Do not compare against stale baselines when the current winner is known.
- Do not hide clock pinning inside a harness without restoring state.
- Do not let a local proxy timing become the headline tok/s.
- Do not emit a performance artifact without git/dirty state.
- Do not tune a failed candidate repeatedly outside the lifecycle loop.
- Do not leave a new live harness as a one-off probe if its result will influence project decisions.

## Cleanup Policy

Historical probes are allowed to remain as provenance only when they are not registered as live evaluator candidates.
If a historical probe becomes live again, first bring it under this contract or expect `decode_eval` to flag it through
`child_artifact_contract`.

The preferred path for new work is:

```text
template/candidate spec
-> structural and policy prune
-> correctness or quality gate
-> local A/B diagnostic gate
-> W==D promotion gate if local passed
-> machine-readable verdict artifact
-> ledger/refutation memory
```
