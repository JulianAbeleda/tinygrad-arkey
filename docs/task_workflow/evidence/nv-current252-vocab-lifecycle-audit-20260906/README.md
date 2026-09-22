# Current252 vocabulary lifecycle audit

Run from the repository root:

```sh
bash extra/llm_research/prefill/run_current252_vocab_audit.sh docs/task_workflow/evidence/nv-current252-vocab-lifecycle-audit-20260906
.venv/bin/python extra/llm_research/prefill/compare_current252_vocab_audit.py docs/task_workflow/evidence/nv-current252-vocab-lifecycle-audit-20260906
```

The shell script serializes all three fresh processes under the GPU lock.
Controls explicitly disable the vocabulary lease; candidate enables it. The
original bracket used the script before default/profile modes were added;
projection runtime sources were unchanged at commit1421da46e. Later source
manifests pin the automatic-selector and profiling runs.

| Arm | Median ms | Output replay | Projection census |
|---|---:|---|---|
| Generic vocabulary A | 53.313233 | 20/20 exact | 252/252 |
| Existing generated vocabulary B | 50.746286 | 20/20 exact | 252/252 |
| Generic vocabulary C | 53.183219 | 20/20 exact | 252/252 |

Recovery: 2.501940 ms (4.6986%) versus mean controls. See `comparison.json`
for exact route symbols, full-logit comparisons, raw-input hashes and scope.
Both control comparisons pass allclose; max absolute difference0.00793672,
mean0.000704779. Each arm replays exactly within itself, not bit-exact between
different vocabulary implementations. Every token is198.

`candidate-default` verifies implicit selection inside the already opt-in
compiler stack: vocabulary override unset, median50.428258ms, one generated
vocabulary main, PASS. The ordinary llama stack is unaffected. Explicit
`NV_COMPILER_Q6_VOCAB_PP512=0` restores the generic compiler-tail path.

`deep-replay-abi-failure.log` preserves a pre-timing failure: the optional
internal-buffer checker assumes the old three-buffer main ABI. Completed arms
use the established20-cycle output/logit replay, not that internal checker.

GPU CSVs are process-boundary observations, not locked clocks. Profiling runs
are mapping evidence only and are excluded from endpoint comparisons.

Fresh llama reference was launched from a writable temporary directory:

```sh
flock -n /tmp/gpu-bench.lock /home/ubuntu/env/llama.cpp/build-cuda/bin/llama-bench -m /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf -p 512 -n 0 -b 2048 -ub 512 -ngl 99 -fa on -r 9 -o json
```

Build ac4cddeb0,
R9 sample0 retained but excluded as the first-use sample. Settled median of
samples1--8 is38.8192285ms; all samples are in `llama-fresh.json`. This is a
fresh sequential cross-harness reference, not a matched prompt/clock protocol.
