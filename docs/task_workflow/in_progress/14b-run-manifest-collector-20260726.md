# Decode run-manifest collector

Task: `LUNA-003`  
Verdict: `PASS`

Reusable collector: `extra/qk/decode/run_manifest.py`.

The module is import-pure. `collect_run_manifest(...)` gathers the Section 16 fields only when called; its CLI writes JSON only after an explicit invocation. It records only environment variables with route/compiler/backend prefixes: `TINYGRAD_`, `AMD_`, `HIP_`, `HSA_`, `ROCR_`, `ROCPROF`, `GGML_`, and `LLAMA_`.

The collector rejects a missing model, required schema fields, empty command argv, missing branch/commit/worktree facts, and an empty `positive_control` mapping. It captures dirty paths via porcelain status, boot ID, model byte identity, and `rocm-smi` power/clock snapshots when available. A failed power query is recorded as unavailable rather than silently omitted.

Focused unit coverage is in `test/unit/test_run_manifest.py`:

- known command argv, branch, dirty paths, and resolved worktree;
- boot-ID fixture;
- explicit before/after power fixture;
- rejection of missing positive controls.

No GPU command, model load, benchmark, profiler, or trace was run by this task.
