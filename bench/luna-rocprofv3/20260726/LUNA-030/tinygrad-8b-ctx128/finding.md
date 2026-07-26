# LUNA-030 tinygrad 8B ctx128 profiler/control finding

Verdict: `TOOL_FAILURE`.

One fresh subprocess was launched under `/tmp/gpu-bench.lock` using
`/opt/rocm/bin/rocprofv3 --kernel-trace --output-directory <absolute-path>/rocprof --`.
It produced only the initial tinygrad model-admission line in `stdout.log` and
rocprofv3 process instrumentation in `stderr.log`. It produced neither a
`rocprof/` dispatch artifact nor the fixed-depth authority JSON requested by
`--out`.

The required in-worker positive control was `route=flash` with a positive
captured program count. It is absent, as are route tile/combine and lowered-UOp
records. No interpretation of profiler emptiness as absent tinygrad execution,
route identity, performance, or correctness is permitted.

No second subprocess was run. LUNA-031 cannot proceed from this evidence:
repair the worker completion/control path first, then launch a new isolated row.
