# LUNA-022 and LUNA-023: llama bounded depth traces

Verdicts: `LUNA-022 PASS`; `LUNA-023 PASS`.

Each context used a fresh locked process and the accepted profiler form: `/opt/rocm/bin/rocprofv3 --kernel-trace --output-directory <absolute path> --output-format json -- llama-bench`. No runtime tracing, statistics, or output-config flags were used.

The ctx512 workload used `-p 0 -n 128 -d 512 -b 512 -ub 512 -fa 1 -ngl 99 -r 1 -o json`. Its nonempty child JSON reports ROCm on the RX 7900 XTX and 57.965381 token/s; its 50,017,288-byte raw trace contains 110,983 dispatches.

The ctx4096 workload differs only by `-d 4096`. Its nonempty child JSON reports 55.961050 token/s; its 53,546,825-byte raw trace contains 118,844 dispatches. Both captures record boot ID, commit, PATH, argv, and unset `GGML_HIP_FORCE_MMQ`/`GGML_HIP_USE_FLASH_ATTN` controls in their manifests.

Kernel metadata names are identical between the two captures. Dispatch behavior is not identical: ctx4096 adds 7,861 dispatches, raises Q4 `mul_mat_q` M128 dispatches from 238 to 1,904, raises `flash_attn_ext_f16` from 40 to 320, and increases `flash_attn_tile` aggregate device duration from 52,876,938 ns to 163,202,374 ns at the same observed `32x16x40` grid and `32x2x1` block. The normalized difference record is `bench/14b-decode-ctx128-depth-decay-20260726/llama/ctx512-ctx4096-dispatch-comparison.json`.

Ledgers retain raw names, dispatch counts, sample grid/block dimensions, total duration, and maximum duration. No timeout, signal, `pkill`, `kill -9`, or interruption was used. No tinygrad workload was run.
