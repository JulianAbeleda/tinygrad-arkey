# Decode context 4096 qualification

Ordinary generated Flash measures 4.691128 ms/token median, 4.579049 ms observed minimum, and 213.168 tok/s over two continuous three-token windows after excluded capture/warmup. Matched llama reports 4.915217 ms/token average (204.694 tok/s). Generated median is 0.224089 ms (4.56%) faster than llama's average, and its observed minimum is 0.336168 ms faster. The aggregates remain explicitly different: generated reports the median/minimum of continuous request windows while llama reports the average of five six-token samples.

The generated run uses the ordinary Flash route at depth 4096 with max-context 8192. Its selected `rollout_jit_flash_live[34]` contains 418 generated programs; the attached census preserves source hashes and renderer sources. Both measured windows held P0 at 2572 MHz with no hardware, thermal, or software throttle reason.

The harness does not record memory usage. External `nvidia-smi` samples observed 117 MiB before launch, 19,880 MiB during fill/capture, a high-water sample of 31,362 MiB out of 32,607 MiB at roughly 6:35 elapsed, and 117 MiB after process exit. These are observed samples rather than a continuously measured peak. The run completed without OOM with about 1,245 MiB remaining at the highest sample.
