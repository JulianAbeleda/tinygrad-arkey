# NV prefill restoration audit — 2026-08-28

## Verdict

The historical Qwen3-8B pp512 near-parity result is **not installed or
reproducible on the current tree**. It must not be used as a current tinygrad
prefill claim.

## Fresh same-session measurements

| implementation | pp512 | status |
| --- | ---: | --- |
| tinygrad current installed path | 433 tok/s (1182.6 ms/chunk) | fallback; candidate census failed |
| llama.cpp `ac4cddeb0`, CUDA, FA on, R5 | 14,074.3 ± 1,175.6 tok/s | valid same-session comparator |

The tinygrad number is diagnostic, not a promoted performance authority: the
authority harness correctly aborted before writing its JSON because all four
promoted candidate identities were missing from the execution census.

## What was found

1. The compact promoted artifact is present and expands to the expected four
   sm_120 roles and identities.
2. The research admission boundary used two spellings for the same target:
   the installed artifact says `NV:sm_120`, while the capability table only
   declared `CUDA:sm120`. Native-spelling capability rows now admit all four
   roles; the focused runtime suites pass (89 tests).
3. Admission is not execution. The loaded model owns the registry, fp16
   overlay, exact route attachments, and graph bindings, but a real TinyJit
   prefill records zero candidate executions. A direct non-JIT graph build
   records all four, localizing the remaining break to the JIT/lazy-lowering
   lifecycle.
4. Retaining candidate compiler contexts through the outer TinyJit scope was
   tested and did not improve wall time; that attempted production edit was
   removed.
5. The older campaign already records that the 44–46 ms fused pp512 rows are
   unreproducible and retained as history, not current authority
   (`docs/task_workflow/input/b3-prefill-host-overhead-scope-20260803.md`).

## Next gate

Trace one pp512 capture at the lazy scheduler boundary and prove, per each of
the four exact shapes, whether the candidate context reaches postrange and
whether the final NV kernel contains the intended tensor-core/LDS schedule.
Only after the execution census is nonzero and the generated kernels are
attributed should whole-prefill correctness and the 512/1024/2048/4096 sweep
resume.
