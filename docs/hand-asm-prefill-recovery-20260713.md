# Hand-ASM tinygrad pp512 recovery

Date: 2026-07-13

Hardware: AMD gfx1100 / RX 7900 XTX

Model: Qwen3-8B-Q4_K_M

Workload: synced whole-model prefill, 512 input tokens at `start_pos=0`

## Definition and verdict

"Hand-ASM tinygrad" means the external handwritten RDNA3 WMMA path emitted by
`build_gemm_lds2`, not the generated-pure scheduler route. The historical result
was real: commit `9de41983aca92246e558118435c2fac8868e707c` reproduced **3394.1
tok/s** with zero relative RMSE. The current tree again reaches the 3.4k class after
restoring lazy LM-head evaluation. The GEMM instruction stream did not regress;
the whole-model graph around it did.

## Frozen kernel identity

The recovered `PREFILL_GEMM_PROFILE=hand_asm_lds2` profile freezes the validated
all-LDS2 schedule:

- `M=512`, `BK=32`, `PAD=16`, `DBUF=0`, `PLRA=1`, `PLRAB=0`, relocation off;
- 2x2 waves, 4x4 WMMA subtiles, 128x128 output tile, 128 threads;
- attn_qo at `(N,K)=(4096,4096)` emits 751 instructions / 4512 bytes;
- serialized instruction-byte SHA256:
  `4a576be126ff08aad7e2df56514098684dd3f9adf7068792adece7127fd1c739`.

The historical and current builders emit byte-for-byte identical attn_qo
streams under this schedule. This route is deliberately labeled an external
handwritten rollback/reference surface, not generated-pure tinygrad.

## Regression audit

An automated commit bisect used the same model, harness, hand-ASM schedule, and
a joint gate of `tok/s >= 3200` plus `rel_rmse <= 0.01`. Some unrelated broken
intermediate commits were skipped. The first bad commit was:

`c85ac30e1f421e61f193faa8e25ea1020cc92b66`

`[qk][prefill] route lm_head through prefill-v2 direct-packed path`

| Revision | Result | Correctness |
|---|---:|---:|
| parent `6959479833` | approximately 3538-3548 tok/s | pass |
| first bad `c85ac30e1` | approximately 2443 tok/s | pass |

The commit changed `logits()` from lazy `self.output(x)` to an eager
`_pf16(self.output, x).contiguous()` during prefill. Inference consumes only
`logits[:, -1, :]`; before the change, tinygrad could push that slice through the
graph and evaluate the 151936-wide vocabulary projection for one row. The eager
contiguous forced the full `512 x 151936` LM-head output first. This was a large,
unnecessary calculation—not a slowdown or correctness defect in the hand-ASM
GEMMs.

The packed Q6_K alternative is also not a fix: its measured full-M LM head took
about 104 ms versus about 17 ms for resident fp16. Both full-M routes lose to
preserving lazy final-token pruning for the inference workload.

## Recovery

LM-head policy is now centralized:

- `PREFILL_LM_HEAD_ROUTE=lazy` is the default and preserves final-token pruning;
- `resident_fp16` explicitly requests a full-sequence fp16 LM head;
- `direct_packed` explicitly requests the full-sequence packed experiment;
- legacy `PREFILL_LM_HEAD_DIRECT=1` remains an alias for `direct_packed`.

The historical hand path is selected without reconstructing six low-level flags:

```sh
PREFILL_V2=1 \
PREFILL_GRAPH_GEMM=1 \
PREFILL_GEMM_PROFILE=hand_asm_lds2
```

## Evidence

- Historical commit `9de41983a`: 3394.1 tok/s, `rel_rmse=0`.
- Historical full-logit A/B: identical SHA256 across five TinyJit runs,
  `fbdf14a5c261e9e753092d3048bd4386435cce2e38d529a4e1c034e78f4787ef`.
- Current named profile: three deterministic prompts have exact greedy-token
  parity against graph-GEMM-off; both isolated children and post-run GPU health
  probes pass. See
  `bench/prefill-whole-synced/hand-asm-lds2-recovered-quality-20260713.json`.
- Current pinned authority: **3354.16 tok/s** / **152.6462 ms**, with 0.122%
  sample CV, complete authority metadata, and a passing route-binding gate. See
  `bench/prefill-whole-synced/hand-asm-lds2-recovered-20260713.json`.
- Controlled installed llama.cpp build `ac4cddeb0 (9592)`, pp512, ten repeats:
  `3138.61 +/- 87.87 tok/s`. The recovered hand-ASM path remains ahead in the
  matched prefill-only comparison.

The previously discussed ~4.4k number is not used as recovery authority: it came
from an under-dispatched raw pipe geometry and did not establish complete output
correctness. The target recovered here is the correctness-backed historical
3.4k class.
