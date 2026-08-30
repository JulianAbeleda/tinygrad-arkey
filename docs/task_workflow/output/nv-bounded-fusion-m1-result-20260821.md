# NV bounded-fusion M1 result (2026-08-21)

Status: measurement record. No production runtime, renderer, scheduler, or
model file was changed. The candidate was activated through the existing
harness-only lease `_rms_affine_gateup_norm_weight`; no loader policy creates
it, so the default decode graph is untouched. Every GPU arm ran as a fresh
process under `/tmp/gpu-bench.lock`. Notation: `O` observed, `I` inferred,
`U` unmeasured.

## 1. Candidate

M1: absorb the FFN RMSNorm epilogue into the fused w1+w3 gate/up GEMV.

Per decode token the 36 FFN-norm chains render

```text
r_16_256                 (scale reduce, ~4.1 us)
E_32_32_4_f14a5cc0       (scale x weight epilogue, ~2.34 us)
q4k_g3_lanemap_gemv_w1w3fused16_12288_4096  (fused gate/up GEMV, ~38.78 us)
```

The candidate applies the control epilogue `(half)((h*s)*w)` at every packed
Q4 load inside the gate/up GEMV (`q4k_g3_lanemap_w1w3_rms_affine16_*`), with
the single fp16 RNE round at the end of the fp32 multiply chain and the
bitwise-exact control scale rendered by the same `r_16_256` program. The
control arm is the landed M2d candidate without the M1 lease, so the only
inter-arm delta is the norm-epilogue absorption.

## 2. Gates

| gate | result | key evidence |
| --- | --- | --- |
| NV render smoke | `PASS` (O) | `survive=true`; 36 `rms_affine16` bodies; 0 `fused16`; `E_32_32_4_f14a5cc0` 37 -> 1; `r_16_256` stays 37 |
| exact full-logits | `PASS` (O) | fp32 SHA-256 `596c73cb4905676c92a91a62be5d1255606ccdfee38c59394e44d0619fed2d96` identical control vs candidate; tokens/shape equal |
| census | `PASS` (O) | 594 -> 558 kernels; the only shifts are the three named families |
| reverse wall bracket | `NO_GO_WALL` (O) | candidate +82.08 us/token slower |
| cost prediction | `FAIL` (O) | measured +82.08 us is outside the predicted `[-59.76, +12.24]` us range |

## 3. Census

| family | control | candidate |
| --- | ---: | ---: |
| `E_32_32_4_f14a5cc0` (ffn-norm epilogue) | 37 | 1 |
| `q4k_g3_lanemap_gemv_w1w3fused16_12288_4096` | 36 | 0 |
| `q4k_g3_lanemap_w1w3_rms_affine16_12288_4096` | 0 | 36 |
| `r_16_256` (scale reduce) | 37 | 37 |
| total kernels | 594 | 558 |
| node-sum mass | 5144.62 us | 5269.69 us |

The fold is exact: 36 epilogues disappear, the fused16 -> rms_affine16 swap is
1:1, and the scale reduce is unchanged, yet the node-sum mass **rises** by
125.07 us. The fused body is 44.575 us/node against the control's 38.78 us
(+5.80 us x 36 = +208.6 us), which outweighs the 84.24 us of removed epilogue
mass. The norm is recomputed once per matrix dot (gate and up), and x is
streamed fp32 rather than the control's fp16 (O/I).

## 4. Wall bracket

Control / candidate / control, settled continuous windows, 5 reps x 32 tokens
(160 timed tokens per arm). Token stream SHA-256
`f25083e5d0a754131283b40c03f52e688fee9f175bea7ae106805e7d628d7905` is
identical across all three arms (O).

| arm | median ms/token |
| --- | ---: |
| control A | 4.7330828 |
| candidate | 4.8151934 |
| control B | 4.7331488 |

| quantity | value |
| --- | ---: |
| candidate minus control A | +82.111 us/token |
| candidate minus control B | +82.045 us/token |
| candidate minus control midpoint | +82.078 us/token |
| promotion bar | -50 us/token vs both controls |
| verdict | `NO_GO_WALL` |

## 5. Verdict

The M1 norm-epilogue absorption is **refuted at this head**. It is bitwise
exact and removes 36 kernels, but the fused gate/up GEMV costs more than the
launch it saves because the scale x weight epilogue re-executes twice per
element (gate and up) and the consumer streams fp32 instead of fp16. The
measured +82.08 us/token regression sits ~70 us beyond even the pessimistic
end of the cost model, so the unmodeled in-kernel critical path and activation
traffic dominate the launch removal.

This closes the last named "residual fold" candidate: the fold-into-GEMV
mechanism does not move the needle toward llama at d512. Combined with the
already measured FFN reduce-output (+67 to +79 us), q/k reduce-output
(+38.7/+43.2 us at its old baseline), vocab top1 (`NO_GO_WALL`), and copy-free
native RMSNorm (+12.5 to +17.1 us), the bounded-fusion direction (H8) is now
comprehensively wall-negative. The remaining gap is scheduling/overlap
(Direction A), not a missing body fold.

## 6. Evidence

- Orchestrator record:
  `docs/task_workflow/evidence/nv-bounded-fusion-m1-20260821/m1-ab.json`
- Child gates and timing:
  `docs/task_workflow/evidence/nv-bounded-fusion-m1-20260821/m1-ab.children/`,
  `.../m1-ab.timing/`
- SHA-256 manifest:
  `docs/task_workflow/evidence/nv-bounded-fusion-m1-20260821/sha256.txt`
- Harness:
  `extra/llm_research/decode/nv_epilogue_absorption_m1_ab.py`

## 7. Command

```text
cd /home/ubuntu/tinygrad-arkey
env PYTHONPATH=. DEV=NV .venv/bin/python \
  extra/llm_research/decode/nv_epilogue_absorption_m1_ab.py \
  --mode ab --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --depth 512 --count 32 --reps 5 --max-context 1024 --settled-continuous \
  --timeout 900 --lock-wait 120 \
  --out docs/task_workflow/evidence/nv-bounded-fusion-m1-20260821/m1-ab.json
```
