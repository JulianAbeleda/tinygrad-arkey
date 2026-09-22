# llama flash_attn_ext_vec matched isolated body (2026-08-16)

Date: 2026-08-16
Branch: `nvidia-bringup-20260731`
Measurement baseline: `86d653651`
Status: **decisive measurement.** The first matched isolated-vs-isolated body
comparison of the two decode flash score kernels, built in response to the
design review (finding 2: the old `4.19 isolated tinygrad vs 3.16 in-situ
llama` was not a valid body comparison).

## 1. The missing measurement

The old comparison subtracted a llama **in-situ** CUPTI number (3.16 us/node,
measured inside its full graph, PDL-enabled) from a tinygrad **isolated**
number (4.19 us, back-to-back launches) and called the difference a "body
gap." That is not a body comparison, and the design review said so. This record
measures llama's kernel in the same isolated harness we use for ours: nsys
CUPTI kernel duration, 400 back-to-back launches on the default stream, 20
warmup, same shape (Hd=128, Hq=32, Hkv=8, Tc=512).

## 2. Probe

`extra/llm_research/microbench/llama_fattn_vec_iso.cu` compiles llama's own
`flash_attn_ext_vec<128, 1, F16, F16, false>` from
`ggml-cuda/fattn-vec.cuh` (HEAD `ac4cddeb0`) with the pinned launch shape:
grid `(1, 2, 32)` (Codex-corrected grid.y=2), block `(32, 4, 1)`, fp16 K/V,
no mask, scale 1/sqrt(128). Correctness gate: finite output, softmax
meta `(max, sum)` sane.

## 3. Matched results (RTX 5090, sm_120, this session)

| kernel | config | isolated us med | n |
| --- | --- | ---: | ---: |
| llama `flash_attn_ext_vec` | grid.y=1 (32 blocks) | 6.176 | 399 |
| llama `flash_attn_ext_vec` | grid.y=2 (64 blocks, pinned) | **4.096** | 399 |
| llama `flash_attn_ext_vec` | grid.y=4 (128 blocks) | 3.136 | 399 |
| tinygrad production tile | S=48 (384 blocks) | **4.160-4.192** | 400 |

Stability: two independent runs per config agree within 0.004 us (grid.y=1:
6.176/6.174; grid.y=2: 4.096/4.096; grid.y=4: 3.136/3.140).

## 4. Conclusion

**The flash score body gap is falsified by matched measurement.** At its pinned
launch config, llama's isolated body is 4.10 us; our production tile is
4.16-4.19 us. They are at parity within ~2%, not 1.33x. The old "3.16 us"
number is a llama in-situ, PDL/graph-install measurement and was never a body
number; subtracting it from an isolated tinygrad body was invalid, exactly as
the review found.

The recoverable flash score mass is therefore not in the kernel body. The
installed difference (our 6.56 us census row vs llama's 3.14-3.16 us in-situ)
is a launch/graph/overlap effect on llama's side (PDL-enabled graph, launch
hiding), which is the overlap axis already measured FLAT on our DAG, not a
score-body deficit.

## 5. Implication for the work queue

- Stop treating the flash score body as a lever. The vec rewrite chased a body
  gap that matched measurement shows does not exist.
- The next flash measurement (if any) is llama's installed launch behavior, not
  the kernel math: same graph embedding, PDL semantics, and kernel ordering.
- The design review's other gates (SASS Q-load check, split-count sweep, NV
  policy boundary) are unaffected; they target the installed axis, not a body
  rewrite.

## Evidence

- `docs/task_workflow/evidence/nv-llama-fattn-matched-iso-20260816.json`
- probe: `extra/llm_research/microbench/llama_fattn_vec_iso.cu`
- llama source: `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh`
- in-situ reference: `nv-llama-d512-node-ledger-20260812.json`
