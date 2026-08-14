# NV flash-score item-3 first-principles record - kernel body is flat, the win is overlap

> **SUPERSEDED 2026-08-13** by `nv-truth-audit-flash-body-20260813.md`. The
> "kernel body flat" conclusion rested on a Python-dispatch-dominated microgate
> that also compared two non-production S=4 configs. Device-side CUPTI timing
> shows the production tile at 4.19 us/node vs llama 3.16 us (+68 us in-situ),
> and the vec kernel at 10.2 us (NCHUNK=2) / 39.6 us (NCHUNK=9), so the body is
> NOT flat and the vec kernel is NOT ready to route. See the audit for the
> corrected ladder.

Date: 2026-08-13
Branch: `nvidia-bringup-20260731` (HEAD `c56d33c14`)
Status: **first-principles verdict (no production change).** Answers "can we
do item 3" for the flash-score row on the 220 -> 245 tok/s ladder. The llama
single-pass kernel body is already built and already measured; first principles
say it cannot move the wall, and the measurement agrees.

## 1. What item 3 is

Item 3 is the flash-score row: the +68.1 us CUPTI delta between tinygrad
(182.0 us / 36 nodes) and llama (113.9 us / 36 nodes), and by extension the
flash half of llama's launch hiding on the fused-ceiling ladder. It has two
independent parts that must not be conflated:

1. **Kernel body** - tinygrad's tiled two-kernel score/PV vs llama's
   single-pass `flash_attn_ext_vec`.
2. **Overlap** - whether the flash execution time is hidden behind the GEMV
   chain or exposed serially.

## 2. First-principles arithmetic - the body cannot be the lever

At d512 the flash score/PV work is tiny:

- Per query head: score = 128 dims x 512 KV = 65,536 MACs; PV = 512 x 128 =
  65,536 MACs; ~131K MACs/head.
- x 32 heads = ~4.2M MACs total. At even a conservative 10 TFLOPS fp16 this
  is under 0.5 us; the RTX 5090 fp16 rate makes it well under 0.1 us.
- K/V stream = 8 KV heads x 512 x 128 x 2 bytes x 2 tensors = 2 MB, ~1 us at
  L2 bandwidth.

The kernel is therefore launch-latency and L2-stream bound, not
instruction-structure bound. The structural deltas that the trace doc listed
(32-lane reduce vs 8-lane groups, per-tile LDS staging vs no staging, Q re-read
vs register-resident Q) change instruction counts that are a small fraction of
the ~65 us graph wall. First principles predict a structural rewrite is flat;
the microgate confirms it.

## 3. What we already measured

`extra/llm_research/decode/nv_flash_vec_llama_microgate.py`, GPU-held,
reverse A/B/A, 200 graph replays x 5 reps, S=4 (llama's split count), same
combine on both arms:

| arm | median us/graph |
| --- | ---: |
| legacy tile control midpoint | 65.4876 |
| llama-vec single-pass candidate | 65.4664 |
| delta | **-0.0213 (flat)** |

Correctness is near-bitwise (fp32 normal max_abs 1.3e-8, fp32 dynamic 4.8e-6,
fp16 normal 7.6e-6, zero cases exact; all finite). So the single-pass body is
correct but has zero wall effect. The kernel rewrite is done and is not the
lever. Evidence: `docs/task_workflow/evidence/nv-flash-vec-llama-microgate-20260813.json`.

## 4. The actual llama advantage is overlap, and most of it is already ours

The pinned same-measurement ledgers (`nv-llama-d512-node-ledger-20260812.json`,
`nv-tinygrad-d512-node-ledger-20260813.json`) show llama runs on one stream but
with 946.4 us of in-graph concurrency (node-sum 4774.4 vs span 3835.2 us);
tinygrad hides zero (span 5380.5 us is larger than node-sum 5149.4 us and
pays a 232.2 us launch gap).

Decomposing llama's hidden mass, most of it is already superseded by our
fusion, not recoverable:

| llama class | hidden behind mmq (us) | tinygrad status |
| --- | ---: | --- |
| quantize_q8_1 | 391.3 | already fused into our GEMV (anchor at parity) |
| rms_norm | 155.9 | already fused (we are -258 us ahead of llama) |
| rope | 32.5 | already fused into q/k epilogue (we are -126.8 us ahead) |
| flash score | 57.8 | **exposed (0 hidden)** |
| flash combine | 85.2 | **exposed (0 hidden)** |

The only overlap mass we have not already eliminated by fusion is the flash
pair: **~143 us** (57.8 score + 85.2 combine) that llama spends behind MMQ and
tinygrad spends serially. That is the real residual of item 3.

## 5. Verdict - can we do it

The kernel body: already built, correct, and flat. First principles say it
cannot win, and the S=4 microgate says exactly that. This half is closed.

The overlap: it is not a flash kernel problem, it is a scheduler problem.
Recovering the ~143 us means co-scheduling the flash kernels with an
independent GEMV chain inside the decode graph, a substrate tinygrad does not
have. The 08-05 causal record named this as the open multi-queue construction
blocker; the P2 co-schedule scan measured a 33 us ceiling for in-graph
co-scheduling on this topology, below the +50 us promotion bar. No GPU arm is
authorized and none clears the gate.

Therefore item 3 is **NOT a kernel-rewrite win**. The next levers stay the
fusion rows: reduce-output epilogue (392 us), vocab aux chain (57.3 us), and
the rope/cast tail. The last ~0.3 ms to full 245 tok/s parity is a scheduler
substrate build, not a flash patch.

## Evidence

- microgate: `docs/task_workflow/evidence/nv-flash-vec-llama-microgate-20260813.json`
- llama node ledger: `docs/task_workflow/evidence/nv-llama-d512-node-ledger-20260812.json`
- tinygrad node ledger: `docs/task_workflow/evidence/nv-tinygrad-d512-node-ledger-20260813.json`
- trace: `docs/task_workflow/input/nv-flash-score-llama-trace-20260813.md`
- overlap hard stop: `docs/task_workflow/input/nv-decode-overlap-p2-verdict-20260812.md`
- causal record: `docs/task_workflow/input/nv-decode-native-flash-causal-record-20260805.md`
