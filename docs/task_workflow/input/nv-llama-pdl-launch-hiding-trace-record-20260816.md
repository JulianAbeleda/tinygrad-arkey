# NV llama launch-hiding trace: single-stream PDL, quantified + substrate check (2026-08-16)

Date: 2026-08-16
Branch: `nvidia-bringup-20260731`
HEAD at analysis: `15da2e215`
Trace: `/tmp/llama_tg10_node_20260812.sqlite` (llama CUPTI, graph 5 = 16 replays x 762 kernels)
Status: **traced and quantified; verdict on native substrate: launch-gap half already wired
(dependent-QMD chain), programmatic half CONSTRUCTION-REQUIRED and economics-negative at HEAD.**

## 1. Question

llama hides ~925 us of kernel time on a single stream. What is the mechanism exactly, and is it
expressible on the native NV runtime (`ops_nv.py`, QMD-based, not CUDA)?

## 2. Trace evidence (first 762-kernel replay, graph 5, verified against the pinned sqlite)

All kernels in the replay run on **one stream (stream 45)**. Replay span 3840.6 us vs node-sum
4766.0 us -> **overlap mass 925.3 us = 19.4% of node sum**.

**543 of 761 same-stream launches (71.4%) begin before their predecessor ends.** This is
same-stream overlap, i.e. PDL behavior, not a multi-stream artifact.

Per-class hidden mass against the `mul_mat_vec_q` anchor (217 nodes, sum 3538.7 us):

| class | sum us | hidden us | hidden % |
| --- | ---: | ---: | ---: |
| quantize_q8_1 | 480.3 | 386.5 | 80.5 |
| rms_norm_f32 | 308.0 | 153.7 | 49.9 |
| flash_attn_combine_results | 120.4 | 83.9 | 69.7 |
| flash_attn_ext_vec | 113.7 | 56.0 | 49.3 |
| rope_neox | 125.3 | 30.5 | 24.3 |
| k_set_rows | 74.8 | 0.0 | 0.0 |
| k_get_rows_float | 2.8 | 0.0 | 0.0 |
| k_bin_bcast | 1.9 | 0.5 | 28.8 |

Overlap is distributed through the whole replay (windows 0-120 / 300-420 / 640-762 show
129.6 / 146.2 / 157.5 us). Timeline sample: `flash_attn_ext_vec` runs 0-2.88 us while the next
`quantize_q8_1` starts at 1.82 us and `mul_mat_vec_q` starts at 2.14 us - all on stream 45.

## 3. Mechanism: CUDA programmatic dependent launch (PDL)

Traced from `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/common.cuh` (HEAD `ac4cddeb0`):

- `common.cuh:1528-1545`: `ggml_cuda_pdl_config` sets `cudaLaunchAttributeProgrammaticStreamSerialization = 1`
  and launches via `cudaLaunchKernelEx` (`common.cuh:1623-1645`).
- Gate: `ggml_cuda_kernel_can_use_pdl` = `attr.ptxVersion >= 90` (Hopper+), so virtually every
  llama CUDA kernel opts in (`common.cuh:1608-1616`).
- Device side: `ggml_cuda_pdl_lc()` calls `cudaTriggerProgrammaticLaunchCompletion()` at kernel
  start (`common.cuh:129-132`); `ggml_cuda_pdl_sync()` calls `cudaGridDependencySynchronize()`
  (`common.cuh:114-118`).
- PDL forces `GGML_CUDA_RESTRICT` to be empty (no `__restrict__`) on PDL builds
  (`common.cuh:1614-1619`).

Net effect: the *next* kernel's grid can begin launching (CTA dispatch, param load, prologue)
while the *current* kernel is still executing; the consumer waits on grid completion before
touching the producer's output. This is what the 71.4% same-stream early launch and the
distributed overlap mass measure.

## 4. Substrate check on the native NV runtime (`ops_nv.py`)

`ops_nv.py` is a native NV driver runtime (QMD + GPFIFO, not CUDA). Two halves:

**Launch-gap half - already wired.** `NVComputeQueue.exec()` chains consecutive same-queue kernels
via the QMD dependent pointer: `dependent_qmd0_pointer`, `dependent_qmd0_action=1`
(`QMD_SCHEDULE`), `dependent_qmd0_prefetch=1`, `dependent_qmd0_enable=1` (`ops_nv.py:166-171`).
The Blackwell QMD v05 struct exposes the full grid-dependency field set (`DEPENDENCE_COUNTER`,
`DEPENDENT_QMD0_ACTION_QMD_DECREMENT_DEPENDENCE`, `WAIT_ON_LATCH_ID`/`ARRIVE_AT_LATCH_ID`), and the
graph runtime already relies on the chain: "in case of NV, optimize when only 1 same-queue
dependency exists, since NV chains 2+ executions in this case, eliminating dependency need"
(`tinygrad/runtime/graph/hcq.py:390-393`). So CPU round-trips between same-queue kernels are
already gone on the native path.

**Programmatic half - CONSTRUCTION-REQUIRED.** True PDL (next grid launches while the current grid
is still running, with the consumer doing a grid-dependency wait before reading producer data)
needs kernel-side trigger/sync instructions (`griddepcontrol`-class PTX) plus QMD grid-dependency
programming. Today:
- no renderer emits a programmatic launch-completion trigger or grid-dependency wait
  (zero `griddep` hits in `tinygrad/renderer/ptx.py`, `cuda.py`, `ops_nv.py`);
- the runtime never programs `WAIT_ON_LATCH_ID`/`ARRIVE_AT_LATCH_ID`/`DEPENDENCE_COUNTER` even
  though the autogen structs contain them.
The hardware fields exist, so this is construction work, not a hard block - but it spans renderer
emission, QMD programming, and a per-pair scheduler policy.

## 5. Economics: the 925 us is not transferable

Most of llama's hidden mass corresponds to kernels tinygrad already fused away:

| llama hidden class | hidden us | tinygrad fate |
| --- | ---: | --- |
| quantize_q8_1 | 386.5 | folded into GEMV bodies (no node left to hide) |
| rms_norm_f32 | 153.7 | fused into norm epilogues |
| rope_neox | 30.5 | fused into q/k epilogue |
| flash score + combine | 139.9 | the only remaining exposed hidden mass |

That is ~571 us of llama's 925 us hidden mass with no corresponding separate node in tinygrad.
The flash pair is the only exposed item, and the fresh ledger
(`nv-decode-path-pseudocode-fresh-ledger-20260816.md`) prices it at +122 us installed with
**matched isolated bodies at parity** (llama 4.10 us vs tinygrad 4.16-4.19 us). The exhaustive
08-13 account (`nv-launch-hiding-substrate-exhaustive-account-20260813.md`) priced total
recoverable launch-hiding at ~18-33 us, not the ~925 us overlap mass.

Baseline note (do not reuse the stale 193.5): production NV census at HEAD is **~205-207 tok/s**
(205.4 at `86d653651`, 207.4 at `ef24c46ae`, token sha unchanged); the 193.5 tok/s figure is the
08-15 pre-Q6-V baseline (`d40b938af`) and the CUDA-route 179.6 tok/s is the separate Route B
multi-stream measurement with no NV reduce-output fusion. llama fresh pair is 245.5-248 tok/s, so
the HEAD gap to llama is ~38-43 tok/s (~0.7-0.8 ms/token).

## 6. Verdict

1. llama's mechanism is single-stream PDL, now quantified from the pinned trace: 925.3 us overlap
   mass (19.4%), 71.4% same-stream early launches.
2. On the native NV runtime, the launch-gap half is already active via the dependent-QMD chain;
   the programmatic (mid-kernel trigger + grid-dependency wait) half is not wired and would need
   renderer + runtime + policy work. It is CONSTRUCTION-REQUIRED, not BLOCKED.
3. It is **not the lever to parity at HEAD**: ~571 us of the hidden mass is already fused away in
   tinygrad, and the remaining flash pair is at body parity with its installed gap priced at
   ~+122 us (launch/graph-install behavior, recoverable mass bounded by the ~18-33 us exhaustive
   account). Building full PDL does not buy the 240 line; the open reduce-output / M1-family rows
   (~750 us census) remain the larger exposed mass, and the composition arithmetic must be rebased
   against the ~205-207 tok/s HEAD baseline, not the stale 193.5.

## Evidence

- Trace: `/tmp/llama_tg10_node_20260812.sqlite`, graph 5, first replay (762 kernels)
- llama source: `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/common.cuh` (PDL: 114-132,
  1528-1545, 1608-1616, 1623-1645)
- native runtime: `tinygrad/runtime/ops_nv.py:141-209` (QMD dependent chain),
  `tinygrad/runtime/graph/hcq.py:390-393` (NV chain optimization)
- QMD v05 fields: `tinygrad/runtime/autogen/nv_570.py` (`DEPENDENCE_COUNTER`,
  `DEPENDENT_QMD0_*`, `WAIT_ON_LATCH_ID`, `ARRIVE_AT_LATCH_ID`)
- prior accounting: `nv-launch-hiding-substrate-exhaustive-account-20260813.md`,
  `nv-decode-path-pseudocode-fresh-ledger-20260816.md`
