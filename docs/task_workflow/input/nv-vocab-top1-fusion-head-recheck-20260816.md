# NV vocab top-1 fusion head recheck - corrected wall attribution

Date: 2026-08-16
Status: audit record. The 08-14 NO_GO_WALL verdict is **confirmed at HEAD**, and the
wall mechanism is **corrected**: the loss is NOT the packed-u64 reduce chain (that
chain is ~9 us FASTER than the legacy tail in isolation).  The +25.5 us/token wall is
in-situ (eager/JIT handoff inside the decode), not in any kernel body.
Branch: `nvidia-bringup-20260731`, HEAD `8d8500982`.

## 1. Why this recheck

The 08-14 record attributed the wall to the scheduler u64 cross-tile reduce
(`r_16_4_1187` "~83.3 us" vs the "~56 us" legacy tail).  A fresh A/B at HEAD and two
isolated CUPTI probes refute that attribution and pin the wall elsewhere.

## 2. Fresh A/B at HEAD (end-to-end, NV backend, lease route)

Harness: `nv_vocab_top1_fusion_ab.py` (control = legacy 4-kernel argmax tail;
candidate = `q6k_vocab_top1_call` packed u64 epilogue + scheduler cross-tile reduce).
Qwen3-8B-Q4_K_M, depth 512, count 8, 3 timing arms, flocked.  Token stream bit-identical
across arms (`7e2ff56b...`).

| arm | median ms/token | tok/s | delta vs control |
| --- | ---: | ---: | ---: |
| control bracket (legacy tail) | 4.804 | ~208 | - |
| candidate (fused epilogue) | 4.830 | ~207 | **+25.5 us/token** |

`cost_gate.result = FAIL` (predicted -50 us, measured +25.5 us, gap 75.5 us, CONTRADICTED).
Verdict stays `NO_GO_WALL`.

Evidence: `docs/task_workflow/evidence/nv-vocab-top1-fusion-ab-head-20260816.json`

## 3. Isolated CUPTI kernel bodies at HEAD (CUDA backend, 200/150 replays)

The isolated kernels do NOT reproduce the wall:

| component | CUPTI median | source |
| --- | ---: | --- |
| legacy tail chain (`r_32_4_1187` + `r_128_16_8_1187` + `r_16_8`) | 51.43 us | `nv-vocab-tail-cupti-head-20260816.json` |
| fused tail chain (`r_16_4_1187` + `r_16_4`) | 42.66 us | same |
| tail delta (fused - legacy) | **-8.77 us (fused faster)** | same |
| GEMV plain `q6k_gen_coop_151936_4096_inkernel` | 324.93 us | `nv-vocab-gemv-epilogue-cupti-head-20260816.json` |
| GEMV `..._epi_vocabtop1` | 325.32 us | same |
| GEMV epilogue delta | **+0.39 us** | same |

So the isolated component arithmetic predicts the fused route should WIN ~8.4 us/token,
opposite to the measured +25.5 us end-to-end.  The 08-14 record's "r_16_4_1187 ~83.3 us"
is stale at HEAD: the same kernel measures 42.1 us isolated.

## 4. In-situ attribution: the wall is the eager/JIT handoff, not a kernel

The +25.5 us wall with favorable isolated bodies means the loss is in how the fused
route's winner crosses the eager/JIT boundary each token.  Two probes confirm:

- The `.clone()` held-copy is REQUIRED: removing it wedges the token stream
  (`4710,32313,32313,...` instead of `4710,32313,11,...`), the exact JIT-replay
  firewall documented in the 08-14 record.
- Removing the clone also collapses overlap: the candidate timing explodes to
  ~39.1 ms/token (control 4.79 ms).  The clone is what lets the eager `item()` read the
  winner without serializing the JIT graph; without a fresh per-token allocation the
  handoff forces a full drain.  So the clone is the mechanism that keeps the fused
  route viable at all, not an idle cost.

The remaining ~35 us of in-situ overhead (candidate +25.5 us against an isolated
prediction of -8.4 us) is the eager/JIT memory-plan interaction of the research route:
fresh key-buffer allocation + held-copy + reduce chained after the GEMV inside the
decode graph, where the legacy argmax path's buffers are stable across tokens.

## 5. Verdict and disposition

- F5 (vocab aux fusion) stays **NO_GO_WALL** with the corrected attribution: the wall
  is in-situ, not the u64 reduce chain.
- Even if the in-situ overhead were removed, the best-case value of F5 is the isolated
  ~8-9 us/token tail saving, far below the +50 us promotion bar and ~+0.4 tok/s.
- Single-pass cross-tile max is not buildable on the PTX renderer (no atomics, no
  grid sync) and, per the substrate status doc, is the only Path A row that was
  substrate-missing; the kernel-body measurement now shows it would only recover
  ~8-9 us, not the wall.
- Path A kernel work on the vocab row is therefore **exhausted**: no further body-level
  lever exists for F5.

## Evidence files

- `docs/task_workflow/evidence/nv-vocab-top1-fusion-ab-head-20260816.json` (end-to-end A/B)
- `docs/task_workflow/evidence/nv-vocab-tail-cupti-head-20260816.json` (isolated tail chain)
- `docs/task_workflow/evidence/nv-vocab-gemv-epilogue-cupti-head-20260816.json` (isolated GEMV epilogue)
