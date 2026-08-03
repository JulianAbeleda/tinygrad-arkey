# NV decode campaign - final wall-time parity vs llama.cpp

Date: 2026-08-02 (measured 2026-08-03 04:20-04:30 UTC)
Status: parity record for the decode-norm campaign tail. Same session, same model
(Qwen3-8B-Q4_K_M), same RTX 5090 (sm_120), NVRTC-compiled CUDA kernels on the tinygrad side.
No promotion is claimed by this record; it is the campaign's final wall-time comparison at
d512/d2048/d4096 after M2/M3/M4/M5 and Path 3 all landed. Reproducible promotion state
(correction per `nv-campaign-forward-review-amendment-20260803.md` section 2.5): M2's
`decode_epilogue_fusion` record is OPEN for `NV:sm_120` (Q6K down-coop in-kernel merge);
M3 norm, M4 Q4K epilogue, M5 combine, and Path 3 semantic RMSNorm are CLOSED. The baseline
this record compares against is the M2-on state.

## Protocol

- tinygrad: `nvidia-bringup-20260731` @ `19cff5c46`, default runtime (M2 open for
  NV:sm_120, M3/M4/M5/Path 3 closed), `--no-fused-prefill` (HEAD's fused prefill attention is a broken AMD-loop-state ABI
  on NV and is excluded from the NV baseline), fixed-depth prompt `[1]*depth`, chunk 32,
  temperature 0.0, nmeas=20, reps=3, median tok/s.
- llama.cpp: CUDA build `ac4cddeb0` (`build-cuda`), `llama-bench -ngl 99 -fa 1 -pg 0,10
  -d <depth> -r 5`, decode row (`n_prompt=0, n_gen=10`), mean tok/s.
- Both measured in the same session back to back; GPU clock state shared.

## Wall-time parity (decode tok/s)

| depth | tinygrad (NVRTC) | llama.cpp | ratio | gap |
| --- | ---: | ---: | ---: | ---: |
| d512 | 172.80 | 248.20 +/- 7.37 | 69.6% | 1.44x |
| d2048 | 161.50 | 235.14 +/- 7.36 | 68.7% | 1.46x |
| d4096 | 149.00 | 225.95 +/- 6.71 | 65.9% | 1.52x |

Correctness pins hold at every depth: token sha256
`9d6b3787cef8c4a7b208df30c05c049f692a5ebc80dd19c2994dd54c18e789b9` 3/3 and first token
`151936` 3/3.

## Kernel census (tinygrad, per token, d512/d2048/d4096 identical)

| class | count |
| --- | ---: |
| total kernels/token | 1021 |
| total kernel us/token | 6187 / 6576 / 7091 |

The decode gap is inside the bandwidth-bound GEMV regime: per-token kernel count is flat with
depth (1021) and the depth penalty is context-side (KV reads), matching llama's own decay
(248.2 -> 226.0, -9.0%; ours 172.8 -> 149.0, -13.8%). The remaining 1.4-1.5x is the
per-kernel GEMV efficiency gap (llama `mul_mat_vec_q` dp4a vs our q4k GEMV family), which is
the pre-existing L4 lever in `nv-performance-campaign-scope-20260801.md` and is NOT part of
the decode-norm campaign.

## What the decode-norm campaign moved

Baseline for comparison (M2-on state before this campaign, recorded in the Path 3 task):
d512 ~172.7 tok/s, 1021 kernels, ~6179us. Final state is unchanged in wall time at d512
(172.80) while proving byte-identical tokens across every landed piece: M3 fused decode
RMSNorm (closed), M4 q4k epilogue fusion (closed), M5 flash-combine fp16 absorption
(closed), Path 3 semantic RMSNorm (closed). All four measured non-landing; none changed the
default runtime's sha. The campaign's verdict stands: norm-family work does not move wall
time while the opaque custom-kernel boundary copy tax exists (M3 reopen condition), and the
next wall-time lever is L4 decode GEMV efficiency, not another norm fusion.

## Artifacts

- llama: same-session `llama-bench` decode rows (mean +/- stddev over 5 reps).
- tinygrad: `/tmp/path3_e2e_probe.py --depth N --nmeas 20 --reps 3 --no-fused-prefill`.
- This record replaces the one-sided decode table in
  `docs/nv-prefill-decode-diagnosis-20260801.md` (which had llama only at d512) as the
  campaign-tail reference.
