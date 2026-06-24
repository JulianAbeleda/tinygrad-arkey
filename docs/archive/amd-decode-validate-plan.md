# Validate-before-close — stress-test the decode conclusion (scope)

Date: 2026-06-15. Before closing the decode investigation, test the UNDER-MEASURED e2e claims that could
refute "the kernel is solved but the e2e wall is structural." Three experiments.

## V1 -- per-kernel e2e breakdown (CRITICAL: could overturn the whole kernel focus)
Question: of the ~33 ms decode token, what FRACTION is the Q4_K GEMVs vs attention vs q8-quant vs norms vs
sampling? If the GEMVs are a MINORITY, our "kernel doesn't translate" is really Amdahl (GEMV was never the
bottleneck), and we've optimized the wrong kernel.
- Method A: PROFILE=1 -> profile.pkl -> ProfileGraphEvent (the JIT replay) -> per-kernel durations from
  ents[st_id..en_id] into sigs (device timestamps) -> bucket by kernel name (r_* GEMV reduces, sdpa/attn,
  copy, E_* elementwise/norm). Decode the sigs properly this time.
- Method B (fallback ablation): time the token with the Q4_K GEMV doing minimal work vs real -> the drop =
  GEMV fraction.
- Gate: GEMV >= ~50% of token -> GEMV dominates, our focus was right, structural-wall stands. GEMV < ~35%
  -> Amdahl; the real bottleneck is elsewhere (attention/quant/sampling) -> REOPEN with the right target.

## V2 -- llama.cpp on THIS machine (the bar is unverified on our hardware)
Question: is the 103.84 tok/s bar real on OUR (possibly degraded) GPU, or a phantom? Run llama.cpp on the
same Qwen3-8B Q4_K_M, same GPU, measure decode tok/s.
- Check if llama.cpp is built/available (ROCm/HIP build). If yes, run; if not, note and skip (don't build
  from scratch unless quick).
- Gate: llama.cpp >> our 30 -> the gap is real on our hardware (our conclusion's premise holds). llama.cpp
  ~= 30 -> the gap is a cross-machine artifact; there may be NO gap on our GPU -> conclusion moot.

## V3 -- healthy-GPU re-measure (remove the degradation confound)
Question: are our e2e numbers (30 tok/s, readraw 54%) a degraded-GPU artifact? Historical: fp 58, readraw 85%.
- Diagnose: rocm-smi clocks/temp/throttle state. Is the GPU throttled?
- Re-run readraw + fp e2e; compare to historical. If they recover toward 85%/58 -> this session was degraded
  and the e2e numbers (incl. the vdot 425->61 gap) are partly artifact -> re-measure the key e2e claims.
- If still 54%/30 on a healthy-looking GPU -> the numbers are real, not degradation.

## Honest framing
These don't re-run known-nulls -- they test the FOUNDATION of the e2e conclusion. The kernel-level findings
(v_dot4 near-saturates standalone; fp dequant caps) are solid regardless. What's at stake is whether the
"structural e2e wall" story is earned or whether (a) the GEMV isn't even the bottleneck (V1), (b) there's no
real gap on our hardware (V2), or (c) the e2e numbers are degradation artifacts (V3). Any of the three could
materially change the conclusion. Report honestly whichever way they fall.
