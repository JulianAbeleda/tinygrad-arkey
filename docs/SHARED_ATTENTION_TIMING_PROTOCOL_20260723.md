# Shared Attention Timing Protocol

## Decision

Unpinned timing is valid for peak-performance and roofline exploration. Pinned timing is required only for reproducible promotion decisions under the current evaluation harness.

The harness check for `clock_pin=true` is a measurement-integrity policy. It does not imply that pinned clocks are faster, or that an unpinned result is invalid as a peak result.

## Evidence from prior runs

- The recent whole-model authority artifacts recorded `clock_pin: null` and no clock-state identity.
- Those results are therefore unpinned smoke evidence, not clock-controlled promotion evidence.
- Older attention replay artifacts also lack clock metadata and must be labeled unpinned.
- No conclusion about pinned versus unpinned performance is valid until both modes are measured on the same candidate and workload.

## Two-track reporting

Every serious performance result should report both tracks when possible:

1. **Peak track:** unpinned clocks; report median, best sample, warmups, repetitions, and observed clock telemetry when available. This answers: “What performance can the system reach?”
2. **Promotion track:** pinned clock state; report the same protocol and a `clock_state_id`. This answers: “Can the candidate reproduce an apples-to-apples improvement?”

Unpinned results must not be silently substituted for pinned promotion evidence. Pinned results must not be presented as the hardware peak unless they are also the best observed mode.

## Required A/B protocol

For each candidate and baseline:

- Keep workload, binary identity, input identity, device, driver, warmups, repetitions, synchronization, and sample statistic identical.
- Run an unpinned session and record `clock_pin=false` or `null` plus any available observed clock telemetry.
- Run a pinned session and record `clock_pin=true` and a stable `clock_state_id`.
- Report median, p10/p90, best sample, and variance for both sessions.
- Preserve separate artifact identities; do not merge pinned and unpinned samples into one authority result.
- Use pinned results for the existing promotion gate; use the better mode for roofline analysis only when its clock state is explicitly stated.

## Current status

The `2180.97 tok/s` warmed 8B `pp512` result is unpinned smoke evidence (`clock_pin: null`) and failed the current `3300 tok/s` promotion threshold. It remains useful for optimization direction, but a pinned-versus-unpinned A/B is required before interpreting the gap as a compiler or kernel limitation.

The first reduced-LDS runtime A/B passed full `512x4096` numeric comparison with zero maximum absolute error. In an isolated guarded run, baseline kernel time was `0.71408 ms` and the one-buffer candidate was `0.56976 ms` (about 20% faster). CPU affinity was pinned to core 0, but the attempted GPU clock pin was rejected by sysfs permissions; the device remained in automatic power mode. These numbers are therefore correctness and optimization smoke evidence, not clock-controlled roofline evidence.

The supported ROCm SMI interface subsequently allowed a controlled run without direct sysfs writes. Manual SCLK level 2 and MCLK level 3 were selected, with `Performance Level: manual` observed throughout, followed by restoration to `auto`. With 3 warmups and 10 synchronized samples on CPU core 0:

- Two-buffer baseline: median `0.45470 ms`, p10 `0.453264 ms`, p90 `0.460812 ms`, best `0.45312 ms`.
- One-buffer candidate: median `0.36742 ms`, p10 `0.365116 ms`, p90 `0.368888 ms`, best `0.36436 ms`.
- Median improvement: `19.2%`.
- Full output numeric error: `0.0` for every measured sample.

This is the first promotion-grade clock-controlled kernel A/B for the reduced-LDS candidate. It is still candidate-only and not model promotion evidence because the one-buffer payload has not been admitted to the production route registry.

## One-buffer registry admission and model-wide stop

The exact Q4_K `qwen3_8b_q4k_m_gfx1100` `attn_qo` `512x4096x4096` one-buffer payload is now eligible for registry selection only when its immutable evidence join is present: the canonical pre-transform identity, zero-spill 20,480-byte LDS compile record, full-output zero-error guarded run, and all ten manual-ROCm-SMI timing samples for the packed identity. Any missing or drifted field declines to the existing two-buffer payload; no other role or buffer-two route is changed.

It remains unpromoted model-wide. The smallest pinned pp512 authority attempt reached memory-capacity preflight but emitted no whole-model report, so there is no producer/body/tail join showing the selected one-buffer identity in the model route census across the actual forward. There is also no same-run whole-model numeric/quality artifact tied to that identity. Those two joins are required before a model-wide promotion claim.
