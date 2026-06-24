# Decode q8 two-lane scope - 2026-06-19

Purpose: execute both post-P8 decode lanes without reopening imported Q4 routing.

## Lane 1 - Research Flag Hardening

Question: is the fused q8 FFN artifact route clean enough to keep as a default-off research flag?

Required gates:

- `Q8_FFN_HANDWRITTEN=1` remains default off;
- no in-process HIP runtime;
- reproducible artifact build and fixed-launch HCQ loader pass;
- graph route passes;
- W==D decode speedup is at least `3%` at all measured contexts;
- actual-route dNLL is `<=0.01`;
- artifact dependency and supported shape/arch boundary are documented.

## Lane 2 - Native Transfer Roadmap

Question: is there a bounded tinygrad-native patch to start, or is this a project-level AMD backend effort?

Required gates:

- use the artifact route as the oracle;
- record current native failures;
- confirm no bounded A2 feature exists;
- list the renderer/scheduler capabilities required;
- define the start gate for native work.

## Non-Goals

- no default route change;
- no new kernels;
- no model-wide imported Q4 routing;
- no native compiler implementation unless a bounded feature clears the start gate.
