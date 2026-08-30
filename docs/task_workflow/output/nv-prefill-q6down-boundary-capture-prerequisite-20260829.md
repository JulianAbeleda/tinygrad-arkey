# NVIDIA pp512 Q6-down boundary-capture prerequisite

## Verdict

**BLOCKED.** The four boundaries are specified, but no packet-local capture
seam has been added or executed yet. The unavailable boundary is **all four**:
compact-Q8 producer, Q6 main, output publication, and rank-preserving residual
epilogue. D1 remains closed.

## Purpose and frozen target

This packet is measurement-only. It must explain the existing Q6-down
regression without changing arithmetic, geometry, model routing, queue policy,
or the installed promotion record.

- Model: Qwen3-8B Q4_K_M, pp512/ubatch512, RTX 5090 sm_120.
- Population: exactly 18 real `ffn_down` Q6_K roles, `(512,4096,12288)`.
- Candidate: existing compact-Q8 producer plus Q6 main route.
- Control: identical graph with resident FP16 down.
- Processes: fresh process per arm and per hot/rotated-cold condition.
- Samples: synchronized R9 wall samples; retain every sample, not only median.

## Required seam

Add a packet-local capture mode to the existing
`extra/llm_research/prefill/nv_compiler_q6k_imma_gate.py`,
`nv_compiler_q6k_model_arm.py`, and
`nv_compiler_q6k_pp512_binding.py` paths. The mode must preserve the current
18-role population and expose four cumulative forced cuts:

1. compact-Q8 producer only;
2. Q6 main only, with producer included in its dependency chain;
3. Q6 main plus output publication;
4. Q6 main plus publication plus rank-preserving residual epilogue.

Each cut must have a distinct begin/end marker at the actual submitted work,
not a host-side estimate. Markers must be associated with role identity and
queue/dependency identity, and must survive graph replay. The capture record
must contain device timestamps, queue-ready/dependency timestamps, launch and
completion sequence, and host observation time.

## Census contract

For every cut and arm record:

- allocations and releases: count and bytes, including workspace and output;
- device-to-device and host-device copies: count and bytes;
- materializations, contiguous conversions, and views crossing the cut;
- queue readiness and dependency wait intervals;
- publication interval and residual-epilogue interval separately;
- exact role count, record count, canonical weight-base count, and zero-copy
  status;
- hot and rotated-cold labels, process identity, git/source hashes, and all
  environment knobs.

The harness must fail closed if any marker is absent, duplicated, reordered,
or attributed to a non-real role. A complete-model wall number without these
records is not a boundary measurement.

## Measurement matrix

Run the four cuts against their corresponding FP16 controls in both conditions:

| condition | candidate | control | required output |
|---|---|---|---|
| hot | all four cuts | matching FP16 cuts | per-sample markers and census |
| rotated-cold | all four cuts | matching FP16 cuts | per-sample markers and census |

Use paired process arms where possible. Report minimum, median, MAD, and every
sample. Reject a boundary attribution unless its independent candidate/control
delta exceeds both `0.25 ms` and `3x` pooled MAD in the same condition. If
boundaries overlap, report the interval as shared exposure and do not assign a
dominant mechanism.

## Acceptance gates

PASS requires all of the following:

- all four markers are independently observable in hot and rotated-cold arms;
- all 18 real roles are present in every candidate arm;
- allocation, copy, materialization, queue, and device-time fields are
  populated or explicitly zero;
- logits/replay correctness is unchanged;
- no route, arithmetic, geometry, or promotion-policy files changed;
- evidence includes raw records and a generated summary.

Any missing field or boundary is BLOCKED with its exact name. A PASS only
authorizes D1 attribution work; it does not authorize an arithmetic redesign.

## Closed lanes

Do not change `tinygrad/`, compiler arithmetic, Q6 K16 correction, launch
geometry, model routing, promotion records, or production defaults. Do not
repeat Q6 geometry sweeps. Do not infer a producer/main/publication/epilogue
cost from the existing whole-model `+1.611334 ms` delta.

## Handoff

The next agent should implement the seam in packet-local harness/evidence
files, run the complete matrix on an idle benchmark GPU, and replace this
BLOCKED verdict only when the raw capture satisfies the contract above.
