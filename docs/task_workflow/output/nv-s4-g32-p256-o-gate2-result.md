# S4_G32_P256 same-byte O discriminator

## Verdict

`STOP_LANE_A`, with zero recovery booked.

The 144-byte symmetric S4 representation is numerically valid and reaches
essentially the same rotated-cold service as the installed 144-byte Q4_K O
kernel, but it does not improve it. Simplifying Q4_K's affine metadata is not
the missing cold-service lever for this production-shaped body.

This is not a quality qualification and not a production route. The S4
weights were deterministic legal research fixtures, not a converted model
artifact.

## Matched construction

- Shape: 4096 rows by 4096 inputs.
- Control: production-rendered residual-fused FP16-activation Q4_K O kernel.
- Candidate: the same CTA/lane partition, FP16 activation, FP32 reduction,
  residual epilogue, 16 blocks per row, and 144 bytes per 256 weights.
- Candidate block: eight stored FP16 scales followed by signed four-bit
  codes in the existing group-pair/word-column lane map.
- Timing: nine balanced repetitions, hot batches, and a 16-copy weight ring
  for independently rotated-cold calls.

## Correctness

All three legal fixtures passed:

- finite output;
- independent scalar double-precision host oracle with zero failing rows;
- output guard zones unchanged; and
- candidate weights, activation, and residual bytewise unchanged.

The oracle fixtures use binary-exact scale and activation values, which is why
the recorded maximum error is zero despite the candidate's FP32 accumulation.

## Result

| condition | installed Q4_K | S4 candidate | candidate recovery |
| --- | ---: | ---: | ---: |
| hot median | 4.981227 us | 5.153173 us | -0.171946 us |
| rotated-cold median | 9.741 us | 9.762 us | -0.021 us |

The predeclared advancement threshold was +0.15 us per cold call. The
candidate therefore fails the performance gate.

An independent R9 confirmation reproduced the decision: hot recovery was
-0.159893 us and rotated-cold recovery was +0.003 us, again far below the
threshold. The sign of the tiny cold difference is noise around parity; the
absence of a material same-byte recovery is stable.

Both kernels issue 14 global-load instructions and stream the same weight
bytes. The candidate compiles to 368 instructions and 46 registers versus
344 and 43 for control, with no spills in either arm. Its signed-nibble
conversion replaces Q4_K metadata work rather than eliminating the cold
load/dependency floor.

Four cache-controlled NCU repetitions close the apparent primitive/wall
loophole. Individual profiler samples changed sign, but the median duration
was 9.952 us for control and 10.048 us for S4, a 0.096 us candidate debt.
Median measured DRAM traffic was effectively identical at about 9.47 MB.
S4 reduced long-scoreboard share by about 1.9 percentage points, but executed
about 10% more instructions; the lower stall fraction did not translate into
a service win. The complete wall and counter evidence therefore agree.

## Accounting consequence

Gate 0 passes as an accounting substrate and Gate 1 passes as an artifact
substrate. Gate 2 rejects the same-byte service-rate claim.

A smaller representation remains scientifically open, but it is a different
claim: fewer compulsory bytes must pay any decode tax, and the artifact must
come from higher-precision weights or a documented calibration/training
authority. The current Q4_K-to-S4 fixtures cannot establish that quality.

## Reproduction and authority

- Harness: `extra/llm_research/decode/nv_s4_g32_p256_o_microgate.py`
- Machine result: `docs/task_workflow/output/nv-s4-g32-p256-o-gate2.json`
- Independent confirmation:
  `docs/task_workflow/output/nv-s4-g32-p256-o-gate2-confirm.json`
- Cache-controlled counter repetitions:
  `docs/task_workflow/output/nv-s4-g32-p256-o-gate2-ncu.json` and
  `docs/task_workflow/output/nv-s4-g32-p256-o-gate2-ncu-repeat{1,2,3}.json`
- Retained CUDA source, binary, and compiler report:
  `docs/task_workflow/output/nv-s4-g32-p256-o-gate2-artifacts/`
