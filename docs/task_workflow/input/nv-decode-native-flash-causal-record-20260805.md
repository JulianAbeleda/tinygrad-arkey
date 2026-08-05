# Native NV d512 flash causal record

Date: 2026-08-05. Route: Qwen3-8B-Q4_K_M d512, `DEV=NV`, RTX 5090 / driver
595.84. Status: **PASS — overlap is sufficient to explain the localized
+247.989 us flash ownership gap; matched flash-body parity is unproven.**

## Question and cheapest decisive test

The non-quantized role partition assigned +247.989 us of the native-versus-
llama device delta to flash score/combine. That allocation must not be read as
"llama's flash kernels are 247.989 us faster": llama runs those nodes
concurrently with MMQ, while native NV executes the graph stream-ordered.

The smallest decisive test is therefore a three-way interval ledger, not a
new tile sweep:

1. take the calibrated class composition from the exact 948-program native NV
   profile, whose calibrated total is the independently measured marker-light
   native device window;
2. take llama's class interval unions from the exact 762-node CUPTI timeline;
3. take the disjoint Shapley ownership from the non-MMQ union partition, which
   is the only additive exposed-time convention available for overlapping
   llama classes.

The first input is already a live native-GPU measurement. Re-running a
`DEBUG=2` per-kernel timer would be strictly less decisive: that timer changes
launch/timing conditions and was already documented to report 7.52/3.39 us
score/combine versus 305.581 us for the calibrated profile population. The
new CPU ledger is executable and checks the exact settled inputs fail closed.

```bash
PYTHONPATH=. .venv/bin/python extra/llm_research/decode/native_flash_causal_ledger.py \
  --native docs/task_workflow/output/nv-decode-native-semantic-profile-ledger-20260805.json \
  --timeline docs/task_workflow/output/nv-decode-llama-d512-timeline-ledger-20260804.json \
  --partition docs/task_workflow/output/nv-decode-nonquant-role-partition-20260805.json \
  --out docs/task_workflow/output/nv-decode-native-flash-causal-ledger-20260805.json
```

## Result

| population | score us | combine us | total us |
| --- | ---: | ---: | ---: |
| native calibrated, serialized | 204.697 | 100.884 | **305.581** |
| llama raw class interval union | 173.474 | 190.242 | **363.716** |
| llama disjoint exposed ownership | 36.430 | 21.162 | **57.592** |

The raw comparison is `305.581 - 363.716 = -58.135 us`: under these recorded
interval measures, native has *less*, not more, flash execution time. Thus a
putative llama raw-flash speed advantage is not required to explain the
positive ownership gap. Because the captures use different profiling and
fusion contexts, this does not establish matched per-body parity.

The valid additive comparison is `305.581 - 57.592 = +247.989 us`. Llama has
only 57.592 us of flash on its exposed non-MMQ critical ownership; its flash
nodes run for 363.716 us in total, with their time largely overlapped with MMQ
and other work. Native has no such scheduling escape hatch in its present
single-GPFIFO, stream-ordered execution, so all 305.581 us is exposed.

For context, the earlier structural tile sweep remains important but answers
a different question: it found a 7.52 us native score row versus a roughly
3.17 us llama score row in a launch-perturbed in-loop timer, bounding a
possible score-kernel improvement near 0.157 ms node-sum. It does **not**
override the token-critical conclusion above, and its actual wall credit was
never demonstrated. The fused combine had already been at per-row parity.

## Causal conclusion and boundary

**Sufficient mechanism:** llama hides flash under MMQ while native serializes
it. This is sufficient to produce the localized ownership delta and agrees
with the independent overlap and native single-stream evidence. It does not
exclude a smaller body or data-layout advantage under a matched experiment.

**Not proven:** that native flash code is optimal, that zeroing flash would
recover 305.581 us at token wall, or that a tile rewrite is the right fix. A
native rewrite has a ceiling of the 305.581 us serialized population and gets
credit only after a token-identical real-token A/B. The existing tile/geometry
search is a NO-GO for incremental structure tuning; reopening it requires a
new thread-decomposition/staging substrate scope.

**Falsifiable next test:** construct a native multi-queue path capable of
launching flash and an independent MMQ chain concurrently, preserve the
exact dependency edges, then show an interval union reduction of at least
5% and a same-session token-wall reduction. Until native RM channel
construction permits that path, the remaining construction blocker—not a
flash kernel swap—is the precise open item.

## Validation

```text
PYTHONPATH=. .venv/bin/pytest -q \
  test/unit/test_native_flash_causal_ledger.py \
  test/unit/test_native_semantic_profile_ledger.py \
  test/unit/test_nonquant_role_partition.py
6 passed
```

No production code, route, default, or GPU state changed.
