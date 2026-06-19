# q8 FFN dual-track route + codegen scope (2026-06-19)

This scope splits the q8 side-channel reopening into two complementary tracks. They are **not mutually exclusive**.

- **Track A: handwritten/backend research route.** Purpose: answer model truth first: dNLL, W==D speed, graph/host
  overhead, and failure modes behind a research flag.
- **Track B: tinygrad codegen transfer.** Purpose: make the primitive owned by tinygrad instead of relying on a
  handwritten/backend escape hatch.

Both tracks use the same oracle target already measured in `q8-ffn-handwritten-oracle-scope-20260619.md`.

## Current authority

Q8H-0 through Q8H-5 passed:

| gate | result |
|---|---|
| real-GGUF handwritten Q4_K x q8_1 `ffn_gate/up` correctness | PASS, max_abs <= 1.91e-6 |
| fused RMSNorm + q8 side-channel producer cost | PASS, incremental 0.92us |
| gate+up lifecycle vs current fp coop | PASS, 1.23x |
| decode EV model | PASS_TO_Q8H6, ~1.05x decode |

So the remaining question is no longer "does the lifecycle work?" It is:

1. Does q8-lossy gate/up pass quality?
2. Can we route it in-model without losing the measured economics?
3. If yes, should the route stay handwritten research-only, or should tinygrad learn to generate the producer/consumer?

## Track A — handwritten/backend research route

### A0 — quality proxy before route build: PASS

Run teacher-forced dNLL where the normal graph remains in place but dense FFN `gate/up` see
`q8_1_dequantize(q8_1_quantize(ffn_norm_output))`.

This does **not** measure performance. It isolates the q8 activation quality risk before spending on HCQ in-model
routing. Probe: `extra/q8_ffn_quality_proxy.py`.

Gate:

- `dNLL <= 0.01` vs the same baseline evaluator and token window.
- If this fails, stop Track A and Track B for this q8 route; the speed oracle is not a model route.

Artifact:

- `bench/q8-ffn-handwritten-oracle/quality_proxy.json`.

Result:

| window | baseline NLL | q8 proxy NLL | dNLL | verdict |
|---|---:|---:|---:|---|
| 32-token sanity | 3.38016 | 3.38437 | +0.00421 | PASS |
| 160-token gate | 2.85548 | 2.85712 | **+0.00165** | **PASS** |

The proxy is quality-only, not a speed route. It proves the q8 activation loss on FFN gate/up is below the current
`dNLL <= 0.01` quality threshold, so the handwritten route is worth building to W==D.

### A1 — HCQ-launchable handwritten artifacts: PASS

Convert the two handwritten HIP kernels into tinygrad-loadable AMD code objects:

- fused RMSNorm/q8 producer: one output fp activation + q8 blocks;
- Q4_K x q8_1 MMVQ consumer: gate/up real-GGUF shape.

No in-process HIP runtime. EBT-1 proved HIP and tinygrad HCQ are mutually exclusive in-process, so this must use the
same class of HCQ code-object launch used by the Tensile extraction work.

Gate:

- HCQ eager launch writes tinygrad-owned buffers correctly.
- No host/device copies beyond existing model buffers.
- Reproduces Q8H-1/Q8H-3 correctness.

Executed as `extra/q8_ffn_hcq_artifact.py`. The probe compiles one raw AMD code object per kernel through
tinygrad's COMGR compiler and launches through tinygrad AMD HCQ. It deliberately uses the raw tinygrad kernel dialect
instead of HIP runtime headers: `__attribute__((device))` helpers, explicit shared storage, raw half conversion, and
no in-process HIP calls.

Artifacts:

- `bench/q8-ffn-handwritten-oracle/hcq_artifact.json`
- `bench/q8-ffn-handwritten-oracle/hcq_artifact_up.json`

| tensor | producer fp max_abs | q8 dequant max_abs | consumer max_abs | verdict |
|---|---:|---:|---:|---|
| `blk.0.ffn_gate.weight` | 4.77e-7 | 0.01165 | 7.15e-7 | PASS |
| `blk.0.ffn_up.weight` | 4.77e-7 | 0.01165 | 1.43e-6 | PASS |

This proves the handwritten producer and consumer are HCQ-loadable artifacts on tinygrad-owned GPU buffers. It does
not yet prove model routing or graph capture; it retires only the "can these kernels be loaded without HIP in-process?"
risk.

### A2 — one-block eager route behind research flag: PERF_FAIL / REDIRECT

Route one dense FFN block:

`ffn_norm side-channel -> q8 gate/up -> silu*up -> current ffn_down`

behind `Q8_FFN_HANDWRITTEN=1`, default off.

Gate:

- block output max_abs vs graph q8 proxy within tolerance;
- eager device-time speedup >= modeled threshold after route overhead;
- no default behavior change.

Executed as `extra/q8_ffn_oneblock_route.py` on block 0 using a real post-attention hidden state. The route launches
the HCQ producer, HCQ gate consumer, HCQ up consumer, then feeds `silu(gate) * up` into the existing `ffn_down`.

Artifact:

- `bench/q8-ffn-handwritten-oracle/oneblock_route.json`

| check | result | verdict |
|---|---:|---|
| route output vs graph q8 proxy max_abs | 0.00137 | PASS |
| route output vs graph q8 proxy mean_abs | 4.44e-5 | PASS |
| q8 proxy vs fp FFN max_abs | 0.00327 | quality context |
| HCQ producer + gate + up lifecycle | 195.0us | FAIL |
| modeled lifecycle target with 20% slack | <=129.2us | FAIL |

The correctness route is real, but the HCQ-loadable COMGR artifact does not preserve the HIP oracle economics.
The producer is also slower in-model because the route casts the fp16 RMSNorm weight to the producer's current
`float*` contract. A hipcc-compiled code object was tested as a recovery path and compiled successfully, but
`AMDProgram` rejected it with `unknown AMD reloc 10`; it is not yet loadable by tinygrad's current HCQ program loader.

**Redirect before A3:** do not graph-capture this slow COMGR artifact. First recover a hipcc/clang-quality
HCQ-loadable artifact, or change the producer/consumer to a tinygrad-owned raw-ASM/codegen form that matches the HIP
oracle timing. Forward scope: `q8-ffn-fast-artifact-and-codegen-transfer-scope-20260619.md`.

Follow-up result: `q8-ffn-fast-artifact-vs-raw-code-result-20260619.md` executes that redirect. The hipcc/LLD
artifact path passes when gate/up are fused into one shared-q8 lifecycle primitive (`114.12us`), while the current
COMGR/raw-code path remains slow (`194.80us`).

### A3 — TinyJit/HCQGraph route: PASS AFTER CONTRACT AUDIT

Make the route graph-capturable so W==D decode can measure it without per-token Python overhead.

Gate:

- graph replay stable across tokens;
- no pointer-staleness from q8 side-channel buffers;
- W==D ctx sweep >= 3% sustained speedup.

Executed follow-up: `q8-ffn-fast-artifact-a3-route-result-20260619.md`.

Initial result:

- eager one-block fast-artifact route **PASS**: `121.38us`, correct vs q8 proxy, no HIP runtime;
- Tensor-visible runtime-cache injection first faulted because placeholder `ProgramInfo` optimized away artifact input
  buffers.

Corrected result:

- `extra/q8_ffn_injection_contract_audit.py` verifies the `ProgramInfo.globals/outs/ins` contract without executing
  artifact runtimes;
- after fixing the placeholder contract and real Q4_K dtype/shape key, eager injected node and TinyJit replay both
  **PASS** vs the q8 proxy (`max_abs 0.00137`);
- W==D decode is now the next gate.

### A4 — final quality/perf verdict

Run:

- dNLL on the accepted token window;
- W==D decode at banked contexts;
- fallback sanity with flag off.

Verdict:

- **PASS:** research route is valid and becomes the target for Track B.
- **QUALITY_FAIL:** stop; do not build codegen.
- **PERF_FAIL:** keep the kernels as oracle assets only; do not route.

## Track B — tinygrad codegen transfer

Track B starts from the same target but does not depend on Track A being shipped. Track A gives it measured target
behavior.

### B0 — capability contract

Add a formal contract for a "lifecycle producer" kernel:

- one per-row reduction over 4096 for RMSNorm sumsq;
- barrier/LDS broadcast of `rinv`;
- per-32 max/scale/quantize;
- multi-output stores: fp normalized activation + q8 block payload/scales;
- graph-visible q8 side-channel object passed to exactly gate/up.

Gate:

- contract documented with shapes, dtypes, lifetimes, aliasing rules, and quality boundary.

### B1 — custom-kernel expressibility spike, second pass

Revisit Q8L-2 with the handwritten oracle as target. Try to express the producer using current UOps plus explicit
`Ops.BARRIER`/local storage instead of the store-group-only idiom.

Gate:

- one generated kernel, not per-stage kernels;
- UOp verification passes;
- DEBUG source contains one producer with barrier/local staging and multi-output stores.

Kill:

- if current custom-kernel UOps still cannot express the two reduction granularities in one kernel, stop B1 and go to
  B2.

### B2 — renderer/codegen capability

If B1 fails, add the minimal renderer/codegen feature needed for this family:

- range model for staged reductions with barriers;
- multi-output stores after a staged reduction;
- stable scheduling around `DEFINE_LOCAL`, `BARRIER`, and post-barrier stores.

Gate:

- small unit test: fused RMSNorm/q8 producer standalone matches the handwritten producer;
- no broad scheduling regressions.

### B3 — generated consumer parity

The existing tinygrad q8 consumer is correct but slower than the handwritten oracle. Decide whether Track B owns only
the producer or also the llama-style consumer.

Options:

- B3a: generated producer + existing q8 consumer. Lower engineering cost, likely smaller speedup.
- B3b: generated producer + generated/ported 128-thread llama-style q8 MMVQ. Higher cost, target is Q8H-4 1.23x.

Gate:

- isolated gate+up lifecycle >= 1.15x over fp coop, same as Q8H-4.

### B4 — model integration

Only after B2/B3 pass:

- add an internal q8 side-channel dataflow for dense FFN gate/up;
- keep env gated and default off;
- run dNLL and W==D.

Gate:

- same as Track A A4.

## Decision matrix

| outcome | action |
|---|---|
| A0 quality fails | stop both tracks; q8 side-channel remains an oracle-only speed trick |
| A0 passes, A route fails perf | keep handwritten kernels as proof; Track B only if producer capability has other uses |
| A route passes, B not started | acceptable as research flag only, not default |
| A route passes, B passes | tinygrad-owned primitive can replace handwritten route |
| B producer passes but consumer lags | route only if W==D still clears >=3%; otherwise keep as codegen asset |

## Why both tracks are useful

Track A answers whether the primitive is worth having in the model. Track B answers whether tinygrad can own the
primitive cleanly. Running Track A first lowers risk for Track B: it prevents building a compiler feature for a route
that later fails dNLL.
