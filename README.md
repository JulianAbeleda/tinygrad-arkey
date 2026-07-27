# tinygrad-arkey

A hard fork of [tinygrad](https://github.com/tinygrad/tinygrad). AMD/RDNA3 focused, aimed at fast local GGUF LLM inference and at moving the hot path toward generated code. No plans to merge upstream.

## What this is

tinygrad is a small deep learning framework by George Hotz that lowers tensor operations to GPU kernels. This fork narrows it to one job: quantized LLM inference on AMD gfx1100, with the search/audit work split between this runtime and a private kernel-search tool.

The main idea is still kernel search, but the bar is strict: a route only becomes default after it is correct, structurally route-bound, fail-closed outside its admitted shapes, and fast enough under the authority harness. The hot decode/prefill routes are generated or descriptor-driven. Fallback availability is shape-specific; an unsafe requested fallback fails loudly rather than pretending to be a rollback.

## Why this is machine search even though the runtime is static

The search happens **offline**, not while a model is serving tokens. A private search/audit tool and the
BubbleBeam/Futuresight path expand structured kernel candidates, ask tinygrad to lower them, and use
correctness, route-attribution, resource, and measured-speed gates to reject or promote each result. The
promoted candidate is then frozen as a descriptor, schedule, or generated UOp lowering in this repository.

```text
model + target facts
  -> structured candidate grammar
  -> machine-expanded candidate set
  -> tinygrad lowering and compilation
  -> correctness + attribution + performance gates
  -> promoted candidate frozen into a deterministic route
  -> normal tinygrad compilation and cached GPU execution
```

Freezing the winner is deliberate. It makes a regression attributable, keeps the exact generated program
inspectable, avoids search and benchmark noise during startup, and provides a stable rollback target. Static
deployment does not make the discovery manual: the candidate provenance and measured selection process are
what make it machine search.

At runtime there is no claim of unrestricted program synthesis or continuous autotuning. Model execution
selects an already promoted route, reconstructs its UOp kernel, and hands it to tinygrad's normal
codegen/renderer/compiler path. The resulting cached GPU binary is the repeated hot path. Some generated
route surfaces currently live under `extra/qk` so their provenance and debugging artifacts remain explicit;
production route adapters may import them while constructing the kernel, but the offline search itself is not
rerun for inference.

In short: **machine-searched offline, statically promoted, compiled by tinygrad, deterministic at runtime.**

## How it differs from upstream tinygrad

* **Hardware:** AMD only, currently gfx1100 / RX 7900 XTX.
* **Scope:** quantized GGUF LLM decode and prefill.
* **Search:** a private tool audits candidates and route policy; tinygrad runs the promoted routes.
* **Defaults:** generated or descriptor-owned where proven; research candidates stay off the production path.
* **Instrumentation:** decode/prefill authority, route attribution, purity census, and compiler-lowering gates are first-class. Decode graph-PMC attribution remains open work.

## Current performance state

Machine: RX 7900 XTX (24 GB), AMD gfx1100. **Decode re-measured from promoted master on 2026-07-27**;
the last complete four-point prefill curve is from **2026-07-24**, with newer bounded 14B endpoint checks
reported separately below. Comparisons use the same box and GGUFs.

Decode is now measured by the **fixed-depth authority** (`extra/qk/decode/decode_runtime_overhead.py`,
`tinygrad.decode.fixed_depth.v2`), which prefills to exactly the stated context before timing, so the ctx
columns are real decode depth. The previous table came from `extra/llm/model_e2e_bench.py`, which decodes
from a **one-token seed** over a growing window — its ctx labels described KV *allocation*, not depth, and
its numbers are not comparable to these. The current llama depth authority is `llama-bench -p 512 -n 40
-d 512,4096 -ngl 99 -r 3 -o json` (`-d` = fixed KV depth), auto clock.

**tinygrad leads llama.cpp on 8B decode at both depths and on 14B at ctx512. At 14B ctx4096 it is now in
practical parity: 62.45 tok/s versus llama's 62.55, a −0.15% median difference inside run variance.** The
G=5 QG2/S32 route plus descriptor-owned width-4 K/V staging recovered the previous deep-context deficit.

This does **not** mean the slope is fixed. Tinygrad still decays about 10.4% from ctx512 to ctx4096 against
llama's 5.6%; it reaches parity at depth because it starts about 5.2% faster. The earlier 59.41 tok/s 14B
ctx4096 result is the pre-promotion baseline, not the current result. The older llama 54.6 figure remains
withdrawn because its method was not recorded and it does not reproduce.

### Decode, tinygrad vs llama.cpp (current promoted master)

| Model | Quant | ctx512 | ctx4096 | vs llama | Notes |
|---|---:|---:|---:|---|---|
| Qwen3-8B | Q4_K_M | **114.19** | **103.07** | **+10.7% / +6.5%** | llama 103.20 ± 1.02 / 96.80 ± 0.20; tinygrad master 2026-07-27. |
| Qwen3-14B | Q4_K_M | **69.70** | **62.45** | **+5.2% / −0.15%** | llama 66.27 ± 0.29 / 62.55 ± 0.30; practical parity at depth. |

At ctx128, production `auto` stays on SDPA and completed at 94.80 tok/s on 8B and 59.89 tok/s on 14B. The
accelerated flash route begins at the configured threshold. The 14B flash candidate owns QG2, S32, and
width-4 cooperative staging structurally; 8B remains on its G=4 width-1 descriptor.

**tinygrad-arkey prefill** (`extra/qk/prefill/prefill_whole_synced.py --mode authority`, whole-prefill tok/s):

| Model | pp512 | pp1024 | pp2048 | pp4096 | decay 1024→4096 |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B Q4_K_M | **3694** | 3624 | 3485 | **3236** | −10.7% |
| Qwen3-14B Q4_K_M | **1945** | 1920 | 1875 | **1785** | **−7.0%** |

**llama.cpp prefill** (`llama-bench -p 512,1024,2048,4096 -n 0 -ngl 99 -fa 1`, same session, same GGUFs):

| Model | pp512 | pp1024 | pp2048 | pp4096 | decay 1024→4096 |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B Q4_K_M | 3347 ±242 | 3410 ±7 | 3317 ±2 | 3158 ±17 | −7.4% |
| Qwen3-14B Q4_K_M | 1845 ±86 | 1817 ±6 | 1758 ±2 | 1642 ±9 | −9.6% |

**Margin (ours vs llama, same session):**

| Model | pp512 | pp1024 | pp2048 | pp4096 |
|---|---:|---:|---:|---:|
| Qwen3-8B | +10.4% * | +6.3% | +5.1% | **+2.5%** |
| Qwen3-14B | +5.4% * | +5.7% | +6.7% | **+8.7%** |

**Newer bounded 14B prefill endpoint authority (2026-07-27):** pp512 measured 2026 / 2029 / 2023 tok/s
(median **2026**) and pp4096 measured 1880 / 1880 / 1878 tok/s (median **1880**). These confirm that the
promoted default is faster than the four-point 2026-07-24 row, but they are not substituted into that
same-session curve or its llama margins. The latest 8B pp512 route-regression smoke was 3768 tok/s; it is a
single bounded check, not a replacement authority row.

\* **The pp512 column is soft.** llama's pp512 is its own least reliable point on this box: ±242 t/s (7%) on
8B, and its curve is non-monotonic there (pp1024 3410 > pp512 3347). A cross-session check found pp4096
reproduced to −0.1% and pp2048 to −0.6%, while pp512 drifted −6.3%. Trust pp1024–pp4096; treat pp512 as
indicative.

Context decay is quoted from **pp1024**, not pp512, because decay inherits the reliability of its baseline
endpoint and pp512 is the unreliable one here. The structural floor for causal attention over 1024→4096 is
**−5.3%** (tokens grow O(S), causal attention O(S²); see
[docs/prefill-roofline-first-principles-20260724.md](docs/prefill-roofline-first-principles-20260724.md)).
**14B at −7.0% is near that floor and better than llama's −9.6%. 8B at −10.7% vs llama's −7.4% is the
clearest remaining prefill weakness.**

Sensitivity, stated because it is large: measured from pp512 instead, the same data reads 8B −12.4% vs llama
−5.6% and 14B −8.2% vs llama −11.0%. And had llama's 8B pp512 landed at its prior-session 3571 rather than
tonight's 3347, llama's own figure would be −11.6% — worse than ours. Any decay claim anchored on pp512 on
this box should be treated as indicative only.

### llama.cpp reference

| Model | Quant | Decode ctx512 | Decode ctx4096 | Notes |
|---|---:|---:|---:|---|
| Qwen3-8B | Q4_K_M | 103.20 ± 1.02 | 96.80 ± 0.20 | **2026-07-26, same session** (`-d 512,4096`). |
| Qwen3-14B | Q4_K_M | 66.27 ± 0.29 | 62.55 ± 0.30 | **2026-07-27, same-session depth authority.** Statistically tied with tinygrad at ctx4096. |

32B is omitted from the current tables because it has not been measured with the fixed-depth decode authority or the current prefill authority. Its old growing-window numbers are not comparable.

Read these as current working numbers, not a universal claim:

- **Decode** — tinygrad leads on 8B at both depths (+10.7% / +6.5%) and on 14B at ctx512 (+5.2%). At 14B
  ctx4096 it is effectively tied (−0.15% by the reported medians). Tinygrad's 14B slope is still steeper,
  10.4% versus llama's 5.6%; matching llama's depth slope remains open.
- **Prefill** — the complete four-context tables above are dated 2026-07-24 evidence, not a fresh sweep of current `master`. Newer bounded 14B endpoint checks improved to 2026/1880 tok/s at pp512/pp4096. The compiler-generated WMMA-LDS candidate set serves its exact admitted 8B roles, while the correctness-gated packed-WMMA route is the admitted 14B path. Unsupported shapes use ordinary tinygrad scheduling.
- **Prefill history** — [docs/prefill-current-state.md](docs/prefill-current-state.md) is a 2026-07-24 campaign ledger. It contains superseded intermediate conclusions and is not the authority for current route behavior or current performance.
- **Prefill on 14B** — live, and it does NOT use the fp16 overlay: 14B's ~29 GB fp16 footprint cannot be admitted on a 24 GB card, so the packed route reads Q4_K/Q6_K storage directly. The older README claim that big models must fall back was only ever true of the *overlay* path. OPEN: 14B end-to-end token parity has not yet been run with this route live.

### Production route selection

Prefill and decode use the same policy contract: `auto`, an explicitly selected accelerated candidate, or
the ordinary fp16 fallback. Candidate discovery remains phase-specific; lifecycle and selection semantics
are shared in `tinygrad/llm/route_selection.py`.

| Phase | Canonical variable | Values | Default behavior |
|---|---|---|---|
| Prefill | `TINYGRAD_PREFILL_ROUTE` | `auto`, `packed_wmma`, `direct_packed`, `fp16` | `auto` selects the promoted route admitted by structural facts; forcing an unsafe quarantined route fails loudly. |
| Decode | `TINYGRAD_DECODE_ROUTE` | `auto`, `flash`, `fp16` | `auto` uses SDPA for shallow context and the admitted flash candidate at the configured threshold. |

The legacy `TINYGRAD_PREFILL_PACKED_WMMA` and `FLASH_DECODE` variables remain compatibility aliases. They do not promise that every model has a usable implementation for every forced mode. New automation should use the canonical variables above. Invalid or quarantined selections fail loudly.

Route ownership is structural rather than model-name based:

- 8B's resident fp16 projections own dense tensor-core warmstarts only after `_pf16_w` overlays are realized.
- 14B's packed-only projections do not receive dense warmstarts; this prevents shape-key collisions with
  unrelated reductions. Its promoted packed-WMMA route remains available.
- The Hq=40/Hkv=8 direct-packed implementation is quarantined after repeatable GPU MMU faults. The quarantine
  applies only to that implementation, not to the shared structural binding used by packed-WMMA.
- `auto` never dispatches the quarantined direct implementation. Forcing it fails during model setup rather
  than after GPU submission.

Validated on 2026-07-27 on RX 7900 XTX/gfx1100: actual ctx128 completed on SDPA for both models; promoted
master decode measured 114.19 / 103.07 tok/s on 8B and 69.70 / 62.45 on 14B at ctx512 / ctx4096; 14B
prefill endpoint medians were 2026 at pp512 and 1880 at pp4096. The 8B warmed pp512 control completed at
3768 tok/s. The single 8B prefill control remains a bounded route-regression check, not a replacement for
the multi-run prefill authority table.

**Measurement discipline:** report prefill/decode throughput only from the authority harnesses via `extra/qk/bench.py` (below). Never report throughput from a `model.generate` TTFT bench; it includes unrelated Python, sampling, and host overhead.

## Running it

You need an AMD GPU, the AMD backend working, and a GGUF model file. Most current gates assume gfx1100 / RX 7900 XTX. Run from the repo root using the project's virtual environment.

`--max_context` defaults to `auto`: at load it probes free VRAM (`rocm-smi`) and admits a safe context from weights, KV, prefill peak, and flash scratch under the configured safety margin, capped at the model's trained context. Admission is a **tier ladder** driven by memory arithmetic and structural support rather than model-name checks:

1. **fp16 KV exact context** — preferred whenever the requested or automatic context fits.
2. **int8 KV exact context** — considered when fp16 cannot fit and the model geometry plus decode route support a quantized cache. The resident cache is int8 with per-(K/V, head, token) fp16 scales; supported flash routes dequantize in-register. Quality and performance require model-specific evidence.
3. **fp16 ring streaming** — an explicitly lossy, bounded physical window for unbounded logical generation when streaming is allowed and the geometry supports it. It is not used to silently satisfy an explicit exact-context request.
4. **refuse loudly** — when no admissible exact tier fits, or a requested streaming mode is unsupported.

An explicit `--max_context N` is admission-checked and may select a supported exact tier to honor the request, failing loudly if none fits. Decode route selection is based on live context and admitted structural facts rather than allocating work merely because `max_context` is large.

```sh
# THE benchmark — the single canonical entry. Dispatches to the synced authority harnesses (prefill + decode),
# each in an isolated subprocess with the correct env. This is the ONLY sanctioned way to report throughput.
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk/bench.py --model /path/to/Qwen3-8B-Q4_K_M.gguf
#   --prefill  : prefill authority only (extra/qk/prefill/prefill_whole_synced.py, synced graph-GEMM pp@L)
#   --decode   : genuine fixed-depth decode (exact prompt prefill, production generate route, repeated medians)
```

**Do not roll your own throughput harness, and never report a `model.generate` TTFT number** — TTFT understates prefill by ~3× (it folds in generate's Python overhead + sampling + host jitter). If you need a route/purity check instead of a number, start with:

```sh
PYTHONPATH=. .venv/bin/python extra/audit/pure_machine_search_default_path_census.py --check
PYTHONPATH=. .venv/bin/python extra/audit/pure_machine_search_default_path_census.py --strict-final-default
```

## Main Files

Start with these files and the documentation map in [docs/README.md](docs/README.md):

* `tinygrad/llm/` — the core runtime (command line, model, model-file loader).
* `extra/qk/decode/decode_harness.py` — decode speed across context lengths.
* `extra/qk/prefill/prefill_whole_synced.py` — prefill speed.
* `extra/audit/pure_machine_search_default_path_census.py` — current generated/default-route census.
* `extra/qk/route_manifest.py` — candidate registry and provenance input. Runtime policy validates selected manifest rows rather than treating the manifest global as semantic authority.
* `extra/qk/decode/flash_decode_attention_spec.py`, `extra/qk/decode/flash_decode_attention_executor.py` — generated flash/decode attention routes.
* `extra/qk/gemv_g3_codegen_lowering.py`, `extra/qk/q6k_route_spec.py`, `extra/qk/prefill/prefill_graph_gemm_route.py` — generated route/runtime surfaces.
* `extra/qk/prefill/packed_wmma_prefill_candidates.py` — the packed-WMMA prefill candidates that are the 14B default (frozen per-(quant,role) geometry + load-time correctness gate).
* `extra/qk/microbench/wmma_peak.cpp` — measured achievable WMMA peak (~105 TFLOPS on gfx1100); use it as the denominator for any efficiency claim.

A private tool owns search/audit adapters, interchange schemas, evaluation policy, ledgers, roofline attribution, and reports. tinygrad owns runtime model facts, route admission and execution, compiler/backend lowering, and hardware gates. Profiler adapters do not imply that decode hardware-counter attribution is complete.

## License

MIT, inherited from tinygrad. See [LICENSE](LICENSE).
