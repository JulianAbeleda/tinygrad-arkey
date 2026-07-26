# tinygrad-arkey

A hard fork of [tinygrad](https://github.com/tinygrad/tinygrad). AMD/RDNA3 focused, aimed at fast local GGUF LLM inference and at moving the hot path toward generated code. No plans to merge upstream.

## What this is

tinygrad is a small deep learning framework by George Hotz that lowers tensor operations to GPU kernels. This fork narrows it to one job: quantized LLM inference on AMD gfx1100, with the search/audit work split between this runtime and [BoltBeam](https://github.com/JulianAbeleda/BoltBeam).

The main idea is still kernel search, but the bar is strict: a route only becomes default after it is correct, route-bound, rollback-safe, and fast enough under the authority harness. The hot default decode/prefill routes are now generated or spec-driven; owned kernels are rollback/reference material, not the target path.

## How it differs from upstream tinygrad

* **Hardware:** AMD only, currently gfx1100 / RX 7900 XTX.
* **Scope:** quantized GGUF LLM decode and prefill.
* **Search:** BoltBeam audits candidates and route policy; tinygrad runs the promoted routes.
* **Defaults:** generated where proven; owned kernels stay as rollback/reference only.
* **Instrumentation:** decode, prefill, route attribution, purity census, and compiler-lowering gates are first-class.

## Current performance state

Machine: RX 7900 XTX (24 GB), AMD gfx1100. **Prefill rows re-measured 2026-07-24**, **decode rows
re-measured 2026-07-26** — both same-session against llama-bench on the same box and GGUFs.

Decode is now measured by the **fixed-depth authority** (`extra/qk/decode/decode_runtime_overhead.py`,
`tinygrad.decode.fixed_depth.v2`), which prefills to exactly the stated context before timing, so the ctx
columns are real decode depth. The previous table came from `extra/llm/model_e2e_bench.py`, which decodes
from a **one-token seed** over a growing window — its ctx labels described KV *allocation*, not depth, and
its numbers are not comparable to these. llama is `llama-bench -p 0 -n 128 -d 512,4096 -ngl 99 -fa 1 -r 3`
(`-d` = KV depth), same session, auto clock.

**tinygrad leads llama.cpp on 8B decode at both depths, and on 14B at ctx512 — but loses 14B at ctx4096.**
The earlier claim of a lead "across all three models at both contexts" no longer holds: it was measured
against llama's 2026-07-03 14B ctx4096 figure of 54.6 tok/s, and llama's deep-context 14B decode has since
improved to 62.87. Our 14B decays -13.1% from ctx512 to ctx4096 against llama's -5.6%, so **14B
deep-context decode is the clearest remaining decode weakness.**

### Decode, tinygrad vs llama.cpp (2026-07-26, same session)

| Model | Quant | ctx512 | ctx4096 | vs llama | Notes |
|---|---:|---:|---:|---|---|
| Qwen3-8B | Q4_K_M | **113.86** | **102.57** | **+10.3% / +6.0%** | llama 103.20 ± 1.02 / 96.80 ± 0.20 |
| Qwen3-14B | Q4_K_M | **68.39** | **59.41** | **+2.7% / −5.5%** | llama 66.58 ± 0.38 / 62.87 ± 0.05. Loses at depth. |
| Qwen3-32B | Q4_K_M | 31.9 | 32.6 | historical | **2026-07-03, growing-window harness — NOT re-measured, not comparable to the rows above.** |

8B decode is up from a 2026-07-16 baseline of 115.31 / 102.89 after recovering two regressions
(`f705fee2f` kernargs back to device-local VRAM, `9542e82f1` recompute cost gate); ctx512 trades ~1% for a
full prefill recovery. The 07-03 attn_v route-miss fix is unconditional (legacy dispatch pruned in
`08945aa94`).

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
| Qwen3-14B | Q4_K_M | 66.58 ± 0.38 | 62.87 ± 0.05 | **2026-07-26, same session.** ctx4096 was 54.6 on 07-03 — llama improved here, and now beats us at this point. |
| Qwen3-32B | Q4_K_M | 31.2 tok/s | 29.7 tok/s | Decode 2026-07-03 (tg128), historical. Prefill not re-measured (we fall back on 32B). |

Read these as current working numbers, not a universal claim:

- **Decode** — tinygrad leads on 8B at both depths (+10.3% / +6.0%) and on 14B at ctx512 (+2.7%), but
  **loses 14B at ctx4096 (−5.5%)**. Our 14B decays −13.1% over that range against llama's −5.6%; 32B is
  unmeasured since 07-03 and its historical lead should not be assumed to still hold.
- **Prefill** — see the two prefill tables above. 8B and 14B both lead llama at every context; the promoted compiler-generated WMMA-LDS candidate set is the default for its four exact admitted fp16 roles on 8B, and the packed-WMMA candidate route (`TINYGRAD_PREFILL_PACKED_WMMA`, default ON, 6/6 combos gated at `max_abs 0.0`) is the default on 14B. Unsupported shapes use ordinary tinygrad scheduling.
- **Prefill current state** — the canonical route, evidence, rollback, and closed-result ledger live in [docs/prefill-current-state.md](docs/prefill-current-state.md).
- **Prefill on 14B** — live, and it does NOT use the fp16 overlay: 14B's ~29 GB fp16 footprint cannot be admitted on a 24 GB card, so the packed route reads Q4_K/Q6_K storage directly. The older README claim that big models must fall back was only ever true of the *overlay* path. OPEN: 14B end-to-end token parity has not yet been run with this route live.
- **Prefill on 32B** — still falls back to the slow universal path (fp16 overlay ~64 GB, and no packed-WMMA geometry is gated for its shapes). Closing that is a real project.

**Measurement discipline:** report prefill/decode throughput only from the authority harnesses via `extra/qk/bench.py` (below). Never report throughput from a `model.generate` TTFT bench; it includes unrelated Python, sampling, and host overhead.

## Running it

You need an AMD GPU, the AMD backend working, and a GGUF model file. Most current gates assume gfx1100 / RX 7900 XTX. Run from the repo root using the project's virtual environment.

`--max_context` defaults to `auto`: at load it probes free VRAM (`rocm-smi`) and admits the largest safe context for the model (weights + KV + prefill-score peak + flash scratch, held under an 0.8 fragmentation margin), capped at the model's trained context. Admission is a **tier ladder**, all driven by the memory arithmetic (no model-name checks):

1. **fp16 KV (lossless)** — used whenever it admits a useful context. 8B/14B admit their full trained context on a 24 GB card this way.
2. **int8 KV-quant (`DECODE_KV_QUANT`, ~0.6% loss)** — auto-escalated when fp16 can't fit but int8 can. The resident KV cache is int8 + a tiny per-(K/V,head,token) fp16 scale; the decode flash route dequantizes in-register (no materialized fp16 KV). This is what lets **32B run long context** (~2800 tokens where fp16 admits <2000), token-identical to fp16 in practice. It's a *capacity* lever, not a decode speedup (attention decode is compute-bound, not bandwidth-bound).
3. **refuse loud** — only when even int8 can't fit a useful context; the message points at the Q4-KV / eviction tiers (follow-ons).

An explicit `--max_context N` is still admission-checked and auto-upgrades through the ladder to honor the request, failing loud only if no tier fits. All of this relies on the seqlen-bound decode attention route (decode work scales with live context, not `max_context`), so raising the cap does not collapse decode.

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
* `extra/qk/route_manifest.py` — runtime-facing route manifest, rollback flags, provenance, and refuted axes. BoltBeam owns the policy/search copy.
* `extra/qk/decode/flash_decode_attention_spec.py`, `extra/qk/decode/flash_decode_attention_executor.py` — generated flash/decode attention routes.
* `extra/qk/gemv_g3_codegen_lowering.py`, `extra/qk/q6k_route_spec.py`, `extra/qk/prefill/prefill_graph_gemm_route.py` — generated route/runtime surfaces.
* `extra/qk/prefill/packed_wmma_prefill_candidates.py` — the packed-WMMA prefill candidates that are the 14B default (frozen per-(quant,role) geometry + load-time correctness gate).
* `extra/qk/microbench/wmma_peak.cpp` — measured achievable WMMA peak (~105 TFLOPS on gfx1100); use it as the denominator for any efficiency claim.

BoltBeam owns model facts, candidate/search schema, evaluation policy, ledgers, roofline attribution, and reports. tinygrad owns runtime execution, compiler/backend lowering, and hardware gates.

## License

MIT, inherited from tinygrad. See [LICENSE](LICENSE).
