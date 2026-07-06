# Minimization Principles

The reduced principles for building the **smallest honest inference stack** — a specialized, hard-fork LLM
inference compiler + runtime that owns the whole path from quantized weights to tokens on the GPU.

Governing frame: this is a **hard fork** (no upstream merge). "Upstream keeps it" is never a reason. The sole
workload is LLM inference (Qwen3 Q4_K/Q6_K) on AMD gfx1100 today, with CUDA/NV/Metal as wanted futures. Multi-backend
is future-scope (keep the backends); training/autograd generality is not (delete it).

**One-sentence reduction:** *A minimal primitive set, machine-generated kernels optimized by rules-as-data, compiled
ahead-of-time into a data artifact, and replayed by a tiny hardware-submission runtime — with a hard
authored/generated boundary so the number you carry reflects only what you truly maintain.*

---

## I. Accounting — what actually counts as your code

### 1. Authored vs generated is a hard boundary; only authored counts.
Generated artifacts (FFI bindings, searched kernels, command graphs) are *derived*: mark them (`@generated`), keep them
reproducible from a committed recipe, exclude them from the budget. If it can't be regenerated, it's authored code
hiding as generated.
- GitHub `@generated` linguist convention: https://github.com/github-linguist/linguist/blob/main/docs/overrides.md
- tinygrad autogen (generated excluded from its line budget): https://github.com/tinygrad/tinygrad
- In-repo: `sz.py` (marker-based authored/generated split), `tinygrad/runtime/autogen/`.

### 2. Kernels and graphs are data, not code.
A searched+frozen kernel is a binary blob + launch descriptor; the model is weights + a command list. Data ships; it
does not sit in the authored surface.
- ggml/GGUF (weights + graph as data): https://github.com/ggml-org/llama.cpp
- microTVM AOT executor: https://tvm.apache.org/docs/v0.13.0/how_to/work_with_microtvm/micro_aot.html

### 3. Size is a symptom of honesty, not the goal.
A line budget is a *forcing function* that exposes accidental complexity. Pursue each cut on merit (dead code,
specialization, honest relocation) and let the number land. Cutting muscle to hit a target is the failure mode.
- tinygrad line-count budget as design discipline: https://github.com/tinygrad/tinygrad

---

## II. Architecture — how the system is shaped

### 4. A minimal primitive set; complex ops emerge by composition.
~12–15 primitives (add/mul/reduce/reshape/permute/pad…) compose into flash-attention, conv, MoE. Never add an op you
can compose.
- tinygrad (12 primitive ops): https://github.com/tinygrad/tinygrad
- Luminal (15 primitive ops): https://github.com/jafioti/luminal

### 5. Specialize to the workload; delete the generality you don't run.
A general framework carries training/autograd, every op, every backend. An inference fork on one arch needs a
fraction — remove the rest by a reachability closure from the live entrypoint (`llm/cli.py` -> `model.py`).
- Applied in-repo: autograd/optimizer removal; the inference-only reachability lens.

### 6. Ahead-of-time everything: push work to compile time, leave nothing at runtime.
The graph is fixed; discover, schedule, and optimize once — offline. The runtime should *replay*, never compile.
- Luminal ("a core tenet … pushing everything to compile time"): https://docs.luminalai.com/blog/intro
- microTVM AOT: https://tvm.apache.org/docs/v0.13.0/how_to/work_with_microtvm/micro_aot.html

### 7. Separate compiler from runtime; ship only the runtime.
The compiler + search is offline tooling (like `clang` — you don't ship it). The product is data (weights + kernels +
command graph) plus a tiny runner. This shrinks the shipped surface without deleting capability.
- IREE (compiler -> VM + HAL runtime): https://iree.dev
- ExecuTorch (AOT export -> bytecode VM): https://pytorch.org/executorch
- llama2.c (700-line fixed-arch runner): https://github.com/karpathy/llama2.c

### 8. The floor is the hardware-submission layer.
Everything reducible collapses to "push command packets to the GPU ring + sample." Below that you delete capability,
not code. That is ~2–4k lines, and it is irreducible.
- In-repo: tinygrad HCQ (`tinygrad/runtime/ops_amd.py`, `tinygrad/runtime/graph/hcq.py`)
- llama2.c as the empirical runner floor: https://github.com/karpathy/llama2.c

---

## III. Technique — how to build the compiler small

### 9. Generate kernels; never hand-write them.
The entire order-of-magnitude gap vs a hand-kernel engine is this one choice: a small general generator + search
replaces N hand-authored kernels across ops x quants x backends.
- tinygrad (12 ops compose all kernels): https://github.com/tinygrad/tinygrad
- llama.cpp (~670k LOC, hand kernels per backend — the counter-example): https://github.com/ggml-org/llama.cpp
- In-repo: `docs/pure-machine-search.md`.

### 10. Optimization is rules-as-data + saturation, not hand-coded passes.
Express rewrites declaratively and explore all equivalent forms with an e-graph instead of sequential backtracking —
a smaller optimizer *and* a better search. The principled replacement for ad-hoc search passes.
- egg (equality saturation): https://dl.acm.org/doi/10.1145/3434304
- TENSAT (tensor graph superoptimization via equality saturation): https://arxiv.org/pdf/2101.01332
- DialEgg (MLIR + egglog): https://dl.acm.org/doi/pdf/10.1145/3696443.3708957

### 11. Backends are data-driven from ISA descriptions, not hand-coded per instruction.
The renderer/assembler is generated from machine-readable ISA tables (encodings, operands), so a new arch is data,
not code.
- ACT (generating compiler backends from accelerator ISA descriptions): https://arxiv.org/pdf/2510.09932
- In-repo: `tinygrad/runtime/autogen/amd/{rdna3,rdna4,cdna}` ISA tables + `tinygrad/renderer/amd/generate.py`.

### 12. Don't fight the compiler — decouple it.
The compiler is *why* you're small; the enemy is its runtime coupling (symbolic vars like `start_pos`, buffer
relocation). Sever the coupling so the runner needs zero compiler at runtime; never rebuild the compiler to "escape" it.
- TVM AOT module RFC (removing runtime graph parsing): https://github.com/apache/tvm-rfcs/blob/main/rfcs/0046-module-based-model-runtime-for-aot.md
- MLIR progressive lowering: https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/

---

## Landscape reference (alternatives / prior art)

| Project | What it demonstrates | Link |
|---|---|---|
| tinygrad | minimal general searching compiler (~thousands of lines, 12 ops) | https://github.com/tinygrad/tinygrad |
| Luminal (Rust) | same philosophy, harder AOT, static-linked, 15 ops | https://github.com/jafioti/luminal |
| llama.cpp | the counter-example: hand kernels x many backends -> ~670k LOC | https://github.com/ggml-org/llama.cpp |
| llama2.c | the runner floor: fixed arch, ~700-line C, zero deps | https://github.com/karpathy/llama2.c |
| egg / egglog / TENSAT | rules-as-data optimization + saturation search | https://arxiv.org/pdf/2101.01332 |
| IREE / ExecuTorch / microTVM AOT | industrial AOT -> tiny-runtime blueprints (steal the artifact format) | https://iree.dev |
| ACT | data-driven backend generation from ISA descriptions | https://arxiv.org/pdf/2510.09932 |

## Sizing notes (from the reduction analysis, 2026-07-06)

- Authored budget when this was written: ~26k (target band ~17–21k by specialization; ~pipeline floor ~10–12k with the
  general runtime kept).
- Generated (derived, unbudgeted): ~158k autogen bindings.
- Slim AOT end-state: shipped runtime ~5k (AMD-only) / ~10–13k (multi-backend); the ~17k of pipeline **relocates** to
  offline compiler tooling — it is not deleted.
- The de-risking experiment before any rewrite: **serialize a captured `TinyJit`/HCQ graph and replay it in a fresh
  process without scheduler/codegen loaded** — measures whether the true runtime floor is ~5k or ~10k.
