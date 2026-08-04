# ExecutableTaskGraph IR - NVIDIA decode-first replay/overlap substrate (AMD/Metal lowering seams)

Date: 2026-08-03 (revised 2026-08-03: NVIDIA-first reframe + parity verdict)

Status: scoped, not implemented. This is a design/analysis pass responding to the
proposal to introduce an abstraction "like an ExecutableTaskGraph, not a CUDAGraph"
(kernel/copy nodes, buffer read/write sets, true dependency edges, reusable
shape/signature, lowered per backend). Branch boundary: tinygrad
`nvidia-bringup-20260731` at `f2270480c`. This document does not authorize
implementation, promotion to `dev`/`exp`/`master`, or any code change.

This revision is NVIDIA-first by decision: the measured parity gap lives on NV decode
(sm_120), so the scope's P0 and the only speed deliverable (D2) target NVIDIA. AMD and
Metal are kept as lowering seams only: the IR design must not block them, but no AMD
or Metal lowering work is in scope. Shared runtime changes still require an AMD control
(house rule).

Bans for this scope: no code changes; no GPU use in this doc-only pass; never touch
`extra/llm_research/microbench/dp4a_peak_cuda*` or `scratchpad/t6_metal_admission_probe.py`;
never commit to `master`/`dev`/`exp`. The GEMV efficiency scope
(`decode-gemv-efficiency-forward-scope-20260803.md`) and the host-overhead scope
(`b3-prefill-host-overhead-scope-20260803.md`) stay separate: this scope is graph
topology and replay substrate, not kernel efficiency and not host launch cost.

---

## 1. Verdict on the proposal

Agree with the shape, with one honest correction: **most of this abstraction already
exists in tinygrad, and the proposal is a formalization of existing `GraphRunner`
infrastructure into an explicit IR, not a greenfield build.** The "not a CUDAGraph"
framing is already true today: `GraphRunner` is backend-agnostic, and `CUDAGraph` is one
lowerer among three (`CUDAGraph`, `HCQGraph`, `MetalGraph`).

What the proposal names, mapped to what exists:

| proposal element | exists today | where |
| --- | --- | --- |
| kernel/copy nodes | `GraphRunner.calls`: `(dev_idx, ast, bufs, device_vars)` per call | `tinygrad/engine/jit.py:381-394` |
| buffer read/write sets | per-call `bufs` list + `w_dependency_map`/`r_dependency_map` with sub-buffer ranges | `jit.py:353-377` (`DepsTracker`) |
| true dependency edges | `DepsTracker.access_resources` computes range-aware RAW/WAR/WAW edges | `jit.py:358-377`; consumed by `CUDAGraph` (`runtime/graph/cuda.py:44`) and `HCQGraph` (`runtime/graph/hcq.py:262-267`) |
| reusable shape/signature | `var_vals_replace`, `launch_dims_replace`, `symbolic_dims`; per-replay param update via `cuGraphExecKernelNodeSetParams`; `graph_cache` weak-keyed on the AST UOp | `jit.py:396-415`, `cuda.py:67-68`, `tinygrad/engine/realize.py:119-123` |
| lower per backend | `CUDAGraph` (cuGraph), `HCQGraph` (queue/signal based), `MetalGraph` (MTLIndirectCommandBuffer), `NullGraph` | `tinygrad/runtime/graph/{cuda,hcq,metal}.py`, `runtime/ops_null.py` |
| fallback = serialized execution | `run_linear` + `graph_cache` miss path | `realize.py:264`, `realize.py:119-123` |

One place the proposal is *less* precise than the existing code: it says "buffer
read/write sets", but `DepsTracker` already handles sub-buffer ranges (offset/end per
base buffer), which is what makes WAR/WAW correct under suballocation. Any IR we design
must keep range-awareness; whole-buffer edges would be a regression.

The genuinely new work is therefore narrow and specific:

- **D1 - explicit IR.** Today the DAG is implicit: `GraphRunner` holds calls + a deps
  tracker, `CUDAGraph` holds its own `CUgraph` node list, and nothing exposes the
  nodes/edges/signature as first-class data for analysis, census, or cross-backend
  fidelity. Formalizing `Linear` -> `ExecutableTaskGraph` (nodes, range-based RW sets,
  edges, signature) makes the graph inspectable and reusable by all lowerers.
- **D2 - dependency-driven batching beyond consecutive calls.** `graph_split_rewrite`
  (`jit.py:236`) only batches *consecutive* admissible calls; it flushes on mixed
  device, unsupported op, batch-size limit, or explicit barrier. Independent kernels
  separated by unrelated work never share a graph. llama's overlap (section 2) is
  exactly this: independent nodes inside one graph.
- **D3 - overlap semantics.** Whether our replayed graphs run independent nodes
  concurrently is unmeasured (section 4). This is the load-bearing unknown; it decides
  how much of D2 is worth building.
- **D4 - signature/reuse rules.** When is a captured graph reusable vs re-instantiated?
  The machinery exists; the rules are not written down or censused.
- **D5 - non-NV lowerers.** AMD (`HCQGraph`) and Metal (`MetalGraph`) lowerers exist and
  stay as seams; Vulkan/OpenCL are not in this fork (section 6).

The proposal's value is not the abstraction itself, it is what the abstraction
unlocks: a dependency-ordered schedule we can reason about, census, and batch
non-consecutively. Everything else in the proposal is already shipped.

## 2. The evidence that makes this a decode parity lever (OBSERVED)

Decode is already graph-replayed and is NOT host-bound:

- The flash-decode rollout replays **6 graph groups per token** (`batched
  32/64/128/256/512/29` = 1021 programs/token, `DEBUG=2` trace); GPU busy is
  5.83 ms of the 6.12 ms wall = **95% busy**.
  (`nv-performance-campaign-scope-20260801.md:624-626`)
- The decode gap is 1.44-1.52x behind llama at d512-d4096 with 46% of measured memory
  bandwidth used (824 GB/s of 1792 GB/s).
  (`nv-decode-parity-final-20260802.md`; `nv-performance-campaign-scope-20260801.md:630`)

llama's graph overlaps independent nodes:

- llama tg graph: graphId 2, 762 nodes, 29 replays; node-sum 5.006 ms but replay
  duration is ~22% below node-sum; independent nodes on one stream run concurrently
  (`nv-decode-gap-decomposition-record-20260803.md:36-53`).
- The record's section 6 explicitly states: *"whether tinygrad can express equivalent
  overlap through its own graph is not established here."*
  (`nv-decode-gap-decomposition-record-20260803.md:107-109`)

Prefill is a second consumer of the same substrate:

- The warm pp512 pass already runs **8 replayed graph groups**; wall 44-46 ms vs GPU
  busy 24.1 ms; first replay's 1.9 s is one-time graph instantiation.
  (`b3-prefill-host-overhead-scope-20260803.md:26-29`)

So the substrate question for decode is not "do we replay?" (we do) but "**does our
replay overlap independent nodes like llama's does?**" That is P0 below.

## 3. Will this take us to parity? No, not alone - the verdict and the five evidenced levers

Honest verdict: **this scope does not take decode to parity by itself.** D1/D4 are
substrate (maintainability + census, zero speed). D2 is the only speed claim, and it
attacks one piece of a four-piece gap. Decode is 95% GPU busy, so the wall is kernel
execution; the d512 wall gap of 1.67 ms/token decomposes (INFERRED from OBSERVED rows,
`nv-decode-gap-decomposition-record-20260803.md:24-32`) into a GEMV like-for-like
delta (~0.50 ms, near its ~0.6 ms cap) and a sequential non-GEMV tail (~1.2 ms).
llama closes its version of the tail by overlap (22% of node-sum hidden); our
opportunity there is limited because our non-GEMV classes are a stream-serialized
tail with real dependencies.

Parity requires the five levers below, in evidence order. Each row carries its
OBSERVED basis, its ceiling, and the isolated same-session A/B that proves the wall
conversion (the kv-store fusion already proved that kernel-sum savings do NOT
automatically convert to wall: 948 -> 948 kernels, wall-neutral 177.8 vs 178.4 tok/s,
`decode-kv-store-chain-fusion-scope-20260803.md`). "Proven" here means: measured
ceiling exists and the conversion experiment is defined, not that the wall win is
already banked.

| # | lever | evidence (OBSERVED) | ceiling | proof experiment |
| --- | --- | --- | --- | --- |
| 1 | Graph-level overlap of independent nodes (this scope's D2) | llama runs 22% below node-sum (5.006 ms sum vs ~3.89 ms replay span); mechanism is inside its CUDA graph | overlap-able fraction of our 6.02 ms decode node-sum; llama's 22% is the upper reference | P0 nsys `--cuda-graph-trace=node` on our replay, then DAG-grouping A/B at d512 |
| 2 | GEMV per-kernel efficiency (dp4a-class achieved bandwidth) | parity record's verdict: residual 1.4-1.5x is per-kernel GEMV efficiency (llama `mul_mat_vec_q` vs our q4k/q6k family); ours 824 GB/s = 46% of 1792 GB/s | like-for-like ~0.5 ms of the 1.67 ms gap (near its 0.6 ms cap); substrate items L2 Q6K partial ~0.25 ms, L4 vocab ~0.14-0.24 ms, flash tile ~0.16 ms (node-sum upper bounds) | per-item diagnostic microbench + isolated d512 wall A/B (`decode-gemv-efficiency-forward-scope-20260803.md`) |
| 3 | Decode kernel-count parity (1021 vs llama's 762; llama has zero add/residual kernels) | kv-chain remainder q-rope+q-cast ~2.5 kernels/layer (~90 kernels); rmsnorm 145 kernels / 431 us at 2/layer vs llama's 1; residual add + ffn 144 kernels / 252 us; vocab scatter 5 kernels / 383 us | ~380 us kernel-sum across the listed classes | one fusion per class, each with the same-session d512 A/B; kv-store precedent says expect wall-neutral until proven otherwise |
| 4 | Depth-side KV read scaling (context growth) | flash score kernel grows 7.78 -> 32.4 us/median d512 -> d4096; ours -14.2% tok/s vs llama -9.3% over that range; fp16 cache dtype is already landed capability-based (`model.py:838-841`, `kv_cache_fp16_eligible`) | the depth-penalty delta (-14.2% vs -9.3%) | same-session d4096 wall run on the fp16-cache route vs fp32 control |
| 5 | Prefill host-side replay (B3) | warm pp512 wall 44-46 ms vs busy 24.1 ms = 1.9x vs llama's 1.15-1.35x envelope; wall-minus-busy residual ~20-22 ms across 10 submits (OBSERVED arithmetic, cause pending instrumentation) | pp512 gap to llama (ours 0.77-1.05x at pp512-4096 today) | B3's same-run poll/submit instrumentation, then whole-schedule graph replay A/B |

Ranking basis: the order is **expected wall impact** (measured ceiling weighted by
conversion confidence), not measured ceiling alone, because the ceilings are not on
one scale. Lever 1's ceiling is genuinely unknown until P0 (llama's 22% is a
reference; the decomposition explicitly says our overlap opportunity is limited
because the non-GEMV classes are a sequential tail). Levers 2-3 are node-sum upper
bounds whose wall conversion is unproven (the kv-store precedent: kernel-sum savings
were wall-neutral). Lever 4 is context-side; lever 5 is a different regime (prefill).
A strict ceiling-only order would put fusion (lever 3) ahead of overlap (lever 1) -
the sequential tail is ~1.2 ms of the 1.67 ms gap - but overlap leads here because P0
is cheap and decisive and it is this scope's deliverable. If P0 reports no overlap,
levers 1 and 3 swap in a revision.

Lever 1 is this scope's deliverable; levers 2-4 are existing or adjacent scopes
(GEMV efficiency scope, fusion scopes, fp16-cache landed); lever 5 is the B3 scope.
They are additive: the GEMV delta and the sequential tail are separate wall pieces, so
closing only one cannot reach parity. This scope must not absorb levers 2-5; it must
produce the topology facts (edges, census, P0 overlap) that make each of their A/Bs
cheaper to size.

## 4. P0 characterization - does our decode replay overlap? (HARD prerequisite)

Nothing is designed, and no claim of a speed path is made, until this measurement
exists. It mirrors the method already used on llama
(`nv-decode-gap-decomposition-record-20260803.md:36-39`): `nsys profile
--cuda-graph-trace=node` on our decode rollout.

Deliverables, all OBSERVED with session/commit/config recorded:

- **P0.1** per-node times for each of the 6 decode groups; node-sum vs replay wall per
  group (the llama 22% comparison).
- **P0.2** concurrency: max simultaneous nodes, overlap fraction per group, and which
  node classes overlap (e.g. residual/add behind GEMV chain, like llama's hidden
  classes).
- **P0.3** the admission census cross-check: which of the 1021 programs are
  consecutive-chain members vs siblings with no true dependency, so we know the
  theoretical grouping ceiling before building anything.

Gates that follow from P0:

- If our groups already overlap at llama-like levels -> the topology is not the decode
  lever; D2 is demoted to "census/cleanup", and the doc is revised before any
  implementation.
- If our groups serialize (node-sum ~= replay wall) -> D2 is the decode lever, and the
  scope proceeds to the design below with a measured target.

The prefill side gets one cheaper row: whether the 8 prefill groups' nodes overlap
(same `--cuda-graph-trace=node` on a warm pp512 run). It does not gate decode work but
it decides whether D2 also serves B3's schedule.

## 5. IR design (proposed; contingent on P0)

### 5.1 The object

`ExecutableTaskGraph` as explicit data, produced by the existing builder and consumed
by per-backend executors:

- nodes: `(kind: KERNEL|COPY, device, program/ast, params: {var_vals, launch dims},
  reads: [(buffer, range)], writes: [(buffer, range)])`
- edges: RAW/WAR/WAW, computed with the `DepsTracker` range algorithm as the canonical
  edge builder (reuse, don't rewrite)
- signature: the reuse key - semantic facts (role, shapes/dtypes, device, target),
  symbolic var_vals, and buffer identities as slots, not concrete allocations

The IR itself is backend-agnostic (nodes/edges/signature carry no backend types), so
AMD and Metal lowerers can consume it later without design change; this scope only
builds and exercises the NV lowering.

### 5.2 Builder and census stay

`graph_split_rewrite` + the typed admission census (`GraphAdmissionObservation`,
reasons, observer) remain the construction path; the IR is its output, not a
replacement. Any new grouping (D2) must keep the census: every kernel keeps its
admission reason and assignment (direct/graph/batch member), and the census gains the
edge information (in-degree, sibling count) that makes the overlap question
answerable from a file instead of a profiler.

### 5.3 Executors

One executor per backend over the same IR. The serialized fallback becomes an executor
too (`run_linear` semantics), so "fallback" is a lowering choice, not a separate code
path. CUDAGraph keeps its per-node param update (`cuGraphExecKernelNodeSetParams`) as
the reuse mechanism. HCQGraph (queue/signal schedule) and MetalGraph (ICB replay) are
**not touched** in this scope; they remain compatible seams by construction (5.1).

### 5.4 Design constraints

- No new subsystem: the IR lives beside `GraphRunner` in `tinygrad/engine/jit.py`
  territory; lowerers stay in `tinygrad/runtime/graph/*`.
- Byte-identity on replay: no reordering that changes outputs; range-aware edges are
  mandatory, whole-buffer edges are banned.
- No kernel changes, no dtype changes, no host-launch work (separate scopes).
- NVIDIA-first: AMD/Metal lowerers are seams, not deliverables; shared runtime changes
  still require an AMD control (house rule, same as B3).

## 6. Backend lowering table - proposal vs in-fork reality (NV in scope, rest seams)

| proposed lowering | in-fork reality | status in this scope |
| --- | --- | --- |
| CUDA -> `cudaGraphExec` | `CUDAGraph` (`runtime/graph/cuda.py:10`): cuGraphCreate/AddKernelNode/AddMemcpyNode/Instantiate/Launch; deps from `DepsTracker` (`cuda.py:44`); per-replay `setParams` (`cuda.py:67-68`) | in scope (the D2 target) |
| HIP -> `hipGraphExec` | `HCQGraph` (`runtime/graph/hcq.py:26`): queue-based with kickoff/timeline signals, multi-queue copies, RDMA, PMC capture - **not** hipGraphExec | seam, out of scope; default is NO hipGraphExec (HCQ is the canonical AMD lowering in this fork) |
| Metal -> indirect/reusable commands | `MetalGraph` (`runtime/graph/metal.py:52`): MTLIndirectCommandBuffer, ICB offset admission, hybrid replay | seam, out of scope |
| Vulkan -> command buffers | no Vulkan backend in this fork (`mesa` appears only as an autogen import in `ops_nv.py:13`) | future; the typed admission census is the extension seam |
| OpenCL -> command buffer + sync | no OpenCL backend in this fork | future |
| fallback -> serialized | `run_linear` (`realize.py:264`); `graph_cache` miss (`realize.py:119-123`) | formalize as an executor over the IR (in scope, low risk) |

## 7. Signature and reuse semantics (D4)

Current state: `graph_cache` is weak-keyed on the graph AST UOp (`realize.py:119`),
symbolic var values and launch dims update per replay (`jit.py:396-415`), buffer slots
bind via PARAM uops (`jit.py:389`), and instantiation is the expensive one-time step
(1.9 s prefill first replay, OBSERVED).

The scope formalizes:

- **signature** = semantic facts + symbolic var_vals + buffer slot identities (shape,
  dtype, device, role, target facts). Concrete allocations are parameters, never part
  of the key.
- **reuse rule**: same signature -> per-replay param update; changed signature ->
  re-instantiate. The rule must be written down because the cost asymmetry (update vs
  instantiate) is the whole point of the substrate.
- **census**: record reuse hit/miss per replay so a re-instantiation regression is a
  test failure, not a mystery slowdown.

Open question for review: how shape change is classified (symbolic var update vs
structural re-capture) for the prefill schedule, where concrete shapes dominate and
llama-style dynamic shapes are the exception.

## 8. Dependency-driven batching (D2, post-P0, NVIDIA-first)

`graph_split_rewrite` flushes on: mixed device, unsupported call op, batch-size limit
(`JIT_BATCH_SIZE` default 32, `helpers.py:240`), and explicit graph barriers
(`jit.py:258-269`). Nothing groups non-consecutive independent calls.

The proposed shape: a DAG scheduler over the IR that collects all calls, computes
range-aware edges, and groups maximal independent admissible subgraphs into shared
graphs, preserving the census and the existing admission gates. This is the only piece
of the proposal that can capture llama's ~22% overlap, and only if P0 shows our replay
serializes. It is exercised on NV (CUDAGraph) first; the scheduler itself is
backend-agnostic so HCQGraph/MetalGraph can adopt it later without rework.

Design questions for review (not answered here):

- Does the scheduler replace `graph_split_rewrite`'s scan or run as a post-pass over
  the captured linear?
- Do grouping decisions change with device state (memory pressure), and does the
  census record the reason either way?
- Are cross-group barriers preserved (the explicit-barrier path exists for a reason)?

## 9. Guardrails

- No code changes from this document; implementation requires a separate scope after
  review and after P0.
- Never commit to `master`/`dev`/`exp`; branch `nvidia-bringup-20260731` only.
- AMD control required for any shared runtime change (jit/graph layer), even under the
  NVIDIA-first framing.
- Keep B3 (host overhead), GEMV efficiency (kernel side), and fusion scopes untouched;
  do not fold their levers into this one. Lever 1 is this scope's only deliverable.
- Evidence classes only: OBSERVED / INFERRED / SCOPED; P0 numbers carry session,
  commit, and config. Do not name a cause from an accounting residual.
- `git diff --check` clean on any future commit.

## 10. Deliverable

This document plus the analytical verdict in the hand-back message. Commit as `[docs]`
on `nvidia-bringup-20260731`. No push unless requested. HARD STOP here for review.

## 11. One-line job

Formalize the existing GraphRunner/DepsTracker/backend lowerers into an explicit
ExecutableTaskGraph IR whose dependency edges and reuse signature are first-class
data, after measuring (P0) whether our NVIDIA decode graphs already overlap
independent nodes.
