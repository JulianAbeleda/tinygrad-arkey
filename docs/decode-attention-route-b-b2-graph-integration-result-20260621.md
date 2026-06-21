# Route B B2 — HCQ graph/AQL launch-integration de-risk: result

Date: 2026-06-21

Executes **Route B B2 only** of `docs/decode-attention-route-a-route-b-full-execution-scope-20260621.md`. B1 found the
vendored llama `flash_attn_tile`+combine **wins 2.96× by GPU-busy** through tinygrad's HCQ but **loses ~2.5× by wall**
because it was launched as **two raw `HCQProgram` dispatches** (2 doorbells, 2 kernarg fills, 2 submits). B2 answers one
question: **does the GPU-busy win survive once tile+combine are folded into ONE HCQ compute queue** (one doorbell,
kernargs baked once, replayed) — i.e. does the integrated **wall** beat `gqa_coop_vec` by ≥1.5×?

## Decision: **`B2_LOCAL_GRAPH_PASS` — the GPU win SURVIVES launch integration (integrated wall 1.65× faster than coop). Route B remains viable → proceed to B3 (owned hand-AMDGCN).**

Folding the two dispatches into one **bound** HCQ queue (one submit/doorbell, kernargs baked, queue body persistent via
`bind()`) drops the wall from **148 µs (B1, 2 raw dispatches) → ~36 µs** — now **1.65× FASTER** than `gqa_coop_vec`
(reproduced 1.64–1.68×), with GPU-busy ~3.6× and unchanged correctness. The B1 launch-overhead penalty is **recovered**
by graph integration. W==D is NOT run here — for the *vendored* kernel it is layout-blocked (below); W==D is B3's job.

## Audit (B2.1): how the batching works — no `tinygrad/` change needed
tinygrad's HCQ compute queue (`tinygrad/runtime/support/hcq.py:143` `exec`, `:230` `submit`; AMD
`tinygrad/runtime/ops_amd.py:333`/`437` `exec`, `:420`/`468` `_submit`, `:694` `signal_doorbell`) already supports
**N execs → one submit → one doorbell**. `HCQGraph` (`tinygrad/runtime/graph/hcq.py`) does exactly this generalized
over a whole JIT graph: one queue per device (`:51`), kernargs baked once (`:33,:39-42`), `bind()`-ed once (`:217`),
replay = patch sints + re-submit (`:263`). A **raw external-`.co` `AMDProgram`** can be driven straight through
`q.exec(prg, args_state, gs, ls)` — it reads only `prg.{prog_addr,rsrc1/2/3,group/private_segment_size,
kernargs_segment_size,aql_prog_addr,enable_dispatch_ptr,wave32}` (all set by `NamedAMDProgram`) — **no `ExecItem`
machinery required**. tile→combine serialize with **no barrier between them**: the AMD `exec` emits `CS_PARTIAL_FLUSH`
at the end + `acquire_mem` at the start (PM4), or relies on the AQL barrier bit (`AQL_HDR`, `ops_amd.py:33`).

## Prototype (B2.2): `extra/qk_llama_flash_attn_tile_hcq_graph_b2.py`
Reuses B1's `build_replay()` (tile+combine `NamedAMDProgram`s, baked kernargs, buffers, numpy ref). Kernargs baked once
into dedicated persistent buffers. Two replay variants:
- **REBUILD** — queue rebuilt python-side each call (`wait().memory_barrier().exec(tile).exec(comb).signal().submit()`),
  one doorbell but the python queue-build is re-paid each call.
- **BOUND** — queue built + `bind()`-ed once; replay = reset a dedicated completion signal + re-submit the persistent
  IB. This is the HCQGraph-ideal a real model-JIT integration gets (near-zero per-call host overhead).
Default queue mode is **PM4** (`AMDComputeQueue`) — the same path `gqa_coop_vec` uses, so the A/B is apples-to-apples.

## Measurements (B2.3) — clock-pinned, @ctx1024 decode shape (KV 1280-padded, parallel_blocks=20)
| metric | llama BOUND (integrated) | llama REBUILD | gqa_coop_vec | ratio (bound) |
|---|---|---|---|---|
| **wall (the B2 gate)** | **~36 µs** | ~131 µs | ~60 µs | **1.65× llama-faster** |
| GPU-busy | 12.4 µs | 12.5 µs | 44.7 µs | 3.60× |
| correctness vs numpy | rel_max 1.3e-3 (fp16) | — | — | OK |
| B1 reference (2 raw dispatches) | — | — | — | wall 148 µs → 0.39× |

Per-replay launch accounting (bound path): **dispatch_count=2, doorbell_count=1, signal_count=3** (2 profile + 1
completion), **host_sync_count=1, graph_replay_count=1**. (B1 was 2 doorbells / 2 kernarg fills / 2 submits.) Artifact:
`bench/qk-decode-attention-route-b-b2/latest.json` (stamped; all required fields).

## Gates
| gate | result |
|---|---|
| correctness rel_rmse ≤ 5e-3 (fp16-class; B1 threshold, exact-1e-3 not meaningful for fp16 K/V) | **PASS** (1.2e-3) |
| GPU-busy stays ~B1 and clearly faster than coop | **PASS** (12.4 vs 44.7 µs, 3.6×) |
| **integrated wall ≥ 1.5× faster than `gqa_coop_vec` @ctx1024** | **PASS** (1.65×, reproduced) |
| policy guard | PASS |
| `git diff tinygrad/` zero | **PASS** (zero) |

## Why W==D is NOT run here (and is B3's job)
Local graph integration passed, which *unlocks* W==D — but a W==D for the **vendored** kernel is **layout-blocked**:
the captured kernarg encodes llama's exact ggml KV-cache byte layout (`nb11=2048`: `[pos][8 kv-heads][128]` fp16),
which the **tinygrad model's KV-cache does not produce**. Routing it in-model would require either per-call
layout-bridging (a transpose/repack kernel that negates the win) or deep model-cache surgery + per-shape kernarg
re-capture — both **out of B2's bounds** ("fixed ctx1024 before generalization") and the vendored kernel is
**non-promotable** anyway. So W==D belongs to **B3's owned kernel**, authored to tinygrad's KV layout, which is
promotable AND can be W==D'd in-model without bridging. No `model.py` route was added (boundary honored).

## Classification & route decision
**`B2_LOCAL_GRAPH_PASS`.** External `.co` kernels **CAN** enter HCQ graph batching (not `BLOCKED_BY_GRAPH_INTEGRATION`),
and once batched the **wall clearly wins** (not `FAILS_GRAPH_ECONOMICS`). The launch-lifecycle bottleneck B1 exposed is
**closed**: the 2.96× GPU win becomes a 1.65× *wall* win under one-doorbell graph integration. **Route B remains the
leading path after launch integration → proceed to B3 (owned hand-AMDGCN/HSACO tile)**, where an owned, promotable
kernel authored to tinygrad's layout is the right vehicle for the in-model W==D.

## Boundaries honored
Vendored kernel **non-promotable**, `default_eligible=false`; no defaults changed; no B3/Route-A work started; **no
`model.py` route, zero `tinygrad/` diff**; HIP Graphs used only as design reference (the route is tinygrad HCQ); no
benchmark headline from local wall; `gqa_coop_vec` comparator SSOT; no closed-lane reopen.
