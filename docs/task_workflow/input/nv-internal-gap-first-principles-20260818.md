# NV decode internal-gap resolution: first-principles (2026-08-18)

Date: 2026-08-18
Branch: `nvidia-bringup-20260731`, HEAD `fc3c6e718`
Status: **resolved.** The ~1060 us "internal gap" in the profiled NV wall
ledger is a PROFILE=1 artifact of the HCQGraph per-replay host instrumentation,
not production GPU idle. The wall equation and every prior audit verdict hold.

## 1. The question

The exact wall account (`nv-240-exact-wall-account-20260817.md`, residual
0.0) says `wall = GPU busy (union) + host gap`, with tinygrad `4788.3 =
4519.3 + 269.0`.  But the same harness's own ledger reports
`internal_gap_us = span - union ≈ 1060-1105 us` (19% of the profiled span).
llama's CUPTI ledger has an internal gap of only **8.2 us** (one 762-node
graph replay per token).  If that ~1060 us were real idle, it would be the
biggest single wall term and the audits would have closed the wall "in
llama's manner" without explaining it.  That is the loose end this note
resolves.

## 2. The finding

The ~1060 us internal gap is **not production GPU idle**.  It is the
PROFILE=1 profiler's per-replay host work at the 4 graph-replay boundaries
that the NV JIT creates inside every token.

### 2.1 Proof by wall arithmetic (same route, byte-identical tokens)

Same-session A/B of the production control route (d512, Qwen3-8B-Q4_K_M,
RTX 5090, token sha `548ff2c...` on both arms):

| arm | median wall us/token |
| --- | ---: |
| PROFILE=0 (production) | 4820.0 |
| PROFILE=1 (ledger harness) | 6275.7 |

The PROFILE=1 GPU span from the ledger at the same runtime code is
**5577.4 us** (union 4514.8 + internal gap 1059.9).  A GPU timeline cannot
span longer than the wall of the same route: `5577.4 > 4820.0`, so the
profiled span is inflated by the profiler, not by idle.  The ~1456 us
PROFILE wall tax covers the ~1060 us internal gap plus the inter-token gap
inflation.

### 2.2 The gap sits entirely at replay boundaries

Every steady token is exactly 5 graph replays with the entry-count signature
`(32, 64, 128, 256, 114)`.  Decomposing one steady token's span:

| replay | entries | GPU span us | boundary gap after (us) |
| --- | ---: | ---: | ---: |
| 0 | 32 | 176.5 | 293.5 |
| 1 | 64 | 438.5 | 311.0 |
| 2 | 128 | 786.2 | 561.5 |
| 3 | 256 | 1839.5 | 2.2 |
| 4 | 114 | 1273.8 | (token end) |

Within-replay gaps are exactly **0.0 us** for every consecutive kernel pair
in all 6 steady tokens; the entire internal gap is the 4 inter-replay gaps.
The last gap (~2 us) is ~0 because the 114-entry vocab replay's host work
finished while the 256-entry replay was still on the GPU.

### 2.3 The mechanism (code)

`HCQGraph.__call__` (`tinygrad/runtime/graph/hcq.py`) does, on every call:

```python
self.kickoff_value += 1
for dev in self.devices: self.last_timeline[dev][0].wait(self.last_timeline[dev][1])
if PROFILE and self.kickoff_value > 1:
  self.collect_timestamps()   # builds + appends the full per-kernel JSON payload
  self.collect_pmc()
```

The wait is unconditional (signal re-arm safety) but instant in production
(it targets the same graph object's previous token, already complete).  Under
PROFILE=1 the per-call `collect_timestamps` builds and disk-appends a JSON
payload of every kernel (`start`/`end`/`duration` strings plus the full deps
list, up to 256 entries and 911 deps) inside the same serialized host path.
That work scales with replay size and matches the measured gap pattern
(260 us before the 64-entry replay, 290 before 128, 510 before 256).
llama is profiled with nsys CUPTI, which inserts nothing into the graph, so
llama's internal gap (8.2 us) is real and ours was not comparable.

## 3. What is real: the 5-replay per-token structure

The 5-replay split is not a dependency or a graphable-kernel barrier: the
admission census (all 872 calls admitted, all 4 boundaries
`batch_size_limit`) shows it is the JIT `JIT_BATCH_SIZE` heuristic
(`tinygrad/engine/jit.py`): `max_batch_size` starts at 32 and doubles after
each flush.  The steady decode replay sizes are (32, 64, 128, 256, 114);
llama runs one 762-node CUDA graph replay per token.

Measured production (PROFILE=0) per-replay host cost (same route, same token
sha, steady tokens 4-10):

| replay | entries | host submit us (median) |
| --- | ---: | ---: |
| 0 | 32 | ~32 |
| 1 | 64 | ~27 |
| 2 | 128 | ~33 |
| 3 | 256 | ~42 |
| 4 | 114 | ~27 |

Per-token host submission sum: **~160-175 us** (5 submissions).  llama pays
one submission (~30 us).  The exact account's host-gap delta
(269.0 vs 168.3 = +100.6 us) is consistent with this structure: our 4 extra
replay submissions (~+120-135 us) offset by a slightly cheaper tail sync.
The L6 submit-ahead result (-6.2 us) is consistent too: reordering across the
token boundary does not remove the per-replay submission cost.

## 4. Status

- Internal-gap row: **CLOSED** as profiler artifact.  Production GPU runs
  back-to-back kernels (`union == node_sum == span ≈ 4515-4519 us`, zero
  overlap), and `wall = union + host` holds with the host term measured.
- The three-front audits (schedule, reduce_output, vocab_aux) verdicts are
  unchanged; nothing was closed "in llama's manner" that this finding
  reopens.
- Real per-token structure vs llama: 5 graph replays + 4 host submissions vs
  llama's 1 replay + 1 submission.  Direct A/B (`JIT_BATCH_SIZE=1024`,
  token-identical): **1 replay is +112.9 us slower** (median wall 4924.4 vs
  4811.5 us), even though its single host submission is cheaper (~120-144 us
  vs ~160-175 us).  The GPU-side single-replay execution is ~+150 us slower,
  so the 5-replay split is already the cheaper configuration and the
  replay-merge lever closes measured-negative.  (The single-graph route also
  crashes under PROFILE=1, consistent with the profiler's per-kernel signal
  cost growing with graph size.)
- The +628.8 us busy delta (llama hides 1125 us behind its mmq anchor; we run
  zero overlap) remains the largest structural lever and is
  CONSTRUCTION-REQUIRED (PDL/overlap substrate), measured wall-neutral on the
  real route.

## 5. Reproduction

- Wall A/B: `scratchpad/nv_decode_wall_node_ledger.py --mode child-inproc`
  with `PROFILE` 0 vs 1; tokens identical across arms.
- Gap decomposition: `/tmp/nv_decode_wall_ledger_head2/replays.jsonl` +
  the interval math in `extra/llm_research/decode/cuda_graph_timeline_ledger.py`.
- llama internal gap: `cuda_graph_timeline_ledger.py --trace
  /tmp/llama_d512_20260817b.sqlite --graph-id 6` (span 3899.5, union 3891.3,
  internal gap 8.2 us).
- Admission census: `observe_graph_admissions` on the capture token
  (all boundaries `batch_size_limit`, batches 32/64/128/256/392).
- JIT batch A/B: `JIT_BATCH_SIZE` env 32 vs 1024 on the same child route.

## 6. Evidence

- `docs/task_workflow/evidence/nv-profile-tax-ab-head-20260818.json`
- `docs/task_workflow/evidence/nv-jit-batch-size-ab-head-20260818.json`
