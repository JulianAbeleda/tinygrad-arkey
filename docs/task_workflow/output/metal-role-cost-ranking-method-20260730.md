# Metal role-cost ranking method

Date: 2026-07-30

Status: MR0 audit plus MR7 methodology only. No hardware was executed and no performance result is claimed.

## Decision

BoltBeam commit `00ec7d4fdbf9b1d81b0dde02243a3054c4c70720` contains a numerically consistent repeated
baseline, but it does not satisfy every MR0 closure gate in
`metal-replay-generated-route-parity-scope-20260729.md`. Its timing rows remain usable as historical baseline evidence;
its `status=complete` and README statement that MR0 is complete are not supported by the retained bundle alone. The
scope-compatible verdict is:

```text
MR0_NUMERIC_BASELINE_VALID
MR0_CLOSURE_INCONCLUSIVE
MR7_BLOCKED_ON_MR0_REPAIR + MR4 + COMPLETED_MR5
```

MR7 must not rank work from program labels, label counts, evenly apportioned graph time, packed-weight byte share, or
the earlier isolated `ffn_gate_up` search. It will join tinygrad-owned semantic identity to MR1 call placement, measure
the exact current generated program through the existing provider, and let BoltBeam apply one centralized eligibility
and ranking rule. At most two role families may enter MR8.

## MR0 evidence audit

### Audited authority

The baseline commit's parent is BoltBeam `9b0e7e9cf33eed1813e3abbf31e4a2b942f09fa0`, the implementation revision
pinned by the run plan. The compact sources at `00ec7d4fdbf9b1d81b0dde02243a3054c4c70720` are:

| Artifact | Git blob |
| --- | --- |
| `bench/metal-qwen3-8b-replay-baseline-20260730/baseline-summary.json` | `585b72698175b83c758e1dced329e911151effc2` |
| `bench/metal-qwen3-8b-replay-baseline-20260730/interleaved-run-plan.json` | `7e60a4263585e57e91f71cdd34facebb82615348` |
| `bench/metal-qwen3-8b-replay-baseline-20260730/llama-samples.json` | `85e522d473ce897e326153ddfa0c0d167211f603` |
| `bench/metal-qwen3-8b-replay-baseline-20260730/runs/tinygrad-01/authority.json` | `1d7b12a37b9436f009b73a78bcd5d096e0273213` |
| `bench/metal-qwen3-8b-replay-baseline-20260730/runs/tinygrad-02/authority.json` | `8abee1f9f2097f890629c565332b38e8b5a52098` |
| `bench/metal-qwen3-8b-replay-baseline-20260730/runs/tinygrad-03/authority.json` | `268c78c4f5ac9351cda22ec912108e58a39c49d1` |
| `bench/metal-qwen3-8b-replay-baseline-20260730/runs/tinygrad-04/authority.json` | `b3a11e397216f796cd0c0447250b53b6ab06ec4f` |
| `bench/metal-qwen3-8b-replay-baseline-20260730/runs/tinygrad-05/authority.json` | `b6a59cceda85d9006a7db32d4bd96bcf0ba92532` |

### Values that reconcile

The five accepted tinygrad authorities reproduce the summary arrays exactly. Their wall samples are 89.187333,
89.026000, 87.835625, 88.257125, and 88.814875 ms; median 88.814875 ms. Their throughput samples are 11.212355,
11.232674, 11.384902, 11.330530, and 11.259375 tok/s; median 11.259375 tok/s and full-range dispersion
1.532480% of the median.

All five tinygrad rows also agree on:

- schema `tinygrad.decode.fixed_depth.v2`, fixed depth 128, one measured decode token, and two warmup decode steps;
- `METAL / MetalDevice`, SDPA, 803 programs per token, and no flash-decode route;
- model path, 5,027,783,488-byte size, mtime, and metadata-identity hash
  `f1e12816d0c29c5952432385bb0bf3b89aeb67825fca6635340acd7fc10134e8`;
- prompt evidence hash `7d25501b7d71ac141ff4cc67ce92a0279e6ac98f1f7b5151629d30bd2fe84431`;
- final token id 33235 and generated-token evidence hash
  `e2df3a607b2e942c63c19887f3a220fee4cfa787c955b93e38524e7d1e68b245`.

The five llama rows reproduce the summary arrays exactly. Their median is 49.662875 ms and 20.135765 tok/s; their
full-range dispersion is 2.156258% of the median. The summary's tinygrad/llama throughput ratio
0.5591729536876738 and wall ratio 1.7883554867608555 recompute exactly.

The GGUF currently at the recorded path hashes to
`d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`, matching the summary and llama artifact.
This read-only recheck confirms the present file; it does not retroactively add a content hash to each tinygrad row.
The pinned tinygrad subtree is also internally valid:
`8139f6725c2e96f66a76515ad6ae5dc50354243c:tinygrad` resolves to
`05b2bf7e1391433bcca4b758e546d03a86481553`.

### Mismatch and missing closure evidence

| Class | Finding | Consequence |
| --- | --- | --- |
| literal mismatch | The run plan and README specify one warmup per runtime; all five tinygrad authorities record `warmup_decode=2` and the retained command passes `--warmup-decode 2`. | The declared protocol does not equal the executed tinygrad protocol. Clarify separate process/model warmups if that distinction was intended. |
| missing cross-runtime evidence | Tinygrad records a final token id/hash. `llama-samples.json` records timings only, so output identity cannot be matched across runtimes. | The MR0 cross-runtime output-identity gate is unproven. |
| hash semantics gap | Tinygrad `identity_sha256` is the SHA-256 of path/size/mtime metadata, as defined by `_model_identity`; it is not the GGUF byte hash used by the summary and llama. | `model_hash_match=true` is not derivable from the five tinygrad authorities. A future row must carry the byte SHA or a content-addressed run-manifest join. |
| provenance gap | The five tinygrad authorities do not carry tinygrad commit/tree, BoltBeam revision, exact target id, OS, Xcode, xctrace, or compiler identity. Those values occur only in the plan/summary or are absent entirely. | Artifact-to-revision and artifact-to-toolchain identity are asserted rather than joined. |
| run-validity gap | The bundle declares serial/cold-process execution and an intended interleaved order, but llama rows have no timestamps and neither runtime has per-row thermal, power, or memory-pressure evidence. | Order, overlap exclusion, and thermal invalidation cannot be independently audited. |
| structural gap | MR0 does not reproduce 726 graph members plus 77 direct calls. The later MR1 census at BoltBeam `9dfbad60f080e70487d71cb0342e5bc7cef69609` does, but it is a different tinygrad revision and capture. | The later census closes MR1 accounting, not the MR0 same-revision reconciliation requirement. |

The MR0 repair is an evidence repair, not a performance-code change: publish a corrected protocol, retain per-row run
manifests with the missing identities and validity facts, include a comparable llama output check, and reissue the
summary as `INCONCLUSIVE` unless all gates are present. Do not overwrite the historical bundle.

## MR7 authority graph

One concern has one authority. No repository gets a second role map or ranker.

| Concern | Reused authority |
| --- | --- |
| workload, target, and whole-token samples | corrected MR0 bundle plus the replay arm retained by MR4 |
| call admission and graph/direct placement | MR1 `tinygrad.graph_admission_census.v1` |
| semantic role and exact tensor facts | `tinygrad.llm.model_facts` and the completed MR5 propagation seam |
| current route/plan and program identity | tinygrad route census plus source/plan/binary hashes |
| exact isolated execution | existing `tinygrad.search_provider.v1` `describe/admit/compile/check/measure` path |
| finite evidence validation and measured rank | BoltBeam |
| portable optimization concepts | MR6 transfer matrix; concepts do not contribute performance scores |

MR1 at `9dfbad60f080e70487d71cb0342e5bc7cef69609` is a valid zero-unknown structural census: 803 logical calls become
726 members in 24 graph batches and 77 direct calls. Its census blob is
`c09e97ac057a186f94834a141a26c99bff362973`. All 803 metadata arrays are empty, so it cannot rank semantic work.
The current MR5 document is explicitly an inventory and identifies the missing propagation seam. MR7 remains blocked
until a new census at the selected MR4 replay strategy carries stable semantic identities.

## Identity and joins

### Semantic identity

MR5 constructs one immutable identity from existing model facts. The ranking projection requires:

```text
model byte hash
phase
normalized production role family
exact tensor/module identity
logical M/N/K
quant storage and packed layout
input/output/scalar-compute/accumulator dtypes
```

Layer index is occurrence metadata, not candidate applicability. `attn_k` and `attn_v` remain distinct tensor
identities even if the existing normalized family is `attn_kv`. Generic operations use an explicit generic category;
missing values use `metadata_unavailable`. A Metal program name or command-buffer label is display-only and is never
parsed to populate any field.

### Execution and placement identities

Keep two orthogonal joins rather than overloading the semantic key:

- execution identity: semantic identity plus ordinary generated route class, source hash, plan hash, binary hash,
  launch geometry, tinygrad/compiler revision, and resolved-target hash;
- placement identity: run id, replay strategy, call index, assignment, graph batch/member index or direct-call index,
  and admission reason.

The MR1 census, MR5 route census, provider result, and whole-token run must agree on the shared fields. A missing or
many-to-many join is never repaired by name parsing.

## Measurement protocol

### 1. Freeze the control

Use the MR4-retained replay strategy, whether that is compact replay or the safe original grouping. Record at least
five valid interleaved whole-token control samples at depth 128 with the corrected MR0 identity/thermal rules. The
median whole-step time is `W`. Retain raw wall samples and selected GPU-interval unions; do not substitute tok/s from a
different session. Depth 512 is an additional required regime before admitting a flash-attention family.

### 2. Reconcile every occurrence

Join the completed MR5 identity census to all MR1 logical calls. For each exact execution identity `e`, compute its
occurrence count `n_e` from call indices, not label frequency. Preserve the exact set of graph batches and direct calls
containing each role family. Require:

```text
sum(graph members) + sum(direct calls) + ignored nodes + explicit failures = logical calls
unknown admission decisions = 0
material calls with metadata_unavailable = 0
```

A graph batch supplies its aggregate duration and complete member set only. Never calculate
`batch_duration / batch_size`, even when every member has the same broad family.

### 3. Measure the exact ordinary program

For every unique material execution identity, reuse `tinygrad.search_provider.v1` in
`shape_mode=exact_workload`. Measure the ordinary current generated control before proposing a candidate:

- exact M/N/K, quant/layout/dtypes, target, compiler, source/plan/binary identity, and launch geometry;
- resident identical buffers and exact role data or a content-addressed equivalent fixture;
- compile once outside timing, at least two warmups, at least seven synchronized GPU samples, `wait=True` Metal
  command-buffer timestamps, raw samples, median, dispersion, and working-set facts;
- correctness against the generic semantic oracle before timing is admitted;
- deterministic/interleaved execution when two equivalent control constructions must be compared.

If multiple layer occurrences share an execution identity, time it once and retain the occurrence count. If source,
plan, shape, layout, or dtype differs, it is a separate identity and must be measured separately. Bounded fixtures,
candidate timings from the 2026-07-29 search, and graph-entry estimates are inadmissible.

### 4. Form whole-token bounds

For role family `r`, let `E_r` be its exact measured execution identities. Define:

```text
I_r = sum over e in E_r of n_e * median_isolated_gpu_time_e

G_r = duration of the union of complete outer GPU intervals that contain any occurrence of r

H_r = min(W, I_r, G_r) / W

maximum modeled saved time_r = H_r * W
maximum modeled throughput gain_r = 1 / (1 - H_r) - 1
```

`I_r` is the 100%-elimination isolated-cost model. `G_r` is a non-additive enclosing bound: a mixed graph batch's full
interval may appear in several role bounds, but it is never split among members and role bounds must never be summed.
Use interval union when timestamps exist. If only aggregate durations exist, sum complete enclosing batches, cap at
`W`, mark the result conservative, and do not use it to break ties.

Bootstrap raw whole-token and isolated samples to report the median and one-sided 95% lower confidence bound for
`H_r`. The model is a screening ceiling, not an achieved speedup: cache behavior, fusion, dependencies, and graph
execution can make isolated substitution inaccurate. MR11 remains the whole-model authority.

### 5. Apply the 5% headroom gate and rank

A role family is eligible only when all identity, reconciliation, correctness, target, and sample gates pass and both
the point estimate and one-sided 95% lower bound of `H_r` are at least 0.05. Five percent means whole-step time that
could be removed under the explicit elimination model, not a five-percent isolated-kernel win.

BoltBeam ranks eligible families lexicographically by:

1. larger lower-confidence headroom bound;
2. larger median headroom;
3. lower normalized measurement dispersion;
4. stable semantic-identity digest as the deterministic tie break.

Select the first two families at most. A third passing family is `DEFERRED_BY_BUDGET`, not silently merged into a
larger search. If no family passes, stop and revisit the trace/roofline; do not divide graph time or broaden the search
to manufacture a target.

## Bottleneck classification

Classification is separate from rank and defaults to `unknown`.

- `memory_bound`: exact packed/logical byte accounting plus exact-role measurements show time scaling with bytes under
  controlled shape/compute changes, or a scope-compatible counter proves the traffic limit. Raw 120 GB/s and the
  existing non-workload-comparable stream proxy cannot establish this class.
- `compute_bound`: exact semantic operations plus controlled exact-role measurements show time scaling with compute at
  stable bytes, or a supported compiler/counter result proves saturation. Advertised peak compute is insufficient.
- `latency_dispatch_sensitive`: an MR4 grouping-only A/B preserves compiled program identities and resident buffers
  while a statistically valid reduction in submission structure reduces the relevant whole-token GPU/wall time. Call
  count alone is insufficient.
- `unknown`: mixed evidence, missing physical traffic/counters, unstable scaling, or inability to isolate the current
  program.

The classification suggests MR8 axes; it never bypasses the 5% gate and does not import AMD implementation details.

## Blocked, inconclusive, and terminal rules

| Result | Rule |
| --- | --- |
| `BLOCKED_PREREQUISITE` | MR0 identity repair, MR4 replay verdict, completed MR5 propagation, or MR6 matrix is absent. No role ranking is emitted. |
| `BLOCKED_IDENTITY` | Any material call lacks semantic identity, a join is many-to-many, program/source/binary identity drifts, or a label would be needed to complete the join. |
| `BLOCKED_PROVIDER` | Exact current-route shape cannot be admitted/compiled/checked/measured by the existing provider, or only a bounded fixture/estimated graph entry is available. |
| `INCONCLUSIVE_MEASUREMENT` | Fewer than required valid samples, thermal/power/memory-pressure invalidation, non-finite timing, excessive or order-correlated dispersion, confidence instability, or unmatched whole-token and isolated sessions. |
| `INCONCLUSIVE_BOUND` | The enclosing graph/direct interval set is incomplete, exact measured identities do not cover all material occurrences, or `I_r` and `G_r` disagree beyond their declared uncertainty without an explanation. |
| `EXCLUDED_NO_HEADROOM` | Evidence is valid but median or lower-confidence `H_r` is below 5%. This is a measured exclusion, not a blocker. |
| `ELIGIBLE` | Every gate passes and 5% headroom clears. |
| `SELECTED_FOR_MR8` | One of the first two eligible role families under the centralized BoltBeam ordering. |
| `DEFERRED_BY_BUDGET` | Eligible but below the first two; preserved with evidence and reopen condition. |

Any blocker at the global prerequisite or identity layer stops MR7. Per-role provider blockers may coexist with a
complete ranking only when the blocked role's enclosing upper bound is already below 5%; otherwise the overall result
is inconclusive because an important family may be missing.

## Required MR7 artifact

BoltBeam should emit the required `role-cost-ranking.json` through its existing evidence/result ledger rather than add
a Metal-only ranker. Each role row records semantic digest, member/call indices, occurrence counts, graph/direct
enclosures, exact provider evidence hashes and raw samples, `I_r`, `G_r`, `H_r`, confidence bound, bottleneck class,
5% gate result, rank state, and reopen condition. The artifact also pins MR0/MR1/MR4/MR5/MR6 content hashes and the
ordinary generated fallback.

No selected candidate, runtime binding, or Metal performance claim exists at MR7. MR8 may construct a bounded
target-neutral population only for `SELECTED_FOR_MR8` rows.
