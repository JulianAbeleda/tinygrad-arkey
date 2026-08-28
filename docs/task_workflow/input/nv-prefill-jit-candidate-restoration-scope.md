# NV prefill JIT candidate restoration scope

Status: completed by `../output/nv-prefill-corrected-tile-result.md`.

Target: dense Qwen3-8B, Q4_K_M source weights, 512-token physical prefill
microbatch, `NV:sm_120`, RTX 5090. MoE, sub-512 prompts, other models, and other
targets are outside this scope.

Primary objective: determine exactly why the current installed prefill path
does not execute the four promoted fp16 Graph-GEMM candidates, restore the
path if the retained candidates remain correct and fast, independently restore
or replace the fused prefill-attention route, and produce a fresh full-model
512/1024/2048/4096 ledger against same-session llama.cpp.

This scope follows test -> invest. A lifecycle hop is repaired only after a
probe proves that hop is the first divergence. No historical throughput number
is promotion evidence.

---

## 1. Current truth and claim boundary

Fresh 2026-08-28 evidence is in
`docs/task_workflow/evidence/nv-prefill-restoration-20260828/`:

| row | observed state |
| --- | --- |
| tinygrad pp512 before restoration | about 433 tok/s, about 1.18 s per 512-token chunk |
| llama.cpp pp512 | about 14.1k tok/s in the same session |
| promoted compact set | present; four expected sm_120 identities expand |
| research typed admission | initially rejected `NV:sm_120`; repaired locally and 89 focused tests pass |
| loaded model | owns fp16 overlay, registry, route attachments, and exact Graph-GEMM bindings |
| direct non-JIT graph construction | records all four candidate roles |
| real TinyJit prefill | records zero candidate roles |
| historical 11.2k pp512 row | reproduced and proven numerically invalid; retired |

The completed repair produces a correctness-qualified 84.025 ms pp512 path
(6,093 prompt tok/s).  All four candidate roles execute on normal `DEV=NV`,
and complete-model logits match the safe path exactly.  The original 46 ms
result combined a dropped LDS publication with duplicated output tiling and is
not an optimization authority.

Therefore these statements are currently forbidden:

- “tinygrad prefill is at llama parity”;
- “the old prefill result only needed an artifact hookup”;
- “the four candidate kernels executed because the model admitted them”;
- “the remaining issue is attention” or “the remaining issue is GEMM” before
  the lifecycle census below assigns the wall.

A current claim becomes admissible only after an ordinary clean-process model
load, without research overrides, proves route identity, correctness, final
binary mechanism, and full-model wall together.

---

## 2. Work accounting

The dense projection work for one 512-token chunk is approximately 7.11 TFLOP.
The four promoted role identities cover the following physical invocations in
a 36-layer model:

| promoted role | physical linears/layer | invocations/chunk | shape `(M,N,K)` | approximate chunk FLOP |
| --- | ---: | ---: | --- | ---: |
| `attn_qo` | Q and O | 72 | `(512,4096,4096)` | 1.24 TF |
| `attn_kv` | K and V | 72 | `(512,1024,4096)` | 0.31 TF |
| `ffn_gate_up` | gate and up | 72 | `(512,12288,4096)` | 3.71 TF |
| `ffn_down` | down | 36 | `(512,4096,12288)` | 1.86 TF |
| total | seven linears/layer | 252 | — | 7.11 TF |

The full wall must be decomposed as:

```text
chunk wall
  = dense projection GPU span
  + prefill-attention GPU span
  + norms / activations / RoPE / KV stores / output work
  + graph and host non-overlapped time
  - real GPU overlap
```

At the current pp512 wall, 7.11 TF / 1.18 s is only about 6 TFLOP/s. The old
45.9 ms row would imply about 155 TFLOP/s across the dense-model accounting.
That scale difference is why the first job is route and final-binary proof,
not small kernel tuning.

---

## 3. Exact lifecycle to observe

Every selected projection must be accounted through these hops:

```text
checked-in compact artifact
  -> exact CandidateRegistry admission
  -> model policy and inventory attachment
  -> per-linear Graph-GEMM binding
  -> route_prefill_linear decision
  -> route_pf16_graph_gemm construction
  -> lazy Tensor AST / warmstart key
  -> postrange option and candidate-context match
  -> optimized KernelInfo candidate_context
  -> rendered NV source and compiled binary
  -> TinyJit captured LINEAR / graph node
  -> replayed GPU program
  -> full-model output and wall
```

Fused attention has a separate chain:

```text
target promotion record + admitted `(Hq,Hkv,T)` geometry
  -> model `prefill_custom_kernel_attn`
  -> custom_kernel_attention construction
  -> FlashPrefillAttentionSpec validation
  -> NV lowering and final binary
  -> one attributed attention service/layer
  -> replayed GPU program
  -> full-model output and wall
```

The two chains join only at the full transformer. Neither chain may borrow the
other's evidence.

---

## 4. Required observability substrate

Add a research-only, context-local observer. It must be inert when absent and
must not select, modify, or rescue a route. Emit JSON records rather than
parsing `DEBUG` prose.

Each dense record must carry:

- process/run identity and git commit;
- TinyJit phase: eager `cnt=0`, capture `cnt=1`, or replay `cnt>=2`;
- layer and linear identity;
- role and exact `(M,N,K)`;
- candidate and candidate-set identities;
- scanned target triple;
- route decision and exact decline reason;
- expected and actual warmstart keys;
- whether opts matched;
- whether a candidate context matched;
- context identity before postrange, after postrange, at PROGRAM, and at
  compiler-cache lookup;
- final function name, source SHA, binary SHA, registers, shared memory,
  spill/local-memory state, grid, and block;
- captured graph group/node identity and execution count.

Each attention record must additionally carry the requested and admitted
attention geometry, spec identity, lowering result, explicit fallback reason,
and final binary identity.

Observer rules:

1. No environment variable may silently enable a production route.
2. No callback may retain Tensor or Buffer objects.
3. The observer cannot affect compiler-cache keys.
4. The same graph without the observer must have identical program and binary
   identities.
5. Unit tests must prove nesting, exception restoration, and no-observer
   behavior.

Likely instrumentation surfaces are:

- `tinygrad/llm/prefill_graph_gemm.py`;
- `tinygrad/llm/prefill_routes.py`;
- `tinygrad/codegen/opt/postrange.py`;
- `tinygrad/codegen/__init__.py`;
- `tinygrad/engine/jit.py`;
- `tinygrad/llm/fused_attention.py`.

Instrumentation belongs in a distinct commit and is removed or retained as a
generic inert diagnostic independently of any repair.

---

## 5. Phase A — prove the observer and locate the first divergence

Run each arm in a fresh process so compiler and JIT caches cannot leak across
arms.

| arm | construction | expected evidence |
| --- | --- | --- |
| A0 | exact registry decode only | four admitted identities |
| A1 | one direct `model.forward`, graph construction only | four unique roles; physical counts 72/72/72/36 |
| A2 | ordinary model call, TinyJit eager call `cnt=0` | same construction counts and 252 candidate programs scheduled |
| A3 | same JIT capture call `cnt=1` | same program identities enter captured LINEAR |
| A4 | replay `cnt>=2` | no Python reconstruction required, but the 252 captured candidate nodes execute |
| A5 | already-warm model entered under the observer | observer truthfully labels pre-existing capture rather than reporting false missing routes |

The current census observes A1 but reports no candidates for the authority
harness. Before changing routing, decide whether this means:

- candidates were compiled before the census started;
- candidate construction occurs in a context not visible to the current
  `ContextVar`;
- the JIT's eager/capture phases use a different route;
- or the candidates truly never enter scheduling.

Pass gate: the observer explains A1–A5 without contradiction, and an injected
known fallback produces a named decline at the exact first hop. If the observer
cannot distinguish precompiled replay from fallback, stop; later performance
measurements are not attributable.

---

## 6. Phase B — candidate context and warmstart-key matrix

For each of the four shapes, capture these values at `apply_opts`:

```text
actual _warmstart_key(k)
expected public warmstart_key(...)
matched opts
matched candidate context
candidate identity
planned opts
apply success or KernelOptError
optimized AST candidate identity
```

Run four controlled arms:

| arm | opts | candidate context | purpose |
| --- | --- | --- | --- |
| B0 | none | none | generic heuristic control |
| B1 | promoted opts | none | isolate schedule-option value |
| B2 | promoted opts | promoted context | exact retained candidate |
| B3 | deliberately wrong identity/context | wrong | prove collision/mismatch fails closed |

Questions this phase must answer, not assume:

1. Does lazy scheduling happen while `warmstart_candidate_state` is active?
2. Does the dense candidate context coexist with the packed-WMMA context map,
   or is one replacing the other?
3. Does the shape key change after semantic markers, contiguous boundaries,
   JIT parametrization, or graph memory planning?
4. Does the same key alias two candidates with different identities?
5. Does `KernelOptError` trigger a generic heuristic fallback, or fail loudly
   when a candidate context is present as intended?
6. Does the lower/program cache distinguish no-context and candidate-context
   kernels?

Pass gate: B2 must show the same candidate identity continuously from the
binding through optimized `KernelInfo` and compiler-cache context. A missing
identity at any hop names the repair boundary. Do not alter the next hop until
this gate passes.

---

## 7. Phase C — final binary mechanism proof

Compile one legal instance of every role in B0/B1/B2. Retain source, binary,
disassembly, and compiler-owned resource facts.

For B2 require:

- tensor-core instructions appropriate to sm_120;
- the declared 128x128x32 tile, 4x2 warp geometry, 256 threads, and 40,960-byte
  active two-buffer LDS contract;
- aligned vector transport consistent with the candidate payload;
- no spill or unexpected local-memory traffic;
- final binary cache identity bound to the candidate identity;
- a binary distinct from B0 whenever the mechanisms are distinct;
- no AMD ABI or target descriptor in the NV binary.

Do not infer mechanism from a source function name. Use final SASS plus
compiler-owned resource metadata. Compare to retained historical artifacts
where available, but do not require byte identity across compiler revisions.

Decision:

- If B2 never changes the binary, the context is being ignored or consumed too
  late: repair the compiler lifecycle.
- If B2 produces the intended binary but the binary is invalid or spills, the
  retained candidate is incompatible with the current compiler: return to
  search, not plumbing.
- If B2 produces a valid intended binary, proceed to isolated timing.

---

## 8. Phase D — per-role correctness and performance

Build a standalone production-shaped gate for each role. Use the same
allocation, inputs, stream, events, cache conditioning, and output oracle for
B0/B1/B2. Run at least R9 randomized/reversed brackets after warmup.

Required rows:

| role | generic | opts only | exact candidate | correctness | resources | hot time | rotated-cold time |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| `attn_qo` | | | | | | | |
| `attn_kv` | | | | | | | |
| `ffn_gate_up` | | | | | | | |
| `ffn_down` | | | | | | | |

Correctness must use finite legal fixtures, output guards, read-only input
checks, deterministic reruns, and a stable higher-precision oracle. Reuse the
historical numerical thresholds only if their exact fixture and semantic
contract are retained; otherwise predeclare new max-absolute, relative-L2,
and downstream-logit gates before timing.

Investment gate per role:

- exact candidate passes semantics;
- final mechanism passes Phase C;
- median improvement exceeds run spread in both bracket directions;
- no resource regression that reverses the win under production cache state.

A losing role is not forced into the candidate set. Search or use the best
proved fallback for that role, then mint a new identity and re-run admission.

---

## 9. Phase E — correct the JIT integration only after D passes

The repair must be the smallest change at the first failed Phase A/B hop. Test
these hypotheses in order:

1. **Census timing only:** candidate was already compiled before the observer.
   Repair attribution; do not change execution.
2. **Context lifetime:** scheduling occurs after the route's mutation or after
   the enclosing context restores state. Move exact candidate state to the
   scheduling owner, not a longer-lived global.
3. **Context-map replacement:** packed and dense context maps overwrite one
   another. Merge them with collision checks at one model-owned boundary.
4. **Warmstart-key drift:** the scheduled AST no longer matches the advertised
   key. Fix the key derivation or binding only after the actual before/after
   key is recorded.
5. **Premature capture:** a fallback graph is captured before attachments are
   installed. Move attachment before the first callable JIT phase and prove
   the old graph cannot be reused.
6. **Compiler-cache alias:** a fallback program is returned for a candidate
   AST. Fix cache identity and add a cross-context cache test.
7. **Candidate incompatibility:** plumbing is correct but the old candidate no
   longer compiles or wins. Re-run the per-shape search; do not disguise a new
   schedule under the old identity.

The already-tested broad idea “retain candidate contexts through the outer
TinyJit scope” did not change the full wall and is not repeated without new
Phase B evidence.

Integration pass gate:

- clean-process TinyJit A2/A3/A4 all account correctly;
- final captured graph contains all 252 dense candidate invocations with the
  expected 72/72/72/36 role counts;
- replay executes those exact binaries;
- zero unexpected candidates and zero silent generic declines;
- default-off/nonmatching targets remain unchanged.

---

## 10. Phase F — independently settle fused prefill attention

The historical fast wall depended on more than the dense GEMMs. The current
repository records an earlier fused NV route failing verification with a
`PACKED_FRAGMENT_LOAD` / native-ABI error. Treat attention as an independent
requalification.

Run three arms on identical Q/K/V tensors and actual model geometry:

| arm | route | purpose |
| --- | --- | --- |
| F0 | ordinary current SDPA | semantic and wall control |
| F1 | currently promoted custom-kernel route | test present plumbing and capture |
| F2 | retained historical fused construction, isolated | determine whether the old mechanism is still expressible |

For each arm retain:

- exact attention geometry and mask semantics;
- output oracle metrics and finite checks;
- explicit custom-kernel trace or fallback reason;
- final source/binary/SASS/resources;
- kernel count, GPU active span, and full attention span per layer;
- graph node count and graph grouping.

Decision tree:

- F1 falls back before compile: repair route/spec admission.
- F1 reaches compile and fails verification: localize the first invalid UOp or
  ABI ownership mismatch, then fix the generic lowering primitive with unit
  coverage.
- F1 compiles and is correct but loses F0: do not promote it; search attention
  topology independently.
- F1 is correct and faster: install only after full-model Phase G.
- F2 works while F1 fails: diff the semantic/spec/lowering chain, not merely
  generated source.

Attention pass gate: one attributed, correct attention service per layer (or
an explicitly proved equivalent split), no hidden SDPA fallback, and a net
full-attention-span win.

---

## 11. Phase G — full-model correctness

Run control and candidate in separate clean processes with identical model,
tokens, seed, context capacity, and target facts.

Required checks:

1. Full pp512 logits are finite.
2. Candidate reruns are deterministic.
3. Greedy token output matches the approved reference.
4. Full logits pass predeclared max-absolute and relative-L2 gates.
5. Top-k set/order and decision margin remain stable.
6. KV cache contents for the populated prefix pass their exact or
   predeclared-tolerance contract.
7. Input weights and token buffers remain bytewise read-only.
8. Output/scratch guards are intact.
9. A pp1024 recurrent two-chunk run proves the second chunk reads the first
   chunk's KV state correctly.
10. Decode after prefill retains the existing decode correctness hash and does
    not change the installed one-token route.

Token equality alone is insufficient. A candidate can preserve argmax while
corrupting logits or KV state.

---

## 12. Phase H — full lifecycle ledger and same-session performance

First bracket current control versus candidate. Only after the candidate is a
clear within-tinygrad win bracket it against llama.cpp.

Protocol:

- fresh processes and recorded environment;
- same GGUF and GPU;
- clock/thermal state recorded;
- R7 or greater, randomized/reversed A/B/B/A ordering;
- warmup sufficient to exclude compilation and graph instantiation;
- synchronized wall time plus GPU active span;
- pp512, pp1024, pp2048, pp4096;
- llama `-ngl 99 -fa 1 -n 0` with the same prompt lengths;
- report mean, median, spread, and every raw sample;
- retain program census and attention trace with each tinygrad row.

The final ledger must include:

| region | tinygrad wall/GPU span | llama wall/GPU span | TG − llama | candidate coverage | explanation |
| --- | ---: | ---: | ---: | ---: | --- |
| dense Q/O | | | | | |
| dense K/V | | | | | |
| dense gate/up/down | | | | | |
| prefill attention | | | | | |
| norms/RoPE/KV/elementwise | | | | | |
| graph/host non-overlap | | | | | |
| total chunk | | | | | |

Also report achieved dense TFLOP/s from the 7.11-TF accounting, wall/busy ratio,
kernel/node count, and percent of the total gap owned by each region.

Claim rules:

- “restored candidate path” requires exact census + final mechanism +
  correctness, regardless of speed;
- “faster prefill” requires a fresh same-session row at the named prompt
  length;
- “prefill parity” requires the declared comparison rule across all four
  lengths, not one favorable point;
- a mixed table must be stated as mixed; do not average away a losing endpoint;
- no warm result may be described as cold TTFT.

---

## 13. Promotion requirements

Promotion is a separate commit after all gates pass.

Required before default-on:

- exact four-role artifact identities or newly minted replacements;
- target-exact admission for `NV:sm_120`;
- unit tests for admission, attachment, route decision, key matching, context
  propagation, cache identity, and fail-closed behavior;
- final SASS/resource records for every distinct kernel body;
- 252-invocation clean-process census;
- fused-attention attribution or an explicit measured decision to retain SDPA;
- full-model correctness packet;
- R7+ four-length within-tinygrad bracket;
- R7+ same-session llama bracket;
- installed-endpoint rerun without research environment variables;
- no regression to decode, AMD compile behavior, Metal importability, or
  unknown-target fallback.

Recommended commit sequence:

1. inert observer and observer tests;
2. exact first-divergence repair, closed by default;
3. isolated candidate evidence and any newly minted artifact;
4. fused-attention repair/requalification, if it passes;
5. full-model correctness and performance evidence;
6. promotion/default flip;
7. installed-endpoint confirmation and ledger update.

Never combine mechanism, evidence, and default flip in one commit.

---

## 14. Stop conditions and branch outcomes

Stop and record rather than investing further when:

- the observer cannot attribute replayed binaries;
- a retained candidate fails isolated correctness;
- the intended final mechanism is absent;
- the candidate loses its isolated R9 gate;
- full-model correctness fails;
- dense recovery is real but attention consumes the recovered wall;
- the candidate wins only with research overrides that cannot be represented
  in the production lifecycle;
- or the four-context wall does not improve beyond run spread.

Possible honest outcomes:

1. **Plumbing restoration:** old candidates remain fast; repair JIT ownership
   and requalify attention.
2. **Partial restoration:** some roles remain fast; mint a mixed candidate set
   and search only losing roles.
3. **Compiler-era invalidation:** old identities no longer produce competitive
   binaries; retire them and perform a fresh sm_120 search.
4. **Attention wall:** dense path recovers but fused attention does not; record
   the new dense ceiling and scope attention separately.
5. **Harness-only error:** candidates already execute and only census is stale;
   repair attribution, then use the measured wall to decide whether any kernel
   work is warranted.

---

## 15. Completion definition

This scope is complete only when one of these is documented with retained
artifacts:

- an installed, correctness-qualified prefill path with an exact lifecycle
  ledger and fresh four-context llama comparison; or
- a precise wall showing the first unrepairable/uneconomic lifecycle boundary,
  with every earlier hop proved and the historical parity claim formally
  retired.

The immediate next executable is Phase A: build the inert lifecycle observer
and run A0–A5. It has the highest information value because it distinguishes a
false census from a real JIT route failure before any production optimization
is attempted.

---

## 16. Reproducible harness and artifact contract

Add one orchestration entry point rather than accumulating temporary scripts:

```text
extra/llm_research/prefill/nv_prefill_candidate_lifecycle.py
```

Required subcommands:

```text
admission       # A0
direct          # A1
jit-phases      # A2-A5, emits one row per TinyJit phase
compile-roles   # B0-B3 and Phase C
microbench      # Phase D
attention       # Phase F
correctness     # Phase G
wall            # Phase H tinygrad arm
summarize       # joins retained child artifacts; never executes a GPU path
```

Every GPU arm runs in its own child process and writes atomically only after
success. The parent treats a missing, partial, non-finite, identity-mismatched,
or nonzero-exit child as FAIL, preserving stdout/stderr separately.

Canonical evidence directory:

```text
docs/task_workflow/evidence/nv-prefill-jit-restoration/
  provenance.json
  admission.json
  direct.json
  jit-phases.json
  compile/
    attn_qo/{control,opts,candidate}/
    attn_kv/{control,opts,candidate}/
    ffn_gate_up/{control,opts,candidate}/
    ffn_down/{control,opts,candidate}/
  microbench.json
  attention.json
  correctness.json
  wall-tinygrad.json
  wall-llama.json
  lifecycle-ledger.json
  result.md
```

`provenance.json` must include commit, dirty paths, Python/tinygrad invocation,
GPU name, PCI identity, driver, firmware-visible architecture, renderer target,
model SHA, candidate-set identity, llama binary commit, environment diff, clock
policy, and exact commands. Large binaries/SASS may live below `compile/`; their
SHA-256 values belong in the joined JSON.

Minimum structural regression suite before any GPU run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  test/unit/test_runtime_specs.py \
  test/unit/test_promoted_prefill_candidate_runtime.py \
  test/unit/test_prefill_graph_gemm_runtime.py \
  test/unit/test_candidate_context_propagation.py \
  test/unit/test_kernel_candidate_context.py \
  test/unit/test_prefill_whole_synced.py
```

Canonical full-wall command remains
`extra/llm_research/prefill/prefill_whole_synced.py`, but it is run only after
the new lifecycle harness passes. Its candidate census must be updated to
distinguish “compiled before observer” from “never selected”; weakening or
removing its hard failure is prohibited.

Llama comparison template:

```bash
/home/ubuntu/env/llama.cpp/build-cuda/bin/llama-bench \
  -m /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  -ngl 99 -fa 1 -p <512|1024|2048|4096> -n 0 -r 7 -o json
```

## 17. Test-to-invest checklist

| checkpoint | proof needed | investment unlocked |
| --- | --- | --- |
| observer | A0-A5 distinguishes construction, capture, replay, and fallback | lifecycle instrumentation may be trusted |
| route | all exact attachments and decline reasons accounted | repair first divergent hop |
| scheduler | key, opts, and identity continuous through postrange | compile the candidate mechanisms |
| binary | intended TC/LDS mechanism, no spill, exact cache identity | isolated GPU timing |
| primitive | correctness plus R9 net win per role | JIT/product integration for passing roles |
| JIT | 252 captured and replayed candidate invocations | full-transformer correctness |
| attention | correct attributed fused route beats SDPA, or measured SDPA decision | full lifecycle wall bracket |
| model | logits, KV, tokens, and decode-after-prefill pass | four-context performance claim |
| wall | R7+ within-tinygrad win and same-session llama ledger | promotion review |
| installed | clean ordinary endpoint reproduces identity and wall | public claim |

No checkpoint may be skipped because a downstream timing looks favorable.
# Snapshot-first correction

Restoration starts from the last measured working prefill state, not from an
open-ended diagnosis of the current fallback.  The reachable authority is
`8c3e762dc`, which records four passes ending at 45.7/44.2 ms with the tuned
fp16 candidate schedule.  An isolated replay of that exact commit is not
self-reproducing: it fails before its first pass at the NV fused-attention
`PACKED_FRAGMENT_LOAD` contract.  The later pinned-tree audit independently
reports the same failure.  The named `04e500079`, `1d4fef2ed`, and `d8bac6914`
objects are absent, no Aug 1--3 unreachable commit survives, and no historical
ABI-bearing cache entry or old worktree remains.

Therefore the historical number is measurement authority, but the committed
tree is not executable-source authority.  The minimum first-principles bundle
to reconstruct is:

1. fp16 overlay projection ownership with `TC`, `UPCAST(1,4)`, then
   `UPCAST(0,2)`;
2. NV fused prefill attention (the retained fused-off control is roughly
   160 ms, so projection restoration alone cannot recover the historical row);
3. renderer-owned native-attention lowering in the correct IR phase: packed
   fragment loads and row-softmax repacks must be expanded before the scalar
   program spec, with the target selected from the renderer rather than
   ambient process state;
4. only after semantic parity, restore the historical graph/replay lifecycle
   and re-bracket the full token path.

Widening the scalar program spec to accept the leaked native operations is not
a fix.  An isolated diagnostic that admitted the first leaked scalar-half
fragment merely exposed the next leaked native row-softmax operation.  This
proves a missing/misordered lowering pass rather than one stale validator row.
