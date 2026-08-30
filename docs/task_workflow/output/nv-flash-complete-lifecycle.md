# Complete Flash decode lifecycle

## Decision

The complete Flash lifecycle is now known well enough to stop treating
`flash score` as an isolated kernel row.

For the qualified dense decode endpoint, both tinygrad and llama execute the
same high-level algorithm:

```text
Q/K/V ready
  -> partitioned score + online softmax + PV partials
  -> global partial combine
  -> O projection
```

They differ in how the graph chooses the physical KV horizon, how producer
work reaches the score kernel, the partial ABI, output dtype, and how the
runtime launches the graph. The remaining tinygrad loss is after Q/K/V
readiness: approximately 60 us/token in score service and 13 us/token in
combine service. Tinygrad is already faster in the fused Q/K completion and
KV-store lifecycle that feeds Flash.

The previously open active-horizon graph policy has now been tested through
the complete token lifecycle. It is exact and emits the intended S6/768 graph
through token 768 and S8/1024 afterward, but it is not promotable. Cold graph
handoff stalls on lazy S8 capture; prewarming both graph pairs removes that
stall but makes the complete candidate slower than both wall controls. No
known positive Flash construction remains uninstalled.

The pre-score entry path is also accounted hop by hop. The prior layer's
gate/up stream is below the cache knee; adding its FFN-down stream is the first
checkpoint that makes the next score cold. The later provider and K/V-ready
hops are neutral or slightly reheating. Thus FFN-down creates the inherited
residency tax and Flash pays it when it next demands historical K/V.

## Scope and evidence labels

This record is exact for the current Qwen3-8B dense decode authority on RTX
5090, batch one, one-token decode, 32 query heads, eight KV heads, head width
128, fp16 KV cache, and the admitted NV sm_120 wide-vector route. It is not a
claim about prefill, MoE, other head geometries, KV quantization, rope-at-read,
or larger context capacities.

- **Source-proven** means the behavior follows directly from the checked
  tinygrad or llama source.
- **Trace-measured** means the installed graph or kernel interval was
  observed in retained profiling evidence.
- **Wall-measured** means a token-hash-preserving reverse wall bracket passed.
- **Inferred** means the explanation reconciles the source and measurements
  but has not itself been isolated.
- **Closed** means a plausible construction was tested and did not convert to
  token wall.

The llama source authority is commit
`ac4cddeb0dbd778f650bf568f6f08344a06abe3a`. The tinygrad installed endpoint
is 4.094502 ms/token, or 244.230 tok/s. The retained llama authority is
4.021721 ms/token, or 248.711 tok/s.

## One-layer lifecycle

### Installed ceiling addendum

The retained S8 figures below describe the pre-ceiling endpoint.  The current
installed dense d512 route selects S6 through Tc=768 and pre-captures distinct
S6 and S8 greedy ping-pong graph pairs.  Its 36-call score and combine rows are
194.048 and 48.448 us/token respectively, versus llama's retained 162.948 and
37.057 us.  The current endpoint is 4060.523 us/token / 246.274 tok/s.

The graph tracker reports about 3925 us for both node sum and device union,
with effectively zero overlap.  Consequently PDL/launch-ahead is not an open
pool on the current graph.  It may be retested only if a future construction
changes score/combine ownership or dependencies.  Full evidence and closed
Q-load, V-schedule, S6-combine, and grouped-ownership tests are in
`docs/task_workflow/output/nv-flash-ceiling-exhaustion-result.md`.

| stage | tinygrad installed path | llama path | status |
|---|---|---|---|
| 1. graph shape | Full cache capacity is 1024; current admitted wide route derives S8 from `MAXC/128` | Used KV extent is padded to a 256-token graph bucket; at the captured depth it is 768 | source-proven |
| 2. attention input | Native RMS norm or the shared RMS/Q8 provider prepares the common activation | Separate attention RMS norm, then three Q8 activation quantizers | source/trace-proven |
| 3. Q producer | Q projection, then one fused head RMS-norm + RoPE completion | Q projection, separate head RMS norm, separate RoPE | source/trace-proven |
| 4. K/V producers | K/V projection bodies, then one fused K head norm + RoPE + K/V cache completion | K projection -> head norm -> RoPE -> K store; V projection/cache write is a separate graph branch | source/trace-proven |
| 5. readiness join | Score has direct data dependencies on Q completion and the combined K/V cache completion | Score has data dependencies on Q RoPE, K store, and V completion | DAG-proven |
| 6. score launch | 32 query heads x eight physical partitions, 128 threads/CTA | 32 query heads x six physical partitions, 128 threads/CTA | trace-measured |
| 7. score body | Wide K/V loads, fp16 dot, online softmax, PV/den/max partial per head and partition | Wide cooperative K/V copies, fp16 dot, online softmax, PV partial plus float2(max,sum) | source/trace-proven |
| 8. combine | One 128-thread CTA/head combines eight partials and writes fp16 | One 128-thread CTA/head combines six partials and writes fp32 | source/trace-proven |
| 9. O handoff | O Q4 projection consumes the typed fp16 combine result and absorbs residual add | Separate Q8 quantizer feeds O MMQ; O owns its fusion behavior | source/trace-proven |
| 10. next layer | Attention result feeds the FFN path, then the next layer repeats | Same semantic transition | DAG-proven |

There are 36 score calls and 36 combine calls per token in both traces. No
Flash cardinality mismatch is hiding in the comparison.

## Lifecycle 1: graph creation and physical horizon

### tinygrad

The model allocates a persistent combined cache with logical layout
`[K_or_V, batch, KV_head, MAXC, head_dim]`. The score reads a cache value that
is ordered after the current token's store. `start_pos` is a bound graph input,
so position changes data and loop predicates without forcing a new graph.

At model setup, route admission resolves renderer capabilities once and caches
them. The installed wide route is fail-closed to the qualified shape and to
physical capacities 768 or 1024. With no explicit research geometry, it sets:

```text
physical partitions S = MAXC / 128
logical live tokens Tc = start_pos + 1
```

The official 1024-capacity graph therefore launches S8 even near logical
depth 512. The score predicate masks results for tokens beyond `Tc`, but the
wide K/V loads have already been formed. Empty upper partitions consequently
execute instructions and issue cache/memory traffic.

The source supports a bounded `token_bound` graph that retains the 1024 cache
stride while compiling only S6/768 service. A closed-lease production selector
now chooses S6 until its semantic bound and S8 afterward. It is retained as
research substrate, not installed policy, because the complete selector wall
gate failed both cold and after dual-graph prewarm.

### llama

The KV cache owns separate per-layer K and V tensors. The buffers are cleared
at allocation so padded positions cannot introduce NaNs. For graph reuse,
`llama_kv_cache::get_n_kv` pads the highest used slot to at least a 256-token
multiple. At the retained depth this produces a 768-token view.

The vector Flash launch receives that 768 extent. Its launch helper computes
six 128-token KV tiles, queries occupancy, and searches eligible parallel
block counts for the best wave efficiency. The observed six partitions are
therefore selected from the physical extent and device occupancy; they are not
a hardcoded model/depth constant.

### Consequence

The first structural difference exists before either score kernel starts:
tinygrad services a 1024-token physical horizon while llama services 768.
This is a graph-policy difference, not a mathematical-attention difference.

The two paths also express logical validity differently. Tinygrad carries
`Tc` into the score program and turns out-of-range scores into negative
infinity, while the admitted wide spelling leaves the underlying K/V loads
ungated. Llama supplies a causal/padding mask to the vector kernel, which adds
negative infinity to padded score positions after their K values are loaded.
Thus neither path avoids all reads inside its selected physical horizon. The
important byte difference is the selected horizon itself: 1024 versus 768.

## Lifecycle 2: Q/K/V production and cache ownership

### tinygrad

The installed attention producer DAG is:

```text
attention RMS/provider
  +-> Q projection -> fused Q head RMS norm + RoPE --------+
  +-> K projection --+                                     |
  +-> V projection --+-> fused K norm + RoPE + KV store ---+-> score
```

The fused completion kernels are real lifecycle wins. Across the token,
tinygrad's Q norm+RoPE row is about 5 us faster than llama, and its K
norm+RoPE/cache row is about 34 us faster. The K/V projection lifecycle is
also ahead before the shared provider is allocated and approximately tied
under a conservative provider allocation.

The score input is a zero-copy bitcast view of the ordered fp16 cache. The
wide-load promotion required preserving the `AFTER(cache, store)` dependency
through that bitcast; materializing a new buffer would lose both the ordering
contract and the intended pointer reinterpretation.

### llama

The graph explicitly expands Q, V, and K producer work, then expands K and V
cache copies before building attention. In the retained CUDA trace, each
layer has the following data DAG:

```text
attn norm -> Q quant -> Q -> Q norm -> Q RoPE --------+
          -> K quant -> K -> K norm -> K RoPE -> store +-> Flash
          -> V quant -> V -----------------------------+
```

The Flash kernel receives fp32 Q and fp16 K/V. For the fp16-K path it converts
Q to half pairs in registers. Separate K and V allocation is not itself the
speed mechanism: the matched tinygrad separate-allocation candidate was
bit-correct but slower.

### Readiness boundary

Tinygrad profiles show a median command-timeline interval of about 2.5 us per
layer between the last direct producer timestamp and score start. Llama PDL-off
shows roughly 0.2 us per layer. That visual difference is not a recoverable
90-us/token pool:

- score-to-combine and combine-to-O waits are zero on tinygrad;
- the legal KV-ready split-phase launch changed token wall by only about
  1.5 us versus the control midpoint and failed the conservative gate;
- full endpoint reconciliation gives tinygrad a slightly smaller total
  non-node remainder than llama, so adding the visible readiness gaps would
  double-count runtime/profile timing.

Disposition: the readiness interval is **trace-visible and conversion-closed**.
It remains part of the scheduling description, not part of the booked Flash
debt.

## Lifecycle 3: score kernel selection and launch geometry

### tinygrad installed kernel

```text
flash_vec_llama_score_pv_32_128_8_widekv16
logical owners: 32 query heads x 8 partitions
threads:        128 per CTA, four warps
```

Each CTA owns one query head. Four GQA query heads map to one KV head, so the
same K/V material is read by four CTAs and expected to be served from L1/L2
after the first request. Within each warp, four eight-lane groups score token
columns. Each lane owns 16 head dimensions and loads them as two aligned
16-byte `uint4` transactions for K and V.

The kernel loads Q into registers, computes fp16-pair dot products, performs
online max/sum softmax, accumulates PV in fp32, and emits one partial for each
head/partition. A completely empty partition emits the defined identity
partial: PV zero, denominator zero, maximum negative infinity.

### llama selected kernel

```text
flash_attn_ext_vec<128,1,F16,F16,false>
grid:  (1, 6, 32)
block: (32, 4, 1)
```

It uses the same one-query-head-per-CTA and four-warp principle. Cooperative
16-byte copies and eight-lane dot groups expose 128 token rows per partition.
The kernel maintains online softmax state and writes unnormalized fp32 PV plus
max/sum metadata when more than one partition is used.

The NCU trace classifies this as a latency/occupancy kernel, not a DRAM
roofline kernel: only a small fraction of peak DRAM is used, while repeated
GQA reads are mostly cache-served. Llama intentionally spends cached bytes to
reduce instruction-heavy staging and expose more independent token work.

### What the wide-load promotion proved

The original tinygrad transcription used scalar half loads and was a true
code-generation wall: it generated about fifteen times llama's L1 traffic and
lost badly. Aligned `uint4` loads reduced L1 traffic into llama's regime,
reduced executed instructions, made the S6 pair competitive in isolation,
and passed the full S8 token-wall bracket.

That history matters: the old scalar no-go was not evidence against llama's
topology. It was missing load grammar and boundary ownership.

## Lifecycle 4: online softmax and partial ABI

Both implementations compute attention in partitions without materializing a
full score row. For each partition they maintain:

```text
local_max
local_denominator = sum(exp(score - local_max))
local_numerator   = sum(exp(score - local_max) * V)
```

Tinygrad writes a flat fp32 partial row of width `head_dim + 2` for every
head/partition: 128 numerator values, denominator, and maximum. With S8 this
is 32 x 8 x 130 fp32 values.

Llama writes fp32 numerator values to `dst_tmp` and a separate float2
`(maximum, denominator)` metadata array. At the captured shape this is 32 x 6
x 128 fp32 values plus 32 x 6 float2 metadata values. The CUDA pool owns these
temporary buffers for the launch lifecycle.

The ABI difference explains part of combine traffic/cardinality, but there is
no evidence that merely changing buffer layout is a large independent score
lever. The dominant known cardinality difference is eight partials versus six.

## Lifecycle 5: combine and O handoff

### tinygrad

The installed combine uses one 128-thread CTA per query head. It first finds
the maximum across eight partitions, computes each partition's exponential
rescale, sums numerator and denominator, divides, and writes fp16. The typed
fp16 output is consumed directly by the Q4 O projection, whose epilogue also
absorbs the residual add.

### llama

The combine also uses one 128-thread CTA per query head, with one thread per
output dimension. It stages six float2 metadata entries in shared memory,
finds the global maximum, rescales six numerator/denominator partials, divides,
and writes fp32. A separate Q8 quantizer then prepares the O projection input.

### Measured service

| region, 36 calls | tinygrad | llama PDL-off | tinygrad debt |
|---|---:|---:|---:|
| score bodies | 222.656 us | 162.948 us adjusted | 59.708 us |
| combine bodies | 50.272 us | 37.057 us | 13.215 us |
| score + combine | 272.928 us | 200.005 us | 72.923 us |

The current tinygrad score-to-combine and combine-to-O timestamp gaps are
zero. The debt is therefore kernel service under production conditioning, not
an exposed launch gap between these nodes.

The combine residual is real but small. Wider-combine and earlier single-stage
constructions were exact/semantically valid but wall-negative or structurally
slow. The current 128-lane wide-route combine already captures the useful
geometry change. No untested 50-us-class combine claim remains.

## Lifecycle 6: cache and working-set conditioning

The installed tinygrad S8 score is about 4.54 us/layer hot and 6.18 us/layer in
the full graph. Llama is 3.81--3.87 us/layer hot and 4.53 us/layer in its
PDL-off production graph. The earlier statement that the hot bodies were tied
compared tinygrad hot with llama production and is withdrawn.

A controlled 96-MiB read working set reproduces most of that transition. An
immediate Q/K/V prefix without the large prior working set is neutral. Llama
also slows after the same conditioner, but by about half as much. This proves:

1. the hot score body is a real but minority 40--44% of the production score gap;
2. immediate producer dependencies are not the current problem;
3. production conversion is causal and owns the remaining 56--60%;
4. split count alone does not explain cold sensitivity, because llama S6 and
   forced S8 react similarly to the conditioner.

The active-horizon counters then identify the concrete structural component.
Changing tinygrad from S8/1024 to S6/768 while retaining the 1024 cache stride
reduces DRAM, L2, L1, and executed instructions by about 25%, with unchanged
register and shared-memory resources. Explicit load gating, separate K/V
allocations, and address coloring did not beat this construction.

Blanket `evict-first` weight loads are also closed. The cache-policy primitive
can protect a synthetic KV footprint, but Q/K/V-only production conversion was
small and applying it to all dense weights regressed the large projections by
destroying useful intra-kernel reuse.

### Exact production entry-hop ledger

The synthetic capacity result has now been grounded in the exact common-layer
production sequence. Captured cubins, launch arguments, and buffer identity
were replayed cumulatively between a score reheat and the timestamped score.

| last completed hop | score | delta from hot | marginal delta | score DRAM reads | L2 read hit |
|---|---:|---:|---:|---:|---:|
| hot | 4.576 us | reference | reference | 0.001 MB | 100.00% |
| previous gate/up | 4.768 us | +0.192 us | +0.192 us | 0.003 MB | 99.98% |
| previous FFN-down | 5.600 us | +1.024 us | **+0.832 us** | **3.692 MB** | **78.79%** |
| shared provider | 5.600 us | +1.024 us | +0.000 us | 3.702 MB | 78.80% |
| Q projection | 5.872 us | +1.296 us | +0.272 us | 4.232 MB | 75.32% |
| paired K/V projection | 5.920 us | +1.344 us | +0.048 us | 4.234 MB | 75.31% |
| Q completion | 5.824 us | +1.248 us | -0.096 us | 4.217 MB | 75.37% |
| K/V completion, full entry | 5.776 us | +1.200 us | -0.048 us | 4.213 MB | 75.43% |

Gate/up streams 54 MiB of packed weights. FFN-down adds 39.375 MiB, bringing
their cumulative weight footprint to 93.375 MiB before other live data on a
96-MiB L2. That is the production capacity crossing. Q adds displacement
pressure, while the completion hops touch current-token state and modestly
reheat the target.

The score itself executes the same 1,097,088 instructions and moves the same
approximately 22.09 MB through L1 and 17.46 MB through L2 in every counter
arm. Only the serving level changes. FFN-down is therefore the first causal
eviction hop; the score is the later payment hop. The exact full-entry penalty
is 43.200 us/token. Matching llama's measured S8 conditioner sensitivity
exposes a narrower 21.312 us/token, but neither amount is booked without a
line/reuse-aware policy that passes token wall.

The matched llama production-order replay shows why cache bytes cannot be
converted directly into time. After gate/up plus down, llama's target refetches
about 3.18 MB from DRAM but pays only 0.128 us. Q then adds 0.832 us with almost
no additional target DRAM traffic, and Q completion recovers 0.192 us. Its pure
capacity response is flat, crosses a sharp knee between 90 and 92 MiB, then
plateaus through 108 MiB. Llama uses normal CUDA cache replacement rather than
an explicit per-layer clear. Its corrected production-flags matched-prefix
penalty is 0.736 us/layer, leaving tinygrad a 0.464-us/layer, or
16.704-us/token, exposure. That would cap the installed endpoint at 245.230
tok/s, but zero is booked because
the small llama completion hops are modeled and a candidate still must reduce
both target DRAM reads and service time.

The complete hot-to-production conversion is larger than the selected prefix:
tinygrad falls 36.35%, while llama falls 16.9--18.9%. That conversion
difference is 33.500--35.804 us/token and owns 56--60% of the measured
59.708-us/token score gap. Direct production NCU confirms llama is cold, not
cache-preserved: 3.166 MB of target DRAM reads and a 75.58% L2 hit rate. The
corrected prefix replay matches that state and is within 0.082 us/layer of
llama's production mean.

## Lifecycle 7: runtime scheduling and PDL

### tinygrad native HCQ graph

The graph carries direct data dependencies through Q completion, KV cache
completion, score, combine, and O. Score-to-combine and combine-to-O start at
zero measured gap. A native split-phase attempt at the KV-ready boundary did
not recover meaningful wall time.

### llama CUDA graph

With PDL disabled, the retained graph uses ordinary launch-completion edges.
With PDL enabled, producer kernels issue a launch-completion signal near entry
and consumers synchronize before reading their inputs. That can allow a
consumer launch to become resident before its producer completes.

PDL is not llama's Flash-body advantage. The retained score-start to
combine-end sum is 207.746 us/token with PDL off and 212.678 us/token with PDL
on. PDL helps llama elsewhere in the complete token lifecycle, but copying it
around tinygrad's existing Flash pair would not create llama's kernel service
rate.

## Token translation and accounting boundary

The endpoint gap is 72.781 us/token, while the gross score+combine difference
is 72.923 us/token. This near equality does not mean every Flash microsecond is
independently recoverable. Tinygrad also has roughly 80 us/token of lifecycle
wins elsewhere that offset its norm, Q/O, and vocabulary losses. The complete
device and wall ledger must be re-run after any Flash promotion.

At the current endpoint:

- eliminating the entire gross Flash difference is only a parity ceiling;
- the measured S6 graph near depth 512 is worth about +0.57 tok/s locally;
- the measured S6 policy averaged over tokens 513 through 768 is worth about
  +0.89 tok/s for that region;
- amortized over a full continuation from 513 through 1024, switching to S8
  after 768 is estimated at about +0.445 tok/s.

Only the fixed S6 brackets are measured local conversions. The full selector
has now been bracketed and is a no-go: cold execution contains a deterministic
lazy-capture stall, while a prewarmed reverse bracket regresses 10.676
us/token, or 0.634 tok/s. The full-window estimate is therefore not bookable.

## Closed, open, and unmeasured surfaces

| surface | disposition | reason |
|---|---|---|
| scalar versus wide K/V loads | closed by promotion | wide loads passed counters, semantics, and token wall |
| one-query-head vector topology | closed by promotion | installed and positive |
| S8 empty upper partitions | conversion-closed for current selector | causal counters and fixed-S6 wall pass, but cold selector stalls and prewarmed selector regresses |
| S7/896 horizon | closed | production wall regression |
| smallest bucket at every 128 tokens | closed as a policy | S5 passes but loses to safe S6; S7 regresses |
| producer-ready launch-ahead | closed | visible timestamp gap did not convert to wall |
| score-to-combine launch gap | closed | measured zero |
| combine-to-O launch gap | closed | measured zero |
| wider combine | closed | device improvement did not convert on the prior route; current route already uses 128 lanes |
| single-stage score+combine | closed for tested mapping | resource/underfill construction was strongly negative |
| separate K/V allocation | closed | exact but slower |
| explicit invalid-load gating | closed | faster hot, slower cold |
| base-address coloring | closed at tested offsets | movement, no portable winning rule |
| blanket streaming/evict-first weights | closed | synthetic pass, production no-go |
| previous FFN-down creates Flash residency tax | mechanism-resolved | exact production prefix and steady-state counters identify the first cache-knee crossing |
| llama producer replacement law | mechanism-resolved | normal replacement is flat, crosses a 90--92 MiB knee, then plateaus; residency bytes and paid service are not one-to-one |
| line/reuse-aware FFN-down streaming | mechanism pass, token construction closed | payload-only eviction removes the entry penalty, but the split-pointer construction regresses the token wall by 69.877 us |
| cross-head shared-memory K/V reuse | closed for tested construction | bit-exact QG2 halves loading warps but adds 0.704 us/layer score service |
| residual equal-geometry cold-service difference versus llama | measured exposure, no construction | 21.312 us/token ceiling after the exact entry reconciliation; zero booked |
| other dense shapes and capacities | unqualified | installed route is deliberately fail-closed |

## Next sequence

1. Keep the installed S8 endpoint unchanged and book zero selector recovery.
2. Reopen cache admission only through an ABI-preserving per-access semantic or
   active-K/V residency mechanism; the split-pointer construction is closed.
3. Require lower full-entry Flash DRAM reads and native latency without slower
   gate/up or FFN-down service before changing the production token path.
4. Keep additional horizon buckets, combine rewrites, and scheduling changes
   closed unless new evidence changes their accounting boundary.
5. Re-run the full lifecycle ledger only after a new primitive-to-token
   construction passes exactness, graph census, and reverse wall.

This sequence treats the active-horizon policy as a learned wall and forces
further work to explain the equal-horizon residual rather than repeat the
fixed-S6 microgate.

## Evidence

Primary source surfaces:

- `tinygrad/llm/model.py`
- `tinygrad/llm/decode_routes.py`
- `tinygrad/llm/flash_decode_attention.py`
- `/home/ubuntu/env/llama.cpp/src/llama-graph.cpp`
- `/home/ubuntu/env/llama.cpp/src/llama-kv-cache.cpp`
- `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/fattn.cu`
- `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/fattn-common.cuh`
- `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh`

Retained measurements and result records:

- `docs/task_workflow/evidence/nv-flash-causal-reopen/post-wide-installed-ledger.json`
- `docs/task_workflow/evidence/nv-flash-causal-reopen/post-wide-installed-ledger/production.profile.jsonl`
- `docs/task_workflow/evidence/nv-flash-causal-reopen/llama-flash-causal-summary.json`
- `docs/task_workflow/evidence/nv-lifecycle-recovery-tests-20260826/llama-pdl-ab/pdl-off-dag.json`
- `docs/task_workflow/evidence/nv-token-lifecycle-vs-llama-20260825/llama-dag.json`
- `docs/task_workflow/evidence/nv-attention-token-lifecycle-reopen-20260824/attention-edge-ledger.json`
- `docs/task_workflow/evidence/nv-flash-wide-conditioning/priority1-conditioning-r3.json`
- `docs/task_workflow/evidence/nv-llama-flash-matched-conditioning-20260826/`
- `docs/task_workflow/evidence/nv-flash-kv-layout-matrix-20260826/`
- `docs/task_workflow/evidence/nv-flash-active-horizon-selector/selector-r9.json`
- `docs/task_workflow/evidence/nv-flash-active-horizon-selector/selector-prewarmed-r9.json`
- `docs/task_workflow/evidence/nv-flash-entry-hop-ledger/entry-native-r1.json`
- `docs/task_workflow/evidence/nv-flash-entry-hop-ledger/entry-hop-summary.json`
- `docs/task_workflow/output/nv-flash-entry-hop-ledger-result.md`
- `docs/task_workflow/output/nv-flash-entry-hop-vs-llama-result.md`
- `docs/task_workflow/output/nv-flash-kernel-to-production-conversion-result.md`
- `docs/task_workflow/output/nv-llama-flash-causal-reopen-result.md`
- `docs/task_workflow/output/nv-llama-flash-wide-load-result.md`
- `docs/task_workflow/output/nv-flash-wide-production-conditioning-result.md`
- `docs/task_workflow/output/nv-flash-active-horizon-result.md`
- `docs/task_workflow/output/nv-flash-active-horizon-selector-result.md`
- `docs/task_workflow/output/nv-attention-token-lifecycle-reopen-result-20260824.md`

Verification: 120 focused Flash route, descriptor, admission, ABI, renderer,
and production tests passed; four target-dependent tests skipped.
