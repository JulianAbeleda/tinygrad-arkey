# NV dense decode byte and topology wall audit

Date: 2026-08-25

Model: Qwen3-8B Q4_K_M

Route basis: installed `e9c3e4edd` device ledger plus current `619cc7688` no-go closure

## Verdict

> Update, 2026-08-25: the first-ranked shared-Q8 Q4/Q4/Q4 triple producer was
> tested through bit-exact, hot, two independent cold, and NCU counter gates.
> The 1,024- and 2,048-CTA grammars are cold-negative. The surviving 4,096-CTA
> grammar is blocked on a typed workgroup-uniform region that can safely contain
> the exact K/V merge barrier. See
> `nv-shared-q8-qkv-producer-result-20260825.md`. No recovery is booked.

The wall is neither **bytes alone** nor **scheduler topology alone**. It is a
coupled **quantized-streaming issue/rate wall**:

- exact projection and vocabulary payload is 4.671 GB per token;
- weight-consuming kernels occupy 3.281 ms, 79.8% of the 4.110 ms device union;
- large bodies reach roughly 1.47-1.63 TB/s, while Q/O reach about 1.11-1.13
  TB/s and the small K/V producer pool reaches about 0.80 TB/s;
- lowering loaded SM clock from 2850 to 1995 MHz with memory fixed at 14001
  MHz increases wall latency by 990 us, or 23.25%;
- the dependency DAG exposes 321.760 us of arithmetic parallel slack, but it is
  almost entirely the K/V branch already assigned to the second queue, and
  production realizes only 3.536 us of resident overlap.

The decisive correction is that streaming bytes do not imply a pure DRAM
bandwidth wall. Dequantization, address/load issue, and reduction work on the
SM determine how quickly those bytes can be consumed. The next lever should
therefore increase the production rate of the small projection population by
changing its physical work granularity, while keeping exact bytes and output
ownership accounted. Generic queue additions and generic lossless weight
compression are not first-order answers.

## 1. Exhaustive exact weight-byte ledger

The GGUF tensor table gives exact packed payload bytes. Every projection and
the vocabulary matrix is consumed once per generated token; the full token
embedding table is resident storage but only one row is indexed and is not
counted as a decode stream.

| payload role | tensors | bytes/token |
| --- | ---: | ---: |
| FFN gate | 36 | 1,019.216 MB |
| FFN up | 36 | 1,019.216 MB |
| FFN down | 36 | 1,252.786 MB |
| attention Q | 36 | 339.739 MB |
| attention O | 36 | 339.739 MB |
| attention K | 36 | 84.935 MB |
| attention V | 36 | 104.399 MB |
| vocabulary output | 1 | 510.505 MB |
| **projection + vocabulary total** | **253** | **4,670.534 MB** |

Norm weights add only 1.233 MB and remain cache-sized. The 350.061 MB token
embedding table is not a full-matrix decode read. Activation vectors,
residuals, scales, and logits are small beside the projection payload; fusing
them can remove launches and cache traffic but cannot materially change the
4.67 GB weight stream.

### Weight bytes joined to current device time

| family | exact payload | current time | aggregate payload rate |
| --- | ---: | ---: | ---: |
| gate/up | 2,038.432 MB | 1,273.856 us | 1,600 GB/s |
| down | 1,252.786 MB | 852.448 us | 1,470 GB/s |
| vocab | 510.505 MB | 313.856 us | 1,627 GB/s |
| Q | 339.739 MB | 299.584 us | 1,134 GB/s |
| O | 339.739 MB | 304.864 us | 1,114 GB/s |
| K/V | 189.334 MB | 235.904 us | 803 GB/s |
| **total** | **4,670.534 MB** | **3,280.512 us** | **1,424 GB/s** |

These are payload/time accounting rates, not a replacement for DRAM counters.
Current isolated-cubin counters independently show full payload-sized cold
reads for the major rows. A true whole-token NCU window was attempted twice,
but NCU cannot observe the production NV backend's direct submission context.
That is retained as an instrumentation wall; no zero-byte or inferred counter
result is substituted.

At the observed aggregate rate, removing 100 MB has a gross arithmetic value
of about 70 us, and removing one percent of weight payload has a gross value
of about 33 us. Those are ceilings before decompression or changed-kernel cost.

### Is exact lossless recompression a large hidden pool?

No evidence supports it. Offline zlib level 1 saves 72.008 MB (1.542%); level
9 saves 76.131 MB (1.630%). This is not an optimal-codec bound, GPU
representation, or speed claim; it is a feasibility diagnostic. Even granting
the stronger result for free would be only about 53 us gross at the measured aggregate rate, before an
already SM-sensitive path pays decompression cost. The packed Q4_K/Q6_K
payloads are therefore not hiding a large generic lossless-compression win.

Larger byte reductions require changing the numerical representation. The
1.316 GB of Q6 projection/vocab payload is the obvious non-bit-exact surface;
requantizing it could be material, but it changes model outputs and belongs in
a quality-evaluated lane, not the exact promotion lane. KV-cache compression
also changes numerics and is context-length dependent; at depth 512 it is not
the dominant 4.67 GB pool.

## 2. Exhaustive dependency/topology ledger

The captured full-token DAG has 454 nodes, 1,129 exact RAW/WAR/WAW edges, no
unknown dependencies, and no skipped spans.

| construction | span | arithmetic saving versus serial |
| --- | ---: | ---: |
| serial node sum | 4,178.400 us | 0 |
| ideal two-queue list schedule | 3,859.776 us | 318.624 us |
| ideal three-or-more-queue schedule | 3,856.640 us | 321.760 us |
| dependency critical path | 3,856.640 us | 321.760 us |

Two queues are sufficient in the resource-free model; queues 3 through 16 add
nothing. This arithmetic model deliberately ignores bandwidth contention,
front-end handoffs, and wait cost, so 321.760 us is a topology ceiling rather
than a wall forecast.

### Where every off-path microsecond lives

| family | total node mass | off-path mass | zero-slack mass |
| --- | ---: | ---: | ---: |
| K/V projection producers | 246.752 us | 231.456 us | 15.296 us |
| K norm/RoPE/cache-ready | 142.976 us | 70.528 us | 72.448 us |
| Q projection | 311.488 us | 11.584 us | 299.904 us |
| startup elementwise | 87.520 us | 8.192 us | 79.328 us |
| every other major family | 3,389.664 us | 0 | 3,389.664 us |

This closes the topology question precisely: the ideal slack is the attention
K/V side branch. Gate/up, down, O, flash score/combine, vocab, and the main Q
path are all on the dependency spine.

Production already assigns all 46 physical K/V producers to queue 1 and the
Q/spine work to queue 0. Prior cold physical gates proved about 2 us/layer of
Q-versus-K/V overlap in isolation, but reapplying installed placement was a
structural no-op and reversing submission order lost 8.176 us/token. The
KV-ready-to-score split-phase construction was also wall-neutral. Thus the
missing 318 us is not an unimplemented queue count; it is an ideal schedule
that assumes competing quantized streamers share resources for free.

## 3. Causal wall discriminator: fixed-memory SM-clock A/B/A

The production graph was run at depth 512 for seven settled windows per arm.
Memory remained at 14001 MHz and every token-stream hash matched.

| arm | loaded SM clock | median wall |
| --- | ---: | ---: |
| high A | 2850 MHz | 4.261669 ms/token |
| low | 1995 MHz | 5.248684 ms/token |
| high C | 2850 MHz | 4.255786 ms/token |
| high midpoint | 2850 MHz | 4.258728 ms/token |

A 30.0% measured loaded-SM reduction caused a 23.25% latency increase. A
two-term inverse-clock fit assigns about 54% of the high-clock wall to the
SM-frequency-scaled term. That number is diagnostic, not a forecast: SM clock
also governs load/address issue and therefore achieved DRAM rate. What it
decisively rejects is a pure fixed-bandwidth byte floor.

## 4. What is actually closed

- **More queues:** closed. Two queues reach the ideal DAG schedule; production
  already uses two for the only material independent branch.
- **Submission-order changes:** closed by exact wall-negative brackets.
- **Generic PDL/readiness:** closed for the measured Q/K-to-score and other
  broad placements; timestamp wait is not recoverable wall.
- **Gate/up-to-down and score-to-combine-to-O boundary gaps:** closed at zero
  measured edge delay.
- **Instruction-only Q/O block unrolling:** closed by the latest exact
  full-token loss despite a tiny cold-kernel gain.
- **Generic exact lossless compression:** low ceiling and wrong cost balance;
  not a first-order route.

## 5. Remaining mechanisms, ranked

### 1. Q+K/V triple-output producers — test next

This is the clean intersection of topology and rate. Today Q and the already
paired K/V producer are separate physical bodies. A triple producer can keep
each output's exact arithmetic order while:

- combining 9.44 MB Q with the smaller K/V stream into a roughly 14-15 MB
  launch;
- reusing the fp16 activation or existing shared-Q8 provider;
- eliminating one projection launch per admitted layer;
- converting unrealized branch slack into a larger body with a better chance
  of sustaining the rate already reached by gate/up and vocab.

The complete Q+K/V pool is 529.488 us/token for 529.072 MB. If a triple body
raises only that pool to 1.2, 1.4, or 1.6 TB/s, the arithmetic recoveries are
about 89, 152, or 199 us/token respectively. None is booked. Test four exact
topologies separately: ordinary Q4/Q4/Q4, shared-Q8 Q4/Q4/Q4, shared-Q8
Q4/Q4/Q6, and ordinary mixed Q4/Q4/Q6. Start with shared-Q8 Q4/Q4/Q4 because
the provider and K/V pair contracts already exist.

Admission order remains: bit-exact outputs, cold complete-span timing and
counters, exact output ownership with no transports, current-route census,
then reps>=9 A/B/A full wall. A miss with an opaque output or incomplete
population is an accounting wall and must be repaired; a complete negative
wall closes that topology.

### 2. Better single-body issue/dequant scheduling

Continue only where current cold counters show a rate deficit and a causal
stall mechanism. Q/O/K/V remain the rate-deficient bodies; gate/up and vocab
are already near the demonstrated large-body rate. The failed Q/O u4 spelling
closes that construction, not all instruction/rate work. New candidates must
change load-level parallelism, integer unpack/dequant scheduling, or CTA work
aggregation enough to raise cold rate—not merely reduce L1/L2-hit loads.

### 3. Numerical byte reduction as a separate lane

Requantizing Q6 weights or compressing KV cache can remove materially more
bytes, but cannot preserve current token hashes. Pursue only with an explicit
quality/evaluation contract. It must not be mixed into the exact parity
ledger.

### 4. Specialized exact weight codec — low priority

Only revisit if a GPU-decodable format demonstrates substantially better than
the small offline zlib saving and its decode cost passes the SM-clock-aware
roofline. The current evidence gives it less expected value than triple-body
rate aggregation.

## Bottom line

Bytes define the size of the problem; SM issue/dequant and projection
granularity define how quickly we pay for those bytes. The scheduler has
already exposed the only large independent branch, but physical overlap cannot
evade shared-resource service. The next clean bet is therefore not “more
overlap” or “compress everything.” It is an exact Q+K/V triple producer that
turns small low-rate streams into one larger high-rate body, followed by the
same full-token wall authority used for every promotion.

Follow-up result (2026-08-25): this bet passed. A typed uniform full-grid
region with separate caller-owned Q/K/V outputs is bit-exact and recovered
16.119 us/token end-to-end in the 7-repetition production bracket (about
0.84 tok/s at the measured endpoint). The gain is service-rate/topology only;
it does not claim DRAM-byte reduction.

Evidence:

- `docs/task_workflow/evidence/nv-byte-topology-wall-audit-20260825/accounting.json`
- `docs/task_workflow/evidence/nv-byte-topology-wall-audit-20260825/queue-sweep.json`
- `docs/task_workflow/evidence/nv-byte-topology-wall-audit-20260825/sm-clock-sensitivity-r7.json`
- `docs/task_workflow/evidence/nv-byte-topology-wall-audit-20260825/weight-zlib-level1.json`
- `docs/task_workflow/evidence/nv-byte-topology-wall-audit-20260825/weight-zlib-level9.json`
- `docs/task_workflow/evidence/nv-byte-topology-wall-audit-20260825/ncu-instrumentation.json`
