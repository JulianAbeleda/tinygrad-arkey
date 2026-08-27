# NV Flash topology transfer audit

## Decision

The transferable part of llama's Flash topology is now narrower and more
useful than "port the kernel." Tinygrad can and already does express the outer
topology that matters: one query head per CTA, four warps, 128 token rows per
CTA, 8-lane K-dot groups, aligned 16-byte K/V loads, online softmax, FP32 PV,
a physical-extent-derived split count, and a 128-lane final combine.

The untransferred part is the inner warp grammar: llama leaves one score owned
by one lane, exchanges the resulting probabilities warp-locally, and emits a
dense K/V load/consume cadence. Tinygrad's current emitter retains eight
scores per 8-lane group and its attempted shared-ownership spelling requires
heavier synchronization. That is the remaining construction target.

## Geometry-normalized causal test

The query-head ownership axis was tested without changing total launch supply.
Every arm launched 192 CTAs and 768 warps; only query heads per CTA and the
compensating split count changed.

| arm | heads/CTA | splits | CTAs | median tile+combine | numerical result |
|---|---:|---:|---:|---:|---|
| QG1/S6 | 1 | 6 | 192 | 71.533 us | reference, bitwise |
| QG2/S12 | 2 | 12 | 192 | 71.606 us | bitwise |
| QG4/S24 | 4 | 24 | 192 | 71.596 us | one FP16 word differs |

QG1 is the fastest arm. More GQA sharing does not recover service when CTA and
warp supply are held constant. One-head-per-CTA is therefore an admissible
transfer and is already present in the research substrate; it is not the
unexplained performance gap.

## Transfer ledger

| llama mechanism | transfer status | evidence / next gate |
|---|---|---|
| one query head per CTA | transferred, positive | geometry-normalized QG sweep favors QG1 |
| four warps / 128 tokens per CTA | transferred | same outer work mapping in QG1/S6 substrate |
| physical-extent-derived partitions | transferred | S6 for padded 768-token extent |
| 8-lane K dot groups | transferred | same cooperative dot width |
| aligned 128-bit K/V requests | transferred and promoted | wide-K/V path and SASS/counter gates |
| online softmax + FP32 PV | transferred | same numerical class; NVIDIA llama is FP32 PV |
| 128-thread/six-part combine | transferred | lane-width-128 combine substrate |
| one score retained per lane | not yet faithfully transferred | current shared arm lowers registers but adds service cost |
| warp-local probability handoff | substrate gap | UOp spelling currently lowers through CTA-strength barriers; renderer-only syncwarp test remains negative |
| llama-like dense K/V issue cadence | not reproduced | prior source batching did not create llama's live-range/SASS topology |
| packed FP16 PV | do not transfer | not used by NVIDIA llama; numerical equivalence is not general |
| PDL | independent lifecycle lever | does not explain llama's Flash-body advantage |
| persisting L2 policy | independent residency lever | primitive recovery did not translate at token wall |

## Exact next construction

Do not alter the outer geometry again. Build a new inner-warp arm inside the
QG1/S6 substrate with these invariants:

1. retain one probability per lane rather than eight replicated score values;
2. distribute it with an explicit warp shuffle or a true warp-scoped shared
   handoff, without a CTA barrier;
3. keep FP32 PV, K/V bytes, 16-byte load width, grid, partial ABI, and combine
   unchanged;
4. require the intended SASS ownership/cadence before timing it;
5. judge isolated hot and production-conditioned cold service, then run a full
   token bracket only if the primitive wins.

This separates what can be copied from llama from what must be enabled in the
tinygrad lowering. A source-level feature that fails to produce the intended
SASS is an information wall, not evidence that the topology itself is bad.

## Inner-warp test outcome

The construction was subsequently completed in three forms.

| arm | exact | registers | static `LDG.E.128` sites | hot NCU | cold NCU | decision |
|---|---|---:|---:|---:|---:|---|
| control | reference | 96 | 32 | 5.376 us | 5.920 us | incumbent |
| one-owner warp shuffle | bit-exact | 70 | 8 | 5.280 us | 6.976 us | cadence collapsed |
| one-owner warp shared | bit-exact | 70 | 8 | 5.280 us | 7.232 us | cadence collapsed |
| one-owner shuffle + forced column unroll | bit-exact | 96 | 32 | 5.376 us | 5.984 us | faithful but no service win |

The forced-unroll arm is the decisive transfer test. It restores the incumbent
load-level parallelism and register population while retaining one score per
lane and native warp shuffles. Executed instructions fall from about 798k to
756k, but hot service ties and cold service is slightly worse. Thus the inner
ownership topology is expressible and transferable, but is not an isolated
performance lever on this kernel. The low-register versions appeared compact
only because ptxas serialized the column loops and removed independent load
sites; their cold regressions are not evidence for lower register pressure.

The installed V-tail schedule also failed to rescue the low-register arm:
it remained bit-exact but measured 5.536 us hot and 6.816 us cold versus
5.312/5.888 us control.

Tests requiring a positive primitive—geometry investment, full Flash-island
bracket, full-token bracket, and promotion—are therefore ineligible. No token
recovery is booked.

## Direct llama cadence reconciliation

The broader retained SASS experiment closes the idea that another generic
load-cadence substrate is still missing. The promoted two-argument launch
bound already gives tinygrad the compact K schedule and enough register budget:

| score body | registers | first 16 K-load span | next 16 V-load span | cold service |
|---|---:|---:|---:|---:|
| llama native | 158 | 32 instructions | 40 instructions | 6.144 us |
| tinygrad before compiler contract | 56 | 226 instructions | 310 instructions | 6.400 us |
| tinygrad promoted compiler contract | 96 | 29 instructions | 331 instructions | 5.888 us |

Tinygrad therefore already crosses llama on the matched score primitive. Its
V loads are staggered rather than clustered, but forcing 16 distinct V values
live is not a recovery: the retained scalar-Q test regresses 5.888 to 6.272 us
cold with identical bytes, sectors, register count, and zero spills. Moving
the wall earlier, splitting the inline assembly, and making the assembly
movable also lose. The compiler's staggered typed-load schedule is beneficial
for tinygrad's current probability/update grammar.

The fresh one-owner forced-unroll arm reaches the same conclusion from the
other direction: it reproduces independent load supply and reduces dynamic
instructions, yet ties hot service and slightly loses cold service. Thus the
score-body topology and cadence substrate are closed as the next token lever.

The surviving llama Flash advantage is no longer the score primitive. Both
runtimes launch a separate partial-state combine at the matched S6 topology,
but llama services that combine/lifecycle boundary faster. Future work should
target the combine implementation and its handoff under the existing faster
score body, not weaken the score body to imitate llama's visual V adjacency.

## Evidence

- `docs/task_workflow/evidence/nv-flash-topology-transfer-20260827/qgroup-normalized-r9.json`
- `docs/task_workflow/evidence/nv-flash-topology-transfer-20260827/warpprob-counter-r9.json`
- `docs/task_workflow/evidence/nv-flash-topology-transfer-20260827/warpprob-vtail1-counter-r9.json`
- `docs/task_workflow/evidence/nv-flash-topology-transfer-20260827/sharedprob-warp-counter-r9.json`
- `docs/task_workflow/evidence/nv-flash-topology-transfer-20260827/warpprob-unroll-counter-r9.json`
- `docs/task_workflow/evidence/nv-flash-load-wall-20260826/tiny-s6-scalar-lb-r11.json`
- `docs/task_workflow/evidence/nv-flash-load-wall-20260826/tiny-s6-scalar-vwall-r11.json`
- `docs/task_workflow/output/nv-flash-launch-bounds-causal-result.md`
- `docs/task_workflow/output/nv-flash-llama-ownership-topology-test-result.md`
- `docs/task_workflow/output/nv-llama-flash-causal-reopen-result.md`
