# Exact Q/O/K/V short-stream mechanism closure

## Decision

`CP_ASYNC_EXACT_BUT_SERVICE_NEGATIVE__CROSS_LAUNCH_PERSISTENCE_NEEDS_NEW_RUNTIME_SUBSTRATE`

No production route was edited and no recovery is booked.

The highest-information executable mechanism not equivalent to the closed
static stripe, multi-row, full-grid, queue, or PDL spellings was exact
global-to-shared asynchronous staging of each Q4_K row.  It directly tested
whether the measured long-scoreboard stall could be converted into useful
overlap while preserving the installed output association.

## Executed gate

The research-only gate compares the installed 4096x4096 Q4_K O body with one-
and two-stage `cp.async` candidates.  It uses three deterministic fixtures,
bitwise output comparison, R9 hot timing, 16-weight-copy rotated-cold timing,
SASS validation, build-resource capture, and NCU counters.

| arm | hot median | rotated-cold median | cold recovery | long scoreboard | dynamic instructions |
| --- | ---: | ---: | ---: | ---: | ---: |
| installed control | 5.015808 us | 9.690 us | -- | 74.69% | 4,800,512 |
| one-stage async | 5.230656 us | 10.169 us | -0.479 us | 63.66% | 4,960,256 |
| two-stage async | 5.427072 us | 10.584 us | -0.894 us | 65.65% | 5,464,064 |

Both candidates are bitwise exact on all fixtures.  Their cubins contain real
`LDGSTS.E.128` instructions, use 45 registers/thread, have no spills, and use
576 or 1152 bytes of shared memory.  DRAM bytes remain essentially unchanged
at about 9.47 MB.  Thus the causal intervention worked mechanically: it
reduced long-scoreboard exposure.  It nevertheless reduced delivered DRAM
throughput, added instructions/barriers/shared traffic, and lost in both hot
and rotated-cold service.  This closes row-local asynchronous staging, not
merely this timing sample.

## Remaining-mechanism audit

| mechanism | novelty/admissibility | disposition |
| --- | --- | --- |
| static inter-projection stripe | already exact-tested with cold/counters | closed |
| runtime atomic row queue inside one grid | changes assignment, not the physical stream count; uniform rows offer no measured imbalance to recover | equivalent to closed queue/stripe service class |
| persistent CTA inside one Q/K/V launch | still terminates at the same QMD and pays the next projection ramp | equivalent to closed full-grid/multi-row class |
| QMD PDL or multiple host queues | can overlap grid boundaries but does not create a resident consumer of future tasks | existing graveyard |
| row-local software prefetch/unroll | exact block unroll already closed; real async staging tested here | closed |
| CTA-cluster/TMA staging | weights are single-consumer row streams, so staging preserves DRAM bytes and adds exchange/synchronization; no cross-projection reuse contract exists | no distinct recoverable work identified |
| service resident across Q/K/V or layers | genuinely new because it can remove physical QMD/ramp boundaries | not executable with the current runtime ABI |

## Precise missing substrate

A genuinely new persistent service must remain resident while successive
producers publish work.  The current graph/runtime submits immutable per-call
kernel arguments through separate QMDs and exposes only host-authored ordering
(dependent-QMD chaining, queue semaphores, and optional PDL latches).  It has no
device-visible task-ring ABI through which a producer can publish a descriptor
containing weight pointer, activation pointer, output pointer, format, shape,
and exact reduction contract to a resident consumer.  It also lacks the
required lifetime and progress contract: persistent-grid admission/residency,
release/acquire epochs, wrap-safe queue ownership, graph replay rebinding, an
abort/drain protocol, and a deadlock-safe reservation model alongside Flash
and FFN grids.

Without that substrate, a purported persistent microgate can only pre-fill a
finite queue before launch.  That is one fused/full-grid kernel and is not a
test of cross-launch ramp removal.  Building such a gate would repeat a closed
spelling under a new name.

## Evidence

- `async-r9-counters.json`: complete exact hot/cold/counter authority.
- `async-artifacts/gate.cu`: rendered control and candidate sources.
- `async-artifacts/gate`: executable cubin container.
- `async-artifacts/ptxas.txt`: register, shared-memory, barrier, and spill log.
- Research source: `extra/llm_research/decode/nv_q4k_o_async_pipeline_microgate.py`.
- Prior static stripe authority: `docs/task_workflow/evidence/nv-qokv-roofline-20260827/stripe-gate-r9.json`.
