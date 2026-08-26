# Dense token lifecycle versus llama

## Decision

The remaining gap is no longer an unidentified collection of slow GEMVs. At
the current dense d512 endpoint, the strongest measured discriminator is token
topology: llama overlaps a large amount of producer/consumer command occupancy,
while tinygrad's device work is almost completely serialized. A later causal
reconciliation corrects one interpretation below: flash score has a slower
installed interval, but its isolated tinygrad body is already faster than
llama. The stable directly actionable service deficit is the 4096-wide norm
family.

This is a lifecycle ledger, not a sum of isolated microbenchmarks. Concurrent
llama kernel durations include time while other kernels are resident and cannot
be interpreted as independent service time.

## Wall authority

Both sides use Qwen3-8B-Q4_K_M, CUDA, flash attention, generation at depth 512,
and twenty generated tokens per timed sample.

| implementation | wall authority | latency (us/token) | throughput (tok/s) | delta from tinygrad |
|---|---:|---:|---:|---:|
| tinygrad | fresh bounded pre-cliff d512 | 4,261.471 | 234.66 | -- |
| llama | fresh r7 official average | 4,021.721 | 248.71 | 239.750 us / 14.05 tok/s |
| llama | settled six-sample median | 3,995.396 | 250.29 | 266.075 us / 15.63 tok/s |

The official average is the comparison authority. The settled median is shown
because llama's first generation repetition was colder than the following six.

## Device topology

| implementation | graph nodes | raw node-duration sum (us) | device span/union (us) | overlapped duration mass (us) |
|---|---:|---:|---:|---:|
| tinygrad current production | current replay | 4,112.864 | 4,109.500 | 3.364 |
| llama current build | 762 | 5,013.179 | 3,896.695 | 1,116.484 |

This is the central finding. Tinygrad executes less total command occupancy,
but almost none of it overlaps. Llama's graph exposes much more command
occupancy than its elapsed span because producer and consumer kernels coexist.
The measured device-span gap is 212.805 us, close to the 239.750 us official
wall gap. Therefore the dominant known gap is device scheduling/topology, with
roughly 27 us left outside that direct span comparison (including profiling and
host-bracket differences). The full 1,116 us overlap mass is not a recoverable
speedup estimate; only the elapsed-span difference is admissible.

## Stable service-rate ledger

The table below uses the retained PDL-off llama oracle to remove concurrent
wait/residency inflation, reconciled against the current tinygrad production
replay. Positive values mean tinygrad spends more device time. It is the best
available like-for-like body ledger; the fresh current llama topology capture
is used separately above to establish the deployed concurrency behavior.

| lifecycle role | tinygrad (us) | llama PDL-off (us) | tinygrad delta (us) | disposition |
|---|---:|---:|---:|---|
| attn/ffn/final 4096 norm | 308.512 | 203.361 | +105.151 | largest stable service deficit |
| flash score | 235.680 | 162.948 | +72.732 | installed-lifecycle deficit; isolated body already faster |
| flash combine | 99.936 | 37.057 | +62.879 | body deficit, but current llama overlaps/waits here |
| Q projection + completion | 298.592 | 249.092 | +49.500 | body deficit; topology-sensitive |
| O projection | 305.440 | 259.809 | +45.631 | body deficit; topology-sensitive |
| Q head norm | 69.248 | 41.281 | +27.967 | smaller stable deficit; fused-body caveat |
| K head norm | 67.424 | 40.832 | +26.592 | smaller stable deficit; fused-body caveat |
| K/V projection + completion | 236.352 | 215.009 | +21.343 | small body deficit |
| vocabulary | 324.544 | 303.908 | +20.636 | small tail deficit |
| gate/up GEMV | 1,273.920 | 1,268.370 | +5.550 | effectively tied; not the next lever |
| misc / embedding | 7.264 | 2.784 | +4.480 | immaterial |
| down GEMV | 853.248 | 855.817 | -2.569 | tinygrad already faster |
| rope + K/V store | 2.624 | 92.866 | -90.242 | tinygrad fusion wins |
| activation quantization | 30.080 | 145.120 | -115.040 | tinygrad fusion wins |

The rows close exactly: tinygrad is +234.610 us in the PDL-off body ledger.
The positive rows total 442.461 us, offset by 207.851 us where tinygrad is
faster. Those positive deltas cannot all be booked because llama's deployed
graph overlaps work and tinygrad changes can shift the critical path.

The Q/K head-norm rows are not perfectly like-for-like: tinygrad's current
kernel also folds RoPE, and the K path can include the cache sink. They remain
useful as a bounded target family, not as separately recoverable sums.

## What is known, and what is closed

Known gaps:

1. **Topology/overlap is the dominant deployed difference.** The current
   traces directly measure a 212.805 us device-span advantage for llama.
2. **4096-wide norm service is the largest stable body deficit.** This is
   distributed across the token, so any intervention must be tested at wall.
3. **Flash score is an installed-lifecycle deficit, not a body deficit.** Its
   isolated tinygrad body is faster than llama; the excess appears only under
   production conditioning/serialization.
4. **Head norms and vocab are real but individually small.** Together they are
   useful only after a clean wall bracket or as enablers of a topology change.
5. **Q/O/combine have body deficits but are concurrency-sensitive.** Their raw
   current llama durations become longer, not shorter, under overlap; raw Nsys
   duration is therefore not an admissible optimization target for these rows.

Closed or deprioritized:

- Gate/up is approximately at llama's wait-free service rate.
- Down is already slightly faster.
- Activation quantization and RoPE/KV-store work are already structurally
  cheaper in tinygrad because of fusion.
- Raw llama command-duration wins caused by concurrent residency are not
  kernel-service wins and must not be chased as such.

## Ranked next tests

1. Reconstruct the current llama overlap by layer and edge, then identify the
   exact producer/consumer pair(s) responsible for the 212.805 us span gap.
   The pass condition is reduced tinygrad device union and wall time, not more
   overlap mass by itself.
2. Run a full-token 4096-norm intervention gate. First prove a representative
   body improvement, then require a repeated wall bracket. Maximum accounting
   exposure is about 105 us; the realistic booking is whatever reaches wall.
3. Reopen flash score only with a construction that changes production
   conditioning or useful overlap. A score arithmetic rewrite is not supported
   by the isolated-body evidence.
4. Revisit Q/O/combine only with edge-conditioned timing. A candidate must
   shorten the critical span or enable producer/consumer coexistence; isolated
   duration alone is insufficient.
5. Treat head norms and vocab as cleanup targets after the topology and first
   two service-rate tests, unless one supplies the missing dependency substrate.

## Evidence

- `docs/task_workflow/evidence/nv-token-lifecycle-vs-llama-20260825/llama-unprofiled.json`
- `docs/task_workflow/evidence/nv-token-lifecycle-vs-llama-20260825/llama-dag.json`
- `docs/task_workflow/evidence/nv-token-lifecycle-vs-llama-20260825/lifecycle-roles.json`
- `docs/task_workflow/evidence/nv-token-lifecycle-vs-llama-20260825/lifecycle-raw-command-roles.json`
- `docs/task_workflow/evidence/nv-post-horizon-d512-ledger-20260825/profile-ledger/production.profile.jsonl`
