# Active-boundary exhaustive test matrix

## Current conclusion

The isolated native-versus-CUDA-graph cadence difference is real, bitwise, and
survives stable descriptor reuse. It is not caused by native graph rebuilding,
missing dependent-QMD prefetch, pushbuffer word count, leading barriers, final
completion placement, or sample order.

The main mechanism is now identified and promoted. Tinygrad requested an L1
system-memory barrier at every grid QMD, including internal edges already
ordered by `dependent_qmd0`. CUDA graph's hardware-accelerated dependency path
does not charge that full barrier at every node. Removing it only from QMDs
that have a same-queue dependent successor recovers 50.501 us in the matched
208-call bridge and 67.530 us/token in the production reverse bracket. The
final QMD retains its system membar.

## Executed matrix

| ID | theory | test | result | disposition |
|---:|---|---|---|---|
| 1 | pointer/scalar ABI mismatch | captured ABI assertions | original scalar omission found and fixed | closed defect |
| 2 | semantic divergence | nonzero final-buffer oracle | 46/46 logical hashes bitwise | pass |
| 3 | arm-order or session drift | both process orderings | graph advantage survives | closed |
| 4 | ordinary CUDA is faster | ordinary launch slope | 3.41--3.45 us/node, much slower | closed |
| 5 | CUDA graph steady replay | first versus steady | first replay pays up to 190.7 us at 512 nodes | measured |
| 6 | fresh native descriptors are unfair | bound stable native replay | 2686.277 versus graph 2613.920 us | closed; 72.357-us gap survives |
| 7 | missing QMD prefetch | prefetch 0/1 | off costs 127.230 us at 208 calls | current prefetch=1 qualified |
| 8 | leading timeline wait | wait-only/none | at most about 2 us movement | closed |
| 9 | leading cache barrier | barrier-only/none | at most about 2 us movement | closed |
| 10 | final QMD release is slow | QMD versus pushbuffer completion | pushbuffer about 1--3 us worse | closed |
| 11 | native emits one PB launch per node | PB word census | dependent chain is 20 words for 1--128 QMDs | closed |
| 12 | direct PCAS is the solution | break every dependent link | synthetic -530 us, but permits illegal overlap/races | rejected as production claim |
| 13 | production has large unused concurrency | retained real DAG critical path | two queues within 2.176 us of dependency critical path | closed as large pool |
| 14 | QMD packing | copied 512-B compact arena | hangs at long chain; construction invalid | no mechanism conclusion |
| 15 | whole CUDA backend wins | matched dense route smoke | CUDA 5.657 ms, NV 4.195 ms, identical token hash | closed |
| 16 | single-cubin artifact | 12-cubin 208-call bridge | corrected graph advantage 68.460--78.356 us | closed |
| 17 | fresh/bound native command buffer | bound queue and fixed QMD addresses | gap remains 72.357 us, 46/46 hashes | closed |
| 18 | pure scheduler versus kernel interaction | matched 208-node no-op chain | native 141.028, CUDA 104.128 us | pure 36.900-us scheduler floor proven |
| 19 | redundant QMD cache invalidations | retain only first invalidation | no material no-op or real-population movement | closed |
| 20 | per-grid L1 system membar | clear only on internal dependent QMDs | 2686.277 -> 2635.776 us; 46/46 hashes | proven and promoted |
| 21 | producer/consumer visibility | 256 alternating pairs, 511 internal edges | zero errors; 459.103 -> 332.855 us | semantic pass |
| 22 | production token correctness/wall | control/candidate/control, reps=9 | identical hashes; -67.530 us/token | production pass |

## Exhaustive remaining list

| ID | question | smallest admissible test | status |
|---:|---|---|---|
| 23 | CUDA versus native remaining ~20-us bridge residual | matched active-body and QMD field attribution | open; main 50.501 us recovered |
| 24 | graph upload flag alone | explicit upload versus warmed ordinary instantiate | closed: 2615.328 versus 2611.936 us at 208, no material movement |
| 25 | graph rebuild/update | rebuild, update, and reuse arms | low priority; stable reuse result already isolates steady path |
| 26 | exact graph boundaries | one/four/five matched bridge executables | open, bounded small by prior merge losses |
| 27 | unique per-layer weights | allocate physical weight population and rerun | open; current interleaved working set exceeds L2 but aliases pointers |
| 28 | instruction/program residency | alternating and unique-program conditioning | partially closed by 12-cubin bridge |
| 29 | two-GPFIFO exact topology | replay captured production assignments/edges in both front ends | open after capture diff |
| 30 | native hardware-only timing | bracket chain with hardware timestamp signals | open; current native drain includes CPU completion observation |
| 31 | clock telemetry | retain SM/memory clocks per arm | open; order reversal already passes |
| 32 | production native candidate | relax internal dependent-QMD membar only | complete, Blackwell default with rollback |
| 33 | production exactness | adversarial visibility plus token stream | complete |
| 34 | production wall | reps>=7 reverse bracket | complete: -67.530 us/token, +3.930 tok/s in bracket |
| 35 | dense-model generality | second dense Q8 0.6B population | -87.145 us/token; semantic comparison inconclusive because controls also diverged |
| 36 | vocabulary body | independent service-rate construction | open, measured 15.800-us ceiling |
| 37 | O body | independent cleanup | open, measured 3.007-us ceiling |

## Command-stream capture requirement

The `mmap64` correction resolved active ownership without a kernel module.
Exactly CUDA channel 1 advances during graph replay. Its first replay uploads a
large resident graph object; steady replay publishes three GPFIFO entries
(start event, compact graph launcher, end event). This agrees with NVIDIA's
documented hardware acceleration for graph dependencies, but the transferable
property was found by field isolation rather than copying a private graph
descriptor: omit the internal `CWD_MEMBAR_TYPE_L1_SYSMEMBAR` and preserve it
at external completion.

### RM ownership census

The non-invasive `LD_PRELOAD` RM capture now resolves allocation classes,
returned handles, GPFIFO offsets and sizes, USERD memory handles and offsets,
and CPU/GPU mapping requests for both arms. Native allocates three 65,536-entry
Blackwell GPFIFO channels from one 3 MiB backing object. CUDA initializes
sixteen 1,024-entry channels from one 2 MiB backing object, with USERD slots
spaced by 12 KiB. This is a real structural difference, but it is not yet a
causal result: the CUDA driver can allocate infrastructure channels that the
graph never uses.

The attempted shortcut of treating RM's returned `pLinearAddress` as a CPU
mapping was rejected. It is not a safely dereferenceable userspace address on
this path. Active ownership therefore needs the already-scoped safe route:
correlate the actual mmap with the RM handle and put a task-scoped hardware
write watchpoint on `USERD + 0x8c`/the doorbell. Only the advancing channel's
GPFIFO/PB/QMD chain is admissible for IDs 18--20.

## Token-rate ledger

The production reverse bracket books 67.530 us/token. In that fresh current-tree
session it moved 4178.963 -> 4111.433 us/token, or 239.294 -> 243.224 tok/s.
Normalized onto the prior installed authority:

```text
4060.523 - 67.530 = 3992.993 us/token
1e6 / 3992.993    = 250.439 tok/s
```

The first pair is the directly measured current endpoint; the second is a
normalization for continuity with the earlier ledger, not a fresh 250.439
tok/s endpoint measurement.

## Evidence

- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/projection-bridge-nv-bound-r1.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/projection-bridge-cuda-bound-pair-r1.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/projection-bridge-nv-bound-prefetch0.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/projection-bridge-nv-bound-prefetch1.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/cuda-graph-first-vs-steady.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/nv-route-smoke.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/cuda-route-smoke.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/native-rm-summary.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/cuda-rm-summary.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/cuda-upload0-r1.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/cuda-upload1-r1.json`
