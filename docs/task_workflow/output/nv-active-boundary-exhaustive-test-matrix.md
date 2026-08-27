# Active-boundary exhaustive test matrix

## Current conclusion

The isolated native-versus-CUDA-graph cadence difference is real, bitwise, and
survives stable descriptor reuse. It is not caused by native graph rebuilding,
missing dependent-QMD prefetch, pushbuffer word count, leading barriers, final
completion placement, or sample order.

The surviving unknown is narrower: CUDA graph's private steady-state
descriptor/scheduling encoding services the same strict chain about 72 us
faster across the 208-call heterogeneous population. Switching the complete
model to the CUDA backend is not viable; that route is about 1.46 ms/token
slower. A native command-stream capture/diff is required before implementation.

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

## Exhaustive remaining list

| ID | question | smallest admissible test | status |
|---:|---|---|---|
| 18 | CUDA versus native QMD fields | capture same strict-chain QMDs and field-diff v5 | RM ownership mapped; payload dump still open |
| 19 | hidden launch descriptor outside QMD | follow graph GPFIFO/PB references and dump CB0/tables | RM ownership mapped; payload dump still open |
| 20 | USERD/doorbell publication cadence | correlate `GP_PUT`, entries, and doorbell writes | allocation census complete; active-channel watchpoint open |
| 21 | QMD memory class/locality | reproduce captured CUDA placement in native sidecar | blocked on capture |
| 22 | dependent action differs | identify CUDA action before testing native alternative | blocked on capture |
| 23 | second dependent slot/window | inspect CUDA chain; strict serial semantics prohibit speculative branch | blocked on capture |
| 24 | graph upload flag alone | explicit upload versus warmed ordinary instantiate | closed: 2615.328 versus 2611.936 us at 208, no material movement |
| 25 | graph rebuild/update | rebuild, update, and reuse arms | low priority; stable reuse result already isolates steady path |
| 26 | exact graph boundaries | one/four/five matched bridge executables | open, bounded small by prior merge losses |
| 27 | unique per-layer weights | allocate physical weight population and rerun | open; current interleaved working set exceeds L2 but aliases pointers |
| 28 | instruction/program residency | alternating and unique-program conditioning | partially closed by 12-cubin bridge |
| 29 | two-GPFIFO exact topology | replay captured production assignments/edges in both front ends | open after capture diff |
| 30 | native hardware-only timing | bracket chain with hardware timestamp signals | open; current native drain includes CPU completion observation |
| 31 | clock telemetry | retain SM/memory clocks per arm | open; order reversal already passes |
| 32 | production native candidate | gated implementation of captured transferable property | blocked on IDs 18--20 |
| 33 | production exactness | full token-stream and multiple-context gates | blocked on candidate |
| 34 | production wall | reps>=7 reverse bracket | blocked on candidate |
| 35 | dense-model generality | second dense shape/quant population | blocked on candidate |
| 36 | vocabulary body | independent service-rate construction | open, measured 15.800-us ceiling |
| 37 | O body | independent cleanup | open, measured 3.007-us ceiling |

## Command-stream capture requirement

The next tool must correlate NVIDIA RM handles rather than guess mappings by
size. It needs to resolve the usermode/doorbell VMA, USERD, GPFIFO memory, GPU
VA mappings, pushbuffer entries, `SEND_PCAS`, QMD, and referenced CB0 or
out-of-line descriptor tables. Capture only the standalone strict-chain bridge
first. Diff one-node, sixteen-node, and 208-node graph executions against the
native authorities.

No production runtime mutation is justified until that diff identifies a
specific transferable property.

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

The bound, ABI-correct, bitwise bridge measures a 72.357-us synthetic movement:

```text
4060.523 - 72.357 = 3988.166 us/token
1e6 / 3988.166     = 250.742 tok/s
```

This is an investment ceiling, not a booking. The whole CUDA route is a
negative control, not a candidate: 5.657 ms/token versus native 4.195 ms/token
in the same short smoke, with identical generated tokens.

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
