# Dense active-boundary recovery campaign

## Objective

Recover as much of tinygrad's non-active token interval as possible. Reaching
llama parity is a checkpoint, not the stopping condition.

The current common-protocol ledger estimates:

| quantity | tinygrad | llama | tinygrad minus llama |
|---|---:|---:|---:|
| kernel-active work | 3701.818 us | 3878.210 us | -176.392 us |
| endpoint wall | 4060.523 us | 4021.721 us | +38.802 us |
| wall minus active | 358.705 us | 143.511 us | +215.194 us |

The 215.194-us difference is diagnostic territory, not a booked recovery
pool. The campaign must convert it into mechanisms before translating it to a
token-rate ceiling.

## Ranked targets

1. **Production-schedule elongation of small QMDs.** Measure submitted-to-
   active and active-to-successor intervals for each physical node. The first
   emphasis is native norms, Flash score/combine, and the small K/V completion
   population because their old command intervals exceed their exact active
   bodies most strongly.
2. **Native HCQ versus CUDA-graph replay of the same cubin sequence.** Keep
   binaries, buffers, ordering, and dependencies fixed. This isolates runtime
   admission/cadence from kernel code and cache state.
3. **Launch-count removal after the tax is measured.** Rank legal producer-
   consumer fusions by `removed QMDs x measured exposed tax`. Do not infer a
   win merely from fewer launches; previous in-GEMV RMSNorm constructions
   increased body work and lost.
4. **Vocabulary service rate.** This remains a confirmed independent
   15.800-us/token active-body debt.
5. **Remaining exact-body conversion.** Convert combine, Q/K completion,
   argmax, and tail rows to the common CUPTI boundary before assigning the
   final residual.
6. **O cleanup.** The confirmed active-body debt is 3.007 us/token.

## Initial discriminators

### Fixed per-QMD tax

Fresh current-HEAD chained replay reproduces the native HCQ no-op floor:

| arm | drain slope |
|---|---:|
| plain no-op QMD | 0.6512 us/kernel |
| timestamp-bracketed no-op QMD | 0.6507 us/kernel |
| plain Q/K cubin | 1.5258 us/kernel |
| timestamp-bracketed Q/K cubin | 1.4973 us/kernel |

Timestamp instrumentation is not the tax. A real native QMD has an
approximately 0.65-us clean-chain floor.

However, applying that constant to the exact projection census is rejected.
The 208 projection/provider calls show only 65.984 us/token of old-command
minus exact-active inflation, or 0.317 us/call. A universal 0.65-us model
predicts 135.200 us. Per-family differences range from -0.270 to +1.158
us/call. The residual is therefore schedule- and predecessor-dependent, not a
uniform charge that can be recovered from every launch.

### Graph grouping

Replay-group consolidation is not promoted as the first investment. The prior
`JIT_BATCH_SIZE=1024` construction was 112.9 us/token slower, and the later
one-graph PDL bracket was dominated by 329.25-us control spread. This does not
prove every continuous-chain implementation is bad, but it rejects the claim
that deleting four group boundaries alone exposes a clean 215-us recovery.

## Decision after the first tests

The next implementation test is a matched native-HCQ/CUDA-graph sequence
bridge, beginning with the small-node chain rather than the full 596-node
token. It should include native RMSNorm, shared provider, Q/K completion,
Flash combine, and representative projection bodies. If the same sequence is
materially faster through CUDA graph replay, invest in native admission and
chain cadence. If it is tied, the remaining tax is production predecessor and
cache topology; instrument those exact transitions rather than rewriting the
runtime globally.

## Matched front-end bridge result

The proposed bridge is now implemented and passes two levels of confirmation.

First, the exact production Q/K completion cubin was replayed as the same
serial chain through native HCQ and CUDA graph:

| front end | drain slope |
|---|---:|
| native HCQ | 1.5258 us/node |
| CUDA graph, run 1 | 1.2947 us/node |
| CUDA graph, run 2 | 1.2941 us/node |

CUDA graph recovers 0.2311--0.2316 us per small node. Ordinary CUDA launches
cost 3.41--3.45 us/node, so the result is specifically graph replay rather
than a generic CUDA-driver advantage.

Second, the bridge replayed all twelve exact installed projection/provider
cubins with their captured ABIs, real family counts, and a deterministic
round-robin serial order. CUDA graph beat native HCQ at every measured prefix
in both runs. At the complete 208-call population:

| confirmation | native HCQ | CUDA graph | recovery |
|---|---:|---:|---:|
| ABI-correct run 1 | 2674.572 us | 2606.112 us | **68.460 us** |
| ABI-correct run 2 | 2681.716 us | 2603.360 us | **78.356 us** |

This rejects the single-hot-cubin artifact theory. A native front-end/cadence
opportunity survives heterogeneous binaries and large projection bodies.

The result is not yet a production booking. The bridge is serial, reuses hot
buffers, and does not reproduce the installed two-queue dependency topology.
Those restrictions are symmetric between arms and make the causal front-end
comparison valid, but the 69.609--82.657-us movement must survive a production
graph bracket before it changes the endpoint. An audit found that the original
four heterogeneous traces omitted scalar argument `700` for three Q8 kernel
families (36 calls). Those traces are invalid and retained only as provenance.
The table above comes from two new ABI-asserted runs with captured pointer and
scalar arguments; the conclusion survives, but the original numeric range is
superseded.

The subsequent semantic oracle initializes every captured logical buffer with
a deterministic nonzero pattern and compares final cross-runtime SHA-256
values after identical replay populations. All **46/46 logical buffer hashes
match bitwise**. The timing construction therefore passes both the ABI and
execution-semantic gates.

Applied mechanically to the installed endpoint, the observed projection-only
range is 3982.167--3992.063 us/token, or 250.50--251.12 tok/s. This is now the
highest measured recovery construction, not merely the old 215-us residual.

## Investment decision

Proceed with the substrate. The next implementation must compare the native
and CUDA graph command encodings for this exact population and identify which
transferable property lowers cadence: QMD reuse, GPFIFO/pushbuffer density,
queue admission fields, or replay submission structure. The first production
candidate should port that property into a new native replay path behind an
environment gate; it should not replace the installed path until exactness and
a reverse wall bracket pass.

## Token-rate interpretation

Recovering the full measured differential would give an illustrative, not
booked, endpoint of about 3845.329 us/token or 260.1 tok/s. Vocabulary and O
alone can only approach roughly 247.4 tok/s. The runtime-boundary campaign is
therefore the only currently open route to a result materially beyond parity.

## Evidence

- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/hcq-dispatch-current.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/cuda-graph-bridge-r1.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/cuda-graph-bridge-r2.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/projection-bridge-nv-r1.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/projection-bridge-nv-r2.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/projection-bridge-cuda-r1.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/projection-bridge-cuda-r2.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/projection-bridge-nv-abi-r1.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/projection-bridge-nv-abi-r2.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/projection-bridge-cuda-abi-r1.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/projection-bridge-cuda-abi-r2.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/frontend-bridge-summary.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/projection-bridge-nv-semantic-v2.json`
- `docs/task_workflow/evidence/nv-active-boundary-targets-20260827/projection-bridge-cuda-semantic-v2.json`
- `docs/task_workflow/evidence/nv-active-body-ledger-20260827/phase2-projection-result.json`
- `docs/task_workflow/output/nv-hcq-dispatch-slope-result-20260822.md`
- `docs/task_workflow/output/nv-active-body-ledger-phase2-result.md`
- `docs/task_workflow/output/nv-w1w3-norm-smem-result-20260822.md`
- `docs/task_workflow/output/nv-edge-aware-pdl-runtime-hook-result-20260821.md`
