# Pure multi-role buffer2 rollout scope

## Goal

Move Qwen3-8B whole-prefill from the current generated two-buffer result toward
the historical 4.4k tok/s line by applying the same proven generated compiler
architecture to the remaining dense prefill roles. Do not add role-specific
kernel implementations or broaden one exact candidate across multiple roles.

## Current authority

Pinned whole-prefill, Qwen3-8B Q4_K_M, 512-token chunks, `K=8`, four warmups,
three rounds:

| Context | Pure scheduler | Gate/up buffer2 | Speedup |
|---:|---:|---:|---:|
| 512 | 1,511 tok/s | 2,431 tok/s | 1.61x |
| 1,024 | 1,473 tok/s | 2,384 tok/s | 1.62x |
| 2,048 | 1,410 tok/s | 2,241 tok/s | 1.59x |
| 4,096 | 1,324 tok/s | 2,019 tok/s | 1.52x |

At ctx512 the current time is 210.6 ms. The 4.4k target is 116.4 ms, so
94.2 ms must still be removed. The gate/up conversion has delivered about 58%
of the total latency reduction from the 338.9 ms pure baseline to that target.

The DEBUG=2 synchronized trace is useful only for kernel inventory. It strongly
distorts runtime and must not be used as the normal Amdahl baseline. It confirms
the remaining role shapes and call counts but not their unsynchronized shares.

## Exact role candidates

Emit four independently hashed exact candidate payloads:

| Role | M | N | K | Calls/layer |
|---|---:|---:|---:|---:|
| `ffn_gate_up` | 512 | 12,288 | 4,096 | 2 |
| `ffn_down` | 512 | 4,096 | 12,288 | 1 |
| `attn_qo` | 512 | 4,096 | 4,096 | 2 |
| `attn_kv` | 512 | 1,024 | 4,096 | 2 |

All four admit and source-compile with the existing schedule:

- tile `128x128x32`;
- waves `4x2`, 256 threads;
- two 20,480-byte LDS slots;
- two chained K16 WMMAs per K32 epoch;
- 40,960-byte active LDS;
- exact divisible M/N/K;
- zero new postrange or stage-builder generalization required.

Each role retains an exact workload, role, target, schedule, and canonical hash.
The compiler path is shared; identities, evidence, and cache entries remain
separate.

## R1: candidate-set registry

Add an immutable validated registry in `extra/qk/runtime_specs.py`:

- set schema `boltbeam.full_kernel_candidate_set.v1`;
- entries `{canonical_identity, payload}`;
- exact index `(profile, role, M, N, K, backend, arch, wave_size)`;
- admission through the existing capability validator;
- reject duplicate exact keys;
- reject identity/payload mismatch;
- reject weak warmstart-key collisions between different candidates;
- one-entry compatibility adapter for the current JSON/hash environment pair.

The set is an orchestration object. Individual candidates remain the compiler
and evidence identity.

## R2: exact route selection

In `extra/qk/prefill_graph_gemm_route.py`:

1. Resolve role and exact `(M,N,K)` before current pipe/LDS policy.
2. Query the admitted candidate registry.
3. If one exact entry exists, install its typed context and schedule opts and
   route through the existing ordinary generated `A @ B.T` transport.
4. Candidate selection may override the role's default `pipe` family with the
   admitted generated LDS/buffer2 schedule.
5. If no entry exists, preserve current behavior byte-for-byte.
6. A set must not globally force or reject roles it does not contain.

No new emitter, HIP kernel, `Ops.INS`, or role-specific `_apply_tc_opt` branch is
permitted.

## R3: scoped compiler bindings

The current warmstart key `(frozenset({M,N}),K)` is sufficient for these four
distinct shapes, but it is weak and loses role/orientation. Add fail-closed
collision detection before installing contexts.

Candidate state must share the same lifetime as model warmstart state:

- `_WARMSTART_OPTS`;
- `_WARMSTART_CANDIDATE_CONTEXTS`;
- `_WARMSTART_LOCAL_STAGE_KEYS`;
- local-stage deny keys.

Install and restore them together around prefill capture. Running another model,
profile, or candidate set in the same process must not inherit stale contexts.

Compiler cache identity already includes candidate schema/hash. Multiple layers
and paired tensors may reuse one role binary while binding different buffers;
different role hashes must never alias.

## R4: per-role proof ladder

For each role, in this order:

1. Candidate-set admission and exact route binding.
2. Source-only compile and emitted lifecycle proof.
3. Final ISA/resources: 40,960 LDS, correct workgroup, zero spills/scratch.
4. Full-output numerical comparison using nonconstant role-specific inputs.
5. Runtime binary equality.
6. Kernel-only pinned timing against that role's ordinary pure scheduler.
7. Five-stage evaluator join on candidate, binary, and clean commit.

Recommended rollout order:

1. `attn_qo`: square shape and two projections/layer.
2. `ffn_down`: largest K and likely largest remaining dense-linear cost.
3. `attn_kv`: smaller N; occupancy/tail behavior requires separate evidence.

Measured whole-prefill contribution, not this heuristic, decides subsequent
search priority.

## R5: combined model gate

Assemble only passing role winners into one candidate-set manifest and run:

- route census proving all four exact identities;
- full-logit or token parity;
- pinned ctx512/1024/2048/4096 whole-prefill authority;
- per-role binary/resource attribution;
- no oracle or hybrid fallback;
- weight-budget and VRAM evidence;
- comparison against pure baseline, gate/up-only, and historical S9.

Candidate-set acceptance is atomic, but a failing role may be omitted while the
passing subset remains benchmarkable.

## R6: BoltBeam search

BoltBeam owns set construction and evaluation orchestration:

1. Derive role shapes from the model profile.
2. Create role-specific populations seeded by the proven schedule.
3. Admit/compile/prove/resource-check each candidate in isolated subprocesses.
4. Run correctness before timing.
5. Persist per-role winners and assemble a set manifest.
6. Run combined whole-prefill and retain the best passing subset.

Tinygrad remains the sole legality/compiler/runtime authority. BoltBeam must not
duplicate tile, LDS, lifecycle, or descriptor rules.

## Weight and model limits

The current generated candidates consume dense fp16 weights after GGUF
dequantization.

- Qwen3-8B: use the existing resident fp16 overlay and admission budget. Reject
  before partial allocation if weights, KV, and scratch do not fit.
- Qwen3-14B: four-role resident fp16 overlay generally cannot fit on a 24GB
  7900 XTX. The experimental chunked overlay has stale-replay/MMU history and is
  not an admissible production path. 14B requires safe weight rebinding or
  packed in-kernel Q4/Q6 decode before whole-model promotion.

## Required tests

1. Four-entry 8B set admission; wrong role/shape/profile rejected.
2. Duplicate exact key, identity mismatch, and weak-key collision rejection.
3. Exact role selection overrides default pipe policy only for admitted entries.
4. Missing role falls back to current route without error.
5. One graph produces four distinct contexts/cache identities.
6. Paired tensors/layers reuse the role binary with different buffers.
7. Candidate state restores across sequential model/profile runs.
8. Per-role numerical, source, ISA, resource, runtime, and timing gates.
9. Combined full-model parity and pinned context sweep.
10. Weight-budget rejection before partial fp16 overlay realization.

## Completion

The rollout is complete when all admitted role candidates have five-stage
authority, the combined route is strict pure with no fallback, and pinned
whole-prefill either reaches the 4.4k line or a new measured residual identifies
the next non-GEMM/LM-head ceiling. DEBUG-synchronized kernel shares are not
sufficient for completion.
