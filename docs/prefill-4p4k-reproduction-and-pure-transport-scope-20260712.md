# Prefill 4.4k reproduction and pure transport scope

## Objective

Recover and explain the historical Qwen3-8B ctx512 4.4k result, then carry the
winning role strategies into a genuinely generated, machine-searchable route.
Do not conflate the performance milestone with the provenance milestone.

## Current evidence

All timings below use Qwen3-8B Q4_K_M, M=512, `K=8`, four warmups, three
rounds, synchronized whole-prefill timing, and pinned RX 7900 XTX clocks.

| State | Commit | ctx512 | Provenance |
|---|---|---:|---|
| Historical S9 combined best | `b1259638d` | **116.01 ms / 4,413 tok/s** | hybrid hand atoms |
| Historical S9 default | `b1259638d` | 116.67 ms / 4,388 tok/s | hybrid hand atoms |
| Current S9 comparison | later tree | about 124.9 ms / 4,099 tok/s | hybrid hand atoms |
| Current gate/up-only policy | `734d2a648` | 127.33 ms / 4,021 tok/s | candidate plus fallback |
| Current all-four buffer2 | `6a44e6f88` | 147.05 ms / 3,482 tok/s | four generated candidates |

The historical combined-best tuning contributes only 0.66 ms versus the
historical default. Therefore most of the current 11.3 ms gap to 4.4k is a
code/route/runtime regression since `b1259638d`, not an unsearched S9 LDS2
axis.

## Definition of 100%

This program is complete only when all of the following are true:

1. The historical 4,413 tok/s result is reproducible or its loss is assigned to
   exact commits/configuration changes under the same benchmark regime.
2. The current fastest policy has honest per-role provenance; no hand atom is
   reported as pure generated.
3. A clean pinned ctx512/1024/2048/4096 sweep and model-output parity artifact
   exist for the accepted performance route.
4. For the pure track, each non-LDS role is graph-bound through compiler-owned
   UOps/Programs with exact candidate identity, correctness, resources, binary
   join, and whole-model timing.
5. Either pure generated reaches 4.4k or a compiler/runtime boundary is proven
   with a minimal failing contract and no remaining low-cost experiment.

## Track A: reproduce 4.4k

### A0. Freeze the benchmark regime

Create one command template and validate:

- exact GGUF path and hash;
- model/profile id;
- ctx512 token/input construction;
- `K=8`, warmups=4, rounds=3;
- synchronized timing and `--pin-clock` with successful pin provenance;
- logits/sampling mode;
- max context and KV allocation policy;
- graph capture count and clean commit;
- route family, rollback state, and candidate-role policy.

Reject comparisons with DEBUG timing, PROFILE dispatch sums, different logits
mode, dirty trees, missing clock pin, or different model weights.

### A1. Re-run the historical authority

Use a separate worktree at `b1259638d` so the current tree is untouched. Run:

1. S9 historical default.
2. S9 historical combined best.
3. At least two repetitions of ctx512.
4. One ctx512/1024/2048/4096 sweep after reproduction.

Acceptance band: ctx512 median/min protocol must reproduce the stored 116 ms
band within 2%, with route attribution remaining external handwritten/hybrid.

### A2. Re-run current controls in the same session

On the current clean tree, run sequentially under the same thermal/session
conditions:

1. current S9 default;
2. current S9 combined-best knobs;
3. gate/up-only candidate policy;
4. all-four candidate policy as a negative control.

Record raw samples, not only minima. The current S9 run determines whether the
regression is still present or was session variance.

### A3. Configuration manifest reconstruction

Extract the exact effective environment and compile/resource identities from
the historical artifacts. Include:

- wait policy, especially cooperative-store/fragment-load counters;
- WM/ WN and wave geometry;
- pipeline/DBUF/PLRAB flags;
- LDS padding/layout and allocation;
- role-selective exclusion policy;
- Q4_K/Q6_K prefill routes;
- LM-head/logits path;
- graph-GEMM and PREFILL_V2 defaults;
- any context or warmstart globals installed during capture.

Do not infer configuration solely from artifact filenames. Fail closed when an
effective setting is unavailable.

### A4. Regression range and bisection

The known good is `b1259638d`; the current tree is the known slow state only
after A2 confirms it. Build a first-parent commit list for files owning:

- `tinygrad/llm/model.py`;
- `tinygrad/llm/prefill_routes.py`;
- `extra/qk/prefill/wmma.py`;
- graph-GEMM route and schedule specs;
- warmstart/postrange state;
- TinyJit/HCQ graph execution and synchronization.

Use separate worktrees and a smoke-to-authority ladder:

1. compile/load smoke;
2. pinned ctx512 `K=2`, two rounds for classification;
3. full `K=8`, four-warmup, three-round confirmation on boundary commits.

Classify each commit as good, bad, invalid, or incomparable. An invalid commit
does not count as bad. Bisect configuration changes separately from code when
environment defaults changed.

### A5. Causal A/B of the first bad change

Once a boundary is found, toggle or minimally revert only the responsible
behavior on the current tree. Require:

- output parity;
- same route identities and model weights except the isolated variable;
- pinned ctx512 recovery;
- resources/graph count explaining the effect;
- no accidental restoration of an unrelated historical path.

Bank the causal result even if the old behavior should not be promoted.

### A6. Performance-route promotion

If the 4.4k behavior is safe, promote it behind an explicit manifest policy,
then run the full context sweep. Label provenance honestly:

- S9 `build_gemm_pipe` / `build_gemm_lds2` raw `Ops.INS` is hybrid/hand atom;
- candidate buffer2 compiler route is generated only for the roles actually
  bound to its exact identity;
- a mixed route is not globally pure merely because one candidate is pure.

## Track B: pure generated 4.4k

Track B starts after Track A identifies the exact winning strategies. It is not
blocked on promoting the hybrid performance result, but it must not borrow the
hybrid provenance label.

### B0. Freeze successful generated work

Keep `ffn_gate_up` on its proven 40 KB LDS buffer2 candidate. Do not reopen
all-four buffer2: pinned role A/B already shows it loses about 20 ms versus the
gate/up-only policy.

### B1. Non-LDS compiler transport contract

The diagnostic compiler can emit generated WMMA structure for the lean pipe
shape, but `lower_wmma_pipe_spec` is intentionally unimplemented. Define a
compiler-owned transport that accepts:

- typed `WMMAPipeSpec` and exact role/workload;
- ordinary graph buffers and launch dimensions;
- compiler-generated UOps/Program, not raw route-local instruction tuples;
- candidate context/hash and binary identity;
- cache-safe graph replay and scoped warmstart state.

The current impedance mismatch is explicit:

- `emit_prefill_gemm_from_spec` expects raw instruction tuples;
- `route_pf16_graph_gemm` feeds those to the `asm_kernel` ABI;
- diagnostic lowering returns a compiled Program/report, not graph-bound UOps;
- adapting it by copying S9 `Ops.INS` would create another hand emitter and is
  forbidden.

### B2. Minimal vertical slice: `attn_qo`

Implement one exact `512x4096x4096` graph-bound candidate before generalizing.
Required gates:

1. host extraction and type validation;
2. compiler-only source/structure proof;
3. no `Ops.INS`, native-ISA source, or hand oracle;
4. full-output nonconstant AMD correctness;
5. resource and no-spill evidence;
6. runtime binary equality;
7. pinned isolated timing;
8. gate/up-plus-attn_qo whole-model A/B.

Reject the abstraction if it cannot preserve ordinary graph ABI and candidate
identity without route-local special cases.

### B3. Extend to `ffn_down` and `attn_kv`

Only after B2 passes, reuse the same transport for exact role candidates:

- `ffn_down`: `512x4096x12288`;
- `attn_kv`: `512x1024x4096`.

Small-N KV requires its own occupancy/resource proof. Do not assume the square
attn_qo geometry transfers.

### B4. Machine search

Seed search from the reproduced S9 non-LDS strategy, expressed through compiler
knobs. Search role-specific geometry, waves, upcast, unroll, vector loads,
register residency, and wait policy only where owned by the compiler contract.

The prior plain scheduler-table search is a recorded negative result: isolated
31.95/35.70/36.53 TFLOPS winners changed whole-model throughput from 4,021 to
4,023 tok/s. Future candidates must bind the actual route identity before
isolated timing is considered transferable.

### B5. Combined pure authority

Assemble only passing exact candidates, prove all role identities in the census,
run parity, and execute pinned ctx512/1024/2048/4096. Compare against:

- current gate/up-only mixed route;
- reproduced historical S9 default and best;
- current all-four buffer2 negative control.

## Decision tree

1. Historical worktree does not reproduce 4.4k:
   investigate machine/model/driver/clock differences before code bisection.
2. Historical reproduces, current S9 does not:
   bisect code/default regression.
3. Current S9 reproduces 4.4k, gate/up-only remains near 4.0k:
   the remaining gap is role implementation/provenance; proceed with Track B.
4. Gate/up-only also reaches 4.4k in repeated current runs:
   bank variance/reproducibility evidence before further optimization.
5. Compiler transport requires raw hand instructions:
   pure Track B is blocked; keep hybrid performance and document the missing
   Program-to-graph ABI rather than relabeling S9.

## Artifacts

Every run must record command/environment, source commit, dirty state, clock pin,
model identity, route/census, raw samples, correctness, and result classification.
Store regression-bisection artifacts separately from candidate-search artifacts.

Primary existing evidence:

- `bench/prefill-whole-synced/raw-hand-s9-combined-best-authority.json`;
- `bench/prefill-whole-synced/raw-hand-s9-combined-default-authority.json`;
- `bench/prefill-lds2-s9/combined-search.json`;
- `bench/prefill-lds2-s9/final-report.json`;
- `bench/prefill-pure-full-kernel/gate-up-only-policy-20260712/`;
- `bench/prefill-pure-full-kernel/multirole-buffer2-candidate-set-v1/`.

## Stop conditions

Stop an experiment only when it is reproduced, falsified by a controlled A/B,
or blocked by a named interface/hardware prerequisite. Do not stop because a
microbenchmark fails to transfer; move to the route-bound surface. Do not keep
searching a surface after repeated whole-model results are within the measured
noise band.
