# Scope: make the 14B packed-WMMA route a legitimate machine-search claim (2026-07-25)

14B prefill is **1945/1920/1875/1785 tok/s** at pp512/1024/2048/4096, beating llama.cpp same-session by
+5.4/+5.7/+6.7/+8.7%. That number is real and reproducible. **It does not currently count as a machine-search
result** under this repo's own contract, and `assert_pure_machine_search` returns `GUARD PASS` only because
the route is invisible to it.

This scope fixes the governance, not the kernel. No performance change is expected or sought.

## What was fact-checked first (so nobody re-litigates it)

**The kernel IS generated.** `extra/qk/prefill/current_prefill_execution_adapter.py:76-87` builds an ordinary
`(a @ b.transpose()).schedule_linear()` and compiles it via `compile_linear` under
`Opt(OptOps.TC, 0, (-1,2,1))` plus a candidate context. So packed-WMMA is tinygrad's own
scheduler-generated matmul shaped by a frozen opt/schedule payload -- NOT a hand-written kernel. An earlier
draft of this analysis called it "a hand-written kernel wearing generated clothes"; that was wrong and is
retracted here.

**The opt values have no demonstrable search provenance.** `PACKED_WMMA_GEOM`
(`extra/qk/prefill/packed_wmma_prefill_candidates.py:38`) cites
`e2e_packed_wmma_bench_q6_nopad.py:56-75` as its source. That file has **never existed in this repo** --
absent from the working tree, from every commit, and from the object database (`git rev-list --all
--objects`). The only in-repo geometry sweep, `bench/wave32-geometry-compile-sweep/latest.json`, is
`"scope": "compile-only"`, has 12 results with `result[0] status: FAIL`, and sweeps
`tiles [16,32,64,128] x waves [[1,1],[2,2],[4,2]]` at fixed `k=256` -- a grid that **cannot** have produced
`tm=256, tn=128, tk=32, wn=2`. So we cannot show the constants are search outputs rather than guesses,
independently of whether they are.

## The six blockers, and which are hard

| # | blocker | kind |
|---|---|---|
| 1 | no `ROUTES` row -> no provenance, no `shape_guards`, nothing for `derive_purity_status` to evaluate | **HARD** |
| 2 | not in `pure_search_guard.HOT_FAMILIES` -> the census structurally cannot see it | **HARD** |
| 3 | opt constants have no reproducible in-repo search provenance | evidence |
| 4 | `PACKED_WMMA_GEOM` keyed `(quant, role)` with no shape term -- 8B and 14B share one entry | design |
| 5 | `rebind_full_kernel_workload` (`runtime_specs.py:307`) overwrites `applicability`, then admission (`:392-395`) validates the claim rebind just wrote -- circular, so legality cannot be enforced | correctness |
| 6 | no authority gate; 14B end-to-end token parity never run with this route live | evidence |

## PHASE 1 -- register the route honestly (this scope; closes 1, 2, 6)

**Claim `provenance: "tinygrad_scheduler_generated"`, NOT `machine_authored_generated`.** This is the
defensible label today: the kernel genuinely comes from tinygrad's scheduler, and
`tinygrad_scheduler_generated` is in `FINAL_DEFAULT_PROVENANCE`, so `derive_purity_status` yields
`search_generated_promoted` for a `promoted_default` row. Claiming `machine_authored_generated` would assert
that an emitter derives extents from descriptor fields, which the frozen shape-blind table does not do.
**Do not overclaim here -- the whole point of this exercise is that the label must be earned.**

1. **`ROUTES` row** `packed_wmma_prefill_generated` in `extra/qk/route_manifest.py`, modelled on the
   `prefill_flash_attention_generated` row: `workload=prefill`, `status=promoted_default`,
   `provenance=tinygrad_scheduler_generated`, roles `[attn_qo, attn_kv, ffn_gate_up, ffn_down]`,
   `quant=[Q4_K, Q6_K]`, `env={"TINYGRAD_PREFILL_PACKED_WMMA": "1 (default)"}`,
   `rollback={"TINYGRAD_PREFILL_PACKED_WMMA": "0 -> direct-packed"}`,
   `baseline_route_id="prefill_q4k_direct_tile4x4_default"`, `expected_kernels`, `forbidden_kernels`,
   `authority_gate` pointing at the new gate.
   **`shape_guards` MUST carry the shape** -- the four exact 14B `(m,n,k)` triples
   `(512,5120,5120)`, `(512,1024,5120)`, `(512,17408,5120)`, `(512,5120,17408)` -- and the 8B triples where
   a geometry entry exists. This is where blocker 4 gets partially contained: the manifest records the
   shapes the route is admitted for even while the geometry table stays shape-blind internally.
2. **`HOT_FAMILIES` entry** `prefill_packed_wmma` in `extra/qk/pure_search_guard.py`, with
   `rollback_active` reading `TINYGRAD_PREFILL_PACKED_WMMA` via `_env_flag(e, "TINYGRAD_PREFILL_PACKED_WMMA", 1)`
   (note the runtime default is 1, so an UNSET key must resolve to 1 -- `_env_flag` exists for exactly this,
   and `test/unit/test_pure_search_guard_boundary.py` pins these defaults against the real getenv defaults).
3. **Authority gate** `extra/qk/prefill/packed_wmma_prefill_promotion_gate.py`, modelled on
   `extra/qk/prefill/prefill_causal_tile_skip_promotion_gate.py` and
   `extra/qk/prefill/prefill_softmax_reduce_fuse_promotion_gate.py`: reads transcribed evidence only, never invents
   numbers, and **fails closed** on any manifest-admitted shape lacking evidence.
4. **Run the missing evidence:** 14B end-to-end token parity with the route live
   (`extra/qk/prefill/prefill_flash_e2e_parity.py --only 14B`) and the 6/6 canary gate results with `max_abs`. The
   throughput evidence already exists (paired same-session, two reps).

## PHASE 2 -- earn `machine_authored_generated` (NOT this scope)

Requires (a) running the opt search in-repo so the constants are reproducible artifacts rather than a
citation to a file that never existed, and (b) keying the geometry by shape so it generalizes. That also
closes the shape cliff, since a searched geometry extends to new shapes and a frozen one cannot -- the same
work as making 32B reachable. Blocker 5 (the circular `applicability` check) should be fixed alongside,
since without it a registered row still cannot be *enforced*.

## Gates for Phase 1
- `assert_pure_machine_search({'PURE_MACHINE_SEARCH_ONLY':'1'})` must PASS **with the new family present**,
  and must report `prefill_packed_wmma -> packed_wmma_prefill_generated (pure)`.
- `validate_manifest()` must return no errors (it enforces `purity_status == derive_purity_status(...)`).
- Whole unit suite failure-set **equality** -- the repo has 51 pre-existing failures; the gate is set-equality,
  not zero. Do not fix a pre-existing failure.
- No throughput change expected. Spot-check 14B pp512 stays ~1945 to prove the registration is inert.
- GPU is a single resource: wrap any GPU run in `flock /tmp/gpu-bench.lock`.

## Why bother, stated plainly
The 14B win is currently a number without a provenance story, on a route the purity system cannot see. The
repo's thesis is that machine search beats hand-written kernels; llama.cpp IS the hand-written-kernel
baseline. Beating it with unregistered frozen constants doesn't advance that thesis -- it just means our
tuning is better than theirs on two shapes. Registering the route honestly (Phase 1) makes the claim
auditable; earning `machine_authored_generated` (Phase 2) makes it true.
