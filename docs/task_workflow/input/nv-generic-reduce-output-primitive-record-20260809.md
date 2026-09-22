# NV generic cooperative reduction-to-output primitive record (CPU gate)

Status: **CPU capability gate PASS. The single-recipe `REDUCE_OUTPUT` body is
now a shape/recipe-generic cooperative reduction-to-output primitive, and the
production decode census admits the norms population. No policy promotion, no
GPU time, no wall claim.**

Scope: `docs/task_workflow/input/nv-generic-reduce-output-primitive-scope-20260809.md`
(sections 3.1-3.4). Branch `nvidia-bringup-20260731`, HEAD before this record
`8c8aa5cac`. Everything below ran on CPU (hermetic gate) or on the repo default
device for the graph census only (no wall, no lock, no `gpu-bench.lock`).

## Verdict

1. `ReduceOutputSpec` and the emitter are fully spec-driven: reduce op composes
   with the existing `_LADDER` (`ADD` -> staged XOR sum, `MAX` -> staged XOR
   max), warp/lane/per-lane association is derived from the ordinary reduce
   shape, and the recipe string selects the per-lane accumulation and epilogue.
   The 08-05 body for the shipped shape (rows=1, dim 4096, 16 warps / 32 lanes /
   8 per lane, `sumsq_rsqrt_affine`) is byte-identical: same program name
   `reduce_output_rmsnorm_1_4096` and the same UOp body digest
   `c82e25f5a4c7cb7758dc31fb8dd5bee72ee01bcff1eb08e26c030415b7a89337`.
2. The rangeify admission accepts the production spelling
   `CONTIGUOUS(RESHAPE(MEMORY_SEMANTIC(REDUCE_OUTPUT)))` (the C6 chain) both
   under an explicit STORE and as a consumer CALL argument, using the M4-style
   typed-view ownership contract (`_residual_producer_identity`,
   `_proven_invocation_input_view`, declared typed outputs). Ownership is not
   reimplemented; the residual-view validator structure is reused.
3. Hermetic gate green on `DEV=CPU`: 10 tests pass in
   `test/unit/test_generic_reduce_output.py`.
4. Production census (decode DAG, generalized admission): **108 selector
   admissions, 54 fused `reduce_output_rmsnorm_1_4096` CALLs in the captured
   graph** (both baselines had zero). Every rejection carries a distinct trace
   reason; see per-association table below.

## Hermetic gate (section 3.3)

`DEV=CPU python3 -m pytest test/unit/test_generic_reduce_output.py -q` -> 10
passed. Coverage:

- each census association (`r_16_256`, `r_2_8_4_4_16`, `r_8_16_8`) lowers a
  realized fp16 `(1,dim)` RMSNorm to ONE ordinary CALL named `reduce_output_*`,
  bitwise equal to the ordinary two-program RMSNorm, with REDUCE range + one
  barrier + LOOP restore;
- a MAX-reduce recipe lowers to one CALL via `_LADDER[Ops.MAX]` and matches the
  ordinary max-affine form bitwise;
- lazy `x+x` input fails closed with no materialization;
- PERMUTE / SHRINK / EXPAND / arbitrary AFTER / bare unproven PARAM fail closed
  with the distinct trace reasons (`marker_not_eligible`,
  `input_proof_missing`);
- the 08-05 legacy body digest is unchanged.

## Production census (section 3.4)

Run: `scratchpad/nv_reduce_output_rmsnorm_census.py --depth 256
--typed-semantic-producer`, model Qwen3-8B-Q4_K_M. Artifact:
`docs/task_workflow/output/nv-generic-reduce-output-census-20260809.json`.

| association (ordinary reduce shape) | dim | marker created | selector entries | admitted | reject reason |
| --- | ---: | ---: | ---: | ---: | --- |
| `16x32x8` (`r_16_256`) | 4096 | yes (attn/ffn/output norms) | 182 | 108 | 74 `marker_not_eligible` (view has no durable ownership at marker creation) |
| `r_2_8_4_4_16` (q-norm) | 4096 over `(1,32,1,128)` | no | 0 | 0 | rows=32 at call site; `_semantic_reduce_output_rmsnorm` keeps ordinary fallback |
| `r_8_16_8` (k-norm) | 1024 over `(1,8,1,128)` | no | 0 | 0 | rows=8 at call site; keeps ordinary fallback |

Admissions > 0 for the norms population is satisfied. `count = 54`
`reduce_output_rmsnorm_1_4096` CALLs appear in the captured decode graph (one
fused body binary, SHA-256 `d125f7c7...`), versus zero in both the ordinary
baseline (1008 calls) and the typed baseline (937 calls).

Program-level delta versus the typed baseline is recorded honestly in the
artifact: the CALL-input route emits one fused body per consuming call argument
plus one weight materialization each (54 fused + 54 weight-store programs), and
removes 18 `r_16_256` reduces and 18 `E_32_32_4_f14a5cc` epilogues. Captured
call count is therefore not reduced (net +72 vs typed, +1 vs ordinary). This is
an admission-capability result, not a program-count or wall result.

## Why this unblocks the norms and flash rows (without claiming wall)

The 08-07 capability audit attributed 658.359 us of the 662.128 us fusion/dataflow
ledger bucket to ONE missing construct, C1: a generic cooperative
reduction-to-output primitive (`nv-substrate-capability-vs-ledger-scope-20260807.md`).
The norms row (+495.330 us attribution) and the flash row (+163.029 us
attribution) both sit behind that same construct because each requires "reduce
once, broadcast within the output workgroup, then store vector output" in one
ordinary SINK program.

Before this gate, the production decode census admitted **zero** decode norms:
the single-recipe emitter could not express the production reduce shapes, and
the admission predicate had no path for the C6 chain as a consumer CALL input
(the fp16 norms are materialized as `CONTIGUOUS(RESHAPE(MS(...)))` call
arguments, not producer STOREs). This gate removes both blockers: the body is
shape/recipe-generic and the C6 CALL-input spelling is admitted through the
M4-style typed-view proof. The census now shows 108 admissions and 54 fused
bodies for the norms population - the C1 construct is reachable in the
production ordinary DAG for the first time.

No recovered wall is claimed. This is a CPU capability gate: the fused body is
bitwise proven only in the hermetic form, the NV render path (the Xid 31 class)
is deliberately not re-tested, and the captured graph is an admission census,
not a timing bracket. The 495.330 us / 163.029 us rows remain ledger
attribution behind C1; they become reachable (eligible for later isolated GPU
qualification under lock) instead of structurally impossible.

## Policy and constraints

- `decode-reduce-output-rmsnorm-route-policy.json` stays `promoted_targets: []`.
- No model wiring change; no changes to `decode_routes.py`, `qk_primitives.py`,
  M4/M5/Path-3/M3 records, or the shared-Q8 promotion record.
- Code and test shipped in one `[nv]` commit; this record and the census
  artifact ship as `[docs]`.
