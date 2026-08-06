# NV norms fusion exact-output A/B record

Date: 2026-08-06
Target: Qwen3-8B-Q4_K_M, d512, RTX 5090 / native `DEV=NV` (driver 595.84),
branch `nvidia-bringup-20260731`
Status: **NO-GO at the boundary-free construction gate; no GPU arm ran**

## Question

Can the ffn/next RMSNorm *epilogue* be absorbed as an ordinary in-core
per-load affine epilogue of the consuming quant kernel under the
boundary-free ordinary-UOp gate, and does the exact-output A/B then book the
norms attribution row? The scope authority
(`nv-fusion-exhaustive-scope-20260805.md`) admits exactly one construction for
epilogue populations: ordinary UOps in-core, no CUSTOM boundary, no adapters,
no `CONTIGUOUS` materialization, no lazy-view stripping. Norm reduce nodes
stay separate; the recoverable ceiling is the epilogue half (215 anchor-child
epilogues / 306.368 us on the authority census).

## Construction and boundary-free gate

Codegen path under test: `y = (x.float() * s).cast(float16) * w` with `s` the
bitwise-shared fp32 reduce scalar `rsqrt(mean(x^2) + eps)`, fused into the
consuming q4k GEMV body as ordinary UOps. The norm reduce nodes stay.

Gate evidence (`tinygrad.nv_norms_fusion_ab.gate.v1`, run CPU-only through the
harness, artifact `/tmp/nv-norms-fusion-gate-20260806.json`):

```text
verdict: CONSTRUCTION_GAP
conditions:
  phase0_verdict_pass                   False
  consumer_is_ordinary                  False
  reduce_absorbable                     False
  candidate_removes_programs_in_graph   False
```

The phase-0 baseline (`tinygrad.nv_boundary_free_ordinary_uop_gate.v1`)
reports the ordinary RMSNorm pair as two programs for both a realized and a
lazy input, with no CUSTOM and no CONTIGUOUS in either row:

```text
realized  program_count 2  contains_custom_kernel False  contains_contiguous False
lazy_add  program_count 2  contains_custom_kernel False  contains_contiguous False
```

The CPU candidate probe is numerically exact but topologically blocked:

```text
fused_epilogue_bitwise_equal          True   (fp16 uint16 view identical)
fused_epilogue_max_abs                0.0
fused_epilogue_contains_custom        False
fused_epilogue_contains_contiguous    False
fused_pair_program_count              2      (reduce + epilogue both stay)
affine_epilogue_program_count         1      (epilogue alone is one ordinary program)
consumer_is_ordinary                  False  (opaque q4k_g3_lanemap_gemv_* custom program)
reduce_absorbable                     False  (no cross-thread reduction-to-output primitive)
```

The construction is bitwise-exact on a d512 row and the epilogue half lowers
as one ordinary elementwise program, but the consuming decode kernel is an
opaque custom program: ordinary UOps cannot fuse an epilogue into its body
without a CUSTOM boundary, and no generic cross-thread reduction-to-output
scheduler primitive exists to absorb the reduce half. No replayable ordinary
program is removed from the decode graph. The boundary-free gate fails, so
the population's GPU arm is HARD-STOPPED before any logits or wall work.

## Logits gate

**NOT AUTHORIZED** (HARD STOP): the exact-output contract is unchanged and
was not weakened. Full fp32 logits SHA-256 over 32 decoded rows identical to
control, identical token stream, per-row argmax equal to the sampled token,
and no stale-return binding. The control-arm full-logit reference from the
P2b redirect-on authority (8 rows, `/tmp/nv_p2b_redirect_on_logits.json`) is
`71c0a2b092cbc2e40c22b42cd4f6f3c84fe56fd40f2bfd008efc5b76be0ae0f0`; no
candidate logits row exists because the construction gate stopped first.

## Census comparison

Static norms census on the redirect-on authority DAG
(`/tmp/nv_p4_redirect_on_dag_20260805.json`, 875 nodes / 4080 edges, ledger
`tinygrad.nv_fusion_population_ledger.v1`, re-run this turn):

| population | nodes | total us | candidates | candidate us | anchor-child epilogues | us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| norms | 362 | 650.752 | 288 | 544.128 | 215 | 306.368 |

Role breakdown: `rmsnorm_reduce` 73 (`r_16_256`), `rmsnorm_epilogue` 72
(`E_32_32_4_f14a5cc0`), `final_rmsnorm_epilogue` 1 (`c6fef356`), q_norm
reduce 36 / epilogue 72, k_norm reduce 36 / epilogue 72. Norms is not
boundary-free eligible (`145` reductions present). The candidate arm would be
required to change only the norms population census, keeping every reduce role
identical and removing only epilogue nodes. No census arm ran; the numbers
above are the static ledger authority for the expected change-set.

## Wall bracket

**NOT AUTHORIZED** (HARD STOP). The serialized reverse
control/candidate/control bracket with settled-continuous windows (`--mode
timing-child --settled-continuous --reps 5`, count 32, depth 512,
max-context 1024, groups 0) requires the boundary-free gate, the exact-logits
gate, and norms-confined census to pass first. No wall samples or median exist
for this row; the promotion gate remains `>= +50 us/token` vs both bracketing
controls with an identical token stream (SHA-256 authority).

## Verdict

**NO-GO** at the boundary-free construction gate. Zero recoverable credit is
booked for the norms fusion population, consistent with the audit's rule that
nothing in the 662.128 us fusion/dataflow attribution is bookable without an
exact-output native A/B per population. The construction, gate result,
logits gate, census comparison, and wall bracket are recorded in
`/tmp/nv-norms-fusion-ab-20260806.json`
(`tinygrad.nv_norms_fusion_ab.v1`, `verdict: NO-GO`).

## HARD STOP notes

1. Boundary-free construction gate returned `CONSTRUCTION_GAP`; no norms GPU
   arm is authorized until `nv_boundary_free_ordinary_uop_gate.py` passes.
2. The exact-output full-logit gate was not run because the construction gate
   failed first; the correctness contract is unchanged.
3. No norms census arm and no wall bracket ran; the `+50 us` promotion gate is
   not evaluated.
4. Closed constructions are not reopened by relabeling:
   `nv_rmsnorm_native_microgate.py`, `nv_rmsnorm_scale_gateup_microgate.py`,
   `nv_reduce_output_rmsnorm_microgate.py`,
   `nv_rmsnorm_scale_gateup_one_layer_qualification.py`; the audit closed the
   REDUCE_OUTPUT wrapper and the typed-CALL producer (0 reducers / 875
   programs). Reopening requires a first-class consumer-owned multi-phase
   scheduler carrier, not a view or adapter widening.

## Validation

```text
python3 -m pytest test/unit/test_nv_norms_fusion_ab.py -q
8 passed in 0.87s
```

Hermetic CPU tests cover the construction predicates (boundary-free gate
verdict, phase-0 rows, fused-epilogue bitwise exactness, ordinary lowering,
affine program count, candidate arm raising `ConstructionGapError`) and the
contract validators (exact-logits gate, census confinement, reverse-bracket
promotion, NO-GO record shape). No GPU is touched by the tests.

## Citations

- `docs/task_workflow/input/nv-fusion-exhaustive-scope-20260805.md` (norms
  population scope, gate, HARD STOP)
- `docs/task_workflow/input/nv-decode-exhaustive-forward-scope-20260805.md`
- `docs/task_workflow/input/nv-decode-final-accounting-audit-20260805.md`
- `extra/llm_research/decode/nv_fusion_population_ledger.py` (census)
- `extra/llm_research/decode/nv_boundary_free_ordinary_uop_gate.py` (gate)
- `extra/llm_research/decode/nv_projection_epilogue_qualification.py` (A/B
  arm pattern)
- `extra/llm_research/decode/nv_shared_q8_progressive_qualification.py`
  (flock wrapper, settled-continuous windows, token-stream SHA-256 authority)
- `extra/llm_research/decode/nv_predispatch_full_logits_qualification.py`
  (`DEFAULT_MODEL`, `_load`, `_prompt`)
- Closed norms constructions: `nv_rmsnorm_native_microgate.py`,
  `nv_rmsnorm_scale_gateup_microgate.py`,
  `nv_reduce_output_rmsnorm_microgate.py`,
  `nv_rmsnorm_scale_gateup_one_layer_qualification.py`
- `nv-reduce-output-rmsnorm-microgate-record-20260805.md`,
  `nv-reduce-output-typed-call-input-reopen-record-20260805.md` (audit
  closures)
