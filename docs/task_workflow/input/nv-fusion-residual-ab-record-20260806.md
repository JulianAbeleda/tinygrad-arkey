# NV residual/cast/contiguous fusion exact-output A/B record

Date: 2026-08-06
Target: Qwen3-8B-Q4_K_M, d512, RTX 5090 / native `DEV=NV` (driver 595.84),
branch `nvidia-bringup-20260731`
Status: **NO-GO at the boundary-free construction gate; no GPU arm ran**

## Question

Can the residual add / cast *epilogue* be absorbed as an ordinary in-core
per-load projection epilogue of the consuming quant kernel under the
boundary-free ordinary-UOp gate, and does the exact-output A/B then book the
residual/cast/contiguous attribution row? The scope authority
(`nv-fusion-exhaustive-scope-20260805.md`) admits exactly one construction for
this population: ordinary UOps in-core, no CUSTOM boundary, no adapters, no
`CONTIGUOUS` materialization, no lazy-view stripping. This row is the only
fully boundary-free-*eligible* census population (145 ordinary epilogues, zero
reductions, zero custom kernels), so it is the cleanest epilogue-row A/B in
the partition; the recoverable ceiling on the authority census is 145
anchor-child epilogues / 174.912 us.

## Construction and boundary-free gate

Codegen path under test: `y = proj_out + residual` with `proj_out` the q4k
GEMV fp32 output and `residual` the block activation, fused into the consuming
q4k GEMV body as ordinary UOps. The admissible rule is the audit's one
construction: no custom kernel boundary, no adapters, no `CONTIGUOUS`
materialization, no lazy-view stripping, exactly as scoped for epilogue
populations.

Gate evidence (`tinygrad.nv_residual_fusion_ab.gate.v1`, run CPU-only through
the harness, artifact `/tmp/nv-residual-fusion-gate-20260806.json`, SHA-256
`a30caa6c63cae04eb52131ad0be81bbdf45337a58b047a95c7360eb58b33b9bb`):

```text
verdict: CONSTRUCTION_GAP
conditions:
  phase0_verdict_pass                   False
  consumer_is_ordinary                  False
  residual_absorbable                   False
  candidate_removes_programs_in_graph   False
```

The phase-0 baseline (`tinygrad.nv_boundary_free_ordinary_uop_gate.v1`)
reports its ordinary epilogue reference as two programs for both a realized
and a lazy input, with no CUSTOM and no CONTIGUOUS in either row:

```text
realized  program_count 2  contains_custom_kernel False  contains_contiguous False
lazy_add  program_count 2  contains_custom_kernel False  contains_contiguous False
```

The CPU candidate probe separates the two facts that determine the verdict.
The residual add itself is an ordinary elementwise program when both operands
are realized fp32 buffers; its real producer is the opaque q4k GEMV custom
program:

```text
residual_add_program_count               1      (proj_out + residual, realized fp32)
residual_add_lazy_program_count          1      (lazy-input variant)
residual_add_contains_custom             False
residual_add_contains_contiguous         False
gemv_plus_residual_program_count         2      (opaque GEMV CALL + ordinary add)
gemv_plus_residual_programs              ['q4k_g3_lanemap_gemv_4096_4096', 'test']
gemv_plus_residual_contains_call         True   (CALL/SINK boundary into the opaque program)
consumer_is_ordinary                     False  (opaque q4k_g3_lanemap_gemv_* custom program)
residual_absorbable                      False  (in-core absorption needs a custom boundary)
candidate_removes_programs_in_graph      False  (the add stays behind the opaque GEMV CALL)
```

The construction is one ordinary program in isolation, but the decode
projection kernel is an opaque custom program: ordinary UOps cannot fuse an
epilogue into its body without a CUSTOM boundary, and no replayable ordinary
program is removed from the decode graph. That boundary is exactly the closed
M4 `Q4KGEMVEpilogue` route (boundary copies, measured wall regression) and the
closed P2b composed attention-O route (adapter copies), neither of which is
reopened here. The boundary-free gate fails, so the population's GPU arm is
HARD-STOPPED before any logits or wall work. `REDUCE_OUTPUT` and the typed-CALL
producer were not used and stay closed.

## Logits gate

**NOT AUTHORIZED** (HARD STOP): the exact-output contract is unchanged and
was not weakened. Full fp32 logits SHA-256 over 32 decoded rows identical to
control, identical token stream, per-row argmax equal to the sampled token,
and no stale-return binding. The control-arm full-logit reference from the
P2b redirect-on authority (8 rows, `/tmp/nv_p2b_redirect_on_logits.json`) is
`71c0a2b092cbc2e40c22b42cd4f6f3c84fe56fd40f2bfd008efc5b76be0ae0f0`; no
candidate logits row exists because the construction gate stopped first.

## Census comparison

Static residual census on the redirect-on authority DAG
(`/tmp/nv_p4_redirect_on_dag_20260805.json`, 875 nodes / 4080 edges, groups 5,
ledger `tinygrad.nv_fusion_population_ledger.v1`, re-run this turn):

| population | nodes | total us | candidates | candidate us | anchor-child epilogues | us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| residual/cast/contiguous | 145 | 174.912 | 145 | 174.912 | 145 | 174.912 |

Role breakdown: `attention_cast` 36 (`E_32_32_4_0a5eb0ac`),
`attention_residual_add_or_ffn_down_cast` 36 (`E_32_32_4_02a9738c`),
`ffn_activation_cast` 36 (`E_128_32_3`), `ffn_residual_add` 36
(`E_32_32_4_81c96a8e`, heuristic), `block_output_contiguous` 1
(`E_32_32_4_86a23e1a`, heuristic). Exact 108 / heuristic 37, mean 1.206 us,
max 3.296 us, reductions 0, epilogues 145, custom kernels 0,
`boundary_free_eligible True`; capture name digest
`49838b8ab2e7118d0c384fb93d2b4c3085b3732f1fe8d5abc69d51d232a6b413`, total
5260.256 us. This is the only boundary-free-eligible population. The candidate
arm would be required to change only the residual population census, removing
145 epilogue nodes with zero reduction-role changes. No census arm ran; the
numbers above are the static ledger authority for the expected change-set.

## Wall bracket

**NOT AUTHORIZED** (HARD STOP). The serialized reverse
control/candidate/control bracket with settled-continuous windows (`--mode
timing-child --settled-continuous --reps 5`, count 32, depth 512,
max-context 1024, groups 0) requires the boundary-free gate, the exact-logits
gate, and residual-confined census to pass first. No wall samples or median
exist for this row; the promotion gate remains `>= +50 us/token` vs both
bracketing controls with an identical token stream (SHA-256 authority).

## Verdict

**NO-GO** at the boundary-free construction gate. Zero recoverable credit is
booked for the residual/cast/contiguous population: the census row is
boundary-free eligible, but the admissible ordinary-UOp in-core construction
is not expressible because the consuming q4k GEMV is an opaque custom program,
and the closed custom-boundary routes are not reopened. The construction, gate
result, logits gate, census comparison, and wall bracket are recorded in
`/tmp/nv-residual-fusion-ab-20260806.json`
(`tinygrad.nv_residual_fusion_ab.v1`, `verdict: NO-GO`, SHA-256
`f99d503533b23138dc68257dabb66b380e89062113c3f3aab598a6daf306b591`).

## HARD STOP notes

1. Boundary-free construction gate returned `CONSTRUCTION_GAP`; no residual GPU
   arm is authorized until `nv_boundary_free_ordinary_uop_gate.py` passes.
2. The exact-output full-logit gate was not run because the construction gate
   failed first; the correctness contract is unchanged.
3. No residual census arm and no wall bracket ran; the `+50 us` promotion gate
   is not evaluated.
4. Closed constructions are not reopened by relabeling: the M4 q4k GEMV
   epilogue route (`m4-q4k-epilogue-measurement-record-20260802.md`,
   `nv-decode-native-epilogue-causal-record-20260805.md`) and the P2b composed
   attention-O route (`nv-p2b-projection-boundary-reopen-record-20260805.md`);
   the REDUCE_OUTPUT wrapper and the typed-CALL producer stay closed
   (`nv-reduce-output-callify-redirect-audit-20260805.md`,
   `nv-reduce-output-typed-call-input-reopen-record-20260805.md`). Reopening
   requires a first-class consumer-owned in-core projection epilogue accepting
   the native fp32 GEMV result and residual directly, not a boundary copy or
   adapter widening.

## Validation

```text
python3 -m pytest test/unit/test_nv_residual_fusion_ab.py -q
8 passed in 1.37s
```

Hermetic CPU tests cover the construction predicates (boundary-free gate
verdict, phase-0 rows, residual-add ordinary lowering for realized and lazy
inputs, the two-program opaque GEMV boundary with CALL, candidate arm raising
`ConstructionGapError`) and the contract validators (exact-logits gate, census
confinement, reverse-bracket promotion, NO-GO record shape, ledger fallback
with the 145/174.912 residual reference). No GPU is touched by the tests.

## Citations

- `docs/task_workflow/input/nv-fusion-exhaustive-scope-20260805.md`
  (residual/cast/contiguous population scope, gate, HARD STOP)
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
- `docs/task_workflow/input/nv-decode-native-epilogue-causal-record-20260805.md`
- `docs/task_workflow/input/m4-q4k-epilogue-measurement-record-20260802.md`
  (closed q4k epilogue route)
- `docs/task_workflow/input/nv-p2b-projection-boundary-reopen-record-20260805.md`
  (closed composed attention-O route)
- `nv-reduce-output-callify-redirect-audit-20260805.md`,
  `nv-reduce-output-typed-call-input-reopen-record-20260805.md` (audit
  closures)
