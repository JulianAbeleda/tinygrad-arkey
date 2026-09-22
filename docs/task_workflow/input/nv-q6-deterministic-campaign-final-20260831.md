# NV Q6 deterministic campaign final ledger

Date: `2026-08-31`

Shape: `M=512, N=4096, K=12288`

Model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`

## Final state

```text
semantic_contract              = trusted_fp16_packed
runtime_range_contract         = nested lexical RANGE/END
physical_main_body             = one
main_grid                      = 170 CTAs
q8_panel1_schedule             = early
q6_fp32_d_publication          = packed-row initial publication
initial_publication_barrier    = combined Q6 + Q8
reduction_policy               = all_partials_ascending
latest_final_qualification     = reduction arm A
current_main_r31_median_us     = 231.232
current_fixup_r31_median_us    = 25.056
current_total_r31_median_us    = 256.256
pinned_llama_total_us          = 209.856
remaining_total_gap_us         = 46.400
remaining_total_gap_percent    = 22.11%
five_percent_target_us         = 220.3488
gap_to_five_percent_target_us  = 35.9072
duplicate_gpu_run_required     = false
```

The latest reduction arm A run is the final promotion qualification for the admitted route. It already covers the admitted packed arithmetic, lexical one-body kernel, combined publication, all-partials reduction, trusted reference, normalized SASS/resources, and locked same-process R31 timing. No duplicate GPU run is needed.

The llama comparison is explicit rather than inferred. The pinned llama baseline is `201.216 us` main plus `8.640 us` fixup, or `209.856 us` total. The current A total is `256.256 us`, exactly `46.400 us` slower by the displayed paired-total medians. The current total is `22.11%` above llama and remains `35.9072 us` above the `220.3488 us` 5% target.

## Proof vocabulary

| Class | Meaning | Limit |
|---|---|---|
| `REF` | Finite output passes the independently compiled wide-direct trusted reference at `rtol=2e-5, atol=2e-3` | Numerical qualification, not a performance claim |
| `BIT` | Required partial, fixup, trace, or final `uint32` arrays are bit-exact | Applies only to the named boundary |
| `TRACE` | Host replay and device observations agree at each named hop | Causal localization for the traced route and lanes |
| `FINITE` | Every member of a predeclared finite variant space was compiled and evaluated | Excludes only that finite space, not every possible algebraic formulation |
| `AST` | The UOp/RANGE/END topology has the required lexical or one-body structure | Structural compiler proof, not binary identity by itself |
| `SASS` | Cubin SHA, instruction families, resources, and classified spans prove the emitted binary contract | Static binary evidence, not hardware counters |
| `R31` | Same-process alternating or balanced R31 paired timing under the exclusive GPU lock | Promotion evidence only after correctness and structural gates pass |

Every promotion below requires the relevant `REF` or `BIT` evidence before `R31`. An unqualified fast arm is diagnostic only.

## Admitted commit chain

| Commit | Subject | Admitted fact |
|---|---|---|
| `0eb13c2ab` | `[codegen][nv] preserve Q6 fp16 weight-scale contract` | Packed FP16-rounded Q6 weight scales preserve the trusted numerical contract and beat both explicit repair and legacy arithmetic |
| `c788b12a8` | `[codegen] preserve nested runtime range lifecycles` | Dependent adjacent runtime ranges are not incorrectly merged across a lexical lifetime boundary |
| `a41b4230f` | `[codegen] keep nested END ranges lexical` | Split-END lowering uses flattened range ownership while preserving explicit lexical END placement |
| `0edb2dac3` | `[nv] admit one-body Q6 Stream-K route` | The 170-CTA route emits one physical compute body and materially beats the duplicated-body anchor |
| `7857aa86e` | `[nv] combine Q6 and Q8 publication barrier` | Early Q8 scheduling with combined initial Q6/Q8 publication is the admitted publication route |

## Sequential gate ledger

### Gate 0: direct dA versus factored dA

Executed command:

```bash
flock -w 1200 /tmp/nv-q6-oracle-gpu.lock \
  env PYTHONPATH=. DEV=NV \
  .venv/bin/python extra/llm_research/prefill/bench_nv_q6_oracle_full_streamk.py \
  --rounds 31 \
  --out docs/task_workflow/evidence/nv-q6-oracle-full-streamk-factor-da-gate0-20260831/result.json \
  --artifacts docs/task_workflow/evidence/nv-q6-oracle-full-streamk-factor-da-gate0-20260831/artifacts
```

| Measure | Direct dA | Factored dA |
|---|---:|---:|
| Trusted-reference max abs | `0.18719482421875` | `0.1871337890625` |
| Trusted-reference failures | `1,758,835` | `1,758,841` |
| GPU fixup versus declared CPU recurrence | bit-exact | bit-exact |
| Main R31 median | `225.344 us` | `285.056 us` |
| Fixup R31 median | `25.472 us` | `26.720 us` |
| Main+fixup R31 median | `251.040 us` | `312.192 us` |
| Whole-kernel instructions | `8,176` | `8,328` |
| Stack | `16 B` | `288 B` |
| LDL / STL | `7 / 14` | `251 / 377` |

Paired direct-minus-factored main was `-59.904 us`; paired total was `-61.280 us`; direct won `31/31` main and `31/31` total pairs.

Proof: `BIT + SASS + R31`, followed by `REF` failure.

Decision: `REJECT_NUMERICAL`. Direct dA proved that factor placement caused substantial spill and timing cost, but it did not repair the approximately `0.187` numerical residual. It remained a useful diagnostic arithmetic form, not an admissible route.

Evidence ledger: `docs/task_workflow/input/nv-q6-direct-da-full-route-gate0-decision-20260831.md`.

### Gate 1: 128-CTA full-K route and 26 associations

Executed command:

```bash
flock -w 1200 /tmp/nv-q6-oracle-gpu.lock \
  env PYTHONPATH=. DEV=NV \
  .venv/bin/python extra/llm_research/prefill/bench_nv_q6_oracle_fullk_tiles.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --rounds 31 \
  --out docs/task_workflow/evidence/nv-q6-oracle-fullk-tiles-gate1-20260831/result.json \
  --artifacts docs/task_workflow/evidence/nv-q6-oracle-fullk-tiles-gate1-20260831/artifacts
```

Isolation contract:

```text
grid                         = (128, 1, 1)
block                        = (256, 1, 1)
tile_grid                    = (4, 32)
epochs_per_cta               = 48 ascending K256 epochs
partial_workspace            = absent
fixup                        = absent
output                       = direct row-major 512x4096
implicit_baselines           = 2
controlled association arms  = 24
total arms                   = 26
```

Result:

```text
verdict                      = FAIL_NUMERICAL_FINITE_SWEEP_EXHAUSTED
trusted_reference            = PASS
tile_aligned_direct          = FAIL
tile_aligned_factored        = FAIL
controlled_sass_proofs       = 24/24
numerically_passing_arms     = 0/26
max_abs_envelope             = 0.187255859375 .. 0.18731689453125
mean_abs_envelope            = 0.0136876795441 .. 0.0136876944453
failure_envelope             = 1,758,795 .. 1,758,882
```

All 24 controlled variants emitted their requested `FMUL/FADD/FFMA` census, with `256 IMMA` and `32 LDSM` in every controlled arm. The direct and factored baselines measured `284.320 us` and `349.056 us`; paired direct-minus-factored was `-63.648 us`.

Proof: `FINITE + SASS + REF failure`.

Decision: the error was not created by Stream-K partial addressing, cross-CTA subtotal association, fixup ordering, any enumerated order-preserving four-term FP32 tree, or the enumerated FMA contraction regimes. The next gate had to localize a real K256 hop below those layers.

Evidence ledger: `docs/task_workflow/input/nv-q6-fullk-tiles-gate1-decision-20260831.md`.

### Gate 2: real K256 prefix and hop localization

Executed command:

```bash
flock -w 1200 /tmp/nv-q6-oracle-gpu.lock \
  env PYTHONPATH=. DEV=NV \
  .venv/bin/python extra/llm_research/prefill/bench_nv_q6_oracle_kprefix_trace.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --depths 1,2,4,8,16,24,32,40,48 \
  --rounds 31 \
  --out docs/task_workflow/evidence/nv-q6-oracle-kprefix-hop-20260831/result.json \
  --artifacts docs/task_workflow/evidence/nv-q6-oracle-kprefix-hop-20260831/artifacts
```

The first divergence occurred at `depth=1`, `epoch=0`, hop `WEIGHT_SCALE_CONTRACT`.

The legacy body effectively used:

```text
dot = integer_scale_weighted_dot(c0, c1, signed_scale0, signed_scale1)
acc = acc + fp32(d) * fp32(dot)
```

The trusted contract requires two independently rounded weight scales:

```text
w0 = fp32(fp16(fp32(d) * fp32(signed_scale0)))
w1 = fp32(fp16(fp32(d) * fp32(signed_scale1)))
acc = acc + fp32(c0) * w0 + fp32(c1) * w1
```

`2,064,507 / 3,145,728`, or `65.629%`, of evaluated `d * signed_scale` products changed under the required FP16 rounding.

Trace classification:

| Hop | Result |
|---|---|
| Packed real Q6/Q8 prefixes | exact |
| Q6 shared publication | exact |
| Q8 publication and traced carriers | exact |
| Decoded Q6 values and scales | exact |
| IMMA `c0/c1` | exact |
| Integer correction terms | exact |
| Lane/output and epoch mapping | exact |
| Legacy weight-scale contract | first mismatch |
| Repaired FP32 steps and accumulator chain | bit-exact |
| Final traced accumulator versus output | bit-exact |

Full 128-tile result:

| Measure | Legacy | Explicit trusted-FP16 repair |
|---|---:|---:|
| Max abs | `0.187255859375` | `0.0001220703125` |
| Mean abs | approximately `0.01368768` | `3.190014581377909e-7` |
| Failures | `1,758,882` | `0` |
| R31 median | `282.112 us` | `299.168 us` |
| Paired repair-minus-legacy | - | `+16.928 us`, `0/31` repair wins |

Proof: `TRACE + BIT + REF + SASS + R31`.

Decision: `PASS_REPAIR_FULL_AB`, but `REJECT_EXPLICIT_IMPLEMENTATION_TIMING`. The semantics became mandatory; the explicit implementation was not promoted.

Evidence ledger: `docs/task_workflow/input/nv-q6-kprefix-hop-localization-decision-20260831.md`.

### Gate 2A: packed trusted-FP16 implementation

Executed qualification command:

```bash
flock -w 1200 /tmp/nv-q6-oracle-gpu.lock \
  env PYTHONPATH=. DEV=NV \
  .venv/bin/python extra/llm_research/prefill/bench_nv_q6_oracle_kprefix_trace.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --depths 1,2,4,8,16,24,32,40,48 \
  --rounds 31 \
  --out docs/task_workflow/evidence/nv-q6-oracle-kprefix-hop-20260831/packed-result.json \
  --artifacts docs/task_workflow/evidence/nv-q6-oracle-kprefix-hop-20260831/packed-artifacts
```

Packed Q6 shared row contract:

```text
words  0..63 = Q6 payload
word       64 = FP32 d bits
words 65..72 = eight packed FP16 (d * signed_scale) pairs
words 73..75 = padding
row_words      = 76
dynamic_shared = 57,344 bytes
```

The packed arm was bit-identical to the explicit trusted-FP16 repair and had zero tolerance failures.

| Arm | R31 median |
|---|---:|
| Legacy | `283.104 us` |
| Explicit trusted-FP16 repair | `299.520 us` |
| Packed trusted-FP16 | `271.616 us` |

Paired packed-minus-explicit was `-28.032 us`; paired packed-minus-legacy was `-11.072 us`; packed won `31/31` in both comparisons.

Packed SASS/resource census:

```text
instructions = 5,056
registers    = 255
stack        = 0 B
LDL / STL    = 0 / 0
IMMA / LDSM  = 256 / 32
LDG / STS    = 109 / 73
BAR          = 5
```

Proof: `TRACE + BIT + REF + SASS + R31`.

Decision: `PROMOTE_TRUSTED_FP16_PACKED`, admitted as `0eb13c2ab`.

Evidence ledger: `docs/task_workflow/input/nv-q6-trusted-fp16-packed-decision-20260831.md`.

### Gate 2B: nested RANGE and lexical END compiler substrate

Focused non-GPU command:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  test/unit/test_nested_runtime_range_lifecycle.py \
  test/unit/test_generic_tc_split_range_axis.py \
  test/unit/test_rangeify_multireduce.py \
  test/unit/test_nv_q6_oracle_streamk_single_body_packed.py
```

Result: `19 passed, 2 skipped`.

The first compiler change prevented dependent adjacent runtime ranges from being merged across lexical lifetime boundaries. The second made split-END analysis use flattened range ownership and split only explicitly ended lexical ranges.

Proof: `AST + focused regression tests`.

Decision: admit `c788b12a8` and `a41b4230f`. This was the required compiler substrate for expressing two runtime segments inside one physical body without corrupting RANGE/END lifetimes.

### Gate 3: genuine 170-CTA one-body route

Executed command:

```bash
flock -w 1200 /tmp/nv-q6-oracle-gpu.lock \
  env PYTHONPATH=. DEV=NV \
  .venv/bin/python extra/llm_research/prefill/bench_nv_q6_oracle_streamk_single_body_packed.py \
  --rounds 31 \
  --out docs/task_workflow/evidence/nv-q6-oracle-streamk-single-body-packed-20260831/result.json \
  --artifacts docs/task_workflow/evidence/nv-q6-oracle-streamk-single-body-packed-20260831/artifacts
```

Both arms passed the trusted reference with max abs `0.00067138671875`, mean abs `0.00002147154009435326`, and zero failures. GPU fixup versus CPU recurrence, required partials, and final outputs were bit-exact.

| Measure | Duplicated-body anchor | Genuine one-body candidate |
|---|---:|---:|
| Cubin SHA256 | `16cffa...` | `1df61553f7ebb9904108c2ed14b0c256abdce067a2ae3a1bfe45fcc86a243e1f` |
| Static instructions | `10,376` | `5,192` |
| Registers | `255` | `255` |
| Stack | `64 B` | `48 B` |
| LDL / STL | `24 / 24` | `12 / 12` |
| IMMA / LDSM | `512 / 64` | `256 / 32` |
| LDG / STS | `218 / 146` | `109 / 73` |
| BAR | `11` | `5` |
| Main R31 median | `308.640 us` | `246.912 us` |
| Fixup R31 median | `25.696 us` | `25.088 us` |
| Total R31 median | `334.560 us` | `271.840 us` |

Paired one-body-minus-anchor was `-61.696 us` main with `31/31` wins, `-0.736 us` fixup with `29/31` wins, and `-62.592 us` total with `31/31` wins. The paired total improvement was `18.71%`.

Proof: `AST + BIT + REF + SASS + R31`.

Decision: `PROMOTE_ONE_PHYSICAL_BODY`, admitted as `0edb2dac3`.

Evidence ledger: `docs/task_workflow/input/nv-q6-one-physical-body-packed-decision-20260831.md`.

### Gates 4 and 5: Q8, FP32 d, and combined publication

Executed four-arm command:

```bash
flock -w 1200 /tmp/nv-q6-oracle-gpu.lock \
  env PYTHONPATH=. DEV=NV \
  .venv/bin/python extra/llm_research/prefill/bench_nv_q6_oracle_publication_gates.py \
  --rounds 31 \
  --out docs/task_workflow/evidence/nv-q6-oracle-publication-gates-20260831/result.json \
  --artifacts docs/task_workflow/evidence/nv-q6-oracle-publication-gates-20260831/artifacts
```

All four arms were finite and passed the trusted reference with max abs `0.00067138671875`, mean abs `0.00002147154009435326`, and zero failures. Required partial and final `uint32` buffers were bit-exact, and GPU fixup matched the CPU recurrence.

The FP32 Q6 `d` value was already part of the admitted packed row at word 64 and was already published in the initial Q6 publication. Therefore a nominal separate FP32-d publication toggle mapped to the existing anchor/no-op contract; it was not an independent semantic candidate. The genuine independent axes were Q8 panel-1 scheduling and whether initial Q6 plus Q8 publication shared one barrier.

| Arm | Instructions | Stack | LDL/STL | BAR | Classified panel-1 span | Main | Fixup | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| early / separate | `5,192` | `48 B` | `12/12` | `5` | `2,221` | `246.752 us` | `25.792 us` | `272.448 us` |
| late / separate | `5,368` | `248 B` | `99/131` | `5` | `443` | `295.520 us` | `26.432 us` | `321.888 us` |
| early / combined | `5,136` | `0 B` | `0/0` | `4` | `2,225` | `231.264 us` | `25.344 us` | `256.672 us` |
| late / combined | `5,144` | `0 B` | `0/0` | `4` | `1,827` | `226.432 us` | `25.568 us` | `252.224 us` |

Every arm retained `255` registers, `0 B` local static memory, `1,024 B` static shared memory, `256 IMMA`, `32 LDSM`, `109 LDG`, `73 STS`, `1,544 FMUL`, `1,024 FADD`, and `0 FFMA`. The classifier found exactly `18/18` Q8 panel-1 loads/stores in every arm.

Sequential paired decisions:

| Comparison | Main delta | Fixup delta | Total delta | Wins | Decision |
|---|---:|---:|---:|---:|---|
| late/separate minus early/separate | `+48.768 us` | `+0.704 us` | `+49.440 us` | `0/31` main, `0/31` total | `NO_GO_LATE_Q8` |
| early/combined minus early/separate | `-15.616 us` | `-0.384 us` | `-16.192 us` | `31/31` main, `28/31` fixup, `31/31` total | `PROMOTE_COMBINED` |
| late/combined minus late/separate | `-68.576 us` | `-0.928 us` | `-69.664 us` | `31/31` main, `30/31` fixup, `31/31` total | timing pass, sequentially ineligible |

Proof: `BIT + REF + SASS-classified factorial controls + R31`.

Decision: `PROMOTE_COMBINED_ONLY`. Standalone late Q8 failed its prerequisite gate and was removed. The admitted route remains early Q8 scheduling with combined initial publication, committed as `7857aa86e`.

Evidence ledger: `docs/task_workflow/input/nv-q6-one-body-publication-gates-decision-20260831.md`.

### Gate 6: reduction policy A/B/B'

Executed command:

```bash
flock -w 1200 /tmp/nv-q6-oracle-gpu.lock \
  env NV_Q6_GPU_LOCK_HELD=1 PYTHONPATH=. DEV=NV \
  .venv/bin/python extra/llm_research/prefill/bench_nv_q6_oracle_reduction_policy.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --rounds 31 \
  --arms all_partials_ascending,llama_standalone_ascending,llama_standalone_nonfinal_first \
  --out docs/task_workflow/evidence/nv-q6-oracle-reduction-policy-20260831/result.json \
  --artifacts docs/task_workflow/evidence/nv-q6-oracle-reduction-policy-20260831/artifacts
```

Arm definitions:

| Arm | Short name | Output policy | Segment order | Final owner writes destination in main |
|---|---|---|---|---|
| `all_partials_ascending` | A | all partials | ascending | no |
| `llama_standalone_ascending` | B | llama standalone final | ascending | yes |
| `llama_standalone_nonfinal_first` | B' | llama standalone final | nonfinal first | yes |

All arms were finite and passed the trusted reference with max abs `0.00067138671875`, mean abs `0.000021467494661919773`, and zero failures. For B and B', predecessor partials were bit-exact to A, direct-final pre-fix output was bit-exact to the output reconstructed from A partials, and final output was bit-exact to A. All final-output SHA256 values were `51ab501f46be3f395263a7655bd204c2397d385496eb3f1d440a3c3e4ef11205`.

Normalized binary/resource result:

| Arm | Main SHA256 | Main instr | Main stack | Main LDL/STL | Main STG | Fixup SHA256 | Fixup instr | Fixup stack | Fixup LDL/STL | Fixup LDG |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| A | `6eb663b3a3fd628e3394a0ce8f8780e108e47f40b887b0a75a0756dcf33e9137` | `5,136` | `0 B` | `0/0` | `64` | `483de2ee3eed3597932a8632f9892377ce054e77bfe34c2420fe5a5d54ff5514` | `672` | `16 B` | `4/8` | `196` |
| B | `1f348eb29ab5d4d3442901af3970bab12b6cbdb69c4f352afc8182083f98c1a6` | `5,224` | `0 B` | `0/0` | `128` | `9d2f425e8072cc2b42b20c477ca251ae2166b25f45de86bbb3b0aecdb1b07358` | `864` | `24 B` | `5/5` | `389` |
| B' | `681a10d51bbb260a259135d357ee46fa0ea2ba900b310068aa0839cc776baebe` | `5,576` | `0 B` | `0/0` | `128` | `9d2f425e8072cc2b42b20c477ca251ae2166b25f45de86bbb3b0aecdb1b07358` | `864` | `24 B` | `5/5` | `389` |

Every main retained `255` registers, `1,024 B` static shared memory, `0 B` local static memory, `256 IMMA`, `32 LDSM`, `109 LDG`, `73 STS`, `4 BAR`, `1,024 FADD`, and `0 FFMA`.

Locked R31 result:

| Arm | Main | Fixup | Required reset | Per-round total median |
|---|---:|---:|---:|---:|
| A | `231.232 us` | `25.056 us` | `0.000 us` | `256.256 us` |
| B | `230.912 us` | `29.888 us` | `0.000 us` | `260.864 us` |
| B' | `231.872 us` | `29.280 us` | `0.000 us` | `261.120 us` |

Paired total B-minus-A was `+4.864 us` with `0/31` B wins, so B was rejected. Paired total B'-minus-B was `+0.448 us` with `10/31` B' wins, below the `3.0 us` materiality requirement and below the required `24/31` wins, so B' was retained only as a control and not promoted.

Proof: `BIT + REF + AST + SASS + balanced R31`.

Decision: `QUALIFIED_REDUCTION_SEQUENCE`. A remains the admitted reduction policy. B proves that moving the final owner write into the main does not improve total time because its more expensive fixup dominates the `0.320 us` main-median reduction. B' proves that nonfinal-first physical order does not rescue that policy and adds substantial main address/control code.

This A result is the final promotion qualification. Do not rerun the GPU merely to restate the current admitted number.

Evidence: `docs/task_workflow/evidence/nv-q6-oracle-reduction-policy-20260831/result.json`.

## Current admitted hop-by-hop route

```text
load real Q6_K blocks and matching Q8 activation tiles
  -> assign the full Stream-K schedule to 170 CTAs
  -> enter one physical compute body per CTA
  -> use lexical nested RANGE/END for each CTA's one or two logical segments
  -> publish the 76-word packed Q6 row
       [Q6 payload | FP32 d | eight packed FP16(d*scale) pairs | pad]
  -> publish initial Q8 data under the same combined initial barrier
  -> keep Q8 panel-1 publication at the admitted legacy-compatible early point
  -> execute the exact Q6 decode, Q8 carrier mapping, and integer corrections
  -> issue 256 IMMA and 32 LDSM in the one-body main
  -> consume each c0/c1 pair with independently FP16-rounded packed weights
  -> accumulate epochs in the traced ascending semantic order
  -> write every logical segment subtotal to its deterministic partial slot
  -> run the ordered all-partials ascending fixup
  -> write the final row-major FP32 output
```

Reference pseudocode:

```python
for cta in range(170):
  for segment in lexical_segments_owned_by(cta):
    q6_row = publish_q6_payload_d_and_packed_fp16_weights(segment)
    q8_panel0 = publish_q8_panel0(segment)
    barrier()  # one combined initial Q6/Q8 publication barrier
    q8_panel1 = publish_q8_panel1_at_early_anchor(segment)

    acc = fp32_zero()
    for epoch in segment.ascending_k256_epochs:
      c0, c1 = exact_imma_and_integer_correction(epoch, q6_row, q8_panel0, q8_panel1)
      w0, w1 = unpack_fp16_rounded_weight_pair(q6_row, epoch)
      acc = trusted_ordered_accumulate(acc, c0, fp32(w0), c1, fp32(w1))

    partials[deterministic_slot(segment)] = acc

for output_tile in range(128):
  output[output_tile] = ordered_fp32_sum(partials[ascending_slots(output_tile)])
```

## Rejected hypotheses and bounded conclusions

| Hypothesis | Evidence | Conclusion |
|---|---|---|
| Factoring or distributing `dA` repairs the output | Gate 0 | Rejected; placement changes speed and spills, not the residual |
| Stream-K partial slots or fixup create the residual | Gate 1 128-CTA no-partial/no-fixup route | Rejected for this route |
| A selectable post-dot FP32 association or contraction repairs the residual | Gate 1 complete declared 26-arm space | Rejected within the enumerated finite space |
| Q6/Q8 publication, decode, IMMA, correction, or mapping is the first mismatch | K256 hop trace | Rejected at the traced first failing epoch; those hops were exact |
| Applying `d` after the integer scale-weighted dot is equivalent enough | K256 hop trace | Rejected; independently FP16-rounded `d*scale` products are required |
| The explicit trusted-FP16 repair is the implementation to keep | Gate 2 timing | Rejected on timing despite correct semantics |
| Two source branches are necessary for two segment counts | One-body Gate 3 | Rejected; lexical nested ranges express one genuine body |
| Late Q8 panel-1 publication is an independent win | Publication Gate 4 | Rejected; late/separate regressed total by `49.440 us` paired |
| FP32 Q6 d needs a new standalone publication gate | Packed/publication contract | Rejected as an independent axis; d is already word 64 of the initial packed Q6 row |
| Separate Q6 and Q8 initial barriers are required | Publication Gate 5 | Rejected; combined publication removed a barrier and spills and won `31/31` total pairs |
| Main-thread final-owner destination write reduces total time | Reduction B | Rejected; total regressed `4.864 us`, `0/31` wins |
| Nonfinal-first segment order rescues standalone-final reduction | Reduction B' | Rejected; `+0.448 us` paired versus B and only `10/31` wins |

These are bounded conclusions. In particular, the publication factorial observed a fast late/combined arm, but late scheduling failed its declared prerequisite as an independent change. That interaction is not admitted and must not silently reappear in the route.

## Deterministic next decision tree

The next work starts from reduction A, not from an older one-body or publication timing. Every candidate uses the same model, shape, trusted reference, exclusive lock, and balanced same-process R31 protocol.

```text
ANCHOR A
  main cubin SHA  = 6eb663b3a3fd628e3394a0ce8f8780e108e47f40b887b0a75a0756dcf33e9137
  fixup cubin SHA = 483de2ee3eed3597932a8632f9892377ce054e77bfe34c2420fe5a5d54ff5514
  total           = 256.256 us

G7: isolate all-partials fixup cost
  freeze main cubin SHA and written-partial SHA
  A0 = current metadata-driven ordered fixup
  A1 = compile the immutable one/two-contributor ownership schedule into addresses/control
  A2 = coalesce or vectorize partial reads without changing scalar FP32 add order
  test exactly one transformation per arm
  require REF + final BIT + normalized fixup SASS + paired R31
  if candidate delta <= -3.0 us and wins >= 24/31:
    promote candidate, freeze new fixup SHA, recompute remaining gap
  else:
    retain A and close that transformation

G8: isolate main address/publication scheduling
  freeze packed arithmetic, ownership, partial layout, and fixup cubin SHA
  first remove address/control work without changing Q8 timing
  optionally test late/combined only as a newly declared interaction gate
  if testing the interaction:
    compare directly against admitted early/combined A
    require exact 18/18 panel-1 classification and no unrelated SASS drift
  require REF + partial BIT + final BIT + normalized main SASS + paired R31
  apply the same -3.0 us and 24/31 promotion threshold

G9: reduction-policy reopening condition
  do not retry B or B' unchanged
  reopen direct-final reduction only if a new representation removes its extra fixup loads
  require a declared recurrence, CPU bit replay, A-equivalent final uint32, and total-time win

STOP CONDITION
  current target = total <= 220.3488 us
  if reached, run the final pinned llama comparison once
  if not reached, record the new exact residual and continue from the newest admitted anchor
```

Required assertions for every next arm:

| Gate | Assertion |
|---|---|
| Correctness | finite, trusted-reference failures `0`, and no tolerance regression |
| Semantic boundaries | required partial, pre-fix, fixup replay, and final `uint32` equality |
| Structure | named UOp/RANGE/END and ownership signatures unchanged outside the isolated toggle |
| Binary | cubin SHA recorded; registers, stack, local/shared bytes, LDL/STL, IMMA/LDSM, LDG/STG/STS, BAR, FADD/FMUL/FFMA recorded |
| Timing | exclusive `flock`, balanced same-process R31, paired median delta, MAD, and wins |
| Admission | at least `3.0 us` material paired total win and at least `24/31` wins unless a stricter predeclared gate applies |

The highest-leverage first branch is G7 because current A spends `25.056 us` in fixup versus llama's `8.640 us`, while B showed that merely moving the final write into main increases fixup loads and loses total time. The second branch is main address/publication scheduling, where the combined barrier already removed all main spills and left a stable zero-stack binary anchor.

## Canonical evidence index

| Stage | Decision/evidence |
|---|---|
| Gate 0 | `docs/task_workflow/input/nv-q6-direct-da-full-route-gate0-decision-20260831.md` |
| Gate 1 | `docs/task_workflow/input/nv-q6-fullk-tiles-gate1-decision-20260831.md` |
| K256 localization | `docs/task_workflow/input/nv-q6-kprefix-hop-localization-decision-20260831.md` |
| Packed trusted FP16 | `docs/task_workflow/input/nv-q6-trusted-fp16-packed-decision-20260831.md` |
| One body | `docs/task_workflow/input/nv-q6-one-physical-body-packed-decision-20260831.md` |
| Publication | `docs/task_workflow/input/nv-q6-one-body-publication-gates-decision-20260831.md` |
| Reduction A/B/B' | `docs/task_workflow/evidence/nv-q6-oracle-reduction-policy-20260831/result.json` |

## Final admission statement

The deterministic campaign moved from a fast but incorrect direct-dA diagnostic to a traced numerical repair, packed that repair without losing its FP16 rounding semantics, supplied the compiler range substrate needed for one genuine body, admitted the 170-CTA one-body route, removed one publication barrier, and rejected two direct-final reduction policies. The current admitted path is semantically qualified and reproducible at `256.256 us` total. It is still `46.400 us` behind the pinned llama `209.856 us` baseline, so the campaign has moved the needle but has not closed the performance target.
