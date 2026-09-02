# NV Q6 reduction policy decision, 2026-08-31

## Decision

`RETAIN_ALL_PARTIALS`

Retain the admitted all-partials ascending reduction policy. Reject the llama-shaped standalone direct-final policy and its nonfinal-first source-order variant. The standalone policy is correct but slower, while nonfinal-first does not produce a material improvement over ascending standalone and adds compiler-visible main-kernel work.

The experimental `streamk_segment_order` selector is removed from the admitted broad CTA builder. The reduction substrate, harness, tests, and evidence remain uncommitted as rejected experiment material.

## Tested arms

| Arm | Policy | Physical segment order | Scratch ownership | Final segment |
|---|---|---|---|---|
| A: `all_partials_ascending` | Standalone fold of every segment | Ascending | All 294 active segments | Scratch |
| B: `llama_standalone_ascending` | Standalone fold with direct final | Ascending | 166 predecessor segments | Direct output |
| B-prime: `llama_standalone_nonfinal_first` | Standalone fold with direct final | Nonfinal first | 166 predecessor segments | Direct output |

B-prime is intentionally source-order confounded. It changes which logical segment is executed by each physical iteration while preserving the physical range and accumulator lifecycle.

## Exact final R31 result

The final result is `QUALIFIED_REDUCTION_SEQUENCE`, with `correctness_passed=true` and GPU lock acquisition recorded as true.

| Arm | Main median | Fixup median | Reset median | Reset-inclusive total median | Total min | Total max |
|---|---:|---:|---:|---:|---:|---:|
| A | 231.232 us | 25.056 us | 0.000 us | 256.256 us | 254.112 us | 258.592 us |
| B | 230.912 us | 29.888 us | 0.000 us | 260.864 us | 259.040 us | 263.040 us |
| B-prime | 231.872 us | 29.280 us | 0.000 us | 261.120 us | 260.064 us | 263.072 us |

| Paired comparison | R31 median delta | Paired MAD | Candidate wins | Range | Policy verdict |
|---|---:|---:|---:|---:|---|
| B minus A | +4.864 us | 0.704 us | 0/31 | +0.768 to +8.832 us | `REJECT_CANDIDATE` |
| B-prime minus B | +0.448 us | 0.832 us | 10/31 | -1.088 to +3.200 us | `RETAIN_CONTROL_NO_MATERIAL_WIN` |

The decision threshold was `max(3.0 us, 3 * paired MAD)` with at least 24/31 candidate wins required for promotion. The threshold was 3.0 us for both comparisons.

## Correctness and hashes

All three arms satisfy every enforced invariant:

- Final output is finite.
- GPU and CPU ordered folds are bit exact.
- Every required scratch slot is finite.
- Every unused scratch slot retains its NaN sentinel.
- Trusted direct-wide comparison has zero failing elements at `rtol=2e-5`, `atol=2e-3`.
- Maximum absolute error versus trusted direct-wide is `0.00067138671875`.
- Mean absolute error versus trusted direct-wide is `0.000021467494661919773`.
- Both standalone arms match A bit-for-bit for predecessor partials, direct-final values, and final output.

| Value | A SHA256 | B SHA256 | B-prime SHA256 |
|---|---|---|---|
| Final output | `51ab501f46be3f395263a7655bd204c2397d385496eb3f1d440a3c3e4ef11205` | same | same |
| Written partials | `924ee3a0166f770e03ce7e3f8ea35356da1f3c04370075cd5965a94921a58155` | `c9f17b4daef9409e1a84d3f7ba9d716bff526afea542785f241ed80927d7796c` | same as B |
| Full partial workspace | `14d509484f4db97f1196aa958632d77e1e12abfe06797a79b21b08c2cf1ae3d8` | `2451ea84d5f99cde3bae27f2cdbad774c3fee193f68dc615932fec33d5386498` | same as B |
| Pre-fix direct output | `d19f20ec1e9731826755438d83552704116eae74dcb81cb398f93bbd83e344c9` | `6f730b09954fac59e5a01fb5dda003f077a0dd98c8706db0635175031a38a186` | same as B |

The A pre-fix output is an untouched NaN sentinel and is not a candidate value.

## Deterministic ownership and accumulation contract

- Shape: `M=512`, `N=4096`, `K=12288`.
- Ownership: 170 CTAs partition 6144 K256 work units.
- Tiles: 128 output tiles, each with 48 K256 work units.
- Segments: 294 total, comprising 166 predecessor segments and 128 final segments.
- Contributors: 90 tiles have two contributors and 38 tiles have three contributors.
- Allocation: 340 scratch slots, two planes of 170 owners.
- Slot identity: `slot = plane * 170 + owner`.
- Fold order: nonfinal contributors in descending owner order, followed by the final contributor.
- Fold arithmetic: an explicit sequential FP32 `__fadd_rn` chain, identical in the GPU fixup and CPU oracle.
- A writes all 294 active scratch slots. B and B-prime write only the 166 predecessor slots and place the 128 final segments directly in the destination.
- The JSON evidence contains every segment descriptor, slot-map row, contributor count, final index, and physical launch order. No descriptor or order is reconstructed heuristically during analysis.

The 31 timed rounds execute in one process. Three-arm launch order follows a balanced six-permutation cycle, repeated five times with the first permutation used once more. Three warmup rounds precede measurement.

No counter state exists. Every active scratch and output destination is fully overwritten by its producer, so the required reset cost is exactly `0.0 us`. Every reported total uses `main_us + fixup_us + required_reset_us`.

## Normalized SASS and resources

| Kernel | Instructions | Registers | Stack | Shared | IMMA | LDSM | LDG | STG | STS | BAR | LDL/STL | FADD | ATOM/MEMBAR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A main | 5136 | 255 | 0 B | 1024 B | 256 | 32 | 109 | 64 | 73 | 4 | 0/0 | 1024 | 0/0 |
| A fixup | 672 | 255 | 16 B | 0 B | 0 | 0 | 196 | 64 | 0 | 0 | 4/8 | 192 | 0/0 |
| B main | 5224 | 255 | 0 B | 1024 B | 256 | 32 | 109 | 128 | 73 | 4 | 0/0 | 1024 | 0/0 |
| B fixup | 864 | 255 | 24 B | 0 B | 0 | 0 | 389 | 64 | 0 | 0 | 5/5 | 192 | 0/0 |
| B-prime main | 5576 | 255 | 0 B | 1024 B | 256 | 32 | 109 | 128 | 73 | 4 | 0/0 | 1024 | 0/0 |
| B-prime fixup | 864 | 255 | 24 B | 0 B | 0 | 0 | 389 | 64 | 0 | 0 | 5/5 | 192 | 0/0 |

| Kernel | Source SHA256 | Cubin SHA256 |
|---|---|---|
| A main | `40cca7e5d0b11d37c7df5843206eaf0a27cbb128dd556f879f7b4f43ace324d3` | `6eb663b3a3fd628e3394a0ce8f8780e108e47f40b887b0a75a0756dcf33e9137` |
| A fixup | `232cdd9fd88d51326419983712f08fdf2962f75fec77119a3faf14bfc7d582a4` | `483de2ee3eed3597932a8632f9892377ce054e77bfe34c2420fe5a5d54ff5514` |
| B main | `100f78ba1a08d1eab4f9beb00f1a3fabbcbe75a608698ee0fddb397c7bde1246` | `1f348eb29ab5d4d3442901af3970bab12b6cbdb69c4f352afc8182083f98c1a6` |
| B fixup | `ee8046c12acbc5deaa69f86438637e5603d0102335c0f7f4ae23dedd064ff6cc` | `9d2f425e8072cc2b42b20c477ca251ae2166b25f45de86bbb3b0aecdb1b07358` |
| B-prime main | `d4c7b2465de3731d6dbf3f36db2e7e94dd3c5ddd1ce51aa3c6f73570b204f176` | `681a10d51bbb260a259135d357ee46fa0ea2ba900b310068aa0839cc776baebe` |
| B-prime fixup | `ee8046c12acbc5deaa69f86438637e5603d0102335c0f7f4ae23dedd064ff6cc` | `9d2f425e8072cc2b42b20c477ca251ae2166b25f45de86bbb3b0aecdb1b07358` |

These are static SASS counts, not hardware performance counters.

## Reflection against llama

The pinned llama split is `201.216 us` main plus `8.640 us` fixup, totaling `209.856 us`.

| Arm | Main gap versus llama | Fixup gap versus llama | Total gap versus llama |
|---|---:|---:|---:|
| A | +30.016 us | +16.416 us | +46.400 us |
| B | +29.696 us | +21.248 us | +51.008 us |
| B-prime | +30.656 us | +20.640 us | +51.264 us |

The direct-final shape is reflected semantically in B and B-prime, but it is not reflected as a performance win. Moving final segments out of scratch saves no measurable main time and makes the standalone fixup more expensive. Reduction policy alone does not explain or close the llama gap.

The next deterministic optimization target remains A. The measured gaps separate into approximately 30 us in the main and 16 us in the all-partials fixup. Any next experiment should preserve A's exact descriptor and fold order, isolate one compiler-visible change, and use the same locked R31 paired protocol.

## Legality and progress safety

Direct-final writeback is legal here because a CTA writes only the final segment that it owns. It does not consume predecessor data or claim that the output is complete. The subsequent standalone fixup kernel launch is the global synchronization boundary, after which predecessor scratch and direct-final output are safe to read and fold.

A 170-CTA in-kernel final-participant reducer is not generally legal with ordinary CUDA block scheduling. There is no grid-wide barrier inside the kernel. A final CTA cannot safely spin until all predecessor CTAs publish because resident spinning CTAs can prevent unscheduled producers from becoming resident, causing a progress deadlock. Legal fusion would require a cooperative launch with a supported grid barrier and a proven residency bound, or a different synchronization primitive with an independently proven progress contract. Neither was implemented or tested.

The generated kernels contain zero `ATOM` and zero `MEMBAR` instructions. No counter state, progress-unsafe spin, or hidden reset is present.

## Proven, inferred, and unknown

### Proven by this run

- A, B, and B-prime produce the same final bits under the frozen fold order.
- B is 4.864 us slower than A by paired R31 median and loses all 31 pairs.
- B-prime is 0.448 us slower than B by paired median and does not cross the 3.0 us materiality threshold.
- B doubles main static global stores from 64 to 128 and raises fixup static loads from 196 to 389.
- B-prime adds 352 static main instructions over B without improving total time.
- The reset-inclusive result contains no reset or synchronization cost omitted from the formula.
- All tested policies remain slower than the pinned llama main-plus-fixup total.

### Inferred from the evidence

- B's regression is consistent with its larger direct-final fixup, extra global-store paths, higher static load count, and larger stack allocation.
- B-prime's extra address and control expressions are consistent with its 352-instruction main-kernel increase.
- The remaining llama gap is more likely in main scheduling, publication, data movement, and fixup specialization than in the choice between these three reduction policies.

These causal statements are inferences because static SASS census is not a hardware-counter attribution.

### Unknown after this run

- Whether a statically specialized all-partials fixup can approach llama's 8.640 us while retaining exact fold order.
- Which main-kernel instruction or dependency regions account for the approximately 30 us main gap.
- Whether cooperative grid synchronization is feasible for this exact device and grid with a proven residency contract.
- Cross-driver and cross-device timing stability.
- End-to-end model impact beyond this isolated `M=512`, `N=4096`, `K=12288` gate.

## Commands

Focused non-GPU tests:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  test/unit/test_nv_q6_oracle_reduction_policy.py \
  test/unit/test_nv_q6_oracle_streamk_single_body_packed.py
```

Result: `8 passed in 1.55s`.

Final GPU qualification:

```bash
flock -w 1200 /tmp/nv-q6-oracle-gpu.lock \
  env NV_Q6_GPU_LOCK_HELD=1 PYTHONPATH=. DEV=NV \
  .venv/bin/python extra/llm_research/prefill/bench_nv_q6_oracle_reduction_policy.py \
  --rounds 31 \
  --out docs/task_workflow/evidence/nv-q6-oracle-reduction-policy-20260831/result.json \
  --artifacts docs/task_workflow/evidence/nv-q6-oracle-reduction-policy-20260831/artifacts
```

Result: exit `0`; lock acquired; `QUALIFIED_REDUCTION_SEQUENCE`.

An initial pre-repair run failed B-prime because the experimental harness adapter classified finality by physical rather than logical segment. The one permitted harness-only repair changed that classifier to the explicit logical segment. The final persisted result above supersedes the diagnostic run.

## Evidence

- `docs/task_workflow/evidence/nv-q6-oracle-reduction-policy-20260831/result.json`
- `docs/task_workflow/evidence/nv-q6-oracle-reduction-policy-20260831/artifacts/`
- `extra/llm_research/prefill/nv_q6_oracle_reduction_policy.py`
- `extra/llm_research/prefill/bench_nv_q6_oracle_reduction_policy.py`
- `test/unit/test_nv_q6_oracle_reduction_policy.py`

No commit was created.
