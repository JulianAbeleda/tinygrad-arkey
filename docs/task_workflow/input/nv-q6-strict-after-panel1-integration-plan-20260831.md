# Q8-only strict-after integration plan

Status: scaffolded and fail-closed pending the core scheduling substrate. No
core/compiler/Q6-builder implementation has been changed.

## Frozen route

| item | frozen identity |
|---|---|
| admitted combined/all-partials main | `6eb663b3a3fd628e3394a0ce8f8780e108e47f40b887b0a75a0756dcf33e9137` |
| admitted ordered all-partials fixup | `483de2ee3eed3597932a8632f9892377ce054e77bfe34c2420fe5a5d54ff5514` |
| main symbol | `nv_q6_oracle_broad_cta_prefetch_combined_publish_oracle_publisher_trusted_fp16_packed_ws_segments_in_cta_streamk_s0` |

The default admitted builder must continue producing the frozen main cubin.
The strict-after candidate must produce a different cubin while preserving the
fixup cubin exactly.

## Exact lower dependency token

The anchor token is binary-selected, not inferred from CUDA source:

```text
phase                    phase-0 arithmetic tail
predecessor IMMA PC      0x9f70
dependency token PC      0x9f80
dependency ordinal       2552
token instruction        FADD R167, R53, R36
first legal LDG ordinal  2553
overwrite barrier        0xa930, ordinal 2707
anchor first STS         0xa990, ordinal 2713
publication barrier      0xaab0, ordinal 2731
```

An LDG immediately after the token has a 160-instruction bound to the anchor's
first STS. Ten independent phase-0 IMMA instructions remain at PCs `0xa060`,
`0xa180`, `0xa230`, `0xa280`, `0xa2b0`, `0xa340`, `0xa4d0`, `0xa670`,
`0xa760`, and `0xa810`. This provides useful latency-cover work without the
candidate's current 2,225-instruction live range.

The future arm must identify this semantic arithmetic token in its compiled
candidate and fail if the mapping is not recoverable. Absolute candidate PCs
may shift; the enforced relation is token ordinal < first panel-1 LDG ordinal,
with all loads before the overwrite barrier and span at most 160.

## Exact 18-word publication contract

| word | global offset | anchor LDG PC/register | shared offset | anchor STS PC |
|---:|---:|---|---:|---:|
| 0 | `0x4800` | `0x1e80 R192` | `0x9800` | `0xa990` |
| 1 | `0x4c00` | `0x1eb0 R191` | `0x9c00` | `0xa9a0` |
| 2 | `0x5000` | `0x1ee0 R190` | `0xa000` | `0xa9b0` |
| 3 | `0x5400` | `0x1f10 R189` | `0xa400` | `0xa9c0` |
| 4 | `0x5800` | `0x1f40 R188` | `0xa800` | `0xa9d0` |
| 5 | `0x5c00` | `0x1f70 R187` | `0xac00` | `0xa9e0` |
| 6 | `0x6000` | `0x1fa0 R186` | `0xb000` | `0xa9f0` |
| 7 | `0x6400` | `0x1fd0 R185` | `0xb400` | `0xaa00` |
| 8 | `0x6800` | `0x2020 R184` | `0xb800` | `0xaa10` |
| 9 | `0x6c00` | `0x2050 R183` | `0xbc00` | `0xaa20` |
| 10 | `0x7000` | `0x20b0 R119` | `0xc000` | `0xaa30` |
| 11 | `0x7400` | `0x20e0 R182` | `0xc400` | `0xaa40` |
| 12 | `0x7800` | `0x2110 R178` | `0xc800` | `0xaa50` |
| 13 | `0x7c00` | `0x2170 R177` | `0xcc00` | `0xaa60` |
| 14 | `0x8000` | `0x21b0 R176` | `0xd000` | `0xaa70` |
| 15 | `0x8400` | `0x21f0 R175` | `0xd400` | `0xaa80` |
| 16 | `0x8800` | `0x2220 R174` | `0xd800` | `0xaa90` |
| 17 | `0x8c00` | `0x2280 R173` | `0xdc00` | `0xaaa0` |

Future register numbers and PCs are observations, not frozen allocation gates.
The 18 logical offsets and their matching shared destinations are frozen.

## Fail-closed integration sequence

1. Wait for the core agent to publish its actual helper/UOp ABI.
2. Bind that released helper explicitly as `MODULE:SYMBOL`; the repository has
   no speculative default import or builder enum.
3. Add an isolated Q8 candidate factory only after the ABI is known. Keep the
   admitted default builder byte-identical.
4. Run the harness in non-GPU mode. Resolve the helper, factory, dependency
   token, anchor hash, candidate compile, normalized SASS, and resources.
5. Stop before GPU on any missing primitive, unmappable token, hash drift,
   incorrect panel mapping, SASS mismatch, spill, or resource failure.
6. Only after all compile gates pass, use the existing locked Gate-7 trusted
   correctness and same-process alternating timing path.

## Hard gates

Correctness:

- trusted-reference exactness;
- candidate/anchor partial uint32 identity;
- candidate/anchor final uint32 identity;
- active outputs finite and unused output slots NaN.

SASS and resources:

- exactly 18 panel-1 LDG and 18 matching STS;
- first LDG after the recovered dependency token and initial combined barrier;
- every LDG before overwrite barrier; every STS between overwrite and publish;
- first-load-to-first-store span `<=160`;
- `IMMA/LDSM/LDS/LDG/STS/STG/BAR = 256/32/176/109/73/64/4`;
- `I2FP/FMUL/FADD/FFMA = 1024/1544/1024/0`;
- instructions `<=5144`, registers `<=255`;
- stack/local/LDL/STL all zero.

Timing:

- `flock -w 1200 /tmp/nv-q6-oracle-gpu.lock`;
- one process, three warmups, alternating randomized R31;
- both main and total paired median delta `<= -3 us`;
- both main and total wins `>=24/31`.

## Isolated files

Present now:

- `extra/llm_research/prefill/bench_nv_q6_oracle_strict_after_panel1.py`;
- `docs/task_workflow/input/nv-q6-strict-after-panel1-integration-plan-20260831.md`;
- `docs/task_workflow/evidence/nv-q6-strict-after-panel1-integration-20260831/preflight.json` after preflight.

Deferred until ABI release:

- the core agent's chosen primitive and its backend-neutral unit test;
- one isolated Q8 candidate factory or minimal default-off builder hook, named
  only after the released ABI is known;
- the full compile/correctness/timing result JSON and decision ledger.

The scaffold imports only the Python standard library before resolving the
explicit helper and factory strings. Therefore the unavailable-core result
cannot compile a kernel, discover a device, or start GPU work.
