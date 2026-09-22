# NV Q4 FFN-down producer-owned Q8_1 DP4A gate - NO-GO, cannot construct as pre-wired (2026-08-13)

Date: 2026-08-13. Target: RTX 5090, native `DEV=NV`, sm_120, driver 595.84. Branch
`nvidia-bringup-20260731`, HEAD `4337ff29e`. GPU idle and `/tmp/gpu-bench.lock` free at
start; no process was killed. This is a measurement-only record; nothing is promoted.

## Verdict

**NO-GO - the gate is unmeasurable as pre-wired on this branch.** The candidate
(`Q4KFFNDownMMVQAdmission(16, owned_input_boundary=True)`) cannot construct: the admission is
reached at the call site but `q4k_ffn_down_mmvq_call` returns `None` for two independent
reasons. The harness's control arm additionally fails its own stale topology check, so no
semantic/topology comparison was produced. The timing bracket ran but is vacuous: the
"candidate" arm fell back to the installed route, so its `+8.62 us/token` is control-vs-control
drift, not a candidate measurement. No code change was made and production is untouched.

## Root cause: the branch moved under the candidate

The producer-owned successor was committed at `d63f0a7c9` (2026-08-12 17:48). Two promotions
landed hours later on the same day and are both live on this HEAD for NV sm_120:

| commit | time | promotion | effect on the control row |
| --- | --- | --- | --- |
| `b46905281` | 08-12 18:53 | M2a `decode_q4k_w1w3_fp16_store` | w1w3 producer stores fp16 in-kernel (`q4k_g3_lanemap_gemv_w1w3fused16_*`); the fp32 activation AFTER the owned boundary was designed to consume no longer exists |
| `8440c51b5` | 08-12 19:45 | M2b `decode_ffn_down_resadd` | ffn_down GEMV absorbs the h+ffn_out add in-kernel and threads `normed_h`; installed kernel renamed to `q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288` |

Promotion resolution confirmed on this HEAD: `decode_q4k_w1w3_fusion_promoted`, `decode_q4k_w1w3_fp16_store_promoted`,
and `decode_ffn_down_resadd_promoted` all return `True` for `("NV","sm_120")`. A control census on
the branch shows `q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288` 18x and
`q4k_g3_lanemap_gemv_w1w3fused16_12288_4096` 36x, with zero `q4k_g3_lanemap_gemv_4096_12288`.

## Why the candidate cannot construct

The call-site probe (admission attached to block 16, owned boundary, real d512 decode loop)
observed:

| call-site fact | observed |
| --- | --- |
| `x.dtype` | `dtypes.half` (M2a fused16 producer; owned boundary requires fp32) |
| `epilogue_inputs` keys | `["normed_h"]` (M2b promoted residual-add route; the call's first guard rejects non-empty epilogue inputs) |
| capability | `("NV","sm_120")` |
| `q4k_ffn_down_mmvq_call` return | `None` (both guards fail) |

Both failing guards live in `tinygrad/llm/q4k_ffn_down_mmvq.py` (line 143 epilogue guard, line
146 owned-boundary fp32 dtype guard), but their preconditions were voided by the M2a/M2b
promotions rather than by a candidate bug. No minimal candidate-only fix restores the pre-wired
gate: the harness control-arm topology check that also fails lives in the harness
(`expected_installed_count` of `q4k_g3_lanemap_gemv_4096_12288`), not in the candidate file, and
the owned-boundary contract itself (remove one fp16 materialize per lease, net-zero nodes) no
longer matches the promoted control.

## Gate 1: semantic + topology + full-logit (not runnable)

`--mode qualify --owned-input-boundary --indices 16 --count 8`. The control child failed its own
topology assertion before any comparison:

```text
production topology failed: {'provider_count': 0, 'consumer_count': 0, 'installed_q4_ffn_down_count': 0,
  'expected_provider_count': 0, 'expected_consumer_count': 0, 'expected_installed_count': 18,
  'adapter_kernel_count': 0, 'pass': False}
```

No token/logit or `exact_topology_delta` numbers exist because the control arm aborts on the stale
installed-kernel name. There is therefore no PASS/FAIL semantic evidence to report.

## Gate 2: settled reverse wall bracket (ran, but vacuous)

`--mode timing --owned-input-boundary --indices 16 --count 20 --reps 3`. All three arms share the
same token stream hash `6700c07ac628c8d6758a1b16144602fe55b82feae49741c4a3133ab10a091aa6` (prelude
13876), and the candidate arm never left the installed route:

| arm | median ms/token |
| --- | ---: |
| control A | `5.156869500` |
| candidate (fallback) | `5.161728250` |
| control B | `5.149341250` |
| control bracket median | `5.153105375` |
| candidate minus control | `+0.008622875` ms (`+8.62 us`) |

The harness's automatic verdict is `NO_GO_WALL`, but it is vacuous: because the admission is a
no-op on this branch, the bracket measured control against control. `+8.62 us/token` is inter-arm
drift and is NOT attributable to the candidate.

## Consequence

The producer-owned Q8_1 DP4A successor remains unmeasured and closed by default. Re-gating it on
this branch requires re-basing the owned-boundary contract against the promoted control (fp16
w1w3 store + in-kernel ffn_down residual add) or closing M2a/M2b for a research arm, plus updating
the harness control-kernel expectation - all out of scope for a measurement-only task. No code was
changed, no promotion record was touched, and `git status` contains only evidence artifacts.

## Evidence

- `extra/llm_research/decode/nv-q4-down-owned-q8-construct-diagnosis-20260813.json` (diagnosis)
- `extra/llm_research/decode/nv-q4-down-owned-q8-bracket-20260813.json` (harness bracket output)
- `extra/llm_research/decode/nv-q4-down-owned-q8-bracket-20260813/control-0.json`
- `extra/llm_research/decode/nv-q4-down-owned-q8-bracket-20260813/candidate-1.json`
- `extra/llm_research/decode/nv-q4-down-owned-q8-bracket-20260813/control-2.json`
