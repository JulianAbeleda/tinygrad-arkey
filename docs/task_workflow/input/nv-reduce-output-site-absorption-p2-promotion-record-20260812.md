# NV reduce-output per-site absorption P2 promotion record (NV sm_120)

Status: **PROMOTED (bar cleared without a waiver). The per-site admission
that the CPU arm could not prove is now established on the GPU arm: under
the candidate conditions the NV sm_120 census admits the fp32 q/k bodies
(`reduce_output_rmsnorm_32_128` x36 + `reduce_output_rmsnorm_8_128` x36)
and the FFN-down 1_4096 bodies (x19), with zero weight materializations and
a net program count strictly below the baseline-context arm. Exact
full-logit SHA-256 is identical to control, the token stream is bit-exact
across every arm of the reverse bracket, and the +50 us/token bar clears
against BOTH bracketing controls (+52.20 / +53.55 us), so the package books
without asking the principal for a waiver. The prior fp32 q/k promotion
record (`nv-reduce-output-fp32-qk-promotion-record-20260812.md`) stays the
promotion source for the already-landed route; this record is the P2
admission + wall evidence for the per-site scope. No policy change: the
route policy file is byte-identical to HEAD (already promoted NV sm_120).**

Scope: `docs/task_workflow/input/nv-reduce-output-site-absorption-scope-20260812.md`
(P2 section). Bracket evidence:
`docs/task_workflow/evidence/nv-reduce-output-site-absorption-p2-ab-20260812.json`.
Branch `nvidia-bringup-20260731` at HEAD `ba07391ed`. Run 2026-08-13 on the
RTX 5090 (NV sm_120, Qwen3-8B-Q4_K_M, fixed depth 512, `--count 32`,
`--reps 5`, `--max-context 1024`) under the shared GPU bench lock via
`extra/llm_research/decode/nv_reduce_output_fp32_qk_ab.py --mode ab`. Every
GPU arm ran as a fresh process (JIT capture / allocator state cannot leak);
the harness self-manages `/tmp/gpu-bench.lock` and was never wrapped in an
outer flock.

## Gate sequence (fixed order, all PASS)

1. **NV render smoke (Xid 31 class)**: candidate survives on sm_120
   (`survive=True`, 594 compiled programs) with the fused bodies present in
   the compiled set: `reduce_output_rmsnorm_1_4096`,
   `reduce_output_rmsnorm_32_128`, `reduce_output_rmsnorm_8_128`; the
   per-block promotion gate is True on all 36 blocks.
2. **Exact full-logit gate**: fp32 stacked-row SHA-256 identical control vs
   candidate (`4ff1b9eaf5308a4e0d44d8ab24b9ca23bc2831c430654b0f6f0724cc0ff84e69`
   both arms, shape `[32,1,151936]`, `tokens_equal=True`,
   `logits_sha256_equal=True`, `gate_pass=True`). Production-harness
   reference at this HEAD (count=8 window) still reproduces the committed
   token stream `9e6664fd...` and full-logit `6ec7227e...` (prelude token
   13876), confirming the A/B ran on the same exact-output substrate as the
   prior booking.
3. **Census contract**: candidate 91 fused bodies (C6 19 + q 36 + k 36)
   matching the committed reference; control 0 bodies; q/k reduce drops
   36 each, q/k epilogue drops 72 each; norms kernels 328 -> 74; honest net
   program delta **-163** (594 vs 757 kernels) with zero weight
   materializations (side-effect ledger shows only the expected
   reduce/epilogue name swaps, no `*_weight_store*` additions);
   `gate_pass=True`, no fail-closed entries.
4. **Reverse wall bracket** control / candidate / control (fresh processes):
   candidate median 5.1955 ms vs control A 5.2477 ms and control B 5.2491 ms
   -> **+52.20 / +53.55 us/token** (bracket median 5.2484 ms -> +52.87 us),
   clearing the +50 us bar against BOTH bracketing controls; token stream
   hash identical across all 15 samples (5 reps x 3 arms).

## Wall bracket (count 32, reps 5, fresh processes, shared lock)

| arm | median ms/token | tok/s |
| --- | ---: | ---: |
| control A | 5.2477391875 | 190.56 |
| candidate | 5.19554365625 | 192.47 |
| control B | 5.2490943125 | 190.51 |
| bracket median | 5.24841675 | 190.53 |

Candidate minus control A = **+52.20 us**, minus control B = **+53.55 us**,
minus bracket median = **+52.87 us**. Bar: +50 us/token against both
bracketing controls -> **PROMOTED** (`promoted=True`,
`all_token_hashes_equal=True`, `settled_continuous=False`). tok/s
translation: 192.47 vs 190.53 = **+1.94 tok/s**.

## Census swap table (candidate vs baseline-context control)

| role / family | control | candidate | delta |
| --- | ---: | ---: | ---: |
| `reduce_output_rmsnorm_32_128` (q) | 0 | 36 | +36 |
| `reduce_output_rmsnorm_8_128` (k) | 0 | 36 | +36 |
| `reduce_output_rmsnorm_1_4096` (FFN-down) | 0 | 19 | +19 |
| q_norm_reduce | 36 | 0 | -36 |
| q_norm_epilogue | 72 | 0 | -72 |
| k_norm_reduce | 36 | 0 | -36 |
| k_norm_epilogue | 72 | 0 | -72 |
| rmsnorm_reduce | 56 | 37 | -19 |
| rmsnorm_epilogue | 55 | 37 | -18 |
| final_rmsnorm_epilogue | 1 | 0 | -1 |
| norms kernels total | 328 | 74 | -254 |
| all kernels (honest net) | 757 | 594 | **-163** |

Weight materializations: zero. Non-norms family shifts reported with exact
program names (`callify_redirect_side_effects`); nothing hidden.

## Decision and scope limits

P2 books the q/k site AND the FFN-down 1_4096 site as the package: the
harness's candidate condition carries all three body families, so the wall
bracket measures the combined route, and the combined delta clears the +50
us bar against both bracketing controls without a waiver (the per-site
arithmetic in the scope priced the package at ~+235 us at 0.6 mapping; the
measured +52.9 us is the realized wall delta of the already-promoted
production route). The CPU-arm blocker is resolved: the q/k site
(`PERMUTE(CAST)` carrier without precompiled-output identity on CPU) is
admissible on NV because the NV substrate supplies that identity, and the
census confirms it.

`promoted_targets` in `decode-reduce-output-rmsnorm-route-policy.json` is
NOT changed: it already lists NV sm_120 from the prior fp32 q/k booking, and
the file is byte-identical to HEAD after the A/B (the harness requires the
control arm to see the closed policy; the policy was temporarily closed for
the run and restored, so this record makes no policy edit).

Raw artifacts: `docs/task_workflow/evidence/nv-reduce-output-site-absorption-p2-ab-20260812.json`
(ab record), `nv-reduce-output-site-absorption-p2-ref-stream-verify-20260813.json`
and `nv-reduce-output-site-absorption-p2-ref-logits-verify-20260813.json`
(production-reference reproduction at this HEAD, count=8 window). The
per-arm children and timing rows live under the harness `--out` directory
(`/tmp/nv-p2/`) and are not committed.
