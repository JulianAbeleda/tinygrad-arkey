# NV decode launch-hiding P2 verdict - fresh-HEAD decision reruns (hard stop)

Date: 2026-08-12
Branch: `nvidia-bringup-20260731` (HEAD `713b2a020` at P1; P2.4 replay-busy
record `337281427`; this record follows)
Status: **P2 complete, hard stop.** All three decision tools rerun on the
fresh duration-bearing DAG and the compiled-descriptor census at HEAD; every
arm fails the +50 us gate. No GPU arm is authorized by the reopen scope
(`nv-decode-overlap-reopen-resource-join-scope-20260812.md`, section 4).

## 1. Why this rerun exists

The 08-05 exhaustive scope closed every in-graph co-schedule arm on the old
topology with a ceiling of 17.952 us; its one open CPU step, the
resource-complementarity join, failed closed because the tinygrad census had
no compiled resource tuples, and the authority DAG it cited was `/tmp` scratch
lost in the restart. P1 (`713b2a020`) rebuilt both: a duration-bearing,
kernel-named DAG at HEAD (596 nodes / 110544 edges / 5 groups / 0 unknown,
digest `b7af576d...`) plus a complete 596-row compiled-descriptor census from
NVProgram objects. This record reruns the scan, the join, and the wait-adjusted
cut forecast on those artifacts.

## 2. Audit

- Fresh DAG: `docs/task_workflow/evidence/nv-dag-duration-head-20260812.json`
  (596 nodes; q4k 180, q6k 37, flash 72, E 105, r_ 94, reduce_output_rmsnorm
  91, rmsnorm_q8_1_llama_provider 17; serialized sum 5493.27 us; baseline
  critical path 4842.38 us).
- Fresh census: `docs/task_workflow/evidence/nv-descriptor-census-head-20260812.json`
  (596 complete tuples; regs 8-96, static smem 1024-10240, local mem 576
  flat, 36 unique binaries; raw 50 MB artifact kept out of repo by design).
- Fresh replay busy (P2.4, `337281427`): 95.6% busy, span 4973.0 us, host gap
  230.1 us vs production wall 5203.1 us, replay factor 0.917 replacing the
  pessimistic 0.883 estimate. The 1.129 ms/token gap is kernel-work-bound,
  not host-bound.

## 3. Arithmetic (all three tools, same DAG + census)

| tool | result | gate | verdict |
| --- | ---: | ---: | --- |
| co-schedule scan (same-group) | ceiling 33.36 us, greedy 32.49 us | +50 us exact CP recovery | CPU_NO_GO |
| co-schedule scan (cross-group ceiling) | fusion-only ceiling 763.46 us is a population bound, not a co-schedule arm | +50 us | not an arm |
| resource join | 150/150 dependency-independent pairs SM-co-resident (CTA bound > 0), best pair recovery 3.97 us | positive bound AND +50 us | INCONCLUSIVE_FAIL_CLOSED |
| wait-adjusted cut forecast | Q -10.474 us, K +25.705 us costed at 4.743 us/wait (recalibrated on this DAG) | +50 us costed | CPU_NO_GO |

Tool adaptations (post-split DAG shape, minimal and fail-closed):

- Scan `classify` now accepts `rmsnorm_*` / `reduce_output_*` as support
  (non-quant, non-flash kernels eligible to hide behind quant/flash hosts);
  the metadata guard is unchanged, so a future quant rmsnorm family cannot be
  misclassified.
- Cut analyzer gained a layout detector: legacy pre-split 9-node blocks vs
  post-split q4k/q6k/reduce_output/rmsnorm family windows between consecutive
  `flash_block_tiled_` nodes; any unrecognized layout raises (fails closed).
- Join now consumes the fresh descriptor census (keyed by call index, exact
  set via `attach_compiled_descriptors`) and computes a complementary CTA
  residency bound per dependency-independent pair under RTX 5090 per-SM
  budgets (65536 regs, 1536 threads, 228 KB smem); fail-closed seam
  preserved.

## 4. Verdict

Every candidate mechanism on this row clears no gate at HEAD:

1. Co-schedule scan ceiling 33.36 us < 50 us.
2. Resource join best complementary pair 3.97 us < 50 us (bound positive,
   recovery too small).
3. Wait-adjusted cuts Q -10.474 us / K +25.705 us < 50 us at the
   recalibrated 4.743 us/wait.
4. Replay busy 95.6% and host gap 230.1 us show the host side is not the
   lever; the gap is in kernel work (the 652 us class table and the
   reduce-output / M1 / vocab rows already scoped for execution).

Therefore the reopen scope's hard stop holds: no GPU arm for launch hiding.
The row stays quarantined until a future occurrence-pinned candidate clears
+50 us at the calibrated wait cost on a fresh DAG. Work proceeds on the
scoped execution rows (reduce-output per-site absorption ~201.3 tok/s, M1
norm epilogue ~193, vocab aux ~194.3) rather than on overlap recovery.

## 5. Gates

- `python3 sz.py` PASS before commit (42845/100000).
- Unit: scan/join/cut/forecast/route-b3/duration-attach suites 56 passed.
- Evidence JSONs persisted:
  - `docs/task_workflow/evidence/nv-co-schedule-scan-head-20260812.json`
  - `docs/task_workflow/evidence/nv-descriptor-census-head-20260812.json`
  - `docs/task_workflow/evidence/nv-overlap-resource-join-head-20260812.json`
  - `docs/task_workflow/evidence/nv-wait-adjusted-forecast-head-20260812.json`

## 6. Evidence

- P1 artifacts: `nv-dag-duration-head-20260812.json` (digest `b7af576d...`).
- P2.4: `nv-replay-busy-head-20260812.json` + `-rep.json` (95.6% busy,
  host gap 230.1 us).
- Raw capture: `/tmp/nv_capture_head_20260812_fox.json` (50 MB, sha256
  `ee5ac3d8...`, referenced by the compact census, kept out of git).
