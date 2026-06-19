# docs/ — map

Single navigation source-of-truth for this fork's docs. The AMD-decode work produced a long
chronological probe log; the **verdicts are folded into the syntheses below** — start there, treat
the dated `*-plan/-result/-probe.md` files as provenance, not current state.

## ⭐ Start here (canonical, post-bank)

- **`prefill-RECONCILIATION-source-of-truth-20260619.md`** — ⭐PREFILL SOURCE-OF-TRUTH. Settles the contradictory
  prefill results under one controlled interleaved matrix. Verdict: concrete-KV = 1.24x byte-identical (shipped,
  ~47% llama); **+Tensile (external .co, research) = 1.76x over concrete = ~86% llama, REPRODUCED** (the old
  "4770/1.76x" is REAL; the "0.997x no-advantage" runs were a high-WMMA-clock outlier — tinygrad WMMA prefill is
  clock-volatile 1449-2675, Tensile is clock-stable ~2640). Supersedes prefill-matmul-RECONCILED / tensile-land /
  transpose-free "Tensile-no-advantage". Artifact: `artifacts/prefill-reconciliation-matrix-20260619.json`.

- **`amd-decode-banked-20260616.md`** — THE entry point. Final decode state (~64 tok/s / 63% llama),
  the full lever map (shipped / tapped / refuted / gated), the machine-search system, resume pointers.
- `amd-decode-beyond-llama-roadmap.md` — the lever map with live statuses (parity vs beyond-llama).
- `gpu-performance-first-principles.md` — **canonical** bytes/math/overhead + roofline reference;
  diagnose the bucket BEFORE optimizing.
- **`../bench/README.md`** — the benchmark results index: every current number, its artifact, and the
  exact command to reproduce it. **Includes "Which harness for decode tok/s — READ FIRST"** (use the clean
  `model.generate`-path CLI/W==D harnesses; the flash auto-bench's ~54 is contaminated, not a tok/s number).
- `qk-decode-banked-reproduce-20260618.md` — banked decode line reproduced on HEAD (68.2/66.4/60.7, W==D,
  host-sync 0%, whole stack default-on) + the harness lesson.
- `amd-decode-capstone.md` — the decode ledger (23 → ~64 tok/s arc).
- `amd-decode-arc-synthesis.md` — synthesis through the primitive lens.

## 8B decode-attention + MMVQ frontier (2026-06-17 → 18) — latest state

The work after the decode bank. Closeouts/results are canonical; the many dated `qk-*` arc docs are provenance.

- **`what-makes-a-performance-primitive-efficient-20260618.md` — READ THIS FIRST for the performance-primitive model and gap.**
  Consolidated source of truth for what makes a performance primitive efficient, using llama.cpp vs tinygrad as the
  case study: decode, lm_head, MMVQ, attention, spec, prefill, machine-search lessons, and every remaining path
  marked shipped/refuted/deferred/open.
- `performance-primitive-external-research-audit-20260619.md` — second-round external research audit across
  arXiv/OpenReview/ChinaXiv. Cites each paper/source, records the claim, checks whether it is true/applicable to this
  tinygrad gfx1100 project, and maps it to current or future primitive rows.
- `primitive-local-observability-search-scope-20260619.md` — scope for building primitive-local tooling instead of a
  generic profiler: read-only ledger first, then schema validators, runner wrappers, deterministic failure
  classifiers, guided search memory, and optional rocprof/SQTT counter plugins.
- `primitive-local-observability-search-result-20260619.md` — **executed PLO-1..PLO-6.** Adds
  `extra/qk_primitive_ledger.py`, a read-only primitive ledger/validator/classifier/search-memory/trace-plugin
  inventory that reconstructs current verdicts from existing artifacts without hardware execution.
- `primitive-local-observability-audit-20260619.md` — replay audit over the primitive ledger, including the TPE-7a
  rebindable-node artifact. Confirms graph-protocol prerequisite PASS while keeping in-model capture and artifact
  policy as remaining gates.
- `primitive-ledger-analysis-audit-20260619.md` — uses the primitive ledger for the intended analysis pass: decode is
  q8/MMVQ lifecycle-limited; prefill is graph/artifact-boundary-limited; broad kernel search is not supported by the
  current evidence.
- `primitive-lifecycle-search-scope-20260619.md` — scope + executed PLS-1..PLS-4 ledger for lifecycle-level search:
  producer placement, activation/weight format, consumer primitive, routing boundary, quality gate, fallback, runner
  bindings, policy candidates, generator, and refutation memory. Adds `extra/qk_lifecycle_search.py` and
  `bench/qk-lifecycle-search/*`; current frontier is q8 decode artifact/native transfer and Tensile prefill
  artifact/native transfer, not broad kernel search.
- `primitive-coverage-gap-scope-20260619.md` — coverage audit scope after the latest decode/prefill integration
  learning. Names rows missing from the map, not necessarily missing implementations: decode B2 runtime/cache identity,
  decode MMVQ artifact/import, prefill transpose-free layout, long-context KV/attention, serving, alternative quant,
  CUDA portability, and tooling visibility.
- `primitive-coverage-map-20260619.md` — **executed PCG-0.** Consolidates the current row map into
  `bench/qk-primitive-coverage/rows.json` with 12 validated rows. Key update: Tensile prefill is refuted as an e2e
  speed route after transpose-free `0.997x`; prefill now points to non-matmul overhead, while decode B2 is closed and
  the remaining decode choice is large project-level MMVQ contract work or the small q8 research flag.
- `decode-large-small-paths-scope-20260619.md` — split decode closeout into the large parity-scale path and small
  research path. Large path: MMVQ contract preservation/source import, `~1.187x` measured target but project-level.
  Small path: q8 FFN artifact route, `1.051-1.063x` and dNLL `+0.002887`, default-off research flag.
- `decode-mmvq-artifact-import-discovery-result-20260619.md` — large-path L1 inventory. llama.cpp has MMVQ source and
  build objects, but no standalone Tensile-like HCQ code-object family; direct TPE-style decode extraction is closed as
  a bounded route.
- `decode-mmvq-large-project-scope-20260619.md` — funded large decode MMVQ project scope. Splits the work into
  source/object import first and native renderer/scheduler transfer second, with P0-P8 gates. Target remains
  `44% -> 54%` in-model HBM over the weight-GEMV bucket, about `1.187x` decode.
- `decode-mmvq-large-project-p0-contract-inventory-result-20260619.md` — **executed P0.** The llama.cpp gfx1100
  `mmvq.cu` object contains `22` Q4_K/Q6_K candidate functions and `22` `.kd` descriptors with `144` byte kernargs.
  Next gate is P1: named-descriptor HCQ load smoke, no HIP runtime, no launch yet.
- `decode-mmvq-large-project-p1-loader-smoke-result-20260619.md` — **executed P1.** Selected Q4_K and Q6_K low-VGPR
  llama descriptors load through tinygrad HCQ (`0x74840`, `0x74e40`), no unsupported relocations, no HIP runtime, no
  kernel launch. Next gate is P2: capture real llama kernargs/grid/local in a separate HIP-only process.
- `decode-mmvq-large-project-p2-kernarg-capture-result-20260619.md` — **executed P2.** Versioned LD_PRELOAD capture
  over llama-bench records `7` real Q4_K/Q6_K `mul_mat_vec_q` launches and reconstructs the `144` byte kernargs.
- `decode-mmvq-large-project-p3-p4-q4-result-20260619.md` — **executed P3/P4 for Q4_K.** Imported llama Q4_K MMVQ is
  correct on `blk.0.attn_output.weight` (`max_abs 1.43e-6`) and reaches `903.9 GB/s` / `94.2%` HBM with single-submit
  HCQ timing. Next gate is P5: q8_1 producer/reuse plus one-role in-model routing.
- `decode-mmvq-large-project-p5-p6-result-20260619.md` — **executed P5/P6 for Q4_K.** Real activation -> q8 producer
  -> imported Q4 consumer is correct and clears the lifecycle device gate (`50.8%` HBM-equivalent vs current
  `attn_q/o` ~`29%`), and the same Q4 template generalizes to `ffn_gate/up`. Next gate is graph-safe Q4 routing; Q6
  remains a parallel coverage track.
- `decode-mmvq-large-project-p7a-graph-route-result-20260619.md` — **attempted P7a.** Runtime-cache graph adapter was
  built, but TinyJit replay faults even with persistent side buffers. Imported Q4 remains valid in eager HCQ; graph use
  now requires first-class raw-kernarg rebind support or native lowering. **Superseded by P7b.**
- `decode-mmvq-large-project-p7b-raw-kernarg-rebind-scope-20260619.md` — P7b scope for making imported raw kernargs
  graph-safe: raw template + declared pointer patches, staged through CPU-side args-buffer proof, eager parity, graph
  micro-smoke, real activation graph proof, then one-block route decision.
- `decode-mmvq-large-project-p7b-raw-kernarg-rebind-result-20260619.md` — **executed P7b.** Raw-kernarg rebind support
  passes: offsets `0/8/56` bind q4/q8/out VAs, eager parity is `max_abs 1.43e-6`, and TinyJit replay of real
  `blk.0.attn_output` activation is stable for `5/5` calls with zero diff vs eager.
- `decode-mmvq-large-project-p7c-one-role-route-scope-20260619.md` — P7c scope for moving the imported Q4 route from
  probe-only to one real model role behind `DECODE_MMVQ_IMPORT_Q4=1`, with persistent q8/out side buffers and no default
  behavior change.
- `decode-mmvq-large-project-p7c-one-role-route-result-20260619.md` — **executed P7c.** `blk.0.attn_output` routes
  through the imported Q4_K path in `model.py`; smoke output shape is `[1,1,4096]`, routed blocks `[0]`. Next gate is
  clock-controlled one-role timing, then q8 quality/dNLL.
- `decode-mmvq-large-project-p7d-one-role-timing-scope-20260619.md` — P7d scope for timing the imported route on the
  true pre-`attn_output` activation with same-process interleaved TinyJit A/B.
- `decode-mmvq-large-project-p7d-one-role-timing-result-20260619.md` — **executed P7d.** Imported route is correct,
  replay-stable, and model-branch reachable, but slower for `blk.0.attn_output`: `0.1396ms` vs baseline `0.1064ms`
  (`0.763x`). Do not expand `attn_output`; next valid diagnostic is `ffn_gate/up` q8 amortization.
- `decode-mmvq-large-project-p7e-gateup-amortization-scope-20260619.md` — P7e scope for the fresh favorable Q4 case:
  `ffn_gate/up`, `12288` rows each, one q8 producer shared by two imported Q4 consumers.
- `decode-mmvq-large-project-p7e-gateup-amortization-result-20260619.md` — **executed P7e.** Imported route remains
  replay-stable but loses for `ffn_gate/up`: `0.2264ms` vs baseline `0.1685ms` (`0.744x`). Imported Q4 decode route is
  closed as a local timing win; remaining value is oracle/native-transfer evidence, not model-wide artifact routing.
- `decode-mmvq-large-project-p8-fused-lifecycle-scope-20260619.md` — P8 scope for the full 1-4 sequence after P7e:
  lower-bound model, current native expressibility, handwritten prototype evidence, and final decision.
- `decode-mmvq-large-project-p8-fused-lifecycle-result-20260619.md` — **executed P8.** Fused q8+gate/up is
  build-worthy by lower bound (`56.83us` vs `153.22us` gate); current native COMGR/DSL attempts fail; the hipcc/LLD
  artifact route clears local gate (`115.24us`, `1.46x`) and graph route passes. Decision: artifact research flag or
  project-level native renderer transfer.
- `decode-q8-two-lane-scope-20260619.md` — post-P8 two-lane closeout scope: harden the default-off q8 artifact
  research flag and separately define the native renderer/scheduler transfer start gate.
- `decode-q8-two-lane-result-20260619.md` — **executed two-lane closeout.** Artifact lane is ready as
  `Q8_FFN_HANDWRITTEN=1` research flag (`1.051-1.063x`, dNLL `+0.002887`, no HIP runtime); native lane is project-level,
  with no bounded `>=30us` q8-specific patch identified.
- `decode-q8-both-lanes-execution-scope-20260619.md` — execution scope for the "do both" decision: accept the q8
  artifact dependency for research-flag use and charter the native AMD scheduler project separately.
- `decode-q8-both-lanes-execution-result-20260619.md` — **executed both lanes.** `Q8_FFN_HANDWRITTEN=1` is accepted as
  research-only/default-off; native transfer is chartered as N0-N4 project-level backend work with a `>=30us` start gate.
- `decode-next12-execution-result-20260619.md` — **executed high-level decode next steps 1-2.** The q8 artifact route is
  the completed research answer; native scheduler work is active as a project charter with N0 complete and N1 now closed
  with no bounded N2 start.
- `decode-n1-attribution-scope-20260619.md` / `decode-n1-attribution-result-20260619.md` — native q8 scheduler
  attribution. Full oracle gap is `73.109us`, but largest bounded attribution is `14.087us`; SQTT capture works but
  local RDNA3 HCQ decode fails, so remaining scheduler/resource movement is project-level tooling/backend work.
- `amd-scheduler-tooling-backend-project-scope-20260619.md` — concrete scope for that project-level fork. Track T
  funds RDNA3 HCQ attribution tooling first; Track B funds the reusable AMD scheduler/resource backend only after a
  measurable feature or explicit backend investment decision.
- `amd-scheduler-tooling-backend-t0t4-b0-result-20260619.md` — first combined execution. T0/T2/T3 pass, SQTT replay is
  structurally decodable but maps only `S_ENDPGM` and no q8 body instructions, so T4 finds no bounded feature; B0 oracle
  suite passes with q8 and Tensile targets.
- `amd-scheduler-tooling-t1-body-mapping-proof-20260619.md` — focused T1 proof. Sweeps baseline, `SQTT_MODE=3`,
  `SQTT_TTRACE_EXEC=1`, and both; all capture q8 wave lifecycle packets but `0` raw body instruction packets, so the
  local register-knob fix is refuted.
- `amd-scheduler-tooling-t1b-att-aqlprofile-result-20260619.md` — **executed both requested T1b paths.** ROCprofiler SDK
  and AQLprofile are installed under `/opt/rocm-7.2.4`; external `rocprofv3 --att` remains blocked by the missing/unstable
  ATT decoder path, while AQLprofile command recovery passes. Transplanting recovered `MASK/TOKEN/CTRL` values into HCQ
  changes trace volume but still yields `0` body instruction packets, so the missing piece is a broader command-sequence
  or ROCprofiler-service detail, not a simple register value.
- `amd-scheduler-tooling-t1c-att-decoder-repair-result-20260619.md` — **executed local ATT decoder repair.** Inspected
  available ROCm packages and tested decoder aliases (`librocprofiler-sdk.so`, legacy `libatt_plugin.so`); no candidate
  produced `rocprofv3 --att` output. External ATT is a ROCm packaging/toolchain blocker until a real
  `librocprof-trace-decoder.so` is installed or built.
- `amd-att-decoder-blocker-scope-20260619.md` — concrete reopen scope for the known ATT blocker: binary decoder
  acquisition, source build from `ROCm/rocm-systems`, known-good ROCm environment, and why an ABI shim is not the first
  path. Gates require `rocprofv3 --att` payloads before returning to tinygrad HCQ SQTT body attribution.
- `amd-att-decoder-solution-result-20260619.md` — **executed D0/D1 and solved the external ATT oracle blocker.**
  ROCprof Trace Decoder `0.1.6` binary passes once HIP controls are compiled/linked coherently against ROCm 7.2 instead
  of Ubuntu HIP 5.7. `rocprofv3 --att` now emits `.att`, decoded UI files, wave JSON, and result JSON for the HIP control.
- `amd-sqtt-oracle-hcq-diff-scope-20260619.md` — next scoped tooling phase after the decoder pass: archive the working
  ROCprofiler ATT oracle, reproduce tinygrad HCQ lifecycle-only SQTT, diff setup/order/targeting, try one env-gated
  command-sequence patch if a bounded delta is found, and close with body-attribution pass/kill.
- `amd-sqtt-oracle-hcq-diff-result-20260619.md` — **executed O0-O5; verdict `KILL_PATCH_NO_BODY`.** ROCprofiler ATT is
  valid and instruction-rich (`110446` decoded wave instruction records), but the only bounded HCQ patch
  (`SQTT_ORACLE_TARGET_CU=1`, with/without AQLprofile raw regs) still produced zero body packets. Track T is closed as a
  small primitive-observability patch; reopening requires broader ROCprofiler command-service integration.
- `amd-rocprofiler-thread-trace-audit-result-20260619.md` — source audit of what that broader integration actually
  means. Verdict: ROCprofiler ATT depends on a profiled HSA queue + AQLprofile-generated vendor AQL packet lifecycle
  (`hsa_amd_profiling_set_profiler_enabled`, profiler-active queue packet, trace-control buffer, code-object markers),
  not one missing SQTT register. Reopen only as AQLprofile packet import/replay or native profiled-HCQ work.
- `amd-rocprofiler-reopen-tracks-scope-result-20260619.md` — scoped and executed the first phase for all three reopen
  options. Verdict: split tooling is the default usable path; AQLprofile packet replay is the only bounded reopen;
  native profiled-HCQ is project-level and should not start from another register sweep.
- `amd-rocprofiler-r1p1-aqlprofile-replay-result-20260619.md` — **executed Track 1 R1-P1.** Forcing tinygrad
  `AMD_AQL=1` is stable but still lifecycle-only (`0` body packets). AQLprofile has nonzero gfx1100 command material,
  but the old command-buffer output is not a direct HCQ replay blob. Remaining reopen requires a v2 AQLprofile packet
  exporter with tinygrad-owned trace/control buffers, or native profiled-HCQ.
- `amd-rocprofiler-r1p2-v2-exporter-scope-20260619.md` — scope for that remaining bounded reopen: v2
  `aqlprofile_att_create_packets` exporter, allocation callback table, tinygrad-mappable buffers, one HCQ AQL dispatch
  replay, and strict body-packet pass/kill gates before any native profiled-HCQ work.
- `amd-rocprofiler-r1p2-v2-exporter-result-20260619.md` — **executed R1-P2 P0.** Corrected the local v2 ABI:
  `aqlprofile_att_profile_t.agent` is `hsa_agent_t`, not `aqlprofile_agent_handle_t`. With the real HSA GPU agent,
  `aqlprofile_att_create_packets` passes for all swept ATT profiles, returns nonzero start/stop packets, and exposes the
  allocation callback table. Next boundary is P1/P2: bind those buffers to tinygrad-submittable GPU VAs and replay around
  one HCQ dispatch.
- `amd-rocprofiler-r1p2-hcq-replay-result-20260619.md` — **executed R1-P2 P1/P2; verdict
  `PASS_BODY_ATTRIBUTION`.** A separate HSA helper exports v2 AQLprofile vendor packets, tinygrad allocates HCQ-owned
  control/command/trace buffers, and the probe patches both raw 64-bit VAs and PM4 `VA >> 12` page-address fields. Full
  `start -> tinygrad body kernel -> stop` replay syncs and yields decodable SQTT body packets (`98,269` body-like
  packets), proving ROCprofiler ATT can be imported into tinygrad HCQ without HIP/HSA in the tinygrad process.
- `amd-att-primitive-attribution-scope-20260619.md` — next scope after ATT replay passed: use imported ATT on real
  tinygrad primitives, first decode MMVQ contract attribution (`76%` standalone HBM -> `44%` in-model) and then prefill
  non-matmul residual attribution. Strictly observability-only; timing authority remains clock-controlled A/B + PMC.
- `amd-att-primitive-attribution-result-20260619.md` — **executed the ATT primitive atlas; verdict
  `PASS_ATT_PRIMITIVE_ATTRIBUTION`, interpretation `ATT_USABLE_NOT_DECISIVE_FOR_INMODEL_GAP`.** ATT now
  body-attributes native tinygrad Q4_K coop (`168,693` body-like packets), imported llama Q4_K MMVQ (`163,942`), and
  pp512 SDPA (`135,442`). This clears the tooling blocker but does not change timing conclusions; next decode use is a
  role-joined in-model ATT pass.
- `amd-att-inmodel-role-join-scope-20260619.md`, `amd-att-inmodel-role-join-result-20260619.md` — **executed first
  role-joined in-model ATT pass; verdict `PASS_INMODEL_ROLE_JOIN_NATIVE_Q4K_COOP`.** `blk.0.attn_output` launches the
  intended `q4k_coop_partial_4096_4096` plus stage-2 reduce/glue in-model, with `16,137` body-like ATT packets. This
  closes runtime/cache identity for that Q4_K role; next ATT target, if any, is higher-share Q6_K `ffn_down`/`lm_head`.
- `decode-standalone-retention-staged-attack-scope-20260619.md` — staged attempt to recover more of tinygrad's
  `~76%` standalone decode MMVQ efficiency in-model. Starts with Q6_K role-joined ATT (`ffn_down`, then `lm_head`),
  then reduce/glue Amdahl, one direct-output/reduce-fusion proof if gated, q8 lifecycle only if still justified, and
  finally project-level scheduler/resource work if all bounded routes fail.
- `decode-standalone-retention-stage1-q6-role-join-result-20260619.md` — **executed Stage 1 Q6_K role join via
  explicit `q6_surface_fallback` after full model load hit a 4.68GB allocation failure.** Both Q6 surfaces launch the
  intended native coop programs (`q6k_coop_partial_4096_12288`, `q6k_coop_partial_151936_4096`) plus reduce/glue, with
  ATT body attribution. No bounded Q6 fallback/wiring fix found; proceed to reduce/glue Amdahl ledger.
- `decode-complete-tooling-scope-20260619.md` — complete tooling scope for the remaining decode lifecycle question:
  join role identity, ATT body attribution, lifecycle accounting, timing authority, reduce/glue Amdahl, and llama
  comparison into one atlas before funding any direct-output/reduce-fusion or scheduler/resource build.
- `decode-complete-tooling-result-20260619.md` — **executed DCT-0..DCT-7.** Adds
  `extra/qk_decode_complete_tooling.py` and `bench/qk-decode-complete-tooling/*`. Verdict:
  `COMPLETE_TOOLING_PASS_WITH_EXPLICIT_GAPS`; reduce/glue is visible but does not clear the build gate, Q6 surface
  equivalence is accepted for visibility not timing, and ATT remains body evidence rather than timing authority.
- `decode-native-mmvq-scheduler-renderer-full-scope-20260619.md` — full scope for the remaining dependency-free
  native decode route after the tooling atlas: a project-level AMD scheduler/renderer path to preserve the MMVQ
  lifecycle contract in-model. Defines NSR-0..NSR-8, start criteria, gates, kill conditions, expected potential, and
  the boundary between q8 research-flag hardening and true native compiler ownership.
- `decode-q8-research-route-hardening-result-20260619.md` — small-path hardening pass. Consolidates W==D, dNLL,
  artifact hashes, fixed-launch boundary, and policy gate; verdict `PASS_RESEARCH_HARDENED_EXISTING_EVIDENCE`.
- `decode-fused-mmvq-integration-next-path-scope-20260619.md` — next base-decode path after the PMU convergence:
  tinygrad's standalone GEMV is stronger than llama's, but in-model weight-GEMV falls to `~44%` vs llama `~54%`.
  Scopes activation/Q8 reuse plus occupancy/launch-shape preservation, starting with measurement-only FMI-1/FMI-2.
- `decode-fused-mmvq-integration-fmi1-fmi2-result-20260619.md` — **executed FMI-1/FMI-2.** The in-model GEMV loss
  atlas passes (`44% -> 54%` projects `1.187x` if recovered across the weight-GEMV bucket), and llama/tinygrad launch
  contract diff passes. Decision: build Track B first, the byte-identical occupancy/launch-shape route.
- `decode-fused-mmvq-integration-fmi4-b1-result-20260619.md` — **executed FMI-4 B1.** Existing env launch-shape knobs
  (`Q4K_COOP_RT`, `Q6K_COOP_RT`, coop on/off) do not move a high-share role by `>=10%`; B1 is closed. Track B remains
  live only as runtime/cache identity or renderer/scheduler work.
- `decode-integration-diagnostic-result-20260619.md` — prefill-style decode localization. Verdict:
  **no single transpose-like tax**; Q4_K stage2 reduce is real but insufficient, q8 lifecycle is capped/lossy, env knobs
  fail, and the remaining large gap is MMVQ in-model contract preservation.
- `decode-fused-mmvq-integration-b2-runtime-cache-result-20260619.md` — **executed PCG-1/FMI-4 B2.** Runtime/cache
  identity closes: in-model decode and direct same-process role calls use the same program/launch identities for
  `attn_q/o`, `ffn_gate/up`, `ffn_down`, `lm_head`, and `attn_k/v`. The hidden wiring-bug route is closed.
- `primitive-pmu-observability-scope-20260619.md` — scope for using installed ROCm profiler tooling as the PMU oracle
  and building only the tinygrad primitive-local attribution layer needed around HCQ.
- `primitive-pmu-observability-result-20260619.md` — PMU-1..PMU-3 result: ROCm PMU works on HIP controls, but tinygrad
  HCQ is invisible to `rocprofv3` in the smoke; redirects to a tinygrad-native HCQ attribution adapter.
- `primitive-hcq-attribution-scope-20260619.md` — PMU-4 scope: tinygrad-native HCQ attribution for eager launches and
  graphs, producing Level-3 runtime/graph evidence without pretending to have PMU counters.
- `primitive-hcq-attribution-result-20260619.md` — PMU-4a..c result: probe-local attribution captures eager HCQ
  launches, HCQGraph construction/replay, and a Tensile runtime row; classifies `rocprof_hcq_visibility_gap` +
  `graph_rebind_ok`.
- `amd-schedule-codegen-exhaustion-scope-20260619.md` — cross-primitive scope for exhausting AMD scheduler/codegen by
  oracle, not as an open-ended compiler ambition. Uses q8 decode and Tensile prefill as authority cases.
- `amd-schedule-codegen-exhaustion-result-20260619.md` — **executed SCE-0/SCE-1.** Builds
  `bench/amd-schedule-codegen-exhaustion/oracle_matrix.json`: 7 feature rows are project-level, 1 artifact-only,
  1 bounded graph/rebind row, 1 tooling-blocked, 1 not worth owning, 1 already expressible. Native q8/prefill
  schedule generation is exhausted as a bounded primitive; remaining native work is a reusable AMD backend project.
- `prefill-address-lowering-renderer-arc-plan-20260619.md` — dependency-free prefill renderer arc. CG-W1.5 validates
  the real warmstarted in-model ffn matmul uses WMMA but is ALU-overhead-bound; CG-W2/2b then refute kernel-level
  coalesced/wide-copy fixes. The only remaining no-deps lever is renderer/opt-level fp16 load vectorization or
  hand-asm WMMA, both project-level.
- `route-a-a3-lds-multiwave-scope-20260619.md` — continuation scope for dependency-free RDNA3 WMMA hand-asm:
  LDS-staged, multi-wave GEMM to chase LLVM/Tensile after single-wave A2 stayed below LLVM.
- `route-a-a3-lds-multiwave-result-20260619.md` — **executed A3 P0/P1 gates.** P0 LDS tile smoke passes
  (RMSE `0.000209`); P1 multi-wave LDS GEMM faults even at `128^3`, so the next valid step is a smaller
  store/load-only address-mapping debug probe before any P2 pipeline/tuning.
- `prefill-tensile-research-measurement-scope-20260619.md` — complete Option A execution scope for Claude: finish the
  bounded JIT-dim step, route extracted Tensile prefill behind `PREFILL_TENSILE_GEMM=1`, and measure pp/dNLL as
  research-only evidence.
- `prefill-tensile-tpe7a-rebindable-node-result-20260619.md` — TPE-7a result: one extracted Tensile kernel object
  can be rebound to current buffers through graph-style kernarg filling; correctness/protocol proof only.
- **`performance-frontier-exhaustion-20260619.md` — latest exhaustion checkpoint.** Bounded decode primitives are
  exhausted; q8/RMSNorm is codegen-deferred; hand-LDS WMMA is refuted; external BLAS ceiling is measured; the bounded
  no-deps prefill WMMA sweep is refuted; EBT-1 kills the HIP-runtime bridge; the only material prefill route left is
  Tensile primitive extraction through HCQ or a codegen/Tensile-class rewrite.
- `qk-decode-per-role-delta-audit-20260618.md` — the quantitative per-role decode gap table (traffic/%peak/time-share/
  Amdahl/status); summed ceilings ~+27–30% ≈ the whole 1.47× llama gap, all behind one q8/full-MMVQ wall.
- `qk-machine-search-primitive-rows-20260618.md` — current machine-search rows (live + closed); supersedes the
  06-17 rows doc. Live/deferred: q8 side-channel, ffn coop sub-gate, attention residual audit, LDS flash-prefill,
  external/raw-HIP boundary/control; closed: quant-weight-reuse-8b, broad mmvq_q4k/q6k, decode_block_fusion,
  hand-LDS WMMA as the prefill lever, and bounded pure-tinygrad WMMA issue/occupancy.
- `q8-mmvq-lifecycle-deep-scope-20260618.md` — deep scope for the only remaining decode MMVQ lifecycle reopening:
  producer-side q8 from fused RMSNorm/apply into Q4_K ffn_gate/up int-dot. Explains what "q8/MMVQ lifecycle"
  means, what is already refuted, phase gates, and why this is low-EV/deep rather than a kernel tweak.
- `q8-mmvq-lifecycle-deep-result-20260619.md` — **executed it: Q8L-0/1 pass, Q8L-2 KILL.** The fused
  per-row→per-32 multi-output producer is NOT expressible via the store-group idiom (needs an LDS-reduction
  flash-style kernel); q8 side-channel is **deferred behind a codegen capability**, not a buildable arc — closes
  the last bounded decode research question.
- `q8-ffn-handwritten-oracle-scope-20260619.md` — research-only oracle scope for the q8 decode reopening: use
  handwritten kernels to test whether the deferred fused RMSNorm→q8 producer plus llama-style Q4_K int-dot consumer
  actually clears correctness, lifecycle speed, block EV, and dNLL gates before funding tinygrad codegen. Includes
  Q8H-0 preflight, Q8H-1 real-GGUF handwritten MMVQ correctness PASS, and Q8H-3/4 producer+lifecycle PASS
  (1.23x gate+up isolated), plus Q8H-5 EV PASS (~1.05x decode model); remaining gate is q8-lossy dNLL/W==D.
- `q8-dual-track-route-and-codegen-scope-20260619.md` — splits q8 into complementary tracks: Track A handwritten/
  backend research route for dNLL/W==D truth, and Track B tinygrad codegen transfer for owning the fused producer and
  q8 MMVQ lifecycle. Adds `extra/q8_ffn_quality_proxy.py`; Track A A0 quality proxy PASS with 160-token dNLL
  +0.00165, so next is HCQ-launchable handwritten route.
- `q8-ffn-fast-artifact-and-codegen-transfer-scope-20260619.md` — forward scope after A2: one-block q8 route is
  correct but COMGR-HCQ artifacts are too slow (`~195us` vs `<=129us` gate). Scopes the two remaining paths:
  hipcc-quality artifact loading through HCQ (`unknown AMD reloc 10` first) and tinygrad-owned raw/codegen transfer.
- `q8-ffn-fast-artifact-vs-raw-code-result-20260619.md` — **executed the two paths.** Path A hipcc/LLD artifact
  loading through HCQ passes when expressed as `producer + fused gate/up consumer` (`114.12us`, correct, no HIP runtime
  in-process). Path B current COMGR/raw-code route remains correct but too slow (`194.80us`). Reopens A3 graph/in-model
  routing only for the fast fused artifact route.
- `q8-ffn-fast-artifact-a3-route-result-20260619.md` — **A3 result.** Fast artifact one-block route passes eagerly
  (`121.38us`, correct vs q8 proxy). Initial Tensor-visible injection faulted; the contract audit found optimized-away
  input buffers and a wrong Q4_K dummy dtype/shape. After fixing both, eager injected node and TinyJit replay PASS
  (`max_abs 0.00137`, no HIP runtime). W==D decode is next.
- `q8-ffn-handwritten-a4-decode-result-20260619.md` — **A4 final gate PASS_RESEARCH.** `Q8_FFN_HANDWRITTEN=1`
  routes dense decode FFN gate/up through the graph-injected q8 artifact. W==D decode improves
  `1.051-1.063x` across ctx 128/512/1024/4096, and actual-route dNLL is `+0.002887` over 160 tokens. Default remains
  off; remaining question is artifact dependency vs codegen/ASM transfer.
- `q8-ffn-codegen-asm-transfer-scope-20260619.md` — **Track B scope + B0/B1 audit.** Disassembles the passing
  hipcc/LLD oracle and slower COMGR route. Both consumers already emit 16 `v_dot4_i32_iu8`; the gap is fused gate/up,
  producer shape, scheduling, and q8 side-channel lifecycle, not a missing dot intrinsic. Next build is a tinygrad-owned
  fused gate/up consumer (`<=60us`) before funding producer renderer work.
- `q8-ffn-codegen-b2a-comgr-fused-result-20260619.md` — **B2a COMGR fused-C result: FAIL_PERF.** The tinygrad-owned
  COMGR fused gate/up consumer is correct (`max_abs <=1.43e-6`) but slow (`146.88us` vs `<=60us`; lifecycle
  `177.72us` vs `<=129.2us`). Closes source-level C reshuffles; remaining B2 path is explicit AMD DSL/ASM or renderer
  scheduling work.
- `q8-ffn-codegen-b2b-asm-consumer-scope-20260619.md` — **B2b AMD DSL/ASM consumer scope + smoke PASS.** Adds
  `extra/q8_ffn_asm_gateup_smoke.py`, which emits `v_dot4_i32_iu8` through `Ops.PROGRAM` and HCQ with no C/hipcc path
  and stores the expected result. Next is a sliced hand-owned fused gate/up consumer: address skeleton -> q8/Q4 load
  skeletons -> one-block dot -> full fused gate/up, gated at `<=60us`. Final B2b verdict: **correctness PASS /
  PERF FAIL**. Full real-GGUF fused gate/up ASM consumer is correct (`max_abs <=1.43e-6`) but slow (`166.649us` vs
  `<=60us`), so native decode ownership is closed as project-level AMD scheduling/compiler work.
- `q8-ffn-amd-scheduler-work-scope-20260619.md` — next-layer scope after B2b: compiler/scheduler work, not primitive
  search. Defines S0-S5: disassembly accounting, reduction audit, address/scale-min audit, load/wait/dot scheduling,
  descriptor/local-id capability, and the decision gate for local hand schedule vs AMD DSL feature vs project-level
  scheduler. Recommendation: run S0 first only.
- `q8-ffn-amd-scheduler-s0-result-20260619.md` — **executed S0 and closed native q8 decode ownership.** tinygrad ASM
  emits the same 16 dot4 ops as hipcc/LLD and fewer static instructions (`218` vs `336`) but is still `166.649us` vs
  `<=60us`; visible deltas are load shape/address scheduling, not a bounded primitive edit. Verdict:
  `S0_CLOSE_PROJECT_LEVEL_SCHEDULER`.
- `q8-ffn-dynamic-scheduler-observability-scope-20260619.md` — scope for option 2 after S0: a tinygrad-native HCQ
  trace/counter bridge for the q8 visible gap. Defines DSO-0..5: q8 HCQ attribution rows, resource/occupancy metadata,
  controlled variant ladder, optional built-in AMD PMC/SQTT attempt, and final classifier
  (`load_shape_bound`, `wait_scheduler_bound`, `closed_project_level`, etc.).
- `q8-ffn-dynamic-scheduler-observability-result-20260619.md` — **executed DSO-0..5.** Classifier:
  `wait_scheduler_bound`. The decisive ladder is body-insensitive: reduction-only/synthetic-dot/load-only variants all
  remain ~0.151-0.153ms vs full ASM 0.166ms, so the visible q8 gap is broader AMD scheduling/work-decomposition/codegen,
  not a bounded load-shape primitive.
- `q8-ffn-amd-scheduler-codegen-project-scope-20260619.md` — complete next-layer scope after DSO: Route A native
  tinygrad AMD scheduler/codegen transfer, Route B artifact/import research route, and Route C schedule-import training
  data. Defines gates for when to reopen q8 producer ownership vs keeping decode closed as compiler roadmap.
- `q8-ffn-artifact-import-route-result-20260619.md` — **executed Route B.** Reproducible hipcc/LLD artifact build,
  fixed-launch HCQ loader, graph injection, and maintenance boundary all pass as **research-only / policy-bound**.
  Isolated lifecycle `115.24us`; graph replay max_abs `0.001373`; default off; no in-process HIP runtime.
- `q8-ffn-route-a-scheduler-codegen-result-20260619.md` — **executed Route A A0/A1.** Oracle contract extraction
  passes, but AMD DSL capability map finds no bounded A2 feature: vector loads ~14us, wait grouping ~0.8us, reduction
  rewrite ~13us, dot4 already solved. Native q8 ownership stays project-level scheduler/codegen roadmap.
- `q8-ffn-route-a-pmu-sqtt-evidence-result-20260619.md` — **post-A1 evidence gate.** tinygrad HCQ-level PMC/SQTT
  collection works for the q8 ASM path (`2` PMC events, `12` SQTT events, ~1.78 MB trace), but local SQTT decode fails
  on the captured RDNA3 blobs and no bounded `>=30us` A2 feature is identified. Route A remains closed for q8 decode
  except as a project-level AMD scheduler/codegen effort.
- `spec-decode-bandwidth-amortization-scope-20260619.md` — reopens spec decode only under the PMU-backed
  weight-read-amortization framing. Keeps the old `decode_spec_verify_shortcut` closed; defines the new
  `decode_spec_weight_amortization_lifecycle` row, whose hard gate is T=K+1 verify `<=1.5x` one T==1 pass plus
  low-sync accept/commit and greedy byte-exactness.
- `spec-decode-bandwidth-amortization-sdb1-sdb2-result-20260619.md` — **executed SDB-1/SDB-2.** Current spec remains
  non-viable (`~0.52x` before runtime with 0.6B K=4) because verify is `4.65x`; reaching `<=1.5x` requires a
  `67.8%` verify cut across Q4_K, Q6_K/lm_head, and attention/reduces. No bounded shared primitive; spec is
  project-level T-cheap batched-forward work.
- `spec-decode-tcheap-batched-forward-project-scope-20260619.md` — project-level decode-only scope for making spec
  viable after SDB-1/SDB-2: a short-block target verify forward, low-sync accept/commit, exact KV protocol, and
  T=K+1 verify `<=1.3-1.5x` one pass. Explicitly not a prefill route and not a bounded kernel edit.
- `spec-decode-tcheap-batched-forward-tbf0-tbf2-result-20260619.md` — **executed TBF-0..TBF-2.** Short-block verify
  IR contract is defined, but current component gates all fail: Q4_K `2.916x`, Q6_K/lm_head `5.831x`,
  attention/reduces `3.061x`, linears group `3.523x` vs the `<=1.5x` T-cheap gate. Stops before TBF-3 until a
  concrete component route exists.
- `spec-decode-component-route-candidates-scope-20260619.md` — next decode-only scope after TBF-0..2: candidate
  routes for grouped short-block quantized linears, short-block causal verify attention, and their combined
  projection. No implementation until a candidate passes its component gate.
- `spec-decode-component-route-candidates-result-20260619.md` — **executed SCR-0..SCR-4.** Candidate attention
  generalization has no bounded proof surface, grouped short-T linears have no shared Q4_K/Q6_K bounded schedule, and
  combined projection has no passing ceilings. Verdict: `PROJECT_LEVEL_CLOSE`; do not build TBF-3 unless a new measured
  component candidate clears `<=1.5x`.
- `llama-kernel-residual-primitive-audit-scope-20260619.md` — scope for auditing llama.cpp's **own** remaining
  primitive headroom: MMVQ residual-to-peak, q8 quant, attention, small-op fusion, graph boundaries, and prefill.
  Separate from the tinygrad-vs-llama gap explanation.
- `llama-kernel-residual-primitive-audit-20260619.md` — result of that audit. llama is not theoretically optimal,
  but fresh `rocprofv3` traces show prompt-free decode is 85.6% MMVQ; q8/RMSNorm is the only moderate non-MMVQ
  decode lifecycle candidate, graph launch overhead is already solved by HIP graphs, and pp512 prefill is 74.4%
  quantized MMQ/matmul rather than attention-limited.
- **Decode-attention wins SHIPPED (byte-identical greedy, default-on):**
  - `qk-8b-attention-fusion-result-20260617.md` — flash-decode threshold 1024→512 (+12.8% ctx520).
  - `qk-8b-flash-variant-result-20260617.md` — `hoisted` exp + L=128 default (+11–29% across ctx).
  - `qk-gqa-coop-vector-load-result-20260617.md` — `gqa_coop_vec` default → decode-attention slope gap CLOSED.
- **Q4_K MMVQ int-dot line — CLOSED:** `qk-mmvq-int-dot-closeout-20260618.md` (**read this**) — the
  consolidated bank. SHIPPED `_sdot4`→native signed dot4 via `__builtin_amdgcn_sudot4` (fixed a latent
  unsigned-bug; value-tested; used by no default path); 128-thread/row sudot4 kernel 57% correct (beats opaque
  52%) but whole-linear REFUTED by the q8-pack wall (reuse ceiling 2 + ~7µs pack floor); int-dot FFN refuted.
  - Key sub-arcs (provenance): `qk-dot4-isa-audit-20260618.md` (the sudot4 fix + RDNA3 dot4 ISA map),
    `llama-q4k-mmvq-scheduler-audit-20260618.md` (llama's MMVQ decomposition),
    `qk-mmvq-llama-scheduler-probe-verdict-20260618.md`, `qk-mmvq-sudot4-full-linear-arc-20260618.md`,
    `qk-q8-activation-lifecycle-verdict-20260618.md`, `qk-mmvq-{codegen,deep-linearizer,fused-coop-row}-*`.
- **Current decode standing:** ~66–69% of llama via the shipped coop + flash-decode routes. Residual MMVQ gap =
  per-thread codegen (tinygrad-internals). 14B/32B pivot deferred per standing preference.

## Active / open frontiers

- `prefill-wmma-lds-tiling-scope-20260619.md` — provenance for the now-refuted Branch A. After decode closed, the surviving high-EV arc:
  PREFILL_V2 forward is ~74% fp16 WMMA matmul emitted with LDS=0; the lever is WMMA operand LDS-tiling (~1.6× pp).
  Decision-first: Phase PWLT-0 is the authority call — Branch A (tinygrad hand-LDS, **triple payoff**: also unblocks
  q8 producer + flash-prefill attention) vs Branch B (external hipBLASLt/rocBLAS, prefill-only). Both feasible
  (assets/libs present); recommendation A-first, B as fallback control.
- `prefill-wmma-lds-tiling-result-20260619.md` — **executed Branch A: PWLT-A1 pass, PWLT-A2 KILL.** Hand-LDS WMMA
  = 1.02× the default matmul (both ~34% peak) → **LDS-tiling is NOT the lever** (IC-served on gfx1100, like decode
  attention). Real headroom is dense WMMA issue / Tensile-class scheduling, not LDS staging.
- `prefill-external-blas-result-20260619.md` — **ceiling/control measured.** Host-only C++ avoids the split-HIP
  compile issue; hipBLASLt reaches 69.8 TFLOPS on ffn_gate/up (1.71× tinygrad) and rocBLAS reaches 70.9/76.7 TFLOPS
  on ffn_down/attn_q/o. This proves a higher GEMM ceiling, but routing remains an external-dependency + HCQ-vs-HIP
  runtime boundary.
- `prefill-external-rawhip-tensile-boundary-scope-20260619.md` — broad external/raw-HIP/Tensile boundary scope
  before EBT-1.
  Starts with the authority decision, then EBT-1 tinygrad-buffer pointer interop, EBT-2 bridge/shape overhead,
  EBT-3 one-block transfer, EBT-4 full warm pp, and fallback lanes for Tensile HSACO or raw-HIP kernels. It also
  states the key gate conflict: strict >=1.5x full pp likely stops because the measured ceiling caps around
  1.4-1.45x before overhead. Superseded as the active plan by the Lane B scope below after EBT-1 killed Lane A.
- `prefill-external-bridge-ebt1-result-20260619.md` — **executed EBT-1: Lane A KILL.** HIP runtime and tinygrad
  HCQ/KFD are mutually exclusive in one process, so in-process rocBLAS/hipBLASLt on tinygrad pointers is closed.
- `prefill-tensile-primitive-extraction-and-codegen-scope-20260619.md` — **current Lane B scope.** Extract the
  selected Tensile primitive and its full launch contract (solution, HSACO, symbol, `.kd`, kernargs, launch geometry,
  workspace) and run it through tinygrad HCQ. Also scopes option 2: only after a working extracted contract exists,
  use it as the target for a tinygrad codegen/Tensile-class schedule transfer.
- `prefill-tensile-tpe4-perf-result-20260619.md` — **executed TPE-4: PASS.** The extracted rocBLAS Tensile
  ffn_gate/up primitive runs through tinygrad HCQ at 66.91 TFLOPS median (0.7703 ms), correct, no copies, no HIP
  runtime in-process. Lane B is now runnable and fast for one fixed shape.
- `prefill-tensile-tpe5-shape-matrix-result-20260619.md` — **executed TPE-5: PASS.** The extracted Tensile primitive
  generalizes: ffn_gate/up 66.8, ffn_down 68.9 (StreamK, no workspace), attn_q/o 58.9 TFLOPS through HCQ — all correct,
  stable, no workspace/aux/layout-copies, one code object + one pointer convention. Weighted model predicts **~1.40×
  full warm pp512** (→ ~2920 tok/s ≈ 95% of llama) if all three are routed, above the 1.25× gate.
- `prefill-tensile-tpe6-block-transfer-result-20260619.md` — **executed TPE-6: REDIRECT.** A whole FFN block
  (gate+up+silu·up+down) routed through the kernels is **exact** (rel 4.8e-4) and copy-free (weights stay natural
  `[out,in]`, run in `[feature,T]` space, zero per-matmul transposes), and the block matmuls hit 61 TFLOPS = **1.53×
  the PREFILL_V2 plateau on GPU time**. But naive per-op routing adds ~6.2 ms host sync overhead (a JIT-less probe
  artifact) that swamps the win end-to-end → realizing it needs a **single-dispatch graph (HCQGraph/TinyJit) runtime
  helper**. Next: build that helper, re-run the block gate, then TPE-7 (no model default; external-artifact policy pending).
- `prefill-own-wmma-kernel-scope-20260619.md` — pure tinygrad/no-deps scope. Key learning: tinygrad's
  WMMA matmul (41 TFLOPS) only *matches* the non-WMMA ALU matmul (40) — it gets **none** of the tensor-core 2×, so
  WMMA units are **stalled, not the bottleneck**. POWN-0 diagnose (occupancy / accumulator-chain / issue-rate) →
  POWN-1 config sweep (LDS-off since IC-served, chase dense WMMA issue + occupancy) gated ≥1.5×. The result below
  banks the bounded no-deps ceiling.
- `prefill-own-wmma-kernel-result-20260619.md` — **executed POWN-1: KILL.** Best config is the existing
  B128x128x16/W2x2 at 42.0 TFLOPS; more waves, bigger tiles, BK32, and noLDS all regress. No bounded no-deps
  prefill WMMA knob reaches the 62 TFLOPS gate.
- `prefill-external-blas-scope-20260619.md` — **DECLINED (no external deps).** rocBLAS/hipBLASLt ceiling-first plan;
  kept as provenance for the bridge analysis (DEV=AMD HCQ vs HIP-runtime). Its PXB-1 ceiling has now been measured
  in `prefill-external-blas-result-20260619.md`.
- **`amd-decode-prefill-v2-increment1-20260617.md`** — **prefill v2 BUILT & WON: ~13x warm prefill** (189→2486
  tok/s, ~83% of llama) via concrete-ubatch + fp16 + realized-weights + warmstart-TC, gated `PREFILL_V2`,
  decode untouched. Quality gate PASSED (dNLL ~0, 8B). Corrects the Stage-0 gate's premise (lazy weights →
  realize/VRAM; per-shape opts; host-overhead confound). Gate: `amd-decode-prefill-v2-gate-20260616.md`.
- **`amd-decode-prefill-v2-increment2-20260617.md`** — **flash-prefill attention: GATED (banked)**. Attention
  is the next prefill bottleneck at long ctx (~51% @ sp=3072) but the tractable approaches are refuted.
- **`amd-decode-prefill-v2-increment2-phase5-correction-20260617.md`** — **CORRECTION + kernel-level
  confirmation**: a custom score-free fused attention kernel IS expressible/correct (bridge + capabilities +
  expressibility proven, `test_flash_prefill_custom_kernel*.py`), but **honest DEBUG=2 GPU time REFUTES it on
  perf (~170–760× SLOWER than SDPA**; the earlier ~2.7× were wall-clock artifacts). Score-free w/o LDS reuse =
  memory-bound; real flash-2 needs LDS tiling (BEAM-territory, hangs gfx1100). Flash-prefill banked; prefill v2
  rests at Increment 1. **Methodology lesson: GPU timing via DEBUG=2 `tm`, never wall-clock around `.realize()`.**
- `amd-decode-prefill-plan.md` — the original prefill diagnosis (~2% of llama; LDS cache-blocking). Superseded
  as the active plan by prefill v2 above, but still the canonical root-cause reference.
- Phase-2 decode docs (2026-06-16): `amd-decode-sequential-tax-profile`, `…-overlap-feasibility-spike`,
  `…-overlap-derisk`, `…-two-queue-probe` (**overlap GATED** on a 2nd compute ring), `…-demotion-search`
  (B3 done), `amd-decode-flash-attention-plan` (flash SHIPPED).
- Direction + status: `structure/Development/machine-search-decode-context-plan-2026-06-16.md`;
  running log `structure/Development/session-handoff.md`.

## Machine-search system (shipped this arc)

The bounded search loop, dogfooded on B3. Code: `extra/qk_search_spec.py` (schema authority),
`extra/qk_nll_eval.py` (decode-path dNLL gate), `extra/qk_demote_search.py` (orchestrator). Result:
`amd-decode-demotion-search-20260616.md`.

## Architecture references (live)

- `amd-decode-harness-architecture.md`, `amd-decode-qk-storage-architecture.md`,
  `amd-decode-primitive-v2-design.md`, `amd-decode-bandwidth-roofline.md`,
  `amd-decode-packed-{load-lowering,qk-tile-design,qk-semantic-op}.md`.

## Historical — the decode-arc probe log

Dated scope/result docs whose verdicts are now captured in the syntheses above. Kept for provenance;
**not current state** (several carry a SUPERSEDED header).

- *"current state" docs, now superseded by the bank:* `amd-decode-current-verdicts.md`,
  `amd-decode-methodology-and-roadmap.md`, `amd-decode-final-report.md`, `amd-decode-hypothesis-statement.md`,
  `amd-decode-consolidated-first-principles.md`, `amd-decode-optimization-plan.md`.
- *bottleneck diagnosis & probes:* `amd-decode-rootcause`, `…-fix-plan`, `…-perlayer-plan`,
  `…-validate-plan`, `…-memory-access-audit`, `…-dequant-instruction-count`, `…-latency-vocabulary`,
  `…-dp4a-vocabulary`, `…-prefetch-plan`, `…-mirage-probe`.
- *kernel/TC/GEMM probes:* `…-option1-result`, `…-option1-corrected`, `…-batched-tc-{plan,result}`,
  `…-warmstart-plan`, `…-verify-loop-plan`, `…-fusion-probe-plan`, `…-vdot-amort-plan`,
  `…-amortized-quant-plan`, `…-scale-and-vdot4-plan` (`amd-loop-…`), `…-semantic-family-b`,
  `…-lossy-quant-search`.
- *levers later synthesized:* `amd-decode-speculative-plan` (B5), `amd-decode-prior-art`.

## Flywheel sub-arc (model-to-kernel triage/generation) — concluded

Read the postmortem first; the learned model added no value at the current feature set, the
native-matmul loop substrate works (decoupled from the decode bar).

- `amd-decode-flywheel-postmortem.md` (read first), `amd-decode-loop-substrate.md`,
  `amd-decode-flywheel-proof-plan.md` (2.6k-line plan), `amd-decode-kernel-optimization-flywheel.md`,
  `amd-decode-ansor-direction.md`, `amd-decode-loop-live-plan.md`,
  `flywheel-judging-rewrite-scope.md`, `flywheel-rewrite-ubuntu-handoff.md`,
  `qwen-json-eval-objective-scope.md`, `research-paper-brief.md`.

## Other subsystems

- **PSP / boot** (separate from decode): `amd-kdb-root-cause.md`, `amd-linux-psp-good-trace.md`,
  `amd-ubuntu-boot-prompts.md`, `amd-remote-dropout-investigation.md`.
- **Reference research:** `amd-rocm-llamacpp-research.md` (llama.cpp/ROCm/MMQ deep dive).

## Upstream tinygrad docs (not fork-specific)

`index.md`, `quickstart.md`, `mnist.md`, `nn.md`, `dtypes.md`, `env_vars.md`, `runtime.md`,
`tinygpu.md`, `tinybox.md`, `showcase.md`.
