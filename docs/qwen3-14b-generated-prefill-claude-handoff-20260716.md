# Qwen3-14B generated-prefill Claude handoff

## 1.9 Current status 2026-07-19: production low-level route layer is CPU-preflight complete

The current `ffn_gate_up` family now has a production-faithful route implementation behind the C8 evidence
interfaces. It reuses tinygrad's frozen PROGRAM loader, AMD runtime cache, fixed-VA buffers, same-device SDMA
transfer, native dispatch helper, direct-packed executable attestor, spawned child containment, queue health probes,
and kernel-fault collector. No HIP launcher or second dispatch path was added.

The candidate and direct routes share the exact same resident FP16 activation Tensor object. Candidate epoch-major
Q4 is uploaded once and retained outside timing; direct N-major packed Q4 is realized once and retained outside
timing. Every candidate invocation produces physical DS4 Q8 from the shared FP16 Tensor inside its outer wall, zeros
the persistent FP32 output, performs four ordered fixed-VA transfers and one exact frozen target dispatch for each of
20 epochs, and retains Q8/output references through post-sync attestation. The attestation checks the runtime object
and runtime-cache binding, uploaded code range, mutual disjointness and stable VAs for every resident source, full Q8
source, fixed buffer, and code range, requested and captured kernarg pointer VAs, program/binary identities, queue,
input, and launch count.

Direct packed calls the production `_run_direct_packed_baseline`; its existing frozen attestor owns output
realization and observes the actual executable after synchronization. A child-local census wraps the instantiated
tinygrad AMD allocator's `_copyout`, so any readback/copyout through that allocator during either measured invocation
blocks the receipt. The guarded parent runs a fresh PM4 child and then a fresh AQL child sequentially, with pre/post
health and fault capture, and does not attempt AQL after any PM4 timeout, fault, reset, unhealthy state, or blocked
child.

CPU-only integration and regression coverage passes 184 tests under
`pytest -q test/unit/test_mmq_ffn_gate_up*.py test/unit/test_mmq_frozen_staged_low_level_session.py test/unit/test_direct_packed_executable_attestor.py`.
Importing the new modules opens no device. The real frozen bundle also reconstructs the exact pointer-only authority:
compact shape `512x17408x256`, full shape `512x17408x5120`, 20 dispatches, and PROGRAM key
`14f2a216a8a7609e8a251fe3869b3fb146fd5d5a8ca0ec468120e0fbcbd54a60`.

This closes implementation preflight, not GPU qualification. The current C8 composition cannot exist until new
family-bound C4/C6/C7 and direct qualification evidence have been collected. The next execution order is:

```text
C4 zero-target-dispatch runtime canary on PM4
  -> C5 candidate prefix 1, then prefix 3 on PM4
  -> C6 candidate full 20 plus matched direct correctness on PM4
  -> PM4 transition sequences and joint C7
  -> repeat the guarded correctness/transition sequence on AQL
  -> freeze C4/C6/C7/direct qualifications and the matched contract
  -> only then run paired C8 timing
```

No GPU was run while building this layer. Numerical correctness, runtime health, transition safety, memory
admission, performance, whole-model behavior, and promotion remain unknown/open.

## 1.8 Current status 2026-07-19: new exact `ffn_gate_up` family passes CPU/static C1-C3

This section supersedes any implication that the historical full-role `ffn_gate_up` runtime evidence is already
bound to the current candidate. A new deterministic compact-K256 staged family has been frozen for the exact
`ffn_gate_up (512,17408,5120)` role from clean revision
`3fa4cd6195e460930417732fb404521e33c9cf3c`:

- family `sha256:4e82044650b7c03a579b42b8dd270389d280c2f5eab9f9004a3bc83dbe79917f`;
- PROGRAM key `14f2a216a8a7609e8a251fe3869b3fb146fd5d5a8ca0ec468120e0fbcbd54a60`;
- retained SINK key `a3a4f98c4ebebfe8f770f2f3f4e611c22f92510845e482d1bc79dfb75963a495`;
- source SHA256 `05875a80074fb19df641fb86f97090b01af1844bf2074ed8d7b10274c15b57cc`;
- HSACO SHA256 `149ba322c1a99c1fa056d25c6230bc8908c27f15fe94b177276c5808eebe8bf3`.

The frozen seven-file target bundle and compact evidence are retained under
`docs/artifacts/qwen3-14b-prefill-ffn-gate-up-staged-3fa4cd619-20260719/`.
Generation was CPU-only through `AMDISARenderer`; provenance records a clean repository, the exact 23-key compiler
environment, no `Device` initialization, and no GPU allocation, queue, or dispatch.

C1 passes the static HSACO audit with 13,020 instructions and a terminal `s_endpgm`. C2 binds the final gfx1100
wave32 descriptor and byte-identical reassembly: grid `136x4x1`, local size `256x1x1`, 57,856-byte LDS, 256
allocated/used VGPRs, 16 SGPRs, zero scratch, and zero VGPR/SGPR spills. C3a/C3b pass for the same family and program.
The exhaustive native certificate covers 139,264 launch coordinates and 19,636,224 projected address evaluations
with no bounds failure, uint32 wrap, or incomplete output read-modify-write coverage. The full C3 certificate SHA256
is `9170c3f71b04a4fe7943cafc4a7b0beaa3d0d41f290d47a07f829e347f8e556f`.

The matched-timing foundation is now fail closed and role exact. It declares one common logical Q4 input and one
resident FP16 activation; requires the candidate to produce Q8 from that FP16 activation inside every measured outer
wall; excludes correctness readback; requires separate persistent PM4/AQL sessions, at least three warmups per route,
at least ten seeded balanced pairs, and exact candidate/direct executable qualifications; and reports a win only if
both queues win. Runtime-owned typed callbacks and typed no-readback output realization flow unchanged through the
session worker and collector. The worker binds a live instantiated-device queue attestation before executing either
route, validates a post-sync observation of the actual queue/route/executable/input after every wall, and retains a
cumulative host-I/O census spanning pre-sync through that observation. Readback or device-to-host
copyout, clock reset across invocations, duplicate receipts, odd/unbalanced pairing, legacy receipt runners,
prequantized launch inputs, retry, queue fallback, or identity drift fail closed. Injected callback captures remain
explicitly ineligible for promotion until the production route builders supply these observed authorities.

This is infrastructure and static certification, not runtime qualification. No GPU has executed this new binary.
C4 zero-dispatch construction, FP16-backed C5/C6 correctness, joint C7 memory admission, mixed-route transition
preflights, matched C8 timing, whole-model validation, and production promotion all remain open:

```text
ffn_gate_up_current_family_c1_c3=PASS
ffn_gate_up_current_family_c4_c8=OPEN
ffn_gate_up_current_family_gpu_execution=false
ffn_gate_up_current_family_correctness=UNKNOWN_NOT_EVALUATED
ffn_gate_up_current_family_performance=UNKNOWN_NOT_EVALUATED
production_promotion=false
default_route=direct_packed
```

## 1.7 Current status 2026-07-19: `ffn_gate_up` performance is `UNKNOWN_NOT_EVALUATED`

This section supersedes the `ffn_gate_up` performance conclusions in §1.2, §1.1, and §0.9. Source inspection of
`run_full_grid_r5_benchmark` proves that both retained `0.277502 ms` generated and `9.351988 ms` direct observations
use `SHAPE=(128,128,256)`; the latter is not a complete `512x17408x5120` role measurement. That bounded pair still
cannot establish a speedup because its timed regions differ: generated times a precompiled resident
`runtime(..., wait=True)` launch, while `_run_direct_packed` includes tensor construction/conversion, realization,
and `.numpy()` readback.

The complete-role generated observations remain useful raw data: `42.882262 ms` synchronized AQL, `27.076011 ms`
queued AQL, and `41.861031 ms` PM4. No complete-role direct-packed observation with the same preparation,
allocation, readback, and synchronization definition was retained. Therefore no full-role ratio, win, or loss is
established. The corrected status is:

```text
ffn_gate_up_correctness=PASS
ffn_gate_up_resources=PASS
ffn_gate_up_performance=UNKNOWN_NOT_EVALUATED
r5_emitted_backend_win=false
r6_ready=false
production_promotion=false
default_route=direct_packed
```

`docs/qwen3-14b-prefill-r5-geometry-20260718.json` now carries v2 candidate/comparator workload and measurement
descriptors and marks the legacy measurements non-comparable. `docs/qwen3-14b-prefill-q4-role-closeout-20260718.json`
and `docs/qwen3-14b-prefill-mmq-one-role-evidence-20260718.json` retain the raw observations but explicitly supersede
the historical `R5_COOP_WIN_READY_FOR_R6`, `34.694x`, and `2.895x slower` claims.

## 1.6 Current status 2026-07-19: exact staged `attn_qo` is transition-disqualified

This section supersedes §1.5 for the exact deterministic staged family. C1-C7 remain passing evidence; the C8
outcome is a safety disqualification, not a timing loss or a generated-route promotion.

The selected family is still:

- role `attn_qo (512,5120,5120)`;
- family `sha256:2cfc30075f8024cee8a927c2c3de2e87eef3db6d83882da69faa0fe0a3cc1e4f`;
- PROGRAM key `3f478e6d89a2de467f6b7d1ca18418cdfd0cdb19de05db1d66608e65a5e6475f`;
- HSACO SHA256 `dfb213624287a8dec10f8646d8c16e49651efee8e0ca27c67ff982b0d6b050bf`.

C7 now passes independently on PM4 and AQL under one live device/software authority. The staged route peaks at
104,988,672 physical bytes on PM4 and 121,765,888 bytes on AQL, both below the shared 25,248,317,440-byte
admitted budget. It materializes no dense FP16 weight. Standalone staged correctness remains zero mismatches with
maximum absolute error `0.00341796875`.

The promotion-grade mixed-route session cannot safely reach matched timing:

1. `staged_candidate -> staged_candidate` passes under one retained PM4 runtime/buffer owner.
2. `direct_packed -> staged_candidate` passes direct packed, then faults on candidate invocation 0 before epoch 0
   completes.
3. The same boundary reproduces under PM4 and AQL, with effective instantiated-queue attestations, no retry, no
   queue fallback, an SQ type-2 memory violation, GPU reset, and healthy post-reset recovery.

The following non-decision diagnostics narrowed the cause but are not inputs to the normalized safety
classification: retaining the direct output allocation does not change the result; candidate code bytes are
identical before and after direct execution; candidate `RSRC3.INST_PREF_SIZE` and `RSRC1.FWD_PROGRESS`
discriminators do not change the result; and offline final-ISA def/use finds no undefined physical-register reads
beyond the ABI live-ins. The retained C3 certificate exhaustively covers 5,873,664 projected final-native global
addresses with no out-of-bounds access or uint32 overflow. The separate LDS audit covers all 688 DS operations; the
maximum effective end is 57,840 bytes within the declared 57,856-byte LDS allocation.

The evidence therefore supports an exact safety decision without claiming a driver root cause: this staged
candidate passes by itself but is not safe after the admitted direct-packed route. Do not run a route-isolated
timing collector and reinterpret it as promotion evidence. Real inference also crosses kernel/route boundaries, so
fresh-process timing would hide the production safety requirement.

The generic memory-adaptive selector now requires every accelerated candidate to carry a search-keyed production
eligibility requirement and matching candidate-bound evidence. Missing, malformed, blocked, or identity-mismatched
evidence rejects the accelerated row; the named `DIRECT_PACKED_FALLBACK` baseline is the only exemption. The runtime
collector revalidates the selected accelerated record before model binding. Existing full-resident-overlay search
continues through its guarded whole-model full-output/route-census authority. The Qo adapter binds both the exact C6
composition and this transition-safety classification, so even a hypothetically faster Qo row is rejected and a
later composition-only eligibility change cannot bypass the retained safety disqualification. Qo is still not
added to the production candidate catalog, and no generated-route runtime binder is claimed.

The durable bundle and decision inputs are under
`docs/artifacts/qwen3-14b-prefill-attn-qo-staged-951d3615c-20260719/`. The normalized transition classification has
identity `sha256:9c7b68d681293876c7ee2542bbc4dc8e055b9f68fce5b7b7d54e6a00143038eb` and records:

```text
attn_qo_staged_c1_c7=PASS
attn_qo_staged_transition_pm4=SQ_TYPE_2_AT_CANDIDATE_EPOCH_0
attn_qo_staged_transition_aql=SQ_TYPE_2_AT_CANDIDATE_EPOCH_0
attn_qo_staged_candidate=DISQUALIFIED
attn_qo_selected_route=direct_packed
attn_qo_c8_status=BLOCKED_AT_C8
attn_qo_timing_c8_status=NOT_EVALUATED
attn_qo_timing_c8_win=false
production_promotion=false
```

`direct_packed` remains the FP16-overlay-free, memory-admitted quantized fallback for this scanned
Qwen3-14B/gfx1100 setting: packed Q4/Q6 weights remain resident and a dense full-model FP16 weight overlay is not
created. Selecting it resolves this exact Qo candidate safely; it does not satisfy the
higher-level objective of finding and promoting a faster generated candidate. Phase 6 mixed-route whole-model
validation and Phase 7 parity remain open.

## 1.5 Current status 2026-07-19: deterministic staged `attn_qo` passes PM4/AQL C1-C6

This section supersedes §1.4 for the selected dense fixed-VA staged family. The direct-layout v3 classification in
§1.4 remains the negative-control result and must not be reopened.

Commit `951d3615c [amd] stabilize progressive drain ordering` fixes the fresh-process compiler reproducibility
blocker. `_serialize_progressive_c_drains` previously selected equally ready drain heads through identity-hashed
`UOp` set order, publishing heap-layout order into post-instruction-selection dependencies, later scheduling,
register allocation, source, and HSACO. The fix uses selected-graph position only as the equal-ready tie-break.
Twenty-seven focused tests and a 92-test staged-family review pass.

Two clean, exact-environment `attn_qo (512,5120,5120)` builds at `951d3615c` are now byte-identical through the
retained SINK, PROGRAM, rendered source, disassembly, HSACO, serialized PROGRAM, bundle manifest, deterministic ustar
archive, and staged-family manifest:

- family identity
  `sha256:2cfc30075f8024cee8a927c2c3de2e87eef3db6d83882da69faa0fe0a3cc1e4f`;
- SINK key `bd77ed89317319ced878964575afdbc59487f23d50240ad7d397d5bd2f9cbe44`;
- PROGRAM key `3f478e6d89a2de467f6b7d1ca18418cdfd0cdb19de05db1d66608e65a5e6475f`;
- source SHA256 `c8b75dd0cf9905d02d74ca5923154669692b2c1df4dd627980c769c82cc021ef`;
- HSACO SHA256 `dfb213624287a8dec10f8646d8c16e49651efee8e0ca27c67ff982b0d6b050bf`;
- both archive SHA256 values
  `35d52dbd52a4add2ba564ab216617569544e12fa0adf12ddbb830be43ed2ecf3`.

C1-C3 pass for that exact family. C2 reports gfx1100 wave32, grid `40x4`, local size 256, 57,856-byte LDS,
256 allocated/used VGPRs, 16 SGPRs, zero scratch, and zero VGPR/SGPR spills. C3a/C3b exhaustively bind the retained
source and final native address expressions; all five ABI bases, 40,960 launch coordinates, 5,873,664 projected
addresses, and exact-once output RMW coverage pass.

The isolated escalation passes independently under both PM4 and AQL:

1. C4 preconstructs the exact runtime with zero target dispatches, no compile/recompile, exact cache/binary binding,
   clean timeline and fault window, and healthy pre/post probes.
2. C5 prefix 1 compares all 2,621,440 outputs with zero mismatches, maximum absolute error
   `0.0001220703125`, exact five-pointer/stage bindings, phase isolation, and clean health.
3. C5 prefix 3 compares all outputs with zero mismatches, maximum absolute error `0.00048828125`; all three
   overwrite/submit/synchronize receipts pass and health remains clean.
4. C6 full 20 compares all outputs with zero mismatches, maximum absolute error `0.00341796875`; all 20 fixed-VA
   staging and target lifecycle receipts pass, with no fallback, HIP path, fault, reset, or unhealthy post-state.

The PM4 and AQL C6 results have identical numerical comparisons and lifecycle counts. Current working evidence is:

- static summary `/tmp/qk-attn-qo-staged-951d3615c-final-static-evidence-20260719.json`,
  SHA256 `9d34e25a24e5b207428ed9f5c8f3bd8d60050cc7a940c8d675d0f87e3c26a6b2`;
- C4 `/tmp/qk-attn-qo-staged-951d3615c-final-20260719-c4-pm4.json`,
  SHA256 `df45aa281b2e12eb73f4801f6c1fa31e2c25570116b3cd5ddcbba45f88192004`;
- C5 prefix 1 and 3 SHA256 values
  `579d55f6a5e753946bc6d0ad6e6f0bd02a65d0796335f839748ba4bd1b5996ff` and
  `c81973ff564047bcb2c296a51b836f840ab8219e8a43ce342fcd28835cbf86f6`;
- C6 `/tmp/qk-attn-qo-staged-951d3615c-final-20260719-c6-pm4-full20.json`,
  SHA256 `7ef898e7efb5562df7c7eb9ee006459348b5f9409e3abc9aaeff73cca73109ad`.
- AQL C4 `/tmp/qk-attn-qo-staged-951d3615c-final-20260719-c4-aql.json`,
  SHA256 `847a2eb181e7d07d5500c632a928e200c021139534827ba38dc8eef8df1d3abc`;
- AQL C5 prefix 1 and 3 SHA256 values
  `4735454e1d8eb0276df84bd72d26737599c8bf9cfc1bd4212799953be81d3299` and
  `31bf5ffaa6349aac2be68dac3e5ac895ecc40de93dbdeb6aa51b445afe71cbc4`;
- AQL C6 `/tmp/qk-attn-qo-staged-951d3615c-final-20260719-c6-aql-full20.json`,
  SHA256 `ad959b8b218b6c6594431329d1978b619147fde98bb112b2b81bbf4210178fd1`.

These `/tmp` paths are not durable promotion assets. Retain the content-addressed bundle and evidence before final
promotion. C7 exact memory admission and promotion-grade C8 matched PM4/AQL timing remain open. The C6 synchronized
samples are `27.137666009366512 ms` under PM4 and `26.153420913033187 ms` under AQL. They are
diagnostic only and already suggest this serialized staged family will be a correctness-qualified fallback against
the retained roughly `9.35 ms` direct-packed comparator. Do not make a C8 decision without the required matched
warmups and randomized paired rounds.

```text
attn_qo_staged_c1_c6_pm4_aql=PASS
attn_qo_staged_c7=OPEN
attn_qo_staged_c8=OPEN
production_promotion=false
default_route=direct_packed
```

## 1.4 Current status 2026-07-19: `attn_qo` direct layout classified; staged family is next

Read this section before the older corrected-v2 lifecycle diagnoses. Section 1.2 remains the historical performance
authority, and `docs/generated-prefill-role-certification-method-20260718.md` is the current prospective
certification method.

The provenance-complete direct-layout v3 `attn_qo (512,5120,5120)` family passes C1-C4 with zero scratch/spills.
Its exact epoch-0 PROGRAM and full allocations pass research-only grids `1x4`, `8x4`, and `9x4`, but `16x4` reports
an SQ type-2 memory violation and reset. Two pointer-biased `1x4` discriminators then isolate the conspicuous
transition region:

- tile 11 crosses 4 MiB relative to the Q4 allocation and passes all 65,536 target values; the other 2,555,904 output
  values remain exact zero; health/fault evidence is clean. Artifact
  `/tmp/qk-attn-qo-v3-c5-single-tile11-retry2-20260719.json`, SHA256
  `49e21dd3684fe28b84db396012e7c0ff552a6c1a0753ea8e4b0f23b54e0ea463`;
- tile 12 begins above that boundary and passes the same target and untouched-output checks with clean health/fault
  evidence. Artifact `/tmp/qk-attn-qo-v3-c5-single-tile12-20260719.json`, SHA256
  `95354f34c19d790334960c747973d84334cb7dbed336bf49ff07981441fa0155`.

These diagnostics refute an invalid individual tile or the relative 4 MiB crossing as the sufficient cause. The
failure requires aggregate execution over the sparse direct full-role Q4 layout. Historical compact fixed-VA
per-epoch Qo already passed exact `40x4`, full 20, with zero mismatches and the same resource envelope. Its retained
full result is `docs/qwen3-14b-prefill-attn-qo-fixed-va-20epoch-pm4-20260718.json`, SHA256
`c532a1677557054018cfca6462b41612ab53dc8ab9351c547e5ca382754a2833`.

Stop schedule changes, direct-grid widening, and tile-by-tile search. Build a certification-grade staged family using
the existing tinygrad emitter/runtime path, run C1-C4, then `prefix 1 -> prefix 3 -> full 20`, followed by C7 memory
admission and C8 complete-role timing. The historical staged executable measured `76.49636000860482 ms` versus the
retained direct-packed `11.41 ms`, so it is performance-rejected and supplies no promotion claim for the new family.

```text
attn_qo_direct_v3_status=BLOCKED_AT_C5
attn_qo_individual_tiles_11_12=PASS_DIAGNOSTIC_ONLY
attn_qo_next_family=dense_fixed_va_per_epoch_staged
production_promotion=false
default_route=direct_packed
```

## 1.3 Current corrected-v2 status 2026-07-18: `attn_kv` native-PM4 lifecycle and full-role correctness pass

Read this section before the historical lifecycle diagnoses below. Section 1.2 remains the performance and promotion
authority: this work closes the corrected stride-aware v2 `attn_kv` runtime/correctness blocker, but does not make the
generated route faster than direct packed or production eligible.

The corrected v2 bundle is
`/tmp/qk-attn-v2-stridefix-picklefix-bundle-20260718`, family
`5d862e43cbf924f5d8c9e239a4fbb3d0601517436b03707e9b6f3d5ebc10d38b`. It contains 20 distinct static-offset
K256 PROGRAMs for `attn_kv (512,1024,5120)`. Every retained binary has zero scratch and zero VGPR/SGPR spills.

The native-PM4 prefix-3 failure was localized more narrowly than the earlier AQL/PM4 description:

- The PM4 census accepted exact target submissions 0 and 1. Target 0 was complete before target 1 could submit.
- Target 1 faulted asynchronously. Target 2 never reached `AMDComputeQueue.exec` or `_submit`.
- The exception surfaced in target-2 `get_runtime`: `AMDProgram.__init__` uploads the next code object and synchronizes,
  which exposed the preceding target fault.
- Epoch 2 itself is not deterministically bad: the isolated ordinal-2 and ordered `[1,2]` probes pass.
- This evidence locates where the fault surfaced. It does not prove a driver-level root cause inside the prior target,
  KFD timeline dependency, or code-object allocation lifecycle.

Commit `8269edefe [qk] preconstruct frozen epoch runtimes` adds a default-off, harness-only discriminator. It calls the
existing `get_runtime("AMD", program)` cache in exact epoch order before target realization; no HIP launcher or runtime
core was added. It fails closed on key/binary/cache drift, duplicate or overlapping code allocations, invalid
entry/descriptor ranges, undrained device timelines, compute dispatch during preconstruction, or failure to reuse the
same runtime objects during scheduler dispatch. The relevant CPU suite passes 111 tests.

The guarded escalation now passes:

1. Full-family no-target canary: all 20 runtime/code objects construct in a fresh child invoked with `AMD_AQL=0`;
   target MMQ dispatch count remains zero, all lifecycle checks pass, the kernel-fault window is empty, and pre/post
   health pass. The artifact does not independently attest queue mode, and code uploads plus health adds still use the
   GPU.
2. Preconstructed prefix 3: three exact PM4 submissions and runtime-object reuse checks pass; 0/524,288 output
   mismatches, maximum absolute error `0.0003662109375`, clean fault window, healthy pre/post probes.
3. Preconstructed full 20: 20 exact PM4 submissions and runtime-object reuse checks pass; 0/524,288 output mismatches
   under combined `rtol=atol=0.003`, maximum absolute error `0.003173828125`, clean fault window, healthy pre/post
   probes.

The comparison authority is the same-session retained full-role producer bytes with the documented FP16 metadata
round trip. The producer diagnostic still reports oracle-rounding drift: 205 Q-value mismatches, 3,344 raw-scale
mismatches, 218 raw-sum mismatches, and after target-half rounding zero scale mismatches but one sum mismatch. Therefore
this is not whole-model llama parity. Timing is unmeasured for this corrected v2 family. Queue-mode admission,
remaining-role coverage, whole-model memory/correctness, decode regression, and matched performance remain open.

The compact durable composition is
`docs/qwen3-14b-prefill-attn-kv-v2-runtime-preconstruction-closeout-20260718.json`.

```text
corrected_v2_attn_kv_spills=0
corrected_v2_attn_kv_native_pm4_full20=PASS
corrected_v2_attn_kv_target_mismatches=0/524288
corrected_v2_attn_kv_performance_measured=false
whole_model_live_census_performed=false
production_promotion=false
default_route=direct_packed
```

## Historical 1.2 status 2026-07-18: superseded `ffn_gate_up` performance accounting

Do not use this section as current `ffn_gate_up` performance authority; §1.7 supersedes it.

All four generated Q4 roles have exact full-role correctness/resource/health evidence. The earlier `ffn_gate_up`
selection was based on a comparison that §1.7 now proves non-comparable:

- `0.277502 ms` and `9.351988 ms` both cover bounded `128x128x256`, but their timed regions differ.
- Complete-role generated observations are `42.8822620306164 ms` with synchronized AQL,
  `27.07601070869714 ms` with queued AQL plus a final synchronization, and `41.86103114625439 ms` with PM4.
- All exact runs pass 8,912,896 values with zero mismatches and maximum absolute error `0.00341796875`; no matched
  complete-role direct observation exists, so these measurements establish no performance decision.

The frozen route remains valuable as a correctness-qualified integration diagnostic. The immutable six-row research
artifact remains unchanged for identity/binding/census work, but it is not a performance-qualified policy and its
`ffn_gate_up` row is not a performance selection.

```text
phase_3_measurements_complete=false
selected_generated_roles=
integration_only_policy_candidate_roles=ffn_gate_up
performance_not_evaluated_generated_roles=ffn_gate_up,attn_kv,attn_qo,ffn_down
performance_qualified_policy=false
production_promotion=false
default_route=direct_packed
```

Authoritative phase state:

| Phase | State |
|---|---|
| 3 — Q4_K role completion | Measurement closeout complete; no generated performance winner. |
| 4 — Q6_K role completion | Complete for the declared two-row direct-packed fallback strategy. |
| 5 — Central candidate policy | Integration machinery complete; performance-qualified policy open. |
| 6 — Mixed-route 14B integration | Open; the current six-row route is diagnostic only. |
| 7 — Parity and promotion | Open and ineligible until a performance-qualified policy passes Phase 6. |

Continue to reuse tinygrad's five-buffer emitter, frozen artifacts, and PM4/AQL runtime. Do not add a HIP launcher.
The two useful tracks are now: diagnose the live integration seam, and search for a candidate that wins under
comparable full-role timing. Only the latter can supply a promotion policy.

## Historical 1.1 status 2026-07-18: all Q4 roles resolved

This section predates the full-role performance-accounting correction in §1.2.

The `attn_kv` Q4 role no longer owns a correctness, resource, health, or repeated-dispatch blocker:

- `d3c605890` stages every changing epoch input—Q4, Q8 values, scales, and sums—through persistent fixed VAs. PM4
  prefix 3 and full 20 epochs pass.
- The AQL prefix-3 result before the queue fix remains `BLOCKED` after two completed epochs.
- `6216e9e4e` fixes AQL publication ordering by writing the packet header last. Post-fix AQL prefix 3 and full 20
  epochs pass.
- Both full-20 runs reuse the frozen PROGRAM without compilation or fallback. They use fixed-VA GPU-SDMA staging,
  one in-place FP32 accumulator, no intermediate readback or external accumulation add, `VGPR=256`, `LDS=57,856 B`,
  and zero scratch. PM4 and AQL each compare 524,288 final values with zero mismatches and maximum absolute error
  `0.002685546875`; their fault windows are empty and pre/post health canaries pass.

The exact five raw artifacts and their hashes are composed in
`docs/qwen3-14b-prefill-attn-kv-role-closeout-20260718.json`. The historical AQL failure must be described precisely:
the retained `sq_intr type 2` evidence is an SQ memory violation followed by a gfxhub page fault and reset sequence.
The log does not establish an instruction-page or instruction-fetch fault.

Correctness does not promote this candidate. The only full-role generated timings are cold isolated one-sample
observations: AQL `19.601495820097625 ms` and PM4 `74.58446017699316 ms`. Neither beats the retained direct-packed
`attn_kv` baseline of `7.89 ms`. Do not call either sample a stable median, matched benchmark, or statistical result.
The honest decision is:

```text
attn_kv_q4_generated_candidate=REJECTED
attn_kv_q4_role_resolution=MEASURED_DIRECT_PACKED_FALLBACK
attn_kv_q4_selected_route=direct_packed
attn_kv_q4_policy_row_changed=false
production_promotion=false
```

The same shared N5120 binary (`e66d0b8c…`) now has exact role evidence for the other two Q4 rows:

- `attn_qo`: PM4 prefix 1, prefix 3, and full 20 pass. The full result has 0/2,621,440 mismatches, maximum absolute
  error `0.00341796875`, clean logs, healthy pre/post canaries, and a single cold isolated timing of
  `76.49636000860482 ms` versus the retained direct-packed `11.41 ms`.
- `ffn_down`: PM4 prefix 1, prefix 3, and full 68 pass. The donor remains explicitly `attn_qo`; the distinct
  `ffn_down` execution fixture is validated for 68 epochs. The full result has 0/2,621,440 mismatches, maximum
  absolute error `0.01123046875` within tolerance, clean logs, healthy pre/post canaries, and a single cold isolated
  timing of `114.26727805519477 ms` versus the retained direct-packed `11.76 ms`.

Neither timing is a matched, warmed, repeated, or statistical claim. Both generated candidates are rejected; their
existing direct-packed policy rows remain unchanged. The all-role composition is
`docs/qwen3-14b-prefill-q4-role-closeout-20260718.json`.

Authoritative phase state:

| Phase | State |
|---|---|
| 3 — Q4_K role completion | Historical conclusion, corrected by §1.2: all four generated roles were measured, but `ffn_gate_up` was not a comparable performance winner. |
| 4 — Q6_K role completion | Complete for the declared two-row direct-packed fallback strategy. |
| 5 — Central candidate policy | Complete at the default-off research boundary; the immutable policy remains unchanged. |
| 6 — Mixed-route 14B integration | Open. |
| 7 — Parity and promotion | Open. |

Phase 6 integration diagnostics may use the unchanged immutable artifact, but the current performance work must search
for a faster generated candidate. No HIP launcher or model/GPU-name branch is justified.

There is no project progress percentage. The phase ledger in
`docs/qwen3-14b-generated-prefill-completion-scope-20260714.md` is the authority.

## Historical 1.0 status 2026-07-18: repeated `attn_kv` dispatch blocked Phase 3

This section predates fixed staging for every input, the PM4 full-20 pass, AQL header-last publication, the AQL
full-20 pass, and the measured direct-packed fallback decision in §1.1.

Read this section first. It supersedes §0.9 and the chronological diagnostics below.

The project is not waiting on spill reduction, Q6 fallback tooling, policy serialization, a HIP launcher, or a live
research adapter:

- Both exact direct-packed Q6 fallback rows are qualified and timed:
  `docs/qwen3-14b-prefill-q6-attn-kv-qualification-20260718.json` and
  `docs/qwen3-14b-prefill-q6-ffn-down-qualification-20260718.json`.
- `docs/qwen3-14b-prefill-six-row-research-policy-20260718.json` is the retained immutable six-row policy. It is
  `research_only` with `production_promotion=false`.
- The exact policy selector, runtime attachments, frozen candidate execution, declared direct-packed fallbacks, and
  actual execution-census events are connected through the explicit/default-off research route. Unknown workload,
  missing bundle/program authority, attachment drift, candidate identity drift, and candidate runtime failure all
  fail closed.
- Exact frozen PROGRAM bundles exist for `attn_kv` `(512,1024,256)` and the shared N5120 geometry
  `(512,5120,256)`. The strict harness reuses the authoritative frozen-role binding and tinygrad's existing
  `get_runtime("AMD", program)` PM4/AQL paths. It does not emit, recompile, or launch through HIP.
- Shared N5120 PROGRAM reuse does not relabel its donor fixture: artifact/donor role and execution role are recorded
  separately, and an `ffn_down` execution gets its own deterministic 68-epoch fixture.

The active blocker is now narrow and retained:

- `docs/qwen3-14b-prefill-attn-kv-fresh-epoch-isolation-20260718.json` records three PM4 fresh-process probes over the
  same frozen artifact. Epochs 0, 1, and 2 each pass exactly one target dispatch with zero mismatches across 524,288
  values, clean kernel-fault windows, and healthy pre/post canaries. Their maximum absolute errors are respectively
  `6.103515625e-5`, `6.103515625e-5`, and `1.220703125e-4`.
- `docs/qwen3-14b-prefill-attn-kv-pm4-aql-differential-20260718.json` records the counterevidence. PM4 and AQL both
  pass a one-epoch prefix. In a three-epoch same-process prefix, both complete two epochs and fault entering the third
  dispatch, followed by SQC memory-violation, gfxhub page-fault, MES queue-removal failure, and reset evidence.
- Because both launch modes fail at the same repeated-dispatch boundary while epochs 0/1/2 pass separately, this is
  not evidence of one deterministic bad epoch, an epoch-2 numerical defect, or a PM4-only packet bug. The precise
  shared runtime/kernel/resource-state cause remains unproven. Do not present the inference as a root-cause fix.

Authoritative phase state:

| Phase | State |
|---|---|
| 3 — Q4_K role completion | Historical state: `ffn_gate_up` correctness-qualified but not yet comparably performance-measured; `attn_kv` repeated-dispatch blocked; `attn_qo` and Q4 `ffn_down` still required full qualification. |
| 4 — Q6_K role completion | Complete for the declared two-row direct-packed fallback strategy. |
| 5 — Central candidate policy | Complete at the default-off research boundary: immutable six-row policy, exact binding, and execution-census implementation exist. |
| 6 — Mixed-route 14B integration | Open: no successful whole-model policy execution/census, memory reconciliation, health, or decode proof. |
| 7 — Parity and promotion | Open: no matched multi-context promotion run for this policy. |

```text
immutable_six_row_policy=true
exact_research_route_binding_implemented=true
actual_execution_census_implemented=true
exact_research_route_default_off=true
whole_model_live_census_performed=false
production_promotion=false
production_dispatch_changed=false
default_route=direct_packed
```

Next work should reuse the frozen PROGRAM and existing tinygrad runtime/harness boundaries. Audit the repeated-dispatch
lifecycle CPU-side, implement only an evidenced lifecycle correction or fail-closed safe execution strategy, then
repeat the bounded `1 -> 3` health/correctness escalation before any longer run. Only after `attn_kv` is stable should
the remaining Q4 roles and Phase 6/7 gates proceed. Do not add a HIP launcher, regenerate the frozen binary merely to
compare launchers, revive the spill-ordering dead end, or claim promotion from fresh single-epoch passes.

There is no project progress percentage. The phase ledger in
`docs/qwen3-14b-generated-prefill-completion-scope-20260714.md` is the authority.

## Historical 0.9 status 2026-07-18: one exact role evidence-ready

This section predates the Q6 fallback qualifications, immutable six-row policy, live default-off research
binding/census, frozen `attn_kv` artifact, and repeated-dispatch differential recorded in §1.0.

Read this section first. The current pushed implementation/evidence head before this documentation update is
`e4a940384`. The old spill, MMU-fault, executable-lifecycle, and “do not run 20 epochs” statements in the chronological
sections below are superseded.

At this checkpoint, the exact Q4_K/Q8_1 `ffn_gate_up` role at `M=512, N=17408, K=5120` appeared evidence-ready to
implement a research opt-in. The corrected accounting in §1.7 blocks that conclusion without asserting a win or loss:

- `docs/qwen3-14b-prefill-r5-geometry-20260718.json` is a retained, same-session
  `q4k-q8-1-mmq-r5-geometry-search.v2` report. Its emitted full-grid row passed all 16,384 outputs with zero
  mismatches and maximum absolute error `3.0517578125e-5`. Three measured rounds gave candidate median/min
  `0.277502/0.269477 ms` for one K=256 epoch. Direct packed also used the bounded `128x128x256` workload, but its
  `9.351988/9.349303 ms` timed region included preparation/allocation/readback absent from the generated region, so
  the historical `34.694x` label is not a speedup claim. The row still retains
  useful source/binary hashes, native resources (`VGPR=256`, `LDS=57,856 B`, `scratch=0`), and correctness evidence.
- `docs/qwen3-14b-prefill-target-frozen-20epoch-pm4-20260718.json` proves the exact full role with one frozen binary,
  20 same-process PM4 launches, in-kernel FP32 accumulation, persistent/preloaded inputs, stable fixed-VA GPU-SDMA
  metadata, and no intermediate readback, external add, recompile, or fallback. All 8,912,896 final outputs match
  under the declared tolerance with zero mismatches; maximum absolute error is `0.00341796875`. Health and fault
  checks pass. Complete-role generated observations are `41.861031 ms` PM4, `42.882262 ms` synchronized AQL, and
  `27.076011 ms` queued AQL; no matched complete-role direct-packed observation was retained.
- `docs/target-epoch-safe-all-attested-20260718.json` independently proves every K=256 epoch in fresh processes:
  zero mismatches across 178,257,920 individually checked values, maximum absolute error `1.220703125e-4`, pinned
  fixture/source identity, and clean per-epoch health.
- `docs/qwen3-14b-prefill-mmq-one-role-evidence-20260718.json` composes the retained R5, strict full-role, and
  independent epoch evidence. Its top verdict is
  `BLOCKED_UNTIL_COOPERATIVE_TILE_WIN`; machine-search R6 reports
  `BLOCKED_NO_COMPARABLE_FULL_ROLE_WIN` and machine-search R7 reports `PASS_TARGET_ROLE_REDUCTION`.
  That R7 label is source-component reduction for this role, not project Phase 7 parity/promotion.

The joined report is deliberately explicit:

```text
one_role_opt_in_eligible=false
research_opt_in_implementation_eligible=true
route_binding_implemented=false
live_route_census_performed=false
promotion_eligible=false
production_dispatch_changed=false
default_route=direct_packed
```

No HIP launcher was built or needed. The work reused tinygrad's existing five-buffer emitter/harness and its PM4 and
AQL launch paths over the same frozen artifact. The frozen static audit, exact PM4/AQL 1- and 3-prefix differential,
strict 20-launch PM4 proof, and independent all-epoch proof all passed.

This advances project Phase 3 for one of four Q4 roles; it does not close Phase 3. Remaining production order:

1. Implement the one-role research opt-in through the live generic registry/admission/runtime path, then run a live
   negative-role and no-hidden-fallback census. The manifest row remains `research_descriptor_only` and unbound.
2. Qualify the remaining Q4 roles: `attn_qo`, Q4 `ffn_down`, and `attn_kv`, including full-role correctness,
   resource/health evidence, and whole-primitive timing.
3. Qualify and policy the two direct-packed Q6 fallbacks (`attn_kv` and Q6 `ffn_down`) with the existing Q6 tooling.
   Pursue a faster generated Q6 route only if measured Q6 share requires it.
4. Emit one immutable six-row policy with exact identities, admission, fail-closed drift/unknown handling, and
   one-change direct-packed rollback.
5. Run project Phase 6 whole-model mixed-route memory reconciliation, route census, correctness/output/token checks,
   GPU health, and decode correctness/performance regression.
6. Run project Phase 7 matched llama/tinygrad contexts 512/1024/2048/4096 in at least three alternating pinned
   sessions and apply the statistical promotion gates. Use Boltbeam attribution only on that exact revision/session
   if a gate misses.
7. Only after all gates pass: production promotion, then autoscan.

## Historical 0.5 audit status 2026-07-17 (head `7b863aaec`)

This section is superseded by §0.9 and retained for chronology. At that time, the repository head was
`7b863aaec` (`[test][qk] honor ProgramInfo global buffer indices`), twelve commits ahead of
`origin/master`; the worktree is clean. This includes the allocator lease fix at `23eaf693b` and the new
fail-closed harness under `extra/qk/mmq_llama_five_buffer_gpu_harness.py`.

The corrected phase-major graph and allocator lease regression coverage are structurally healthy:

- `PYTHONPATH=. PYTHONHASHSEED=0 pytest -q test/unit/test_mmq_llama_five_buffer_full_kernel.py`: **13 passed**.
- The focused rematerialization/end-liveness/span suites: **23 passed**.
- `test/unit/test_amd_isa_wmma.py`: **36 passed, 4 failed**. The four 16-subtile multi-output failures are the
  same spill-free-gate failures reproduced on the pre-change `423f6ff83` baseline; they are historical and not a
  regression from `23eaf693b`.
- The new fail-closed GPU harness tests: **3 passed**. A matched-environment run reached a real AMD dispatch but
  synchronization reported `MMU fault: 0x7F8827C51000 | NotPresent=1`; no numerical verdict was captured. The harness
  preserves this structured blocker rather than treating it as a pass.

The full K=256 `to_program` path now **emits successfully** under the matched environment
(`PYTHONHASHSEED=0 REGALLOC_ADDR_REMAT=1 REGALLOC_END_NO_SOURCE_LIVE=1`). Graph/codegen selection reports
`10,101` UOps and a pre-allocation peak of `312` live virtuals at UOp `5938`; the full allocator run then reports
`REGALLOC_SPILLS: count=0 stack_size=0`, and `compile_llama_five_buffer_full_kernel` returns
`emitted=True program=True blocker=''`. The CPU-only compile takes several minutes in the long register-selection
pass, so this is a valid zero-spill emission result but not a GPU correctness or timing result. The `23eaf693b`
change reserves the serialized high A/B fragment lease and has a focused unit test; the end-to-end run above is the
evidence for the full-grid gate.

K=512 exact FP32 state-carry and phase-major lifecycle tests pass structurally. Its exact compile fails before
regalloc in `_register_stage_leases`: 128 independent C roots request low-accumulator leases `[v8, v1032)`, over
the 256-VGPR file, because epoch-aware A/B fragment provenance is not yet proven. K=256 has since been launched on
the AMD device, but synchronization reported an MMU `NotPresent=1` fault before output comparison. No shape has been
numerically compared with llama or performance-measured; the dispatch fault is the next acceptance gate.

Date: 2026-07-16  
Repository: `/home/ubuntu/tinygrad-arkey`  
Branch: `master`  
Handoff revision: `412d7998f` (`[amd] recover propagated progressive C roots`)  
Remote state at handoff: `master == origin/master`, clean worktree

## 0. Status update 2026-07-17: bounded emission gate CLOSED

The §15 assignment (spill-free bounded emission) is complete and pushed. Sections 1, 4 and 7 below describe the
pre-`5982d83c1` state and are retained for history; read this section first.

Accepted commits, oldest to newest:

- `8a8ffc6ae [amd] select native fp16 multiply for half2 metadata`
- `00a79b247 [codegen] fail fast on ops that reach regalloc unselected`
- `5982d83c1 [amd] guard chain-head A/B loads on the previous chain release`

The 112 spills were **two** stacked blockers, which is why the single-blocker framing in §7 misleads:

1. **A/B chain-head lifetime.** A multi-output-tile kernel has one chain per subtile and every chain reloads the same
   physical high A/B pair (`ab_key = "wmma_ab"`, `amd.py`). Accumulate tiles WAR-guard that shared run
   (`dep = (prev.src[0],)`); a chain head has no prior tile and takes only `_wmma_carrier_order_deps(tile.src[2])`, so
   eight heads opened simultaneously onto one capacity-1 run. `_serialize_progressive_c_drains` already ordered each
   head WMMA after the previous chain's release frontier, but the head's `DS_LOAD_B128` operands carried no such edge
   and floated ahead of it: the ordering edge was on the wrong node. Guarding the loads closed it.
2. **Missing typed half-MUL selection.** isel selected `MUL` for float32 and ints only, so the half2 metadata
   recurrence's genuine fp16 multiply survived unselected. Its children lowered to raw-bit machine ops, leaving a
   mixed `ushort x half` node. Post-isel this *looks* like malformed oracle algebra; it is not. Both source graphs are
   clean (bounded 3,239 nodes / 346 MULs, full-grid 7,802 / 966, zero mixed-dtype). Do not "fix" it upstream.

Exact bounded oracle, matched environment (`PYTHONHASHSEED=0 REGALLOC_ADDR_REMAT=1 REGALLOC_END_NO_SOURCE_LIVE=1`):

| metric | at `412d7998f` | at `5982d83c1` |
|---|---:|---:|
| spill requests | 112 (`v0` at 2580) | **0** |
| stack | 448 B | **0** |
| emitted instructions | none (fail-closed) | **8,287, all encodable** |
| `v_wmma_i32_16x16x16_iu8` | 128 | **128** |
| `ds_load_b128` | 256 | **256** |
| `v_mul_f16_e32` | 0 (unselected) | **12** |
| post-selection UOps / peak live | 9,481 / 158 @ 5908 | **9,481 / 158 @ 5908** |

No scratch or buffer traffic. The bounded probe still sets no full-grid claim.

Two traps recorded so they are not re-entered:

- **`regalloc_rewrite` indexes `ctx.uops` BY POSITION** via `i = next(ctx.idx)`. Any op with no ISA rule is never
  visited, shifts every later index by one, and surfaces as a `KeyError` on an unrelated vreg — the allocator blaming
  itself for a backend gap. `00a79b247` makes this fail fast and name the op. If you see a mystery `KeyError` in
  regalloc, suspect selection first.
- **gfx11 VOP f16 literals live in the low 16 bits.** `_vop2_f`'s `float()` path encodes an fp32 pattern
  (`0.3f -> -0.0027h`). Inline constants survive it and everything still encodes, so "the program encodes" does not
  prove literal correctness. `_vop2_h` owns the fp16 path.

### Next owning problem: full-grid is NOT gated on spills

**Superseded by §0.1: both SPEC gates are now closed and full-grid reaches regalloc. The GEP/AFTER diagnosis below is
wrong about the verifier and is retained only as history.** The spill gate was only one of the listed full-grid
blockers. §13.1 was still closed at that point, and its remaining blockers were
graph/spec-shaped: `compile_llama_five_buffer_full_kernel(build_llama_five_buffer_full_kernel(128,128,256))` fails
**before** register allocation, in SPEC type verification. Verify each `BLOCKER` line independently rather than
assuming the resource one was the last.

Two of these have now been walked. Reproduce with:

```bash
env PYTHONHASHSEED=0 REGALLOC_ADDR_REMAT=1 REGALLOC_END_NO_SOURCE_LIVE=1 python3 -c "
from extra.qk.mmq_llama_five_buffer_full_kernel import (build_llama_five_buffer_full_kernel,
                                                        compile_llama_five_buffer_full_kernel)
compile_llama_five_buffer_full_kernel(build_llama_five_buffer_full_kernel(128,128,256))"
```

**Cleared at `e5efb9707`** — `AFTER(ADD, STORE)`. Do not re-diagnose this:

```text
UOp verification failed at 3842 on Ops.AFTER dtypes.float 2
  [(Ops.ADD, dtypes.float, None), (Ops.STORE, dtypes.void, None)]
```

The source sink is legal (7,802 nodes; `AFTER` src0 is only `DEFINE_LOCAL`/`INDEX`/`BITCAST`/`STACK`; zero
`AFTER(ADD, STORE)`). The illegal node is manufactured *during codegen rewriting*: a bare `STACK(float.vec(8))` is
spec-legal as written but expands away, dropping its effect order onto the scalar FP32 update. Fix was the no-op
typed BITCAST carrier the bounded release already documents. **Lesson: the source graph being clean does not mean the
verified graph is; check post-rewrite before blaming the oracle.**

**Current exact victim** — a different verifier blocker, further down:

```text
UOp verification failed at 3327 on Ops.GEP dtypes.float 1
  [(Ops.BITCAST, dtypes.float.vec(8), None)] (0,)
```

Known so far, so it is not re-derived:

- That node **passes `spec_program` in isolation** (`GEP(BITCAST(STACK(float.vec(8))))`, arg `(0,)` -> `True`). So the
  real node differs in a field `type_verify`'s message does not print. `shape` is the first suspect: `spec.py:244`
  (`False if x.dtype.count > 1 and (x.dtype.count,) != x.shape`) is shape-sensitive.
- Note that rule would fire on the vec(8) `BITCAST`, not the `GEP`, so the reported node may not be the guilty one.
- Method: build the sink, run `full_rewrite_to_sink` with `SPEC=0`, walk the real post-rewrite nodes, and diff the
  failing node against the synthetic one that passes. Do not reason from the error string alone.
- **Do not widen `spec_shared`.** It is shared across backends, and the bounded probe proves the graph-side idiom
  works. Prefer a graph fix at the producer.

Unchanged and still true: the four 16-subtile AMD tests, two `test_current_prefill_execution_adapter` rows, and
`test_q4k_wmma_tiled_no_hand_scan_is_clean` fail identically at `412d7998f` and at `5982d83c1`. They are historical,
not regressions. Do not xfail them.

## 0.1 Status update 2026-07-17 (later): both SPEC gates CLOSED, full-grid reaches regalloc

Accepted and pushed: `4b153b8e9 [qk][mmq] order full-grid stores through the pointer, not a no-op value bitcast`.

The §13.1 verifier blocker is closed. It was **one** defect seen through two different verifiers, not two:

- `_full_grid_sink` built `value = UOp(Ops.BITCAST, store.src[1].dtype, (store.src[1],))` — a BITCAST to the dtype it
  already has. Codegen folds that no-op away and the `.after()` it carried lands on the scalar FP32 update, so
  `spec_program` rejected `AFTER(ADD, STORE)` at `codegen/__init__.py:196`.
- `e5efb9707` read that as a missing movement carrier and wrapped the accumulator vectors in a typed BITCAST. That
  stopped the AFTER only by keeping a `vec(8)` alive across the writeback, which blocks the `GEP(STACK, i) -> src[i]`
  fold and left 64 `GEP(BITCAST(...))` in the **source** sink — rejected by `spec_tensor` at `codegen/__init__.py:77`.

**The two error strings came from two different `type_verify` call sites.** `spec_tensor` has no general GEP rule (only
GEP over `WMMA`/`SHAPED_WMMA`/`LOAD`, `spec.py:131,143,147`); the general rule at `spec.py:250` is `spec_program`-only.
The handoff's "that node passes `spec_program` in isolation" was true and irrelevant — it was never checked against
`spec_tensor`. **Always confirm WHICH verifier failed before diagnosing the node.** `shape` was never involved.

Fix: revert the accumulator carrier and drop the value-side edge. The pointer's `INDEX` is a real movement value and
already totally orders the 64 stores. No arithmetic, rounding, or ABI change.

### Current exact blocker: full-grid register pressure (genuinely resource-shaped)

Matched env (`PYTHONHASHSEED=0 REGALLOC_ADDR_REMAT=1 REGALLOC_END_NO_SOURCE_LIVE=1` + `REGALLOC_DEBUG`/`_PRESSURE`/`_SPILLS`):

```text
REGALLOC_DEBUG: 13547 uops, PEAK 452 live vregs @ uop 5147, pool=255
REGALLOC_PRESSURE: spill_request=v0 at=3014 pool=255
REGALLOC_SPILLS: count=96 stack_size=381
```

Spill victims: 80 `V_CVT_I2F`, 11 `V_MUL`, 4 `V_ADD`, 1 `V_CMPLT_I`. Peak composition: 184 `V_CVT_I2F`, 129 `V_IMUL`,
64 `GLOBAL_LOAD`, then a thin tail. Bounded oracle is unregressed at `4b153b8e9`: 9,481 UOps, peak 158 @ 5908, 0 spills,
0 stack.

The decisive measurement — **the full-grid math is identical to the bounded probe**:

| metric | bounded | full-grid (128,128,256) |
|---|---:|---:|
| `v_wmma_i32_16x16x16_iu8` | 128 | 128 |
| `ds_load_b128` | 256 | 256 |
| total UOps | 9,481 | 13,547 |
| peak live vregs | **158** | **452** |
| `V_CVT_I2F` live at peak | **8** | **184** |

Same WMMA and LDS counts, same 512 total `V_CVT_I2F`. Full-grid adds 4,066 UOps and 64 `GLOBAL_STORE` (at 11,568..13,543).
So the extra 294 live virtuals are **not** new math — they come from the writeback seam.

All 184 drains live at the peak are defined in UOps 2,841..5,321 and every one is consumed by a `V_MUL` at 5,930..8,935.
The `V_MUL`/`V_ADD` pairs there are the exact serial recurrence (`previous + scale*C`), consumed one term at a time, so
each drain is held for thousands of UOps waiting its turn. WMMAs are not all hoisted first (they span 2,818..13,089) —
the pile-up is local to that region.

**REFUTED at `8124ddaa9` — see §0.2. The real cause was a missing ordering edge in the full-grid producer, and
`_accumulator_vectors` is not a contributor at all. Retained only as history; do not revive.** Leading hypothesis: `_accumulator_vectors` transposes eight scalar WMMA
lanes across eight oracle subtiles (vector `e` = `STACK(lane_0..lane_7)` with `subtile -> e`). That transposition
requires all eight subtiles' drains co-live to assemble any one vector, whereas the bounded probe drains each subtile
independently and keeps only 8 `V_CVT_I2F` live. Note 184 != 64, so the transposition alone does not obviously account
for the whole peak — do not assume it is the only contributor.

Next probe: map each of the 184 peak-live `V_CVT_I2F` back to its subtile/lane and its consuming `V_MUL`, and check
whether the co-liveness is forced by the transposition or is a scheduling choice `pressure_schedule` could undo.

**Rails for this blocker:** the recurrence order and rounding are fixed authority — do not reassociate
`(previous + scale*C) + bias` or introduce FMA/MULACC to shorten the chain. Do not enable AMD scratch.

### Measurement caveat (new, important)

Full-grid spill/peak numbers are **environment-sensitive and load-sensitive**:

- with `REGALLOC_DEBUG_PRESSURE=1`: 452 @ 5147 / 96 spills / 381 B — reproduced 2/2.
- without it: 465 @ 5355 / 95 spills / 377 B — reproduced 3/3.
- one early run under concurrent load reported 444 @ 5098 / 85 spills. Same env as the 452 run, different result.

Each environment is self-consistent when the machine is otherwise idle, but the load-dependent outlier is unexplained
and was not chased. **Run the full-grid reproducer on an idle machine and compare only matched environments.** The
bounded oracle is deterministic (9,481 / 158 @ 5908 / 0) and is the trustworthy comparator.

## 0.2 Status update 2026-07-17 (later still): drain ordering applied, blocker moves to the writeback

Accepted and pushed: `8124ddaa9 [qk][mmq] order full-grid WMMAs behind the preceding lane drain`.

§0.1's hypothesis (`_accumulator_vectors` transposition) was **wrong and is now refuted twice over**. Do not revive it:

1. `_accumulator_vectors` groups by subtile, **not across** subtiles: `vector[e] = STACK([lane[l].substitute(subtile -> e)
   for l in range(8)])` substitutes the *same* `e` into every lane. Assembling one vector needs one subtile's eight
   **lanes**, never all eight subtiles. Measured: `vector[e]`'s weakint consts are `[e]` only.
2. It is structurally inert regardless. `build_wmma_writeback` immediately GEPs each lane back out and
   `GEP(STACK, i) -> src[i]` folds, so the STACK never reaches the machine graph. Measured: `GEP(vector[3], 5)` is
   already `Ops.ADD` and is *identical* to `lanes[5].substitute(subtile -> 3)`.

**The real cause was a missing ordering edge in the full-grid producer.** The oracle keeps the integer WMMA chain and
the eight FP32 lane chains as separate algebraic dependencies, so a legal schedule issues every WMMA before consuming
any C lane. `_bounded_accumulator_drain`'s docstring in `extra/qk/mmq_llama_full_kernel.py` has named this exact
failure all along — the bounded probe has carried the edge for a long time and the full-grid seam simply never got it.

Two facts that discriminated **forced dataflow** from **scheduling choice**, and are worth reusing as a method:

- `pressure_schedule` provably *could not* fix it: it is block-local (`regalloc.py:103-111` splits blocks at
  `BARRIER`/`RANGE`/`END`), the drains were defined in blocks `(2685,4217)`/`(5070,5474)` while their `V_MUL` consumers
  lived in blocks `(5915,6060)`..`(8845,8990)`, with `BARRIER`s between. Barrier indices are identical in its input and
  output — it permutes only *within* blocks and had no legal hoist available.
- But it was not dataflow-forced either, proved by construction: adding the edge changed nothing but the order.

The repair extracts `order_wmma_behind_lane_drain(epochs, tag_prefix)` from `_bounded_accumulator_drain` and applies it
in `_full_grid_sink` before `_accumulator_vectors` substitutes, so edges replicate into all eight subtiles.

| metric | before | after |
|---|---:|---:|
| `V_CVT_I2F` live at peak | 184 | **64** |
| peak live vregs | 465 | **372** |
| spills | 95 | **54** |
| stack | 377 B | **213 B** |

Selected math unchanged (128 `V_WMMA_I8`, 256 `DS_LOAD_B128`, 512 `V_CVT_I2F`, 640 `V_MUL`, 960 `V_ADD`, 64
`GLOBAL_STORE` — identical to pre-repair). Bounded oracle byte-identical (9,481 / 158 @ 5908 / 0 / 0). All 54 focused
tests pass. `test_full_grid_orders_each_wmma_behind_the_preceding_lane_drain` locks the invariant and was confirmed to
fail with the producer fix stashed.

### Current exact blocker: writeback address math and load lifetimes

54 spills still fail closed, so §13.1 stays open, and **the owning problem has moved**. Peak composition inverts:

```text
REGALLOC_DEBUG: 13659 uops, PEAK 372 live vregs @ uop 5049
  132  V_IMUL
   64  GLOBAL_LOAD
   64  V_ADD
   64  V_CVT_I2F
```

`V_IMUL` (132) + `GLOBAL_LOAD` (64) now dominate, not the drains. Spill victims are still mostly `V_CVT_I2F` (the
residual 64 = 8 subtiles x 8 lanes), plus one long-lived `V_CMPLT_I` (v315, range 563..13656 — a whole-program
predicate worth its own look).

Two candidate next moves, in order:

1. **The 132 `V_IMUL` and 64 `GLOBAL_LOAD` at the peak** are the writeback/grid address math. This is the largest
   contributor and is untouched by any repair so far. Start here; `a293391b0 [codegen] localize constrained leases at
   consumers` is the precedent for shrinking def-to-use distance on address carriers.
2. **The residual 64 drains are cross-subtile co-liveness** the per-subtile edge does not cover. Ordering subtile `e`'s
   WMMAs after subtile `e-1`'s updates needs edges injected *post*-substitution. **Unverified and risky:**
   `amd.py:1883-1886` explicitly warns that ordering on FP32 update nodes "can compose into a cycle". Extending
   `_serialize_progressive_c_drains` from the CVT release frontier to the update frontier is the natural candidate —
   note that serializer **does fire** on full-grid (128 machine WMMAs, 64 selected roots, 63 edges applied); it orders
   chain heads on the CVT frontier, which frees the C lease but does not force the FP32 consumer. It is insufficient,
   not absent. Do not re-diagnose it as a bail-out.

Rails unchanged: no reassociation of `(previous + scale*C) + bias`, no FMA/MULACC, no rounding-boundary move, no AMD
scratch, no `spec_shared` widening.

## 0.3 Status update 2026-07-17 (later still): cross-subtile serialization, 9 spills left

Accepted and pushed: `cd989ad0e [qk][mmq] serialize full-grid subtiles behind the preceding subtile's drains`.

§0.2's residual-64 note was right: the binding pressure after the intra-subtile fix was **cross-subtile** C-carrier
co-liveness. The eight subtiles share no integer WMMA node (the subtile index only touches the Q4/A operand), so a
legal schedule still issued all eight WMMA chains before consuming any subtile's drains — 8 subtiles x 8 lanes co-live.
`order_wmma_behind_lane_drain` runs *before* the subtile substitution and cannot see across subtiles.

Fix: `_accumulator_vectors` now takes the epoch-0/group-0 chain head and, per substituted element `e>0`, orders `e`'s
chain head behind element `e-1`'s eight drains. The chain head reaches every lane of its subtile (second WMMA takes
first as its accumulator input, every lane chains through `second.gep(i)`), so one edge serializes the whole subtile.
Strictly increasing element index => acyclic (verified by a cycle-detecting toposort in review).

| metric | before | after |
|---|---:|---:|
| `V_CVT_I2F` live at peak | 64 | **8** |
| peak live vregs | 372 | **273** |
| spills | 54 | **9** |
| stack | 213 B | **29 B** |

`V_CVT_I2F` at peak (8) now equals the bounded probe. Selected math unchanged (128 `V_WMMA_I8`, 256 `DS_LOAD_B128`,
512 `V_CVT_I2F`, 640 `V_MUL`, 960 `V_ADD`, 64 `GLOBAL_STORE`); op-count diff vs parent is only +21 `AFTER` (7x3) and
+9 `BITCAST`. Bounded oracle byte-identical (9,481 / 158 @ 5908 / 0 / 0). A Fable review audited math-neutrality, cycle
safety, per-subtile reach, and the substitution-target liveness — all SAFE with direct evidence.

**Method note that keeps paying off:** for these three full-grid repairs, the diagnosis that held was always (a) show the
would-be fixing layer *cannot legally act* — `pressure_schedule` is block-local (`regalloc.py:103-111`) — and (b) prove
non-forcedness *by construction* by adding the edge and measuring. Reason from the string and you get the 132-`V_IMUL`
red herring; the `V_IMUL` are fully rematerialized by `REGALLOC_ADDR_REMAT` and cost 0 spills.

### Current exact blocker: residual 9 spills, mixed mechanism

Matched env: `PEAK 273 @ 4384`, 9 spills / 29 B. The 9 victims are **3 `V_CVT_I2F`, 2 `GLOBAL_LOAD`, 2 `V_MUL`, 1
`V_ADD`, 1 `V_CMPLT_I`** — no longer a single class, so no single edge will close it. Peak composition is now `128
V_IMUL` (remat, 0 spills) + `64 GLOBAL_LOAD` + a thin tail; real physical demand is ~264-273 against a 255 pool, i.e.
only ~9-18 over. Candidate probes, unverified:

- **`v315` (`V_CMPLT_I`, range 563..13656)** is a whole-program integer predicate reused across every epoch/load, so it
  is live end to end and independent of the drains (it survived both drain fixes). It originates in the cooperative DS4
  panel-staging lane map, not the writeback. A recompute-at-uses localization (analogous to the address remat, cf.
  `a293391b0`) is the natural fix for this one register. Low individual payoff but clean.
- **The 2 `GLOBAL_LOAD` + 2 `V_MUL` + 1 `V_ADD`** sit near the peak at ~4384; inspect their ranges with
  `REGALLOC_DEBUG_DETAIL` and check whether a localization or a small ordering nudge frees them.
- **The 3 residual `V_CVT_I2F`** are the tail of the drain serialization; a milder cross-subtile edge (order behind
  `e-1`'s CVT frontier rather than its full accumulator — the `prior_drains` knob in `_accumulator_vectors`) might trade
  the last few without over-constraining. Try this only if the cheaper localizations above do not reach zero.

Because the peak is only ~9-18 over pool and the victims are mixed, the endgame is likely two or three small
localizations, not one more big ordering edge. Rails unchanged (no reassociation, FMA/MULACC, rounding move, scratch,
or `spec_shared` widening).

## 0.4 Status update 2026-07-17 (final for this session): 9 spills traced, one dead end burned

No new code commit — this section is diagnosis only. Head is `6f65ef31c`, worktree clean, 5 stashes intact.

### The 9 spills, traced to source (matched env)

Each spill victim's def / srcs / consumer / live-range, mapped to the oracle. **Matched env** — the same env the
9-spill baseline is defined in. (Do NOT trace under an added `SPILL_TRACE`-style debug flag: it perturbs allocator
traversal to ~16 spills and reweights the classes. An earlier trace did exactly this and mis-ranked the metadata as
dominant; the corrected matched-env ranking is below. **Lesson: trace in the SAME env you measure in.**)

| class | count | def <- srcs | consumer | span | what it is |
|---|---:|---|---|---|---|
| `V_CVT_I2F` | 3 | `<- V_WMMA_I8` | next chain `V_WMMA_I8` | 17-65 | residual WMMA drain tail |
| `V_MUL` | 2 | `<- V_CVT_H2F, V_CVT_H2F` | `V_MUL`/`V_ADD` | 63-236 | half2 `scale`/`bias` metadata (`recurrence.py:184-185`) |
| `GLOBAL_LOAD` | 2 | `<- V_OFFSET, S_LOAD_PTR` | `DS_STORE` | **~9600** | CSE'd Q8 panel, reused across all 8 subtiles' LDS stagings |
| `V_ADD` | 1 | `<- V_MUL, V_MUL` | `V_ADD` | 262 | recurrence accumulation |
| `V_CMPLT_I` | 1 (`v315`) | `<- V_AND, V_CONST, V_CONST` | `END` | ~13100 | whole-program lane predicate |

### REFUTED — do not retry: ordering the half2 metadata

The tempting "symmetric twin" move (order each group's `scale`/`bias` metadata behind the prior group's drain, like the
C-drain fix) **makes it dramatically worse**. Verified by construction (matched env, full-grid 128,128,256):

| variant | peak | spills |
|---|---:|---:|
| baseline `6f65ef31c` | 273 | **9** |
| metadata behind prior drain (new void barriers) | 282 | 27 |
| metadata behind existing release (barrier-free, minimal edge) | 267 | **74** |

There are 128 metadata muls (8 subtiles x 8 groups x 2). They are *schedulable* (the `dm`/`ds` loads depend only on the
phase `publish` barrier, not on drains, so `pressure_schedule` — block-local — cannot move them), but they contribute
only ~6 registers to the peak (273->267) while serializing 128 uniform loads into the tight drain windows explodes
spills 8x. The metadata is NOT the dominant class and NOT the lever. Cross it off.

### For the next owner (Codex): two honest options, and the recommendation

**The cheap ordering-edge move is exhausted and now fights itself.** The residual is dominated by the `V_CVT_I2F` drain
tail (3/9) and the two long-span Q8 `GLOBAL_LOAD`s (2/9) — and those loads are span ~9600 *because* the cross-subtile
serialization (`cd989ad0e`) pins the shared Q8 panel live across all 8 subtiles. More serialization lengthens that
span. So closing the last 9 is a genuine trade, not another `.after()`:

1. **Grind the spill gate to 0.** Candidate levers, each UNVERIFIED, each needs measure-by-construction:
   - The Q8 `GLOBAL_LOAD` (biggest single spans): decide whether the panel is re-staged per subtile (redundant DS_STOREs
     that could share) or genuinely 8 chunks. If shareable, stage once; if not, the tension with serialization is real
     and may need a re-load-vs-hold decision (registers vs LDS/mem traffic). This is a design call, not a mechanical edge.
   - `v315`: recompute-at-uses localization (analogous to address remat, cf. `a293391b0`) — one clean register.
   - The 3 `V_CVT_I2F`: milder cross-subtile edge via the `prior_drains` knob in `_accumulator_vectors` (order behind
     `e-1`'s CVT frontier, not its full accumulator). Risk: over-constraint blew up the metadata; measure every step.
   - Peak is only ~9-18 over a 255 pool, so 2-3 small wins plausibly reach 0 — but none are proven and the Q8 one is a
     design decision.
2. **Stand up the GPU correctness harness FIRST, before grinding to 0.** This is the RECOMMENDED path. Rationale below.

**Why correctness-first is the better use of effort.** Everything to date is CPU-side compile / register allocation
(`DEV=PYTHON`, no execution). Zero output has ever been checked against the frozen llama comparator. Closing the spill
gate proves the kernel *compiles*, not that it is *correct* or *fast* — those are separate, larger, unmeasured gates
(§13.1 correctness, §13.2 performance = the actual promotion authority). The dominant uncertainty in any ETA is what the
FIRST GPU run surfaces (writeback layout `col*N+row`, five-buffer ABI offsets, launch geometry) — none of which the spill
work touches. One correct number collapses far more uncertainty than 9->0 spills. The compile foundation is solid for
this: clean source graphs, math-neutral ordering edges (op counts identical, rounding boundary untouched), asserted ABI
in `__post_init__` — so a wrong number will point at the writeback/ABI logic, not at allocator hacks.

**What has NOT been done, so no one over-claims:** never run on GPU; never numerically validated; only the smallest full
grid (128,128,256) compiled — real roles are 512x1024x5120 up to 512x5120x17408, where more epochs = more `consumer_seam`
ordering and the aligned-tail/dynamic-offset paths first get exercised; performance entirely unmeasured. The 7 historical
failing tests (four 16-subtile, two prefill-adapter, one q4k-tiled) still fail identically on a clean tree — verify before
blaming any change.

Rails unchanged (no reassociation of `(previous + scale*C) + bias`, no FMA/MULACC, no rounding-boundary move, no AMD
scratch, no `spec_shared` widening, no model/VRAM/GPU/route branching).

## 0.5 Status update 2026-07-17 (Codex): full-grid GPU proof, R5 win, R6/R7 fail-closed

The recommended correctness-first path now has a concrete bounded result. Commits `472ca6da9`, `72e58d322`,
`ef9fb08c8`, `e78551257`, `f6fa66b96`, `76b721184`, `1b7748e26`, `1a26a721a`, `2643551d4`, `48c11957d`, and
`661aea564`, `fea30a7ee`, `1b525a2a5`, `1b7b249cd`, `1c01325de`, `26edaa8f2`, and `78101b7c9`
are pushed on `master`.

* The emitted full-grid 128x128x256 PROGRAM passed an in-process AMD probe: 0/16,384 output mismatches against the
  fp16-rounded Q8 DS4 oracle, max absolute error 3.05e-5, mean 1.12e-8. Resource evidence is vgpr=256, LDS=57,856,
  scratch=0, wave32, with distinct source/binary hashes. This is the `q4k_q8_1_mmq_amd_isa_full_grid_v0` backend.
* Same-session timing on identical quantized inputs is full-grid 0.287 ms minimum versus direct-packed 8.980 ms
  minimum (about 31.2x minimum speedup). This is R5 bounded evidence only; it is not a production route claim.
* R5 ranking now recognizes an emitted full-grid win even when the oracle row ranks faster. `promotion_eligible` stays
  false and R6 remains `BLOCKED_ROLE_SHAPE_INTEGRATION`; no role was changed and direct-packed remains default.
* The exact R6 target is `ffn_gate_up` 512x17408x5120. The probe covers only 128x128x256. The route-shape artifact
  records 20 required K-epoch launches over the full 4x136 M/N grid, plus Q4/DS4 repacking and accumulation (the
  full-grid sink already owns/scatters all M/N tiles in each launch).
  A manifest-scoped smoke artifact now passes the negative-role (`attn_qo`, `ffn_down`, `attn_kv`) and direct-packed
  rollback/default checks; these subgates are true, but they do not substitute for target-shape execution.
* A monolithic K=512 compile is concretely blocked by `NotImplementedError: vgpr lease exceeds virtual pool`. An
  explicit per-store LOAD+ADD accumulation sink was prototyped, but its two-launch 128x128x512 probe exceeded the hard
  six-minute compile deadline with no structured result. The safe follow-up delegates accumulation to tinygrad
  elementwise add over fresh partial outputs and **passes** the bounded 128x128x512 proof: 0/16,384 mismatches,
  max abs 2.44e-4, vgpr=256, LDS=57,856, scratch=0. This is adapter evidence only; production repack, 20-launch
  scheduling, role census, and no-hidden-fallback proof remain absent. R7 rows carry both the per-store timeout and
  the bounded elementwise result instead of claiming source-clone conversion.
* The lazy owner manifest now lets the actual 14B shape `(512,17408,256)` build and emit one K-epoch PROGRAM in
  182.23s: vgpr=256, LDS=57,856, scratch=0, owner count 8,912,896, source SHA `b8923985…`, binary SHA
  `21908e0b…`. This closes the target-shape compile/resource slice, not the 20-epoch GPU correctness/health gate.
  A reduced 256x256x256 one-epoch dispatch then exposed the next blocker: 65,425/65,536 mismatches (max abs 840.9)
  despite the same zero-scratch resources; 128x128 remains the only numerically passing grid. The target 20-epoch
  run also hit an HCQ wait timeout (signal 9 not set, current 8). Treat this as a grid-scaling address/writeback
  defect, not a promotion or GPU-health pass. The diagnostic launch is explicitly `global_size=[2,2,1]`,
  `local_size=[256,1,1]`; all four 128x128 tiles fail, so the issue is not isolated to a nonzero tile origin.

The through-line is unchanged: the first real GPU result collapsed more uncertainty than further spill grinding, but
the 31x bounded win is not evidence for a 14B route. Never run a production role on this candidate until the K-tiled
adapter has exact numerical evidence, resource/health evidence, negative-role tests, and a no-hidden-fallback census.

## 0.6 Status update 2026-07-18: multi-grid address defect repaired

The reduced-grid blocker was isolated to the split Q8 producer, not WMMA writeback or workgroup dispatch. Its K-record
address callbacks assumed a fixed 128-row record stride. For `M>128`, K-record 1 therefore read the preceding rows;
the first real multi-grid run exposed this as 65,425/65,536 mismatches. Commit `a3ae968a9` derives the full M stride
from the physical five-buffer allocation, with a focused `M=256` producer test. Commit `e70398880` separately fixes
AMD's packed workgroup-ID SGPR mapping (`gidx1`-only uses s2) and records the ABI regression test; this converted the
earlier M-only MMU fault into a structured numeric result.

Fresh native probes after both fixes are exact and remain fail-closed:

- `M=256,N=128,K=256`, global size `[1,2,1]`: 0/32,768 mismatches, max error 6.1e-5.
- `M=256,N=256,K=256`, global size `[2,2,1]`: 0/65,536 mismatches, max error 6.1e-5 across all four tiles.
- The bounded R5 `128x128x256` probe remains exact (0/16,384) and measures roughly 32–33x versus direct-packed in the
  same session; current source/binary hashes are retained in the machine-search artifact output.

The exact target-role (`ffn_gate_up`, `512x17408x5120`) compile still emits with zero scratch/spills, but its 20-epoch
GPU dispatch has not yet produced a structured correctness result. The default 30-second HCQ wait was insufficient;
an extended-wait run drove the GPU for roughly 14 minutes and then the amdgpu driver reset the device (`GPU reset
succeeded; device wedged, but recovered through reset`) before the child could emit JSON. A subsequent one-epoch
diagnostic also exceeded a 600-second outer deadline, with amdgpu `sq_intr` errors recorded in the kernel log. Treat
this as a target workload health/scale blocker, not a correctness pass: R6 role integration and production route
promotion remain blocked until a bounded target run completes without a GPU reset/error and reports numerical
comparison/resource evidence.

The harness now supports bounded N-tile chunking. The exact target shape with one K epoch and 16-N-tile chunks passes
0/8,912,896 mismatches (max error 1.22e-4) in 3.38 seconds of same-session GPU time, proving the chunked address
views and target geometry are numerically sound. A two-epoch run with the same 16-tile chunks also hit a 600-second
HCQ timeout (the second epoch did not signal); the full 20-epoch run has the same health failure. One-tile chunks
subsequently produced an MMU `NotPresent` fault. Chunking is diagnostic only and is not promoted.

An additional host-FP32-accumulation diagnostic lets two epochs complete without the GPU elementwise-add launch, but
reports 9,031 mismatches (max error 141.8), so it is not a promotion substitute; epoch 0 and epoch 1 run alone are
each exact. This narrows the remaining defect to sequential multi-epoch/chunk state or resource lifecycle.

## 0.7 Status update 2026-07-18: target lifecycle isolation and fail-closed role schema

The target-role harness now has bounded lifecycle controls that do not alter the generated program or route policy:
one full-N dispatch per K=256 epoch ('n_chunk_tiles=136'), persistent output/input buffers, preloaded all-epoch
Q4/Q8 inputs, optional per-epoch numerical checks, and an explicit device-synchronize diagnostic. The exact target
shape is still 'ffn_gate_up' (512,17408,5120).

The one- and two-epoch full-N controls are exact: the persistent two-epoch run reports 0/8,912,896 mismatches,
both epoch checks pass, max absolute error 3.662e-4, and resources vgpr=256, LDS=57,856, scratch=0.
This separates the earlier N-chunk failure from arithmetic and proves the persistent full-N path for a bounded
prefix. It did not prove the later all-epoch preloaded Q4 view: that path was subsequently found to flatten
`[N,epoch,144]` storage while indexing it as contiguous `[epoch,N,144]`.

The required 20-epoch target run remains a health blocker. It fails before structured result serialization at the HCQ
timeline (signal 30/current 29 with preloaded inputs and no host per-epoch copies; the explicit synchronize variant
repeats the failure at signal 32/current 31). Thus allocator reuse, per-epoch SDMA copies, host checks, and the
elementwise accumulation choice have each been isolated without producing a full-target pass. The blocker artifact is
recorded as 'target-role-20epoch-preloaded.json'; it is not correctness or promotion evidence.

Kernel logs for the controlled sync attempt show SQ memory-violation and gfxhub page-fault evidence followed by MES
queue-removal failure and a GPU reset (the HCQ timeout is therefore a driver-visible health failure, not merely a
Python deadline). The retained fields do not establish an instruction-page or instruction-fetch fault. Earlier
overlapping child experiments can leave stale queues, so this log is treated as health evidence only and does not
identify a mathematical kernel bug.

The machine-search schema now accepts a target-role artifact only when it independently proves the exact role/shape,
all 20 K epochs with full-N dispatch, GPU-side FP32 accumulation, preloaded/persistent buffers, zero-mismatch finite
output, Q4/Q8 repack identities, resource/source/binary identity, same-session timing, and no hidden fallback.
R6 remains fail-closed until that artifact exists; R7 can only mark the three exercised source components as owned
atoms when the same measured target artifact carries source-revision/ownership evidence. Existing defaults remain
production_dispatch_changed=false and direct_packed.

## 0.8 Status update 2026-07-18: all target epochs exact under fresh-process isolation

Commit `68d57d9e4` adds a fail-closed process-per-epoch diagnostic. It compiles the exact target K=256 PROGRAM once on
the CPU, serializes it, and loads it in one fresh spawned AMD process per epoch. A partial is admitted to the host
aggregate only after its independent DS4 oracle comparison passes, the epoch's kernel-log window contains no AMD
fault/reset marker, and a known-safe tiny add passes in another fresh spawned process. The harness is permanently
marked `diagnostic_only=true`, `promotion_eligible=false`, `production_dispatch_changed=false`, with direct-packed
still the default.

The complete exact-role sweep now passes:

- Shape/role: `ffn_gate_up` `(512,17408,5120)`, all 20 K=256 epochs, full grid `[136,4,1]`.
- Every epoch reports 0/8,912,896 mismatches, for 0 mismatches across 178,257,920 individually checked values.
- All outputs/references are finite. Maximum absolute error over the 20 epoch comparisons is 1.2207e-4.
- Kernel times range from 0.690 to 0.705 ms (13.958 ms summed); total worker time including deterministic input/oracle
  generation and readback is 122.06 seconds.
- Every epoch passed its post-dispatch health canary and kernel-log check; no `sq_intr`, page fault, MES failure, reset,
  wedged-device, or VRAM-loss marker was observed.
- The measured program remains vgpr=256, LDS=57,856, scratch=0, wave32, with distinct source/binary identities.

The full evidence is `docs/target-epoch-safe-all-20260718.json`. This proves there is no bad K epoch and strongly
localizes the earlier four-plus-launch failure to same-process repeated-launch/input-buffer/queue lifecycle, not MMQ
arithmetic or a target-grid address defect. It does **not** close R6: the sweep uses fresh processes and host
aggregation, while the strict gate still requires one healthy same-process 20-launch adapter with preloaded/persistent
buffers, GPU-side FP32 accumulation, final full-K oracle comparison, same-session timing, and no hidden fallback.

The follow-up no-target preload canary also passes. It uploads and byte-round-trips exactly 50,135,040 bytes (the full
20-epoch Q4 capacity), with a known-safe tiny add between upload and readback. The upload advances the AMD timeline
cleanly through signal 24, the tiny add/readback through signal 29, and the full readback through signal 53. Payload
and roundtrip SHA256 are both `879d7bb7…`; the independent post-run health canary passes and the kernel-fault window
is empty. Evidence is `docs/target-q4-preload-canary-20260718.json`. Therefore neither the large SDMA transfer nor a
timeline value around 30 is sufficient to reproduce the strict-run fault. The next safe discriminator is the exact
four-buffer preload plus target PROGRAM runtime/code upload with **no target dispatch**, followed only then by a
bounded target-only same-process launch sequence if runtime construction remains healthy.

Construction-level audit then found and repaired a separate preloaded adapter defect. `_random_q4_words` stores blocks
as `[N,epoch,144]`; a single `Buffer.view` epoch base requires `[epoch,N,144]` contiguous storage. The old preloaded
path copied the N-major flattening and advanced `epoch*N*36` words, so it did not point at the requested epoch. The
preload now explicitly transposes/packs epoch-major storage, with a CPU regression proving that every epoch view is
byte-identical to `q4_blocks[:,epoch,:]`. This explains the earlier massive mismatches from the experimental
preloaded-input path. The driver fault still needs a clean retest after the corrected layout; the old fault artifact
must not be treated as evidence against the corrected adapter.

The corrected exact-buffer runtime-init canary also passes without invoking the target kernel. It preloads the full
epoch-major Q4 and Q8 buffers plus the device-zero accumulator, reaching timeline signal 30; constructs/uploads the
exact target PROGRAM, reaching signal 31; and runs only a tiny health add, reaching signal 36. The child and
independent postflight health probe pass, with no kernel-log fault. Evidence is
`docs/target-preloaded-runtime-init-canary-20260718.json`. This rules out the corrected bulk input layout,
allocation/copy, accumulator initialization, and target code upload. The next boundary is one full-N target launch
from a corrected epoch view, followed by its independent epoch oracle comparison.

That corrected one-shot boundary passes as well. Epoch 0 dispatches once from the full epoch-major preloaded buffers
in 0.692 ms and reports 0/8,912,896 mismatches, max absolute error 1.2207e-4, with all values finite. Timeline signal
30 completes, the kernel-log window is empty, and the independent postflight health canary passes. Evidence is
`docs/target-preloaded-single-epoch-0-20260718.json`. The old signal-30/current-29 artifact therefore does not
reproduce after correcting the Q4 layout. Next isolate repeated target launches without accumulation, then add the
GPU accumulator only after that target-only prefix remains healthy.

A corrected four-launch target-only prefix completes all four GPU submissions without a queue fault (timeline signals
30 through 33), but the final epoch-3 output is not exact: 19,139/8,912,896 mismatches, first at `[48,1280]`, max
absolute error 222.53. There is no accumulation and only one final readback, so this is a repeated-target/preloaded-view
numeric defect rather than the old driver reset. Evidence is
`docs/target-preloaded-target-only-epochs-0-3-20260718.json`. The next discriminator is epoch 3 alone from the same
full corrected preload: a pass means prior launches are required to trigger the defect; a failure means the nonzero
preloaded view is still wrong.

Epoch 3 alone from the identical full corrected preload passes 0/8,912,896 mismatches (0.704 ms, max absolute error
1.2207e-4), with a clean kernel log and healthy postflight canary. Evidence is
`docs/target-preloaded-single-epoch-3-20260718.json`. The nonzero epoch view is therefore correct; the four-launch
mismatch requires earlier launches in the same process. Narrow the first failing transition with two- and, only if
needed, three-launch target-only prefixes before changing accumulation or route policy.

The transition is now bounded: epochs 0→1 pass with 0 mismatches at epoch 1, while epochs 0→1→2 complete all timeline
signals but leave epoch 2 with 24,277 mismatches (first `[48,384]`, max absolute error 202.38). Evidence is
`docs/target-preloaded-target-only-epochs-0-1-20260718.json` and
`docs/target-preloaded-target-only-epochs-0-2-20260718.json`. With no accumulation, no intermediate readback, and no
queue fault, launch 3 is the first failing same-process transition. Next compare persistent output reuse against
fresh, held output buffers per launch while keeping the runtime and preloaded inputs identical.

Fresh held outputs do not repair launch 3. With three distinct output allocations retained until final readback, epoch
2 still has 21,234 mismatches (first `[48,1280]`, max absolute error 204.05), while timeline signals 30–32 complete
and the kernel log remains clean. Evidence is
`docs/target-preloaded-target-only-fresh-epochs-0-2-20260718.json`. Output alias/reuse is therefore ruled out. Next
repeat one exact epoch three times (`[0,0,0]`) to distinguish repeated invocation state from changing buffer-view
kernargs.

Both persistent- and fresh-output `[0,0,0]` sequences pass exactly on launch 3, including a final uninitialized fresh
output that cannot be masked by stale correct output values. Evidence is
`docs/target-preloaded-target-only-sequence-0-0-0-20260718.json` and
`docs/target-preloaded-target-only-fresh-sequence-0-0-0-20260718.json`. Repeated invocation count, cached runtime,
and output allocation are therefore healthy when inputs stay unchanged; corruption requires epoch input/view changes.
The next split changes Q4 epochs while holding Q8 fixed, then changes Q8 while holding Q4 fixed, to identify which LDS
producer lifecycle can observe stale prior-launch data.

The split identifies Q8 unambiguously. Q4 `[0,1,2]` with Q8 fixed at 0 passes the third fresh output exactly; Q4 fixed
at 0 with Q8 `[0,1,2]` produces 18,203 mismatches (first `[48,0]`, max absolute error 193.18). All target signals
complete and the kernel log stays clean. Evidence is `docs/target-preloaded-q4-change-q8-fixed-20260718.json` and
`docs/target-preloaded-q4-fixed-q8-change-20260718.json`. The repeated-launch stale state is in the Q8 staging side,
not Q4, output reuse, runtime count, or queue health. Next split Q8 values from Q8 scale/sum metadata to choose the
exact LDS producer subpath before editing ordering.

That final split localizes the corruption to Q8 scale/sum metadata. With Q4 and metadata fixed at epoch 0, changing
only Q8 values through `[0,1,2]` passes the third fresh output with 0/8,912,896 mismatches and max absolute error
1.2207e-4. With Q4 and values fixed at epoch 0, changing only scale/sum metadata through `[0,1,2]` completes timeline
signals 30–32 without a kernel fault but leaves 18,224 mismatches (first `[48,640]`, max absolute error 229.44).
Evidence is `docs/target-preloaded-q8-values-change-metadata-fixed-20260718.json` and
`docs/target-preloaded-q8-values-fixed-metadata-change-20260718.json`. This is diagnostic evidence, not promotion:
the next safe step is to distinguish changing metadata pointers from changing metadata contents at a fixed address,
then repair only the evidenced metadata load/publish seam. Do not revive broad half2 ordering: the earlier matched
compiler experiment worsened spills from 9 to 74.

The pointer/content discriminator is now matched on one compiled PROGRAM, removing cross-process compiler-variant
ambiguity. All four fresh workers loaded serialized artifact `d9e9eadd…` / binary `26a67539…`. Both changing-address
modes fail on launch three: views into the full preloaded arrays leave 20,219 mismatches, and distinct preloaded
one-epoch allocations leave 21,223. Both fixed-address modes pass with 0/8,912,896 mismatches and max absolute error
1.2207e-4: synchronous host refresh, and the production-shaped path that copies from the full preloaded GPU arrays
into one persistent fixed-VA metadata slot using GPU SDMA. The Q4 and Q8-value inputs remain fixed throughout.
Evidence is `docs/target-matched-binary-metadata-storage-bisect-d2d-20260718.json`; the earlier
`target-fixed-address-metadata-refresh-20260718.json` and `target-dedicated-preloaded-metadata-20260718.json` are
retained only as unmatched precursors because their binaries differed. An independent postflight tiny-add health
probe passed and the kernel log contained no fault/reset marker.

This falsifies stale metadata contents and the speculative LDS-ordering repair as the observed cause. The safe adapter
contract is now: preload all epoch metadata on the GPU, copy the selected epoch into persistent one-epoch scale/sum
staging allocations, and bind those same two VAs to every target launch. Next integrate that lifecycle into the strict
same-process target-role harness, preserve the D2D copies in timing/evidence, and rerun a three-epoch prefix before the
20-epoch R6/R7 gate.

The first strict integration attempt exposed a second, independent runtime boundary and was stopped. Exact timeline
mapping showed epoch-0 metadata SDMA copies and target MMQ completed through signal 33; the separate
`(accum + partial).realize()` launch requested signal 34, triggered SQ memory-violation evidence, timed out, and was
followed by a gfxhub page fault and GPU reset. The retained log does not identify an instruction-page or
instruction-fetch fault. The reset recovered and the independent tiny-add health probe passed. A one-epoch
host-accumulation control then passed exactly with clean logs and stable metadata VAs
(`docs/target-role-stable-metadata-host-prefix-1-20260718.json`), proving target + SDMA + full readback can be healthy.
A three-epoch host-accumulation run passed epoch 0 exactly but faulted during epoch 1 before its check, with the wrapper
capturing the SQC/page-fault/reset markers and stopping (`docs/target-role-stable-metadata-host-prefix-3-20260718.json`).
The earlier external-add failure is retained as
`docs/target-role-stable-metadata-prefix-3-20260718.json`.

Therefore fixed metadata staging closes the numerical corruption but does not yet make the old external-add/intermediate
readback adapter promotion-safe. Do not run the 20-epoch strict gate from that adapter. The existing full-kernel builder
has an `accumulate=True` form that folds FP32 accumulation into the target writeback; audit and compile that form
CPU-only first. If it remains spill/scratch free with the same ownership/ABI, it is the preferred next bounded GPU
probe because it removes both the separate elementwise kernel and per-epoch readback lifecycle.

That safer adapter is now implemented and its first escalation result is mixed, so R6/R7 remain closed. The
`accumulate=True` target program emits with 256 VGPR, 57,856 B LDS, zero scratch/spills, the same five-buffer ABI and
136x4x1 / 256x1x1 grid, and exactly 64 additional global loads plus 64 FP32 vector adds at writeback. The opt-in
`target_in_place_fp32_add` harness zeros one persistent output, reuses it across K epochs, performs no external add
and no intermediate readback, and now fail-closes on unhealthy preflight, timeout fault logs, or unhealthy postflight.
The independent epoch orchestrator also records deterministic fixture hashes and per-epoch health attestations for
later strict evidence composition. These changes are pushed in `b133d99de`.

The isolated one-epoch GPU gate passed the full 512x17408 output with 0/8,912,896 mismatches, no non-finite values,
maximum absolute error 1.2207e-4, stable fixed-VA GPU-SDMA metadata staging, clean kernel logs, and healthy pre/post
tiny-add probes (`docs/target-role-in-kernel-accum-prefix-1-20260718.json`). The three-epoch escalation was stopped on
its first target launch: SQ memory-violation evidence was followed about 30 seconds later by a gfxhub page fault, MES
queue-removal failure, and GPU reset. The post-reset health probe passed
(`docs/target-role-in-kernel-accum-prefix-3-20260718.json`). Both attempts compiled the exact same source
`beed14d6…` and binary `78eefc23…`, so this is not a spill regression or a cross-compile variant. Do not run the
20-epoch gate yet. The active blocker is now intermittent AMD executable/queue lifecycle behavior across isolated
target attempts; audit code upload, entry-point lifetime, and launch state before another target dispatch.

### Historical revised completion plan: completed with existing tinygrad PM4/AQL launch paths

Public AMD/ROCm history makes this a known fault class rather than a reason to resume broad spill or metadata search.
AMD's KFD architect has documented Tinygrad reproductions involving low-level queue/synchronization behavior and Navi3
MES recovery, while public ROCm reports contain the same `sq_intr` type-2, page-fault, queue-removal, and reset
sequence. The exact root cause here is still unproven. In particular, the current tinygrad PM4 path already emits an
`ACQUIRE_MEM` instruction-cache invalidation before dispatch, so an extra blind cache flush is not an evidenced fix.

The remaining work must proceed in this order:

1. Compile once and retain one launcher-neutral artifact: HSACO bytes, serialized PROGRAM, generated source, ELF
   descriptor/resources, ABI/grid/local shape, deterministic fixture hashes, and disassembly. Never compare two fresh
   compiles when diagnosing this fault.
2. Audit that frozen binary CPU-only for image/section bounds, descriptor and entry-point arithmetic, branch targets,
   termination, unexpected indirect control flow, and instruction encodings.
3. Reuse two existing tinygrad launch paths over the exact same frozen artifact and five buffers: direct PM4/KFD and
   AQL (`AMD_AQL=1`). No HIP launcher is required. Neither path may recompile or silently fall back.
4. Record program/code-object VA, entry address, all five buffer VAs/sizes/offsets, kernarg VA and pointer words,
   runtime identity, launch count, kernel-log window, timeout result, post-run tiny-health result, and any available
   wave PC or amdgpu coredump reference.
5. Run a fail-closed matched matrix: one launch under each launcher, then three launches under each launcher. Stop at
   the first numeric mismatch, non-finite output, SQC/page-fault/MES/reset marker, timeout, or failed health probe.
6. Classify before editing:
   - both launchers fail: repair generated ISA/control flow or kernel synchronization;
   - only tinygrad PM4 fails: repair its program-upload/queue/dispatch contract;
   - only repeated launches fail: repair executable, address, or queue lifetime;
   - outcome follows VA/entry changes: repair allocation/mapping lifetime.
7. Implement only the evidenced fix and retain a focused regression test. Do not tune spills, revive the rejected
   half2 ordering experiment, add model/GPU-name branches, or treat a recovered reset as a pass.
8. Reuse one passing frozen artifact for strict 1 -> 3 -> 20 same-process target escalation with stable GPU-SDMA
   metadata staging and in-place FP32 accumulation, no intermediate readback, exact final oracle comparison, clean
   fault logs, and healthy pre/post canaries.
9. Regenerate the all-20 independent epoch artifact with fixture and per-epoch health attestations, compose it with the
   strict same-process artifact, and require both for R6.
10. Pass R6/R7 and the negative-role/direct-packed fallback checks, then update the one-role opt-in/promotion artifact,
    commit and push all retained evidence, and only then claim completion.

## 1. Executive state

The project is building a generated tinygrad prefill route for non-fitting quantized models, using Qwen3-14B as the
proof workload. The immediate goal is generated Q4_K/Q8_1 MMQ parity or better against the frozen llama.cpp comparator.
The model is selected by the user, but compiler route selection must remain a function of workload and hardware facts,
never a model-name, fixed-VRAM, or fixed-GPU branch.

Do not use a percentage for the current state; the phase gates are the authority. Spill-free emission, exact frozen
binary identity, bounded same-session timing, full-role numerical correctness, PM4/AQL prefix equivalence, and two
complementary 20-epoch correctness proofs are complete for one Q4 `ffn_gate_up` role.

The active blocker is now coverage and integration, not compiler spills:

- three Q4 roles remain unqualified;
- two Q6 fallback rows remain outside the final policy contract;
- the candidate is still a static `research_descriptor_only` row with no live registry/runtime binding or route
  census;
- no immutable six-row policy exists;
- whole-model Phase 6 and matched multi-context Phase 7 have not run.

The default remains direct-packed and no production dispatch changed.

## 2. User intent and non-negotiable constraints

The user wants the work continued through the logical endpoint, but with these boundaries:

1. Stop before performance autoscan. Autoscan is allowed only after a manually selected generated route proves beyond
   llama parity.
2. The model remains user-selected. GPU capability, VRAM, memory admission, and eventually candidate choice should be
   discovered from facts, not flags or hardcoded model/GPU cases.
3. Do not create an `8B` route and a `14B` route. The architectural distinction is fitting versus non-fitting resource
   conditions.
4. Do not permit hidden dense dequantization, scratch spilling, or a handwritten/oracle kernel to satisfy a generated
   route claim.
5. Preserve exact llama recurrence ordering and rounding. Do not introduce FMA/MULACC, reassociate
   `(previous + scale*C) + bias`, or move fp16/fp32 conversion boundaries without a new correctness authority.
6. Keep commits coherent and small. Hooks are enabled. Push accepted commits to `origin/master`.
7. Preserve unrelated user changes and stashes. Never rewrite or delete user stashes.
8. Temporary authored-code budget is 35,000 lines. After parity proof, perform another prune and restore the 30,000-line
   target. Do not add speculative frameworks when an existing primitive can be reused.
9. Decode and prefill are separate benchmark authorities. The current work is generated prefill; decode must later pass
   a regression gate.
10. GPU may be used now, but do not consume it while another explicitly reported benchmark run owns it. The active
    compiler reproducer is CPU-only.

## 3. Canonical objective and promotion authority

The canonical scope remains:

- `docs/qwen3-14b-generated-prefill-completion-scope-20260714.md`

The final promotion authority is whole-prefill synchronized tokens/second at prompt/context lengths 512, 1,024, 2,048,
and 4,096. Llama and tinygrad must run sequentially in at least three alternating pinned-clock sessions with raw samples
retained.

Required final gates:

- No accepted context below 98% of llama.
- Every declared context median exceeds llama.
- Paired aggregate 95% lower confidence bound exceeds 1.00.
- Aggregate geometric-mean target is at least 105% of llama.
- Project target is at least 2,000 tok/s.
- Decode remains correct and within the declared regression tolerance.
- If any context or aggregate misses, run a fresh Boltbeam timing pass on the exact candidate revision and reconcile
  candidate roles, activation preparation, attention, launch/synchronization, and residual work.

Kernel TFLOP/s and Boltbeam are attribution evidence, not promotion authority.

## 4. Current phase ledger

| phase | state | closing condition |
|---|---|---|
| Scope and architecture | complete | Canonical decision tree and no-model-branch rules are documented. |
| Six-row evidence and attribution | complete | Four Q4 and two Q6 workload rows are exact and identity-qualified. |
| Source-pinned oracle foundation | complete | Exact llama structure is adapted to the five-buffer ABI and full-grid seam. |
| Spill-free generated emission | complete for exact `ffn_gate_up` artifact | Frozen AMD program has zero scratch and exact resource/disassembly evidence. |
| Q4 role qualification | one of four roles evidence-qualified | Qualify `attn_qo`, Q4 `ffn_down`, and `attn_kv`; then prove aggregate timing. |
| Q6 fallback qualification | incomplete | Qualify the two direct-packed Q6 rows and decide policy from measured share. |
| Six-row policy | incomplete | Emit one immutable policy with exact bindings or declared rollback for every row. |
| One-role live research opt-in | not implemented | Bind generically and pass live negative-role/no-hidden-fallback census. |
| 14B mixed-route end to end | not started | Correctness, route census, memory/resource proof, GPU health, decode regression. |
| Matched llama/Boltbeam qualification | not started | Multi-context statistical parity/beyond-parity gates pass. |
| Autoscan and post-proof prune | deliberately paused | Start only after beyond-parity proof. |

## 5. Target architecture

```text
loaded GGUF tensor facts + user-selected model
  -> typed workload inventory
  -> target capability + memory admission
  -> candidate identity and exact packed ABI
  -> Q8 activation producer
  -> generated Q4_K/Q8_1 MMQ contraction
  -> resource/correctness evidence
  -> immutable manual six-row policy
  -> 14B end-to-end proof
  -> matched llama comparison + Boltbeam attribution
  -> only then performance autoscan
```

The current source-pinned MMQ authority is:

- Tile: `128x128x256`.
- Eight waves, 256 threads.
- 57,856 bytes LDS.
- Persistent decoded Q4 panel.
- Two K128 Q8 phases.
- Eight K32 correction groups / sixteen signed i8 WMMAs per K256 epoch.
- Exact half2 metadata recurrence.
- Split five-buffer ABI:
  - slot 0: fp32 output;
  - slot 1: Q4 packed `uint32`;
  - slot 2: Q8 values `int8`, physical `[K/128, M, 128]`;
  - slot 3: Q8 scales `fp32`, physical `[K/128, M, 4]`;
  - slot 4: Q8 original sums `fp32`, physical `[K/128, M, 4]`.

Scale and sum are independently converted to fp16 only at LDS half2 staging, preserving llama rounding.

## 6. Relevant code map

### Oracle and generated full-grid seam

- `extra/qk/mmq_llama_candidate_plan.py`
  - Source-pinned physical schedule and identity.
- `extra/qk/mmq_llama_oracle_epoch.py`
  - Exact epoch construction.
- `extra/qk/mmq_llama_oracle_recurrence.py`
  - Exact K32 recurrence and rounding authority.
- `extra/qk/mmq_llama_full_kernel.py`
  - Bounded callback graph and release seams.
- `extra/qk/mmq_llama_record_producers.py`
  - Typed producer witnesses.
- `extra/qk/mmq_llama_five_buffer_graph.py`
  - Five-buffer composition across epochs.
- `extra/qk/mmq_llama_five_buffer_full_kernel.py`
  - Full-grid ownership, dynamic offsets, aligned-tail policy, row-major writeback, fail-closed compilation.

### AMD selection, scheduling, and allocation

- `tinygrad/renderer/isa/amd.py`
  - `_progressive_c_assignment`: symbolic root path cover.
  - `_acc_base`: physical C-fragment allocation/reuse.
  - `_try_wmma_kmajor_phase`: alternate K-major selection path. A conditional chain-major experiment here was proven
    inactive for the exact workload and reverted; do not assume this function owns the residual path.
  - `_selected_wmma_roots`: retained at `412d7998f`; recovers one-step carrier markers and propagated multi-step
    machine-chain markers, walking to the earliest machine head.
  - `_serialize_progressive_c_drains`: finds each chain tail, collects eight conversion drains, and orders reuse heads.
  - `_post_isel_structural_lifetimes`: composes store chaining, progressive-C serialization, and address localization.
  - `_build_wmma_from_packs` / `_build_wmma_tile`: fixed A/B/C machine fragment construction.
- `tinygrad/codegen/late/regalloc.py`
  - `pressure_schedule` / `_pressure_schedule_block`: generic pre-regalloc pressure-aware topological ordering.
  - `LinearScanRegallocContext`: live ranges, rematerialization, spill diagnostics, fail-closed no-spill allocation.
- `tinygrad/codegen/__init__.py`
  - Ordering of pressure scheduling, backend cleanup, allocation, and post-regalloc lowering.

### Policy/evidence

- `extra/qk/runtime_specs.py`
- `extra/qk/prefill/six_row_policy_artifact.py`
- `extra/qk/prefill/q4k_q8_five_buffer_role_gate.py`
- `bench/prefill-pure-full-kernel/`

## 7. Current exact blocker

Compiler spills are no longer the active blocker. One exact `ffn_gate_up` kernel has closed emission, resource,
correctness, health, bounded timing, and independent/strict epoch evidence.

The exact blocker to production promotion is incomplete route coverage and integration:

1. the one-role candidate has no live generic binding or runtime negative-role/fallback census;
2. three Q4 roles and two Q6 fallback rows are not qualified under the same final policy contract;
3. no immutable six-row policy exists;
4. project Phase 6 whole-model and Phase 7 matched multi-context gates have not run.

The allocator traces below this section are retained as historical repair evidence, not current work instructions.

## 8. Historical spill reproduction (superseded)

Run from repository root:

```bash
env \
  PYTHONHASHSEED=0 \
  REGALLOC_ADDR_REMAT=1 \
  REGALLOC_END_NO_SOURCE_LIVE=1 \
  REGALLOC_DEBUG=1 \
  REGALLOC_DEBUG_PRESSURE=1 \
  REGALLOC_DEBUG_SPILLS=1 \
  python3 -m pytest -q -s \
  test/unit/test_mmq_llama_full_kernel.py::test_bounded_wave_probe_drains_symbolic_subtiles_without_claiming_a_full_grid
```

Expected current result:

- Pytest reports `1 passed` because the test currently expects the fail-closed `NotImplementedError`.
- The diagnostic must report 112 spills and a 448-byte stack.
- A green pytest alone is **not** success.

When genuine zero-spill emission is achieved, update the bounded test so it no longer expects the spill exception and
instead asserts a real program/resource proof. Do not change the test early merely to make it fail or pass differently.

Fast structural suite:

```bash
python3 -m pytest -q \
  test/unit/test_amd_isa_structural_lifetimes.py \
  test/unit/test_regalloc_pressure_schedule.py \
  test/unit/test_regalloc_end_liveness.py
```

Oracle/full-grid structural suites:

```bash
python3 -m pytest -q \
  test/unit/test_mmq_llama_full_kernel.py \
  test/unit/test_mmq_llama_five_buffer_graph.py \
  test/unit/test_mmq_llama_five_buffer_full_kernel.py
```

Avoid launching duplicate exact-oracle processes. Several previous workers accidentally ran two copies, wasting minutes
and making measurements harder to attribute.

## 9. Historical spill-repair scope (completed and superseded)

This section records the earlier spill investigation for provenance. Its A/B lifetime work reached zero-scratch
emission and is no longer the owning plan. The revised launcher-neutral completion plan above controls current work.

### Step A: prove the selected A/B pair-to-consumer mapping

Before changing scheduling, record for the first spill region:

- each `DS_LOAD_B128` base UOp;
- its fixed physical base and four-lane fragment range;
- its unique numeric WMMA consumer;
- zero-code alias consumers separately;
- its barrier/AFTER ordering dependencies;
- the previous and next load pair using the same physical A/B runs.

Do this post-selection and before regalloc cleanup. Remove temporary instrumentation before committing.

The expected result is seven future pairs opened before their seven consumers. If that is not reproduced, stop and
explain the changed path instead of implementing the plan below from assumption.

### Step B: repair ownership at the AMD selected-lifetime layer

Preferred repair boundary:

- AMD post-selection structural lifetime logic, close to `_serialize_progressive_c_drains` and the selected machine
  fragment graph; or
- the exact AMD selector that emits the residual `DS_LOAD_B128` pair, if the pair order is already wrong at creation.

Required invariant:

```text
previous A/B pair -> matching WMMA consumes pair
                  -> next A/B pair using same physical run may load
                  -> next matching WMMA
```

The ordering must be derived from physical lease identity and actual machine consumers, not from Qwen, 14B, exact
M/N/K, hardcoded UOp ordinals, or a route flag.

If multiple physical A/B runs truly exist, independent runs may overlap. Only pairs mapped to the same constrained run
must serialize.

Potential implementation shape:

1. Discover selected `DS_LOAD_B128` definitions and their fixed base constraints.
2. Follow zero-code fixed aliases but identify the actual machine WMMA consumer.
3. Group load pairs by the physical A/B lease tuple.
4. Order consumers using existing symbolic/machine dependencies; fail closed on ambiguity.
5. Add a zero-code ordering edge from each previous consumer to the next pair's load address/dependency position, while
   preserving the canonical DS-load operand shape and all barrier dependencies.
6. Let `pressure_schedule` place the now-ready pair adjacent to its consumer.

Do not add a generic scheduler rule that assumes every unary one-choice instruction is a zero-code alias. That experiment
was rejected as semantically unsafe. If scheduler lookthrough is needed, use an explicit renderer-owned alias contract.

### Step C: acceptance for the A/B repair

Minimum retained evidence:

- The selected mathematical graph is unchanged.
- Exact WMMA count is unchanged.
- A/B load values, addresses, widths, and fixed bases are unchanged.
- Barriers and release witnesses remain in backward slices.
- At most one A/B pair per shared physical lease is live at a consumer boundary.
- Spill count falls materially from 112.
- No new spill category is introduced.
- Focused AMD and regalloc tests pass.

Closing evidence for the current phase:

- Zero spills.
- Zero scratch/stack.
- Final program emits.
- Resource proof identifies the executed binary.

If a safe repair reduces 112 to a smaller nonzero value, it may be retained as a coherent generic improvement, but the
phase remains open and the next exact victim must be reported.

## 10. Do not repeat these rejected paths

| experiment | result | reason not to revive unchanged |
|---|---|---|
| speculative 12 KiB cooperative descriptor | removed | The source-pinned llama tile is the authority. |
| graph-level Q8 packing | ~232 ms on smallest role | Correct algebra, unusable lifecycle/performance. |
| dequant-once fp16 route | ~4.7 ms on smallest role | Far slower than fused candidate. |
| metadata-only PARAM/SPECIAL carriers | peak worsened in matched A/B | Cosmetic, not owning pressure. |
| atomic eight-register WMMA result span | virtual peak fell, physical saturation remained | Broke UOp invariant and hit unsupported multi-register spilling. |
| vec8 recurrence bundle | reverted | Bundle-side physical improvement was not proven. |
| generic alias-closure scheduler | spills worsened to 516 | Reduced one overlap but delayed other releases; initial alias inference was also unsafe. |
| conditional chain-major `_try_wmma_kmajor_phase` | exact metrics unchanged | The real workload did not select that branch; commit `5913c65c4` was reverted by `63b31be2d`. |
| regalloc replay of `DS_LOAD_B128` | prohibited | Shared-memory loads are not safe rematerialization. |
| enabling AMD scratch | prohibited | Violates performance/resource contract and hides the compiler defect. |

## 11. Accepted compiler repairs and what they proved

Recent accepted commits, oldest to newest:

- `27a910ed6 [amd] preserve wide fragment release ordering`
  - Carries completed FP32 release dependencies through selected LDS loads.
- `272d73af8 [codegen] reduce wide pipeline register pressure`
  - Generic pressure-aware scheduling.
- `c05188028 [codegen] rematerialize renderer-safe register fills`
  - Allows safe V_CONST replay at ordinary uses.
- `a293391b0 [codegen] localize constrained leases at consumers`
  - Shrinks a DS-load-to-WMMA distance from 242 UOps to at most 24 in the measured case.
- `1de98cc7f [codegen] stop END body register liveness`
  - Removes false completed-body machine liveness; matched peak 184 to 180 in its original comparison.
- `0d91841d3 [codegen] prioritize pressure release in late scheduling`
  - Prioritizes immediate release before opening later recurrence generations.
- `5e9a5c5dc [amd] recover progressive C markers from carriers`
  - Activates the existing serializer for reachable marked carriers; matched spills 525 to 185.
- `412d7998f [amd] recover propagated progressive C roots`
  - Handles multi-step chains whose marked carrier is flattened into a tail machine WMMA; walks to earliest machine
    head; removes all retained WMMA-result spill requests and reduces spills 185 to 112.

Review these commits before changing the same mechanisms. Preserve their focused invariants.

## 12. Known measurement and test caveats

1. `peak_candidate_slots=255` is a union of registers that live virtuals may occupy, not an additive physical-VGPR
   requirement. Ordinary scalar values advertise most of the pool. Use assigned constrained-run overlap and spill
   victims as the actionable evidence.
2. A fail-closed bounded test currently passes by expecting the failure.
3. Detailed debug modes alter allocator traversal cost and have produced different absolute spill counts. Compare only
   matched before/after environments.
4. `/tmp/mmq_marker_before.json` and `/tmp/mmq_marker_after.json` were useful local captures but are not versioned
   authorities and may disappear. Reproduce facts from the committed revision.
5. Four older 16-subtile AMD tests fail historically at the same spill-free gate. They predate the recent scheduler
   commits. Do not mark them xfail merely to hide the resource defect.
6. The old ushort-cast test counted every `v_bfe_u32` mnemonic and was fixed by `2ef1174a9` to inspect packed-load-owned
   extracts only.
7. Keep llama and tinygrad benchmark runs sequential. Concurrent residency is not an admissible comparison.

## 13. After zero-spill bounded emission

Do not jump directly to autoscan. Execute these phases in order.

### 13.1 Full-grid compile and correctness

- Compile `mmq_llama_five_buffer_full_kernel` for aligned full-grid ownership.
- Verify row-major writeback remains `col*N + row` for the oracle's A/B orientation.
- Prove exact five-buffer ABI and launch geometry.
- Verify final ISA has no scratch/spills and resource allocation is within target limits.
- Run full-output correctness, immutable input checks, output guards, timeout isolation, and post-dispatch GPU health.
- Retain the exact candidate identity and binary/ISA proof.

### 13.2 Performance qualification

Fresh Q4 role budgets:

| role | shape MxNxK | current legacy candidate | direct-packed comparator |
|---|---|---:|---:|
| `attn_kv` | 512x1024x5120 | ~5.9996 ms | ~0.4571 ms |
| `attn_qo` | 512x5120x5120 | ~27.927 ms | ~0.920 ms |
| `ffn_down` | 512x5120x17408 | ~67.731 ms | ~3.107 ms |
| `ffn_gate_up` | 512x17408x5120 | ~66.872 ms | ~2.762 ms |

Exact `attn_kv` stage attribution:

- Q8 producer: ~0.0875 ms.
- MMQ: ~5.9160 ms.
- Total: ~6.0070 ms.
- About 98.5% of the tax is MMQ, not activation production.

The new oracle must be timed as the whole primitive, including activation preparation. Q4 target is at least 60 aggregate
logical TFLOP/s unless a newer whole-model role budget proves an equivalent result.

### 13.3 Six-row policy

- Four Q4 rows must bind exact correctness/performance-qualified candidates.
- Two Q6 rows already have qualified direct-packed fallback evidence:
  - `attn_kv` max error 0.015625;
  - `ffn_down` max error 0.015625.
- Build one immutable six-row manual policy.
- Missing evidence, identity drift, and unknown workload must fail closed.
- Keep production promotion false until the full end-to-end gate passes.

### 13.4 Phase 6 end to end

- Execute the manual mixed route on Qwen3-14B.
- Record route census proving every intended packed role fired.
- Reconcile memory/admission and final resources.
- Verify GPU health and output correctness.
- Run decode correctness/performance regression separately.
- Prove one-change rollback to direct packed.

### 13.5 Phase 7 parity and Boltbeam

- Run matched llama/tinygrad prefill at contexts 512, 1,024, 2,048, and 4,096.
- Use at least three alternating pinned-clock sessions.
- Retain raw samples and immutable revisions.
- If any gate misses, run Boltbeam immediately on that exact candidate and optimize the largest measured residual.
- Only after beyond-parity proof may performance autoscan begin.

## 14. Commit and handoff discipline

- Inspect `git status --short --branch` before and after every change.
- Do not bundle diagnostics, compiler repair, policy identity update, and benchmark artifact into one commit.
- Remove temporary logging before committing.
- Run `git diff --check`.
- Preserve user and unrelated changes.
- Push accepted commits.
- If an experiment is ineffective, revert its worktree edits before moving on. Avoid adding a speculative default-off flag
  merely to preserve a failed experiment.
- Record exact before/after measurements near the owning scope document when a blocker class changes.

## 15. Definition of this handoff's completion

Claude's immediate assignment is complete only when:

1. The complete source-pinned bounded oracle emits a real program.
2. Final AMD evidence reports zero spills and zero scratch/stack.
3. The selected mathematical graph, packed ABI, WMMA family/count, barriers, and rounding authority remain unchanged.
4. Focused AMD/regalloc tests and the updated bounded emission test pass.
5. The accepted repair is committed and pushed with an exact evidence note.

That closes only the current compiler-emission gate. Continue through full-grid correctness, performance, six-row policy,
Phase 6, and Phase 7 as described above, then stop before autoscan.

## 16. Recommended first 90 minutes

1. Read this document and the canonical completion scope.
2. Inspect `412d7998f`, especially `_selected_wmma_roots` and `_serialize_progressive_c_drains`.
3. Run the exact reproduction once and confirm 112 `DS_LOAD_B128` spills.
4. Add temporary post-selection diagnostics for the first seven residual A/B pairs and their seven WMMA consumers.
5. Determine the exact selector/lifetime function that opens those pairs early. Do not assume `_try_wmma_kmajor_phase`;
   that path was already changed experimentally with zero effect on the exact workload.
6. Add a focused structural test expressing one-A/B-pair-per-shared-lease ownership.
7. Implement the smallest AMD-local, ownership-derived ordering repair.
8. Run focused tests, then one exact oracle run.
9. Keep the patch only if it is semantically safe and materially reduces the 112 residual spills; continue until zero.
