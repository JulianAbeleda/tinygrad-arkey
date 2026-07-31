# M1a — readiness assessment and first Metal geometry population

Date: 2026-07-30

Status: **Phase 1 complete, verdict PARTIALLY READY (narrow, named gaps). Phase 2 complete: a 23-tuple /
20-identity Metal-legal geometry population for `ffn_gate_up`, recorded, not measured.** Compile-only
throughout; no GPU workload was run. No `PACKED_WMMA_ROUTES` row was added. This document was **not** committed
to git — see "Why this is not committed" at the end.

Repos read: `tinygrad-arkey-exp` (`exp`, clean at `1e503f115`), `BoltBeam` (`main`, clean at `d93102c`). Neither
was pushed or modified.

## Phase 1 — readiness

### Q1. `geometry` — what constrains `(tm, tn, tk, wm, wn, bc)` on Metal

Two constraints are directly evidenced in code, plus one that is enforced only if the caller declares it:

1. **LDS.** `PackedWmmaRoute._candidate_context` (`tinygrad/llm/packed_wmma_prefill.py:137-149`) builds
   `_TileGeometry` with `stride=80`, `a_end = tm*80`, `lds_windows = (A:[0,a_end], B:[a_end, a_end+tn*80])`, so
   `_TileGeometry.lds_bytes` (`packed_wmma_prefill.py:74-75`) `== (tm+tn)*80`. **This is not the real LDS
   footprint when `bc>1`.** `_candidate_context` (line 147) feeds it into
   `KernelStage1PipelinePlan(bc, geometry.lds_bytes, 1)`, and `KernelStage1PipelinePlan.active_lds_bytes`
   (`tinygrad/codegen/opt/kernel_pipeline.py:102-103`) is `buffer_count * slot_bytes`, i.e. the true allocation is
   `bc * (tm+tn)*80`. PG2 checked `lds_bytes=25600 < 32768` (`docs/task_workflow/output/precontract-target-generalization-pg2-result-20260730.md:19`)
   for AMD's `ffn_gate_up` row, which has `bc=1`, so `active_lds_bytes == lds_bytes` there and the distinction was
   invisible. It is not invisible in general: AMD's own `ffn_down` Q4_K row, `(256,128,32,8,2,2)`
   (`packed_wmma_prefill.py:42-43`), has `lds_bytes=30720` and `bc=2`, i.e. `active_lds_bytes=61440` — legal on
   AMD's larger LDS budget, illegal on Metal's declared `max_threadgroup_memory_bytes=32768`. **The correct
   Metal legality bound is `bc*(tm+tn)*80 <= 32768`**, not `(tm+tn)*80 <= 32768`; for `bc=1` this is `tm+tn<=409`,
   for `bc=2` it tightens to `tm+tn<=204`.
2. **Threads.** `geometry.threads = wm*wn*32` (`packed_wmma_prefill.py:142`, wave_size hardcoded 32 — matches
   Metal's declared `subgroup_width=32`). Constrained by `max_threads_per_threadgroup=1024` ⇒ `wm*wn<=32`.
3. **TC axis divisibility.** At TC-opt application time, `postrange.py::_apply_generic_tensor_core_opt`
   (`tinygrad/codegen/opt/postrange.py:394-398`) pads any M/N/K axis whose extent is not divisible by
   `tc.dims[i]`, requiring `opt_level>=2` for the pad. Metal's tensor core is `dims=(8,8,8)`
   (`tinygrad/codegen/opt/tc.py:181`), so the axis extents the TC opt selects must be multiples of 8 (or eat a
   forced PADTO). `tk=32` is used, unmodified, by **all six** existing `PACKED_WMMA_ROUTES` rows regardless of
   role or quant (`packed_wmma_prefill.py:36-47`); `PackedWeightTransform.block_elems` is fixed at 256
   (`tinygrad/codegen/opt/packed_weight.py:147`, `155-156`), independent of `tk`. I could not establish that
   `tk=32` is mathematically forced rather than an inherited AMD campaign choice — flagging it as unconfirmed
   rather than assuming either way.
4. **The generic BoltBeam-side checker only covers two of these.** `propose_legal_dimensions`
   (`extra/llm_research/bubblebeam_futuresight.py:111-141`) and `build_static_legality` (same file, `144-167`)
   check `schedule.tile.{m,n,k}` divisibility against workload shape, `schedule.launch.threads <= max_threads`,
   and `static_constraints.max_local_memory_bytes <= max_threadgroup_memory_bytes` **only if the candidate
   declares that field** (default is `null`, which is vacuously legal). Neither function knows about the
   `(tm+tn)*80` formula, `bc`, or TC-dims divisibility — those are `packed_wmma_prefill.py`/`postrange.py`-specific
   and must be computed and then written into `static_constraints.max_local_memory_bytes` by whoever builds the
   candidate. This is exactly what phase 2 below does.

No new blocker: Q1 is answerable and computable today. The six existing rows' *values* do not transfer (PG2
already says this); the *formulas* do, once corrected for the `bc` multiplier above.

### Q2. `canonical_identity` — can BoltBeam mint one for Metal + 8B today?

**Yes, the general mechanism is target-neutral and already proven on Metal — but the one helper whose name
matches the packed_wmma_prefill.py docstring wording ("buffer-2 candidate templates") is AMD-locked and does not
work for Metal as-is.**

- `canonical_identity` is exactly `FullKernelCandidate.candidate_hash` — `hashlib.sha256(canonical_json).hexdigest()`
  (`BoltBeam/boltbeam/search/spec.py:258-260`). This is pure content hashing; nothing in it inspects backend.
- The **v2** schema (`_validate_v2_candidate`, `spec.py:328-421`) is explicitly documented as "the one
  target-neutral authoring format ... target capability authority is injected by the caller" (`spec.py:167-172`).
  `_validate_target_identity` (`spec.py:423-430`) requires only a non-empty `backend` string plus a SHA-256
  `resolved_target_hash` — no AMD restriction.
  `SearchRow.__post_init__` (`spec.py:494-499`) explicitly bypasses the legacy AMD-only `validate_backend` for v2
  candidates, requiring only that the row's declared backend match the candidate's own target.
  `BACKENDS = frozenset({"AMD"})` / `validate_backend` (`spec.py:69,121-123`) are used **only** by the legacy v1
  `SearchRow` fallback path and unconditionally by `AcceptedPolicy` (`spec.py:555`, a *different*, decode/QK-search
  artifact type unrelated to `PackedWmmaRoute`) — neither is on the path a v2 `FullKernelCandidate` takes.
- This is not theoretical: `BoltBeam/bench/metal-qwen3-8b-20260729/ffn-gate-up-search-request.json` is a real,
  already-executed v2 request with `target_id: apple_m4_10c`, `backend: METAL`, `arch: Apple9`, observed facts
  `max_threadgroup_memory_bytes: 32768`, `max_threads_per_threadgroup: 1024` — i.e. Metal identities have already
  been minted, validated, and searched over on this exact machine (see Q4).
- **The gap:** `BoltBeam/boltbeam/full_kernel_candidate_set.py::build_qwen3_8b_buffer2_candidate_set`
  (lines 125-149) — the function whose "clone one admitted buffer-2 schedule into four exact 8B workload
  descriptors" docstring matches `packed_wmma_prefill.py:32-34`'s "buffer-2 candidate templates" phrase almost
  verbatim — hardcodes `QWEN3_8B_PROFILE = "qwen3_8b_q4k_m_gfx1100"` (line 11) and requires the seed's
  `workload["profile"]` to equal it (line 133-134). It operates on **v1**-shaped payloads (its rebind only
  touches `workload.profile/role/shape` and `applicability`, leaving the v1-only `schedule.pipeline.buffer_count`
  check at line 130 intact). There is no Metal-profile seed template anywhere in the repo for this helper to
  clone from. **This specific convenience function cannot mint a Metal identity today**; a Metal-target seed
  (or a v2-schema equivalent of this helper) would need to be authored first.

### Q3. `canary_max_abs_error` — is exact 0.0 achievable, or is it AMD-specific?

**The reference computation is backend-agnostic; the code that runs it is hardcoded to AMD in three call sites.**
This is a real, narrow blocker distinct from Q2's.

- `extra/llm_research/prefill/packed_wmma_correctness_canary.py::_activation/_q4_blocks/_q6_blocks/_decode_selected_q4/_decode_selected_q6/build_artifact`
  (lines 41-132) compute a fully independent, pure-numpy fp16 reference: a synthetic activation with **exactly
  one nonzero K element per row** (line 41-48) collapses the GEMM to a single scaled dequant term per output
  element, decoded by hand from the packed Q4_K/Q6_K bytes (lines 94-117). This has no GPU, no tinygrad, no
  backend dependence — the same artifact and reference apply to any device.
- The mechanism `packed_wmma_prefill.py:96-97,104-109` calls "the isolated GPU canary" is
  `install_production_qualification_verifier` → `verify_production_row` →
  `run_canary(...)` in `extra/llm_research/prefill/packed_wmma_correctness_canary.py:135-151`. That function
  hardcodes `device="AMD"` in **three places**: `prepare_current_prefill_compile(payload, entry.canonical_identity, device="AMD")`
  (line 143), `compile_device="AMD", runtime_device="AMD"` in the bundle builder (line 145), and
  `health_probe=make_tiny_health_probe(device="AMD")` (line 150). None of these is a keyword with a default that
  `run_canary`'s own signature exposes — the literal is inline.
- `extra/llm_research/prefill/packed_wmma_production_canary.py:16` additionally hardcodes
  `PROFILE = "qwen3_14b_q4k_m_gfx1100"` as the seed-template profile `_payload_for_production_row` rebinds
  (lines 19-32) — same AMD-only-seed-template gap as Q2, and the same root cause (no Metal seed template exists).
- By contrast, the layer *underneath* the canary is already device-generic:
  `extra/llm_research/prefill/host_safety_canary.py` and `isolated_guarded_executor.py` both take `device` as a
  real parameter with no AMD default baked into the call sites that matter (`host_safety_canary.py:93,99,113,127,136,208`).
  So the fix, if made, is shaped exactly like PG0/PG1a/PG1's already-completed work: thread an explicit `device`
  parameter from `verify_production_row`/`install_production_qualification_verifier` down through `run_canary`
  into the three literal-AMD call sites, plus supply a Metal-target seed payload. It is not a deep rearchitecture.
- I could not establish whether achieving exactly `0.0` (not merely small) on Metal is guaranteed by the
  single-nonzero-K-element activation design working as intended, versus something about it depending on AMD's
  specific accumulation order — the design (one nonzero term, decoded independently in numpy) gives no
  mathematical reason to expect a Metal-specific rounding difference, but this was never run on Metal, so I am
  reporting the code-level finding (hardcoded AMD device, three sites) rather than asserting the numeric outcome.

### Q4. Measurement loop — does BoltBeam have a driver for `describe/admit/compile/check/measure`?

**Yes, complete, already wired to Metal, and already exercised end-to-end on this machine.** No gap.

- `BoltBeam/boltbeam/search/full_kernel_search.py::run_full_kernel_search` (lines 144-154) validates a request
  (`validate_request`, lines 73-114), builds candidates via `instantiate_candidates`/`instantiate_candidate_rows`,
  and calls a `worker` per candidate, collecting `MEASURED`/`BLOCKED`/`REJECTED_*` outcomes with score-based
  ranking.
- `BoltBeam/boltbeam/search/tinygrad_full_kernel.py::TinygradWorkerConfig`/`TinygradSearchProviderWorker` is the
  subprocess adapter; `BoltBeam/boltbeam/data/tinygrad_search_compatibility.json` pins the contract:
  `"provider_actions": ["describe","admit","compile","check","measure"]`, protocol
  `"tinygrad.search_provider.v1"`, and — notably — its **default command is
  `["python3","extra/llm_research/search_provider.py","--backend","METAL"]`**. Metal is the pinned default, not a
  special case.
- `tinygrad/extra/llm_research/search_provider.py:19` (`ACTIONS = ("describe","admit","compile","check","measure")`)
  and lines 221-227 map `schedule.transforms[].op` directly to `OptOps[name]` (`from tinygrad.codegen.opt import
  Opt, OptOps; Opt(OptOps[transform["op"]], transform.get("axis"), transform.get("arg"))`) — so `"op": "TC"` is
  already generically supported; no code change is needed to express a TC opt through this transport.
- This has already run on Metal on this exact machine:
  `BoltBeam/bench/metal-qwen3-8b-20260729/ffn-gate-up-search-result.json` is a `status: COMPLETE` result with
  `counts: {measured_correct: 12, blocked: 0, rejected_unsupported: 1}` against the pinned
  `apple_m4_10c`/`METAL`/`Apple9` target, correctness oracle `"canonical_packed_reference"`, real median-ns
  measurements, and `bench/metal-qwen3-8b-20260729/README.md` documents the exact replay command
  (`boltbeam.cli search-full-kernel ... --worker-command '[".venv/bin/python","-m","extra.llm_research.search_provider"]'`).
  That prior run was **decode** phase (`m=1`) with only `LOCAL`/heuristic transforms (no `TC` row was in its
  candidate space) — it demonstrates the loop mechanism, not a TC-opt prefill search — but the mechanism itself
  needs no further construction.

### Verdict: PARTIALLY READY

Two of four questions (geometry legality, the measurement loop) are unqualified READY. The other two share one
root cause, named precisely:

**Gap — no Metal-target "buffer-2" seed candidate template exists yet**, in either repo. Concretely:
`BoltBeam/boltbeam/full_kernel_candidate_set.py::build_qwen3_8b_buffer2_candidate_set` is locked to
`profile="qwen3_8b_q4k_m_gfx1100"`, and `tinygrad/extra/llm_research/prefill/packed_wmma_production_canary.py`'s
`PROFILE = "qwen3_14b_q4k_m_gfx1100"` seed lookup feeds the actual canary that would produce
`canary_max_abs_error`. Both need a Metal-profile seed authored before they can run for Metal, and the canary
runner additionally hardcodes `device="AMD"` in three call sites in
`extra/llm_research/prefill/packed_wmma_correctness_canary.py::run_canary` that must be parameterized. This is
the same *shape* of fix PG0/PG1a/PG1 already made elsewhere (derive/parameterize instead of hardcode) — narrow,
not architectural — but it is real, unfinished work, not a hypothetical. Until it is done, `gate_combo` cannot
legitimately admit any Metal row, because nothing today can produce a `canary_max_abs_error` for one.

This satisfies the phase-2 trigger ("READY or the gap is narrow"); it does **not**, on a strict reading, satisfy
"phase 1 is green" for the separate commit rule — see the closing note.

## Phase 2 — first Metal-legal geometry population, `ffn_gate_up`

Target facts (Q1, verified `max_threadgroup_memory_bytes=32768`, `max_threads_per_threadgroup=1024`, wave/subgroup
size 32, `tc.dims=(8,8,8)`). Workload: `Q4_K`, role `ffn_gate_up`, shape `m=512, n=12288, k=4096` (Qwen3-8B
Q4_K_M gate/up, prefill). Method, exactly as instructed:

1. **`propose_legal_dimensions`** (`extra/llm_research/bubblebeam_futuresight.py:111`) filtered candidate values
   for `schedule.tile.{m,n,k}` (divisibility against 512/12288/4096) and `schedule.launch.threads` (`<=1024`).
   Run for real; all supplied multiples-of-8 candidates passed (they were pre-filtered to be sensible):
   `tile.m ∈ {8,16,32,64,128,256,512}`, `tile.n ∈ {8,...,1024}` (14 values), `tile.k ∈ {8,...,1024}` (8 values),
   `launch.threads ∈ {32,64,...,1024}` (32 values). `dimension_mapping` packaged these into the documented
   per-axis legal sets.
2. From those legal per-axis sets I hand-built **coupled rows**, not a cartesian product: 9 `(tm,tn)` pairs —
   `tm ∈ {64,128,256}` × `tn ∈ {32,64,128}` — each legality-checked jointly against the *corrected* Q1 formula
   `bc*(tm+tn)*80 <= 32768`, paired with 1-2 compatible `(wm,wn)` splits per pair (`wm | tm/8`, `wn | tn/8`,
   `wm*wn<=32`), `tk=32` (matching all six existing rows, flagged in Q1 as unconfirmed-required rather than
   assumed), and `bc=1` for all 9 pairs plus `bc=2` for the 5 pairs whose tighter bound `tm+tn<=204` is satisfied
   (`(64,32),(64,64),(64,128),(128,32),(128,64)`). This produced **23 legal `(tm,tn,tk,wm,wn,bc)` tuples**, none
   exceeding `active_lds_bytes<=32768` or `threads<=1024`.
3. Each tuple was written into a `boltbeam.full_kernel_candidate.v2` row via
   `{"schedule.tile.m":tm, "schedule.tile.n":tn, "schedule.tile.k":tk, "schedule.launch.threads":wm*wn*32,
   "static_constraints.max_local_memory_bytes": bc*(tm+tn)*80}` against a seed candidate (target
   `apple_m4_10c`/`METAL`/`Apple9`, `resolved_target_hash` and `model_sha256` reused verbatim from
   `bench/metal-qwen3-8b-20260729/ffn-gate-up-search-request.json` since they are this machine's already-recorded
   facts, `schedule.transforms=[{"op":"TC","axis":0,"arg":[-1,2,1]}]` matching the forced opt every existing row
   uses at `packed_wmma_prefill.py:163`, `schedule.compute.family="packed_wmma_dequant_fused"`,
   `correctness.oracle="canonical_packed_reference"`). Fed to `BoltBeam.instantiate_candidate_rows(seed, rows,
   max_candidates=256)` (`BoltBeam/boltbeam/search/full_kernel_candidates.py:49`) — run for real, not simulated.

**Result: 20 unique `candidate_hash` identities from the 23 legal tuples** (all well under the `max_candidates=256`
cap in both `instantiate_candidate_rows` and `instantiate_candidates`). The 3 duplicates are a genuine, load-bearing
finding, not an error: for the three `(tm,tn) ∈ {(256,32),(256,64),(256,128)}` pairs, *both* proposed `(wm,wn)`
splits saturate `threads=1024` (e.g. `wm=32,wn=1` and `wm=8,wn=4` both give `32*32` no — `wm*wn*32=1024` either
way), and BoltBeam's v2 schema records only `schedule.launch.threads` (the product) and `schedule.tile.{m,n,k}` —
it has **no field for the `(wm,wn)` split itself**, and (confirmed while assembling the seed) **`schedule.pipeline`
in v2 has only `stage_count` — no `buffer_count` field at all** (`spec.py:349`, strict-keys `{"stage_count"}`,
vs. v1's `{"buffer_count","stage_count","epoch_graph"}` at `spec.py:284`). This is exactly why
`packed_wmma_production_canary.py::_payload_for_production_row` builds its oracle payload in the **v1** wire
format instead of v2 — v1 is the only schema that can express `buffer_count` at all. `bc` is therefore recorded
in this population as an out-of-schema annotation (the table below), not inside the v2 candidate payload; a
future promotion pipeline for Metal will face the identical v1-vs-v2 schema mismatch the AMD pipeline already
resolved by using v1.

Re-validated all 20 through BoltBeam's own generic checker for good measure:
`build_static_legality(workload_facts, target_facts)` (`bubblebeam_futuresight.py:144`) via `classify_candidates`
— **20 accepted, 0 rejected.**

Full population (`tk=32` throughout; `bc` and `wm,wn` are the tinygrad-side annotation not present in the v2
hash):

| tm | tn | wm | wn | bc | threads | active LDS | candidate_hash (v2) |
|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 32 | 8 | 1 | 1 | 256 | 7680 | `3dae0a988a7095bc063c88f36d9d027ea000688dfbcd95b43e34411ba9a928dc` |
| 64 | 32 | 8 | 4 | 1 | 1024 | 7680 | `8157bb74e76ab7941194ba7e179bea40421badef2fdc228e2b0d84e8e01f8ea8` |
| 64 | 64 | 8 | 1 | 1 | 256 | 10240 | `02d808bfab6ea4f19f5aa338f235e5437ddb032c811e74ee26d9d0ed9c1dc5ec` |
| 64 | 64 | 4 | 8 | 1 | 1024 | 10240 | `ea1f11b9fecb2344a5866b1723f263219f1134484a02efe8ababdf6527462179` |
| 64 | 128 | 8 | 1 | 1 | 256 | 15360 | `2f8f873ef8199ef41b72d646e351fda0ee975c5d1faf01abf503d104df8eedad` |
| 64 | 128 | 2 | 16 | 1 | 1024 | 15360 | `adc9786c0fd0cad28ae1f76b4b30fcf092677b8fed9c10f4bb475902d07fe1da` |
| 128 | 32 | 16 | 1 | 1 | 512 | 12800 | `b03a8e5c126bace567794a09884a4bb57a6f1529cd976d7d2172d5e674d2c53d` |
| 128 | 32 | 8 | 4 | 1 | 1024 | 12800 | `6e7864e9225f44cf8c48aa729e3c6d2ce5532f8d2107aea60603f7bc6b52981b` |
| 128 | 64 | 16 | 1 | 1 | 512 | 15360 | `07dfd84cdca548bddaf9aced0936b1f33194e1b344a8af747dc830a5dad92498` |
| 128 | 64 | 4 | 8 | 1 | 1024 | 15360 | `2a1bec5c9524378835bec24453d07509dc8158ae4d7e4c277ae9de0d25859f2e` |
| 128 | 128 | 16 | 1 | 1 | 512 | 20480 | `f83e91a45e25d9067f43f4db9ee3839dfb125afa28514985cb38b32742370426` |
| 128 | 128 | 2 | 16 | 1 | 1024 | 20480 | `114b2472d3ed4fd29effff38d693148530ddd0c43a24edf1bc6c968245de9bd8` |
| 256 | 32 | 32 or 8 | 1 or 4 | 1 | 1024 | 23040 | `9f7b25c7d8217a9fb67b28c5949b03ba94c4bdd36c46a6100302a14394451de4` (both splits collide) |
| 256 | 64 | 32 or 4 | 1 or 8 | 1 | 1024 | 25600 | `89d8737528288092c42ca868fe761c8b93ada12437e81155188bb9477084ac08` (both splits collide) |
| 256 | 128 | 32 or 2 | 1 or 16 | 1 | 1024 | 30720 | `027d63a0a45a4c29cb56ced10584b21bc35cdc0a605bad1db5f202949e1acb22` (both splits collide) |
| 64 | 32 | 8 | 1 | 2 | 256 | 15360 | `dee13db762ff169473679585c3a81550cee416f8d40139da4e28077e128b3164` |
| 64 | 64 | 8 | 1 | 2 | 256 | 20480 | `73a41bc1965b920369ee91606125725c155360f92757d82d1cb775a4a437aae6` |
| 64 | 128 | 8 | 1 | 2 | 256 | 30720 | `348b825bcfd421a399d1be6f061c09ba59cbbce6aa988b73c8fe8d0e1905d1e2` |
| 128 | 32 | 16 | 1 | 2 | 512 | 25600 | `2e03aafce7e5532dbc008193df8215a2695b182d00280214d8e91c8fdc35f8af` |
| 128 | 64 | 16 | 1 | 2 | 512 | 30720 | `8d74abfe42dec137e7d47e64e5045afd8a0c7bbeeba7af35f1eea3f8fc798747` |

Seed candidate hash (before row substitution): `3dae0a988a7095bc063c88f36d9d027ea000688dfbcd95b43e34411ba9a928dc`
(identical to the first row, since that row's values equal the seed's own tile/threads/LDS defaults — expected,
not a bug).

### What measuring this population would require

In order, none of which was done here:

1. **Close the Q2/Q3 gap** — author a Metal-profile seed template usable by
   `build_qwen3_8b_buffer2_candidate_set` (or a v2-schema-native equivalent), and thread an explicit `device`
   parameter through `run_canary`/`verify_production_row`/`install_production_qualification_verifier`
   (`extra/llm_research/prefill/packed_wmma_correctness_canary.py`,
   `extra/llm_research/prefill/packed_wmma_production_canary.py`) so the isolated GPU canary can target Metal
   instead of the three literal `"AMD"` call sites. Without this, no row from this population can acquire a real
   `canary_max_abs_error`, and `gate_combo` cannot admit it regardless of measured speed.
2. **Resolve the `bc`/`(wm,wn)` schema gap** noted above — decide whether the eventual promotion record carries
   `bc` as a v1-format oracle payload (as AMD's does) or extends v2 with an explicit pipeline `buffer_count`
   field; either way, the 3 collided identities above mean `bc`/`(wm,wn)` cannot be promoted as distinguishable
   v2 candidates until that decision is made.
3. **Run the already-working loop** — `boltbeam.cli search-full-kernel` against
   `extra/llm_research/search_provider.py --backend METAL` (Q4; no construction needed, already proven on this
   machine for this exact role, just not yet for a prefill/TC-opt candidate space) — to get real `median_ns` per
   candidate and `correctness.passed` against the same `"canonical_packed_reference"` oracle used by the prior
   decode-phase run.
4. **A finalist/control comparison**, matching the discipline `mr9_semantic_search.py::_decision`
   (`BoltBeam/boltbeam/search/mr9_semantic_search.py:143-205`) already enforces: a `tinygrad_heuristic.v1` control
   row measured alongside the candidates, ≥2 stable repeats of the top finalists, and a minimum win fraction
   before any row is a promotion candidate — not merely "fastest of 20."
5. **GPU time on this Apple M4** — none was used in this task; every number above is a compile-time/derivation
   result (LDS/thread arithmetic, real `FullKernelCandidate` construction and hashing, real
   `propose_legal_dimensions`/`build_static_legality` calls), not a measurement.

## Why this is not committed

The task's commit rule is "commit only what phase 2 produces, and only if phase 1 is green." Phase 1's verdict
here is **PARTIALLY READY** with two named, real gaps (Q2's missing Metal seed template, Q3's AMD-hardcoded
canary runner) — not the unqualified "verdict GREEN" that PG2 recorded. Reading the commit rule strictly, that
does not clear the bar, even though it does clear the separately-stated, looser "READY or the gap is narrow"
bar for doing phase-2 work at all. This document is therefore written but left **uncommitted**; whether to commit
it (and under what prefix — `[docs]`, matching PG2's own commit, is the obvious choice) is left to whoever reads
this result, consistent with the instruction that a partial verdict is itself valuable output.
