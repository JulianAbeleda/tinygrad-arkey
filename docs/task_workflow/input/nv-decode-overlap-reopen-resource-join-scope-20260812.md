# NV decode launch-hiding reopen scope - resource-join capture at HEAD

Date: 2026-08-12
Branch: `nvidia-bringup-20260731` (HEAD `a8b560457`, fp32 q/k promoted;
gap attribution `nv-decode-gap-attribution-same-session-20260812.md` step 8)
Status: **implementation/tooling scope. Authorizes rebuilding the
duration-bearing kernel-named DAG and the compiled-descriptor census at HEAD
(CPU-only + one lock-held capture), rerunning the co-schedule scan and the
resource join, and only then the two GPU arms named below.** Follows the
standing process: audit -> arithmetic -> implement. No production default
changes; no promotion without the +50 us wall bar.

## 1. Why this scope exists

The gap attribution (same-session pair, d512, Qwen3-8B-Q4_K_M / RTX 5090)
leaves one open row on the way to llama parity: the launch-hiding/overlap
delta. llama replays one CUDA graph at 94% GPU busy (span 3835 us + host gap
212 us = 4074 us wall); tinygrad replays its JIT at ~92% busy (busy ~4790 us
estimate + host/gap ~413 us = 5203 us wall). The kernel-work delta is the
652 us class table; the rest (~476 us) is launch hiding and host behavior.

The 08-05 mechanism enumeration (`nv-overlap-exhaustive-scope-20260805.md`)
closed every in-graph co-schedule arm on the old topology: ceiling 17.952 us
vs the +50 us gate (greedy 10.016, best pair 1.792), two-queue cuts closed at
the calibrated 3.1865 us/wait (Q-cut -10.474, K-cut +42.962 < 50). Its one
open CPU step, mechanism (e), was the resource-complementarity join, which
failed closed for lack of compiled resource tuples. This scope reopens (e)
at HEAD with the census that has since changed twice (715 -> 630 -> 594
kernels via M2a-d + fp32 q/k), and rebuilds the evidence the 08-05 verdict
depends on, which was lost in the machine restart.

## 2. Audit (fresh, 2026-08-12)

### 2.1 What exists in-repo (consumers of the authority DAG)

- `nv_co_schedule_candidate_scan.py`: scores dependency-independent
  (support behind quant/flash host) pairs; gate +50 us exact critical-path
  recovery; needs a duration-bearing, kernel-named DAG (names q4k/q6k/flash
  hosts + E_/r_ support).
- `nv_dependency_closed_cut.py` + `nv_wait_adjusted_cut_forecast.py`: queue
  cuts at the calibrated wait cost; gate +50 us.
- `nv_overlap_resource_join.py` + `route_b3_dag_attribution.py`
  (`attach_compiled_descriptors`, fail-closed seam): resource-complementarity
  join; currently `INCONCLUSIVE_FAIL_CLOSED` (tinygrad census has no
  compiled resource tuple; llama manifest has no local-memory field).
- `route_b3_dag_attribution.py --capture-cuda`: aligned logical/physical
  capture of the executed decode (kernel-named nodes, group ids, semantic vs
  planner-alias edges). 08-04 record: 1021 kernels / 6 groups, alignment
  PASS. The capture seam is device-agnostic; NV needs a mode flag + descriptor
  emitter (section 5).
- `cuda_duration_attach.py`: attaches CUPTI durations to the aligned capture.
  NV has no CUPTI; durations come from the DEBUG=2 prime trace instead.

### 2.2 What is missing (lost or never built)

1. The authority DAG `/tmp/ln_20260805.json` (875 nodes / 4080 edges, digest
   `49838b8a...`) and its builder were `/tmp` scratch; both are lost in the
   restart. The 08-05 verdicts cite it as "not regenerated".
2. A fresh full-token capture at HEAD (`full_token_dag_capture.py --capture`,
   run this session, evidence
   `docs/task_workflow/evidence/nv-dag-presplit-head-20260812.json`) records
   **874 nodes, 3303 edges, 333 cross-group** but only `E_/r_` pre-split
   names: **zero q4k/q6k/flash host nodes and no durations** (profile attach
   has no source at HEAD). It cannot feed the scan as-is.
3. No compiled-descriptor emitter exists for NV. The NV backend already
   exposes what the join requires on each compiled kernel
   (`tinygrad/runtime/ops_nv.py`): `regs_usage`, `shmem_usage`,
   `lcmem_usage`, plus launch grid/block and the program binary for SHA-256.

### 2.3 Fresh numbers this scope is built on

- llama side (same session): `quantize_q8_1.hidden_behind_mmq_us`
  = **391.297 us** (was 445.954 at the 08-04 composition); per-replay
  overlap accounting 946 us on 762 nodes; inter-replay host gap median
  **212.5 us**.
- tinygrad side: production wall **5.2031 ms/token**; prime kernel-sum
  **5424.9 us**; census **594 kernels** at HEAD (630 post-M2b/M2c, 715
  pre-M2a).
- tinygrad host/gap estimate at HEAD: wall 5203 - busy ~4790 = **~413 us**
  (92% busy), vs llama 212 us + in-graph launch (94% busy). The 0.883
  replay factor is a campaign estimate, not a HEAD measurement; one audit
  item below re-measures it.

## 3. Arithmetic

### 3.1 What each mechanism can actually return (not the llama bucket)

llama's 391 us hidden mass is not transferable: it is llama's own
quantize_q8_1 cost structure (217 nodes) hidden inside its mmvq interval.
tinygrad folded that work into its GEMV bodies, so the tinygrad-side
co-schedule ceiling is the support-behind-host exposure of the tinygrad DAG,
measured at 17.952 us on the old topology. The same holds at HEAD until the
fresh scan says otherwise; the resource join cannot exceed the topology
ceiling (08-05 exhaustive scope, mechanism (e)).

| mechanism | ceiling (old topology) | gate | status |
| --- | ---: | --- | --- |
| in-graph co-schedule (scan) | 17.952 us (greedy 10.016) | +50 us exact CP recovery | re-measure at HEAD |
| two-queue cuts (wait-calibrated) | Q -10.474 / K +42.962 | +50 us costed | closed; re-gate only on fresh DAG |
| resource-complementarity join | capped by scan ceiling | positive CTA bound AND +50 us | this scope's arm |
| host-gap compression | ~174-413 us gap | host-forward scope mechanisms | mostly closed/landed; re-measure |

### 3.2 Tok/s translation at the 5.2031 ms baseline

- Scan ceiling ~18-40 us (if it moved): +0.7 to +1.5 tok/s. Even a gate
  pass does not reach the +17 tok/s the naive 391 us llama bucket suggests.
- Host-gap recovery: every 25 us of host time is +1 tok/s; the gap delta vs
  llama is ~174-413 us -> +7 to +16 tok/s only if a mechanism survives the
  host-forward closures (predispatch, copies, JIT signature, scalar copyout
  were measured; see the host-forward scope for what remains).
- The row's honest ceiling at parity is the sum of the scan ceiling and a
  surviving host mechanism; this scope's job is to produce the evidence that
  decides it, not to book it.

## 4. Gates (hard stop, mirrors 08-05)

No GPU arm for overlap recovery until one of these passes on a fresh
duration-bearing DAG from the current closed model graph:

1. Co-schedule forecast clears **+50 us** exact critical-path recovery
   (`nv_co_schedule_candidate_scan.py`).
2. The resource join names a dependency-independent pair with a positive
   complementary CTA residency bound AND exact recovery >= 50 us
   (`nv_overlap_resource_join.py` after descriptor census augmentation).
3. A fusion population clears the boundary-free gate and books an
   exact-output native A/B (mechanisms c/d; the reduce-output folds are the
   active workstream).

Cross-queue cuts stay quarantined until a future occurrence-pinned cut clears
+50 us at the calibrated wait cost.

## 5. Implement plan (tooling first, all CPU-only except the capture)

### P1: NV aligned capture + descriptor emitter (this scope's scaffold)

1. Add `--capture-nv` (or extend `--capture-cuda`) to
   `route_b3_dag_attribution.py` for the NV backend; record per occurrence:
   ordered `CallRecord` (index, name, accesses, group_id, identity) plus one
   compiled descriptor carrying exactly
   `binary_sha256, grid, block, registers_per_thread, static_smem_bytes,
   dynamic_smem_bytes, local_mem_bytes` from the NV compiled kernel
   (`regs_usage`, `shmem_usage`, `lcmem_usage`, launch geometry, program
   binary SHA). Missing/extra/partial rows raise; names never substitute for
   occurrence identity (the existing `attach_compiled_descriptors` contract).
2. NV duration attach: join the DEBUG=2 prime trace
   (`/tmp/tg_debug_probe_20260812.log`) to capture occurrences positionally
   within each group, mirroring `cuda_route_aligned_census.align_capture`;
   any signature mismatch fails closed (group marked UNKNOWN, never zeroed).
3. Emit the HEAD duration-bearing kernel-named DAG (the `ln_20260805.json`
   replacement) with schema digest recorded.

### P2: rerun the decision tools at HEAD

1. `nv_co_schedule_candidate_scan.py` on the fresh DAG -> ceiling / greedy /
   top pairs vs +50 us.
2. `nv_overlap_resource_join.py` with the fresh census -> complementary pair
   or `INCONCLUSIVE_FAIL_CLOSED` again.
3. `nv_wait_adjusted_cut_forecast.py` on the fresh DAG (quarantine re-check).
4. Re-measure the tinygrad steady replay busy (the 0.883 factor) with a
   dedicated replay probe so the host-gap row has a HEAD number.

### P3: GPU arms (only if a gate passes)

1. Native co-schedule probe: the multi-compute-queue device slice
   (`nv-multi-compute-queue-execution-scope-20260803.md`): does GB202 /
   driver 595.84 co-schedule independent kernels from two channels?
2. Two-queue span A/B on the named pair: exact-token sha, census, reverse
   wall bracket, +50 us promotion bar, promotion record.

### P4: promotion contract

Standard workstream gates: exact-output native A/B, census PASS, logits
SHA-256 identical, +50 us reverse wall bracket, promotion record in
`docs/task_workflow/input/`.

## 6. Evidence

- `docs/task_workflow/evidence/nv-llama-d512-node-ledger-20260812.json`
  (llama node ledger: span 3835 us, hidden 391.297 us, gap 212.5 us)
- `docs/task_workflow/evidence/nv-tinygrad-prime-gap-table-20260812.json`
  (prime table: 582 rows / 5424.9 us)
- `docs/task_workflow/evidence/nv-dag-presplit-head-20260812.json`
  (fresh pre-split capture: 874 nodes / 3303 edges / 333 cross-group;
  records the host-node and duration gap this scope closes)
- Raw: `/tmp/llama_tg10_node_20260812.sqlite`,
  `/tmp/tg_debug_probe_20260812.log`
