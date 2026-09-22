# NVIDIA prefill missing-substrate closure

Date: 2026-08-29

## Verdict

The three previously missing execution substrates are implemented and executable.
This does not promote any candidate into model composition; packet-level performance
gates remain authoritative.

## E1 many-row vocabulary

Substrate verdict: **PASS**.

- Corrected compact-Q8 byte ordering and per-32-value scale broadcast.
- Full `(1,1,151936)` output is finite and nonzero.
- Full-logit oracle: `max_abs=0.093423`, relative L2 `0.005383`.
- Candidate and reference argmax both equal `8503`.
- Matched lifecycle R9 runner and evidence now exist.

Packet verdict: **STOP**. Candidate median is `16,449.309 us` versus
`3,871.647 us` control, a `+12,577.662 us` delta (about `4.25x` slower).
E2 remains unauthorized.

Authorities:

- `tinygrad/llm/q6k_vocab_manyrow.py`
- `extra/llm_research/prefill/nv_vocab_e1_lifecycle_r9.py`
- `docs/task_workflow/evidence/nv-vocab-manyrow-e1-r9-20260829.json`
- `docs/task_workflow/output/nv-prefill-vocab-manyrow-primitive-20260829.md`

## D0 HCQ submission observation

Substrate verdict: **PASS**.

- Added an inert-by-default Buffer observer for actual allocation, copy-in, and
  copy-out calls.
- Added opt-in `HCQ_SUBMISSION_OBSERVER_JSON` records after completed HCQ graph
  submissions.
- Records contain device timestamps, segment-local dependencies,
  dependency-ready timestamps, queue/device identity, metadata, PID, and invocation.
- Observer-off self-test emitted no records.
- Observer-on self-test captured four submissions, eight graph entries, and all
  three Buffer event kinds: `alloc`, `copyin`, and `copyout`.

Packet verdict: **BLOCKED**, unchanged. The existing Q6-down capture was taken
before this substrate existed and does not contain paired FP16/Q6 hot/cold R9 at
the four forced lifecycle boundaries. D1 remains unauthorized until that new
measurement campaign names one dominant removable boundary or stops the lane.

Authorities:

- `tinygrad/device.py`
- `tinygrad/runtime/graph/hcq.py`
- `extra/llm_research/prefill/nv_hcq_submission_observer_selftest.py`
- `docs/task_workflow/evidence/nv-prefill-q6down-d0-20260829/submission-observer-selftest.json`
- `docs/task_workflow/evidence/nv-prefill-q6down-d0-20260829/submission-observer-selftest.jsonl`

## F1 Flash vector primitive

Substrate verdict: **PASS**.

- NVRTC compile and launch execute on RTX 5090 `sm_120`.
- Exact launch geometry is `(32,8,1)x(128,1,1)` with GQA mapping
  `kv_head = q_head // 4`.
- Frozen full-output oracle passes: `max_abs=3.01972e-5`,
  `mean_abs=4.70298e-7`.
- All outputs are finite and written; Q/K/V read-only hashes pass.
- Shared memory is `32768` bytes and binary size is `13632` bytes.

Packet verdict: **STOP / incomplete qualification**. The installed-Flash
comparator, exact 36-call hot/cold R9 driver, and usable register/spill/occupancy
counters do not yet exist. F2 remains unauthorized.

Authorities:

- `extra/llm_research/prefill/nv_flash_vkv_primitive.py`
- `docs/task_workflow/evidence/nv-prefill-flash-vector-topology-20260829/f1-result.json`
- `docs/task_workflow/output/nv-prefill-flash-vector-primitive-20260829.md`

## Next authorized work

1. Run a fresh D0 four-boundary paired FP16/Q6 hot/cold campaign using the new
   observers. This is the only lane still blocked by previously missing
   measurement substrate.
2. Do not spend composition time on E1; its matched lifecycle result is decisively
   slower.
3. Build the installed-Flash comparator and exact 36-call replay before making any
   F1 performance or F2 composition claim.
