# M4 residual_add S4 gate run record - GPU section-6 gate, substrate landed

Date: 2026-08-06
Branch: tinygrad `nvidia-bringup-20260731`, HEAD `c594799c1` (S1-S3 deltas landed, tree
clean)
Status: **gate run record, FAIL. The open arm (production residual fold ACTIVE) crashes at
render time on NV with a weakint `SPECIAL` `type_verify` failure; the section-6 gate cannot
pass at any depth. No promotion: `decode-q4k-epilogue-resadd-route-policy.json` stays
CLOSED (`promoted_targets: []`), 0 credit booked.** Authorities:
`m4-resadd-landing-scope-20260806.md` section 3 (gate items 1-5),
`m4-resadd-rangeify-substrate-scope-20260806.md` S4 (re-run protocol).

## 1. Protocol

Same-session, lock-held (`flock -w 600 /tmp/gpu-bench.lock`), Qwen3-8B-Q4_K_M
(`/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`), nmeas 20, reps 3, median tok/s,
temperature 0, chunk_size 32, fused prefill attention disabled (house convention).
Runner: `extra/llm_research/decode/m4_resadd_section6_gate.py`, per-arm fresh
subprocesses (`--arm {closed,open,record} --depth {512,2048,4096}`), each under
`timeout 1200`. Open mode = module override
`mrp._DECODE_Q4K_EPILOGUE_RESADD_PROMOTED_TARGETS = frozenset({("NV","sm_120")})` with the
production fold ACTIVE; closed = default records; record = checked-in policy JSON.
GPU: NVIDIA GeForce RTX 5090 (sm_120), ~21.8GB free with the resident llama-server
processes (daycare/arkey, pre-existing, not gate interference).

## 2. Results

| arm | d512 | d2048 | d4096 |
| --- | --- | --- | --- |
| closed | **PASS** 180.982 tok/s (reps `[6.538, 181.019, 180.982]`), sha `227ad3ce...` 3/3, first `271` 3/3, census 948 kernels / epi 0 / legacy 72 / copy class 1 / resadd 72 | **pre-existing HCQ hang** (`Wait timeout: 30000 ms!`, `runtime/support/hcq.py:300`; "NV synchronization failed before finalizing") | **pre-existing HCQ hang** (same) |
| record | **PASS** 180.376 tok/s, sha `227ad3ce...` 3/3, first `271` 3/3, census 948 / epi 0 / legacy 72 / copy 1 / resadd 72 | **pre-existing HCQ hang** | not run (capped by the known hang; record == closed proven at d512) |
| open | **FAIL: render crash** (deterministic, reproduced 2/2) | **FAIL: same render crash** | compiled and ran kernels on GPU, then pre-existing HCQ hang |

pg3 legacy render sha `27857cb8ca03` for `q4k_g3_lanemap_gemv_4096_4096` unmoved in every
arm that produced a result (closed d512, record d512).

## 3. Failure mode (open arm)

Open d512 and d2048 crash deterministically at render time in the precompile-kernels walk:

```text
RuntimeError: UOp verification failed at 31 on Ops.SPECIAL dtypes.weakint 1
[(Ops.CONST, dtypes.weakint, 4096)] gidx0
```

`tinygrad/uop/spec.py:69` (`type_verify`) via `tinygrad/codegen/__init__.py:337`
(`if SPEC: type_verify(sink, spec_program)`) inside `_full_rewrite_to_sink`. The failing
node is a weakint-typed `SPECIAL gidx0` with src `CONST weakint 4096`. `spec_program`'s
weakint catch-all (`(UPat(GroupOp.All, dtypes.weakint), lambda: False)`, `spec.py:490`)
matches first; the permissive `SPECIAL` rule (`spec.py:237`, in `spec_shared`) is ordered
after the catch-all in the concatenated matcher and never fires for weakint `SPECIAL`s.
The old `bad reshape` schedule crash is gone (schedule time succeeds; 1620 kernels open),
so this is a NEW blocker surfaced by actually running the fold on NV - exactly the class
the S3 CPU-only host proof could not see (CPU render has no gpudims/`SPECIAL`).

## 4. Gate verdict

FAIL. Gate item 1 (open-mode wall vs M2-on baseline with a positive delta) is unattainable:
the open arm cannot render on NV at d512/d2048, and d4096 is capped by the pre-existing
HCQ hang (which also reproduces on closed/record arms, confirming it is environmental and
not fold-related). Items 2-4 (census, pins, pg3 sha) are consequently unattainable in open
mode. Closed/record d512 pass with the re-derived pins (`227ad3ce...`, first `271`), so the
landing wiring is dormant-correct and the tree itself is sound.

## 5. Decision

- `decode-q4k-epilogue-resadd-route-policy.json`: **stays CLOSED**, `promoted_targets: []`.
- 0 credit booked. No recovery row added.
- Substrate deltas D1/D2 NOT reverted (correct, unit-locked; hermetic gate 68 passed incl.
  `test_m5_typed_boundary` 27/27 on this tree).
- Follow-on scope required: make weakint `SPECIAL` renderable on NV (either let the
  `spec_shared` `SPECIAL` rule precede the weakint catch-all, or type the folded residual
  index as int32 at the emitter/`pm_index_is_shrink` boundary), then re-run S4.

## 6. Evidence

`/tmp/m4_gate_closed_d512.json`, `/tmp/m4_gate_closed_d2048.json`,
`/tmp/m4_gate_closed_d4096.json`, `/tmp/m4_gate_open_d512.json`,
`/tmp/m4_gate_open_d2048.json`, `/tmp/m4_gate_open_d4096.json`,
`/tmp/m4_gate_record_d512.json`, `/tmp/m4_gate_record_d2048.json`; tracebacks in
`/tmp/m4_gate_open_d512.err`, `/tmp/m4_gate_open_d2048.err`,
`/tmp/m4_gate_open_d4096.err`. Missing arm: record d4096 (not run; record mode proven ==
closed at d512, d4096 capped by the known hang).
