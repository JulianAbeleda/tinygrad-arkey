# M4 residual_add S4 gate run record - GPU section-6 gate, substrate landed

Date: 2026-08-06
Branch: tinygrad `nvidia-bringup-20260731`, HEAD `4f3a84858` (S1-S3 deltas landed, tree
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

## 7. Follow-on investigation 2026-08-07: the blocker is the precompile bodies, not weakint SPECIAL

A full open-arm scan (no early break; fake NV sm_120 facts, CPU, Qwen3-8B-Q4_K_M,
`/tmp/m4_s5_scan.py` + `/tmp/m4_s5_diag*.py`) established the real shape of the blocker.

**The weakint SPECIAL was the first failure, not the root.** The open-arm schedule embeds
raw precompile artifacts in every composite landing kernel, not only 8: ~630 of the 1620
open kernels fail at render, all with the same weakint `SPECIAL` `type_verify` crash. Each
composite's SINK is a small shell over:

1. `GETTUPLE(FUNCTION(FFNBlock._run, precompile=True), 0)` - the whole FFN precompile body
   (3733+ nodes, 23-245 args) spelled raw, because `rangeify.resolve_function` skips
   precompile functions (`schedule/rangeify.py:576`).
2. `AFTER(buf, CALL)` dependency markers (8-44 per kernel) whose CALL bodies are raw
   custom-kernel ASTs (`q4k_g3_lanemap_gemv_*`, `flash_*`, `q6k_gen_*`). All embedded CALL
   bodies render standalone (verified 44/44) and have structurally matching standalone
   schedule items (35-36/36, 18/18 by node/store census), so they are redundant in-body
   spellings of standalone kernels.
3. `MEMORY_SEMANTIC` role markers on the inlined values.

**A codegen-side resolution pass was tried and is insufficient.** A pass at the top of
`_full_rewrite_to_sink` that inlines the precompile GETTUPLE (substituting body PARAMs for
function args), resolves `AFTER(buf, CALL)` to `buf`, and strips `MEMORY_SEMANTIC` clears
the weakint `type_verify` failure and renders the pure-shell composites. But the inlined
precompile bodies also carry `AFTER(PARAM, CALL)` **as 4096-wide vector values** (dtype
`float`, shape `(4096,)`, consumed by `RESHAPE(..., (1,1,4096))` then `ADD`), and there is
no codegen lowering for a value AFTER: rendering fails with
`float4096* alu0 = (data2_4096+data679_4096)` (pointer-typed ADD), and the natural load
spelling `RESHAPE(INDEX(buf, RANGE(4096)), (1,1,4096))` is rejected by UOp shape validation
(`bad reshape: () -> (1, 1, 4096)`). The value-AFTER read is tensor-level semantics that
only rangeify's buffer/load machinery can lower.

**Conclusion:** the fix belongs in the scheduler, not codegen: when the resadd fold admits
a precompile function output into a composite, the precompile FUNCTION body must be resolved
at rangeify time (so `handle_after`/INDEX lowering see the value AFTERs), or the emitter
must spell the reads with explicit flat loads. The S4 gate stays FAIL, the record stays
CLOSED, 0 credit. No tree change landed (the codegen pass was reverted; worktree clean).
