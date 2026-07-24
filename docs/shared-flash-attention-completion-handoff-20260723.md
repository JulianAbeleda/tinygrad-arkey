# Shared FlashAttention Completion Handoff (2026-07-23)

## Big picture (one line)
We now have a mostly complete compiler-side proof of score-resident fused attention (online softmax + PV + QK), but one backend ordering boundary still blocks the final roofline target for 8B/14B prefill: preserving the C-window lifetime/ordering so no extra score/probability traffic or register pressure is introduced.

## What this work targeted
The aim was to land one fused flash-attention route that covers:
- both fp16 and non-fp16 paths,
- both 8B and 14B prefill/throughput work,
- decode and prefill paths,
- without hand-written special-case kernels,
- without materializing full score/probability buffers,
- and with measurable AMD/ROCm roofline-level resource utilization.

## State at a high level
- Composite reduction handling is no longer tied to one hardcoded path:
  - combine logic generalized into registry-driven combine functions,
  - composite route now supports multi-input/stateful reduction behavior,
  - optimizer no longer has attention-specific carve-outs for this pathway.
- REDUCE-slot plumbing exists and is tested:
  - `REDUCE_SLOT` exists in enum/spec/dispatch,
  - multi-output reduction reduction-path semantics were formalized with structural checks.
- No-range composite path matured significantly:
  - stack/early forms can be handled with generic combine-chain execution.
- Attention pattern handling evolved toward auto rewrite:
  - `CompositeReduce` can carry V input via `v_uop`,
  - online softmax PV path has been represented as 3-slot reduction state,
  - pre-existing explicit flash route tests and partial auto-rewrite coverage exist.
- Shared state / rotating-PV architecture is substantially implemented:
  - phase ABI probe and capture added,
  - rotating PV markers, publications, drains, and typed slots added,
  - lane-private softmax bridge contract defined and then corrected,
  - pointer-to-PV/sequence flow and bridge correctness established.

## What is currently working
- End-to-end fused attention route no longer fails on core algorithmic correctness in the established paths.
- Decode routing and partial prefill shapes compile and run with score-resident behavior.
- AMD route now consistently reaches 0 spills and avoids separate score/probability materialization.
- Current best observed kernel profile for the target composite route:
  - `201 VGPR`, `11776 LDS`, `0 spills`, `8 QK + 8 PV` windows,
  - full 8QK/8PV path remains functionally present.
- The main gate that remains is not a math issue; it is an emitted-order issue.

## Current blocker in one sentence
`CStyleLanguage` (generic statement emission) does not preserve the strict rotating PV sequence order required for this route, so C windows are still loaded too early and compete with softmax state in VGPR, preventing the final <=192 VGPR target and forcing extra live pressure.

## Why this is still 22% away from completion
### Evidence-based blocker profile
1. Existing compiler scheduling markers (`AFTER`, `GROUP`, rotate sequence metadata) are not sufficient to force backend statement order for this case.
2. Render traversal naturally emits WMMA and C sources in a non-blocking order, which causes:
   - `wait`/publish barrier not reliably anchoring each PV block,
   - all C loads hoisted before PV WMMA windows,
   - register peak shifted onto the QK+PV overlap window.
3. Disassembly and register profiles confirm the peak is now caused by C window overlap, not bridge/softmax internal logic.
4. Attempts that inject ad-hoc serial semantics in UOp graph were either too invasive or did not survive to final renderer output.

## Correct path forward (what a handoff engineer should do)
1. Add a native rotating-PV sequence op (typed UOp/OpSpec path, tiny scope, no generic scheduler rewrite).
2. Keep generic `AFTER` semantics unchanged; do not mutate global rendering order globally.
3. Lower that native op in the HIP renderer as a strict sequence:
   - publication sync/boundary,
   - C window LDS load,
   - PV WMMA,
   - store of computed C window back to LDS.
4. Tie this op to AMD backend emission with explicit scheduling contract so ordering is preserved per block.
5. Replace current rotating-PV synthetic sequencing with this native op and preserve verifier invariants.
6. Re-run minimal focused pass + disasm checks to verify:
   - C loads no longer prehoisted,
   - sequence strictly block-serialized as intended,
   - VGPR <=192 is crossed on the target route.

## Why this is the right abstraction
This is a compiler design boundary, not a correctness math rewrite:
- We already proved the high-level fused representation is valid,
- we already removed algorithm-specific carve-outs,
- we already validated lane-private bridge correctness,
- the missing contract is strict emission behavior.

A generic IR-level solution would risk side-effects outside attention and could disturb existing scheduling assumptions. A local native op preserves blast radius.

## Immediate acceptance checklist (completion criteria)
- `<=192 VGPR` target reached for the main prefill composite route on the same geometry where 201 VGPR is currently observed.
- `0 spills/scratch` preserved.
- Score/probability tensors remain non-materialized end-to-end.
- Decoder and prefill 8B/14B paths use same core composite mechanism.
- Non-fp16 path preserves parity and numeric tolerances at the same acceptance gates already used.

## Handoff instructions for continuation
- This is the final scheduler/compiler gate before full architecture completion.
- Focus implementation order around backend contract first, then route-level integration:
  1) define op,
  2) verifier/type checks,
  3) HIP/Amd renderer sequence,
  4) replace marker-based sequence,
  5) re-run minimal evidence gate.
- Do not spend cycles on adding additional benchmark scaffolding until this ordering op lands.
- Keep changes local and narrow so regression surface stays low.

## Notes for reviewers
- Do not confuse this with benchmark-methodology cleanup: the compiler route is not yet complete, so benchmark noise here mostly reflects ordering not measurement bias.
- The historical high gains (prototype prefill/ decode uplift) remain useful as directional evidence, but final 8B/14B parity depends on this backend blocker closure.

