# Selected generated prefill lifecycle audit

Scope: Qwen3-8B Q4_K_M, RTX 5090, pp512; current252 explicit compiler composition.
This audit distinguishes generated projection coverage, the complete selected
token path, and kernel-active performance. They are different claims.

## Reconciliation

| Component | Selected current252 path | Existing evidence / decision |
|---|---|---|
| Gate/up | 72 U8 Stream-K mains, producers and active fixups | Installed; shared-load and register experiments do not establish a major gain. Historical same-runtime 72-role lifecycle was 19.348799 ms generated versus 14.154342 ms llama. This includes producers/fixups and is not pure body timing. |
| Q/O | 72 plain generated projections | Installed. The earlier duplicate-O fix belongs to the native llama binding; current census has 72 Q/O mains, not an extra 36 O calls. Complete Q/O Stream-K replacement was slower. |
| K and mixed V | 36 K and 36 V generated projections | Installed, including the final 18 Q6 V roles. |
| Mixed down | 18 generated Q4 and 18 generated Q6 | Installed. Q6 main+fixup already qualified against live llama; that result does not prove every projection family is faster. |
| Vocabulary | Generic `r_1187_32_4_16_2_2_2_4_8_*` plus two elementwise calls before this fix | **Qualified omission repaired:** current252 off/on/off is 53.313233 / 50.746286 / 53.183219 ms. Existing four-warp vocabulary now follows the compiler stack; rollback is `NV_COMPILER_Q6_VOCAB_PP512=0`. |
| Flash | 36 `nv_sm120_q16_grid_hd128_loop_attention` calls | The faster previously promoted MMA Flash is a llama cubin; adding it would change the generated-only objective. The clean-room replacement was rejected at 1.337 ms/layer. |
| Terminal FFN pruning | Full-batch final FFN | Existing hook was tested and regressed on the earlier composition. It is not a qualified omitted win. |
| Q8 pair reuse / epilogues | Existing producer and support topology | Reuse was below its investment threshold; fused epilogue and other tested substitutions regressed. No unbooked standalone promotion is established. |

## Timing-boundary correction

The August 27 common-protocol **decode** audit established selected active
bodies at 3494.138 us versus llama 3656.606 us, while the endpoint was slower.
Its approximately 215 us boundary residual is diagnostic, not a removable
budget. These GEMV results must not be transferred to pp512 GEMMs.

Current252 `accounting.json` labels HCQ command intervals `active_us`. Those
are not CUDA/CUPTI kernel-active intervals. Its selected invocation 8 records
56.893216 ms device span, 56.877248 ms interval union, and 15.968 us exposed
idle. The README's 13.728 us idle does not match this selected invocation.
Neither small exposed idle nor a large charged role proves that execution
boundaries within command intervals are free. The observer-bearing span must
not be subtracted from the unprofiled 53.302259 ms median.

The historical classifier's final branch assigns every other name to support.
Its zero-unknown count is therefore not a semantic proof. The adapter now
reports unresolved support launches/time and `semantic_closure` separately.

The next body-versus-boundary attribution must use exact current binaries and
launch/argument contracts under a common protocol, preserving producer and
fixup scope. The old three-buffer wide NCU bridge cannot launch the current
five-buffer Stream-K binary unchanged.

## Qualification scope correction

The current composed harness's optional `--deep-replay` fails at
`_graph_stage_buffers`: it assumes `outs=(0,), ins=(1,2)` for the Stream-K
main. The audit preserved the failure before timing. Existing 20-cycle
claims mean alternating-request output/logit replay, not a passing current
internal-buffer replay audit. The vocabulary bracket uses that established
output protocol plus exact selected PROGRAM counts and full-logit comparison.

## Evidence

- `nv-active-body-ledger-phase2-result.md`: common-protocol decode correction.
- `nv-llama-gap-closure-ledger-20260830.md`: native O ownership recovery.
- `evidence/nv-prefill-current252-hcq-20260906/accounting.json`: selected physical ledger.
- `evidence/nv-p5-gateup-r31-20260905/stream31.json`: same-runtime projection lifecycle.
- `evidence/nv-vocab-logits-r9-20260905/`: older generated vocabulary candidate.
- `evidence/nv-llama-fattn-mma-pp512-model-20260830/README.md`: native Flash qualification.
- `evidence/nv-cleanroom-flash-model-20260830/README.md`: generated Flash rejection.
- `nv-prefill-ranked-campaign-closure-20260829.md`: terminal pruning rejection.
- `evidence/nv-current252-vocab-lifecycle-audit-20260906/`: new off/on/off bracket.

## Qualified vocabulary integration

The matched bracket recovers 2.501940 ms (4.6986%) against mean controls;
controls differ by 0.130014 ms. Every arm passes 20 alternating output replay
cycles and retains 252 projection mains/producers/canonical weights with zero
V/down overlays. All 151,936 logits are finite, token remains 198, and maximum
candidate/control difference is 0.00793672 (mean 0.000704779), within the
established rtol 0.02 / atol 0.5 contract. The emitted symbol is
`q6k_v_four_warp_fp16_direct_151936_4096`, exactly once in the candidate and
absent from both controls. The generic vocabulary main disappears.

A separate automatic-selection run, with the vocabulary override absent,
passes at 50.428258 ms and selects the same one vocabulary main. This is a
confirmation, not an additional independently booked speedup. Selector tests
cover compiler mode, explicit rollback, explicit llama preference, conflicting
explicit leases, target architecture, shape, and vocabulary size.

Fresh llama pp512 R9 at build ac4cddeb0 measures 38.8192285 ms after excluding
sample zero under the retained first-use convention. Against the bracket's
50.746286 ms generated candidate this is an 11.9270575 ms difference. This is
a fresh sequential cross-harness reference, not a same-prompt or locked-clock
cross-runtime bracket: llama uses its own prompt and does not perform the same
token/logit inspection. It replaces historical 35--38 ms references for this
session, but does not allocate the remaining gap to kernel bodies.

The repaired omission proves lifecycle composition can recover material time
without changing projection kernels. Further body optimization remains gated
on common-boundary attribution; the audit does not claim the entire residual
is either kernel execution or graph overhead.

## Refreshed selected-path ledger

The post-fix profile closes 1,630 physical command intervals. Its 52.970624 ms
device span is observer-bearing mapping evidence; endpoint authority remains
the unprofiled bracket above.

| Component | Launches | HCQ command interval ms |
|---|---:|---:|
| Projection mains | 252 | 41.665376 |
| Projection producers | 252 | 0.929600 |
| Projection fixups | 108 | 1.544096 |
| Flash | 36 | 3.322176 |
| Vocabulary including classified epilogues | 4 | 0.334688 |
| Classified norm/conversion | 73 | 0.924992 |
| Activation/multiply | 72 | 0.974272 |
| Unresolved support | 829 | 3.242016 |
| Input and token transfer | 4 | 0.018240 |

Exposed interval gaps total 0.015168 ms; no overlap is observed. Vocabulary
fell from 3.208096 ms in the previous trace to 0.334688 ms. Other components
also drift between the two profile sessions; only the unprofiled A/B/C books
the 2.501940 ms recovery. The original 883 support launches included 54 known
producers: the adapter now separates them, leaving 829 unresolved. This is
honest partial semantic attribution, not zero-unknown closure.

Next attribution target: exact current projection main active durations versus
their enclosing command intervals. The 41.665376 ms main interval charge
must not be called pure arithmetic time. Separately resolve the 3.242016 ms
support bucket by buffer roles and generated source before calling it copies,
idle, or removable overhead. No new kernel design is justified by the naming
of either bucket alone.
