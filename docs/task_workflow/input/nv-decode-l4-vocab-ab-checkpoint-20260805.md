# NV decode L4 vocab fusion A/B - checkpoint (invalid run, anomaly under static investigation)

Date: 2026-08-05
Status: session checkpoint, HALT. User requested the state be saved in a doc and work
stopped; no GPU runs and no push until unpaused. Branch boundary: tinygrad
`nvidia-bringup-20260731` at `e3e96cb73` (tracked tree clean except the new harness in
section 8).

## 1. Where this sits in the campaign

- L4 vocab substrate fusion is the top remaining ranked item: the forward scope
  (`nv-parity-and-beyond-forward-scope-20260803.md`) states L4 as "SCOPED; sole GO as a
  landing warrant, NOT a wall ranking" with the isolated same-session d512 wall
  measurement PENDING. This session ran exactly that wall measurement.
- On record before this session: the fused shape is bit-identical and faster in
  isolation (315.9 us vs llama 303.75 us, `l4-vocab-substrate-fusion-measurement-record-20260803.md`);
  the fused kernel was live in census at `499bf4f5f` (1021 -> 1020 kernels/token,
  `q6k_gen_coop_151936_4096_inkernel` 325.09 us, scalar reduce gone,
  `like-for-like-cap-settling-record-20260803.md`); fixed-depth wall at that boundary was
  BELOW PARITY (d512 174.724 vs 248.111 +/- 7.75 tok/s, ratio 0.704,
  `l4-vocab-substrate-fusion-implementation-record-20260803.md`).
- Ledger context (`nv-decode-parity-campaign-reconciled-ledger-20260805.md`): five booked
  recoveries total 270.467859625 us/token; fixed authority 5.612310 ms/token vs llama
  3.966140 ms/token; strict accounted remainder 1375.702140375 us/token.

## 2. Protocol

- Harness: `scratchpad/nv_decode_l4_vocab_fusion_ab.py` (new, untracked), modeled on
  `scratchpad/nv_decode_predispatch_ab.py`: same-session reverse bracket
  (control_a0 / candidate_a / control_a1), 3 settling + 30 steady tokens, `DEV=NV`,
  flock `/tmp/gpu-bench.lock`, `tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()`,
  `Transformer.from_gguf("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf", 4608)`,
  `model.generate([1]*512, chunk_size=32, temperature=0.0)`.
- Toggle: control arms set `dataclasses.replace(model.output.route_admission,
  epilogue_fusion_promoted=False)`; the candidate keeps the loaded (fusion-admitted)
  admission. Load-time asserts passed: `q6k_storage` present, `out_features == 151936`,
  `route_admission.fusion_admitted` True.
- Run output: `/tmp/nv_decode_l4_vocab_fusion_ab_20260805.json`.

## 3. Results - INVALID as a wall comparison

| arm | wall us/token median | vocab programs captured |
| --- | ---: | --- |
| control_a0 | 5597.9955 | `q6k_gen_coop_151936_4096` + `q6k_vocab_scalar_reduce_151936_4096` + scatter (`E_1187_32_4`, `r_32_4_1187`, `r_128_16_8_1187`) |
| candidate_a | 5603.8215 | same external_sum chain, NO `_inkernel` kernel |
| control_a1 | 5611.6415 | same external_sum chain |

Token SHA is identical across all arms (`f8f4d055...`), which is expected because the
fused output is bit-identical, so the correctness gate passed trivially. But the
candidate arm never captured `q6k_gen_coop_151936_4096_inkernel`: the toggle never
produced the fused graph, so **no wall delta conclusion is valid**. Do not report a
fused-vs-external wall result from this run.

## 4. The anomaly

The selection rule at HEAD (`tinygrad/llm/decode_routes.py:273-279`) is
`"in_kernel" if fusion_admitted and binding.use_coop and binding.row_tile * Q6K_POS_EXTENT <= 32
else "external_sum"`. With `Q6K_POS_EXTENT = 16` and the NV sm_120 coop row_tile of 2
(`decode_kernels.py:397-402`, 2*16 = 32 legal), an admitted vocab head should have
rendered the fused kernel. The candidate arm instead rendered the identical
external_sum chain as both controls, while the same box at `499bf4f5f` demonstrably ran
the fused kernel in census. The loaded admission is asserted True at harness start, so
the divergence is between load-time admission facts and trace-time selection.

## 5. Static findings so far (no GPU)

1. `git diff 499bf4f5f..HEAD` for `decode_routes.py` shows the Q6K execute section
   unchanged; the diff adds only Q4K / w1w3 / kv-store routes (47 commits in the range).
   `decode_kernels.py` Q6K section is likewise unchanged.
2. The closed-default promotion record
   `tinygrad/llm/generated/decode-epilogue-fusion-route-policy.json` still promotes
   `("NV", "sm_120")`.
3. Contradiction with the on-record live census at `499bf4f5f` (fused kernel present,
   scalar reduce gone) means either the trace-time facts differ from the load-time facts
   on this branch state, or the harness never observed a fresh trace (below).

## 6. Open hypotheses

1. Capture reuse: `captured_program_names()` reads `model.rollout_greedy_jit*.captured`
   after each arm, and all three arms report identical hash-suffixed scatter names, which
   smells like every arm reused ONE capture (the first arm's). If so, candidate_a never
   re-traced and the toggle was never observed - a harness design flaw, not a code
   regression.
2. The vocab head linear consulted at trace time is not the object that was toggled.
3. `binding.row_tile` resolved to 4 (capability architecture not `sm_120` at trace time),
   making `in_kernel` illegal regardless of fusion admission.
4. Something in the 47 commits between `499bf4f5f` and `e3e96cb73` changed
   install/selection (the `qk_primitives.py` diff removes `QKPrimitiveEligibility` etc.);
   check `QKPrimitiveCapability.satisfied` and the route-admission construction paths.

## 7. Resolution path (after user un-pauses)

- Add debug prints to the harness: capability facts, row_tile, and admission per arm;
  inspect `jit.captured` before/after each arm; force a fresh jit capture between arms.
- Or rerun the like-for-like `DEBUG=2` prime-trace census harness at HEAD to see which
  vocab kernel actually executes.
- No GPU work until the user unpauses; do not push (GitHub secret scan previously
  blocked on a flagged API-key string in `docs/task_workflow/output/`).

## 8. Files and repo state

- Harness (untracked): `scratchpad/nv_decode_l4_vocab_fusion_ab.py`
- Run output: `/tmp/nv_decode_l4_vocab_fusion_ab_20260805.json`
- HEAD: `e3e96cb73`, branch `nvidia-bringup-20260731`, clean tracked tree except the
  harness above.
- References: `nv-parity-and-beyond-forward-scope-20260803.md`,
  `l4-vocab-substrate-fusion-implementation-scope-20260803.md`,
  `l4-vocab-substrate-fusion-measurement-record-20260803.md`,
  `l4-vocab-substrate-fusion-implementation-record-20260803.md`,
  `like-for-like-cap-settling-record-20260803.md`,
  `nv-decode-parity-campaign-reconciled-ledger-20260805.md`.
