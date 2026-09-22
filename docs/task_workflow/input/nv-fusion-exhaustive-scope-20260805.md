# NV decode graph exposure / fusion / dataflow - exhaustive scope

Date: 2026-08-05
Status: **scope decided; CPU dev landed; all GPU arms parked**
Authority: `nv-decode-final-accounting-audit-20260805.md`,
`nv-decode-parity-p6-residual-priority-ledger-20260804.md`
Constraint in force: no GPU use. Every item below is CPU-only or gated behind
a CPU-verified gate that must pass before any token-wall arm.

## Question and answer

What exactly is the 662.128 us/token fusion/dataflow attribution inside the
support-work term, population by population, and what exact-output native A/B
would book each row? The composed authority is native `5.3242440 ms/token`
(187.82 tok/s) versus llama `4.0056768 ms/token` (249.65 tok/s). The audit
locates `+1108.082 us/token` of serialized native support work and splits it
into `662.128 us` fusion/dataflow-and-body attribution and `445.954 us`
llama-hidden overlap. **No row in that 662.128 us is bookable without an
exact-output native A/B per population.** This scope enumerates the
populations, the native node materialization for each, what llama fuses behind
MMQ/MMVQ, the exact A/B contract, the arithmetic, the gate, and what is
CPU-doable now. It also ships the CPU-only exhaustive population ledger and its
hermetic tests.

The family arithmetic closes exactly against the audit authority:

```text
norms 495.330 + flash 163.029 + residual/cast/contiguous 240.106
  + vocab/feedback 71.215 + rope/kv 17.965 - llama Q8 pack 325.517
  = 662.128 us fusion/dataflow-and-body attribution
hidden overlap 79.324 + 84.959 + 0.213 + 0 + 15.579 + 265.879
  = 445.954 us
662.128 + 445.954 = 1108.082 us support-work exposure
```

## Population census on the current closed-model DAG

All counts below come from `extra/llm_research/decode/nv_fusion_population_ledger.py`
run on the redirect-on authority capture `/tmp/nv_p4_redirect_on_dag_20260805.json`
(875 nodes, 4080 edges; the P4 wait-cost authority DAG). The 948-node semantic
partition capture `/tmp/nv_native_semantic_dag_v2_20260805.json` is quarantined
for queue-cut forecasting, but its exact role census (from
`nv-decode-nonquant-role-partition-20260805.json`) is the role authority and
matches this ledger node-for-node where the graphs share programs.

| population | nodes | sum us | mean us | max us | exact | heuristic | anchor-child epilogues | us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| quant_core | 217 | 3948.128 | 18.194 | 322.976 | 217 | 0 | 0 | 0 |
| norms | 362 | 650.752 | 1.798 | 3.840 | 362 | 0 | 215 | 306.368 |
| residual/cast/contiguous | 145 | 174.912 | 1.206 | 3.296 | 108 | 37 | 145 | 174.912 |
| flash | 72 | 306.016 | 4.250 | 6.592 | 72 | 0 | 0 | 0 |
| rope/kv | 72 | 116.128 | 1.613 | 2.848 | 72 | 0 | 54 | 89.792 |
| vocab/feedback | 7 | 64.320 | 9.189 | 39.552 | 7 | 0 | 1 | 1.888 |
| **total** | **875** | **5260.256** | | | **838** | **37** | | |

Status `PASS`: 0 unclassified nodes. The 37 heuristic nodes are the
redirect-on-only `E_32_32_4` program identities (36 `81c96a8e` ffn-residual
adds, 1 `86a23e1a` block-output contiguous), flagged `exact=False` and counted
in residual/cast/contiguous; the 948 capture's equivalent role is
`fab82d40` (ffn_residual_add + block_output_contiguous), so the population
attribution is stable across constructions. Anchor-child means a support node
with an incoming edge from a quant or flash node; the capture graph carries
planner alias edges, so this is a candidate census, not an exact dataflow
claim. The per-node ledger is machine-readable in the tool output.

## Per-population detail

Notation: `A/B` means a same-session native-NV paired bracket, median over
interleaved `A/B/A` repetitions, with the treatment arm changing only the named
population's node census. Correctness contract (all populations): full-logit
fp32 SHA-256 over 32 decoded rows identical, the 32-token stream identical,
per-row `argmax` equal to the sampled token. Acceptance: `median(B)-median(A)`
with paired dispersion, promotion gate `+50 us`, changed-node census recorded,
and the population's gate passed first. A family row books once; rows are never
stacked (audit rule 4 and ledger stop conditions).

### norms (+574.654 family delta; 495.330 us attribution; +19.27 tok/s ceiling)

(a) **Native materialization** (redirect-on authority): 362 nodes, 650.752 us
raw. 145 reductions (`r_16_256` x73, `r_2_8_4_4_16` x36, `r_8_16_8` x36) and
217 epilogues (`E_32_32_4_f14a5cc0` x72, `E_32_32_4_c6fef356` final x1,
`E_4_2_8_16_4`/`E_2_8_16_4_4` q-norm x72, `E_2_8_16_4`/`E_8_2_16_4` k-norm
x72). Scaled partition total 673.211 us; raw/calibrated ratio ~1.016. 215
ordinary epilogues are anchor children (306.368 us).

(b) **Llama**: `rms_norm_f32` stays a separate kernel (145 calls = 1 initial +
36 x q/k/ffn/next). Llama fuses no norm into an MMQ body; it exposes 98.557 us
and hides 79.324 us behind MMQ via driver co-scheduling. The family delta is
exposure, not llama-side fusion.

(c) **A/B design**: arm A = closed model graph; arm B = ffn/next norm
*epilogue* absorbed as an ordinary in-core epilogue of the consuming quant
kernel (ffn epilogue into `w1w3fused`, next epilogue into the following
pipeline). Norm reduce nodes stay (see gate). Correctness contract as above.

(d) **Arithmetic**: ceiling 495.330 us/token; at the composed baseline
`1000/(5.3242440-0.495330) = 207.09 tok/s` (+19.27). Only the measured delta
books.

(e) **Gate**: `nv_boundary_free_ordinary_uop_gate.py` must pass (ordinary UOps
in-core, no custom boundary, no `CONTIGUOUS`, no lazy-view materialization).
Current verdict `CONSTRUCTION_GAP`: the reduce+epilogue pair has no generic
cross-thread reduction-to-output scheduler primitive, so the 145 reduce nodes
are not absorbable and bound the recoverable ceiling to the epilogue half
(217 nodes).

(f) **CPU-doable now**: the gate is CPU-only and already run
(`CONSTRUCTION_GAP` baseline recorded). A candidate in-core norm-epilogue
construction can be codegen-checked CPU-only (lowering as ordinary UOps, no
`CUSTOM`/`CONTIGUOUS`, exact-output compare against the reference pair on a CPU
d512 row).

### residual/cast/contiguous (+240.319 family delta; 240.106 us attribution; +8.87 tok/s ceiling)

(a) **Native materialization**: 145 nodes, 174.912 us raw (v2 capture: 216
nodes, 244.608 us; scaled partition 240.762 us). Roles: attention_cast 36
(`E_32_32_4_0a5eb0ac`), attention_residual_add/ffn_down_cast 36
(`E_32_32_4_02a9738c`), ffn_activation_cast 36 (`E_128_32_3`), ffn_residual_add
36 (`E_32_32_4_81c96a8e`), block_output_contiguous 1 (`E_32_32_4_86a23e1a`).
All 145 are ordinary elementwise epilogues and all 145 are anchor children
(174.912 us) - this is the only fully boundary-free-eligible population.
The redirect-on construction already materializes 71 fewer `E_32_32_4` than
the 948 capture (35 block-output contiguous + 36 of the shared
residual/down-cast identity).

(b) **Llama**: fuses the residual add into 35 of 36 O projections (final O
nonfused; P6-C record); activations stay fp16 end-to-end so no fp32-to-fp16
cast nodes exist; no materialized block-output contiguous. Llama exposes only
0.443 us and hides 0.213 us of elementwise work - the fusion boundary absorbs
this family almost entirely.

(c) **A/B design**: arm A = closed graph; arm B = one ordinary-UOp in-core
projection epilogue absorbing one role (e.g., attention_residual_add into the
attn-o quant epilogue, ffn_down_cast into the ffn-down epilogue). Changed-node
census confined to residual/cast/contiguous; correctness contract as above.
This is the audit's single admissible construction.

(d) **Arithmetic**: ceiling 240.106 us/token -> `196.69 tok/s` (+8.87).

(e) **Gate**: `nv_boundary_free_ordinary_uop_gate.py` passes as the acceptance
predicate. Every custom-boundary predecessor is closed and earns zero credit:
attention-O +69 us NO-GO, llama-O +21.2 us NO-GO, FFN-down epilogue neutral
(+0.348 us), RMSNorm semantic wrapper +60.802 us / 110 lazy-view kernels
NO-GO (`nv-projection-epilogue-qualification` and
`native-epilogue-causal-ledger` records).

(f) **CPU-doable now**: codegen-check the in-core epilogue CPU-only (compile
the quant core + fused epilogue as ordinary UOps, assert no
`CUSTOM`/`CONTIGUOUS`/adapter kernels, exact-output vs the reference two-node
composition on CPU).

### flash (+247.989 family delta; 163.029 us attribution; +5.93 tok/s ceiling)

(a) **Native materialization**: 72 custom kernels, 306.016 us raw:
`flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128` x36 (score) +
`flash_fused_gmax_combine_32_128` x36 (combine). Scaled partition 305.581 us
(score 204.697, combine 100.884). Flash nodes are anchors; 0 anchor children.

(b) **Llama**: score+combine kernels are co-scheduled with MMQ, not fused into
it (hidden 84.959 us, exposed 57.592 us). Llama's raw flash class interval
union is 363.716 us vs native's calibrated 305.581 us: overlap/exposure is
sufficient to explain the ownership gap, and flash-body parity remains unproven
(audit finding 3).

(c) **A/B design**: the missing flash-body-parity arm. Arm A = closed graph;
arm B = a tinygrad-owned flash score/combine body rewrite, flash-node census
unchanged (72 in, 72 out), correctness contract as above. The
fp32-to-fp16 output boundary change belongs to residual/cast/contiguous's A/B,
not this one, to keep census disjointness.

(d) **Arithmetic**: ceiling 163.029 us/token -> `193.75 tok/s` (+5.93).

(e) **Gate**: flash is a custom-kernel family; the boundary-free rule governs
epilogue absorption, not the flash body. Gate = exact-output contract +
same-session paired bracket + flash-confined census. Any epilogue change that
touches flash output must additionally pass boundary-free (owned by the
residual population).

(f) **CPU-doable now**: the native flash construction can be codegen-checked
CPU-only (lowering + exact-output vs the eager fp32 reference attention on a
single d512 row).

### vocab/feedback (+71.215 family delta; 71.215 us attribution; +2.55 tok/s ceiling)

(a) **Native materialization**: 7 nodes, 64.320 us raw (v2: 9 nodes, 73.632 us;
scaled 72.474 us): token_feedback `E_c9699af0` x1 (+ `E_2_c8a3207c` in the 948
capture), vocab_sampler `E_16_4_2_8_16_2_4_4` x1, `E_1187_32_4` x1,
`r_32_32_4_32_4` x1, suffix reduces `r_32_4_1187` (39.552 us, the largest
non-quant node) / `r_128_16_8_1187` / `r_16_8`. The vocab-head core
(`q6k_gen_coop_151936_4096_inkernel`, 322.976 us) is in quant_core.

(b) **Llama**: `get_rows` sampler exposed 1.259 us; no token-feedback node
(feedback stays host-side).

(c) **A/B design**: the predispatch oracle seam
(`nv_predispatch_full_logits_qualification.py`) already owns this family's
causal signal (-69.166 us combined, unbooked pending full-logit oracle). The
sampler-tail fusion arm is separate; packed greedy argmax is closed NO-GO
(71.874 -> 142.647 us, P2 record) and must not be reopened without a different
mechanism.

(d) **Arithmetic**: ceiling 71.215 us/token -> `190.37 tok/s` (+2.55); the
booked amount is the predispatch A/B delta, not the attribution.

(e) **Gate**: full-logit equality is the admission rule for the predispatch
A/B; the boundary-free rule does not apply (sampler nodes are reductions).

(f) **CPU-doable now**: the full-logit qualification tooling and this ledger.

### rope/kv (+33.543 family delta; 17.965 us attribution; +0.64 tok/s ceiling)

(a) **Native materialization**: 72 nodes, 116.128 us raw (scaled 116.79 us):
rope_q `E_16_32_4_2` x36, kv_store_k_rope_cast `E_8_8_16_2` x18,
kv_store_k_rope_cast_with_q6_partial_reduce `r_8_8_16_2_4` x18. 54 ordinary
epilogues are anchor children (89.792 us) on the alias-edge census.

(b) **Llama**: K RoPE is fused into the KV-cache store path by construction
(`build_qkv`/`build_attn` expand Q, V, K so K can fuse rope into the cache
write; `nv-decode-llama-kv-role-mapping-20260805.json`, audit finding 22).
Exposed 83.247 us (rope 42.138 + kv_set_rows 41.108), hidden 15.579 us.

(c) **A/B design**: the KV store chain is already one fused store kernel/layer
and its construction A/B was wall-neutral (`kernel_delta 0`,
`native-epilogue-causal-ledger`); do not reopen. The open arm is rope_q
absorption into the flash score input path, rope/kv census confined.

(d) **Arithmetic**: ceiling 17.965 us/token -> `188.46 tok/s` (+0.64).

(e) **Gate**: exact-output contract + census confinement; the KV store chain
stays closed (wall-neutral).

(f) **CPU-doable now**: nothing new; the KV role mapping is CPU-derived.

### llama Q8 pack (-59.639 family delta; native recoverable 0)

(a) **Native materialization**: 0 nodes. Native activations are already fp16.

(b) **Llama**: `quantize_q8_1` runs 217 times, one directly before every MMQ
(adjacency validated in `llama_tinygrad_role_manifest.py`); exposed 59.639 us,
hidden 265.879 us. The -325.517 us attribution row offsets llama's hidden
overlap; it is a llama cost, not a native target.

(c) **A/B design**: n/a on native. Native equivalents are the quant-substrate
gates: generic Q8/int8 DP4A Gate-1 FAIL +1.172/+1.352 us (P6-A), Q4/Q6 rows
P6-B/C/D.

(d) **Arithmetic**: 0 us recoverable; tok/s unchanged.

(e) **Gate**: `nv_ffn_q8_cooperative_microgate.py`,
`nv_llama_q6_included_microgate.py`.

(f) **CPU-doable now**: those gates are CPU-only.

## Additional populations from the semantic partition

The 948-capture role partition names finer disjoint roles inside the families
(all node counts verified against the ledger on both captures):

| family | roles (counts) |
| --- | --- |
| norms | initial_rmsnorm (2), q_norm (108), k_norm (108), ffn_rmsnorm (72), next_or_final_rmsnorm (72) |
| flash | flash_score (36), flash_combine (36) |
| residual/cast/contiguous | attention_cast (36), attention_residual_add / ffn_down_cast (36), ffn_activation_cast (36), ffn_residual_add (36), block_output_contiguous (36 in 948; 1 in 875) |
| vocab/feedback | token_feedback (2 in 948, 1 in 875), vocab_sampler (7 in 948, 6 in 875) |
| rope/kv | rope_q (36), kv_store_k_rope_cast (18), kv_store_k_rope_cast_with_q6_partial_reduce (18) |
| quant_core | attn_q (36 q4k 4096x4096), attn_k (36 q4k 1024x4096), attn_v (18 q4k + 18 q6k partial 1024x4096), attn_o (36 q4k 4096x4096), ffn_gate_up (36 q4k w1w3fused 12288x4096), ffn_down (18 q4k + 18 q6k coop 4096x12288), vocab head (1 q6k coop 151936x4096) |

Quant core aggregate on the redirect-on authority: 217 nodes / 3948.128 us;
the +302.788 us quant-core delta is a separate P6 ledger row (P6-A/B/C/D), not
fusion attribution, and is not bookable here.

## Recovery ordering

1. residual/cast/contiguous - the only fully boundary-free-eligible population;
   the ordinary-UOp in-core construction is the audit's one admissible route.
2. norms - epilogue half only; gated on a boundary-free construction passing
   the CPU gate (reduce half is CONSTRUCTION_GAP).
3. flash - flash-body parity arm settles the audit's unproven-body question;
   custom family, exact-output contract.
4. vocab/feedback - predispatch oracle A/B, full-logit admission rule.
5. rope/kv - rope_q absorption; KV store chain stays closed.
6. llama Q8 pack - no native arm; feeds quant-substrate P6 gates only.

Within each row, the CPU gate runs first, then the exact-output census arm,
then the paired wall bracket; a row is booked once with `delta_us`,
dispersion, token identity, and changed-node census.

## HARD STOP

No GPU arm for any fusion/dataflow population until all of:

1. The population's CPU gate passes: `nv_boundary_free_ordinary_uop_gate.py`
   for epilogue populations (norms epilogue, residual/cast/contiguous), the
   full-logit oracle rule for vocab/feedback, exact-output census confinement
   for flash/rope.
2. The arm has an exact-output contract: full-logit fp32 SHA-256 over 32 rows,
   identical token stream, per-row argmax equal to the sampled token.
3. The arm changes only the named population's node census (no added custom
   boundary, adapter, or copy nodes; no quant node count change).
4. The paired same-session bracket clears the +50 us promotion gate with
   reported dispersion; rows book once and are never stacked.

Closed constructions are not reopenable by relabeling: attention-O +69 us,
llama-O +21.2 us, FFN-down epilogue +0.348 us neutral, RMSNorm semantic wrapper
+60.802 us / 110 lazy-view kernels, KV store chain wall-neutral, packed greedy
argmax NO-GO. A new construction must be a different mechanism.

## Dev this turn (CPU-only)

- `extra/llm_research/decode/nv_fusion_population_ledger.py`: exhaustive
  population ledger (schema `tinygrad.nv_fusion_population_ledger.v1`), CLI
  `--dag --out`; classification, per-population stats, anchor-child fusion
  candidates, boundary-free eligibility, fail-closed validation.
- `test/unit/test_nv_fusion_population_ledger.py`: hermetic synthetic-DAG
  tests; 8 passed.
- This scope document.

No production default changes, no GPU work, no behavior changes while the GPU
ban is in force.

## References

- `nv-decode-final-accounting-audit-20260805.md` (location PASS / recovery FAIL)
- `nv-decode-parity-p6-residual-priority-ledger-20260804.md` (family rows, P6 stop conditions)
- `nv-decode-nonquant-role-partition-20260805.json` (exact 731-node role authority)
- `native_epilogue_causal_ledger.py` / its output (closed epilogue constructions)
- `nv_boundary_free_ordinary_uop_gate.py` (CONSTRUCTION_GAP gate baseline)
- `nv_projection_epilogue_qualification.py` (attention-O / llama-O NO-GO arms)
- `nv_predispatch_full_logits_qualification.py` (vocab/feedback oracle seam)
- `llama_tinygrad_role_manifest.py` (Q8/MMQ adjacency, role mapping)
- `decode-norm-fusion-paths-forward-20260802.md` (M3 norm record, copy tax)
- `decode-kv-store-chain-fusion-scope-20260803.md` (KV chain fused)
