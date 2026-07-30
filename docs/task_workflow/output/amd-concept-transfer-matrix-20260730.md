# AMD optimization-concept transfer matrix for Metal

Date: 2026-07-30
Packet: MR6 from `metal-replay-generated-route-parity-scope-20260729.md`
Audit revision: `007e6b743fbafca62142a3262a0e56838d017dcd`
Machine authority: `amd-concept-transfer-matrix-20260730.json`

## Verdict

The AMD campaign contains portable **questions and candidate axes**, not a Metal recipe. Five rows are relevant to
the first Qwen3-8B inventory but remain unranked until MR5/MR7 supplies stable Metal role identity and measured role
cost. Flash decode is deferred to the required depth-512 evaluation. The large-shape, Q6_K long-K coverage, and G=5
attention wins do not match the first 8B workload and must not enter its initial candidate population.

This packet authorizes **zero** Metal candidates and changes no runtime or policy. Every exact Metal mechanism remains
`unknown`; ordinary generated Metal remains the control and fail-closed fallback.

## Reading the evidence

Historical artifacts later deleted from the active tree remain addressable by commit and Git blob. Reproduce any
citation with:

```sh
git show <full-commit>:<artifact-path>
git rev-parse <full-commit>:<artifact-path>
```

The blob IDs in the JSON are content identities, not performance identities. AMD numbers prove only the named AMD
workload at the cited revision. They do not establish a Metal primitive, threshold, bandwidth ceiling, or speedup.

Classification vocabulary:

- `portable_concept`: target-neutral mechanism or candidate axis.
- `compiler_work`: a target compiler primitive must first be demonstrated.
- `backend_specific_implementation`: AMD implementation details that must not transfer.
- `irrelevant_first_metal_workload`: retain only as a later reopen concept.

Metal status vocabulary:

- `observed`: measured on the exact Metal workload.
- `supported`: backend/compiler surface exists, but the exact route is unmeasured.
- `unknown`: exact support or competitive behavior is not established.

## Transfer summary

| Id | Historical win/fix | Primary class | First Metal workload | Metal status | MR6 disposition |
| --- | --- | --- | --- | --- | --- |
| `q4k_g3_lane_partition` | Generated Q4_K G3 packed-word lane partition | portable concept | exact 8B roles exist; unranked | unknown | transfer mechanism only |
| `q6k_cooperative_generated` | Cooperative/spec-generated Q6_K GEMV | portable concept | possible Q6_K roles; unranked | unknown | transfer spec shape, not AMD values |
| `q4k_attn_qo_cooperative` | Role-specific Q4_K q/o coalescing | portable concept | exact shape exists; unranked | unknown | transfer only after role-cost proof |
| `flash_decode_regime_selection` | Split-KV flash plus measured crossover | portable concept | depth 128 is below the AMD winning regime | unknown | defer to depth 512; search threshold anew |
| `q4k_large_shape_coverage` | 14B/32B G3 route coverage | irrelevant first workload | shapes absent from 8B | unknown | retain audit pattern; exclude |
| `attn_k_route_coverage` | Missing Q4_K attn_k route | portable concept | required census row | unknown | transfer coverage invariant, not G3 route |
| `attn_v_route_coverage` | Later missing Q4_K attn_v route | portable concept | required census row | unknown | transfer coverage invariant, not G3 route |
| `q6k_ffn_down_longk` | 14B/32B long-K Q6_K coverage | irrelevant first workload | historical change was 0% on 8B | unknown | exclude initial search |
| `generated_g5_k_only_attention` | Generated G=5 K-only local staging | irrelevant first workload | first model has G=4 | unknown | exclude G=5; retain locality axis conditionally |

## 1. Generated Q4_K G3/lane-partition GEMV

**Portable mechanism.** At decode M=1, lanes cooperatively own contiguous packed Q4_K words for one output row,
dequantize owned words in registers, accumulate partials, and reduce to the row output. The target-neutral descriptor
axes are lane ownership, rows per subgroup, block groups, words per group, load width, dequant placement, reduction,
and output form. Factorization must derive from the live subgroup extent and exact Q4_K/K-block facts.

**Historical applicability.** Qwen3-8B `ffn_gate_up` 12288x4096, `ffn_down` 4096x12288, and `attn_qo` 4096x4096.
Generated G3 was token-identical and route-clean, tracking the owned AMD warp route within -0.13% to +0.41% at
contexts 512-4096.

**Compiler work.** Coalesced aligned packed loads, register dequant/accumulation, target-derived subgroup ownership,
and a correct subgroup or partial-buffer reduction.

**Do not transfer.** Wave32, the 4x8 lane factorization, `ds_bpermute`, AMD warp-reduce code, gfx1100 route ids,
environment defaults, launch geometry, or bandwidth observations.

**Metal mapping.** `unknown`. The portable provider loop proves ordinary generated Metal Q4_K matvec execution, not
profitable subgroup packed ownership or reduction. Reopen only if MR7 attributes at least 5% of whole-step time to an
exact Q4_K role and MR8 can derive all legality from Metal facts.

Evidence:

- `59ff850c1b8a7e5b7ee68ea7640c4326949cbb80`,
  `bench/qk-lanemap-template-ir/latest.json`, blob `59ad6cb23205b054b07207b666d270786fa0d7e5`:
  separates five topology degrees of freedom and losslessly re-emits the three roles; static only.
- `dc90b54a6fe2e53c718975c3cf7e7abc25ad0b8b`,
  `bench/amd-isa-backend-g3-weight-promotion/latest.json`, blob `aba9aa656308246210567cb1ea467eb530478092`:
  AMD token/route/rollback/speed-equivalence gate.
- `08bcd1d9b2ddac389130f3600c113bae80820300`,
  `bench/tg-p2-q4k-g3-policy-driven/latest.json`, blob `ac65623b214b6817d7c4aa10ade79de45f933890`:
  per-tensor machine-emitted selection and strict fallback proof.

Ordinary fallback: generated Metal Q4_K matvec/dequant without an exported selected plan.

## 2. Generated/cooperative Q6_K decode paths

**Portable mechanism.** Replace one-row-per-thread serial Q6_K GEMV with cooperative K ownership: adjacent lanes read
adjacent packed positions, write per-lane partials, and reduce in a generated second stage. Keep `lm_head`, `ffn_down`,
and attention applicability separate because split count and output size change the cost.

**Historical applicability.** On Qwen3-8B, `lm_head` 151936x4096 improved whole decode by 17.7-19.2%, and `ffn_down`
4096x12288 added 12.5-13.2%. The later spec-generated conversion reproduced `lm_head`, `ffn_down`, and `attn_v`
1024x4096 byte-for-byte; its worst generated/shipped timing ratio was 1.0106 and model tokens matched 24/24.

**Compiler work.** Coalesced Q6_K packed loads, local K decomposition, register accumulation, explicit partial traffic,
and a correct generated reduction.

**Do not transfer.** The AMD 16-position lane mapping, row tile 4, local sizes, packed-u16 lowering choices, gfx1100
thresholds, route names, or rollback flags.

**Metal mapping.** `unknown`: Q6_K load quality, partial-buffer cost, legal geometry, and depth-128 role share are all
unmeasured. Reopen only after MR7 selects an exact Q6_K role and an isolated compiler probe validates the mechanism.

Evidence:

- `d77da2bcc02bf4687070d9db07f23e4184e85db9`,
  `docs/qk-mmvq-q6k-lm-head-arc-20260617.md`, blob `1a5ffdb729882db2d72437eafbec5dfe9445d7b6`.
- `00e51531d2f0fef2463f884c9481a0ca9151f4d7`,
  `docs/qk-mmvq-coop-ffn-down-result-20260617.md`, blob `237ff2a374e21edfffc1755f467a47eccf320b49`.
- `06b1608e8fa99a329a9f3580e501e9ecc9ccd525`,
  `bench/tg-p3-q6k-generated-coop/latest.json`, blob `f6474e1216a35860ee3ac36a7f9ca4d2022a2ca2`.

Ordinary fallback: exact-role generated Metal Q6_K dequant/matvec.

## 3. Q4_K attention q/o cooperative route

**Portable mechanism.** Apply cooperative packed ownership only after evidence shows a role-specific coalescing
pathology. Quant type alone is insufficient: on AMD, q/o benefited while Q4_K gate/up did not clear the isolated gate.

**Historical applicability.** Qwen3-8B q/o 4096x4096 rose from 169 to 258 GB/s isolated (1.47-1.52x) and improved whole
decode 5.5-6.4% at contexts 512-4096, byte-identical. Gate/up was only 1.16-1.18x and remained on its ordinary path.

**Compiler work.** Packed Q4_K loads, a local packed-word lane, partial outputs, and a generated sum reduction.

**Do not transfer.** The 4096x4096 runtime hard guard, eight-lane mapping, AMD local sizes, or a “Q4_K means coop”
policy.

**Metal mapping.** `unknown`: exact q/o exists, but no Metal evidence establishes poor coalescing, reduction
amortization, or material whole-step share. Reopen only on MR7 role-cost plus isolated headroom evidence.

Evidence:

- `d94468a4ca645d601f7835fb92d4e9bfdef37c92`,
  `docs/qk-mmvq-coop-q4k-attn-result-20260617.md`, blob `c58af797a028d84e26dff251791c21f862532d60`.
- `c42f055c5ec93b8cd0fc6b136e7acd61aef90a41`, historical `tinygrad/llm/model.py`: exact AMD role/shape guard and
  fallback. The hard guard documents historical execution; it is not the portable descriptor.

Ordinary fallback: generated Metal Q4_K projection matvec.

## 4. Flash-decode threshold and implementation regimes

**Portable mechanism.** Split the KV sequence, compute online max/sum/weighted-value state, and combine only where
the added fixed work is amortized. Keep SDPA below a measured target/workload crossover. Route per decode token rather
than letting graph capture order freeze the first regime.

**Historical applicability.** Qwen3-8B, Hq=32, Hkv=8, Hd=128, T=1. AMD flash regressed at contexts 128 and 256,
crossed near 384, and increasingly won through 4096. The safe default moved from 1024 to 512 only after a
byte-identical +12.8% real-generate result near ctx520. A later fix made short-start sessions actually cross graphs.

**Compiler/runtime work.** Symbolic decode length, split-KV work, online state, stable combine, and identity-preserving
dual-graph routing or an equivalent ordinary interface.

**Do not transfer.** Threshold 384/512/1024, split counts, AMD local sizes, the owned HIP tile, S=4/S=48 regimes,
OCML calls, or AMD occupancy observations.

**Metal mapping.** `unknown`. Depth 128 is inside the historical losing regime, so flash is not an initial candidate.
Reopen at the required depth-512 evaluation only after MR5 identity and MR7 attention cost, then search the Metal
crossover rather than importing 512.

Evidence:

- `a8694ba732d520eb4a8ad0725e91c699da3530b0`, `bench/qk-flash-search/flash-search.json`, blob
  `34bdce845d0896551e7059c4f559aa6b90da50f4`: initial measured crossover.
- `d590b55f38b4f162b6b4ed24c4cad27eb883c9f1`, `bench/qk-flash-decode-auto-20260617/result.json`, blob
  `9d77f8d2fbb4eb5976f1c83fec6d1589011cf761`: conservative auto regimes.
- `191223099d56819229c0cab46c0c3ebbc404acbb`, `docs/qk-8b-attention-fusion-result-20260617.md`, blob
  `ecadc2b44f5ac87a3e16be2d5e2022e9515d2779`: measured 1024-to-512 cutover.
- `4e7dc1a4fdad3246024e983b5a9e4fe73a7750e8`, `docs/decode-lowctx-nonflash-gap-result-20260630.md`, blob
  `dff7d5f27621365dd07071559e9c0eb3c892c4c1`: graph-crossover/capture-order correction.

Ordinary fallback: generated Metal SDPA/attention.

## 5. Larger-shape Q4_K G3 coverage

**Portable lesson.** Kernel representability, route coverage, and shape tuning are three different gates. The AMD G3
kernel was numerically correct for six 14B/32B shapes, while hardcoded 8B guards missed 100% of their Q4_K linears;
structural binding improved 14B by 8-9%, but the reused topology remained `SEARCH_SPACE_INCOMPLETE`.

**Do not transfer.** `(K/256)%4==0`, `N%32==0`, the “ANYSHAPE” flag, AMD G3 topology, or model names.

**Metal mapping.** `unknown` and irrelevant to the first exact 8B workload. Exclude from initial MR8. Reopen only when
a later Metal model has uncovered larger Q4_K shapes and a route census proves fallback cost.

Evidence: `baafb13215b0d34ca1bcc4e4a421a4ac97b92a54`,
`docs/qwen-14b-32b-truegen-q1432-result-20260630.md`, blob
`6e09f4fd52ca2d930ca7c813185d5c41d9287e9b`.

Ordinary fallback: generated Metal route for every unmatched shape.

## 6. `attn_k` route coverage

**Portable mechanism.** Complete semantic-role coverage with an explicit census. Coverage does not choose a route.
On AMD, omitted Q4_K `attn_k` silently fell through to lazy dequant; routing it to G3 won strongly on 14B/32B, while
an initially different 8B route regressed 4%. That protected control is why “same quant/role family” cannot authorize
a global route.

**Historical applicability.** 14B `attn_k` 1024x5120 owned 38% of decode. Generated packed routing improved 27.8 to
44.5 tok/s at ctx128 and 27.1 to 42.8 at ctx512. The later paired large-shape policy also improved 32B.

**Do not transfer.** `attn_k -> G3`, the large-model default, AMD shape rules, or kernel-name inference.

**Metal mapping.** `unknown` pending MR5 stable identity and MR7 cost. MR5 must emit an `attn_k` census result even if
the ordinary fallback is correct.

Evidence:

- `9bda1ba0e0474bd00131cfe22e2c96cb3615efd1`,
  `docs/qwen-14b-32b-attn-k-route-miss-result-20260630.md`, blob
  `77d2cb8d8a571af51c0cd34944e830c55e5f3113`.
- `dc4c9f9f8bf9400941538f7f78538a3a19d9d7d2`, `bench/models/qwen/amd-rx7900xtx-gfx1100.md`, blob
  `ec734003c743451876a2f5cf04f5b0ef8fa13312`.

Ordinary fallback: generated Metal linear/dequant.

## 7. `attn_v` route coverage

**Portable mechanism.** Audit every semantic role independently. Fixing `attn_k` did not reveal that the adjacent
`attn_v` role was also omitted. Model facts and binding counts, not adjacency or matching shape, own completeness.

**Historical applicability.** Q4_K `attn_v` 1024x5120 fell to about 24 GB/s and 12.5% of 14B decode. The AMD route
fix was byte-identical and improved 8B by 8.9% and 14B by 13.3%.

**Do not transfer.** `attn_v -> G3`, `DECODE_ROUTE_ATTN_V`, AMD bandwidth reasoning, or “same shape as attn_k” as a
route rule.

**Metal mapping.** `unknown` pending MR5/MR7. It remains a separate mandatory census row even when its shape matches
`attn_k`.

Evidence:

- `6ee3bec5da729c1d7d297b0ac276390ae18d03f4`, historical `tinygrad/llm/route_policy.py`, blob
  `311ee951e44dee56d9c08e7f6d8811277a8be0be`: exact omission, rollback, and measured results in commit and code.
- `8f93398659f823322bd1a19e541f8cf55b023f7b`, historical `README.md`, blob
  `09a307ac9178a6a076ef4e77ad3c6bebcff9c2f7`: post-fix multi-model authority table.

Ordinary fallback: generated Metal linear/dequant.

## 8. Larger-shape Q6_K `ffn_down` coverage

**Portable lesson.** Replace model-dimension routing with measured structural candidates, while keeping semantic role
exclusions explicit. The AMD long-K cooperative rule improved 14B by 17% and 32B by 20%, but moved 8B by 0% because
8B already had coverage.

**Do not transfer.** `K>=8192`, `N<100000`, the AMD coop route, or its default.

**Metal mapping.** `unknown` and irrelevant to the initial 8B search. Exclude from MR8; reopen for a later model only
after an exact route census proves uncovered Q6_K `ffn_down` cost.

Evidence: `8920268217f67ccf271726e30bdb8a2a4d1a6834`,
`docs/qwen-14b-32b-l3-q6k-ffn-down-longk-result.md`, blob
`83d105f5e900195911145c42406d7c86c519a930`.

Ordinary fallback: generated Metal Q6_K `ffn_down`.

## 9. Generated G=5 K-only attention

**Portable mechanism.** Make locality an explicit candidate axis: stage K in local memory while reading V through a
measured warm-cache path. K-only is legal only when producer/consumer cache evidence supports it. `KV_BOTH` and the
ordinary path remain separate controls.

**Historical applicability.** Qwen3-14B Hq=40, Hkv=8, G=5, Hd=128. The generated UOp K-only route halved AMD local
memory from 8192 to 4096 bytes, removed V staging, and improved whole decode 7.8% at ctx512 and 14.7% at ctx2048. The
original K+V G=5 block tile had regressed about 80%; locality, not grouping alone, made the later win.

**Compiler work.** Generated local memory/barrier, target-derived subgroup/workgroup geometry, online softmax, efficient
K staging, measured cache-resident V loads, and resource/static-instruction evidence.

**Do not transfer.** G=5, 160 threads, TK=16, L=128, AMD `ds_bpermute`/`fdot2`, the historical producer label, AMD
cache-warm inference, or K-only as a default.

**Metal mapping.** `unknown` and not applicable to Qwen3-8B G=4. Exclude G=5 from initial MR8. Reopen only the locality
axis if MR7 selects attention and a controlled Metal probe demonstrates V residency plus a K-only win against both
K+V and ordinary attention.

Evidence:

- `3058b03c9463c3cde899d18b9a0448e6e4599b19`, `docs/gp5-final-report.md`, blob
  `2898e819e1980344ac26715ad9a701ee6b7fca3a`: generated provenance, correctness, exact applicability, mechanism,
  whole-model gains, and rollback.
- Same commit, `bench/g5-block-tile/compiler_pathology_v1.json`, blob
  `8d525c8f5a7161216cfc77ea20b852c2123d130e`: AMD 8192-byte local-memory/resource diagnosis.

Ordinary fallback: generated Metal SDPA/attention.

## MR6 gate result

- Every named historical win/fix has an exact commit and recoverable artifact or source blob.
- Portable mechanism, compiler work, and AMD-only implementation are separated for each row.
- Metal capabilities remain explicit `unknown`; no AMD defaults or speed claims cross targets.
- Route completeness is independent of route selection.
- Large-shape and G=5 rows are excluded from the first 8B population.
- The matrix authorizes no promotion, runtime binding, or candidate ranking.

MR6 is complete as a history/code-analysis packet. MR7 owns the next decision: measured Metal role cost may select at
most two role families for MR8.
