# NV vocab-head aux scatter-chain fusion scope (the L4 tail)

Date: 2026-08-12
Branch: `nvidia-bringup-20260731` (HEAD `a8b560457`; same-session gap
attribution ladder step 2)
Status: **implementation/test scope. Authorizes fusing the four vocab-head
aux kernels (`E_1187_32_4`, `r_32_4_1187`, `r_128_16_8_1187`, `r_16_8` -
~57.3 us/token) into the landed `q6k_gen_coop_151936_4096_inkernel` vocab
GEMV epilogue (or into a packed-argmax reduce), keeping the top-1 token
bit-exact (`9e6664fd...`). No production default changes, no promotion
without the +50 us wall bar.** Process: audit -> arithmetic -> implement
(standing pipeline, `0515f2539`).

## 1. Why this scope exists

The same-session gap attribution (d512, Qwen3-8B-Q4_K_M / RTX 5090) prices the
vocab head at **380.8 us/token** vs llama's single mmvq node at **303.6 us**.
The GEMV itself is near parity (323.5 vs 303.6, 1.07x); the delta is the
**57.3 us aux scatter chain** that llama does not pay. The L4 substrate fusion
(08-03) already moved the vocab head to the `in_kernel` cooperative route and
booked ~-85 us; this scope closes the tail the L4 scope named as its baseline
stack and left standing.

## 2. Audit (fresh same-session trace, kernels 2122-2126)

Read-only extraction from `/tmp/tg_debug_probe_20260812.log` (same attribution
trace; 582 rows in window):

| kernel | count | sum us | role |
| --- | ---: | ---: | --- |
| `q6k_gen_coop_151936_4096_inkernel` | 1 | **323.5** | vocab GEMV, coop in-kernel (landed L4) |
| `E_1187_32_4` | 1 | **4.4** | logits elementwise over 151936 rows |
| `r_32_4_1187` | 1 | **39.2** | staged reduce of the 32x4 tile |
| `r_128_16_8_1187` | 1 | **11.4** | staged reduce |
| `r_16_8` | 1 | **2.3** | final argmax/top-1 reduce |
| **aux total** | 4 | **57.3** | llama: 0 |
| **vocab total** | 5 | **380.8** | llama: 303.6 (1 mmvq node) |

Llama reference (same-session nsys ledger): one `mmvq` node, 303.6 us, with
the top-1 selection done on the host from the 151936-row logits buffer; no
separate argmax kernels on the GPU. Our 4 aux kernels are the GPU-side
top-1 chain that survived the L4 landing.

Prior row history (do not relitigate):

| attempt | date | verdict |
| --- | --- | --- |
| L4 vocab substrate fusion (coop in-kernel head) | 08-03 | **LANDED** (~-85 us vocab-path saving, 407.2 -> 322.24 us; AB checkpoint 08-05) |
| packed argmax microgate tooling | 08-05+ | exists (`extra/llm_research/decode/nv_packed_argmax_microgate.py`); diagnostic surface for exactly this chain |

## 3. Arithmetic

The aux kernels are pure top-1 plumbing: their input is the 151936-row logits
buffer produced by the GEMV; their output is one token id (and the sampled
logit for the host). Removing them means moving the argmax into the GEMV
epilogue (per-tile max + index, cross-tile reduce in-kernel) or into a single
packed reduce. This is body-adding for the GEMV (small: max+idx per lane is
register-local) and pure removal for the aux chain, so the 0.6 body-adding
mapping is conservative.

| basis | census us | map | wall us | new ms/token | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline (production HEAD) | - | - | - | 5.2031 | 192.19 |
| aux chain at 1:1 (pure removal) | 57.3 | 1.0 | 57.3 | 5.1458 | **194.3** |
| aux chain at 0.6 (body-adding floor) | 57.3 | 0.6 | 34.4 | 5.1687 | **193.5** |
| GEMV body delta (323.5 vs 303.6) | 19.9 | - | - | - | ~+0.8 |

The +50 us promotion bar: the aux chain alone does not clear it at either
mapping; this row books as a package with the GEMV body delta or against the
bar as measured on the reverse wall bracket (same situation as the M2d fp16
combine landing, which booked +35.8 us below the bar on the M2b/M2c package).

Correctness contract: the top-1 selection must stay bit-exact. The token
stream sha256 `9e6664fd1d67a6124e786daaa1d895bdb64b972c3991c54dd5fcc6cea16f6881`
is the gate; ties in argmax (identical logits) must resolve identically to
today's `r_16_8` semantics, so the fused reduce must use the same comparison
(first-index-wins or last-index-wins as currently implemented - pinned by a
unit test before the fold).

## 4. Implement plan

### P1: CPU

1. Pin the current top-1 semantics: unit test on `r_16_8`-equivalent argmax
   (comparison order, tie handling) over the 151936-row logits shape.
2. Extend the vocab GEMV epilogue (or the packed argmax microgate recipe) to
   carry per-tile (max, index) and reduce them in-kernel; render/SASS review
   for the max+idx compare (no float casts that could reorder ties).
3. Hermetic gate on DEV=CPU: fused route produces the identical token id over
   the qualification prompt set; logits buffer (if still materialized) stays
   byte-identical.

### P2: real-token A/B (GPU, lock-held)

1. Single-arm A/B at d512 (the L4 checkpoint discipline,
   `nv-decode-l4-vocab-ab-checkpoint-20260805.md`): candidate vs control,
   exact token sha, census (aux 4 -> 0, GEMV 1:1 swap), reverse wall bracket.
2. Book the row (alone or as a package); promotion record in
   `docs/task_workflow/input/`.

## 5. Gates (hard stop)

1. Bit-exact top-1: token sha256 `9e6664fd...` on every rep; tie semantics
   pinned by unit test before the fold.
2. Census contract: `E_1187_32_4`, `r_32_4_1187`, `r_128_16_8_1187`,
   `r_16_8` each 1 -> 0; `q6k_gen_coop_151936_4096_inkernel` 1:1 (or a
   documented epilogue variant).
3. Exact-output native A/B, reverse wall bracket, +50 us promotion bar (or
   package booking per the M2d precedent), promotion record.
4. NV sm_120-only route (L4 constraint): no AMD/default-device behavior change.

## 6. Evidence

- `docs/task_workflow/evidence/nv-tinygrad-prime-gap-table-20260812.json`
  (vocab_aux class: 5 kernels / 380.8 us)
- `docs/task_workflow/evidence/nv-llama-d512-node-ledger-20260812.json`
  (llama vocab mmvq 303.6 us)
- `docs/task_workflow/input/nv-decode-gap-attribution-same-session-20260812.md`
  (ladder step 2)
- `docs/task_workflow/input/l4-vocab-substrate-fusion-implementation-scope-20260803.md`
  and the L4 AB checkpoint (prior landing)
- Raw: `/tmp/tg_debug_probe_20260812.log` (kernels 2122-2126)
