# NV reduce-output per-site absorption scope (fp32 q/k + FFN-down sites)

Date: 2026-08-12
Branch: `nvidia-bringup-20260731` (HEAD `a8b560457`, fp32 q/k promoted;
same-session gap attribution ladder step 5)
Status: **implementation/test scope. Authorizes per-site admission of the
remaining `reduce_output_rmsnorm` kernels through the landed generic
cooperative reduction-to-output primitive, one site at a time, each with its
own hermetic gate and real-token exact-output A/B. No production default
changes, no promotion without the +50 us wall bar.** The phase6 mechanism is
closed and is not re-run (NO-GO, candidate 18.5 us SLOWER). Process:
audit -> arithmetic -> implement (standing pipeline, `0515f2539`).

## 1. Why this scope exists

The same-session gap attribution (d512, Qwen3-8B-Q4_K_M / RTX 5090) shows the
`reduce_output` family as the largest single open kernel-sum row: **89 kernels
/ 392.0 us/token** that llama does not pay at all (its mmvq absorbs the output
reduce in-kernel). The ladder prices the row at **~201.3 tok/s** at the 0.6
body-adding census-to-wall mapping, i.e. about +9 tok/s from the 192.19
production baseline. The Q4 FFN-down GEMV row closed NO-GO the same day
(`nv-q4kd-load-pattern-measurement-record-20260812.md`), so this row is now
the largest open epilogue mass on the quant side.

The generic primitive that this row needs is already shipped: the 08-09
capability gate (`nv-generic-reduce-output-primitive-record-20260809.md`)
made `REDUCE_OUTPUT` shape/recipe-generic and proved admission for the norms
population in the production decode census. What it did NOT prove is
body-free program removal: the CALL-input route emitted one fused body per
consuming call argument plus one weight materialization each (net +72
programs in that census). This scope therefore gates each site on body-free
removal, not on admission alone.

## 2. Audit (fresh same-session trace, kernels 1545-2126)

Read-only extraction from `/tmp/tg_debug_probe_20260812.log` (the same
attribution trace; 582 rows in window):

| family | count | sum us | mean us | placement |
| --- | ---: | ---: | ---: | --- |
| `reduce_output_rmsnorm_32_128` | 35 | **129.6** | 3.70 | follows the q/k GEMVs (fp32 q/k route, attention side) |
| `reduce_output_rmsnorm_8_128` | 35 | **110.9** | 3.17 | follows the q/k GEMVs (fp32 q/k route, attention side) |
| `reduce_output_rmsnorm_1_4096` | 19 | **151.5** | 7.97 | follows the FFN-down GEMVs (residual-add side) |
| **total** | 89 | **392.0** | 4.40 | llama: 0 (absorbed in-kernel) |

Same-session llama reference: the nsys node ledger has no reduce-output class;
its mmvq `has_fusion=true` variants carry the output reduce in-kernel. The
delta is 392.0 us/token, 100% recoverable in principle.

Prior row history (do not relitigate):

| attempt | date | verdict |
| --- | --- | --- |
| fp32 q/k route (`reduce_output` on q/k) | 08-10/12 | **BOOKED +83.5 us wall** (promoted `a8b560457`); the reduces are its remaining epilogue cost |
| phase6 route efficiency (reduce into a single fused program) | 08-10 | **NO-GO: candidate 18.5 us/token SLOWER** (5.420 vs 5.401 ms); gates PASS but wall bracket fails |
| generic reduce-output primitive (C1) | 08-09 | CPU capability gate **PASS**; admission proven, body-free removal NOT proven (net +72 in the CALL-input census) |
| M1 norm chains via the primitive | 08-12 | scope only (this row's sibling, `nv-m1-norm-epilogue-generic-primitive-scope-20260812.md`); same body-free admission contract |

## 3. Arithmetic

Census-to-wall mapping (observed in the fp32 q/k booking at ~0.61):
**body-adding changes map ~0.6; pure kernel removal maps ~1.0**. The reduces
are body-free removal targets ONLY if the fused body does not add weight
materializations (the 08-09 CALL-input lesson) and no surviving kernel grows.

| site | census us | 0.6 map wall us | new ms/token | tok/s | +delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp32 q/k site (32_128 + 8_128) | 240.5 | ~144 | 5.059 | **~197.7** | +5.5 |
| FFN-down output site (1_4096) | 151.5 | ~91 | 5.112 | **~195.6** | +3.4 |
| both sites | 392.0 | ~235 | 4.968 | **~201.3** | +9.1 |
| both sites at 1:1 (pure removal) | 392.0 | 392 | 4.811 | **~207.9** | +15.7 |

The +50 us promotion bar: both sites together clear it at the 0.6 mapping
(~235 us wall); each site alone does not (144 / 91 us), so the sites must be
booked as a package or against the bar as measured per-site on the reverse
wall bracket. The 32_128/8_128 pair is one site (same consumer chain); the
1_4096 site is independent.

Why phase6 failed and what changes here: phase6 rebuilt the reduce into a
single fused program and added body work to the surviving chain. This scope
instead admits each existing `reduce_output_rmsnorm` shape through the
already-generic primitive with the M4-style typed-view ownership contract,
and requires the census to show the SAME program count (1:1 swap) or a
strict reduction with zero weight materializations.

## 4. Implement plan

### P1: per-site admission (CPU)

1. Admit `reduce_output_rmsnorm_32_128` and `_8_128` (q/k site) through the
   generic `ReduceOutputSpec` with the production spellings from the fp32 q/k
   route; the hermetic suite in `test/unit/test_generic_reduce_output.py`
   must pass with no new weight materializations.
2. Admit `reduce_output_rmsnorm_1_4096` (FFN-down site) the same way; keep the
   `sumsq_rsqrt_affine` recipe byte-identical to the shipped body digest
   (`c82e25f5...`).
3. Production decode census on DEV=CPU showing body-free program removal
   (per-site program count identical or reduced, no `*_weight_store`-style
   additions), logits SHA-256 identical to `9e6664fd...`.

### P2: real-token A/B (GPU, lock-held)

1. Single-site A/B at d512: candidate vs control, exact-token sha, census
   contract, reverse wall bracket (the `nv_reduce_output_fp32_qk_ab.py`
   discipline, self-managed lock).
2. Book each site that clears the +50 us bar; if neither clears alone,
   book the package against the combined bar.
3. Promotion record in `docs/task_workflow/input/` with the wall bracket,
   census swap table, and logits sha.

## 5. Gates (hard stop)

1. No phase6 relitigation: the 18.5 us-slower mechanism stays closed.
2. Each site passes the boundary-free ordinary-UOp gate and the hermetic
   admission before any GPU arm.
3. The census must show body-free removal: no weight materializations, no
   surviving-body growth (the 08-09 net +72 lesson).
4. Exact-output native A/B, logits SHA-256 identical, reverse wall bracket,
   +50 us promotion bar, promotion record.

## 6. Evidence

- `docs/task_workflow/evidence/nv-tinygrad-prime-gap-table-20260812.json`
  (reduce_output class: 89 / 392.0 us)
- `docs/task_workflow/evidence/nv-llama-d512-node-ledger-20260812.json`
  (llama side: no reduce-output class; mmvq absorbs it)
- `docs/task_workflow/input/nv-decode-gap-attribution-same-session-20260812.md`
  (ladder step 5; per-site tok/s ceilings)
- Raw: `/tmp/tg_debug_probe_20260812.log`
