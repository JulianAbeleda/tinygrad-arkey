# NV 220 composition scope - the composed full-token topology and its fold map

Date: 2026-08-15
Branch: `nvidia-bringup-20260731` (HEAD `62b4ec839`)
Status: **definition scope. No code, no measurement, no GPU session.** This is
the follow-on to `nv-substrate-definition-20260815.md`: that doc established
that 220 is a *composed topology*, not another primitive. This doc defines the
composition concretely - the end-state node census, the fold map from each
remaining support class to its absorbing GEMV, the dependency order, and the
gates. Nothing is authorized for implementation until the fold map is reviewed.

## 1. The target in one sentence

Absorb every remaining non-flash support kernel body-free into the GEMV that
consumes it, so one decode token runs ~253 fused-quant GEMVs plus the ~72 flash
and a handful of irreducible nodes, instead of today's 582+ serial kernels.

The composition is **fusion**, not overlap. The 08-13 node ledger already
reconciled this: llama's 946 us of overlap mass and our ~652 us fusion ceiling
are the same support work viewed from two sides. We already captured the
quantize/norm/rope portion of llama's overlap by fusing it; the remainder is
the open support classes below, absorbed into GEMV epilogues.

## 2. Baseline and target (all numbers pinned)

| quantity | value | source |
| --- | ---: | --- |
| current production wall | 5.102 ms/token = ~196 tok/s | 08-14 vocab A/B control arm |
| target | 4.545 ms/token = 220 tok/s | gap-attribution ladder step 7 |
| wall to remove | ~557 us | subtraction |
| llama reference | 4.074 ms/token = 245.5 tok/s | same-session pair |

The 220 number is the "all class deltas closed at 1:1" ceiling
(`nv-decode-gap-attribution-same-session-20260812.md` step 7). Realistic
body-adding folds map ~0.6 census-to-wall; pure kernel removal maps ~1.0. The
composition below is ordered so that pure-removal folds come first and
body-adding folds last, which is what makes the ceiling reachable rather than
optimistic.

## 3. The fold map (exhaustive)

Each row is one open support class on the current production `DEV=NV` route,
its census mass, the absorbing GEMV, the mechanism, and its status. Census
masses are carried from the 08-12 same-session attribution (authoritative for
production) unless another record supersedes it.

| # | support class | census us/token | absorbing GEMV | mechanism | status |
| --- | --- | ---: | --- | --- | --- |
| F1 | reduce-output fp32 q/k (`32_128` + `8_128`, 70 kernels) | 240.5 | attention Q/K GEMV epilogue | generic reduce-output primitive C1, body-free admission | primitive shipped; body-free removal unproven |
| F2 | reduce-output FFN-down (`1_4096`, 19 kernels) | 151.5 | FFN-down GEMV epilogue | generic reduce-output primitive C1 | primitive shipped; body-free removal unproven |
| F3 | M1 norm chains (`r_16_256` + `E_32_32_4_f14a5cc0`, 72 kernels) | 229.5 | gate/up `w1w3fused16` GEMV | reduce-output primitive: reduce+epilogue -> one fused body | primitive shipped; old body-adding fold NO-GO +81.9 us |
| F4 | attention K/V extras + fp32 q/k epilogue (residual/plumbing) | ~247.6 | attention epilogue / residual folds | M4-style typed-view residual fold | M4 machinery landed; remainder partially folded |
| F5 | vocab aux (`E_1187`, `r_32_4_1187`, `r_128_16_8_1187`, `r_16_8`) | 57.3 | vocab GEMV (single pass) | cross-tile max inside the vocab GEMV | u64 packed two-program path NO-GO -1.11% |
| F6 | flash score structural excess | 166.5 (90 at floor) | n/a | not foldable; sits on the critical path | tile sweep NO-GO; structural, separate item |
| | **total open** | **~1093** | | | |

The census-to-wall mapping: F1-F5 pure removal sums to ~926 us census; at ~0.6
realistic map that is ~557 us wall, which is exactly the 196 -> 220 delta.
F6 is the only row that is not a fusion target; it is structural and is
handled separately, not inside this composition.

## 4. The end-state census

| class | today (production) | composed |
| --- | ---: | --- |
| quantized matvec GEMVs | ~253 (q4k 2812.5 us + q6k 911.8 us) | ~253 (bodies unchanged) |
| reduce-output | 89 kernels / 392 us | 0 (absorbed by F1+F2) |
| M1 norm chain | 72 kernels / 229.5 us | 0 (absorbed by F3) |
| residual/plumbing | ~247.6 us spread | ~0 residual body (absorbed by F4) |
| vocab aux | 4 kernels / 57.3 us | 0 (absorbed by F5) |
| flash score + combine | 72 kernels | 72 (combine unchanged; score to floor F6) |
| **token** | **582+ serial kernels** | **~253 GEMV + ~72 flash + <10 irreducible** |

This is the "one long fused-quant GEMV anchor" from the substrate definition,
stated precisely: the anchor is the existing 253-kernel GEMV chain with all
support folded into its epilogues, not a new mega-kernel.

## 5. Dependency order

Folds are ordered so each step is a strict program-count reduction with no
surviving-body growth, and no step depends on a later one.

1. **F3 then F1/F2 share the same primitive** (`ReduceOutputSpec` body-free
   admission). Do F3 first: it is the simplest 2->1 collapse and its old
   failure (+81.9 us) was body-adding, so it is the cleanest proof of the
   body-free path. Then apply the same proof to F1 and F2 per-site.
2. **F4 residual/plumbing** rides the M4 typed-view residual-fold machinery
   that is already landed; the remaining unfused pieces are site-by-site
   epilogue admissions.
3. **F5 vocab aux** last, because it needs a new single-pass cross-tile max
   mechanism inside the vocab GEMV; the packed-u64 carry is a usable substrate
   for that single pass even though the two-program chain lost.
4. **F6 flash score** is out of the fusion order; its only known movement is
   the structural floor (90 us), which is a separate first-principles flash
   rewrite, not a fold.

## 6. Gates (hard)

Every fold must pass, in order:

1. boundary-free ordinary-UOp gate (no custom-program boundary, no CONTIGUOUS);
2. hermetic CPU admission + production decode census showing body-free program
   removal (no weight materializations, no surviving-body growth - the 08-09
   net +72 lesson);
3. exact-output native A/B: token stream and full-logit SHA-256 identical to
   control, changed-node census confined to the named class;
4. reverse wall bracket with +50 us promotion bar, booked per-site or as a
   package where the scope already requires it (F1+F2 package).

## 7. What this composition is NOT

- not the llama `quantize_q8_1` anchor - we fused that work away; the anchor
  mechanism does not transfer (`nv-anchor-verdict-20260814.md`);
- not per-shape GEMV tuning - Q4 FFN-down and Q6 attention-V are closed NO-GO;
- not launch hiding or multi-stream - the CUDA multi-stream lowerer buys ~0 on
  the chain-like DAG and the overlap ceiling is the same mass as this fusion
  ceiling;
- not 245 - full parity additionally needs the closed per-shape GEMV rows and
  llama's launch hiding, which are outside this scope.

## 8. Review checkpoint

The only claim this scope makes that has not already been measured is that the
F1-F5 folds compose to ~557 us wall without adding body work. Every individual
fold and its ceiling is already pinned in its own record. The review should
challenge the composition (does fold A's body-free construction survive once
fold B is also active on the same GEMV chain?), not re-derive any individual
ceiling.
