# NV M1 norm epilogue scope via the generic reduce-output primitive (body-free admission)

Date: 2026-08-12
Branch: `nvidia-bringup-20260731` (HEAD `48b8fa696`, tok/s ladder landed on the
same-session gap attribution)
Status: **implementation/test scope. Reopens the M1 norm-epilogue row
(192.19 -> ~193 tok/s, the ledger's next line) through a BODY-FREE admission
on the now-generic cooperative reduction-to-output primitive. The dead fold
(epilogue INTO the w1w3fused GEMV body, NO-GO +81.92 us/token, cost gate
CONTRADICTED) is not re-run; the fresh construction is one fused
`reduce_output_rmsnorm_1_4096` body per ffn-norm chain replacing the
`r_16_256` + `E_32_32_4_f14a5cc0` pair with no body work added to any
surviving kernel. The primitive admission must be demonstrated CPU-first
(hermetic gate + production decode census showing body-free program removal)
before any NV arm, lock, or wall bracket. No policy promotion, no model wiring
change in the CPU phase.**
Process: audit -> arithmetic -> implement (standing pipeline, `0515f2539`).

## Why

Production decode is 192.19 tok/s (5.2031 ms/token, d512, Qwen3-8B-Q4_K_M,
RTX 5090 sm_120, token stream `9e6664fd...`). The same-session gap
attribution prices the next ladder step at **~-22 us wall for 193 tok/s**
(exact -21.8 us), and the ledger's line to 193 is the **M1 norm epilogue row**
(rank 4: `r_16_256` + `E_32_32_4_f14a5cc0` ffn-norm chains, 229.5 us census,
"~197-193 line"). The M2a/M2b/M2c/M2d residual/cast/contiguous row is
exhausted; M1 is the next in-row item.

Prior norm attempts and their closures:

| attempt | date | verdict | reason |
| --- | --- | --- | --- |
| M3 opaque fused norm kernel | 08-02 | NO-GO | 144 input-boundary copies + 72 output materializations, -3% decode (173.45 -> 168.42 tok/s) |
| boundary-free ordinary-UOp fusion | 08-06 | NO-GO | `CONSTRUCTION_GAP`: consumer GEMV is an opaque custom program; no cross-thread reduction-to-output primitive existed |
| M1 epilogue fold into w1w3fused (`q4k_gate_up_rms_affine_qualification_call` + `_rms_affine_gateup_norm_weight`) | 08-12 | **NO-GO +81.92 us/token, cost gate CONTRADICTED** | body-adding: epilogue re-executes per matrix dot (R=2), x streams fp32 16KB vs fp16 8KB, `r_16_256` retained; unmodeled in-kernel critical path + activation traffic dominated |
| generic cooperative reduction-to-output primitive (C1) | 08-09 | **CPU capability gate PASS** | 108 selector admissions / 54 fused bodies in the decode census; admission capability proven, body-free program removal not yet proven (CALL-input route added 54 weight materializations) |

The C1 unblock is shipped (shape/recipe-generic `REDUCE_OUTPUT`, M4-style
typed-view admission, CPU gate green), so M1 can be reopened as a body-free
admission through the primitive instead of as a fold into the dominant GEMV
family. That is the whole point of this scope: the mechanism that lost 81.9 us
added body work to a 1394.5 us census kernel family; the primitive mechanism
removes one program per chain (2 -> 1) with zero surviving-body change.

## Audit

### Fresh trace chains (same-session prime window, kernels 1545-2126)

Read-only extraction from `/tmp/tg_debug_probe_20260812.log` (the same
attribution trace; 582 rows in window):

| family | count | sum us | mean us | median us |
| --- | ---: | ---: | ---: | ---: |
| `r_16_256` (ffn scale reduce, `ed256c4a...`) | 36 | **146.3** | 4.06 | 3.85 |
| `E_32_32_4_f14a5cc0` (fp16 norm epilogue, `f14a5cc0...`) | 36 | **83.2** | 2.31 | 2.30 |
| **chain total (36 adjacent r->E pairs)** | 36 | **229.5** | 6.38 | 6.21 |

Every `r_16_256` is immediately followed by one `E_32_32_4_f14a5cc0` (36/36
adjacent pairs), then the fused `w1w3fused16` gate/up GEMV consumes the fp16
normed hidden state. This is the chain the generic primitive would collapse:
reduce + epilogue -> ONE `reduce_output_rmsnorm_1_4096` body per chain, the
same fp16 output ABI the GEMV reads today.

llama reference (same-session nsys ledger,
`docs/task_workflow/evidence/nv-llama-d512-node-ledger-20260812.json`):
`rms_norm` 145 nodes, node-sum 307.619 us = **2.12 us/node**, one fused
`rms_norm_f32` kernel per norm, norm arithmetic never enters the matmul. The
fused-body end-state is llama-shaped: our chain mean of 6.38 us versus llama's
2.12 us single node is the gap the primitive closes.

### Why the old fold lost +81.92 us (body-adding cost)

The dead fold moved the norm epilogue INTO the `q4k_g3_lanemap_gemv_w1w3fused16`
GEMV body. The cost model (`nv_fusion_cost_model.py`,
`COST_PREDICTION` in `nv_epilogue_absorption_m1_ab.py`) stated the premise
arithmetic:

| term | value | effect |
| --- | --- | --- |
| redundancy R | 2 | the epilogue `(half)((h*s)*w)` re-executes once per matrix dot (gate AND up) |
| x traffic | fp32 16KB vs fp16 8KB | doubles activation reads across all rows of the 12288-wide GEMV |
| `r_16_256` scale reduce | retained (bitwise contract) | the reduce program does NOT fold away |
| unmodeled | in-kernel critical path, activation traffic | register pressure / occupancy on a ~38.7 us kernel |

Prediction formula: `blocks x [(R-1) x M_removed - R x launch_us]` =
`36 x [(2-1) x 2.29 - 2 x 1.5]` = **-25.6 us** point (range -61.6 to +10.4).
Measured: **+81.92 us** (candidate SLOWER), outside the range on the opposite
side of zero -> **CONTRADICTED**, and the campaign failed closed. The census
contract itself was a tell: `r_16_256` stayed 37 == 37 and `w1w3fused16` swapped
1:1 to `w1w3_rms_affine16` - the fold MOVED work into the GEMV body, it did not
remove programs. The epilogue census it could ever drop (83.2 us) was swamped
by body cost added to the dominant 1394.5 us GEMV family.

### What the 08-09 generic census proved, and did not

The generic primitive record (`nv-generic-reduce-output-primitive-record-20260809.md`)
and census artifact (`docs/task_workflow/output/nv-generic-reduce-output-census-20260809.json`):
108 selector admissions, 54 fused `reduce_output_rmsnorm_1_4096` bodies in the
captured decode graph, removed 18 `r_16_256` + 18 `E_32_32_4_f14a5cc0` - but
the CALL-input route emitted one fused body per consuming call argument PLUS
one weight materialization each (54 fused + 54 weight-store programs), so the
captured call count was NOT reduced (net +72 vs the typed baseline). The record
states it honestly: admission capability proven, **body-free program removal
not yet proven** in the production DAG. The hermetic single-consumer
STORE/CALL form still fuses 2 -> 1 in isolation, and the microgate measured
the fused body -1.253 us vs the ordinary two-program RMSNorm pair in isolation
(55.374 -> 54.121 us). The M1 admission probe must close exactly that gap: one
fused body per chain, no weight materialization, no w1w3fused body change.

## Arithmetic

### Chain census -> wall translation

Census-to-wall mapping (attribution rule, observed in the fp32 q/k booking at
~0.61): **body-adding changes map ~0.6; pure kernel removal maps ~1.0**
(kernel time + launch gap). The old fold sat at the 0.6 end AND moved work
into a heavier body, so it went negative. A body-free admission through the
generic primitive is the removal end: two ordinary programs per chain become
one fused body, and no surviving kernel changes.

| basis | census us | mapping | wall us | new ms/token | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline (production HEAD) | - | - | - | 5.2031 | 192.19 |
| M1 chain census (fresh trace) | 229.5 | 1:1 (removal) | 229.5 | 4.9736 | **~201.1** |
| M1 chain census | 229.5 | 0.6 (body-adding floor) | 137.7 | 5.0654 | ~197.4 |
| ladder realistic step | - | - | ~22 | 5.1811 | **~193.0** |

The ladder needs only **-22 us wall** for 193 (5.2031 -> 5.1811 ms). The chain
census is 229.5 us, so the row has large headroom: even a modest conversion
clears the +50 us promotion bar, and at 1:1 census recovery the row lands at
**~201 tok/s**. The realistic ~22 us ladder figure reflects that the 36 small
kernels are launch-bound and partially hidden (queue depth, 1.5-2.0 us
`E_32_32_4` floor), so not all census converts; the +50 us promotion bar sits
between the realistic floor and the 1:1 ceiling. Per-chain, the fused body
closes 6.38 us -> ~2-3 us (llama floor 2.12 us/node): the eligible removal per
chain is the pair sum minus the fused body's own cost.

### Why the primitive construction is body-free where the fold was not

| property | dead fold (NO-GO) | generic primitive admission (this scope) |
| --- | --- | --- |
| epilogue location | inside the w1w3fused GEMV body (R=2) | inside the fused `reduce_output_rmsnorm_1_4096` body (R=1, once per chain) |
| surviving kernels changed | GEMV bodies rewritten, `fused16 -> rms_affine16` 1:1 | w1w3fused byte-identical, consumes the same fp16 ABI |
| `r_16_256` | retained 37 == 37 (bitwise contract) | removed 36 -> 0 with the epilogue, one body per chain |
| census direction | moved work (net program delta only the epilogue drop) | pure removal, 2 -> 1 per chain |
| mapping end | 0.6 body-adding | ~1.0 removal |

## Implement plan

Standing order: admission probe (CPU-first) -> hermetic test (CPU) ->
single-layer real-token A/B at d512 -> full A/B (exact-token sha + census +
reverse wall bracket, +50 us bar).

1. **Admission probe on the ffn-norm chains through the generic primitive
   (CPU-first).** Production decode DAG census with the generalized admission
   using the producer-STORE / declared-typed-output spelling (M2a/M4/M5
   machinery: `_DECLARED_TYPED_OUTPUTS`, `_validated_typed_view`,
   `epilogue_absorption_admitted`-style ownership proof) so each ffn-norm chain
   renders ONE `reduce_output_rmsnorm_1_4096` body replacing `r_16_256` +
   `E_32_32_4_f14a5cc0`, with the norm weight read from the loader-owned fp16
   buffer and NO per-call weight materialization. Success contract: 36 chains
   admitted, `r_16_256` 36 -> 0, `E_32_32_4_f14a5cc0` 36 -> 0, fused bodies
   0 -> 36, net -36, every other program count unchanged; each rejection (if
   any) carries its distinct trace reason. This is the gate that proves the
   body-free claim the 08-09 census explicitly left open.
2. **Hermetic test (DEV=CPU).** Extend `test/unit/test_generic_reduce_output.py`
   with the ffn-chain fixture: a realized fp16 `(1,4096)` RMSNorm (the decode
   ffn epilogue recipe) lowers to ONE `reduce_output_rmsnorm_1_4096` CALL,
   bitwise equal to the ordinary `r_16_256` + `E_32_32_4_f14a5cc0` pair, with
   the expected REDUCE range + one barrier + LOOP restore and no weight-store
   program; lazy `x+x` and PERMUTE/SHRINK/EXPAND/unproven-PARAM inputs fail
   closed with the existing distinct trace reasons; the 08-09 legacy body
   digest (`c82e25f5...`) stays pinned. No GPU, no lock.
3. **Single-layer real-token A/B at d512 (NV, under lock, after the CPU gates
   pass).** One block's ffn-norm chain through the fused body vs the M2d
   candidate control: exact full-logit fp32 SHA-256 identical, token stream
   identical, per-row argmax == sampled token; census confined to that layer
   (chain 2 -> 1, no unrelated shift).
4. **Full A/B.** Reverse control/candidate/control wall bracket under
   `flock -w 60 /tmp/gpu-bench.lock` with settled-continuous windows, fresh
   processes per arm, token-stream SHA `9e6664fd...` identical across all
   arms; census with expected drops derived from the freshly measured control
   arm (`r_16_256` 36 -> 0, `E_32_32_4_f14a5cc0` 36 -> 0, fused 0 -> 36, all
   M2 families byte-identical, honest net -36); **+50 us/token promotion bar
   vs BOTH bracketing controls**; cost-prediction reconciliation with a
   body-free R=1 contract (CONFIRMED / EXPLAINED with named residual causes;
   CONTRADICTED fails closed).

## Gates

- **HARD STOP: do not re-run the dead fold.** The
  `q4k_gate_up_rms_affine_qualification_call` / `_rms_affine_gateup_norm_weight`
  construction is closed: NO-GO +81.92 us/token, cost gate CONTRADICTED. No
  re-bracket, no re-litigation, no lease variant of it. The reopen exists ONLY
  as a body-free admission through the generic reduce-output primitive.
- **The primitive admission must be demonstrated CPU-first.** Hermetic gate
  green on `DEV=CPU` and the production decode census showing body-free
  program removal (net -36 with no weight materializations and no w1w3fused
  body change) must pass BEFORE any NV arm, `/tmp/gpu-bench.lock`, or wall
  bracket runs.
- CPU phase constraints: no GPU command, no lock, no `python3 sz.py`, no policy
  promotion, no model wiring change; `decode-reduce-output-rmsnorm-route-policy.json`
  stays `promoted_targets: []`; scratch in `/tmp` only.
- Exact-output contract is unchanged and never weakened: full fp32 logits
  SHA-256 over the stacked rows identical to control, identical token stream
  (`9e6664fd...`), per-row argmax equals the sampled token, no stale-return
  binding. Logits gate precedes census and bracket.
- Census gate fails closed on any unrelated program-count shift; expected
  drops derive from the freshly measured control arm (never a stale constant -
  note the historical M1 harness counted 37 `E_32_32_4_f14a5cc0`; the fresh
  window counts 36 chains, so the control arm is the authority).
- Promotion requires +50 us/token vs BOTH bracketing controls with identical
  token hashes; the cost-prediction reconciliation must CONFIRM or EXPLAIN the
  measured delta (body-free R=1 contract), never CONTRADICT.

## Evidence

- `docs/task_workflow/input/nv-decode-gap-attribution-same-session-20260812.md`
  (fresh same-session numbers; M1 row = 36 chains, 229.5 us census; ladder:
  ~193 needs -22 us wall; ~201 at 1:1 census recovery)
- `docs/task_workflow/evidence/nv-tinygrad-prime-gap-table-20260812.json`
  (582-kernel prime table; classes `r_reduce` / `e_elementwise`)
- `/tmp/tg_debug_probe_20260812.log` (raw prime trace; this scope's chain
  extraction: 36 `r_16_256` 146.3 us + 36 `E_32_32_4_f14a5cc0` 83.2 us =
  229.5 us, mean 6.38 / median 6.21 us per chain)
- `docs/task_workflow/evidence/nv-llama-d512-node-ledger-20260812.json`
  (llama `rms_norm`: 145 nodes, 307.619 us node-sum, 2.12 us/node)
- `docs/task_workflow/input/nv-epilogue-absorption-route-scope-20260810.md`
  (M1 ledger line; cost-model contract; census-to-wall 0.61/1.0 mapping)
- `docs/task_workflow/input/nv-generic-reduce-output-primitive-scope-20260809.md`
  and `nv-generic-reduce-output-primitive-record-20260809.md` (C1 scaffolding,
  CPU gate PASS, per-association admission table)
- `docs/task_workflow/output/nv-generic-reduce-output-census-20260809.json`
  (108 admissions / 54 fused bodies; honest net +72 note; weight
  materialization caveat)
- `docs/task_workflow/input/nv-fusion-norms-ab-record-20260806.md`,
  `decode-norm-fusion-paths-forward-20260802.md`,
  `m3-fused-norm-measurement-record-20260802.md` (prior norm fusion attempts)
- `docs/task_workflow/input/nv-quant-gemv-llama-audit-20260812.md` (M1 NO-GO
  +81.92 us, cost gate CONTRADICTED; per-shape deficit table)
- `extra/llm_research/decode/nv_epilogue_absorption_m1_ab.py` (dead-fold
  harness: COST_PREDICTION, census swap rules)
- `extra/llm_research/decode/nv_fusion_cost_model.py` (predicted-wall-delta
  model, CONFIRMED/EXPLAINED/CONTRADICTED reconciliation)
- `docs/task_workflow/input/nv-reduce-output-rmsnorm-microgate-record-20260805.md`
  (fused body -1.253 us vs the ordinary pair in isolation)
