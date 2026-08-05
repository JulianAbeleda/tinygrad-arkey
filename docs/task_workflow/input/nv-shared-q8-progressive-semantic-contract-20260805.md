# NV shared-Q8 progressive semantic contract (2026-08-05)

Status: measured and closed **WALL NO-GO**. Closed lease only; no default or promotion; zero parity-ledger credit.

## Real-model progression

The fused RMSNorm-to-Q8 qualification leases blocks `1..N` for `N = 1, 2, 4, 18, 35`. Block 0 remains ordinary because its attention input originates at the embedding/gather boundary and the observed marker provenance is `CAST`, not the exact `REDUCE_OUTPUT` RMSNorm source required by the fused provider. The 35-block arm therefore covers every block whose source satisfies the typed fused-provider contract.

This progression includes both real GGUF topologies, Q4/Q4/Q6 and Q4/Q4/Q4. Each leased block must produce exactly one fused provider call, and all three consumers must use that provider.

## Two-level correctness contract

Primitive correctness does not become approximate. Before any model comparison, the checked-in fused-provider oracle must contain at least three cases, report `PASS`, and show bitwise equality with zero packet mismatches in every case.

The model route intentionally adopts llama-compatible Q8_1 activations, so comparison with tinygrad's direct floating-point consumer route is a semantic-change gate rather than a bitwise-equivalence gate. It is predeclared as:

- every returned logit is finite;
- sampled tokens equal the control and every sample equals its own returned-logit argmax;
- candidate and control argmax agree at every measured position;
- the unordered top-10 set is stable; ordered top-10 equality is additionally reported;
- aggregate relative L2 is at most `1e-3`;
- `2 * max_abs_delta / minimum_control_top1_margin < 1.0`, a conservative sufficient bound against an argmax inversion under the observed L-infinity perturbation.

The historical `max_abs <= 0.01` result remains present under `historical_max_abs_gate`, including its PASS/FAIL value, but is explicitly non-authoritative for this intentional Q8 semantic change. It is not deleted, renamed, or silently weakened.

## Measurement closeout

The predeclared progressive gate was executed on native NV. The checked-in primitive oracle passed bitwise first. Model results were:

| Fused blocks | Relative L2 | Max abs | Top-10 ordered | Tokens / argmax | Semantic verdict |
|---:|---:|---:|---|---|---|
| 1 | 0.000556768345 | 0.013356686 | exact | exact | PASS |
| 2 | 0.000485254505 | 0.009447813 | exact | exact | PASS |
| 4 | 0.000763045013 | 0.017748833 | exact | exact | PASS |
| 18 | 0.001067328993 | 0.019977570 | exact | exact | **FAIL** (`> 0.001`) |

Every measured arm was finite, retained exact provider counts, preserved the top-10 ordering, and stayed below the perturbation/minimum-margin bound. The 18-block arm nevertheless failed the independently predeclared relative-L2 threshold. Per progressive hard-stop discipline, the 35-block arm was not run. None of the measured arms reduced whole-token program count.

The largest passing arm, fused-g4, then received included-cost reverse-bracket wall measurement:

- control A: `5.5653619 ms/token`;
- control B: `5.5579222 ms/token`;
- bracketed control: `5.56164205 ms/token`;
- candidate: `5.58010875 ms/token`;
- candidate minus control: `+0.01846670 ms/token` (`+18.4667 us`, `-0.33094%` speedup).

Every timing replicate produced token hash `55f7a13b620816570f19e2f5786d86743b141cc2c72cdcef8cb5019af59f3aec`. The candidate is therefore correct under the g4 semantic contract but slower at included-cost wall.

Raw local evidence (not committed because it contains large per-token graph/logit payloads):

- `/tmp/nv-shared-q8-fused-progressive-20260805.json`, SHA256 `03a230d07df89ee46fd365fab4441a6a4cd4ef31be50624270a85fe91a2d59d2`;
- `/tmp/nv-shared-q8-fused-g4-timing-20260805.json`, SHA256 `b1f48fd28f160e6bf8ccf6bbe2ecb4d738e6f35cd4306b63f1edd3cf86ddca21`.

Verdict: **WALL NO-GO, zero ledger credit.** The mixed-format ABI and exact fused-provider substrate remain useful closed research infrastructure, but neither counts nor semantic admissibility substitute for the measured wall regression. Reopen only with a construction that removes additional token work or materially accelerates the quantized consumers, followed by the same semantic and included-cost gates.

## Reconciliation with the synthetic `-37.498 us/group` microgate

The earlier mixed Q4/Q4/Q6 microgate is not contradictory evidence and does
not reopen this wall verdict. Its `included-cost` label applies only to the
synthetic projection island that it timed; it is not an included-cost
measurement of the construction installed by the fused model lease.

The two candidates differ in load-bearing ways:

- The synthetic gate calls a Tensor-built `q8(x)` and passes separate packed
  int8 data and fp32 scale buffers to three `(weight, xp, xs)` consumers. The
  model route instead emits `rmsnorm_q8_1_llama_provider_4096`, stores llama's
  single packed uint32 Q8_1 ABI with fp16 metadata, and calls `(weight, xp)`
  consumers. They are different provider and consumer programs, not the same
  kernels embedded in a larger graph.
- The synthetic gate starts from an already-normalized random activation and
  reduces all Q/K/V outputs to scalar timing sinks. It therefore does not
  reproduce the real RMSNorm producer boundary or preserve Q/K/V through the
  downstream rope, KV-store, and attention consumers.
- Its measured topology is Q4/Q4/Q6 only. Blocks `1..4` in the real `fused-g4`
  arm contain three Q4/Q4/Q6 groups and one Q4/Q4/Q4 group. No Q4/Q4/Q4
  timing result was recorded by that microgate.

The real graph census also removes the structural premise behind multiplying
`37.498 us` by the number of leased blocks. Control and `fused-g4` both contain
947 programs, have the identical graph partition `[32, 64, 128, 256, 467]`,
and have the same five batch boundaries. Across blocks `1..4`, the candidate
removes four ordinary RMSNorm elementwise programs and four RMSNorm reductions,
four Q4 4096-row consumers, five Q4 1024-row consumers, three Q6 partials, and
three Q6 reductions. It adds four fused providers, the matching 4/5/3 packed-Q8
consumers, and four replacement generic boundary programs. Thus the provider
and boundary work exactly consume the launch-count reduction suggested by the
synthetic island. On the currently serialized native graph there is no changed
batch partition or overlap effect that can explain away the wall result.

The cheapest decisive reopen is a two-topology **exact production-island**
reverse bracket, not a rerun of the synthetic gate:

1. For Q4/Q4/Q6 and Q4/Q4/Q4 separately, compare ordinary RMSNorm plus the
   installed Q/K/V programs against the current fused RMSNorm-Q8 provider plus
   the current packed-ABI consumers.
2. Preserve materialized Q, K, and V outputs in both arms; do not replace them
   with scalar `.sum()` sinks.
3. Profile and sum the named changed programs in both the isolated island and
   the full token. If the exact island regresses, the construction itself is
   closed. If it wins by the amount required to cover the full-token result
   while the full token still regresses, only then is a downstream
   boundary/allocator interaction established.

Until that exact-production test passes, the synthetic result remains
direction-only evidence and the measured `+18.4667 us/token` wall regression
remains authoritative.

## Reproduction commands

Progressive semantic qualification:

```sh
DEV=NV PYTHONPATH=. python3 extra/llm_research/decode/nv_shared_q8_progressive_qualification.py \
  --mode fused-qualify --out docs/task_workflow/output/nv-shared-q8-fused-progressive-20260805.json
```

Included-cost reverse-bracket timing of the largest passing arm:

```sh
DEV=NV PYTHONPATH=. python3 extra/llm_research/decode/nv_shared_q8_progressive_qualification.py \
  --mode fused-timing --fused-groups 4 --count 16 --reps 3 \
  --out docs/task_workflow/output/nv-shared-q8-fused-g4-timing-20260805.json
```

The timing parent launches fresh control/candidate/control children under the GPU flock, synchronizes around steady decode, includes provider and consumer work in token wall time, checks token hashes, and reports candidate-minus-bracketed-control wall. Load-time resident norm-weight construction is excluded symmetrically from token time.
