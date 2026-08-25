# NV numerical weight-byte reduction campaign

## Outcome

No token-wall recovery is booked. Post-hoc Q6_K reduction reaches a real byte
and kernel opportunity, but the representations tested do not satisfy the
predeclared model-quality contract.

| route | physical byte effect | primitive result | quality result | decision |
| --- | ---: | ---: | --- | --- |
| Q6_K to Q4_K, V | 31.43% fewer selected bytes | about 0.64--0.67 us faster per tested projection | local relative L2 up to 0.0821 | stop |
| Q6_K to Q4_K, down | 31.43% fewer selected bytes | about 29.25 us/token summed isolated opportunity | local relative L2 up to 0.1018 | stop |
| Q6_K to Q5_K, V+down | 16.19% fewer selected bytes | direct packed substrate passes; representative V is 0.32 us faster and down is materially faster | local cosine minimum 0.998982, just below 0.999 | promote to full-logit discriminator |
| Q6_K to Q5_K, one down block | 6,684,672 bytes removed | direct kernel agrees with independent Q5 dequantized matvec to about 5e-7 relative L2 | recurrent full-logit aggregate relative L2 0.002232 | quality no-go |

The single-block full-model arm remained finite and preserved all four greedy
token IDs. It passed the `1e-3` full-logit threshold on the first decode step
(`0.000696`) but accumulated through recurrent state to `0.003454` on step
four. The aggregate was `0.002232`. Because one block already fails the
contract, expanding the arm to all 18 Q6 down blocks or running a token-wall
bracket would spend measurement time on an inadmissible model.

## What this establishes

The byte lever itself is genuine. Q5 sidecars remove compulsory DRAM payload,
and the direct device substrate both computes the intended Q5 values and has
positive isolated service value. The wall is numerical propagation, not a
missing packed kernel and not merely the earlier local cosine cutoff.

It also rules out treating token agreement over a short trace as sufficient:
the token IDs can remain unchanged while the full distribution has already
crossed the quality budget.

## Updated wall ledger

| lever family | information status | booked recovery |
| --- | --- | ---: |
| exact weight service-rate tuning | size-aware stream/ramp wall; tested geometry closed | 0 |
| exact non-weight removal | largest remaining bodies are compulsory or causally closed | 0 |
| post-hoc Q6 to Q4 | byte value passes, local quality fails | 0 |
| post-hoc Q6 to Q5 down | byte and kernel substrate pass, recurrent full-logit quality fails | 0 |
| calibrated or training-aware lower-byte weights | untested; requires a new representation/model artifact and quality authority | unmeasured |
| removal of a complete physical stream/ramp | still valid in principle; obvious exact compositions tested so far are closed | unmeasured |

The installed endpoint is unchanged by this campaign. The next honest large
lever is no longer another packing spelling of the same post-hoc conversion.
It is either (a) a calibrated/training-aware mixed-precision artifact that
assigns the quality budget selectively, or (b) a new topology that eliminates
a physical stream/ramp while retaining the already measured service rate.
Both require a new causal construction before any tok/s credit.

Decision: `BYTE_AND_KERNEL_PASS__RECURRENT_QUALITY_WALL__NO_BOOKING`.

## Selective-precision follow-up

All 18 Q6-down blocks were subsequently tested as independent Q5 singletons
under the same four-step recurrent full-logit gate. All logits were finite and
all greedy tokens agreed, but **zero of 18** singletons satisfied `1e-3` on
both aggregate and every recurrent step. The best placement was block 2 at
aggregate `1.500e-3`; the population median was `2.868e-3`.

The reason is not ordinary local quantization error. Singleton full-logit error
is negatively correlated with both standalone projection error (`-0.521`) and
weight error (`-0.498`). Block 6 is the clearest counterexample: it has the
lowest local error but is the second-worst full-model placement. Downstream
amplification and cancellation determine admissibility.

A row-selective discriminator retained the largest standalone-error rows of
the best block in Q6. Keeping 25% and 50% improved aggregate error, but no arm
passed every recurrent step; keeping 75% became worse again. The response is
non-monotonic, proving that standalone row-error ranking is not a safe mixed-
precision policy. A reopen now requires end-to-end calibration sensitivity or
direct subset search. Uniform block selection and weight-error row selection
are closed.

Follow-up decision:
`NO_ADMISSIBLE_SINGLETON__WEIGHT_ERROR_ROW_SELECTOR_NON_MONOTONIC`.

### End-to-end sensitivity investment gate

Block 2 was divided into four equal physical output-row shards. Each shard's
signed recurrent full-logit delta was measured directly, rather than inferred
from weight error. Exhaustive additive search across all 15 nonempty shard
subsets predicted **zero** subsets that pass every recurrent step. The most
favorable predicted 50%-Q5 subset, shards 2+3, was then run directly. It saved
3,342,336 physical bytes but measured aggregate `1.109e-3` and maximum-step
`1.439e-3`, confirming the no-go.

This closes coarse end-to-end row-shard selection before production-kernel
investment. Finer learned/group calibration is not disproven, but its maximum
value is now paired with substantially greater representation and search cost;
it is not the next ranked engineering lever.

Investment decision:
`COARSE_END_TO_END_ROW_SENSITIVITY_NO_GO__NO_HYBRID_KERNEL_INVESTMENT`.

Evidence:

- `docs/task_workflow/evidence/nv-numerical-byte-reduction/q6-v-feasibility.json`
- `docs/task_workflow/evidence/nv-numerical-byte-reduction/q6-down-feasibility.json`
- `docs/task_workflow/evidence/nv-numerical-byte-reduction/q6-q5-feasibility.json`
- `docs/task_workflow/evidence/nv-numerical-byte-reduction/q5-direct-microgate.json`
- `docs/task_workflow/evidence/nv-numerical-byte-reduction/q5-logits-control.json`
- `docs/task_workflow/evidence/nv-numerical-byte-reduction/q5-logits-block0.json`
- `docs/task_workflow/evidence/nv-numerical-byte-reduction/q5-logits-block0-comparison.json`
- `docs/task_workflow/evidence/nv-numerical-byte-reduction/q5-sensitivity-census.json`
