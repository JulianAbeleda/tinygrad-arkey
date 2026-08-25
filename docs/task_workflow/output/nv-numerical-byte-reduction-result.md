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

Evidence:

- `docs/task_workflow/evidence/nv-numerical-byte-reduction/q6-v-feasibility.json`
- `docs/task_workflow/evidence/nv-numerical-byte-reduction/q6-down-feasibility.json`
- `docs/task_workflow/evidence/nv-numerical-byte-reduction/q6-q5-feasibility.json`
- `docs/task_workflow/evidence/nv-numerical-byte-reduction/q5-direct-microgate.json`
- `docs/task_workflow/evidence/nv-numerical-byte-reduction/q5-logits-control.json`
- `docs/task_workflow/evidence/nv-numerical-byte-reduction/q5-logits-block0.json`
- `docs/task_workflow/evidence/nv-numerical-byte-reduction/q5-logits-block0-comparison.json`
