# NV non-weight exact ledger

## Outcome

The non-weight audit does not expose another material unclosed bit-exact pool.
No token-wall recovery is booked.

The latest installed profile contains approximately `0.83 ms/token` outside
the weight-consuming bodies.  That number is real device work, but it is not
equivalent to removable headroom:

| bucket | device mass | current classification |
| --- | ---: | --- |
| flash score | 234.048 us | compulsory body; tested cache/readiness/topology constructions closed |
| flash combine | 99.872 us | wider exact reduction failed token wall |
| norms, RoPE and quant provider | 474.112 us | coupled accounting; mostly compulsory, provider advantage must be retained |
| native argmax | 8.512 us | already promoted near-floor replacement |
| miscellaneous | 11.776 us | small gross pool before replacement cost |

The norm row is deliberately not presented as 474 us of opportunity.  It
includes Q/K RMSNorm+RoPE work and the fused Q8 provider responsible for a
large activation-quant advantage.  Prior decomposition put the legal coupled
norm/provider comparison near parity.

## New causal test

The largest remaining exact body hypothesis was the 19-call
`reduce_output_rmsnorm_1_4096` population.  Its installed spelling reproduces
the ordinary 16-partial reduction tree by redundantly evaluating each serial
partial across lanes.  The discriminator computed each of those exact
partials once, retained their left-to-right combination, scale rounding, and
512-thread affine epilogue.

It was elementwise bit-exact on deterministic nonzero fp16 inputs, but direct
NV execution was slower:

```text
control A   5.408 us/call
candidate   7.936 us/call
control C   5.408 us/call
delta      +2.528 us/call
```

The construction therefore stops at the primitive gate.  No model admission
or token-wall test is warranted, and the rejected code was removed.

## New frame

The exact token path now has two measured walls:

1. Weight bodies are near a size-aware streaming/ramp wall.
2. Non-weight mass is predominantly compulsory or already closed by complete
   causal tests.

This does not claim an immutable hardware floor.  It means the next large
campaign should not budget all weight-rate or non-weight time as recoverable.
The largest unclosed lever is numerical weight-byte reduction under an
explicit quality contract.  Exact work should reopen only when a construction
removes a physical stream, bytes, or a previously unaccounted boundary.

Decision: `NO_MATERIAL_UNCLOSED_EXACT_NON_WEIGHT_POOL`.

