# NV Q/O/K/V exact service-rate causal census

## Result

The current Q4 projection rate deficit is real, size-dependent, and not an
unaccounted byte reduction opportunity.  Fresh cache-flushed counters at the
installed vector spelling measured:

| shape | DRAM bytes | cold duration | effective DRAM rate | size-aware model |
| --- | ---: | ---: | ---: | ---: |
| Q/O 4096x4096 | 9,452,800 | 9.376 us | 1.008 TB/s | about 8.67 us |
| K/V 1024x4096 | 2,374,912 | 4.608 us | 0.515 TB/s | about 4.63 us |

K/V lands essentially on the previously fitted `3.27 us + bytes/1.75 TB/s`
ramp.  Q/O is roughly 0.7 us above it, so only Q/O advanced to the causal stall
gate.

The installed Q/O vector body uses 43 registers/thread, executes 4,784,128
instructions, and spends 70.85% of active-warp issue time stalled on the long
scoreboard.  MIO throttle is only 0.12%, math-pipe throttle 3.38%, short
scoreboard 1.32%, and wait 3.65%.  This identifies memory dependency latency,
not arithmetic pipe saturation, as the remaining local mechanism.

## Translation audit

The causal mechanism does not create a new admissible construction by itself.
Every currently supported exact way to expose more independent memory work has
already passed through a complete gate:

| construction | complete result |
| --- | --- |
| vector header/data loads | installed; fewer instructions and higher cold rate |
| two-block software unroll | bit-exact, slower hot and cold |
| four-block software unroll | bit-exact, small cold win, lost 8.646 us/token at the wall |
| 2/4/8 independent rows per CTA | bit-exact, all slower in the primitive gate |
| four-warps-per-row / Q8 provider | Q/O constructions previously wall-negative or numerically inadmissible |
| Q+K/V full-grid aggregation | shared subset booked; ordinary and broader output contracts wall-negative |

The high scoreboard percentage therefore cannot be multiplied by body time as
recoverable headroom.  It is the symptom of streamed weights on a short body;
the tested exact latency-hiding spellings either reduce occupancy, add
coordination, or fail to compose into the token wall.

No code candidate advances and no token-wall recovery is booked.

## Updated ranking

1. A genuinely new producer-consumer ownership construction that removes a
   complete physical stream/ramp, excluding the already tested QKV, queue, and
   PDL forms.
2. Training/calibration-aware byte reduction with a new model artifact and the
   recurrent quality authority.  This remains the largest numerical ceiling.
3. Vocab-projection stream-rate work only if it changes the full physical
   stream, not its already-promoted argmax tail.
4. Reopen Q/O scheduling only with a distinct latency-hiding mechanism not
   equivalent to unrolling, multi-row CTAs, four-warp ownership, or QKV
   aggregation.

Decision:
`RATE_DEFICIT_AND_LONG_SCOREBOARD_CONFIRMED__KNOWN_EXACT_TRANSLATIONS_CLOSED`.

