# NV context-tile wall and corrected ledger

## Finding

The apparent 230-versus-235 endpoint regimes are primarily a deterministic
context-length step, not unexplained session variability. Production flash
decode uses 48 splits and 16-token tiles. Its aligned work per split is:

```text
aligned = ceil(ceil(context / 48) / 16) * 16
```

At context 769 this rises from 16 to 32, doubling the score tile-loop extent
for all 48 splits. A depth-512 continuous run showed seven windows at
4.224--4.233 ms/token, a straddling window, then seven windows at
4.379--4.386 ms/token. A shifted 16-token-window run reproduced the transition
at the same position. Per-window clock, temperature, and throttle observations
did not coincide with the step.

## Causal geometry test

S=64 retains one 16-token tile per split through context 1024. Above the
boundary it executes 64 aligned tile units rather than S=48's 96, while paying
for a wider combine.

| context band | geometry | wall | result |
| --- | --- | ---: | --- |
| below 769 | S=48 control midpoint | 4.230932 ms/token | reference |
| below 769 | S=64 | 4.251197 ms/token | loses 20.265 us/token |
| above 769 | S=48 control midpoint | 4.412206 ms/token | 226.644 tok/s |
| above 769 | S=64 | 4.274724 ms/token | 233.933 tok/s |

Above the boundary S=64 recovers **137.482 us/token**, or **7.289 tok/s**, with
identical token hashes. A global replacement is wrong because S=64 loses below
the boundary. The measured policy is adaptive: S=48 through context 768, S=64
from 769 through 1024. Integration still requires a second captured decode
graph or equivalent typed geometry selection; the static environment override
is research substrate, not a production admission.

## Corrected endpoint ledger

| row | measured position | tok/s | relative to wall | relative to retained llama |
| --- | --- | ---: | --- | --- |
| tinygrad pre-boundary S=48 | context below 769 | about 236.6 | local S=48 winner; 59.6 us from 240 at the representative median | about 178 us / 10.4 tok/s behind the retained d512 llama authority; position-window matching remains a caveat |
| tinygrad post-boundary S=48 | context above 769 | 226.6 at d800 bracket | avoidable alignment cliff, not a true hardware ceiling | unmeasured at matched d800 llama context |
| tinygrad post-boundary S=64 | context above 769 | 233.9 at d800 bracket | recovers 137.5 us of the cliff; next wall requires a fresh ledger after adaptive integration | unmeasured at matched d800 llama context |
| retained llama authority | historical d512 authority | 247.0 | comparison reference | reference |

The earlier single-number 233--235 endpoint description mixed different
context bands. Future ledgers must name the measured position interval and
flash geometry. No cross-context tok/s comparison may be booked as an
optimization.

## Closure ledger

| lever | status | true closure or conditional? | reopen trigger |
| --- | --- | --- | --- |
| Q5 whole-block and coarse row selection | quality no-go | conditional on post-hoc Q5 and current contract | calibrated/trained artifact or direct fine-group quality authority |
| generic exact weight scheduling | size/ramp wall | conditional on current packed formats and physical streams | fewer bytes or removal of a complete stream/ramp |
| extra queues | closed for current DAG | strong topology closure | changed DAG or complementary-resource work |
| tested QKV full-grid producers | wall-negative | construction closure | different producer-consumer ownership/critical path |
| S=48 flash above context 768 | alignment cliff | **not a ceiling** | adaptive split count; S=64 passes |
| S=64 below context 769 | wall-negative | context-conditional closure | do not use below boundary |
| adaptive S48/S64 policy | measured substrate pass | not installed | build second graph, exact semantic gate, crossing-boundary wall bracket |

Decision: `CONTEXT_ALIGNMENT_WALL_FOUND__ADAPTIVE_SPLIT_SUBSTRATE_PASS`.
