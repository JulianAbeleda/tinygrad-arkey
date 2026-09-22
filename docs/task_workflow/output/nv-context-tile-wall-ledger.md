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
| tinygrad pre-boundary S=48 | bounded d512 pre-cliff authority | 234.66 | local S=48 winner; 94.8 us / 5.34 tok/s from 240 | 213.1 us / 12.36 tok/s behind the retained d512 llama authority |
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

## Adaptive dual-graph implementation gate

A closed-default dual-graph substrate now preserves the S=48 graph and owns a
separate S=64 capture selected from logical `start_pos`. The first crossing
run exposed an 11.7-second lazy-capture pause on the boundary token. Warming
the ordinary sampled graph did not fix the composed route because direct-
greedy ping-pong owns two distinct captures. Prewarming both actual feedback
graphs before prompt execution removed the pause completely.

| arm | median | boundary behavior | tokens |
| --- | ---: | --- | --- |
| installed S=48 control | 4.359039 ms/token | steps to about 4.405 ms after boundary | reference hash |
| prewarmed adaptive S48/S64 | 4.272964 ms/token | all eight windows 4.268--4.290 ms | exact same hash |

The mixed-band recovery is **86.074 us/token**, approximately **229.41 to
234.03 tok/s (+4.62 tok/s)**. The steady post-boundary recovery remains
137.482 us/token (+7.29 tok/s at the d800 bracket).

The substrate stays closed-default pending a target promotion record and an
explicit startup/graph-memory admission decision. The token path is complete;
there is no remaining correctness or boundary-capture wall.

Implementation decision:
`DUAL_GRAPH_TOKEN_PATH_PASS__PROMOTION_POLICY_AND_STARTUP_COST_PENDING`.

## Deployment-cost closure and single-graph route

The dual-graph path is not blocked by VRAM, but it is blocked by construction
latency.  A fresh-process phase census measured the normal S48 ping-pong pair
and then the additional S64 pair.  The S64 pair adds 165,428,760 allocator
bytes, but requires 185.93 seconds to trace/schedule/capture.  A second process
with a warm compiler cache still required 184.33 seconds, so this is repeated
full-model graph construction rather than a one-time binary-cache miss.

The cleaner deployment route is request-static geometry.  On the same composed
depth-704, 16-token, eight-window crossing workload, selecting S64 before the
first decode capture produced 4.268195 ms/token with the exact reference token
stream.  Against the 4.359039 ms/token S48 crossing control, this recovers
90.844 us/token, or approximately 229.41 to 234.29 tok/s (+4.88 tok/s).  It also
avoids both the second graph's 165 MB and its boundary construction pause.

This does not authorize global S64.  S64 remains measured slower below context
769.  The admissible production shape is therefore a request-level horizon
decision: select one graph before decode when the request is expected to run
far enough beyond the cliff to repay the pre-cliff loss.  With the measured
band costs, a request beginning at context 704 repays that loss after roughly
ten post-boundary tokens.  The current generator does not own an output-horizon
input, so automatic broad promotion remains closed pending that typed input and
a reverse bracket around its decision boundary.

The graph-local S64 lease is now independently admitted at the kernel guard;
it no longer relies on `FLASH_DECODE_COARSE_SPLIT=64` being present in the
process environment.  Ordinary calls remain S48 and arbitrary leased geometry
continues to fail closed.

Updated decision:
`DUAL_GRAPH_DEPLOYMENT_NO_GO__SINGLE_GRAPH_HORIZON_POLICY_PASS`.

## Bounded promotion

The request-horizon policy is now wired into `Transformer.generate` and the
HTTP completion path passes the client's explicit `max_tokens`.  Admission is
intentionally narrow: max context at most 1024, prompt length at least 704,
and a declared horizon reaching context 779.  This is the measured payback
boundary for the worst admitted start; missing horizons and all other requests
retain the existing S48 route.

The integrated composed gate used the typed horizon input with no
`FLASH_DECODE_COARSE_SPLIT` environment override and no adaptive dual-graph
lease.  Its eight settled windows measured 4.273808 ms/token with the exact
reference token stream.  Relative to the 4.359039 ms/token S48 crossing
control, the installed bounded policy recovers 85.231 us/token, approximately
229.41 to 233.98 tok/s (+4.57 tok/s).  This is booked for the qualified request
band only; the pre-boundary endpoint remains unchanged.

Promotion decision:
`SINGLE_GRAPH_HORIZON_POLICY_BOOKED_FOR_QUALIFIED_REQUESTS`.
