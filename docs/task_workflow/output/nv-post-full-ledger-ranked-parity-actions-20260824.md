# NV ranked actions after the full llama ledger audit

Date: 2026-08-24  
Starting point: **4503.391 us/token = 222.055 tok/s** fresh  
Conservative campaign point: **4515.396 us/token = 221.465 tok/s**  
Targets: **227**, then **240**, then fresh llama **247.061 tok/s**

## Promotion discipline

Each action follows the existing process:

1. state a causal mechanism and a maximum recoverable wall budget;
2. measure the current production-conditioned control, not a historical or
   scalar fallback;
3. require exact outputs and a positive isolated/counter gate where relevant;
4. run fresh A/B/A unprofiled wall brackets, normally `reps >= 7`;
5. book only the wall recovery that survives both controls;
6. rebuild the full ledger after every promotion.

Profile-only, hot-cache, cross-session, and non-additive overlap projections
remain unbooked.

## Ranked execution queue

### 1. Assign the 82.557 us host/outside-union term

This is the highest-confidence unassigned wall pool and nearly covers the
98.105 us needed for 227 by itself.  Capture matched CPU/API timestamps around
one settled graph replay together with GPU graph start/end and final sync.
Partition graph submission, driver launch, synchronization, token
materialization, and inter-token boundary time.

Maximum arithmetic ceiling if all were recoverable:

```text
4503.391 - 82.557 = 4420.834 us/token = 226.202 tok/s
```

The output of this step is a named implementation target, not a booked gain.

### 2. Reconcile current gate/up and down streaming-rate gaps

Together these pools occupy `2235.872 us/token`, more than half of tinygrad's
device union.  Retained role evidence assigns about `53.65 us` to gate/up and
`57.91 us` to down, but recent Q4/Q6 no-go tests prove that isolated gains can
vanish at the wall.  Re-measure current exact cubins in cold production order,
including bytes and achieved DRAM rate, before another implementation.

Combined retained ceiling, if causal and additive:

```text
111.56 us -> 4391.831 us/token -> 227.695 tok/s
```

Prefer a mechanism that reduces transferred bytes or demonstrably raises cold
DRAM rate.  Do not reopen four-warp gate/up or Q6 packed-lane spelling.

### 3. Isolate flash score-to-combine boundary cost

Current flash score plus combine costs `342.912 us/token`.  Score arithmetic
has already tested near/faster than llama in isolation, while retained role
residuals are `82.24 us` score and `66.69 us` combine.  Wider combine geometry
was wall-negative, so the next test must target the production boundary:
materialization, cache state, launch dependency, or a fused consumer contract.

Retained combined ceiling: **148.93 us/token**.  Require a causal timeline
change before building another kernel-body variant.

### 4. Re-audit the post-native-argmax vocab path

The installed vocab path is `321.440 us/token`: `312.576 us` main plus `8.864
us` native argmax.  Native argmax is already landed, so the historical `65.44
us` vocab residual must be refreshed before reuse.  Split main GEMV/reduction,
tail, and boundary cost against the same llama build.  Only pursue a new main
or tail algorithm if the refreshed body gap survives.

### 5. Refresh K/V after pair fusion and producer-sink landings

The current K/V projection pool is `201.056 us/token`, but the retained
`73.12 us` residual predates the latest ordinary/shared pair and producer-sink
changes.  It is therefore a remeasurement target, not a valid recovery
projection.  Compare the current paired call graph with llama before any more
fusion or queue work; prior broad concurrency and placement probes are closed.

### 6. Q/O production-boundary residuals

Current Q is `303.952 us/token`; O is `307.392 us/token`.  Retained residuals
are `51.42 us` and `48.86 us`.  Their isolated bodies are already close to or
faster than retained llama bodies, making handoff/cache/launch placement the
only justified reopen.  Take this after flash and K/V because the causal
boundary instrumentation from those actions can be reused.

### 7. Norm/accounting only as a fused-chain problem

Do not rank the apparent `+115.42 us` norm row alone.  Tinygrad simultaneously
wins `113.98 us` in activation quantization and fuses Q/K RoPE work into its
norm programs.  Treat the pair as approximately balanced until a matched
chain-level wall construction proves otherwise.

## Milestone composition, not a forecast

The following illustrates the scale required; it is not bookable because the
ceilings come from different measurement domains:

| cumulative hypothetical recovery | latency | throughput |
| --- | ---: | ---: |
| host/outside-union 82.557 us | 4420.834 us | 226.202 tok/s |
| plus gate/up 53.65 us | 4367.184 us | 228.981 tok/s |
| plus down 57.91 us | 4309.274 us | 232.058 tok/s |
| plus flash combine 66.69 us | 4242.584 us | 235.705 tok/s |
| plus refreshed vocab ceiling 65.44 us | 4177.144 us | 239.398 tok/s |

This shows why the order is useful: a validated host term plus one projection
win can cross 227; reaching almost 240 requires several independent wins.  To
match the fresh 247.061 tok/s llama result still requires another `128.819 us`
beyond that illustrative 239.398 tok/s stack.

## Immediate next action

Start with the matched host/API/GPU boundary trace.  It is the only open,
newly measured pool large enough to put 227 within one additional modest
device win, and its instrumentation also separates launch-boundary residuals
for flash, K/V, Q, and O.
