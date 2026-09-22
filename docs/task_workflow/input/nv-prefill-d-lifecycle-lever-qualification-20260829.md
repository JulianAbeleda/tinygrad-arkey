# Q6-down lifecycle lever qualification

Date: 2026-08-29. Packet: `D-L0`.

## Question

Does allocation, host copy, graph copy, or buffer materialization recur inside
the timed Q6-down prefill route and therefore merit a full boundary matrix?

## Test

Run three fresh Q6-down residual arms in order: observer off, observer on,
observer off. All arms use PROFILE=1 so the only changed variables are the
Buffer callback and HCQ JSON export. Each arm runs one warmup and R9.

The observer arm must retain token 198, the exact census, nine HCQ submission
records, complete `program` versus `graph_copy` classification, and less than
2 percent wall perturbation relative to the two controls.

## Decision

- `STOP` if observer validity fails.
- `STOP` if the valid timed R9 contains zero allocation, copyin, copyout,
  graph-copy, and buffer-materialization events. Lifecycle cannot explain a
  recurring prefill wall when it does not recur inside the timed submissions.
- `PASS` only if at least one real lifecycle event recurs. PASS authorizes a
  new `D-L1` hot/cold boundary matrix; it does not authorize optimization.

Kernel arithmetic, queue policy, model composition, and promotion are frozen.
