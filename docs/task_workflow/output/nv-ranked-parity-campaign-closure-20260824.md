# NV ranked parity campaign closure

Date: 2026-08-24

## Outcome

Every action in the ranked post-ledger queue was adjudicated with causal gates.
Three newly accepted optimizations were promoted, documented, tested, and
committed. The fresh installed endpoint is **4355.023 us/token = 229.620
tok/s**, above 227 but still below llama's retained **247.016 tok/s**.

| action | complete-information result | decision | booked |
| --- | --- | --- | ---: |
| host/outside union | same-token marker leaves 4.576 us after marker cost | accounting artifact, close | 0 |
| gate/up | old wall miss was an opaque-output boundary; typed route passes | promote | 53.329 us |
| down Q6 | old no-go disabled composed graph; current wall passes | promote | 10.607 us |
| flash | single-stage exact but +55.036 us/included graph; wider combine wall-negative | close | 0 |
| vocab | exact-order shared staging +183.530 us; global already ~1.542 TB/s | close | 0 |
| K/V | mixed pair has +3.126 us full-wall midpoint and complete topology | promote | 3.126 us |
| Q/O | exact predecessor closure; no positive full-wall candidate for remainder | close tested axes | 0 |
| norm/quant | semantic norm+RoPE already promoted; remaining rows offset at chain level | retain | 0 new |

## Why prior failures were revisited

An expected-pass failure was treated as unresolved whenever the test lacked
current topology, typed output ownership, cold production behavior, or full
wall accounting. This changed two decisions:

1. Gate/up's candidate created an uncounted output materialization because it
   lacked the declared typed-output contract. Restoring that contract removed
   the extra nodes and produced a 53.329 us wall win.
2. Q6 packed-lane's old wall harness forced direct greedy and feedback
   ping-pong off. On the current composed path it passed at +10.607 us in
   reps=9 and +30.246 us in reps=15.

The same rule closed the apparent host opportunity: once measured in one token
domain, the `82.557 us` cross-tool residual collapsed to `4.576 us` after
instrumentation cost.

## Promotions

| commit | promotion | rollback |
| --- | --- | --- |
| `f1a4d8dbc` | typed four-warp vector gate/up | `TINYGRAD_Q4K_GATE_UP_FOUR_WARP_DISABLE=1` |
| `01f86fc65` | Q6 down packed lane map | `TINYGRAD_Q6K_FFN_DOWN_PACKED_LANEMAP_DISABLE=1` |
| `7a18a43ff` | shared-Q8 mixed Q4/Q6 K/V pair | `TINYGRAD_SHARED_Q8_Q4Q6_KV_PAIR_DISABLE=1` |

The mixed pair uses the updated policy requested during the campaign: a
bit-exact candidate with complete topology accounting and a positive full-wall
midpoint is promoted even if control drift prevents it from beating each
individual control.

## Final position

```text
fresh tinygrad wall       4355.023 us/token = 229.620 tok/s
retained llama wall       4048.325 us/token = 247.016 tok/s
remaining wall gap         306.699 us/token = 17.396 tok/s
tinygrad device union     4187.750 us/token
llama device union        3888.240 us/token
device-union gap           299.510 us/token
conservative booked wall  4448.334 us/token = 224.803 tok/s
```

The 227 milestone is achieved by the fresh installed authority. Reaching 240
still requires `188.357 us/token`; llama parity requires `306.699 us/token`.
The next campaign should start from the rebuilt role table, not from the old
82.557 us host residual or previously closed body spellings.

Evidence and the machine-readable final ledger are under
`docs/task_workflow/evidence/nv-ranked-parity-campaign-20260824/`.

Verdict: `RANKED_QUEUE_COMPLETE_FRESH_227_PASSED_LLAMA_PARITY_OPEN`.
