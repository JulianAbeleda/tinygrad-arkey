# Decode Owned q8 Lifecycle Successor Scope - 2026-06-20

Verdict: `PASS_DECODE_OWNED_Q8_LIFECYCLE_SUCCESSOR_SCOPE_READY`

This scopes the route-level decode path that can move while ATT is blocked. It does not promote anything and does not
change defaults.

The current q8 artifact is a hardened opt-in route. The owned successor is the tinygrad-owned version of that lifecycle:
one q8 activation producer/cache reused by the `ffn_gate` and `ffn_up` Q4_K consumers, with the same quality, fallback,
coverage, and W==D gates.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_owned_q8_lifecycle_successor_scope.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_owned_q8_lifecycle_successor_scope_result.json
```

## Object

`OwnedQ8LifecycleSuccessor` owns:

| component | contract |
|---|---|
| producer | q8 activation producer/cache from post-norm decode activation |
| reuse | reuse count `2`, shared across `ffn_gate` and `ffn_up` |
| consumers | two Q4_K packed q4/q8 dot consumers |
| fallback | existing default tinygrad decode |
| initial policy | default off |
| supported target | Qwen3-8B Q4_K_M-style dense FFN, `4096 -> 12288`, gfx1100 |

## Parity Targets

The successor must target the measured q8 artifact, not the slower native local schedule rows:

| target | value |
|---|---:|
| artifact lifecycle | `115.24us` |
| modeled oracle lifecycle | `107.642us` |
| artifact speedup vs P7e baseline | `1.463x` |
| W==D min speedup | `1.051x` |
| W==D median speedup | `1.060x` |
| quality max dNLL | `0.002225` |
| quality threshold | `0.01` |

## Phases

| phase | exit gate |
|---|---|
| OQ8-1 object contract | structural probe instantiates the successor from existing artifacts |
| OQ8-2 artifact parity harness | one matrix names baseline, artifact, and successor target rows |
| OQ8-3 owned producer candidate | q8 bytes/scale semantics and dNLL gate match artifact policy |
| OQ8-4 owned consumer candidate | correctness and lifecycle `<= artifact target` before W==D |
| OQ8-5 promotion decision | quality, fallback, coverage, W==D timing, ownership, and attribution reconciled |

## Boundaries

- do not default-on the q8 artifact through this scope;
- do not resume local native MMVQ schedule edits without ATT or a route-level parity gate;
- do not start BEAM/search until the owned successor has a lowerable candidate and measured objective;
- do not treat the external artifact as owned implementation.

Next executable probe:

```text
extra/qk_decode_owned_q8_lifecycle_successor_object_probe.py
```
