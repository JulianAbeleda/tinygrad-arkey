# Decode Route Decision Closeout

Date: 2026-06-20

## Verdict

`PASS_DECODE_ROUTE_DECISION_CLOSEOUT`

This pass corrects the DPT-1 next-action after accounting for the already-completed P7/P8 artifacts. The graph-safe Q4
source-import route was already attempted and measured. It should **not** be continued as a speed path.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_route_decision_closeout.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_route_decision_closeout_result.json
```

## Decisions

| route | decision | evidence |
|---|---|---|
| current default decode | keep promoted default | banked W==D authority |
| imported llama Q4 MMVQ graph route | **closed as speed route** | P7d `attn_output` speedup `0.763x`; P7e `gate/up` speedup `0.744x` |
| fused q8/MMVQ artifact | keep hardened opt-in | P8 artifact local speedup `1.46x`; Q8P gates passed; default-off policy |
| native tinygrad MMVQ renderer | project-level blocked | readiness `ROADMAP_ONLY`; no `>=30us` attributed feature |

## Consequence

The remaining decode parity work is no longer "make imported Q4 graph-safe." That route is correct and useful as an
oracle, but loses after lifecycle costs.

The coherent choices now are:

1. keep current default decode;
2. use `Q8_FFN_HANDWRITTEN=1` as a hardened opt-in oracle/research route;
3. scope a broad native MMVQ renderer/scheduler transfer against the q8 artifact oracle.

No BEAM/search yet. Search is still premature until native decode lowering exists.
