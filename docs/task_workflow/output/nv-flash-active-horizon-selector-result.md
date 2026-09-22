# Flash active-horizon selector result

## Decision

Do not promote the S6-through-768/S8-after selector.

The fixed S6 primitive win is real, and the production selector is
semantically correct, but it does not survive the complete token lifecycle.
The selector is now conversion-closed for the current dense endpoint.

## What was tested

The closed lease adds a distinct TinyJit identity for the bounded wide S6
graph and selects it for `512 <= start_pos < 768` (`Tc <= 768`). At
`start_pos == 768`, the same generation switches to the installed S8 graph.

The production-shaped authority starts with a 704-token prompt and consumes
nine continuous eight-token windows. It therefore observes S6 service,
crosses the Tc=768 boundary, and observes S8 service in one token stream. The
test uses fresh control, candidate, control processes and a separate graph
profile.

## Results

| gate | result |
|---|---|
| selector boundaries | pass |
| token stream through transition | exact; all three hashes equal |
| graph identity | pass; trace contains both S6 and S8 score programs |
| cold transition | fail; lazy S8 capture creates a 17.316 s/token eight-token window |
| cold steady median, excluding the capture outlier | candidate -7.618 us/token, +0.456 tok/s |
| dual-pair prewarm | pass; both S6 and S8 greedy ping-pong pairs captured before generation |
| prewarmed transition | pass; no boundary outlier |
| prewarmed reverse wall | fail; candidate +10.676 us/token, -0.634 tok/s versus control midpoint |

The cold robust median is not admissible evidence for promotion because its
outlier filter classifies the deterministic graph compilation stall as
external contention. The raw window shows that the stall is part of the
unwarmed production lifecycle.

Prewarming proves that this is a removable operational wall, but removing it
does not recover the proposed token win. With both graph pairs already
captured, the candidate loses to both flanking controls. The selector also
adds minutes of cold load-time compilation in this fresh-process authority.

## Accounting

The current installed endpoint remains 4.094502 ms/token, or 244.230 tok/s.
No recovery is booked. The earlier full-continuation estimate of roughly half
a token per second is superseded as an actionable claim by the measured
selector no-go.

This closes a specific translation failure:

```text
faster bounded S6 score primitive
  -> exact alternate graph
  -> exact S6/S8 token transition
  -> cold graph-capture wall
  -> prewarm removes wall
  -> complete prewarmed token path is slower
```

Future Flash work must expose a new construction. Repeating fixed-S6 timing or
adding more graph buckets is not sufficient. The remaining question is why
the equal-horizon tinygrad score service is more cold/working-set-sensitive
than llama, and whether that mechanism can be changed without another graph
identity or a load-time compilation tax.

## Evidence

- `docs/task_workflow/evidence/nv-flash-active-horizon-selector/selector-r9.json`
- `docs/task_workflow/evidence/nv-flash-active-horizon-selector/selector-r9/candidate-profile.profile.jsonl`
- `docs/task_workflow/evidence/nv-flash-active-horizon-selector/selector-prewarmed-r9.json`
- `extra/llm_research/decode/nv_flash_active_horizon_selector.py`
