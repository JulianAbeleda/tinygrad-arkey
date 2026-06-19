# Primitive-local observability summary

- generated: `2026-06-19T07:31:19+00:00`
- commit: `5ae877964`
- observations: `11`
- validation: `PASS`
- search sessions: `3`
- runner smoke: `PASS`

## Verdict Ledger

| primitive | phase | role | verdict | bottleneck | next action | evidence |
|---|---|---|---|---|---|---:|
| `attention_kv` | `prefill` | `flash_prefill_reuse_free` | `REFUTED` | `bandwidth` | `do_not_reopen_without_new_evidence` | `1` |
| `mmvq_decode` | `decode` | `q8_sidechannel_ffn_gate_up` | `DEFERRED` | `pack_lifecycle` | `wait_for_named_capability` | `2` |
| `prefill_tensile` | `prefill` | `attn_q_o` | `PASS` | `occupancy_or_issue` | `eligible_for_next_gate` | `3` |
| `prefill_tensile` | `prefill` | `ffn_down` | `PASS` | `occupancy_or_issue` | `eligible_for_next_gate` | `3` |
| `prefill_tensile` | `prefill` | `ffn_gate_up` | `PASS` | `unknown` | `eligible_for_next_gate` | `3` |
| `prefill_tensile` | `prefill` | `ffn_gate_up` | `PASS` | `occupancy_or_issue` | `eligible_for_next_gate` | `3` |
| `prefill_tensile` | `prefill` | `weighted_shape_matrix` | `PASS` | `unknown` | `route_only_after_graph_gate_and_policy` | `3` |
| `prefill_wmma` | `prefill` | `ffn_gate_up` | `KILL` | `occupancy_or_issue` | `do_not_reopen_without_new_evidence` | `2` |
| `runtime_boundary` | `graph_integration` | `ffn_block` | `REDIRECT` | `graph_boundary` | `graph_integration_next` | `3` |
| `runtime_boundary` | `graph_integration` | `tensile_rebindable_node` | `PASS` | `graph_boundary` | `eligible_for_next_gate` | `3` |
| `spec_decode` | `spec_verify` | `verify_forward` | `CLOSED` | `unknown` | `do_not_reopen_without_new_evidence` | `3` |

## Reconstructed Required States

- q8/MMVQ lifecycle: deferred behind codegen capability.
- pure-tinygrad WMMA bounded sweep: killed/refuted.
- Tensile extraction TPE-5: pass/generalizes.
- TPE-6 block transfer: redirect to graph integration.
- TPE-7a rebindable node: pass; in-model graph capture remains the next gate.
- spec decode shortcut: closed.

## Runner Registry

- `session:tpe5_shape_matrix_replay`: `prefill_tensile` via `extra/qk_tensile_shape_matrix.py`
- `session:tpe6_runtime_boundary_replay`: `runtime_boundary` via `extra/qk_tensile_block_transfer.py`
- `session:tpe7a_rebindable_node_replay`: `runtime_boundary` via `extra/qk_tensile_rebindable_node.py`

## Runner Smoke

- `session:tpe5_shape_matrix_replay`: `PASS` (2 replay artifacts)
- `session:tpe6_runtime_boundary_replay`: `PASS` (1 replay artifacts)
- `session:tpe7a_rebindable_node_replay`: `PASS` (1 replay artifacts)

## Trace / Counter Plugin Inventory

- mode: `inventory_only_no_trace_collection`
- rocprofv3: `/opt/rocm/bin/rocprofv3`
- rocprof-compute: `/opt/rocm/bin/rocprof-compute`
- tinygrad SQTT example files: `8`
- rocprof trace artifacts: `2`
- PMU probe: `REDIRECT_HCQ_NATIVE_ADAPTER`
- HCQ attribution: `rocprof_hcq_visibility_gap,graph_rebind_ok`

## Principle Check

- read-only over existing artifacts by default;
- no model route/default changes;
- correctness and device time remain decision authority;
- root-cause claims are evidence-level labeled;
- optional counters/traces are plugins, not blockers.
