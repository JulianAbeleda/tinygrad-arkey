# B0.3 gate/up isolated correctness and counters

Status: STOP.

The real `blk.0.ffn_gate.weight` fixture passed G0 for control and all three
default-off B0.2 variants. Outputs had identical hashes, all 6,291,456 values
were finite and written, and packed weights and compact-Q8 records remained
read-only.

Matched root NCU reports are retained in
`docs/task_workflow/evidence/nv-prefill-gateup-schedule-b03-20260829/`.
No variant demonstrated both required counter movements: tensor duty did not
show an eligible increase and long-scoreboard exposure did not fall above
counter noise. The fragment reorder's 346.720 us median versus 352.864 us
control is below the packet's 10 us service-noise threshold; metadata reorder
and double buffer had no service win. No variant is eligible for B0.4.

Evidence: `result.json`.
