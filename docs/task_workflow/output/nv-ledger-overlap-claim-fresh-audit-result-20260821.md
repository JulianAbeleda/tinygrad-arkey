# NV ledger overlap-claim fresh audit result

Date: 2026-08-21

Commit: `6570abc025514273faa100c66b979e531585a1e1`

Scope:
`docs/task_workflow/input/nv-ledger-overlap-claim-fresh-audit-scope-20260821.md`

Status: fresh measurement record. No production change, no promotion, no new
architecture claim. Every GPU arm ran as a fresh process under
`flock /tmp/gpu-bench.lock`.

## 1. Verdict

The overlap claim is **supported at current HEAD**, with one correction to
the wording used before this audit. Tinygrad is not *exactly* serial: its
steady decode token contains 7 short overlapping kernel pairs, or about
0.19% of kernel residence time. Llama contains 870 overlapping pairs, or
about 22.56% of residence time. The comparison is still decisive, so the
reference claim is now phrased as "near-serial versus llama-scale overlap,"
not "zero overlap."

All six belief-flip gates passed. The measured wall gap this session is
+703.452 us/token (smaller than the prior +717.505 us); the device-union gap
is +842.682 us; S1 is again the largest segment at +632.595 us; tinygrad's
anchor bodies are again slightly faster than llama's. The useful-body
portion of llama's overlap remains `unmeasured` because no per-kernel
wait-exit timestamps were captured in this packet.

## 2. Fresh locked numbers

| quantity | tinygrad us | llama us | delta us |
| --- | ---: | ---: | ---: |
| unprofiled wall | 4731.955 | 4028.503 | +703.452 |
| device union | 4733.250 | 3890.568 | +842.682 |
| node sum | 4742.464 | 5023.823 | -281.359 |
| overlap mass | 9.214 | 1133.255 | -1124.041 |
| host/launch residual | -1.295 | +137.935 | -139.230 |
| device span | 4843.000 | 3898.776 | +944.224 |

The identity closes with no residual:

```text
delta_union = delta_node_sum - delta_overlap
+842.682    = -281.359 - (-1124.041) = +842.682

wall gap = device gap + host residual
+703.452 = +842.682 - 139.230
```

The tinygrad overlap mass of 9.214 us is not profiler noise hidden by a
rounded ledger: the raw HCQ profile shows 7 overlapping pairs, with a
maximum pair overlap of 2.75 us and a minimum inter-kernel gap of -3.0 us.
Llama's raw CUPTI replay shows 870 overlapping pairs and a minimum gap of
-4.96 us. The share-of-node-sum comparison is 0.19% versus 22.56%.

## 3. Gate results

| gate | observable | verdict |
| --- | --- | --- |
| G1 llama overlap real | 870 overlapping pairs, negative minimum gap | supported |
| G2 tinygrad near-serial | 7 overlapping pairs, 9.214 us mass | supported |
| G3 identity closes | both identities above, residual 0.000 | supported |
| G4 location is S1 | S1 is the largest segment | supported |
| G5 anchor bodies | tinygrad 2947.750 vs llama 2997.927 | supported |
| G6 roofline | tinygrad moves more bytes at lower effective bandwidth | supported |

## 4. Where the gap lives

| segment | tinygrad us | llama us | delta us |
| --- | ---: | ---: | ---: |
| S0 | 180.000 | 31.042 | +148.958 |
| S1 | 1149.500 | 516.905 | +632.595 |
| S2 | 186.500 | 32.355 | +154.145 |
| S3 | 0.000 | 15.969 | -15.969 |
| S4 | 169.250 | 30.658 | +138.592 |

The S1 delta of +632.595 us is the largest segment and, combined with S0/S2/
S4, accounts for the exposed-support deficit. S3 remains a tinygrad win
because the GLU is already fused into the FFN-down epilogue.

Anchor versus support reconciliation:

| row | tinygrad us | llama us | delta us |
| --- | ---: | ---: | ---: |
| Q/O/gate-up/down anchor union | 2947.750 | 2997.927 | -50.177 |
| per-layer support exposed | 1415.750 | 591.583 | +824.167 |
| vocab tail | 369.750 | 303.746 | +66.004 |
| interval accounting residual | - | - | +2.688 |
| device union total | 4733.250 | 3890.568 | +842.682 |

The result is the same qualitative shape as the 20260820 ledger: the GEMV
anchors are not the deficit, the serial support chain is. The fresh numbers
are slightly different because this is an independent same-day capture and
unprofiled wall, not a reproduction of the prior session to the digit.

## 5. Roofline

| route | byte estimate | effective bandwidth |
| --- | ---: | ---: |
| tinygrad | ~5.04 GB/token | 1065.1 GB/s |
| llama | ~4.70 GB/token | 1166.7 GB/s |

The byte totals are accounting estimates carried from the 20260820 roofline
brief, not hardware DRAM counters. The measured read peak on this RTX 5090
is 1700-1792 GB/s. The same inference holds on fresh wall: tinygrad moves
more bytes but keeps the memory system idle during the serial support
chain, while llama achieves higher effective bandwidth with fewer bytes.

Serialization counterfactual on the fresh llama data:

```text
llama wall 4028.503 us -> 248.231 tok/s
force llama kernels serial: 5161.758 us -> 193.732 tok/s
llama's overlap is worth ~54.5 tok/s
```

## 6. Measurement notes and limits

- The tinygrad wall is a control/candidate/control A bracket, 24 samples x 4
  reps per arm; all three arms share token SHA
  `1e73e557e48b0c2f0792318e1a306f06a1412cd9800ba7a1e667b9c09c4a1254`.
- The llama wall is the fresh unprofiled 20-token row; this llama-bench
  build does not expose a token-stream SHA, so the tinygrad bracket is the
  SHA-gated arm.
- The llama Nsight trace did not finalize on its own after llama-bench
  exited, so the trace was finalized with an interrupt; the exported SQLite
  contains 74,174 events and the parser retained 27 complete steady replays
  after dropping 2 warmups.
- Interval overlap is measured here, not simultaneous useful traffic.
  Per-kernel wait-exit timestamps were not part of this packet, so the
  useful-body portion of the 1133.255 us remains `unmeasured`.
- `PROFILE=1` and `HCQ_GRAPH_PROFILE_JSON` had to be set in the process
  environment before import; setting them inside the harness process is
  too late for the current runtime.

## 7. Evidence

`docs/task_workflow/evidence/nv-ledger-overlap-audit-20260821/`:

- `tinygrad-capture.json`, `tinygrad-profile.jsonl`
- `tinygrad-wall-{control-a,candidate-1,control-b}.json`,
  `tinygrad-wall-bracket.json`
- `llama-unprofiled.json`, `llama-trace-bench.json`
- `llama-trace.nsys-rep`, `llama-trace.sqlite`, `llama-graph-dump.txt`
- `llama-dag.json`, `tinygrad-canonical.json`, `ledger.json`,
  `sensitivity.json`, `overlap-claim-audit.json`
- `sha256.txt` covers every retained file

## 8. Reusable reference statement

> On the RTX 5090 at `6570abc02`, llama hides 1133.255 us of its
> 5023.823 us kernel residence time (22.56%) while tinygrad hides only
> 9.214 us of its 4742.464 us (0.19%). Tinygrad's GEMV anchor bodies are
> 50.177 us faster, and S1 is 632.595 us more exposed. The measured wall
> gap is +703.452 us, and the interval identity closes exactly.
