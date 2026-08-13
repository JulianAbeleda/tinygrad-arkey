# NV decode gap - tinygrad CUPTI node ledger vs llama (2026-08-13)

Date: 2026-08-13
Branch: `nvidia-bringup-20260731` (HEAD `a985c60ca`)
Status: **measurement record.** First same-measurement CUPTI node ledger of the
tinygrad d512 decode token, paired against the pinned llama ledger. Answers
"exactly why are we slower" as a direct per-class subtraction rather than the
DEBUG=2 prime-token estimate used before.

## 1. Method

Both ledgers are CUPTI `CUPTI_ACTIVITY_KIND_KERNEL` rows from
`nsys profile --cuda-graph-trace=node --resolve-symbols=false`, so node-sum,
span, overlap mass and per-class exposure are the same kind of measurement on
both sides.

- llama: pinned trace `/tmp/llama_tg10_node_20260812.sqlite`, graphId 5,
  762 nodes, 16 replays (`nv-llama-d512-node-ledger-20260812.json`).
- tinygrad: fresh capture `/tmp/tg_node_20260813.sqlite` (HEAD `a985c60ca`,
  `DEV=CUDA CUDA_GRAPH_STREAMS=1`, the canonical d512 decode harness,
  9 replays). The decode token is captured as six CUDA graphs
  `graphId 2,5,8,11,14,17` with `32+64+128+256+512+29 = 1021` nodes; the
  matching replay of every graph is concatenated into one per-token timeline.
- Tool: `extra/llm_research/decode/nv_tinygrad_node_ledger.py`
  (`tinygrad.cuda_graph_timeline_ledger.v1`, same metric definitions as the
  llama ledger).

Both numbers are profiled; profiled intervals are not the unprofiled token
wall. The profiler inflates small-kernel node-sum more than large-kernel
node-sum, so the absolute node-sum delta below is an upper bound on the true
work delta.

## 2. Headline (d512, same measurement)

| metric | llama | tinygrad | delta |
| --- | ---: | ---: | ---: |
| kernels / token | 762 | 1021 | +259 |
| node-sum (GPU work) | 4774.4 us | 5149.4 us | +375.0 us |
| span (exposed time) | 3835.2 us | 5380.5 us | +1545.3 us |
| overlap mass (hidden work) | 946.4 us | 0.0 us | -946.4 us |
| span discount | +19.7% | -4.5% | |
| internal gap | 8.0 us | 232.2 us | +224.2 us |
| inter-replay host gap | 212.5 us | 269.4 us | +56.9 us |

The dominant fact is not the extra raw work (+375 us) but **overlap**:
llama hides 946 us of non-GEMV work behind its 217-kernel GEMV chain; tinygrad
hides zero. tinygrad's span (5380 us) is larger than llama's span (3835 us)
by 1545 us, and its negative span discount means it even pays 232 us of
launch gap between its serial kernels.

The earlier co-schedule probe (`nv-co-schedule-scan-head-20260812.json`) showed
this overlap is not recoverable by multi-stream scheduling: the 307 support
nodes are data-dependent (on the critical path), so the co-schedule ceiling is
only 33 us. Recovering llama's 946 us therefore means **fusion** (absorb the
support kernels into the GEMV epilogues), which is exactly what llama already
does.

## 3. Per-class subtraction

Classes are grouped so both sides partition the token. llama's
`quantize_q8_1` is folded into tinygrad's `gemv` anchor; llama's output-reduce
and vocab aux are fused into its `mmq`, so those rows are zero for llama and
non-zero for tinygrad.

| semantic class | llama n | llama us | tinygrad n | tinygrad us | delta us |
| --- | ---: | ---: | ---: | ---: | ---: |
| GEMV + folded quant (anchor) | 434 | 4024.7 | 253 | 4130.3 | +105.6 |
| flash score | 36 | 113.9 | 36 | 182.0 | +68.1 |
| flash combine | 36 | 120.5 | 36 | 89.0 | -31.5 |
| norms (reduce + epilogue) | 145 | 307.6 | 362 | 416.2 | +108.6 |
| rope + kv store | 108 | 201.0 | 72 | 67.1 | -133.9 |
| residual + small elt plumbing | 3 | 4.8 | 234 | 119.2 | +114.4 |
| vocab non-GEMV aux | 0 | 0.0 | 10 | 129.8 | +129.8 |
| output-reduce epilogue | 0 | 0.0 | 18 | 15.7 | +15.7 |
| **total** | **762** | **4774.4** | **1021** | **5149.4** | **+375.0** |

Reading the deltas:

- **Support kernels are the whole story.** tinygrad emits 696 support kernels
  (`norms` 362 + `residual_cast` 234 + `rope_kv` 72 + `vocab_aux` 10 +
  `reduce_output` 18) totaling 748 us that llama has fused into 217 GEMVs.
  The residual/norm/vocab rows alone are +353 us of the +375 us node-sum gap.
- **GEMV bodies are near parity.** The anchor delta is +105.6 us; the vocab
  GEMV (`q6k_gen_coop_151936_4096`, 401 us vs llama's 303.6 us vocab mmq) is
  most of it, consistent with the closed per-shape GEMV sweep.
- **We already win** rope/kv (-133.9 us, fused into GEMV epilogues) and
  flash combine (-31.5 us).
- **flash score is +68 us** at 182 us vs llama's 113.9 us; the tile sweep
  already showed this is structural (zero-load floor 5.3 us is still 1.7x
  llama), not geometry.

## 4. Tok/s translation

Baseline tinygrad unprofiled wall 5.2031 ms = 192.19 tok/s; llama pair wall
4.0741 ms = 245.45 tok/s. In the 190-205 band ~25 us/token saved ~= +1 tok/s
(census-to-wall mapping is 0.6 for body-adding changes, ~1.0 for pure kernel
removal).

| row | us/token | ceiling |
| --- | ---: | ---: |
| norms | +108.6 | ~+4 tok/s at 1:1 removal |
| residual + small elt plumbing | +114.4 | ~+5 tok/s at 1:1 removal |
| vocab non-GEMV aux | +129.8 | ~+5 tok/s at 1:1 removal |
| flash score | +68.1 | ~+3 tok/s at floor |
| GEMV anchor | +105.6 | closed per-shape (Q4 FFN-down/vocab NO-GO) |

The fused-ceiling arithmetic is unchanged from the 08-12 attribution: closing
all open class deltas at 1:1 lands at ~4.55 ms/token ~= 220 tok/s. The 946 us
overlap loss is the same mass as that fusion ceiling viewed from the exposure
side: llama's 20% overlap is its support work already fused into the GEMV
chain, so the two numbers are the same target, not additive. Full wall parity
additionally requires matching llama's launch hiding (the last ~0.3 ms).

## 5. Verdict

The gap is structural, not a one-kernel defect: llama runs 762 kernels and
hides 20% of them behind its GEMV chain; tinygrad runs 1021 kernels fully
serially. The next lever is the generic reduce-output/norm/residual/vocab
fusion primitive, not more per-shape GEMV tuning. This confirms the standing
process direction: trace llama -> validate arithmetic -> trace the e2e
dependency graph -> implement the fusion that removes the 687 support kernels.

## Evidence

- `docs/task_workflow/evidence/nv-tinygrad-d512-node-ledger-20260813.json`
  (schema `tinygrad.cuda_graph_timeline_ledger.v1`, this ledger)
- `docs/task_workflow/evidence/nv-llama-d512-node-ledger-20260812.json`
  (pinned llama reference)
- raw tinygrad trace: `/tmp/tg_node_20260813.sqlite`
  (sha256 `560a3c4d9c3d2bdfb639d28771981998f5c8c8ef83ce72f8559609b9a752cee4`)
- tool: `extra/llm_research/decode/nv_tinygrad_node_ledger.py`
