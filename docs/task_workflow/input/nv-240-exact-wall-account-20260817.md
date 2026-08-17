# NV exact wall account at HEAD: what exactly blocks 240 (2026-08-17)

Date: 2026-08-17
Branch: `nvidia-bringup-20260731` (HEAD `07e9b2abe`, F5 keys.clone landed)
Status: **exact, additive, zero-residual wall account, same session.** Every
microsecond of the llama-vs-tinygrad d512 gap is attributed. No row is an
estimate; no class table is a crosswalk with a caveat. This supersedes the
"not additive" ledgers from 08-12/08-13/08-15/08-16 as the gap authority.

## 1. Headline (d512, Qwen3-8B-Q4_K_M, RTX 5090, same session)

| side | tok/s | ms/token | source |
| --- | ---: | ---: | --- |
| llama (`ac4cddeb0`, CUDA build, fresh `llama-bench -p 512 -n 10 -d 512 -r 5`) | **246.37** | **4.0589** | unprofiled; same session nsys node ledger (graphId 6, 762 nodes, 47 steady replays) |
| tinygrad (HEAD, production control route) | **208.84** | **4.7883** | unprofiled W-method timing child; NV HCQGraph node ledger (594 nodes, 34 steady tokens) |
| **gap** | | **+729.4 us/token** | |

Both node ledgers are the same measurement: per-kernel GPU durations summed
over one steady decode token, profiled in the same session. tinygrad is
profiled natively on the NV HCQ backend (`PROFILE=1` +
`HCQ_GRAPH_PROFILE_JSON`, per-kernel signal timestamps); llama is profiled with
nsys CUPTI at graph-node granularity.

## 2. The wall equation (exact, residual 0.0)

`wall = GPU busy (kernel union) + host gap`, where `GPU busy = node sum -
overlap mass`. Each row is a mean over the same steady tokens/replays, so
every sub-table adds exactly.

| term | tinygrad | llama | delta (tg - llama) |
| --- | ---: | ---: | ---: |
| wall (unprofiled) | 4788.3 | 4058.9 | **+729.4** |
| GPU busy (kernel union) | 4519.3 | 3890.5 | **+628.8** |
| host gap (wall - union) | 269.0 | 168.3 | **+100.6** |

Check: `628.8 + 100.6 = 729.4` exactly (residual 0.0000 us).

## 3. Why the GPU-busy delta is +628.8 us (the overlap row)

llama does MORE kernel work per token than tinygrad (node sum 5015.7 vs
4519.3, delta **-496.3 us** in tinygrad's favor), yet llama's GPU is busy
**less** wall time. The entire inversion is llama's in-graph overlap:

| term | tinygrad | llama | delta |
| --- | ---: | ---: | ---: |
| node sum (all kernels) | 4519.3 | 5015.7 | -496.3 |
| overlap mass (node sum - union) | 0.0 | 1125.1 | -1125.1 |
| GPU busy (union) | 4519.3 | 3890.5 | **+628.8** |

Check: `-496.3 - (-1125.1) = +628.8` exactly. tinygrad's decode runs one
serial NV compute queue with zero kernel overlap; llama's CUDA graph overlaps
~1125 us of its per-token kernel mass (quantize_q8_1, norms, rope, flash)
behind its mmq anchor.

## 4. The class table (sums exactly to the node-sum delta)

Common taxonomy (llama classes folded to match tinygrad roles): llama
`mmq + quantize_q8_1` = tinygrad `gemv` (we fold quant in-kernel); llama
`rope + kv_set_rows` = tinygrad `rope_kv`; llama `get_rows` =
`residual_cast`; llama `elementwise` = `other`. llama has no `reduce_output`
or `vocab_aux` kernels because those epilogues are absorbed in-kernel there.

| class | tinygrad us | llama us | delta |
| --- | ---: | ---: | ---: |
| gemv (incl. folded quant) | 3477.4 | 4138.0 | -660.6 |
| reduce_output | 312.1 | 0.0 | **+312.1** |
| norms | 206.2 | 307.7 | -101.4 |
| flash_score | 213.1 | 173.6 | +39.4 |
| flash_combine | 99.6 | 189.7 | -90.1 |
| rope_kv | 100.5 | 201.9 | -101.4 |
| vocab_aux | 59.5 | 0.0 | **+59.5** |
| other (small elt/reduce plumbing) | 49.1 | 1.9 | +47.2 |
| residual_cast | 1.9 | 2.9 | -0.9 |
| **node sum** | **4519.3** | **5015.7** | **-496.3** |

The class deltas sum to **-496.3 us exactly** (residual 0.0000). tinygrad is
already *faster than llama* on gemv, norms, rope_kv, flash_combine. The only
kernel-work rows where tinygrad is behind are `reduce_output` (+312.1, the
unabsorbed fp32 q/k + FFN-down output reduces) and `vocab_aux` (+59.5).

## 5. The 240 target in this account

240 tok/s = 4166.7 us/token. The path from 208.84:

| lever | mass | ceiling (1:1 wall) | status |
| --- | ---: | ---: | --- |
| reduce_output absorbed in-kernel | 312.1 us node | +~312 us wall | open row; measured FLAT for body-free fold at 08-13 HEAD, needs a wall-gated A/B at this HEAD |
| vocab_aux tail removed | 59.5 us node | +~60 us wall | F5 NO_GO_WALL (L2-cold u64 reduce); keys.clone() landed but row stays open |
| flash_score | 39.4 us | +~39 us | structural (4.16 vs 4.10 us isolated at parity); no new mechanism |
| **overlap (llama hides 1125 us; we hide 0)** | 1125.1 us | the other side of +628.8 | measured FLAT on NV multi-queue at 08-15; CONSTRUCTION-REQUIRED (anchor-shadow policy) |
| host gap (269.0 vs 168.3) | 100.6 us | +~101 us | eager/JIT handoff; graph substrate exhausted at 08-13 |

Kernel-work rows (reduce_output + vocab_aux + flash_score) total ~411 us of
node mass; even at perfect 1:1 wall recovery that lands at ~220 tok/s, not
240. The remaining ~318 us to 240 is the overlap+host row: llama's launch
hiding (in-graph overlap) plus its smaller host gap. **240 is not reachable by
fusing more kernel bodies at HEAD; it requires the overlap substrate (anchor
shadow / in-graph concurrency) that the 08-15 audit found
CONSTRUCTION-REQUIRED on native.**

## 6. What this changes vs the old ledgers

- The 08-12 "+652 us class crosswalk" and 08-16 "+308 us non-additive node
  crosswalk" are replaced by an account where class deltas sum to the node-sum
  delta and the wall delta decomposes with residual 0.0000.
- The old "tinygrad does more kernel work" framing is falsified at HEAD:
  tinygrad node sum (4519.3) is *below* llama (5015.7). We are not behind on
  kernel mass; we are behind on overlap and host gap.
- `reduce_output` is confirmed as the largest single open kernel-work row
  (+312.1 us) and the only class with meaningful mass where llama absorbs and
  we do not.

## 7. Reproduction

- tinygrad ledger: `scratchpad/nv_decode_wall_node_ledger.py` (NV HCQGraph
  profiler, steady-token signature filter `(32,64,128,256,114)`), then
  `scratchpad/nv_240_exact_wall_reconcile.py`.
- llama ledger: nsys `--cuda-graph-trace=node` on
  `llama-bench -p 512 -n 10 -d 512 -r 5`, `cuda_graph_timeline_ledger.py
  --trace ... --graph-id 6`.
- Evidence: `docs/task_workflow/evidence/nv-240-exact-wall-account-20260817.json`.
