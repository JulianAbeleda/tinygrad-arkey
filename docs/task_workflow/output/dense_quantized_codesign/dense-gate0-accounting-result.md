# Dense representation Gate-0 result

Accounting is admissible; kernel and quality gates remain open. Exposure is bytes streamed by dense projections per decode token, not booked tok/s. `embedding` is resident but excluded from streamed projection exposure; tied/alias status is unproven and embedding/vocab remain independently charged.

## Role fixtures

Separate real fixtures cover Q, K, V, O, gate, up, down, and vocab: `blk.0.attn_q.weight` (5120x5120), `blk.0.attn_k.weight` and `blk.0.attn_v.weight` (1024x5120), `blk.0.attn_output.weight` (5120x5120), gate/up (17408x5120), down (5120x17408), and `output.weight` (151936x5120). `token_embd.weight` is resident embedding.

## Candidate populations

- R1 `S4_G32_P256`: all projection roles, plumbing gate.
- R2 `S4_G64_P256`: source-Q4_K projection tensors only, pending recurrent quality.
- R3 `S5_G32_P256`: source-Q6_K-sensitive projection tensors only, pending recurrent quality.

The ledger retains theoretical all-role columns for comparison, not as an admissible mixed-format plan. Per-tensor payload/padding and streamed aggregates are in [dense_tensor_accounting_ledger.json](dense_tensor_accounting_ledger.json).

## Role byte ledger

These are resident source bytes and theoretical full-role candidate bytes.
They are not quality-qualified populations.

| role | source | R1 | R2 | R3 |
| --- | ---: | ---: | ---: | ---: |
| Q | 589,824,000 | 589,824,000 | 557,056,000 | 720,896,000 |
| K | 117,964,800 | 117,964,800 | 111,411,200 | 144,179,200 |
| V | 144,998,400 | 117,964,800 | 111,411,200 | 144,179,200 |
| O | 589,824,000 | 589,824,000 | 557,056,000 | 720,896,000 |
| gate | 2,005,401,600 | 2,005,401,600 | 1,893,990,400 | 2,451,046,400 |
| up | 2,005,401,600 | 2,005,401,600 | 1,893,990,400 | 2,451,046,400 |
| down | 2,464,972,800 | 2,005,401,600 | 1,893,990,400 | 2,451,046,400 |
| vocab | 638,131,200 | 437,575,680 | 413,265,920 | 534,814,720 |
| embedding (resident only) | 437,575,680 | 437,575,680 | 413,265,920 | 534,814,720 |

The source dense projections stream 8,556,518,400 packed bytes per decode
token under this accounting. R1 is a same-byte experiment only on source
Q4_K tensors; replacing Q6_K tensors with R1 changes both bytes and quality.

## Population exposure, not speed

| candidate population | tensors | mixed streamed bytes | reduction versus source |
| --- | ---: | ---: | ---: |
| R1 on all projections, plumbing only | 281 | 7,869,358,080 | 8.03% |
| R2 on source-Q4_K projections | 240 | 8,202,624,000 | 4.14% |
| R3 on source-Q6_K projections | 41 | 8,202,526,720 | 4.14% |

These reductions are arithmetic exposure. They do not include calibration,
corrections, provider work, achieved DRAM rate, lifecycle translation, or
recurrent quality. Consequently they are not predicted or booked token-rate
gains.

## Reproduction

```sh
python3 docs/task_workflow/output/dense_quantized_codesign/build_ledger.py \
  --input docs/14b-role-facts-inventory-20260710.json \
  --output docs/task_workflow/output/dense_quantized_codesign/dense_tensor_accounting_ledger.json \
  --dense-verified
python3 -m pytest -q docs/task_workflow/output/dense_quantized_codesign/test_dense_accounting.py
```

The `--dense-verified` flag is an explicit assertion about the supplied model
facts. The builder does not infer density from a filename.
