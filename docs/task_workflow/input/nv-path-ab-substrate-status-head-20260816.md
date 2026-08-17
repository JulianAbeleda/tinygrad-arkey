# NV Path A / Path B substrate status at HEAD (2026-08-16)

Date: 2026-08-16
Branch: `nvidia-bringup-20260731`
HEAD: `6bbaa6221`
Status: **read-only reconciliation. No GPU session. Answers "do we have the
substrate for Path A (kernel work) or Path B (anchor shadow / overlap)?" at
HEAD with the measured disposition of every sub-row.**

## 1. Position at HEAD

| side | tok/s | ms/token |
| --- | ---: | ---: |
| tinygrad NV (Q6 V + Q6 FFN-down landed) | ~205-207 | ~4.8-4.9 |
| llama fresh / pair | 247.98 / 245.45 | 4.037 / 4.074 |
| CUDA route baseline (Route B harness) | ~179.4 | 5.57 |

Gap to llama at HEAD: ~38-43 tok/s (~0.7-0.8 ms/token). The 08-15 audit's
`193.5` baseline is stale; the two Q6 four-warp promotions are what moved it.

## 2. Path A (kernel work): substrate present, largely consumed

Substrate = in-kernel quant GEMV + four-warp fp16 geometry + reduce-output
primitive + epilogue absorption. Proven and promoted 08-16:

- Q6 FFN-down four-warp fp16: -39.0 us/token, WALL_PASS
  (`nv-q6k-ffn-down-four-warp-fp16-promotion-scope-20260816.md`)
- Q6 attention-V four-warp fp16: -147.35 us/token, WALL_PASS
  (`nv-q6k-v-four-warp-fp16-promotion-20260816.md`)

Remaining Path A rows at HEAD:

| row | census us | disposition |
| --- | ---: | --- |
| reduce-output epilogue (q/k + ffn-down) | ~378 | primitive exists; body-free fold measured FLAT (wall-blocked) |
| M1 norm chains + E/r plumbing | ~238 | body-free FLAT, body-adding NO-GO (closed) |
| flash score installed gap | +122 | bodies at parity; gap is launch/graph behavior, not body |
| vocab aux (F5) | ~52 | only unbuilt mechanism; packed-u64 carry exists, single-pass cross-tile max not built |

The reduce-output and M1 folds are values-blocked (rendered, measured FLAT), not
substrate-blocked. The only Path A item that is substrate-missing is the vocab
single-pass cross-tile max (~50 us, ~+2 tok/s if it lands).

## 3. Path B (anchor shadow / overlap): no buildable substrate at HEAD

| sub-route | status | measured evidence |
| --- | --- | --- |
| Route A native multi-compute channel | CONSTRUCTION_BLOCKED | RM rejects `NVA06F_CTRL_CMD_BIND` (`NV_ERR_INVALID_ARGUMENT`); without BIND extra channels never execute (`nv-decode-overlap-phase0-measurement-record-20260804.md`) |
| Route B CUDA multi-stream graph | present, measured FLAT | 1-4 streams, tokens bitwise identical, wall flat ~179-180 tok/s; width-4 q/k/v is bandwidth-bound (`nv-overlap-route-b-head-wall-record-20260815.md`) |
| PDL (llama's actual single-stream mechanism) | economics-negative | launch-gap half already wired via dependent-QMD chain; programmatic half CONSTRUCTION-REQUIRED but recovers ~18-33 us (`nv-llama-pdl-launch-hiding-trace-record-20260816.md`) |

Conclusion: Path B has no buildable substrate at HEAD. llama's ~925 us overlap
mass is mostly quantize/norm/rope that tinygrad already fused away, so no route
converts that mass to wall on our decomposition.

## 4. Honest target

Path A's big lever (Q6 core) is banked; the audit's Path A ceiling of
~215-222 tok/s requires vocab (~50 us) plus the reduce/norm folds that
measured FLAT. Path B is blocked on driver construction, flat on CUDA
multi-stream, and negative on PDL. Anything past ~215-222 is a Path B
substrate problem that is unsolved at HEAD.

## Evidence

- `nv-240-audit-reconciled-20260815.md` (path decomposition, Path A ceiling)
- `nv-substrate-definition-20260815.md` (capability vs wall block)
- `nv-q6k-v-four-warp-fp16-promotion-20260816.md`,
  `nv-q6k-ffn-down-four-warp-fp16-promotion-scope-20260816.md` (Q6 landed)
- `nv-decode-overlap-phase0-measurement-record-20260804.md` (Route A blocked)
- `nv-overlap-route-b-head-wall-record-20260815.md` (Route B FLAT)
- `nv-llama-pdl-launch-hiding-trace-record-20260816.md` (PDL economics)
- `nv-flash-score-floor-test-head-20260816.md` (flash body parity, floor falsified)
- `nv-220-composition-review-outcome-20260815.md` (fusion-to-220 falsified)
