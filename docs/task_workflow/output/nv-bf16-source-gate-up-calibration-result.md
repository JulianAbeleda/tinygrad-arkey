# BF16-source gate/up calibration result

## Outcome

The official Qwen3-8B BF16 source is usable on this machine through streamed
shards, but the first compact contracts do not yet pass quality. The source
removes the prior double-quantization ambiguity; it does not by itself make the
136-byte symmetric packet admissible.

## Machine and acquisition

- Removed the user-authorized `Qwen3-14B-Q4_K_M.gguf` (8.4 GiB).
- Downloaded the official Qwen3-8B index and shard 1 of 5.
- Official total BF16 checkpoint size: 16,381,470,720 bytes.
- Shard 1 contains block 0 gate/up BF16 tensors, each 12288x4096.
- The layer-at-a-time workflow fits; full simultaneous source duplication is
  unnecessary.

## Correct comparison protocol

The serving control is Q4_K, so replacing one layer with BF16 changes both the
candidate format and the old Q4_K error. A raw-BF16 block-0 arm was therefore
added. It preserves the recurrent token sequence. BF16-derived candidates are
judged against that mixed-model BF16 baseline, not directly against Q4_K.

| BF16-derived block-0 contract | Stack relative L2 vs BF16 block | Tokens | Verdict |
|---|---:|---|---|
| U4Z8 max scale, 136 B/block | 0.343869 | diverges after row 1 | stop |
| U4Z8 local weight-MSE scale, 136 B/block | 0.357569 | diverges after row 1 | stop |
| affine U4 group-64, 140 B/block | 0.003799 | preserved | misses 0.001 limit |
| affine U4, one-vector activation-weighted scale | 0.004259 | preserved | overfit; misses limit |

The exact production gate/up input was captured below the compiled-function
boundary. Its feature energy is highly nonuniform, confirming that activation-
aware calibration is relevant. One vector is not a calibration corpus and made
the result worse.

## Current disposition

The higher-precision-source route is feasible but not complete. The next gate
is a multi-prompt/multi-position activation collector followed by held-out
calibration. It must separate calibration and qualification prompts. Only a
contract below 0.001 stacked relative L2 at one layer proceeds to progressive
layer dosing and packed-kernel work.

No production route changed. No recovery is booked. The installed endpoint
remains 245.948 tok/s.
