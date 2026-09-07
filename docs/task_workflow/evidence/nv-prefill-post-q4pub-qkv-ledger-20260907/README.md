# QKV residual and explicit shared-producer discriminator

Under the promoted 44.627939 ms selected control midpoint, diagnostic native QKV measures 41.009238 ms (40.653790 minimum), leaving a 3.618701 ms residual. The native arm removes the selected Q/K/V region's 90 classified Q8 producers and replaces them with 36 shared native producers, a net 54-launch change.

At the existing `TransformerBlock._attention.project_qkv` seam, the retained dual flat/tile producer was tested without source-key pairing. Its flat record is byte-for-byte compatible with both generated Q/Q4V and Q6V (655360 uint32, identical dtype/shape/stride); tile output preserves K's ABI. Block0 Q6V and block4 Q4V consumers match all independent-producer Q/K/V outputs exactly across five cycles and keep input/weights read-only.

The producer lifecycle rejects integration: R15 dual is 176.562 us versus 100.850 us for independent flat+tile and 104.506 us for independent flat+tile+Q6-flat. Thus removing 54 launches with this producer adds 72.056 us on each Q6 layer. No model wiring was added. The remaining QKV debt must come from body/conditioning differences, not launch count alone.
