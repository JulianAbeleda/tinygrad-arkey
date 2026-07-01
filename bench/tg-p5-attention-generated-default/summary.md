# TG-P5 8B Attention: Generated G5 vs Owned HIP

Verdict: **TG_P5_REFUTE_GENERATED_ATTENTION_SLOWER**

Generated G5 block-tile flash decode generalizes correctly to the 8B geometry (Hq=32/Hkv=8, G=4) and is
token-identical + route-bound to the owned oracle, but it is **slower**, so owned HIP stays the default.

| ctx | owned tok/s | generated tok/s | % of owned | token_match | route_bound |
|---|---|---|---|---|---|
| 512 | 107.8 | 94.4 | 87.6% | True | True |
| 4096 | 97.9 | 93.6 | 95.6% | True | True |

Decision: keep `decode_attention_owned_two_kernel` (owned HIP) as the 8B default per the TG-P5 stop rule
(do not force purity by slowing the model). The 8B owned attention remains the one honest `external_handwritten_kernel`
purity debt; full default purity is not reachable here without a decode regression.
