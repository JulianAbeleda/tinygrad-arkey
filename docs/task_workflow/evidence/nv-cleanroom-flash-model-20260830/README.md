# Clean-room pp512 Flash one-query CTA rejection

Decision: **REJECT**. No production route was promoted.

The exact fp16 kernel and HCQ binding pass the full `[1,32,512,128]` output oracle with max absolute error `4.470348e-08`. Its standalone launch is `1337.344 us` for one model layer.

The earlier `38.65 us/layer` claim was a unit error: the standalone launch was divided by the model's 36 layers even though the launch itself already represents exactly one layer.

The strict default-off whole-model smoke measured `84.924811 ms` versus the promoted baseline near `38.47 ms`, with the same token `198`. The approximately `46.45 ms` regression agrees with 36 invocations of a roughly `1.3 ms` kernel.

Root cause: the architecture launches one CTA per `(query head, query token)`, or 16,384 CTAs per layer. A competitive design must process a query tile per CTA and reuse staged K/V across those queries; vectorized K loads alone cannot offset the excessive CTA and repeated-KV work.

The temporary model seam was removed after this smoke. The standalone kernel, binding, and oracle remain research artifacts only.
