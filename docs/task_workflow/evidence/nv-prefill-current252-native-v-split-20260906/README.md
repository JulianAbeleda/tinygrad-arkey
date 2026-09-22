# Current252 V quant split and QKV interaction ledger

Against the generated control mean 47.818056 ms, native Q4 V is 47.387569 ms and saves 0.430487 ms; native Q6 V is 47.422113 ms and saves 0.395943 ms. Both are 18-call populations. Together with Q (+0.634061 ms) and K (-0.084644 ms), isolated substitutions explain 1.375848 ms of the 3.133466 ms combined QKV advantage. The remaining 1.757618 ms is a composition interaction.

The generated graph launches 108 role-local Q/K/V producers: Q36, K36, Q4-V18, Q6-V18. Combined native QKV launches 36 shared DS4 producers for Q+K(+Q4-V where applicable) and 18 D4 producers for Q6-V, removing 54 producer calls. The residual includes this producer sharing plus changed graph handoff/queue placement and must not be assigned to a body in isolation.

Every arm PASSes token198, canonical252/252, zero overlays, Q8 census216, and exact replay5. Each native V arm explicitly proves 18 native main and 18 native fixup calls. Native programs remain diagnostic only.
