# FMI-4 B1 knob probe

Verdict: `FAIL_B1_NO_ENV_KNOB_CLEARS_GATE`.

- `attn_k/v` best `q4_rt8`: `1.014x` baseline (`28.7%` HBM)
- `ffn_down` best `q6_rt16`: `1.004x` baseline (`86.0%` HBM)
- `ffn_gate/up` best `default`: `1.0x` baseline (`37.2%` HBM)

Existing env launch-shape knobs do not close Track B; next Track B surface is runtime/cache identity or renderer project.
