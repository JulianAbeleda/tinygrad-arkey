# NV installed-island Phase 9 coupled 4096 norm/provider audit

Date: 2026-08-22
Branch: `nvidia-bringup-20260731`
HEAD: `6570abc025514273faa100c66b979e531585a1e1`

Evidence: `docs/task_workflow/evidence/nv-installed-islands-20260822/phase9/`

## Findings, ordered by wall severity

`MEASURED` The `attn/ffn/final 4096 norm` row (`+125.439 us`) is not an
independent recovery claim. It is coupled to the `activation quant` row
(`-114.432 us`, tinygrad advantage):

```text
4096 norm       +125.439 us   (tinygrad 328.800 - llama 203.361)
activation quant -114.432 us  (tinygrad  30.688 - llama 145.120)
net              +11.007 us
```

Any accounting that removes a norm cost while reintroducing the quant
advantage as separate work is rejected. The raw `125.44 us` number is never a
standalone ceiling.

`MEASURED` The 17 `rmsnorm_q8_1_llama_provider_4096` nodes are the frozen
control. Each one fuses the attention-input norm with the Q8 activation quant
and runs at `1.728 us` installed. llama pays `attn_norm 2.752 + Q_quant 0.672
+ K_quant 0.672 + V_quant 0.672 = 4.768 us` for the same semantic unit. This
provider is the source of the `-114.432 us` quant advantage and must be
preserved in any alternative route.

`MEASURED` The 94 standalone norm kernels split into three semantic families:

| family | nodes | node_sum | median P | grid | block |
| --- | ---: | ---: | ---: | --- | --- |
| `reduce_output_rmsnorm_1_4096` | 19 | 136.672 us | 7.008 us | [1,1,1] | [32,16,1] |
| `r_16_256` | 37 | 123.232 us | 3.232 us | [1,1,1] | [16,1,1] |
| `E_32_32_4` | 38 | 68.896 us | 1.792 us | [32,1,1] | [32,1,1] |

`MEASURED` llama folds each semantic norm into one `rms_norm_f32` kernel with
block `[1024,1,1]` (1024 threads): attn_norm `2.752 us`, ffn_norm `2.784 us`,
final_norm `2.688 us`. tinygrad materializes the reduce and the normalize as
separate kernels, and its attn norm reduce is under-parallelized.

## Exact-body / clean-HCQ / production decomposition

`MEASURED` Every family was replayed with its exact production cubin (nsys
2000 reps, body `B`) and clean chained-HCQ drain (slope `C`). `P` is the
per-call production command-interval median, `D = C - B`, `R = P - C`,
identity residual zero in every row.

| kernel | P | B | C | D | R |
| --- | ---: | ---: | ---: | ---: | ---: |
| `reduce_output_rmsnorm_1_4096` | 7.008 | 4.864 | 5.338 | 0.474 | 1.670 |
| `r_16_256` | 3.232 | 2.816 | 3.266 | 0.450 | -0.034 |
| `E_32_32_4` (f14a5cc, x37) | 1.792 | 0.608 | 1.065 | 0.457 | 0.727 |
| `E_32_32_4` (5a5673a, x1) | 1.152 | 0.576 | 1.020 | 0.445 | 0.132 |
| `rmsnorm_q8_1_llama_provider_4096` | 1.728 | 1.184 | 1.633 | 0.449 | 0.095 |

`MEASURED` Role-wide, the standalone norm mass is:

```text
body B     219.712 us   (19 x 4.864 + 37 x 2.816 + 38 x 0.608)
dispatch D  43.022 us   (94 x ~0.46)
residual R  58.098 us   (19 x 1.670 + 37 x -0.034 + 38 x 0.727)
total      320.832 us   (median-based; census mean-based 328.800)
```

tinygrad pure norm body (`219.712 us`) is only `+16.35 us` above llama's
`203.361 us`. The `+125.44 us` census deficit is dominated by install mass:
tinygrad runs 94 physical kernels against llama's 73, so the split
reduce/expand structure and its residual dominate, not arithmetic.

## Verdicts

```text
reduce_output_rmsnorm_1_4096  BODY_DOMINANT   (B = 69% of P; 512-thread attn norm, 4.864 us vs llama 2.752)
r_16_256                      BODY_DOMINANT   (B = 87% of P; 16-thread ffn reduce at llama parity)
E_32_32_4                     MIXED           (B 34% / D 25% / R 41%; normalize split-out)
rmsnorm_q8_1_llama_provider   FROZEN_CONTROL  (efficient fused norm+quant; preserves -114.43 us)
```

`INFERRED` The standalone norm deficit has two independent terms:

```text
attn-norm body   (4.864 - 2.752) x 19 = 40.1 us   BODY
ffn split-out    (3.43 - 2.784) body + extra launch x 37  INSTALL + small body
```

The `r_16_256` reduce is already at llama body parity; the ffn deficit is the
separate `E_32_32_4` normalize node that llama absorbs into the same
1024-thread kernel.

## Positional mapping confirmation

`MEASURED` `E_32_32_4`, `r_16_256`, and `reduce_output_rmsnorm_1_4096` are
semantically RMSNorm reduce/normalize kernels by shape and buffer contract
(4096-element fp16 row, 4-byte rms scalar, or 8192-byte in/out with weight).
They are not rope. The rope/store `E_16_32_4_2`, `E_8_8_16_2`, and
`E_16_4_2_8_16_2_4_4` families remain positionally mapped and are not
reclassified by this phase.

## Decision

`MEASURED` The net legal ceiling for the coupled norm/provider island is
`+11.007 us`, not `125.44 us`. It does not outrank the earlier islands on a
non-double-counted basis. The standalone-norm body deficit (`~40 us` attn norm)
is a real but secondary target, and any recovery must preserve the 17-provider
path. One body-topology scope for the 19 standalone attn norms is admissible,
ranked below the FFN/KV/flash/tail islands.

## Ledger snapshot

```text
node_sum   = 4677.920 us (tinygrad) / 3878.254 us (llama)
union      = 4671.500 us (tinygrad) / 3878.254 us (llama PDL-off)
overlap    = 6.420 us (tinygrad) / 0 us (llama PDL-off)
wall       = 4771.423 us (fresh control)
host_gap   = unmeasured single-domain
useful_body = unmeasured
booked_recovery = 0.000 us
remaining_to_240 = 604.756 us
```
