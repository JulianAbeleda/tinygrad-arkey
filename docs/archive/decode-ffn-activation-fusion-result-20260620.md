# Decode FFN Activation Fusion Result (Deliverable 4)

Date: 2026-06-20

Verdict: `NO_CHEAP_FFN_FUSION_PASSES` — the cheap lever (removing the forced `.contiguous()`) does not eliminate
the target kernel `E_49152`, because the consumer `ffn_down` is a custom Q6_K GEMV that requires a realized
input. Real elimination needs a fused custom kernel (gate/up-epilogue or ffn_down-prologue), specced below and
**deferred**. Default decode behavior NOT changed.

## Target (from Deliverable 3)

`E_49152_32_3` = the FFN `silu(gate) * up` activation = **1.24 ms/token** (56% of elementwise, flat across ctx,
present in both baseline and q8). It is **launch/dispatch-overhead-bound**: ~147 KB of HBM traffic (~0.15 µs of
work) costing ~33 µs/call × 36 calls. Decode is GPU-execution-bound on ~1074 tiny kernels (D≈W, host-sync 0%), so
the only lever is **removing the launch via fusion**, not a faster elementwise (a custom elementwise kernel has
the same per-launch overhead).

## Candidate tested

### C1 — Remove forced silu materialization (`FFN_ACT_VARIANT=no_contig`)

Current default (model.py:804): `self.ffn_down(self.ffn_gate(x).silu().contiguous() * self.ffn_up(x))`. The
`.contiguous()` is a known wart (line 797 "TODO: remove the need for this contiguous"). Hypothesis: dropping it
lets the scheduler avoid a standalone activation kernel.

| config | ctx1024 wall | E_49152 ms | elementwise ms |
|---|---:|---:|---:|
| default | 14.98 | 1.24 | 2.20 |
| `no_contig` | 14.86 | **1.22** | 2.07 |

Result: **no win on the target.** `E_49152` is unchanged (1.22 vs 1.24 ms) — `ffn_down`'s custom GEMV forces
`silu(gate)*up` to materialize regardless of the `.contiguous()`. Only a small (~0.13 ms) cast-glue cleanup,
within noise of the wall. (Tested behind a default-off env flag, since reverted — no core change retained.)

## Conclusion

Removing the contiguous does not capture the 1.24 ms. The activation must materialize because the consumer is a
hand-written GEMV kernel that reads a realized buffer. Real elimination requires fusing `silu(gate)*up` into a
custom kernel so it never round-trips through a standalone launch.

### Deferred real candidate (build spec)

Two options, both custom-kernel work:
1. **Fused gate/up-with-activation producer**: extend the fused `ffn_gateup` custom kernel (or the q4k gate/up
   GEMV epilogue in `extra/q4_k_gemv_primitive.py`) to compute `silu(gate)*up` and emit the 12288-vector
   activation directly — eliminating `E_49152`'s 36 launches.
2. **Fused ffn_down-prologue**: extend the q6k `ffn_down` GEMV (`extra/q6_k_gemv_primitive.py`) to take `gate`
   and `up` as inputs and apply `silu(gate)*up` on load.
- Gate: removes/shrinks `E_49152`, recovers ≥0.5 ms@1024, exact greedy/dNLL within policy; then W==D (≥3%@1024,
  no ctx4096 regress); q8 route still works or marked q8-only.
- Effort: bounded custom-kernel build (~1-2 days), not a cheap candidate. Expected recovery up to ~1.24 ms →
  ~73 tok/s @1024 (baseline). Recommended as the #2 decode build (after attention reduce/stat fusion).

## Commands

```bash
FFN_ACT_VARIANT=no_contig PYTHONPATH=. python3 extra/qk_decode_elementwise_cost_split.py \
  --child-out /tmp/_ffn_nocontig.json --mode baseline --ckpts 512 1024 --nmeas 12 --warmups 8
```

## Boundary

No decode default changed. The `FFN_ACT_VARIANT` test flag was reverted out of `tinygrad/llm/model.py`.
