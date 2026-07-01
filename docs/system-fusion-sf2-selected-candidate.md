# System Fusion SF2 — Selected Candidate: decode_silu_gate_fusion

Date: 2026-07-01. Follows SF1.

## Selected candidate

**decode_silu_gate_fusion**

Remove the intermediate `.contiguous()` call on `tinygrad/llm/model.py:1017` behind the flag `DECODE_FUSE_SILU_GATE` (default-off). When the flag is on, tinygrad's scheduler can fuse `silu(gate) * up` into a single kernel instead of two.

## Root cause

`model.py:1017` (non-fused gate/up path):
```python
return self.ffn_down(self.ffn_gate(x).silu().contiguous() * self.ffn_up(x))
```

The `.contiguous()` between `.silu()` and `* self.ffn_up(x)` forces tinygrad to materialize the silu(gate) intermediate buffer globally. This creates two separate kernel launches:
- `E_136_32_4` — silu(gate), 17408 elements, 40 calls/step
- `E_136_32_4n1` — gate×up multiply, 17408 elements, 40 calls/step

Without `.contiguous()`, tinygrad's scheduler can fuse these into one kernel.

The comment at line 1010 explicitly marks this: `# TODO: remove the need for this contiguous`.

## Implementation

Flag: `DECODE_FUSE_SILU_GATE` (default=0, rollback by setting to 0)

Change at model.py:1017:
```python
if getenv("DECODE_FUSE_SILU_GATE", 0):
  return self.ffn_down(self.ffn_gate(x).silu() * self.ffn_up(x))
return self.ffn_down(self.ffn_gate(x).silu().contiguous() * self.ffn_up(x))
```

Only the non-fused gate/up path (line 1017) is modified. The `ffn_gateup` (B1 fused weight) path at line 1013 and the `Q4K_UNFUSE` path at line 1016 are NOT touched.

## Why this candidate

1. **Root cause found**: the exact code location and the reason (.contiguous() forces materialization) are known.
2. **One-line fix**: minimal change, minimal risk surface.
3. **Generated path**: no handwritten kernel, no new UOp primitive — tinygrad's existing scheduler handles fusion.
4. **Non-numeric**: same computation, different kernel schedule. Expected rel_rmse ≈ 0.
5. **Amdahl**: 1.26% at ctx512 (40 launches × 2 kernels → 40 launches × 1 kernel).
6. **Default-off**: gate behind DECODE_FUSE_SILU_GATE=0 so it does not affect shipped behavior until promoted.

## Rollback policy

Set `DECODE_FUSE_SILU_GATE=0` (already the default). This restores the `.contiguous()` path.

## Reopen conditions

- If SF3 W==D shows LOW_AMDAHL_NO_MOVEMENT: the fusion fires but scheduler overhead or launch amortization swamps the benefit. Reopen when a broader scheduler pass fuses multiple elementwise ops at once (the residual_add and rmsnorm_scale groups together with the silu group).
- If SF3 shows CORRECTNESS_FAIL: record the exact failure mode (numeric divergence, shape mismatch, etc.) before re-trying.
- If refuted: do_not_retry=False — the root cause is real; a correctness failure here means a tinygrad scheduler bug, which is fixable.

## BoltBeam candidate reference

See `boltbeam/data/candidates.json`: `decode_silu_gate_fusion`.
