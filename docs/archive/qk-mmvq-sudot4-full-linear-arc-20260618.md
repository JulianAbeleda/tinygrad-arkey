# Q4_K MMVQ sudot4 full-linear arc — VERDICT: FAIL the full-linear gate (q8-pack wall + lossy) 2026-06-18

After the signed-dot4 (`sudot4`) breakthrough made the 128-thread/row kernel **correct at 57% peak** (beating
opaque 52% at the KERNEL level), this arc built the full-linear isolated gate WITH the q8 activation pack cost
included. **Result: the whole-linear FAILS — the mandatory q8 pack eats the kernel win, and the path is q8-lossy
vs the byte-identical fp coop.** Do NOT route. RX 7900 XTX / gfx1100, Q4_K ffn_gate/up 12288×4096. Commit at
build: `d9be577d3`.

## Phase 0/1 — full-linear isolated measurement (q8 pack included, fresh inputs)

| component | µs | % peak | correct? |
|---|---|---|---|
| base fp (no q8 pack) | 77.4 | 41 | byte-identical |
| fp coop (no q8 pack) | 66.1 | 48 | **byte-identical** |
| opaque asm (prior best) | 60.5 | 52 | ✓ |
| **sudot4-128 kernel ALONE** | **55.0** | **57** | ✓ rel 0.006 (q8-lossy) |
| q8 signed pack (4 kernels) | 29.7 | — | — |

**Whole-linear (q8 pack + kernel):**
- single (1 pack + 1 kernel): **82.6µs = 0.80× fp coop** (slower).
- paired gate+up (1 pack + 2 kernels): **137.6µs = 1.12× base, 0.96× fp coop.**

**Gate: ≥1.3× base FALSE (1.12×) · ≥1.15× coop FALSE (0.96×) · ≥1.05× opaque FALSE. FAIL.**

## Why it fails — the q8-pack wall (the decisive arithmetic)
- The sudot4 kernel saves ~11µs/linear vs fp coop (55 vs 66µs).
- The q8 activation pack costs 29.7µs (4 kernels: 2× max-reduce ~15µs + quantize 7.9µs + signed-pack 6.9µs),
  amortizable only over gate+up (2 linears that share the FFN-norm activation) → **~15µs/linear**.
- **15µs added > 11µs saved → no net whole-linear win.** A fully fused pack (~7µs best case → 3.5µs/linear)
  would give ~1.13× coop — still under the 1.15× gate.
- The fp dequant path (coop) needs **no** q8 pack at all — that structural advantage keeps it competitive.

## Second, independent reason not to route: LOSSY
sudot4 quantizes activations to q8_1 (rel 0.006) → NOT byte-identical greedy. Every shipped MMVQ_COOP route
(lm_head, ffn_down, attn_q/o) and the fp coop are **byte-identical**. Routing a q8-lossy path that is at best
~neutral whole-linear, in place of an exact path, is a strictly worse trade. (Would also require a dNLL pass.)

## Verdict vs the scheduler-probe verdict
The dot4-ISA audit correctly overturned "the decomposition isn't the lever at the KERNEL level" — with native
signed dot4 the 128-thread kernel genuinely reaches 57% correct. **But the WHOLE-LINEAR reintroduces the
q8-pack wall** (the same wall that sank the earlier dp4a / Family-A int-dot attempts): any int-dot path pays a q8
activation-quant cost the fp path avoids, and for Q4_K ffn_gate/up that cost ≈ the kernel speedup. So:
- KERNEL level: sudot4 wins (57% > 52% opaque > 48% coop). **Real, banked.**
- WHOLE-LINEAR level: sudot4 does NOT beat the byte-identical fp coop. **Not routed.**

## What WOULD tip it (not pursued now)
1. A single fused q8 quant+pack kernel ≤ ~5µs (vs 29.7µs) AND amortized across >2 linears, OR
2. the sudot4 kernel tuned to ≥~63% (more headroom over the pack), AND
3. a dNLL pass clearing the q8-lossy quality bar.
All three would be needed; current isolated whole-linear is below the gate.

## Durable outcomes (banked)
- The `sudot4` `_sdot4` fix (correct native signed dot4) — shipped capability, value-tested, used by no default
  path but available.
- The 57%-correct 128-thread kernel structure (recorded; revivable if the q8 pack becomes cheap).
- The q8-pack-wall quantified: int-dot's mandatory activation quant ≈ its kernel win for Q4_K ffn_gate/up.

## Files
`[docs]` this; `bench/qk-mmvq-sudot4-full-linear/baseline.json`. No `[codegen]`/`[nn]`, no routing, no defaults.
The kernel probe was inline (not committed to model/primitive). The `sudot4` helper fix + value test were
shipped in the prior commit (`d9be577d3`).
