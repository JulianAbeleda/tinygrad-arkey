# Decode fused-tile occupancy + roofline baseline (2026-06-26)

Purpose: a derived, reusable baseline for choosing the **split count S** of a generated split-KV decode
tile, so occupancy is set by math (matching the owned route) rather than guessed. Used by the
fused-xlane-score-PV tile (`docs/decode-fused-xlane-score-pv-tile-scope.md`) and any future generated
decode-attention tile.

Hardware (gfx1100 / RX 7900 XTX), from `extra/qk_split_kv_economics_audit.py`:
`CU_COUNT = 96`, `HBM_PEAK_GBPS = 960`.

## How the fine-tuned (owned) kernel achieves occupancy

The owned route is Flash-Decoding split-KV. It uses occupancy purely as a function of the split count:

- `DECODE_ATTN_AMDGCN_S = 48` splits (from `qk_decode_attention_fused_score_state_pv_attribution.py:209`).
- Grid `(Hkv, S)` → `tile_workgroups = Hkv · S = 8 · 48 = 384` → `384 / 96 = ` **4.0 workgroups/CU**.
- Grid is keyed on `Hkv` (not `Hq`), so the `G = Hq/Hkv = 4` query heads share each wave → GQA K-reuse is
  preserved while still launching 384 tiles.
- A separate LSE combine merges the 48 partials per output.

So the "magic" is just S=48. There is no special primitive — split-KV is the occupancy lever.

## The occupancy law (derives S=48)

HBM-bound kernels need enough resident waves per CU to keep memory requests in flight. The latency-hiding
condition is `waves/CU ≥ k`, with `k ≈ 4–8` for memory-bound kernels. With one wave per tile:

$$\text{wg} = H_{kv}\cdot S,\qquad \frac{\text{wg}}{\text{CU}} = \frac{H_{kv}\cdot S}{96}\ \ge\ k
\ \Rightarrow\ S \ge \frac{k\cdot \text{CU}}{H_{kv}} = \frac{4\cdot 96}{8} = 48.$$

So **S=48 is `4·CU/Hkv`** — the owned route's constant is derivable, not tuned.

| S (splits) | wg = Hkv·S | wg/CU = S/12 | regime |
|---:|---:|---:|---|
| 4 | 32 | 0.33 | starved (request-limited; = the owned *combine*'s latency-bound regime) |
| 12 | 96 | 1.0 | marginal |
| 24 | 192 | 2.0 | adequate |
| **48 (owned)** | **384** | **4.0** | saturating |
| 96 | 768 | 8.0 | headroom (test if combine stays cheap) |

## Roofline floors (why matched S ⇒ matched bandwidth)

Decode attention reads K and V over the whole cache (fp16): bytes/token = `2·Hkv·Hd·2 = 4096` per
token-position × `Tc`. Floor time = bytes / `960 GB/s`:

| Tc | KV bytes | roofline floor @ 960 GB/s |
|---:|---:|---:|
| 512 | 2.10 MB | 2.18 µs |
| 1024 | 4.19 MB | 4.37 µs |
| 4096 | 16.78 MB | 17.48 µs |

A tile approaches its floor only at ≳4 wg/CU. At S=48 the generated tile is in the same occupancy regime
as the owned tile, so — with the fast primitives present (fdot2 + LDS + cross-lane, proven by the
microgate) — it should reach comparable HBM utilization **by construction**. At S=4 (0.33 wg/CU) it is
request-starved and runs multiples over the floor.

## Decision rule for the generated tile

- **In-model split count: `S = 48` fixed, `L = ceil(Tc/S)` tokens per split** (mirrors `DECODE_ATTN_AMDGCN_S`),
  e.g. `Tc=512 → L=11`, `Tc=4096 → L=86`. Expose it as an env knob (`..._S`, default 48) so the economics
  sweep can confirm the occupancy proxy.
- Keep grid `(Hkv, S)` (GQA reuse), not `(Hq, S)`.
- Acceptance: `tile_occupancy_proxy_wg_per_cu ≈ 4.0` in the attribution economics pre-gate, and the
  combine fraction stays small (the residual term below).

## The residual: the combine (after occupancy is matched)

Once the tile is at 4 wg/CU, the remaining W==D term is the LSE combine that merges the S=48 partials.
The economics tool already characterizes it: latency-bound (`combine_workgroups = 32 ≪ 96 CU`, effective
`GB/s ≪ peak`), "fixable with more parallelism / fusion." It is a bounded second-order cost (split-KV
reduction economics), the same tax the owned route pays — **not** occupancy starvation. Larger S helps the
tile but enlarges the combine, so `S ≈ 48` is the balance the owned route already struck.

## Validation plan (this baseline → next actions)

1. **Sweep (correctness across split granularity):** the microgate sweeps `L` (→ `S = ceil(Tc/L)`),
   recording the occupancy proxy per case, and must stay numerically correct as splits get fine (small L).
   Confirms the layout is split-count-invariant.
2. **Port:** wire the validated tile into `qk_flash_decode.py` (raw `cache_kv` 5D, `S=48`), run the route
   gate + attribution economics pre-gate (expect `wg/CU ≈ 4.0`, `has_v_dot2`/`has_lds` true).
3. **W==D:** only if economics clears; the roofline above is the floor to compare against.

## Outcome (2026-06-26): occupancy is necessary but NOT sufficient

The S=48 tile was run through W==D (`docs/decode-fused-xlane-score-pv-tile-wd-result.md`). At ctx 4096 the
occupancy was near-matched (3.58 wg/CU) yet the route was still 99× slower (~1665× over the roofline per
layer) — **compute-bound on generated ISA, not memory-bound**. Occupancy is a real, necessary lever (this
baseline stands), but the binding constraint is generated-codegen *code quality*, not occupancy or lane
layout. This baseline remains the correct split-count reference for any future tile whose ISA is made
efficient enough to become memory-bound.
