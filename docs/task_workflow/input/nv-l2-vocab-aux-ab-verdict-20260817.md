# NV L2 vocab_aux: fresh A/B verdict (2026-08-17)

Date: 2026-08-17. Branch `nvidia-bringup-20260731`, HEAD `2e270c963`.
GPU: RTX 5090 (idle, lock held by harness children). Harness:
`extra/llm_research/decode/nv_vocab_top1_fusion_ab.py` (fresh-process A/B,
control vs `_decode_vocab_top1_lease`). Evidence:
`/tmp/nv_l2_vocab_ab_20260817.json`, `/tmp/l2_arm_vocab_timing.py` captures.

Status: **NO-GO, recorded honestly. The fused route is bit-exact and passes
the old topology gate, but the wall is neutral (-1.55 us) and the cost model
is CONTRADICTED. The gate itself was stale: the lease route does not remove
the argmax tail, it RENAMES it (4 kernels -> 2). The stale-prefix gate is
fixed in this record.**

## 1. A/B result (fresh, both arms, token sha identical)

| arm | median ms/token | wall delta |
| --- | ---: | ---: |
| control (legacy 4-kernel tail) | 4.808820 | - |
| candidate (vocab_top1 lease) | 4.807268 | **-1.55 us** |

- token streams equal: **yes** (sha `7e2ff56b...` both arms)
- timing token hashes equal: yes
- verdict: WALL_PASS (candidate < control) but by 1.55 us, not the predicted
  -50 us; cost gate result **FAIL / CONTRADICTED** (gap 48.4 us)

## 2. Why the wall is neutral: the tail was renamed, not removed

Per-arm vocab kernel captures (DEBUG=2, one token):

| control kernels | us | candidate kernels | us |
| --- | ---: | --- | ---: |
| `q6k_gen_coop_151936_4096_inkernel` | 322.18 | `..._epi_vocabtop1` | 320.67 |
| `E_1187_32_4` (x2) | 2.98 + 4.16 | `E_1187_16_4` | 2.43 |
| `r_32_4_1187` | 39.49 | `r_16_4_1187` | 41.09 |
| `r_128_16_8_1187` | 11.33 | - | - |
| `r_16_8` | 2.08 | - | - |
| tail total | 60.04 | tail total | 43.52 |

The scheduler lowers `packed_argmax_from_tile_keys` into its own elementwise +
reduce pair (`E_1187_16_4` + `r_16_4_1187`). So the "fused" route removes
~16.5 us of tail node-sum, and the wall moves 1.55 us - the tail was already
largely hidden behind the GEMV anchor. The old gate checked only
`E_1187_32_4` / `r_32_4_1187` / `r_128_16_8_1187` / `r_16_8` prefixes and
missed the renamed pair.

## 3. Ledger correction

The ledger row "L2 vocab_aux +59.5 us node, +211.5 tok/s ceiling at 1:1" was
an upper bound that assumed the tail is on the wall. The A/B proves the
transfer is ~10%: removing ~16.5 us of node moves the wall 1.55 us. The tail
is hidden mass, not wall mass - same lesson as L5 (node_sum != wall when
hidden). L2's real ceiling is ~2-3 us of wall; the row should be closed or
re-scoped as hidden mass, not promoted on the 59.5 us ledger figure.

## 4. Gate fix (committed with this record)

`nv_vocab_top1_fusion_ab.py` now counts a `tail_family` set (any E_/r_ kernel
containing a 1187 vocab extent, or `r_16_8`) and fails the lease arm if any
tail-family kernel survives. The renamed pair (`E_1187_16_4`,
`r_16_4_1187`) is now caught: lease child fails topology with
`tail_family_program_count: 2`, proving the fix bites.

## 5. What this means

1. L2 does not clear the +50 us bar; no promotion. The fused route as
   implemented is not worth shipping on its own (bit-exact but wall-neutral).
2. To make L2 real, the cross-tile argmax reduce must run IN the GEMV
   epilogue (no second kernel). The custom `emit_q6k_vocab_top1_reduce_kernel`
   was tried and cost ~0.89 ms/token; the scheduler reduce is cheaper but
   still a separate kernel. Both are recorded; neither is a wall win today.
3. The ledger's L1+L2 +17.6 tok/s claim must drop L2: L1 remains the only
   buildable wall row (fresh 384.1 us node).
