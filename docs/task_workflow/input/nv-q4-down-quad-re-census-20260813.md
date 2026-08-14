# NV Q4 FFN-down quad-u128-smem re-census - in-loop NO-GO, tokens exact (2026-08-13)

Date: 2026-08-13
Branch: `nvidia-bringup-20260731` (HEAD `0f2f83a07`)
Status: **measurement record.** Reopens and re-closes the Q4 FFN-down load-pattern row after the
authoritative floor correction (`nv-gemv-core-deficit-correction-20260813.md`): llama Q4 FFN-down
floor **19.23 us/node** (346.209 us / 18 nodes), installed control **26.75 us** in-loop. The
standalone winner `q4kd_16row_128thr_u128_quad_xsmem` (11.43 us standalone) is measured in the real
d512 decode loop here. Verdict: **NO-GO in-loop** (34.48 us median, +8.2 us/node over control),
production untouched. Token stream is bit-identical across all arms (the only gate that passed).

## 1. The question

The 08-12 sweep closed this row NO-GO against a copy-pasted wrong floor (11.776 us/node, which is
attention-O's value). With the corrected llama floor (19.23 us/node), the standalone quad-u128-smem
row (11.43 us) clears the standalone-equivalent target (~13.3 us at this shape's 1.44x offset) and
the row reopens. The MC2 lesson says standalone is not evidence: the gate/up quad won standalone
(22.2 vs 23.2 us) but regressed in-loop (49.2 us, -5% wall). Only the in-loop census counts.

## 2. Implementation (research-only, closed by default)

- `q4k_g3_lanemap_gemv_kernel(rows, k, lanes, epilogue, load_style="scalar")` in
  `tinygrad/llm/decode_kernels.py`: `load_style="quad"` emits the single-projection quad-u128-smem
  geometry adapted from the w1w3 quad (16 rows/block x 8 lanes/row, pure uint4 weight loads, x
  staged to shared memory once per launch and read in-loop as uint4, 3-step XOR ladder 4/2/1 over
  the 8 row lanes, epilogue adds `extra[0][row]`) under the new name
  `q4k_g3_lanemap_gemv_quad_epi_ffnresadd_4096_12288`. Default `scalar` is byte-identical to the
  installed kernel (verified: identical 583-uop topo dump vs HEAD).
- `decode_routes.py`: a harness attaches `Q4KFFNDownQuadAdmission` to specific leased Q4_K FFN-down
  linears (4096x12288, role `ffn_down`); the quad spelling is used only when that marker AND the
  m2b `ffn_down_resadd` epilogue admission are both present. Normal model loads have no marker, so
  production runs the installed scalar path unchanged.
- Census harness: `extra/llm_research/decode/nv_q4_down_quad_census.py` (d512, real
  `model.generate` loop, per-kernel DEBUG=2 census under `JIT=2` reproducing the house per-kernel
  regime, settled reverse wall bracket under `JIT=1` production CUDA-graph decode).

## 3. Protocol

Qwen3-8B-Q4_K_M, d512 (`[1]*512` house census prompt), max_context 1024. Each arm is one fresh
process under `flock -w 90 /tmp/gpu-bench.lock`. Wall phase: prelude + 6 warmup tokens, then 3
settled windows of 20 tokens, median + MAD high-side filter, token sha256 + first token pinned.
Census phase (after resetting all model JITs): prelude + 6 warmup, then 3 DEBUG=2 census tokens,
per-kernel medians for `q4k_g3_lanemap_gemv_quad_epi_ffnresadd_4096_12288` (candidate) and
`q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288` (installed control). Lease set: blocks
4, 8, 13, 19, 25, 29 (6 of the 18 Q4 FFN-down blocks). Bracket order: control-a, candidate,
control-b.

## 4. Measured numbers

| arm | census control us (n) | census quad us (n) | wall ms/token | token sha256 (first) |
| --- | ---: | ---: | ---: | --- |
| control-a | 26.285 (54) | - | 5.1564 | `37e9538d...` (5708) |
| candidate | 26.24 (36) | **34.48** (18, min 33.82, max 35.04) | 5.1975 | `37e9538d...` (5708) |
| control-b | 26.24 (54) | - | 5.1620 | `37e9538d...` (5708) |

Full sha256: `37e9538d8403dd464ebe3823c115b094c93982c2308ea45892eae6f8eb6819a7`, first token 5708,
identical in all three arms (census-phase tokens 5708/198/474 and prelude 82 identical everywhere).

## 5. Gate results

| gate | requirement | measured | result |
| --- | --- | --- | --- |
| per-kernel floor | quad in-loop median < 19.23 us/node | 34.48 us | **FAIL** |
| per-kernel control | quad in-loop median < 26.75 us (and < measured 26.24-26.29) | 34.48 us | **FAIL** |
| token pins | token sha256 + first token identical to control | identical (all arms) | PASS |
| reverse wall bracket | candidate < control, token hash equal | +38.3 us/token (5.1975 vs 5.1592) | **FAIL** |

Verdict: **NO-GO**. The quad geometry is token-exact but regresses in-loop: 34.48 us median vs the
installed 26.24-26.29 us, and the wall moves +38.3 us/token on the 6 leased blocks. This is the
same phenomenon as the MC2 gate/up quad (standalone win, in-loop regression): the smem-staged
16-row/8-lane shape wins standalone (11.43 us, L2-resident microbench) but loses in the real loop.
Standalone-to-in-loop offset is ~3.0x for the quad (11.43 -> 34.48) vs ~1.4x for the scalar control
(18.5 -> 26.3), consistent with the in-loop smem/occupancy penalty the MC2 record documented.

## 6. Consequence

The Q4 FFN-down load-pattern row closes NO-GO again, now against the correct floor. The quad
emitter stays in the tree as research-only (admission-gated, production byte-identical); it is NOT
promoted and no policy record is touched. The +302.8 us core deficit remains open via the DP4A
substrate path (producer-owned Q8_1 successor, `nv-q4k-ffn-down-mmvq-included-cost-and-one-layer-
record-20260805.md` section 4), not the load-pattern row.

## 7. Evidence

- `extra/llm_research/decode/nv-q4-down-quad-re-census-20260813.json` (bracket verdict + gates)
- `extra/llm_research/decode/nv-q4-down-quad-re-census-20260813/control-a.json`
- `extra/llm_research/decode/nv-q4-down-quad-re-census-20260813/candidate.json`
- `extra/llm_research/decode/nv-q4-down-quad-re-census-20260813/control-b.json`
- `extra/llm_research/decode/nv_q4_down_quad_census.py` (reproducible harness)
