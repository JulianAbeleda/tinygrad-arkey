# Post-Owned-Attention Holistic Primitive Audit — Result (2026-06-23)

## 1. Verdict: **POST_DEFAULT_AUDIT_COMPLETE** — decode bottleneck has shifted off attention/GEMV

This is a **DEFAULT audit** (`AUTHORITY_LOCK_DEFAULT`): owned AMDGCN attention is default-on (flipped), Q4K_GEMV_WARP
forced via env (default-eligible, default-off). Headline: **weight-GEMV is at llama parity** and **attention is
near-parity** — the decode bottleneck moved to the **KV-copy materialization** and **unfused small-ops/activation**.
tinygrad is now **~85–88% of llama @ctx1024**. Next bounded lever = **runtime-KV (needs a fresh diagnostic)**;
ISA audit confirmed the owned tile emits every intended primitive with no spills.

## 2. Authority / config
HEAD `a7d30052`. gfx1100, Qwen3-8B-Q4_K_M. Owned attention **default-on** (commit ecdefc86e; gated
gfx1100/Qwen3-8B/B=1/T=1/Hq32/Hkv8/Hd128/ctx≥512; fp16 cache; `DECODE_ATTN_AMDGCN_TILE=0` disables). Q4K_GEMV_WARP
**default-off** (env-forced where noted). llama oracle reused unchanged (rocprofv3 b9592). (`authority.json`.)

## 3–4. Current tinygrad vs llama tok/s + % of llama (`wd.json`, canonical harness, real decode `tok_s_W`)
| ctx | gqa (TILE=0) | owned default | Δ | llama-bench | tinygrad % of llama |
|---|---|---|---|---|---|
| 512 | 76.1 | **85.8** | +12.7% | ~97.7 | ~88% |
| 1024 | 74.0 | **85.4** | +15.4% | ~97.4 | **~88%** |
| 2048 | 71.0 | **84.3** | +18.7% | — | — |
| 4096 | 67.1 | **82.1** | +22.4% | ~92.4 | ~89% |
Route fires by default (owned cache fp16, nodes>0), falls back at ctx<512 / TILE=0 / unsupported shape. Byte-identical
incl. short-ctx SDPA. `POST_DEFAULT_WD_CONFIRMED`.

## 5. W==D confirmation
Confirmed on shipped code (above). No regression, no accidental gqa fallback, fp16 cache confirmed.

## 6. Time-tax diff (`time_tax.json`, GPU-busy µs/token @ctx1024; `TIME_TAX_DIFF_REUSED_LLAMA_ORACLE`)
| lane | tinygrad | llama | gap µs | note |
|---|---|---|---|---|
| **weight GEMV** (FFN+proj+lm_head) | 7620 | 7686 | **−66** | **PARITY** — Q4K_GEMV_WARP closed it |
| **attention compute** | 663 | 507 | +156 | **NEAR-PARITY** — owned tile |
| **KV-cache copy** (E_49152) | 1545 | 204 | **+1341** | full-MAXC materialization; gqa pays it too |
| norm/rope/small (genuine) | 2345 | 1062 | +1283 | unfused small kernels |
| FFN activation (silu) | 1540 | 464 | +1076 | unfused vs llama epilogue-fused |

tinygrad GPU-busy sum ~13.7ms > wall 11.7ms (~2ms overlap); llama serial ~9.97ms. **Wall gap +1.73ms/token.**

## 7. Corrected bucket map (`corrected_buckets.json`, `POST_DEFAULT_BUCKETS_CORRECTED`)
Re-rendered kernels reclassified the heuristic `norm_rope_small_ops` (28%): it **mislabels the KV copy** —
`E_49152_32_3`(758µs)+`E_49152_32_3n1`(787µs) = the full-MAXC store materialization, NOT norm/rope. After
reclassification: FFN-GEMV parity, attention near-parity, and the residual gap is **KV-copy (1.3ms) + unfused
small-ops (1.2ms) + unfused activation (1.0ms)**.

## 8. Holistic primitive lifecycle (`primitive_lifecycle.json`)
| lane | blocker |
|---|---|
| KV-cache copy | `RUNTIME_GRAPH_LIFECYCLE_GAP` (materialization coupled to @function persistence) |
| norm/rope/small-ops | `ISA_CODEGEN_GAP` (many unfused kernels) |
| FFN activation | `ISA_CODEGEN_GAP` (no GEMV-epilogue fusion) |
| weight GEMV | `ALGORITHM_NOT_WORTH_IT` (parity) |
| attention compute | `WD_TRANSFER_REFUTED` (already won; residual 156µs not worth it) |

## 9. ISA / code-object audit (`isa_primitive_audit.json`, `ISA_PRIMITIVES_CONFIRMED`)
llvm-objdump/readelf on the hipcc code objects:
- **`owned_flash_tile_gqa`**: 56 VGPR, 26 SGPR, **8KB LDS**, **0 scratch (no spills)**, kernarg 44B. Flags:
  `has_v_dot2`✓ (v_dot2×2), `has_lds`✓ (ds_store×22/ds_load×1), `has_cross_lane`✓ (ds_bpermute×5),
  `has_vector_global_load`✓ (global_load×23), `has_spill`✗. **Intent matches ISA** — fdot2, LDS staging, warp
  reduce all emitted as designed.
- **`owned_flash_combine`**: 26 VGPR, 0 LDS, 0 scratch, fp32 fma×8 + exp×4 (log-sum-exp), global_load×4 — confirms
  the latency-bound, under-occupied combine.
- Tooling works (`/opt/rocm/llvm/bin/{llvm-objdump,llvm-readelf}` on `/tmp/b4_*.co`) — an ISA-audit tool is feasible.

## 10. Stacking (`stacking.json`, `STACKING_CONFIRMED`)
@ctx1024: q4k-only 74.3, owned-only 76.9, **both 86.2** (additive — FFN-GEMV and attention are disjoint lanes).
@ctx4096: q4k 67.4, owned 74.4, **both 82.7**. Caveat: ctx1024 `gqa_no_q4k=34.4` is a cold-clock/non-warp-path
anomaly (canonical default ~66–74) — used only as a lower bound; the additive conclusion holds on the warm configs.

## 11. Runtime-KV decision (`runtime_kv_decision.json`, `RUNTIME_KV_NEEDS_NEW_DIAGNOSTIC`)
The full-MAXC copy (E_49152 ~1.5ms) is **still present** (FO2 removed the cast, not the store materialization; gqa
pays it too). Gqa-era MAXC-shrink showed +1.5ms/+8 tok/s, but the owned route changed the overlap and the prior
"opaque-append NaN" was entangled with the now-fixed owned-tile dtype bug. **Run two cheap diagnostics first**:
(a) MAXC-shrink A/B on the owned route to confirm ≥5% wall impact post-FO2; (b) re-test the opaque append + fixed
owned tile + fp16 cache for byte-identical multi-step decode. Reopen only if both favorable.

## 12. Next primitive ranking (`next_primitive.json`, `NEXT_PRIMITIVE_RUNTIME_KV`)
| rank | lane | gap ms | blocker | exp. W==D | conf | bounded | first gate |
|---|---|---|---|---|---|---|---|
| 1 | runtime-KV / KV-copy | 1.3 | RUNTIME_GRAPH_LIFECYCLE | ~+5–9% | MED | yes | MAXC-shrink A/B + opaque-append re-diag |
| 2 | small-ops/activation fusion | 2.3 | ISA_CODEGEN | uncertain (overlap) | LOW-MED | no (broad) | prototype 1 fusion, check W==D |
| 3 | attn q/o-proj → warp GEMV | 0.1 | ALGORITHM_NOT_WORTH_IT | marginal | LOW | yes | only if cheap |
| 4 | ISA-audit tooling | 0 | NEEDS_PRIMITIVE_AUDIT | enabling | HIGH | yes | package the proven flow |

Runtime-KV is #1 (most bounded + on critical path + owned-tile blocker now fixed). Small-ops fusion is a bigger
GPU-busy gap but heavily overlapped and codegen-unbounded. **No `NO_BOUNDED_8B_PRIMITIVE_REMAINS`** — but the project
is near llama parity with attention+GEMV solved; **generalizing the promoted routes** to other models/shapes
(`NEXT_PROJECT_GENERALIZE`) is a reasonable strategic alternative to more 8B grinding.

## 13. Project synthesis update
Supersedes the "attention exhausted / runtime-KV next / B4 W==D fail" framing **and** the prior tinygrad-vs-llama
gap maps. New reality: **attention + weight-GEMV are at/near llama parity**; the residual ~12% is KV-copy lifecycle +
small-op/activation fusion. Updated `docs/README.md` + `session-handoff.md` (superseding notes only; history intact).

## 14. Artifacts and commands
- `bench/qk-post-owned-attention-default-audit/{authority,wd,time_tax,corrected_buckets,primitive_lifecycle,isa_primitive_audit,stacking,runtime_kv_decision,next_primitive}.json`.
- W==D: `... QK_CKPTS=512,1024,2048,4096 .venv/bin/python extra/qk_decode_runtime_overhead.py` (default) vs `+ DECODE_ATTN_AMDGCN_TILE=0`.
- Buckets: `DEV=AMD JIT=1 DECODE_ATTN_AMDGCN_TILE=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 .venv/bin/python extra/qk_decode_time_tax_audit.py`.
- ISA: `clang-offload-bundler --unbundle` + `/opt/rocm/llvm/bin/{llvm-objdump -d, llvm-readelf --notes}` on `/tmp/b4_tile_s47_*.co`.

## 15. Files changed
Audit-only (no implementation). New: this result doc + 9 bench artifacts. Updated: `docs/README.md`,
`structure/Development/session-handoff.md`. No source/default changes.

## 16. Working tree status
No source or default changes (audit only). New audit artifacts + result doc + doc updates. No 14B/32B, no
runtime-KV implementation, no new kernels, no default flips.
