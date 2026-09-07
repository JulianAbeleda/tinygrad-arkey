# NV pp512 CTA-local Q-RoPE staging seam (2026-09-07)

A typed experimental stage allocated four non-overlapping 2048-half Q tile slots (8192 halves / 16 KiB) alongside the retained 4 KiB K stage. Each of 128 threads wrote 64 values, covering all 8192 positions exactly once. Each warp selected its own slot, Q head/tile from the production group mapping, and paired d/d+64 with the fp32 `[512,128]` cos/sin table. One barrier published the stage before the KV RANGE; existing shared fragment lowering consumed the fp16 tile inside the loop.

Fresh-cache isolated production-shape numerical probes were bit-exact against explicit E2048 RoPE+cast followed by current Flash (`max_abs=0`, `mean_abs=0`, finite). Unlike direct-load fusion, isolated timing was neutral rather than showing a multi-millisecond resource regression.

Full graph qualification did not reach execution. Two verifier failures exposed call-substitution's weakint `lidx0` form; after exact verifier admission, the attempted local reconstruction of an int32 thread id rendered a duplicate `lidx0` declaration in NVRTC. The correct continuation is to preserve and cast the builder's existing lane/warp provenance through the typed stage, rather than create a second SPECIAL. Per the bounded retry rule, all compiler/model edits are reverted here. No performance or correctness claim is made for the full graph.
