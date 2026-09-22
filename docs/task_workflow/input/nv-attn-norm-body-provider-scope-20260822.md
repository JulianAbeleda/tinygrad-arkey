# NV attn norm body provider-preserving scope (2026-08-22)

Status: implementation scope; standalone attention norm topology only. The 17
shared norm+Q8 providers are a frozen control.

1. Row and edge: `reduce_output_rmsnorm_1_4096` (19 x 7.008 us). Edge:
   attention input -> norm -> Q/K/V projection.
2. Dominant term: BODY. 4.864 us body versus llama attn_norm 2.752 us
   (block [32,16,1]); D 0.474, R 1.670.
3. Code paths: standalone attn norm topology (1024-thread reduce+normalize
   like llama); no provider-path change.
4. Legality: generic RMSNorm over 4096 fp16; target-derived shape only.
5. Fallback: fold normalize into the same kernel; preserve the 17 provider
   nodes untouched so the -114.43 us quant advantage is not reintroduced as
   separate work.
6. Contract: exact norm output; token SHA identical.
7. Arms: isolated = B/C retained (`phase9/`); installed = reverse bracket.
8. Census gate: 19 nodes unchanged or fewer; provider count stays 17.
9. Reverse wall bracket, +50 us promotion bar.
10. Rollback: revert norm; non-regression on non-NV targets.
11. Projected ceiling 40.1 us (net +11.0 us if the provider advantage is
    lost), labelled unmeasured.
12. Prohibited: model-name or block-list dispatch.
