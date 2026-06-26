# Decode Attention Online-Softmax+PV Tile P2 Result

## Verdict

`ONLINE_PV_TILE_STRUCTURAL_ROUTE_CLEAN`

P2 adds the first structural generated/search-owned route identity for the primitive-complete online-softmax+PV tile path.

This is not a speed promotion gate. It proves route binding, materialization hygiene, and token-sample correctness before P3/P4 lane/reduction/dot work.

## Candidate

- Candidate id: `decode_attention_online_pv_tile_structural_p2`
- Flag: `DECODE_ATTN_ONLINE_PV_TILE=1`
- Program: `flash_online_pv_tile_whole_cache_32_128`
- Manifest: `bench/qk-search-spaces/decode_attention_online_softmax_pv_tile_v1.json`
- Artifact: `bench/qk-decode-attention-online-pv-tile/latest.json`

## Command

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_attention_online_pv_tile_gate.py
```

## Gate Result

| Check | Result |
|---|---|
| owned tile absent | pass |
| owned combine absent | pass |
| `E_49152` absent | pass |
| selected-route buffer identity | pass |
| token sample matches owned baseline | pass |
| `flash_online_pv_tile_whole_cache_32_128` present | pass |
| stale A3.10 `flash_tile_prob_partial_pv_whole_cache_*` absent | pass |
| old `flash_prob_32` absent | pass |
| old `flash_partial_coop_vec_whole_cache_*` absent | pass |
| score + online-PV tile + global metadata + combine lifecycle signature present | pass |

## Captured Generated Attention Signature

| Program | Status |
|---|---|
| `flash_score_whole_cache_32_128` | present |
| `flash_max_32` | present |
| `flash_online_pv_tile_whole_cache_32_128` | present |
| `flash_gmax_32` | present |
| `flash_den_32` | present |
| `flash_combine_32_128` | present |
| `owned_flash_tile_gqa_whole` | absent |
| `owned_flash_combine` | absent |
| `E_49152` | absent |

Token sample matched owned baseline:

```text
[315, 24231, 6009, 979, 220, 576]
```

## Interpretation

P2 is complete.

The generated route now has a distinct primitive-complete-path program identity. It is still a structural shell: it uses the external per-split max path and does not yet prove lane-owned register online-softmax state, packed-dot transfer, or cross-lane reduction transfer.

Do not promote this route for speed.

## Next Step

Proceed to P3: lane ownership and reduction mapping.

P3 must answer:

| Question | Required evidence |
|---|---|
| Which lanes own head, split, GQA group, and D? | explicit lane-map artifact |
| Where do `m`, `l`, and `acc[D]` live? | structural route/resource report |
| Is cross-lane reduction emitted or blocked? | source/ISA attribution or precise `SEARCH_BLOCKED_BY_CODEGEN` |
| Does the route preserve T=1 split-KV parallelism? | workgroup/lane report |
