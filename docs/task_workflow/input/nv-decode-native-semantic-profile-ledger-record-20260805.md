# Native NV d512 semantic device-window ledger

Date: 2026-08-05. Route: `DEV=NV`; Qwen3-8B-Q4_K_M; d512; RTX 5090 / driver
595.84. Status: **PASS — the native-versus-llama graph-device delta is
reconciled to 1.143 us, with a disjoint top-level semantic partition.**

## Result

The native marker-light device window is 5291.424 us. The independent llama
unprofiled graph span is 3889.808 us, so the authority device delta is
1401.616 us.

Three `PROFILE=1 HCQ_GRAPH_PROFILE_JSON` captures each reproduced the settled
five native groups `32/64/128/256/468 = 948`. Their serialized timestamp sums
were 5327.392, 5375.968, and 5377.312 us. Because per-node timestamps perturb
the route, they are used only for composition. The median-total capture is
scaled by `5291.424 / 5375.968 = 0.984273716` to the independent native window.

| disjoint device population | calibrated native us | llama us | delta us |
| --- | ---: | ---: | ---: |
| non-quantized serialized work vs aggregate exposed non-MMQ union | 1408.818 | 300.736 | **+1108.082** |
| quantized cores vs MMQ union | 3882.604 | 3579.816 | **+302.788** |
| llama internal graph gaps | 0.000 | 8.111 | **-8.111** |
| **profiled-term equation** | | | **+1402.759** |

The equation differs from the unprofiled authority delta by **1.143 us**,
well inside `max(50 us, 2%)`. The 1.143 us is expected from independently
medianed profiled llama terms versus the unprofiled llama span; it is not an
unassigned kernel bucket.

This changes the priority order. About 1.108 ms of the 1.402 ms device deficit
is not the quantized core itself: it is native serialized elementwise,
reduction, and attention support work that llama mostly overlaps behind MMQ.
The remaining aggregate quantized-core deficit is about 0.303 ms. Across all
three profile compositions that calibrated quantized total ranges from
3882.517 to 3906.375 us, so the conclusion and rank are insensitive to the
profile replicate.

## Native composition

The admission observer attached exact program identities to all 217 native
quantized calls. This closes layer, tensor, role, shape, and quant-storage
identity for Q/K/V/O, fused gate/up, down, and vocab. The remaining 731 calls
have exact kernel identities and are partitioned without overlap:

| native class | calibrated us |
| --- | ---: |
| elementwise | 653.999 |
| reductions | 449.238 |
| flash score | 204.697 |
| flash combine | 100.884 |
| **non-quantized total** | **1408.818** |

Do not subtract individual llama class exposure rows from these four rows:
llama's individual exposure rows are non-additive. Only its aggregate exposed
non-MMQ union, 300.736 us, is valid in the wall equation. A finer causal split
requires disjoint interval ownership or real family A/Bs.

The quantized-role diagnostic ranks the positive candidates:

| ordered role pairing | native us | llama us | diagnostic delta us |
| --- | ---: | ---: | ---: |
| native attention V / llama V 1024 projection | 382.465 | 165.053 | **+217.412** |
| FFN down | 1045.094 | 867.076 | **+178.018** |
| native attention K / llama K 1024 projection | 152.381 | 117.376 | +35.005 |
| vocab | 314.432 | 303.618 | +10.814 |
| fused gate/up | 1360.219 | 1364.038 | -3.819 |
| attention Q | 316.227 | 342.881 | -26.654 |
| attention O | 311.786 | 418.401 | -106.615 |

These role rows are independently medianed and therefore do not sum exactly
to the aggregate MMQ row. The exact role mapping is now closed: Qwen3 builds
Q/K/V but `build_attn` intentionally expands Q/V/K, and states that the nodes
are kept in that order so K can be expanded later for RoPE-to-KV-cache fusion.
Thus, every captured layer's first 1024 MMVQ is V and its second is K. The
source proof and complete 36-layer census are in
`nv-decode-llama-kv-role-mapping-record-20260805.md`.

The attention-V and FFN-down rows identify the positive quantized populations.
The earlier 18-member Q6_K family graph A/B remains a family-level CUDA
diagnostic, not evidence assigning its recovery to K or V alone. Fused
gate/up is not the current aggregate core blocker.

## Method and artifacts

`full_token_dag_capture.py` now serializes metadata already present in its
graph-admission observer. This is an attribution-only correction: it does not
change graph construction, programs, dependencies, or dispatch. The raw HCQ
profile exporter still emitted null metadata, so the parser aligns its entries
to the observer DAG by exact ordinal and exact kernel name and fails closed on
any topology or identity mismatch.

The durable parser is
`extra/llm_research/decode/native_semantic_profile_ledger.py`; the compact
payload is
`docs/task_workflow/output/nv-decode-native-semantic-profile-ledger-20260805.json`.
The three raw captures and observer DAG remain under `/tmp`; their hashes are
embedded in the compact payload. The llama trace and semantic manifest hashes
are also embedded.

Validation:

```text
PYTHONPATH=. .venv/bin/pytest -q \
  test/unit/test_native_semantic_profile_ledger.py \
  test/unit/test_full_token_dag_capture.py
9 passed
```

No production route, promotion, graph schedule, or default changed.
