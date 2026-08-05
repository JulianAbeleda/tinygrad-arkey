# NV Q4 cooperative subset precision-budget record — 2026-08-05

## Verdict

The cumulative-depth stop was unnecessarily coarse.  A non-consecutive,
default-off lease of blocks **1–12 and 14–18** passes the full-logit contract
and improves settled wall beyond g12.  Block 13 is the precision boundary.

The selected 17-block subset has relative L2 `8.992186e-4`, exact token stream,
argmax, top-10 set and top-10 order, 34 exact fused providers, 86 cooperative
Q4 consumers and zero legacy shared-Q4 consumers across two captures.  Adding
block 13 fails narrowly at `1.002240e-3`; leasing every block 1–18 is the prior
`1.271441e-3` failure.  Nothing is promoted or enabled by default.

## Search method

The settled base is blocks 1–12.  The six previously bundled additions are
blocks 13–18 (the same six sometimes described as zero-based groups 12–17).
The search used:

1. fresh g0 and g12 full-logit anchors;
2. six fresh `g12 + one block` GPU semantic arms;
3. an exhaustive CPU superposition/ranking of all 64 subsets using their
   signed logit-delta vectors, not scalar error alone;
4. adaptive fresh-GPU validation of four boundary triples, one quad and the
   two decisive five-extra candidates;
5. settled cumulative A/B/A for the passing maximal-cardinality subset; and
6. a direct g12 / subset / g12 incremental bracket to prevent double booking.

Every GPU child held `/tmp/gpu-bench.lock`.  Each semantic arm used eight exact
autoregressive d512 logits and the existing primitive oracle, binding check,
two-capture census and `1e-3` authority threshold.

## Non-monotonic precision result

| candidate on top of g12 | relative L2 | verdict |
|---|---:|:---:|
| block 13 | `9.37254e-4` | PASS |
| block 14 | `8.68353e-4` | PASS |
| block 15 | `8.89663e-4` | PASS |
| block 16 | `6.82152e-4` | PASS |
| block 17 | `8.69234e-4` | PASS |
| block 18 | `1.05828e-3` | FAIL |
| blocks 14,15,16 | `7.45900e-4` | PASS |
| blocks 14,15,16,17 | `7.16702e-4` | PASS |
| blocks 14,15,16,17,18 | `8.99219e-4` | **PASS** |
| blocks 13,14,15,16,17 | `1.00224e-3` | FAIL |

Block 16 reduces the total perturbation even though it introduces another Q8
boundary.  Errors therefore cancel through later transformer blocks; precision
budget is not monotonic in leased depth and cannot be allocated by per-block
norm or cumulative prefix alone.

The compact semantic artifact is
`docs/task_workflow/output/nv-q4-cooperative-subset-precision-search-20260805.json`.

## Settled wall

The maximal passing subset ran the corrected protocol: six untimed decode
calls, then five uninterrupted 32-token windows per fresh arm, reverse A/B/A,
with no rejected high-side samples and one exact 160-token stream hash.

| arm | median ms/token |
|---|---:|
| unleased control A | `5.415914625` |
| maximal subset | `5.393325000` |
| unleased control C | `5.428464844` |
| control midpoint | `5.422189734` |

Cumulative recovery is `28.864734 us/token` (`+0.5352%`).  Because the prior
ledger already books g12, this cumulative row must not be added to it.

The decisive incremental bracket was g12 / maximal subset / g12:

| arm | median ms/token |
|---|---:|
| g12 A | `5.396238125` |
| maximal subset | `5.387827313` |
| g12 C | `5.404340594` |
| g12 midpoint | `5.400289359` |

The five added blocks recover **`12.462047 us/token`** beyond g12
(`+0.2313%`), with identical 160-token streams and no rejected samples.  This
is the only new causal-ledger credit.  Artifacts:

- `docs/task_workflow/output/nv-q4-cooperative-maximal-subset-settled-timing-20260805.json`
- `docs/task_workflow/output/nv-q4-cooperative-g12-vs-maximal-incremental-timing-20260805.json`

## Disposition

- Keep the explicit subset research-only and default-off.
- Book `12.462047 us/token` incremental recovery beyond the existing g12 row;
  do not also book the `28.864734 us/token` cumulative row.
- Do not run g35: the full 1–18 set remains a semantic failure.
- Any broader search must preserve explicit block identities.  A scalar
  `fused_groups=N` prefix is no longer an adequate precision-policy model.
