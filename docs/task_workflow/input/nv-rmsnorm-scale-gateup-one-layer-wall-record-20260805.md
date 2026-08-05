# NV scale-only RMSNorm -> Q4 gate/up: one-layer wall record

## Decision

**WALL_NO_GO.** Do not expand the explicit lease from block 0 to 4/8/18/36.

The raw-input scale-only consumer has a real isolated included-cost advantage,
but it is slower in the settled production token graph. This record closes the
construction at its first required production wall gate; it does not make a
claim about a general RMSNorm lowering or change any model-load policy.

## Scope and construction

The default-off lease owns the exact NV decode FFN shape only:

* raw residual `h`, shape `(1,1,4096)`;
* scalar `rsqrt(sum(h^2)/4096 + eps)`;
* fp16 RMSNorm weight; and
* the existing Q4_K W1/W3 `(12288,4096)` consumer.

The consumer rounds `fp16(h * scale)` and then `fp16(* weight)` at every
packed-Q4 load, and sends the resulting 12288 gate/up activation through the
ordinary `ffn_down`. Normal model loads do not create the lease.

## Preconditions passed

* Identity affine (`scale=1`, `weight=1`) is bit-exact against the existing
  packed-Q4 W1/W3 consumer.
* The isolated production-shape included-cost gate measured 86.6772 us versus
  a 93.10635 us A/A midpoint: **-6.42915 us**.
* A d512, four-token block-0 full-logit lease preserved sampled tokens,
  argmax, and ordered top-10. Relative L2 was `0.000507516`, within the
  `1e-3` qualification limit.

## Settled wall A/B/A

All arms used fresh processes, DEV=NV, d512, max-context 1024, the same
composed decode contract, six unmeasured warmup tokens, five continuous
16-token windows (80 timed tokens), and the same token-stream SHA256
`db615a6ca0c48eb7a24978182128c6a5eebff7a650e89173bcf2a63ffe1f6c6e`.

| Arm | Median ms/token |
| --- | ---: |
| A: no lease | 5.415789625 |
| B: block-0 lease | 5.4275018125 |
| A2: no lease | 5.422475125 |

Baseline midpoint: `5.419132375 ms/token`.

Candidate delta: **+0.0083694375 ms/token (+8.369 us/token)**.

## Interpretation

The 6.43-us standalone saving does not survive the full graph; the admitted
one-layer construction is 8.37 us/token slower. Possible graph-level causes
(extra raw/weight/scalar dependencies, scheduling, or loss of an existing
RMSNorm materialization advantage) remain unseparated, but are not a reason
to promote or scale a wall regression. Any revival needs a new construction
and a fresh one-layer A/B/A gate.

## Artifacts

* `/tmp/nv-rmsnorm-scale-gateup-timing-20260805.json`
* `/tmp/nv-rmsnorm-scale-gateup-correctness-20260805.json`
* `/tmp/nv-rmsnorm-scale-gateup-one-layer-baseline-20260805.{json,npz}`
* `/tmp/nv-rmsnorm-scale-gateup-one-layer-lease0-count4-20260805.{json,npz}`
* `/tmp/nv-rmsnorm-scale-gateup-wall-{a,b,a2}-20260805.json`
