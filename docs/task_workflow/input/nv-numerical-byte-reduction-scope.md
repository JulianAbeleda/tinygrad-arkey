# NV numerical weight-byte reduction campaign scope

## Objective

Determine whether selected Q6_K decode weights can be represented as Q4_K to
reduce compulsory DRAM payload and improve token wall under an explicit quality
contract.  The source GGUF remains unchanged; experimental packed sidecars are
held only by the candidate process.

## Funnel

1. Real-weight feasibility: reference-dequantize Q6_K and reference-quantize
   Q4_K, with exact packed-byte accounting.
2. Projection quality: measure relative L2, cosine similarity, max error, and
   finite outputs across deterministic real-model activations.
3. Primitive device value: compare direct Q6_K and Q4_K kernel execution.
4. Whole-model quality: full-logit relative L2 at most `1e-3`, finite logits,
   deterministic greedy-token agreement, and a bounded corpus dNLL/perplexity
   regression gate.
5. Whole-token value: fresh-process control/candidate/control wall bracket.

The local projection gate (`relative_l2 <= 0.05`, `cosine >= 0.999`) is only a
feasibility filter.  It cannot promote a numerical model change.  Attention V
is tested first; Q6 FFN-down advances only if the representation and execution
substrate pass.
