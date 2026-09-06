# Current252 internal-buffer deep replay repair

The deep replay collector assumed every generated projection used the plain three-buffer ABI `(outs=(0,), ins=(1,2))`. Gate/up Stream-K uses five buffers with `(outs=(0,1,2), ins=(3,4))`, where output is slot 0 and the Q8 record is slot 4. The collector now handles both generated contracts and reports unexpected alternatives explicitly.

The repaired current252 run passes exact replay across two cycles for gate/up outputs and shared pair records, K, Q/O, Q4 V, Q6 V, all KV state, logits, and token 198. It observes 72 unique gate/up outputs with 36 unique pair-reused records, 36 K outputs/records, 72 Q/O outputs/records, and 18 outputs/records for both V families. Down identities are not yet present in this collector's stage mapping and remain an explicit evidence gap rather than a vacuous qualification claim.

A follow-up maps generated Q4 and Q6 down mains by their exact selected symbols. The resulting run observes 18 unique records and outputs for each down family and passes exact replay for both, closing the earlier explicit stage-census gap.
