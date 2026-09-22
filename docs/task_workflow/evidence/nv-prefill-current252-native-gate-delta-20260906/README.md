# Current252 native gate/up role delta

The current252 harness now admits a diagnostic-only native gate/up substitution
while retaining the generated K, Q/O, V, down, Flash, vocabulary, support graph,
prompt, synchronization, and R9 timing boundary. Native/generated/native
medians are 43.428460 / 47.808165 / 43.963482 ms. Generated gate/up therefore
costs 4.112194 ms against the mean native controls on the exact current graph.

Every arm retains 252 projection mains and canonical packed weights with zero
V/down overlays. Five replay cycles per arm are exact. Native versus generated
logits keep token 198, are finite, and pass rtol 0.02 / atol 0.5 with maximum
absolute difference 0.12075853. The native cubins are diagnostic oracles only;
the generated production route and ordinary-selection goal are unchanged.
