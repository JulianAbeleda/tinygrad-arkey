# Wide Q4 FFN-down live R31 gate

The down asset now uses the compiler's wide `128x128`, 256-thread geometry while retaining its exact `(M,N,K)=(512,4096,12288)` Q4_K contract and graph-owned Q8 record. The dynamic candidate and live llama graphs were interleaved for 31 calls.

Fresh result: finite and tolerance-correct on two activations, with 18 Q8 producers and 18 generated mains versus 18 llama mains plus 18 fixups. Candidate median `6.537605 ms`; llama median `3.733550 ms`; first-input maximum absolute difference `0.01215595`. The wide candidate is faster than the prior tileK64 candidate but remains slower than llama and stays opt-in.
