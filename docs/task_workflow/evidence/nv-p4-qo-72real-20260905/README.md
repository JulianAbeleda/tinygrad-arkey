# Q/O 72-real-weight proxy (2026-09-05)

`nv_compiler_q4k_qo_72real_proxy.py --rounds 3` captured 36 Q and 36 O canonical GGUF buffers. Packed and FP16 paths each captured 72 calls; outputs were finite, replay-exact, and distinct across sampled real weights. Packed timing was 7.785 ms minimum (108.126 us/call). The comparator is an expanded FP16 proxy and the 61.5 us llama value is historical, so this is diagnostic only and does not qualify production Q/O.
