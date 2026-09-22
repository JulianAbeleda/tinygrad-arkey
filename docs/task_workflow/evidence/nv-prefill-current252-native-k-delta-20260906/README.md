# Current252 native K-only role delta

Generated/native-K/generated medians are 47.811322 / 47.902701 / 47.824791 ms. Native K changes wall by -0.084645 ms against mean controls (negative means native is slower), so K is closed as neutral/slower and does not explain the combined QKV gain.

All arms PASS token 198 with exact five-cycle replay, canonical 252/252 weights, zero overlays, and 216 Q8 producers. Native K explicitly has 36 main and 36 fixup calls; generated Q/O remain 72. Native binaries are diagnostic only.
