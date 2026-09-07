# Q Stream-K fixup + RMSNorm + RoPE fusion

A generated-only isolated fixture extended the exact Q single-owner fixup with 128-wide RMSNorm, RoPE, fp16 conversion and direct Flash head-major output. Q's 128x128 fixup tiles align one 128-d head per column tile; each 128-thread block reduces one token/head row. Canonical ABI is fp16 norm weight[128] plus fp32 frequency[512,128].

The staged gates pass: raw fixup is exact (`max_abs=0`); RMSNorm differs by at most 1.91e-6; full RoPE/cast/layout agrees with max_abs 0.00390625 and mean_abs 4.82e-8. The matched baseline includes standard fixup plus canonical RMSNorm and RoPE. It measures 68.399 us median / 66.686 us minimum versus fused 75.572 / 73.709 us. The fused service regresses 7.173 us and cannot help the >0.5 ms model gate, so no production wiring is retained.
