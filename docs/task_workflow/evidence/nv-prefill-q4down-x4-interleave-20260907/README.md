# Q4-down x4 plus WMMA-interleave transfer

The gate/up native-Q4-A fragment, logical XOR4 store remap, and WMMA-update
interleave transfer cleanly to the generated Q4-down Stream-K topology.  The
typed candidate uses physical 4096x512 tiles, K=12288, 170 owners, canonical
Q4 weights, the existing compact Q8 record, and transposes only direct/fixup
ownership back to logical 512x4096.

The canonical 18-role gate passes all 54 producer/main/fixup calls, both
distinct activations, and native comparison (max_abs 0.012156, mean_abs
4.31e-5).  Candidate SASS retains 288 IMMA and 64 STG.E.64 while changing
control/candidate resources from REG255/STACK56 to REG243/STACK0, LDL/STL
14/14 to 0/0, LDS720 to504, and adds LDSM72/SHFL128.

Adjacent R9 runs normalized to the stable native arm estimate the generated
population improvement at 0.411104 ms: current generated 5.370841 ms versus
x4/interleave 4.959737 ms.  This positive estimate is below the frozen 0.5 ms
model promotion gate, so it remains explicit/default-off and was not wired
into the model.  A disjoint compatible gain is required before composition.
