# Q4-down SASS comparison

Using the actual generated `q4_down_streamk` and llama Q4 main symbols, an initial nvdisasm pass counted static instruction occurrences. The figures are not loop-normalized runtime counts; the first LDS matcher also missed plain LDS opcodes. They are retained as raw disassembly observations only until opcode parsing is corrected. A restrict-qualified transformed source now compiles as an opt-in experiment; no runtime claim is made yet.
