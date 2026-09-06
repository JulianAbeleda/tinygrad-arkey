# Q4-down SASS comparison

Using the actual generated `q4_down_streamk` and llama Q4 main symbols, nvdisasm counted generated/llama respectively: 288/256 MMA instructions, 18/9 barriers, 234/114 LDG, 0/88 LDS, 67/128 STG, and 239/817 address instructions. Generated has 12.5% more MMA and twice the barriers and global loads; llama uses shared loads and more address arithmetic. The evidence supports reducing repeated generated packed metadata loads and barriers across K64 panels.
