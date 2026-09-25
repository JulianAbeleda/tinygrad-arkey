#!/bin/bash
# NOTE: `gpu-run` (the exclusive-GPU wrapper from the original scratchpad) was not copied here;
# it lived at the scratchpad root, not under spec/. Replace `$S/gpu-run time N` with your own
# exclusive-GPU invocation, or call the ours_*.sh scripts below directly.
S=/home/ubuntu/tinygrad-self-training/docs/nemotron-vllm-parity/bench/spec
$S/gpu-run time 1500 $S/ours_prof.sh 8 200 1024 b8
$S/gpu-run time 1500 $S/ours_prof.sh 64 2000 1024 b64
$S/gpu-run time 1800 $S/ours_prof.sh 128 2000 1024 b128
$S/gpu-run time 2400 $S/ours_pf.sh 10000 256 ssd float ssd256
