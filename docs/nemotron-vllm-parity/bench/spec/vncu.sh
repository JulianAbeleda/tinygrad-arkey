#!/bin/bash
# usage: vncu.sh tag skip count regex -- args for vllm_ncu.py
S=/home/ubuntu/tinygrad-self-training/docs/nemotron-vllm-parity/bench/spec
tag=$1; skip=$2; count=$3; rx=$4; shift 4
export PATH=/home/ubuntu/vllm-bench/.venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda VLLM_USE_FLASHINFER_SAMPLER=0 TRITON_PRINT_AUTOTUNING=1
cd /home/ubuntu/vllm-bench
sudo -E env PATH=$PATH HOME=$HOME ncu --profile-from-start off --target-processes all --graph-profiling node \
  --kernel-name "regex:$rx" --launch-skip $skip --launch-count $count \
  --section SpeedOfLight --section MemoryWorkloadAnalysis --section LaunchStats --section Occupancy \
  -f -o $S/vncu_$tag .venv/bin/python $S/vllm_ncu.py "$@" > $S/vncu_$tag.log 2>&1
echo "exit $?" >> $S/vncu_$tag.log
sudo chown ubuntu:ubuntu $S/vncu_$tag.* 
