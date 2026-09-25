#!/bin/bash
# usage: ours_prof.sh B P CAP tag
S=/home/ubuntu/tinygrad-self-training/docs/nemotron-vllm-parity/bench/spec
cd /home/ubuntu/tinygrad-self-training
rm -f $S/ours_$4.jsonl
PROFILE=1 HCQ_GRAPH_PROFILE_JSON=$S/ours_$4.jsonl DEV=NV PYTHONPATH=. python3 $S/ours_prof.py $1 $2 $3 $S/ours_$4_meta.json 2>&1 | grep -v "^\s*$" | tail -5
