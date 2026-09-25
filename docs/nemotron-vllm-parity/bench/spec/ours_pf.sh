#!/bin/bash
# usage: ours_pf.sh L PIECE MAMBA PREC tag
S=/home/ubuntu/tinygrad-self-training/docs/nemotron-vllm-parity/bench/spec
cd /home/ubuntu/tinygrad-self-training
rm -f $S/ourspf_$5.jsonl
PROFILE=1 HCQ_GRAPH_PROFILE_JSON=$S/ourspf_$5.jsonl DEV=NV PYTHONPATH=. python3 $S/ours_pf_prof.py $1 $2 $3 $4 $S/ourspf_$5_meta.json > $S/ourspf_$5.log 2>&1
grep -E "RESULT|rep |Error" $S/ourspf_$5.log | tail -6
