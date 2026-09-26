#!/bin/bash
cd /home/ubuntu/tinygrad-self-training
O=/home/ubuntu/storage/audit-nongemm/dec_runs; mkdir -p $O
for b in 32 128; do DEBUG=2 DEV=NV PYTHONPATH=. timeout 300 python3 /home/ubuntu/storage/audit-nongemm/mdec_iso.py $b 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > $O/mdec_b$b.log; grep "^MDEC" $O/mdec_b$b.log; done
