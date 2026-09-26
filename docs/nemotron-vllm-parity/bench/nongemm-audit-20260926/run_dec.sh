#!/bin/bash
# decode non-GEMM isolations: sampler tail variants, attention CUDA-core vs tensor-core dots, norm default vs BEAM
cd /home/ubuntu/tinygrad-self-training
O=/home/ubuntu/storage/audit-nongemm/dec_runs; mkdir -p $O
for b in 32 128; do for v in 0 1 2; do
  V=$v DEBUG=2 DEV=NV PYTHONPATH=. timeout 300 python3 /home/ubuntu/storage/audit-nongemm/tail_iso.py $b 20 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > $O/tail_b${b}_v$v.log; grep "^TAIL" $O/tail_b${b}_v$v.log
done; done
for cfg in "32 200 150 256" "128 2000 150 256"; do for t in 0 1; do
  set -- $cfg
  TC=$t DEBUG=2 DEV=NV PYTHONPATH=. timeout 300 python3 /home/ubuntu/storage/audit-nongemm/att_iso.py $cfg 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > $O/att_b$1_tc$t.log; grep "^ATT" $O/att_b$1_tc$t.log
done; done
for b in 32 128; do
  DEBUG=2 DEV=NV PYTHONPATH=. timeout 300 python3 /home/ubuntu/storage/audit-nongemm/norm_iso.py $b 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > $O/norm_b$b.log; grep "^NORM" $O/norm_b$b.log
  BEAM=4 DEBUG=2 DEV=NV PYTHONPATH=. timeout 600 python3 /home/ubuntu/storage/audit-nongemm/norm_iso.py $b 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > $O/norm_b${b}_beam.log; grep "^NORM" $O/norm_b${b}_beam.log
done
