#!/bin/bash
# all SSD variants, one Mamba layer at the prod 1024-token piece; per-kernel times from DEBUG=2 (non-JIT rep 0)
cd /home/ubuntu/tinygrad-self-training
O=/home/ubuntu/storage/audit-nongemm/ssd_runs; mkdir -p $O
for v in "float 256 0" "float 256 1" "split 256 1" "bf16 256 1" "split 256 0" "bf16 256 0" "float 128 1" "bf16 128 1" "split 128 1"; do
  set -- $v
  NOGEMM=1 MAT=$3 DEBUG=2 DEV=NV PYTHONPATH=. timeout 300 python3 /home/ubuntu/storage/audit-nongemm/ssd_iso.py $1 $2 4 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > $O/$1_$2_m$3.log
  grep "^rep" $O/$1_$2_m$3.log
done
