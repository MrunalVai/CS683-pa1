#!/bin/bash

OUT=task2_l2.txt
rm -f "$OUT"

for N in 256 512 1024 2048
do
    for STAGE in naive simd prefetch optimized
    do
        echo "================ $STAGE SIZE=$N ================" >> "$OUT"
        perf stat \
        -e instructions,cpu_core/l2_rqsts.miss/,L1-dcache-load-misses,LLC-load-misses \
        ./bin/matmul $STAGE $N $N $N >> "$OUT" 2>&1
    done
done
