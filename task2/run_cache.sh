#!/bin/bash
OUT=report_data/cache_metrics.txt
rm -f "$OUT"
for N in 256 512 1024 2048
do
    for STAGE in naive simd prefetch optimized
    do
        echo "================ $STAGE SIZE=$N ================" >> "$OUT"
        perf stat -e instructions,cycles,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses ./bin/matmul $STAGE $N $N $N >> "$OUT" 2>&1
    done
done
