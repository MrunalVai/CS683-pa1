#!/bin/bash
OUT=report_data/task1b_baseline_cache.txt
rm -f "$OUT"
for K in 3 5
do
    for N in 256 512 1024 2048 4096
    do
        echo "================ NAIVE SIZE=$N K=$K ================" >> "$OUT"
        perf stat -e instructions,cycles,L1-dcache-loads,L1-dcache-load-misses ./bin/conv naive $N $N $K >> "$OUT" 2>&1
    done
done
