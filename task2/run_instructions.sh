#!/bin/bash
OUT=report_data/instructions.txt
rm -f "$OUT"
for N in 256 512 1024 2048
do
    for STAGE in naive simd prefetch optimized
    do
        echo "================ $STAGE SIZE=$N ================" >> "$OUT"
        perf stat -e instructions,cycles ./bin/matmul $STAGE $N $N $N >> "$OUT" 2>&1
    done
done
