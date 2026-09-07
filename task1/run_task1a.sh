#!/bin/bash
OUT=report_data/task1a_performance.txt
rm -f "$OUT"
for K in 3 5
do
    for N in 128 256 512 1024 2048 4096
    do
        echo "================ K=$K SIZE=$N ================" >> "$OUT"
        ./bin/conv naive $N $N $K >> "$OUT" 2>&1
        ./bin/conv reorder $N $N $K >> "$OUT" 2>&1
        ./bin/conv unroll $N $N $K >> "$OUT" 2>&1
    done
done
