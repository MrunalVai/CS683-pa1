#!/bin/bash
OUT=report_data/simd_instructions.txt
rm -f "$OUT"
for K in 3 5
do
    for N in 256 512 1024 2048
    do
        echo "================ NAIVE SIZE=$N K=$K ================" >> "$OUT"
        perf stat -e instructions,cycles ./bin/conv naive $N $N $K >> "$OUT" 2>&1
        echo "================ SIMD SIZE=$N K=$K ================" >> "$OUT"
        perf stat -e instructions,cycles ./bin/conv simd $N $N $K >> "$OUT" 2>&1
    done
done
