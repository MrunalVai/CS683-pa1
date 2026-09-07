#!/bin/bash

OUT=task1_simd128.txt
rm -f "$OUT"

for K in 3 5
do
    for N in 128 256 512 1024 2048
    do
        echo "================ SIZE=$N K=$K ================" >> "$OUT"
        ./bin/conv naive $N $N $K >> "$OUT" 2>&1
        ./bin/conv simd $N $N $K >> "$OUT" 2>&1

        echo "----- PERF SIMD128 SIZE=$N K=$K -----" >> "$OUT"
        perf stat -e instructions,cycles ./bin/conv simd $N $N $K >> "$OUT" 2>&1
    done
done
