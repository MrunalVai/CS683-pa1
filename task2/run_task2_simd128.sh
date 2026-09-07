#!/bin/bash

OUT=task2_simd128.txt
rm -f "$OUT"

for N in 128 256 512 1024 1536 2048
do
    echo "================ SIZE=$N ================" >> "$OUT"

    ./bin/matmul naive $N $N $N >> "$OUT" 2>&1
    ./bin/matmul simd $N $N $N >> "$OUT" 2>&1

    echo "----- PERF SIMD128 SIZE=$N -----" >> "$OUT"
    perf stat -e instructions,cycles ./bin/matmul simd $N $N $N >> "$OUT" 2>&1
done
