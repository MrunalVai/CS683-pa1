#!/bin/bash
OUT=report_data/matrix_sizes.txt
rm -f "$OUT"
for N in 128 256 512 1024 1536 2048
do
    echo "================ SIZE=$N ================" >> "$OUT"
    ./bin/matmul all $N $N $N >> "$OUT" 2>&1
done
