#!/bin/bash
OUT=report_data/task1_final.txt
rm -f "$OUT"
for K in 3 5
do
    for N in 128 256 512 1024 2048 4096
    do
        echo "================ SIZE=$N K=$K ================" >> "$OUT"
        ./bin/conv all $N $N $K >> "$OUT" 2>&1
    done
done
