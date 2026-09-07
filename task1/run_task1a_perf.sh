#!/bin/bash
OUT=report_data/task1a_perf.txt
rm -f "$OUT"
for N in 256 512 1024 2048
do
    for STAGE in naive reorder unroll
    do
        echo "================ $STAGE SIZE=$N K=3 ================" >> "$OUT"
        perf stat -e instructions,cycles ./bin/conv $STAGE $N $N 3 >> "$OUT" 2>&1
    done
done
