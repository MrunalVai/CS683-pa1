#!/bin/bash
OUT=report_data/prefetch_distance.txt
rm -f "$OUT"
cp src/matmul_prefetch.cpp src/matmul_prefetch_backup.cpp
for PD in 8 16 32 64 96 128 192 256
do
    sed -E "s/BI=16,BJ=32,PD=[0-9]+/BI=16,BJ=32,PD=$PD/" src/matmul_prefetch_backup.cpp > src/matmul_prefetch.cpp
    make clean >/dev/null
    make >/dev/null || exit 1
    echo "================ PD=$PD ================" >> "$OUT"
    for N in 512 1024 2048
    do
        echo "SIZE=$N" >> "$OUT"
        ./bin/matmul prefetch $N $N $N >> "$OUT" 2>&1
    done
done
mv src/matmul_prefetch_backup.cpp src/matmul_prefetch.cpp
make clean >/dev/null
make >/dev/null
