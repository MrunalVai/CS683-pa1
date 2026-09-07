#!/bin/bash
OUT=report_data/prefetch_distance_perf.txt
rm -f "$OUT"
cp src/matmul_prefetch.cpp src/matmul_prefetch_backup.cpp
for PD in 16 32 64 128 256
do
    sed -E "s/BI=16,BJ=32,PD=[0-9]+/BI=16,BJ=32,PD=$PD/" src/matmul_prefetch_backup.cpp > src/matmul_prefetch.cpp
    make clean >/dev/null
    make >/dev/null || exit 1
    echo "================ PD=$PD ================" >> "$OUT"
    perf stat -e instructions,cycles,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses ./bin/matmul prefetch 1024 1024 1024 >> "$OUT" 2>&1
done
mv src/matmul_prefetch_backup.cpp src/matmul_prefetch.cpp
make clean >/dev/null
make >/dev/null
