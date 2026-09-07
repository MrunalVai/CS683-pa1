#!/bin/bash
OUT=report_data/prefetch_levels.txt
rm -f "$OUT"
cp src/matmul_prefetch.cpp src/matmul_prefetch_backup.cpp
for HINT in T0 T1 T2 NTA
do
    sed "s/_MM_HINT_T0/_MM_HINT_$HINT/g" src/matmul_prefetch_backup.cpp > src/matmul_prefetch.cpp
    make clean >/dev/null
    make >/dev/null || exit 1
    echo "================ HINT=$HINT ================" >> "$OUT"
    for N in 512 1024 2048
    do
        echo "SIZE=$N" >> "$OUT"
        ./bin/matmul prefetch $N $N $N >> "$OUT" 2>&1
    done
done
mv src/matmul_prefetch_backup.cpp src/matmul_prefetch.cpp
make clean >/dev/null
make >/dev/null
