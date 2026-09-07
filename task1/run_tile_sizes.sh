#!/bin/bash
OUT=report_data/tile_sizes.txt
rm -f "$OUT"
cp src/conv_tile.cpp src/conv_tile_backup.cpp
for TILE in 8 16 32 64 128 256
do
    sed -E "s/int tile_size = [0-9]+;/int tile_size = $TILE;/" src/conv_tile_backup.cpp > src/conv_tile.cpp
    make clean >/dev/null
    make >/dev/null || exit 1
    echo "================ TILE=$TILE ================" >> "$OUT"
    for N in 256 512 1024 2048 4096
    do
        echo "SIZE=$N K=3" >> "$OUT"
        ./bin/conv tile $N $N 3 >> "$OUT" 2>&1
    done
done
mv src/conv_tile_backup.cpp src/conv_tile.cpp
make clean >/dev/null
make >/dev/null
