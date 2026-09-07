#!/bin/bash

set -e

mkdir -p report_data

echo "===== TASK 2 DATA COLLECTION STARTED ====="

echo "===== SYSTEM INFO ====="
lscpu > report_data/lscpu.txt
lscpu -C > report_data/cache_info.txt 2>&1 || true
uname -a > report_data/system.txt
g++ --version > report_data/compiler.txt
lscpu | grep -i -E "Model name|Socket|Core|Thread|MHz|avx|avx2|avx512" > report_data/cpu_summary.txt || true
grep -m1 '^flags' /proc/cpuinfo > report_data/cpu_flags.txt || true
cat /proc/cpuinfo | grep "model name" | head -1 > report_data/cpu_model.txt || true
lscpu | grep -i -E "avx|avx2|avx512" > report_data/simd_support.txt || true
lscpu | grep -o 'avx512[^ ]*' > report_data/avx512_support.txt || true

echo "===== PERF EVENTS ====="
perf list | grep -i "L1-dcache" > report_data/perf_l1_events.txt || true
perf list | grep -i "L2" > report_data/perf_l2_events.txt || true
perf list | grep -i "LLC" > report_data/perf_llc_events.txt || true
perf list | grep -i "prefetch" > report_data/perf_prefetch_events.txt || true
perf list | grep -i "sw_prefetch" > report_data/sw_prefetch_events.txt || true

echo "===== BUILD ====="
make clean > report_data/build.txt 2>&1
make >> report_data/build.txt 2>&1

echo "===== DEFAULT RUN ====="
./bin/matmul > report_data/default_run.txt 2>&1

echo "===== CORRECTNESS TESTS ====="
./bin/matmul all 127 131 137 > report_data/correctness_127.txt 2>&1
./bin/matmul all 255 259 263 > report_data/correctness_255.txt 2>&1
./bin/matmul all 513 509 517 > report_data/correctness_513.txt 2>&1

echo "===== MATRIX SIZE PERFORMANCE ====="
./run_sizes.sh

echo "===== INSTRUCTION COUNTS ====="
./run_instructions.sh

echo "===== CACHE METRICS ====="
./run_cache.sh

echo "===== PREFETCH DISTANCE SWEEP ====="
./run_prefetch_distance.sh

echo "===== PREFETCH LEVEL SWEEP ====="
./run_prefetch_levels.sh

echo "===== PREFETCH DISTANCE PERF ====="
./run_prefetch_distance_perf.sh

echo "===== SOFTWARE PREFETCH ACCESS ====="
perf stat -e sw_prefetch_access ./bin/matmul prefetch 1024 1024 1024 > report_data/sw_prefetch_access.txt 2>&1 || true

echo "===== FINAL TASK 2 COMPARISON ====="
./run_task2_final.sh

echo "===== LLAMA DEMO ====="
make llama-demo > report_data/llama_demo.txt 2>&1 || true

echo "===== PACKAGING ====="
tar -czf task2_report_data.tar.gz report_data src

echo "===== DONE ====="
echo "Upload task2_report_data.tar.gz to me."
