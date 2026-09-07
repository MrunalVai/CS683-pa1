#!/bin/bash

set -u

OUT="task2_missing_data.txt"
TMP=".task2_missing_tmp"

CPU=0
SWEEP_SIZES="512 1024"
WIDTH_SIZES="256 512 1024"
PD_VALUES="8 32 64 128 256"
HINT_VALUES="T0 T1 T2 NTA"

rm -f "$OUT"
rm -rf "$TMP"
mkdir -p "$TMP"

cp src/matmul_prefetch.cpp "$TMP/matmul_prefetch.cpp"
cp src/matmul_simd.cpp "$TMP/matmul_simd_256.cpp"
cp src/matmul_optimized.cpp "$TMP/matmul_optimized_256.cpp"

MSR_ORIGINAL=""

log(){
    echo "$*" | tee -a "$OUT"
}

build(){
    make clean >/dev/null 2>&1
    if ! make >/dev/null 2>>"$OUT"; then
        log "BUILD FAILED"
        exit 1
    fi
}

restore(){
    cp "$TMP/matmul_prefetch.cpp" src/matmul_prefetch.cpp 2>/dev/null || true
    cp "$TMP/matmul_simd_256.cpp" src/matmul_simd.cpp 2>/dev/null || true
    cp "$TMP/matmul_optimized_256.cpp" src/matmul_optimized.cpp 2>/dev/null || true

    if [ -n "${MSR_ORIGINAL:-}" ] && command -v wrmsr >/dev/null 2>&1; then
        sudo wrmsr -p "$CPU" 0x1a4 "0x$MSR_ORIGINAL" >/dev/null 2>&1 || true
    fi

    make clean >/dev/null 2>&1 || true
    make >/dev/null 2>&1 || true
}

trap restore EXIT INT TERM

run_stage(){
    STAGE=$1
    N=$2
    taskset -c "$CPU" ./bin/matmul "$STAGE" "$N" "$N" "$N" >> "$OUT" 2>&1
}

perf_all(){
    STAGE=$1
    N=$2

    taskset -c "$CPU" perf stat \
        -e cpu_core/instructions/ \
        -e cpu_core/cycles/ \
        -e cpu_core/L1-dcache-load-misses/ \
        -e cpu_core/l2_rqsts.miss/ \
        -e cpu_core/LLC-load-misses/ \
        -e cpu_core/sw_prefetch_access.any/ \
        ./bin/matmul "$STAGE" "$N" "$N" "$N" >> "$OUT" 2>&1
}

log "============================================================"
log "TASK 2 - MISSING DATA ONLY"
log "============================================================"

build

log ""
log "============================================================"
log "2A: SOFTWARE PREFETCH INSTRUCTION COUNTS"
log "============================================================"

for N in $SWEEP_SIZES
do
    log ""
    log "######## 2A SW_PREFETCH_COUNT N=$N ########"

    taskset -c "$CPU" perf stat \
        -e cpu_core/sw_prefetch_access.any/ \
        ./bin/matmul prefetch "$N" "$N" "$N" >> "$OUT" 2>&1
done

log ""
log "============================================================"
log "2A: PREFETCH DISTANCE CACHE METRICS"
log "============================================================"

for PD in $PD_VALUES
do
    sed -E "s/PD=[0-9]+/PD=$PD/" \
        "$TMP/matmul_prefetch.cpp" > src/matmul_prefetch.cpp

    build

    log ""
    log "######## 2A PREFETCH_DISTANCE PD=$PD N=1024 ########"

    perf_all prefetch 1024
done

cp "$TMP/matmul_prefetch.cpp" src/matmul_prefetch.cpp
build

log ""
log "============================================================"
log "2A: CACHE FILL LEVEL CACHE METRICS"
log "============================================================"

for HINT in $HINT_VALUES
do
    sed "s/_MM_HINT_T0/_MM_HINT_$HINT/g" \
        "$TMP/matmul_prefetch.cpp" > src/matmul_prefetch.cpp

    build

    for N in $SWEEP_SIZES
    do
        log ""
        log "######## 2A PREFETCH_HINT=$HINT N=$N ########"

        run_stage prefetch "$N"
        perf_all prefetch "$N"
    done
done

cp "$TMP/matmul_prefetch.cpp" src/matmul_prefetch.cpp
build

log ""
log "============================================================"
log "2C: COMBINED PREFETCH DISTANCE SWEEP"
log "============================================================"

for PD in $PD_VALUES
do
    sed -E "s/PD=[0-9]+/PD=$PD/" \
        "$TMP/matmul_optimized_256.cpp" > src/matmul_optimized.cpp

    build

    for N in $SWEEP_SIZES
    do
        log ""
        log "######## 2C COMBINED_PD=$PD N=$N ########"

        run_stage optimized "$N"

        if [ "$N" = "1024" ]; then
            perf_all optimized "$N"
        fi
    done
done

cp "$TMP/matmul_optimized_256.cpp" src/matmul_optimized.cpp
build

log ""
log "============================================================"
log "2C: COMBINED CACHE FILL LEVEL SWEEP"
log "============================================================"

for HINT in $HINT_VALUES
do
    sed "s/_MM_HINT_T0/_MM_HINT_$HINT/g" \
        "$TMP/matmul_optimized_256.cpp" > src/matmul_optimized.cpp

    build

    for N in $SWEEP_SIZES
    do
        log ""
        log "######## 2C COMBINED_HINT=$HINT N=$N ########"

        run_stage optimized "$N"

        if [ "$N" = "1024" ]; then
            perf_all optimized "$N"
        fi
    done
done

cp "$TMP/matmul_optimized_256.cpp" src/matmul_optimized.cpp
build

log ""
log "============================================================"
log "2C: COMBINED SOFTWARE PREFETCH COUNTS"
log "============================================================"

for N in $SWEEP_SIZES
do
    log ""
    log "######## 2C SW_PREFETCH_COUNT N=$N ########"

    taskset -c "$CPU" perf stat \
        -e cpu_core/sw_prefetch_access.any/ \
        ./bin/matmul optimized "$N" "$N" "$N" >> "$OUT" 2>&1
done

log ""
log "============================================================"
log "2C: COMBINED 128-BIT SIMD + PREFETCH"
log "============================================================"

cat > src/matmul_optimized.cpp <<'EOF'
#include <immintrin.h>
#include "matmul.h"

static inline float hsum128_opt(__m128 v){
    v=_mm_hadd_ps(v,v);
    v=_mm_hadd_ps(v,v);
    return _mm_cvtss_f32(v);
}

void matmul_optimized(const float* A,const float* B,float* C,int M,int N,int K,int lda,int ldb,int ldc){
    const int BI=32,BJ=32,PD=64;

    for(int i0=0;i0<M;i0+=BI){
        int ie=i0+BI<M?i0+BI:M;

        for(int j0=0;j0<N;j0+=BJ){
            int je=j0+BJ<N?j0+BJ:N;

            for(int i=i0;i<ie;++i){
                const float* a=A+(long)i*lda;
                int j=j0;

                for(;j+3<je;j+=4){
                    const float* b0=B+(long)(j+0)*ldb;
                    const float* b1=B+(long)(j+1)*ldb;
                    const float* b2=B+(long)(j+2)*ldb;
                    const float* b3=B+(long)(j+3)*ldb;

                    __m128 s0=_mm_setzero_ps();
                    __m128 s1=_mm_setzero_ps();
                    __m128 s2=_mm_setzero_ps();
                    __m128 s3=_mm_setzero_ps();

                    int p=0;

                    for(;p+3<K;p+=4){
                        if(p+PD<K){
                            _mm_prefetch((const char*)(a+p+PD),_MM_HINT_T0);
                            _mm_prefetch((const char*)(b0+p+PD),_MM_HINT_T0);
                            _mm_prefetch((const char*)(b1+p+PD),_MM_HINT_T0);
                            _mm_prefetch((const char*)(b2+p+PD),_MM_HINT_T0);
                            _mm_prefetch((const char*)(b3+p+PD),_MM_HINT_T0);
                        }

                        __m128 av=_mm_loadu_ps(a+p);

                        s0=_mm_fmadd_ps(av,_mm_loadu_ps(b0+p),s0);
                        s1=_mm_fmadd_ps(av,_mm_loadu_ps(b1+p),s1);
                        s2=_mm_fmadd_ps(av,_mm_loadu_ps(b2+p),s2);
                        s3=_mm_fmadd_ps(av,_mm_loadu_ps(b3+p),s3);
                    }

                    float r0=hsum128_opt(s0);
                    float r1=hsum128_opt(s1);
                    float r2=hsum128_opt(s2);
                    float r3=hsum128_opt(s3);

                    for(;p<K;++p){
                        float av=a[p];
                        r0+=av*b0[p];
                        r1+=av*b1[p];
                        r2+=av*b2[p];
                        r3+=av*b3[p];
                    }

                    C[(long)i*ldc+j+0]=r0;
                    C[(long)i*ldc+j+1]=r1;
                    C[(long)i*ldc+j+2]=r2;
                    C[(long)i*ldc+j+3]=r3;
                }

                for(;j<je;++j){
                    const float* b=B+(long)j*ldb;

                    __m128 s=_mm_setzero_ps();
                    int p=0;

                    for(;p+3<K;p+=4){
                        if(p+PD<K){
                            _mm_prefetch((const char*)(a+p+PD),_MM_HINT_T0);
                            _mm_prefetch((const char*)(b+p+PD),_MM_HINT_T0);
                        }

                        s=_mm_fmadd_ps(
                            _mm_loadu_ps(a+p),
                            _mm_loadu_ps(b+p),
                            s
                        );
                    }

                    float r=hsum128_opt(s);

                    for(;p<K;++p)
                        r+=a[p]*b[p];

                    C[(long)i*ldc+j]=r;
                }
            }
        }
    }
}
EOF

build

for N in $WIDTH_SIZES
do
    log ""
    log "######## 2C SIMD_WIDTH=128 N=$N ########"

    run_stage optimized "$N"

    taskset -c "$CPU" perf stat \
        -e cpu_core/instructions/ \
        -e cpu_core/cycles/ \
        ./bin/matmul optimized "$N" "$N" "$N" >> "$OUT" 2>&1
done

log ""
log "============================================================"
log "2C: COMBINED 256-BIT SIMD + PREFETCH"
log "============================================================"

cp "$TMP/matmul_optimized_256.cpp" src/matmul_optimized.cpp
build

for N in $WIDTH_SIZES
do
    log ""
    log "######## 2C SIMD_WIDTH=256 N=$N ########"

    run_stage optimized "$N"

    taskset -c "$CPU" perf stat \
        -e cpu_core/instructions/ \
        -e cpu_core/cycles/ \
        ./bin/matmul optimized "$N" "$N" "$N" >> "$OUT" 2>&1
done

log ""
log "============================================================"
log "2A + 2C: HARDWARE PREFETCHER ENABLED/DISABLED"
log "============================================================"

if command -v rdmsr >/dev/null 2>&1 && command -v wrmsr >/dev/null 2>&1
then
    sudo modprobe msr >/dev/null 2>&1 || true

    MSR_ORIGINAL=$(sudo rdmsr -p "$CPU" 0x1a4 2>/dev/null || true)

    if [ -n "$MSR_ORIGINAL" ]
    then
        log "CPU=$CPU"
        log "Original MSR 0x1a4=$MSR_ORIGINAL"

        log ""
        log "######## HW_PREFETCH=ORIGINAL N=1024 ########"

        run_stage prefetch 1024
        perf_all prefetch 1024

        run_stage optimized 1024
        perf_all optimized 1024

        ORIGINAL_DEC=$((16#$MSR_ORIGINAL))
        DISABLED_DEC=$((ORIGINAL_DEC | 15))
        DISABLED_HEX=$(printf "%x" "$DISABLED_DEC")

        sudo wrmsr -p "$CPU" 0x1a4 "0x$DISABLED_HEX"

        CHECK=$(sudo rdmsr -p "$CPU" 0x1a4)

        log ""
        log "Hardware-prefetch-disabled MSR=$CHECK"

        log ""
        log "######## HW_PREFETCH=DISABLED N=1024 ########"

        run_stage prefetch 1024
        perf_all prefetch 1024

        run_stage optimized 1024
        perf_all optimized 1024

        sudo wrmsr -p "$CPU" 0x1a4 "0x$MSR_ORIGINAL"

        RESTORED=$(sudo rdmsr -p "$CPU" 0x1a4)

        log ""
        log "Restored MSR=$RESTORED"

        MSR_ORIGINAL=""
    else
        log "Could not read MSR 0x1a4."
    fi
else
    log "msr-tools not installed."
fi

restore

log ""
log "============================================================"
log "DONE"
log "============================================================"
log "Send me: $OUT"