// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

static inline float hsum256_ps(__m256 v){
    __m128 lo=_mm256_castps256_ps128(v);
    __m128 hi=_mm256_extractf128_ps(v,1);
    __m128 s=_mm_add_ps(lo,hi);
    s=_mm_hadd_ps(s,s);
    s=_mm_hadd_ps(s,s);
    return _mm_cvtss_f32(s);
}

void matmul_simd(const float* A,const float* B,float* C,int M,int N,int K,int lda,int ldb,int ldc){
    for(int i=0;i<M;++i){
        const float* a=A+(long)i*lda;
        int j=0;
        for(;j+3<N;j+=4){
            const float* b0=B+(long)(j+0)*ldb;
            const float* b1=B+(long)(j+1)*ldb;
            const float* b2=B+(long)(j+2)*ldb;
            const float* b3=B+(long)(j+3)*ldb;
            __m256 s0=_mm256_setzero_ps(),s1=_mm256_setzero_ps(),s2=_mm256_setzero_ps(),s3=_mm256_setzero_ps();
            int p=0;
            for(;p+7<K;p+=8){
                __m256 av=_mm256_loadu_ps(a+p);
                s0=_mm256_fmadd_ps(av,_mm256_loadu_ps(b0+p),s0);
                s1=_mm256_fmadd_ps(av,_mm256_loadu_ps(b1+p),s1);
                s2=_mm256_fmadd_ps(av,_mm256_loadu_ps(b2+p),s2);
                s3=_mm256_fmadd_ps(av,_mm256_loadu_ps(b3+p),s3);
            }
            float r0=hsum256_ps(s0),r1=hsum256_ps(s1),r2=hsum256_ps(s2),r3=hsum256_ps(s3);
            for(;p<K;++p){
                float av=a[p];
                r0+=av*b0[p];
                r1+=av*b1[p];
                r2+=av*b2[p];
                r3+=av*b3[p];
            }
            C[(long)i*ldc+j+0]=r0;C[(long)i*ldc+j+1]=r1;C[(long)i*ldc+j+2]=r2;C[(long)i*ldc+j+3]=r3;
        }
        for(;j<N;++j){
            const float* b=B+(long)j*ldb;
            __m256 s=_mm256_setzero_ps();
            int p=0;
            for(;p+7<K;p+=8)
                s=_mm256_fmadd_ps(_mm256_loadu_ps(a+p),_mm256_loadu_ps(b+p),s);
            float r=hsum256_ps(s);
            for(;p<K;++p)r+=a[p]*b[p];
            C[(long)i*ldc+j]=r;
        }
    }
}
