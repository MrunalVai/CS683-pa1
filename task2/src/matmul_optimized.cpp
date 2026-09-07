#include <immintrin.h>
#include "matmul.h"

<<<<<<< HEAD
void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your best combined implementation.
    matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
}
=======
static inline float hsum256_opt(__m256 v){
    __m128 lo=_mm256_castps256_ps128(v);
    __m128 hi=_mm256_extractf128_ps(v,1);
    __m128 s=_mm_add_ps(lo,hi);
    s=_mm_hadd_ps(s,s);
    s=_mm_hadd_ps(s,s);
    return _mm_cvtss_f32(s);
}

void matmul_optimized(const float* A,const float* B,float* C,int M,int N,int K,int lda,int ldb,int ldc){
    const int BI=32,BJ=32,PD=64;

    for(int i0=0;i0<M;i0+=BI){

        int ie=i0+BI<M?i0+BI:M;

        for(int j0=0;j0<N;j0+=BJ){

            int je=j0+BJ<N?j0+BJ:N;
            int i=i0;

            for(;i+1<ie;i+=2){
                const float* a0=A+(long)i*lda;
                const float* a1=A+(long)(i+1)*lda;
                int j=j0;

                for(;j+3<je;j+=4){
                    const float* b0=B+(long)(j+0)*ldb;
                    const float* b1=B+(long)(j+1)*ldb;
                    const float* b2=B+(long)(j+2)*ldb;
                    const float* b3=B+(long)(j+3)*ldb;

                    __m256 s00=_mm256_setzero_ps(),s01=_mm256_setzero_ps(),s02=_mm256_setzero_ps(),s03=_mm256_setzero_ps();
                    __m256 s10=_mm256_setzero_ps(),s11=_mm256_setzero_ps(),s12=_mm256_setzero_ps(),s13=_mm256_setzero_ps();
                    
                    int p=0;

                    for(;p+7<K;p+=8){

                        if(p+PD<K)
                        {
                            _mm_prefetch((const char*)(a0+p+PD),_MM_HINT_T0);
                            _mm_prefetch((const char*)(a1+p+PD),_MM_HINT_T0);
                            _mm_prefetch((const char*)(b0+p+PD),_MM_HINT_T0);
                            _mm_prefetch((const char*)(b1+p+PD),_MM_HINT_T0);
                            _mm_prefetch((const char*)(b2+p+PD),_MM_HINT_T0);
                            _mm_prefetch((const char*)(b3+p+PD),_MM_HINT_T0);
                        }

                        __m256 va0=_mm256_loadu_ps(a0+p),
                        va1=_mm256_loadu_ps(a1+p),
                        vb0=_mm256_loadu_ps(b0+p),
                        vb1=_mm256_loadu_ps(b1+p),
                        vb2=_mm256_loadu_ps(b2+p),
                        vb3=_mm256_loadu_ps(b3+p);
                        s00=_mm256_fmadd_ps(va0,vb0,s00);
                        s01=_mm256_fmadd_ps(va0,vb1,s01);
                        s02=_mm256_fmadd_ps(va0,vb2,s02);
                        s03=_mm256_fmadd_ps(va0,vb3,s03);
                        s10=_mm256_fmadd_ps(va1,vb0,s10);
                        s11=_mm256_fmadd_ps(va1,vb1,s11);
                        s12=_mm256_fmadd_ps(va1,vb2,s12);
                        s13=_mm256_fmadd_ps(va1,vb3,s13);
                    }
                    float r00=hsum256_opt(s00),
                    r01=hsum256_opt(s01),
                    r02=hsum256_opt(s02),
                    r03=hsum256_opt(s03),
                    r10=hsum256_opt(s10),
                    r11=hsum256_opt(s11),
                    r12=hsum256_opt(s12),
                    r13=hsum256_opt(s13);

                    for(;p<K;++p){
                        float x0=a0[p],x1=a1[p];
                        float y0=b0[p],y1=b1[p],y2=b2[p],y3=b3[p];r00+=x0*y0;r01+=x0*y1;r02+=x0*y2;r03+=x0*y3;r10+=x1*y0;r11+=x1*y1;r12+=x1*y2;r13+=x1*y3;
                    }
                    C[(long)i*ldc+j+0]=r00;C[(long)i*ldc+j+1]=r01;C[(long)i*ldc+j+2]=r02;C[(long)i*ldc+j+3]=r03;
                    C[(long)(i+1)*ldc+j+0]=r10;C[(long)(i+1)*ldc+j+1]=r11;C[(long)(i+1)*ldc+j+2]=r12;C[(long)(i+1)*ldc+j+3]=r13;
                }
                for(;j<je;++j){
                    const float* b=B+(long)j*ldb;__m256 s0=_mm256_setzero_ps(),s1=_mm256_setzero_ps();int p=0;
                    for(;p+7<K;p+=8){__m256 vb=_mm256_loadu_ps(b+p);s0=_mm256_fmadd_ps(_mm256_loadu_ps(a0+p),vb,s0);s1=_mm256_fmadd_ps(_mm256_loadu_ps(a1+p),vb,s1);}
                    float r0=hsum256_opt(s0),r1=hsum256_opt(s1);for(;p<K;++p){r0+=a0[p]*b[p];r1+=a1[p]*b[p];}C[(long)i*ldc+j]=r0;C[(long)(i+1)*ldc+j]=r1;
                }
            }
            for(;i<ie;++i){
                const float* a=A+(long)i*lda;
                for(int j=j0;j<je;++j){const float* b=B+(long)j*ldb;__m256 s=_mm256_setzero_ps();int p=0;for(;p+7<K;p+=8)s=_mm256_fmadd_ps(_mm256_loadu_ps(a+p),_mm256_loadu_ps(b+p),s);float r=hsum256_opt(s);for(;p<K;++p)r+=a[p]*b[p];C[(long)i*ldc+j]=r;}
            }
        }
    }
}
>>>>>>> 29a4bf3 (Add Task2 Optimized Matmul)
