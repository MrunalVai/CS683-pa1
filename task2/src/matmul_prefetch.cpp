#include <immintrin.h>
#include "matmul.h"

static inline float hsum256_pref(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);

    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);

    return _mm_cvtss_f32(s);
}

void matmul_prefetch(const float* A,const float* B,float* C,int M,int N,int K,int lda,int ldb,int ldc)
{
    const int BI = 16;
    const int BJ = 32;
    const int PD = 64;

    for (int i0 = 0; i0 < M; i0 += BI){
        int ie = (i0 + BI < M) ? i0 + BI : M;

        for (int j0 = 0; j0 < N; j0 += BJ){
            int je = (j0 + BJ < N) ? j0 + BJ : N;

            for (int i = i0; i < ie; ++i){
                const float* a = A + (long)i * lda;
                int j = j0;

                // Process four output columns at a time.
                for (; j + 3 < je; j += 4){
                    const float* b0 = B + (long)(j + 0) * ldb;
                    const float* b1 = B + (long)(j + 1) * ldb;
                    const float* b2 = B + (long)(j + 2) * ldb;
                    const float* b3 = B + (long)(j + 3) * ldb;

                    __m256 s0 = _mm256_setzero_ps();
                    __m256 s1 = _mm256_setzero_ps();
                    __m256 s2 = _mm256_setzero_ps();
                    __m256 s3 = _mm256_setzero_ps();

                    int p = 0;

                    for (; p + 7 < K; p += 8){
                        if (p + PD < K){
                            _mm_prefetch((const char*)(a + p + PD),_MM_HINT_T0);
                            _mm_prefetch((const char*)(b0 + p + PD), _MM_HINT_T0);
                            _mm_prefetch((const char*)(b1 + p + PD),_MM_HINT_T0);
                            _mm_prefetch((const char*)(b2 + p + PD),_MM_HINT_T0);
                            _mm_prefetch((const char*)(b3 + p + PD),_MM_HINT_T0);
                        }

                        __m256 av = _mm256_loadu_ps(a + p);

                        s0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b0 + p),s0);
                        s1 = _mm256_fmadd_ps(av,_mm256_loadu_ps(b1 + p),s1);
                        s2 = _mm256_fmadd_ps(av,_mm256_loadu_ps(b2 + p),s2);
                        s3 = _mm256_fmadd_ps(av,_mm256_loadu_ps(b3 + p),s3);
                    }

                    float r0 = hsum256_pref(s0);
                    float r1 = hsum256_pref(s1);
                    float r2 = hsum256_pref(s2);
                    float r3 = hsum256_pref(s3);

                    for (; p < K; ++p){
                        float av = a[p];

                        r0 += av * b0[p];
                        r1 += av * b1[p];
                        r2 += av * b2[p];
                        r3 += av * b3[p];
                    }

                    C[(long)i * ldc + j + 0] = r0;
                    C[(long)i * ldc + j + 1] = r1;
                    C[(long)i * ldc + j + 2] = r2;
                    C[(long)i * ldc + j + 3] = r3;
                }

                // Handle remaining columns.
                for (; j < je; ++j){
                    const float* b = B + (long)j * ldb;
                    __m256 s = _mm256_setzero_ps();
                    int p = 0;
                    for (; p + 7 < K; p += 8){
                        if (p + PD < K){
                            _mm_prefetch(
                                (const char*)(a + p + PD),
                                _MM_HINT_T0);

                            _mm_prefetch(
                                (const char*)(b + p + PD),
                                _MM_HINT_T0);
                        }

                        s = _mm256_fmadd_ps(
                            _mm256_loadu_ps(a + p),
                            _mm256_loadu_ps(b + p),
                            s);
                    }
                    float r = hsum256_pref(s);
                    for (; p < K; ++p){
                        r += a[p] * b[p];
                    }
                    C[(long)i * ldc + j] = r;
                }
            }
        }
    }
}
