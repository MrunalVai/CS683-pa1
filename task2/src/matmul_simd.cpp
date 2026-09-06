// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

void matmul_simd(const float *A, const float *B, float *C, int M, int N, int K,
                 int lda, int ldb, int ldc)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc[8];
            const float *a = A + static_cast<long>(i) * lda;
            const float *b = B + static_cast<long>(j) * ldb;
            __m256 vacc = _mm256_setzero_ps();
            for (int p = 0; p < K; p += 8) {
                __m256 va = _mm256_loadu_ps(&a[p]);
                __m256 vb = _mm256_loadu_ps(&b[p]);
                vacc = _mm256_fmadd_ps(va, vb, vacc);
            }
            __m256 vacc2 = _mm256_permute_ps(vacc, 0b11101110);
            vacc = _mm256_add_ps(vacc, vacc2);
            vacc2 = _mm256_permute_ps(vacc, 0b01010101);
            vacc = _mm256_add_ps(vacc, vacc2);
            _mm256_storeu_ps(acc, vacc);
            C[static_cast<long>(i) * ldc + j] = acc[0] + acc[4];
        }
    }
}
