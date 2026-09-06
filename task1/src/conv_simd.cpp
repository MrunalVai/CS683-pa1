// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your AVX2 implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        // step by 8 since W is mathematically guaranteed to be a multiple of 8
        for (int ox = 0; ox < W; ox += 8) {
            
            // initialize a 256-bit vector accumulator with zeros (holds 8 floats)
            __m256 v_acc = _mm256_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    
                    // broadcast a single kernel weight across all 8 vector lanes
                    __m256 v_ker = _mm256_set1_ps(ker[ky * K + kx]);
                    
                    // load 8 unaligned input floats from memory
                    __m256 v_in = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx)]);
                    
                    // fused Multiply-Add: v_acc = (v_ker * v_in) + v_acc
                    v_acc = _mm256_fmadd_ps(v_ker, v_in, v_acc);
                }
            }

            // store the 8 computed output pixels back to memory
            _mm256_storeu_ps(&out[oy * W + ox], v_acc);
        }
    }
}
