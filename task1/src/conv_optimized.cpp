// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>
#include <algorithm>
#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    // TODO(student): replace this placeholder with your best combined implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    // TILING: Tuned cache block sizes to fit within L1/L2 caches
    const int TH = 256;
    const int TW = 256;

    // 1. TILING (Outer Loops): Process the image in cache-resident blocks
    for (int y0 = 0; y0 < H; y0 += TH) {
        for (int x0 = 0; x0 < W; x0 += TW) {
            
            int y_max = std::min(y0 + TH, H);
            int x_max = std::min(x0 + TW, W);

            for (int oy = y0; oy < y_max; ++oy) {
                int ox = x0;
                
                // 2. UNROLLING + SIMD: Process 4 vector registers (32 pixels) per iteration
                for (; ox <= x_max - 32; ox += 32) {
                    
                    // 3. REGISTER BLOCKING: Hold accumulators in hardware registers
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();

                    // 4. REORDERING: Kernel loops inside the spatial blocks to hit L1 cache
                    for (int ky = 0; ky < K; ++ky) {
                        for (int kx = 0; kx < K; ++kx) {
                            
                            // Broadcast the scalar kernel weight to all lanes
                            __m256 v_ker = _mm256_set1_ps(ker[ky * K + kx]);
                            const float* in_ptr = in + (oy + ky) * in_stride + (ox + kx);
                            
                            // Load 32 unaligned input floats and perform Fused Multiply-Add
                            acc0 = _mm256_fmadd_ps(v_ker, _mm256_loadu_ps(in_ptr + 0), acc0);
                            acc1 = _mm256_fmadd_ps(v_ker, _mm256_loadu_ps(in_ptr + 8), acc1);
                            acc2 = _mm256_fmadd_ps(v_ker, _mm256_loadu_ps(in_ptr + 16), acc2);
                            acc3 = _mm256_fmadd_ps(v_ker, _mm256_loadu_ps(in_ptr + 24), acc3);
                        }
                    }
                    
                    // Write back to the L1 data cache exactly once per pixel block
                    float* out_ptr = out + oy * W + ox;
                    _mm256_storeu_ps(out_ptr + 0, acc0);
                    _mm256_storeu_ps(out_ptr + 8, acc1);
                    _mm256_storeu_ps(out_ptr + 16, acc2);
                    _mm256_storeu_ps(out_ptr + 24, acc3);
                }
                
                // Remainder loop to handle image widths that are multiples of 8 but not 32
                for (; ox < x_max; ox += 8) {
                    __m256 acc0 = _mm256_setzero_ps();
                    
                    for (int ky = 0; ky < K; ++ky) {
                        for (int kx = 0; kx < K; ++kx) {
                            __m256 v_ker = _mm256_set1_ps(ker[ky * K + kx]);
                            const float* in_ptr = in + (oy + ky) * in_stride + (ox + kx);
                            acc0 = _mm256_fmadd_ps(v_ker, _mm256_loadu_ps(in_ptr), acc0);
                        }
                    }
                    _mm256_storeu_ps(out + oy * W + ox, acc0);
                }
            }
        }
    }
}
