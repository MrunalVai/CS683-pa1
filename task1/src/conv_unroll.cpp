// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    // TODO(student): replace this placeholder with your unrolled implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        // Unroll the inner spatial loop by 8 (since W is a multiple of 8)
        for (int ox = 0; ox < W; ox += 8) {
            // Maintain 8 separate accumulators to break dependency chains
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    float k_val = ker[ky * K + kx];
                    int in_idx = (oy + ky) * in_stride + (ox + kx);

                    // Compute 8 adjacent output pixels simultaneously
                    acc0 += in[in_idx + 0] * k_val;
                    acc1 += in[in_idx + 1] * k_val;
                    acc2 += in[in_idx + 2] * k_val;
                    acc3 += in[in_idx + 3] * k_val;
                    acc4 += in[in_idx + 4] * k_val;
                    acc5 += in[in_idx + 5] * k_val;
                    acc6 += in[in_idx + 6] * k_val;
                    acc7 += in[in_idx + 7] * k_val;
                }
            }

            // Write the accumulated results to memory exactly once
            int out_idx = oy * W + ox;
            out[out_idx + 0] = acc0;
            out[out_idx + 1] = acc1;
            out[out_idx + 2] = acc2;
            out[out_idx + 3] = acc3;
            out[out_idx + 4] = acc4;
            out[out_idx + 5] = acc5;
            out[out_idx + 6] = acc6;
            out[out_idx + 7] = acc7;
        }
    }
}
