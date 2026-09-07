#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    int p = K / 2;
    int in_stride = W + 2 * p;  // padded row stride

<<<<<<< HEAD
    const int tile_size = 32;
    const int tile_h = tile_size;
    const int tile_w = tile_size;
=======
    // Tile dimensions to evaluate L1-D cache behavior
    const int TH = 16;
    const int TW = 16;
>>>>>>> 29a4bf3 (Add Task2 Optimized Matmul)

    for (int y0 = 0; y0 < H; y0 += tile_h) {
        int y_max = (y0 + tile_h < H) ? y0 + tile_h : H;
        for (int x0 = 0; x0 < W; x0 += tile_w) {
            int x_max = (x0 + tile_w < W) ? x0 + tile_w : W;
            for (int oy = y0; oy < y_max; ++oy) {
                int ox = x0;
                for (; ox + 7 < x_max; ox += 8) {
                    float a0 = 0.0f, a1 = 0.0f,  a2 = 0.0f,a3 = 0.0f,a4 = 0.0f, a5 = 0.0f,a6 = 0.0f,a7 = 0.0f;
                    for (int ky = 0; ky < K; ++ky) {
                        const float* row =
                            in + (oy + ky) * in_stride + ox;
                        const float* kr = ker + ky * K;
                        for (int kx = 0; kx < K; ++kx) {
                            float k = kr[kx];
                            const float* q = row + kx;
                            a0 += q[0] * k;
                            a1 += q[1] * k;
                            a2 += q[2] * k;
                            a3 += q[3] * k;
                            a4 += q[4] * k;
                            a5 += q[5] * k;
                            a6 += q[6] * k;
                            a7 += q[7] * k;
                        }
                    }
                    float* o = out + oy * W + ox;
                    o[0] = a0;
                    o[1] = a1;
                    o[2] = a2;
                    o[3] = a3;
                    o[4] = a4;
                    o[5] = a5;
                    o[6] = a6;
                    o[7] = a7;
                }
                for (; ox < x_max; ++ox) {
                    float acc = 0.0f;
                    for (int ky = 0; ky < K; ++ky) {
                        const float* row = in + (oy + ky) * in_stride + ox;
                        const float* kr = ker + ky * K;
                        for (int kx = 0; kx < K; ++kx) {
                            acc += row[kx] * kr[kx];
                        }
                    }
                    out[oy * W + ox] = acc;
                }
            }
        }
    }
}