#include "convolution.h"
#include <algorithm>

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    // Tile dimensions to evaluate L1-D cache behavior
    const int TH = 64;
    const int TW = 64;

    // Outer loops: Iterate over the image in TH x TW tiles
    for (int y0 = 0; y0 < H; y0 += TH) {
        for (int x0 = 0; x0 < W; x0 += TW) {
            
            int y_max = std::min(y0 + TH, H);
            int x_max = std::min(x0 + TW, W);

            // Inner loops: Standard naive convolution restricted to the current tile
            for (int oy = y0; oy < y_max; ++oy) {
                for (int ox = x0; ox < x_max; ++ox) {
                    float acc = 0.0f;
                    
                    for (int ky = 0; ky < K; ++ky) {
                        for (int kx = 0; kx < K; ++kx) {
                            acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                        }
                    }
                    
                    out[oy * W + ox] = acc;
                }
            }
        }
    }
}