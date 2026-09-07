// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>
#include "convolution.h"

void conv_optimized(const float* in,float* out,const float* ker,int H,int W,int K){
    const int p=K/2,s=W+2*p;
    const int TH=64,TW=512;
    if(K==3){
        const __m256 k0=_mm256_set1_ps(ker[0]),k1=_mm256_set1_ps(ker[1]),k2=_mm256_set1_ps(ker[2]),k3=_mm256_set1_ps(ker[3]),k4=_mm256_set1_ps(ker[4]),k5=_mm256_set1_ps(ker[5]),k6=_mm256_set1_ps(ker[6]),k7=_mm256_set1_ps(ker[7]),k8=_mm256_set1_ps(ker[8]);
        for(int y0=0;y0<H;y0+=TH){
            int ye=y0+TH<H?y0+TH:H;
            for(int x0=0;x0<W;x0+=TW){
                int xe=x0+TW<W?x0+TW:W;
                for(int y=y0;y<ye;++y){
                    const float* r0=in+y*s;
                    const float* r1=r0+s;
                    const float* r2=r1+s;
                    float* o=out+y*W;
                    int x=x0;
                    for(;x+31<xe;x+=32){
                        __m256 a0=_mm256_mul_ps(_mm256_loadu_ps(r0+x),k0),a1=_mm256_mul_ps(_mm256_loadu_ps(r0+x+8),k0),a2=_mm256_mul_ps(_mm256_loadu_ps(r0+x+16),k0),a3=_mm256_mul_ps(_mm256_loadu_ps(r0+x+24),k0);
                        a0=_mm256_fmadd_ps(_mm256_loadu_ps(r0+x+1),k1,a0);a1=_mm256_fmadd_ps(_mm256_loadu_ps(r0+x+9),k1,a1);a2=_mm256_fmadd_ps(_mm256_loadu_ps(r0+x+17),k1,a2);a3=_mm256_fmadd_ps(_mm256_loadu_ps(r0+x+25),k1,a3);
                        a0=_mm256_fmadd_ps(_mm256_loadu_ps(r0+x+2),k2,a0);a1=_mm256_fmadd_ps(_mm256_loadu_ps(r0+x+10),k2,a1);a2=_mm256_fmadd_ps(_mm256_loadu_ps(r0+x+18),k2,a2);a3=_mm256_fmadd_ps(_mm256_loadu_ps(r0+x+26),k2,a3);
                        a0=_mm256_fmadd_ps(_mm256_loadu_ps(r1+x),k3,a0);a1=_mm256_fmadd_ps(_mm256_loadu_ps(r1+x+8),k3,a1);a2=_mm256_fmadd_ps(_mm256_loadu_ps(r1+x+16),k3,a2);a3=_mm256_fmadd_ps(_mm256_loadu_ps(r1+x+24),k3,a3);
                        a0=_mm256_fmadd_ps(_mm256_loadu_ps(r1+x+1),k4,a0);a1=_mm256_fmadd_ps(_mm256_loadu_ps(r1+x+9),k4,a1);a2=_mm256_fmadd_ps(_mm256_loadu_ps(r1+x+17),k4,a2);a3=_mm256_fmadd_ps(_mm256_loadu_ps(r1+x+25),k4,a3);
                        a0=_mm256_fmadd_ps(_mm256_loadu_ps(r1+x+2),k5,a0);a1=_mm256_fmadd_ps(_mm256_loadu_ps(r1+x+10),k5,a1);a2=_mm256_fmadd_ps(_mm256_loadu_ps(r1+x+18),k5,a2);a3=_mm256_fmadd_ps(_mm256_loadu_ps(r1+x+26),k5,a3);
                        a0=_mm256_fmadd_ps(_mm256_loadu_ps(r2+x),k6,a0);a1=_mm256_fmadd_ps(_mm256_loadu_ps(r2+x+8),k6,a1);a2=_mm256_fmadd_ps(_mm256_loadu_ps(r2+x+16),k6,a2);a3=_mm256_fmadd_ps(_mm256_loadu_ps(r2+x+24),k6,a3);
                        a0=_mm256_fmadd_ps(_mm256_loadu_ps(r2+x+1),k7,a0);a1=_mm256_fmadd_ps(_mm256_loadu_ps(r2+x+9),k7,a1);a2=_mm256_fmadd_ps(_mm256_loadu_ps(r2+x+17),k7,a2);a3=_mm256_fmadd_ps(_mm256_loadu_ps(r2+x+25),k7,a3);
                        a0=_mm256_fmadd_ps(_mm256_loadu_ps(r2+x+2),k8,a0);a1=_mm256_fmadd_ps(_mm256_loadu_ps(r2+x+10),k8,a1);a2=_mm256_fmadd_ps(_mm256_loadu_ps(r2+x+18),k8,a2);a3=_mm256_fmadd_ps(_mm256_loadu_ps(r2+x+26),k8,a3);
                        _mm256_storeu_ps(o+x,a0);
                        _mm256_storeu_ps(o+x+8,a1);
                        _mm256_storeu_ps(o+x+16,a2);
                        _mm256_storeu_ps(o+x+24,a3);
                    }
                    for(;x+7<xe;x+=8){
                        __m256 a=_mm256_mul_ps(_mm256_loadu_ps(r0+x),k0);
                        a=_mm256_fmadd_ps(_mm256_loadu_ps(r0+x+1),k1,a);
                        a=_mm256_fmadd_ps(_mm256_loadu_ps(r0+x+2),k2,a);
                        a=_mm256_fmadd_ps(_mm256_loadu_ps(r1+x),k3,a);
                        a=_mm256_fmadd_ps(_mm256_loadu_ps(r1+x+1),k4,a);
                        a=_mm256_fmadd_ps(_mm256_loadu_ps(r1+x+2),k5,a);
                        a=_mm256_fmadd_ps(_mm256_loadu_ps(r2+x),k6,a);
                        a=_mm256_fmadd_ps(_mm256_loadu_ps(r2+x+1),k7,a);
                        a=_mm256_fmadd_ps(_mm256_loadu_ps(r2+x+2),k8,a);
                        _mm256_storeu_ps(o+x,a);
                    }
                }
            }
        }
        return;
    }
    for(int y0=0;y0<H;y0+=TH){
        int ye=y0+TH<H?y0+TH:H;
        for(int x0=0;x0<W;x0+=TW){
            int xe=x0+TW<W?x0+TW:W;
            for(int y=y0;y<ye;++y){
                float* o=out+y*W;
                for(int x=x0;x<xe;x+=8){
                    __m256 a=_mm256_setzero_ps();
                    for(int ky=0;ky<K;++ky){
                        const float* r=in+(y+ky)*s+x;
                        const float* kr=ker+ky*K;
                        for(int kx=0;kx<K;++kx)a=_mm256_fmadd_ps(_mm256_loadu_ps(r+kx),_mm256_set1_ps(kr[kx]),a);
                    }
                    _mm256_storeu_ps(o+x,a);
                }
            }
        }
    }
}