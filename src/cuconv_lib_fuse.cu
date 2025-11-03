#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "cuconv_api.h"


__global__ void fused_conv_kernel(
    const float* __restrict__ input,    
    const float* __restrict__ filters,   
    float* __restrict__ output,          
    int H, int W,
    int pad_h, int pad_w,
    int R, int S,
    int C, int M,
    int Ho, int Wo
){
    int m = blockIdx.y;                                       // כמות הפילטרים שהוגדרו בהרצה
    int outPixelIdx = blockIdx.x * blockDim.x + threadIdx.x;  // כמות הפקסלים במוצא לפי נתוני ההשקה
    if (m >= M || outPixelIdx >= Ho * Wo) return;

    int outY = outPixelIdx / Wo;
    int outX = outPixelIdx % Wo;

    extern __shared__ float sFilter[];   // הכרזת מערך של זיכרון משותף לכל בלוק שגודלו הוגדר בהשקת הקרנל

    float acc = 0.0f;

    
    for (int fy = 0; fy < R; ++fy){
        for (int fx = 0; fx < S; ++fx){
            for (int d = threadIdx.x; d < C; d += blockDim.x){    // כל ת'רד טוען חלק מהפילטר לזיכרון המשותף
                int fOff = (((m * R + fy) * S + fx) * C) + d; // אינדוקס לחד מימדי להיסט הפילטר.....................
                sFilter[d] = LDG(&filters[fOff]);           // טעינה של הפילטר בצורה מקבילית לתוך הזיכרון המשותף של כל בלוק
            }
            __syncthreads();  // מחכים עד שכל הת'רדים יסיימו את עבודתם ורק אז ממשיכים כלומר יטעינו את הפילטר לבלוק 

            int inY = outY - pad_h + fy;//לוגי PADDING קואורדינטות קלט עם 
            int inX = outX - pad_w + fx;

            if ((unsigned)inY < (unsigned)H && (unsigned)inX < (unsigned)W){  // כל ת'רד בודק ואז מבצע
                for (int d = 0; d < C; ++d){               // מכפלה על ציר העומק בקלט ובפילטר 
                    int inIdx = d * H * W + inY * W + inX;    //  אינדוקס של הקלט למערך חד מימדי
                    acc += LDG(&input[inIdx]) * sFilter[d];   // ביצוע המכפלה והסכימה וקבלת תוצאת הקונבולוציה
                }
            }

            __syncthreads(); // מחכים עד שכל הת'רדיפ יסיימו
        }
    }

    
    int outIdx = m * Ho * Wo + outPixelIdx;  // אינדוקס לפיקסלי הפלט
    output[outIdx] = acc;  // השמה של תוצאת הקונבולוציה במערך הפלט
}

// API 
extern "C" int cuconv_conv_forward(
    const float* x, const float* w, float* y,
    int N, int C, int H, int W,
    int M, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dil_h, int dil_w,
    int groups)
{
    
    if (N != 1 || groups != 1) return -1;
    if (stride_h != 1 || stride_w != 1) return -2;
    if (dil_h != 1 || dil_w != 1) return -3;

    
    int Ho = H + 2*pad_h - R + 1;
    int Wo = W + 2*pad_w - S + 1;

    
    int threads = 256;
    int blocks_x = (Ho * Wo + threads - 1) / threads;
    dim3 grid(blocks_x, M, 1);
    size_t shmem = (size_t)C * sizeof(float);  

    fused_conv_kernel<<<grid, threads, shmem>>>(
        x, w, y,
        H, W,
        pad_h, pad_w,
        R, S,
        C, M,
        Ho, Wo
    );
    cudaGetLastError();
    return 0;
}
