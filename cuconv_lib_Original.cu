#include <cuda_runtime.h>
#include "cuconv_api.h"


__global__ void scalar_prods_kernel(     //����� �� ����� ������
    const float* __restrict__ input,      //����� �� ���� ������ ���� ���� ��������� ���� ����� ��� ������� �����������
    const float* __restrict__ filters,   
    float* __restrict__ partial_outputs, 
    int H, int W,
    int pad_h, int pad_w,
    int filterH, int filterW,
    int depth, int numFilters,
    int outH, int outW)
{
    int fyfx = blockIdx.z;                      //   ����� ������ ���� �� ������ ������ ������ ����� ����� ������� ������ �� �����      
    if (fyfx >= filterH * filterW) return;      // ����� ��� ������� ��� ������ �������
    int fy = fyfx / filterW;
    int fx = fyfx % filterW;

    int filterIdx = blockIdx.y;                //  ����� ������ ���� ���� �������� ������ ������� ���� 
    if (filterIdx >= numFilters) return;       // ����� ��� ������� ���� ������ ���� ��� ������ ������

    int outPixelIdx = blockIdx.x * blockDim.x + threadIdx.x;  // ����� ����� �'���� �� ������
    if (outPixelIdx >= outH * outW) return;     // ����� ��� ������ �'���� �������

    int outY = outPixelIdx / outW;             // ����� ����� ���� ���� �'�� ����
    int outX = outPixelIdx % outW;

    
    int inY = outY - pad_h + fy;              //����� ����� ���� ���� �'�� ����
    int inX = outX - pad_w + fx;

    float sum = 0.0f;                                                    
    if ((unsigned)inY < (unsigned)H && (unsigned)inX < (unsigned)W) {   // ����� ��'���� �� ������ ��� ����� �����
        for (int d = 0; d < depth; d++) {                               //�� �'�� ����� �� ���� �� ������ �����               
            int inIdx = d * H * W + inY * W + inX;                                 //����� ������ �� ����� ������ �� ���� ��'����
            int fOff  = (((filterIdx * filterH + fy) * filterW + fx) * depth) + d;     //  ����� ������ �� ����� ������ �� ������� ������
            float xv = input[inIdx];
            float wv = filters[fOff];
            sum += xv * wv;
        }
    } else {
        sum = 0.0f;
    }

    int partialIdx = ( (filterIdx * (filterH * filterW) + fyfx) * outH * outW ) + outPixelIdx;    // ����� ������ �� ����� ��'���� ������ ������� �������
    partial_outputs[partialIdx] = sum;
}


__global__ void sum_kernel(                                          //����� �� ������ ����
    const float* __restrict__ partial_outputs,                       //  ����� �� ���� ������� ������� ������ ���� ���� ��������� ���� ������           
    float* __restrict__ output,                                     //   ����� ������ ��� ���� ���� ������� �������
    int outH, int outW,
    int numFilters, int kernelSize       
){
    int filterIdx   = blockIdx.y;                              
    if (filterIdx >= numFilters) return;  

    int outPixelIdx = blockIdx.x * blockDim.x + threadIdx.x;           //������� ��� �'�� �� ���� ����� �����
    if (outPixelIdx >= outH * outW) return;    

    float sum = 0.0f;
    for (int k = 0; k < kernelSize; k++) {                                  // ����� ��� ������� �� ������
        int pIdx = ((filterIdx * kernelSize) + k) * outH * outW + outPixelIdx;      // ����� ������� �� ����� ������ ���� ���� �� ������ �����
        sum += partial_outputs[pIdx];
    }
    int outIdx = filterIdx * outH * outW + outPixelIdx;              // ����� ������ �� ����� ������ ������ ���� ������� �� �����������
    output[outIdx] = sum;   // ���� �����������
}

//  API 
extern "C" int cuconv_conv_forward(                // ������� ������� ����� ���� ���� �� ���� ���� ��� ��� ��� ������ ����
    const float* x, const float* w, float* y,
    int N, int C, int H, int W,
    int M, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dil_h, int dil_w,
    int groups)
{
    if (N != 1 || groups != 1) return -1;              // ����� ������ ����� �� ��������
    if (stride_h != 1 || stride_w != 1) return -2;
    if (dil_h != 1 || dil_w != 1) return -3;

    int Ho = H + 2*pad_h - R + 1;              // ����� ���� ���� �����
    int Wo = W + 2*pad_w - S + 1;

    const int kernelSize = R * S;                 
    const bool is1x1 = (kernelSize == 1);         // ����� ����� ���� ��� ��� ���� ������� ����� �� ����� 1

    float* d_partial = nullptr;                           // ����� ����� �� ���� ������� �������
    if (!is1x1) {                                 
        size_t partial_sz = (size_t)kernelSize * M * Ho * Wo;        //�� ����� ����� ���� �� ����� ���� ����� ������ ������ �� ������ ������� �������
        cudaMalloc(&d_partial, partial_sz * sizeof(float));
    }
    float* out_or_partial = is1x1 ? y : d_partial;      //   ����� �� ����� ��� ����� ����� ���� �� ��� ���� ������ �� ������ �� ������� ������� 

    
    {
        int threads = 256;                                // ����� ���� ��'���� �����
        int blocks_x = (Ho * Wo + threads - 1) / threads;            // ���� ���� �� ���� ����� �� ������ ���� ���� �� ���� ������
        dim3 grid(blocks_x, M, kernelSize);                        //   ����� �����
        scalar_prods_kernel<<<grid, threads>>>(                       //    ���� �����
            x, w, out_or_partial,
            H, W,
            pad_h, pad_w,
            R, S,
            C, M,
            Ho, Wo
        );
        cudaGetLastError();               // ������� ������� �� ���� ������� �� ������ ������� ������
    }

   
    if (!is1x1) {                          //      ����� ������ ���� ����� �� ���� ����  
        int threads = 256;
        int blocks_x = (Ho * Wo + threads - 1) / threads;
        dim3 grid(blocks_x, M, 1);            //    ��� ����� �� ��� ��� �1
        sum_kernel<<<grid, threads>>>(
            d_partial, y, Ho, Wo, M, kernelSize
        );
        cudaGetLastError();
    }                                           

    if (!is1x1) cudaFree(d_partial);     //   �� ����� ����� ����� �� ���� �� ������� ������� ���� �� ������� �������          
    return 0;
}

