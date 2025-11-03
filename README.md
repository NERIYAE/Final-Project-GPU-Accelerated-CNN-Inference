# Final-Project-GPU-Accelerated-CNN-Inference

CUDA C++ implementation of the 2D convolution **forward pass** for CNN inference.  
Includes a **split** design (dot products + reduction) and a **fused** kernel that accumulates and writes outputs in one pass. 
Focus: memory coalescing, shared memory for filters, read only cache.

## Overview
- Data layout: **NCHW**, N=1, stride=1, symmetric padding.
- Kernels: `scalar_prods_kernel`, `sum_kernel`, `fused_conv_kernel`.
- CPU baseline for numerical validation.
- Optional profiling with **Nsight Compute**.

## Project layout
- `src/cuconv_lib.cu` — CUDA kernels.
- `src/main.cpp` — CLI, data setup, GPU run, validation hooks.
- `CPU_RUN.cpp` — CPU version.
