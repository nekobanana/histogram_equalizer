#ifndef HISTOGRAM_EQUALIZER_SCANNER_CUH
#define HISTOGRAM_EQUALIZER_SCANNER_CUH
#define HIST_SIZE 256
__global__ void Brent_Kung_scan_kernel(int *X, int *Y, int InputSize) {
    __shared__ int XY[HIST_SIZE];
    int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (i < InputSize) {
        XY[threadIdx.x] = X[i];
    }
    if (i + blockDim.x < InputSize) {
        XY[threadIdx.x+blockDim.x] = X[i+blockDim.x];
    }
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * 2 * stride -1;
        if (index < InputSize) {
            XY[index] += XY[index - stride];
        }
    }
    for (int stride = InputSize/4; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if(index + stride < InputSize) {
            XY[index + stride] += XY[index];
        }
    }
    __syncthreads();
    if (i < InputSize) Y[i] = XY[threadIdx.x];
    if (i + blockDim.x < InputSize) {
        Y[i + blockDim.x] = XY[threadIdx.x+  blockDim.x];
    }
}
#endif //HISTOGRAM_EQUALIZER_SCANNER_CUH
