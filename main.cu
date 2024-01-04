#include <iostream>
#include <opencv2/opencv.hpp>
#include "Scanner.cuh"

#define COLOR_DEPTH 8
#define HIST_SIZE (2 << (COLOR_DEPTH - 1))
#define BLOCK_DIM 16

__constant__ int d_CDF[HIST_SIZE];

__global__ void histogram(const unsigned char* image, int* hist, int width, int height) {
    __shared__ int HIST[HIST_SIZE];
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int t_id = tx + ty * width;
    int t_id_block = threadIdx.x + threadIdx.y * BLOCK_DIM;
    if (t_id_block < HIST_SIZE) {
        HIST[t_id_block] = 0;
    }
    __syncthreads(); // dopo inizializzazione shared memory
    if (tx < width && ty < height) {
        atomicAdd(&(HIST[image[t_id]]), 1);
    }
    __syncthreads(); // per sincronizzare i thread all'interno dello stesso blocco prima di sommare
                     // i contenuti delle shared memory sulla global memory
    if (t_id_block < HIST_SIZE) {
        atomicAdd(&(hist[t_id_block]), HIST[t_id_block]);
    }
}
__global__ void equalizer(unsigned char* image, int cdf_val_min, int width, int height) {
    __shared__ unsigned char IMG[BLOCK_DIM * BLOCK_DIM];
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int t_id = tx + ty * width;
    int t_id_block = threadIdx.x + threadIdx.y * BLOCK_DIM;
    if (tx < width && ty < height) {
        IMG[t_id_block] = image[t_id];
    }
    __syncthreads(); // dopo inizializzazione shared memory
    if (tx < width && ty < height) {
        IMG[t_id_block] = (d_CDF[IMG[t_id_block]] - cdf_val_min) * (HIST_SIZE - 1) / (width * height - cdf_val_min);
    }
    __syncthreads();
    if (tx < width && ty < height) {
        image[t_id] = IMG[t_id_block];
    }
}
int main() {
//    cv::Mat image = cv::imread("/home/quacksort/CLionProjects/histogram_equalizer/images/0.bmp", cv::IMREAD_GRAYSCALE);
    cv::Mat origImage = cv::imread("/home/quacksort/CLionProjects/histogram_equalizer/images/0.bmp");

    if (!origImage.data)
    {
        printf("Image not found\n");
        return -1;
    }
    int width = origImage.cols;
    int height = origImage.rows;
    cv::Mat imageHSV;
    cvtColor(origImage, imageHSV, cv::COLOR_BGR2HSV);
    cv::Mat HSVchannels[3];
    split(imageHSV, HSVchannels);
    cv::Mat image = HSVchannels[2];
    unsigned char* h_img = image.data;

//    int *h_img = (int *)malloc(width * height * sizeof(int));
//    for (int row = 0; row < height; row++)
//    {
//        for (int col = 0; col < width; col++)
//        {
//            h_img[row * width + col] = image.at<uchar>(row, col);
//        }
//    }

    int h_hist[HIST_SIZE];
    for (int i = 0; i < HIST_SIZE; i++) {
        h_hist[i] = 0;
    }
    unsigned char *d_img;
    int *d_hist;
    cudaMalloc(&d_img, width * height * sizeof(unsigned char));
    cudaMalloc(&d_hist, HIST_SIZE * sizeof(int));
    cudaMemcpy(d_img, h_img, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hist, h_hist, HIST_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    dim3 block_dim = dim3(BLOCK_DIM, BLOCK_DIM);
    dim3 grid_dim = dim3((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM);
    histogram<<<grid_dim, block_dim>>>(d_img, d_hist, width, height);
    auto r1 = cudaDeviceSynchronize();
    auto r2 = cudaGetLastError();
    cudaMemcpy(h_hist, d_hist, HIST_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

//    for (int i = 1; i < HIST_SIZE; i++) {
//        printf("h_hist[%d]: %d\n", i, h_hist[i]);
//    }

    //Calcolo la CDF
    int h_cdf[HIST_SIZE];
    int *d_cdf;
    cudaMalloc(&d_cdf, HIST_SIZE * sizeof(int));

    Brent_Kung_scan_kernel<<<1, HIST_SIZE/2, HIST_SIZE>>>(d_hist, d_cdf, HIST_SIZE);
    r1 = cudaDeviceSynchronize();
    r2 = cudaGetLastError();
    cudaMemcpy(h_cdf, d_cdf, HIST_SIZE * sizeof(int), cudaMemcpyDeviceToHost);


    for (int i = 1; i < HIST_SIZE; i++) {
        printf("CDF[%d]: %d\n", i, h_cdf[i]);
    }
    int cdf_val_min = h_cdf[HIST_SIZE - 1];
    //    h_cdf[0] = h_hist[0];
    for (int i = 1; i < HIST_SIZE; i++) {
//        h_cdf[i] = h_cdf[i-1] + h_hist[i];
        if (h_cdf[i] > 0) {
            cdf_val_min = h_cdf[i];
            break;
        }
    }
//    printf("min: %d\n", cdf_val_min);

    cudaMemcpyToSymbol(d_CDF, h_cdf, HIST_SIZE * sizeof(int));
//    cudaMemcpy(d_img, h_img, width * height * sizeof(int), cudaMemcpyHostToDevice);
    equalizer<<<grid_dim, block_dim>>>(d_img, cdf_val_min, width, height);
    r1 = cudaDeviceSynchronize();
    r2 = cudaGetLastError();
    cudaMemcpy(h_img, d_img, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_img);
    cudaFree(d_hist);

//    char *char_img_eq = (char *)malloc(width * height * sizeof(char));
//    for (int i = 0; i < width * height; i++)
//    {
//        char_img_eq[i] = static_cast<char>(h_img[i]);
//    }

    cv::Mat valueEq = cv::Mat(width, height, CV_8U, h_img);
    HSVchannels[2] = valueEq;
    cv::merge(HSVchannels, 3, imageHSV);
    cv::Mat eqImageBGR;
    cvtColor(imageHSV, eqImageBGR, cv::COLOR_HSV2BGR);
    cv::imwrite("/home/quacksort/CLionProjects/histogram_equalizer/images/result.bmp", eqImageBGR);

//    free(h_img);
//    free(char_img_eq);

    return 0;
}
