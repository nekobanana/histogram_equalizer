#include <iostream>
#include <opencv2/opencv.hpp>

#define COLOR_DEPTH 8
#define HIST_SIZE 2 << (COLOR_DEPTH - 1)
#define BLOCK_DIM 16

__global__ void histogram(const int* image, int* hist, int width, int height) {
    __shared__ int HIST[HIST_SIZE];
    __shared__ int IMG[BLOCK_DIM * BLOCK_DIM];
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int t_id = tx + ty * width;
    int t_id_block = threadIdx.x + threadIdx.y * BLOCK_DIM;
    if (t_id_block < HIST_SIZE) {
        HIST[t_id_block] = 0;
    }
    __syncthreads();
    if (tx < width && ty < height) {
        IMG[t_id_block] = image[t_id];
    }
    __syncthreads();
    if (tx < width && ty < height) {
        atomicAdd(&(HIST[IMG[t_id_block]]), 1);
    }
    __syncthreads();
    if (t_id_block < HIST_SIZE) {
        atomicAdd(&(hist[t_id_block]), HIST[t_id_block]);
    }
}

int main() {
    cv::Mat image = cv::imread("/home/quacksort/CLionProjects/histogram_equalizer/images/0.bmp", cv::IMREAD_GRAYSCALE);
//    cv::Mat image = cv::imread("/home/quacksort/CLionProjects/histogram_equalizer/images/0.bmp");
    cv::imwrite("/home/quacksort/CLionProjects/histogram_equalizer/images/test.bmp", image);
    if (!image.data)
    {
        printf("Image not found\n");
        return -1;
    }
    int width = image.cols;
    int height = image.rows;
    int *h_img = (int *)malloc(width * height * sizeof(int));
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            h_img[i * width + j] = image.at<uchar>(i, j);
        }
    }

    int *h_hist = ((int *)malloc(HIST_SIZE * sizeof(int)));
    int *d_img;
    int *d_hist;
    cudaMalloc(&d_img, width * height * sizeof(int));
    cudaMalloc(&d_hist, HIST_SIZE * sizeof(int));
    cudaMemcpy(d_img, h_img, width * height * sizeof(int), cudaMemcpyHostToDevice);
    dim3 block_dim = dim3(BLOCK_DIM, BLOCK_DIM);
    dim3 grid_dim = dim3((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM);
    histogram<<<grid_dim, block_dim>>>(d_img, d_hist, width, height);
    auto r1 = cudaDeviceSynchronize();
    auto r2 = cudaGetLastError();
    cudaMemcpy(h_hist, d_hist, HIST_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_img);
    cudaFree(d_hist);


    auto *test_hist = ((int *)malloc(HIST_SIZE * sizeof(int)));
    for (int i = 0; i < HIST_SIZE; i++)
    {
        test_hist[i] = 0;
    }
    for (int i = 0; i < width * height; i++)
    {
        test_hist[h_img[i]]++;
    }
    bool equal = true;
    for (int i = 0; i < HIST_SIZE; i++)
    {
        equal = equal && test_hist[i] == h_hist[i];
//        printf("hist[%d]: %d\n", i, h_hist[i]);
    }
    if (equal) {
        std::cout << "SUCCESS" << std::endl;
    }
    else {
        std::cout << "FAILURE" << std::endl;
    }

    free(h_img);
    free(h_hist);
    free(test_hist);

    return 0;
}
