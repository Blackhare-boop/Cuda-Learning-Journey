#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024
#define TILE_SIZE 16

// 优化版核函数：使用共享内存
__global__ void matMulOptimized(const float* A, const float* B, float* C, int n) {
    // 声明两辆共享内存“小推车”
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    // 核心：按块循环（比如 1024 的矩阵，要循环 1024/16 = 64 次）
    for (int m = 0; m < n / TILE_SIZE; m++) {
        // 1. 把全局显存的数据搬进共享内存（每个线程只搬一个数）
        sA[threadIdx.y][threadIdx.x] = A[row * n + (m * TILE_SIZE + threadIdx.x)];
        sB[threadIdx.y][threadIdx.x] = B[(m * TILE_SIZE + threadIdx.y) * n + col];

        // 2. 极其重要：等全班同学都搬完，才能开始算！不然同步不一致
        __syncthreads();

        // 3. 在共享内存里做乘法
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        // 4. 极其重要：算完了，等大家算完这一块，再去搬下一块
        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

int main() {
    size_t size = N * N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f; 
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    // 这里调用的是优化版的核函数
    matMulOptimized<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("🚀 优化版矩阵乘法耗时: %f 毫秒 (ms)\n", milliseconds);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printf("验证结果: C[0] = %.0f (理论应为 %d)\n", h_C[0], N * 2);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
