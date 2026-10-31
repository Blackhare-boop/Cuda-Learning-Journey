#include <stdio.h>
#include <cuda_runtime.h>

// 矩阵维度：为了测试明显，我们设为 1024 x 1024
#define N 1024

// 朴素版核函数：每个线程算一个元素
__global__ void matMulNaive(const float* A, const float* B, float* C, int n) {
    // 计算当前线程负责的行 (row) 和 列 (col)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        // 致命缺陷：每次循环都去全局显存 (DRAM) 里读数据！
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    size_t size = N * N * sizeof(float);

    // 1. 分配 CPU 内存
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // 初始化数据（随便塞点值）
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f; // 这样乘出来结果全是 N * 2.0
    }

    // 2. 分配 GPU 显存并拷贝
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 3. 设置线程块 (二维的！16x16 = 256个线程)
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 4. 计时并执行
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matMulNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // 等 GPU 算完

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("朴素版矩阵乘法耗时: %f 毫秒 (ms)\n", milliseconds);

    // 5. 拷回结果验证前 3 个
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printf("验证结果: C[0] = %.0f (理论应为 %d)\n", h_C[0], N * 2);

    // 6. 释放内存
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
