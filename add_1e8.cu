#include <stdio.h>
#include <cuda_runtime.h> //CUDA的灵魂，实现很多操作
#include <ctime>


__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {   //global告诉CPU，该指令让GPU去跑
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  //类比班级、座位号，确认学号
    if (idx < N) {  //防止越界
        C[idx] = A[idx] + B[idx];  //存储元素
    }
}


int main() {
    int N = 100000000;  //需要计算1000个数据
    size_t size = N * sizeof(float);    //总共的字节数

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);  //在CPU中划出空间

    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }  //赋予数组初始数

    // --- CPU 飙车开始 ---
    clock_t cpu_start = clock();
    for (int i = 0; i < N; i++) {
        h_C[i] = h_A[i] + h_B[i];
    }
    clock_t cpu_end = clock();
    double cpu_time = double(cpu_end - cpu_start) / CLOCKS_PER_SEC;
    // --- CPU 飙车结束 ---


    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);  //在GPU中划出空间

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);  //将CPU中的数据拷贝到DPU中

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;  //确认班级数

    clock_t gpu_start = clock(); // 计时开始！

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);  //用4个班执行256个线程，去执行GPU上vectorAdd函数

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);  //把GPU的结果搬回CPU

    // 这里换成了 C 语言的 printf
    for (int i = 0; i < 5; i++) {
        printf("%.0f + %.0f = %.0f\n", h_A[i], h_B[i], h_C[i]);
    }  //执行打印验证

    // 看看第 999 个（也就是最后一个）数据算得对不对
    printf("%.0f + %.0f = %.0f\n", h_A[999], h_B[999], h_C[999]);

    cudaDeviceSynchronize(); // 强制 CPU 等待 GPU 算完
    clock_t gpu_end = clock();
    double gpu_time = double(gpu_end - gpu_start) / CLOCKS_PER_SEC;

    printf("🚀 CPU 算 %d 个加法耗时: %f 秒\n", N, cpu_time);
    printf("⚡ GPU 算 %d 个加法耗时: %f 秒\n", N, gpu_time);


    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);  //释放GPU显存与CPU内存

    return 0;
}
