# 🚀 CUDA Learning Journey

这是我的 CUDA 编程学习记录与实践仓库。在这里，我将从零开始，记录从底层硬件原理到高性能 GPU 代码优化的每一步。

## 🛠️ 开发环境 (Environment)
*   **OS**: Windows 11 + WSL2 (Ubuntu)
*   **GPU**: NVIDIA GeForce RTX 5070 (8GB VRAM)
*   **CUDA Toolkit**: 13.1
*   **Compiler**: nvcc 13.1.80

## 📂 代码目录 (Contents)
*   `add_1000.cu`: 基础的向量加法 (Vector Addition)，包含 CPU 与 GPU 结果比对。
*   `add_1e8.cu`: 针对 1 亿个数据量级的性能压测。通过引入 `cudaDeviceSynchronize()` 并利用 `clock()` 进行精准测时，直观对比 CPU 与 GPU 的算力差距。

## 📊 性能基准测试 (Benchmark)
在 1000万 (10 Million) 个 float 数据的向量加法测试中：
*   **CPU 耗时**: ~ 0.63 秒
*   **GPU 耗时**: ~ 0.08 秒
*   **加速比**: **~ 8x** 

> *注：目前的代码仅为基础实现，后续将引入 Shared Memory (共享内存) 和 CUDA Streams (流) 进一步压榨 RTX 5070 的性能。*

## 🗺️ 学习路线 (Roadmap)
- [x] 搭建 WSL2 + CUDA 开发环境
- [x] 掌握 Host 与 Device 内存管理及数据拷贝 (`cudaMalloc`, `cudaMemcpy`)
- [x] 理解 Grid, Block, Thread 的层级调度
- [x] 向量加法与性能基准测试
- [ ] **矩阵乘法 (Matrix Multiplication) 与 Shared Memory 优化 (Tiling)**
- [ ] 归约算法 (Reduction)
- [ ] 学习并复现 FlashAttention 核心算子

---
*Keep Coding, Keep Optimizing!*
