

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include "softmax.cuh"


#define TIME_WARMUP 10
#define TIME_ITERS 100

#define PROFILE_WARMUP 0
#define PROFILE_ITERS 1

// #define WARMUP PROFILE_WARMUP
// #define ITERS PROFILE_ITERS

#define WARMUP TIME_WARMUP
#define ITERS TIME_ITERS



#define CUDA_CHECK(x)                                                         \
    do {                                                                      \
        cudaError_t e = (x);                                                  \
        if (e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n",                       \
                    cudaGetErrorString(e), __FILE__, __LINE__);               \
            exit(1);                                                          \
        }                                                                     \
    } while (0)



static void softmax_cpu(const float* in, double* out, int rows, int N)
{
    for(int r =0; r< rows; r++)
    {
        double max_x = -INFINITY;
        for(int c=0; c < N; c++) max_x = std::max(max_x, (double)in[r*N + c]);
        double s = 0.0;
        for(int c=0; c < N; c++) s += std::exp((double)in[r*N + c] - max_x);
        for(int c=0; c < N; c++) out[r*N + c] = std::exp((double)in[r*N + c] - max_x) / s;
    }
}

using SoftmaxFn = void (*)(const float*, float*, int, int, cudaStream_t);


void softmax_v0(const float* in, float* out, int M, int N, cudaStream_t stream) {
    dim3 block((M + 63) / 64);
    dim3 grid(64);
    _softmax_v0<<<grid, block, 0, stream>>>(in, out, M, N);
}

void softmax_v0_warp(const float* in, float* out, int M, int N, cudaStream_t stream) {
    dim3 block(64);
    dim3 grid((M + 63) / 64);
    _softmax_v0<<<grid, block, 0, stream>>>(in, out, M, N);
}

void softmax_v1(const float* in, float* out, int M, int N, cudaStream_t stream) {
    dim3 block(32);
    dim3 grid(M);
    _softmax_v1<<<grid, block, 0, stream>>>(in, out, M, N);
}

void softmax_v2(const float* in, float* out, int M, int N, cudaStream_t stream) {
    dim3 block(32);
    dim3 grid(M);
    _softmax_v2<<<grid, block, 0, stream>>>(in, out, M, N);
}

void softmax_v3(const float* in, float* out, int M, int N, cudaStream_t stream) {
    dim3 block(32);
    dim3 grid(M);
    _softmax_v3<<<grid, block, 0, stream>>>(in, out, M, N);
}

void softmax_v4(const float* in, float* out, int M, int N, cudaStream_t stream) {
    dim3 block(128);
    dim3 grid(M);
    _softmax_v4<<<grid, block, 0, stream>>>(in, out, M, N);
}

void softmax_v4_256(const float* in, float* out, int M, int N, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(M);
    _softmax_v4<<<grid, block, 0, stream>>>(in, out, M, N);
}

void softmax_v5(const float* in, float* out, int M, int N, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(M);
    _softmax_v5<<<grid, block, 0, stream>>>(in, out, M, N);
}

void softmax_v6(const float* in, float* out, int M, int N, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(M);
    _softmax_v6<<<grid, block, 0, stream>>>(in, out, M, N);
}

void softmax_v7(const float* in, float* out, int M, int N, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(M);
    _softmax_v7<<<grid, block, 0, stream>>>(in, out, M, N);
}

void softmax_v7_512(const float* in, float* out, int M, int N, cudaStream_t stream) {
    dim3 block(512);
    dim3 grid(M);
    _softmax_v7<<<grid, block, 0, stream>>>(in, out, M, N);
}


void softmax_v7_1024(const float* in, float* out, int M, int N, cudaStream_t stream) {
    dim3 block(1024);
    dim3 grid(M);
    _softmax_v7<<<grid, block, 0, stream>>>(in, out, M, N);
}

struct Variant {const char* name; SoftmaxFn fn;} ;
static const Variant softmax_variants[] = {
    {"1. v0_thread_per_row", softmax_v0},
    {"2. v0_thread_per_row_warp", softmax_v0_warp},
    {"3. v1_block_per_row", softmax_v1},
    {"4. v2_block_per_row_warp_shfl", softmax_v2},
    {"5. v3_block_per_row_warp_shfl_float4", softmax_v3},
    {"6. v4_block_per_row_warp_shfl_float4_multiwarp", softmax_v4},
    {"7. v4_256_block_per_row_warp_shfl_float4_multiwarp", softmax_v4_256},
    {"9. v5_256_online_shfl_float4_multiwarp_strided", softmax_v5},
    {"10. v6_256_online_shfl_float4_multiwarp_strided", softmax_v6},
    {"11. v7_256_online_shfl_float4_multiwarp_strided", softmax_v7},
    {"12. v7_512_online_shfl_float4_multiwarp_strided", softmax_v7_512},
    {"13. v7_1024_online_shfl_float4_multiwarp_strided", softmax_v7_1024},


};

static float time_kernel(SoftmaxFn fn, const float* d_in, float *d_out, int M, int N, int warmup = WARMUP, int iters = ITERS)
{
    for(int i =0; i< warmup; i++) fn(d_in, d_out, M, N, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for(int i =0; i< iters; i++) fn(d_in, d_out, M, N, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));     

    return ms / iters;
}


int main(int argc, char** argv)
{
    const int M = (argc > 1) ? atoi(argv[1]) : 65536; // 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576
    const int Ns[] = {1024, 2048, 4096, 8192, 16384};
    // const int Ns[] = {128, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576};
    
    const int check_rows = std::min(M, 256);   // CPU 검증은 앞쪽 일부 행만

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s, M = %d\n\n", prop.name, M);

    const char* csv_path = (argc > 2) ? argv[2] : "1_softmax/results/softmax_versions.csv";
    FILE* csv = fopen(csv_path, "w");
    if (!csv) { fprintf(stderr, "cannot open %s\n", csv_path); return 1; }
    fprintf(csv, "gpu,M,variant,N,time_us,eff_bw_gbps,max_abs_err\n");

    printf("%-50s %8s %12s %12s %14s\n", "variant", "N", "time(us)", "effBW(GB/s)", "max_abs_err");

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.f, 10.f);

    for (int N : Ns) {
        const size_t numel = (size_t)M * N;
        const size_t bytes = numel * sizeof(float);

        std::vector<float> h_in(numel), h_out(numel);
        for (auto& v : h_in) v = dist(rng);

        std::vector<double> h_ref((size_t)check_rows * N);
        softmax_cpu(h_in.data(), h_ref.data(), check_rows, N);

        float *d_in = nullptr, *d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

        for (const auto& v : softmax_variants) {
            // 1) 정답 확인
            CUDA_CHECK(cudaMemset(d_out, 0, bytes));
            v.fn(d_in, d_out, M, N, 0);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, (size_t)check_rows * N * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            double max_err = 0.0;
            for (size_t i = 0; i < (size_t)check_rows * N; ++i)
                max_err = std::max(max_err, std::fabs((double)h_out[i] - h_ref[i]));

            // 2) 시간 측정 및 effective bandwidth (이상적 트래픽: 읽기 1번 + 쓰기 1번)
            const float ms = time_kernel(v.fn, d_in, d_out, M, N);
            const double gbps = 2.0 * bytes / (ms * 1e-3) / 1e9;

            printf("%-50s %8d %12.2f %12.1f %14.3e\n", v.name, N, ms * 1e3, gbps, max_err);
            fprintf(csv, "%s,%d,%s,%d,%.2f,%.1f,%.3e\n", prop.name, M, v.name, N, ms * 1e3, gbps, max_err);
        }

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }
    fclose(csv);
    return 0;
}
