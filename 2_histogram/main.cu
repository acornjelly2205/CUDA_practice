#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <climits>
#include <algorithm>
#include <cub/cub.cuh>

#include "histogram.cuh"

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




using HistogramFn = void (*)(const uint32_t*, uint32_t*, size_t, int, cudaStream_t, uint32_t*, uint32_t*);
enum class Dist { Uniform, Hot90, Sorted, AllSame };

static const char* dist_name(Dist d)
{
    switch (d) {
        case Dist::Uniform: return "uniform";
        case Dist::Hot90:   return "hot90";    // 90%가 한 bin → 같은 주소 contention
        case Dist::Sorted:  return "sorted";   // 값 분포는 uniform, 순서만 정렬 → warp 안에서 같은 bin
        case Dist::AllSame: return "all_same"; // 최악 contention
    }
    return "?";
}





void histogram_naive(const uint32_t* d_input, uint32_t* hist, size_t N, int num_bins, cudaStream_t stream, uint32_t* d_input_dup, uint32_t* d_sorted_input) {
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    _histogram_naive<<<grid, block, 0, stream>>>(d_input, hist, N, num_bins);
}

void histogram_float4(const uint32_t* d_input, uint32_t* hist, size_t N, int num_bins, cudaStream_t stream, uint32_t* d_input_dup, uint32_t* d_sorted_input) {
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    _histogram_float4<<<grid, block, 0, stream>>>(d_input, hist, N, num_bins);
}

void histogram_sub(const uint32_t* d_input, uint32_t* hist, size_t N, int num_bins, cudaStream_t stream, uint32_t* d_input_dup, uint32_t* d_sorted_input) {

    size_t shared_mem_size = num_bins * sizeof(uint32_t);
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    _histogram_sub<<<grid, block, shared_mem_size, stream>>>(d_input, hist, N, num_bins);
}

void histogram_warp_aggr(const uint32_t* d_input, uint32_t* hist, size_t N, int num_bins, cudaStream_t stream, uint32_t* d_input_dup, uint32_t* d_sorted_input) {

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    size_t shared_mem_size = num_bins * sizeof(uint32_t);
    _histogram_warp_aggregate<<<grid, block, shared_mem_size, stream>>>(d_input, hist, N, num_bins);
}


void histogram_warp_aggr_512(const uint32_t* d_input, uint32_t* hist, size_t N, int num_bins, cudaStream_t stream, uint32_t* d_input_dup, uint32_t* d_sorted_input) {

    dim3 block(512);
    dim3 grid((N + block.x - 1) / block.x);
    size_t shared_mem_size = num_bins * sizeof(uint32_t);
    _histogram_warp_aggregate<<<grid, block, shared_mem_size, stream>>>(d_input, hist, N, num_bins);
}



void histogram_warp_aggr_1024(const uint32_t* d_input, uint32_t* hist, size_t N, int num_bins, cudaStream_t stream, uint32_t* d_input_dup, uint32_t* d_sorted_input) {

    dim3 block(1024);
    dim3 grid((N + block.x - 1) / block.x);
    size_t shared_mem_size = num_bins * sizeof(uint32_t);
    _histogram_warp_aggregate<<<grid, block, shared_mem_size, stream>>>(d_input, hist, N, num_bins);
}


void histogram_with_raidx_sort(const uint32_t* d_input, uint32_t* hist, size_t N, int num_bins, cudaStream_t stream, uint32_t* d_input_dup, uint32_t* d_sorted_input) {

    
    dim3 block(256);
    dim3 grid(1);
    radix_sort_pass<<<grid, block, 0, stream>>>(d_input_dup, N, d_sorted_input, 0);
    radix_sort_pass<<<grid, block, 0, stream>>>(d_sorted_input, N, d_input_dup, 8);
    radix_sort_pass<<<grid, block, 0, stream>>>(d_input_dup, N, d_sorted_input, 16);
    radix_sort_pass<<<grid, block, 0, stream>>>(d_sorted_input, N, d_input_dup, 24);

    dim3 block_hist(256);
    dim3 grid_hist(1);

    histogram_with_sorted_input<<<grid_hist, block_hist, num_bins * sizeof(uint32_t), stream>>>(d_input_dup, hist, N, num_bins);
}






static void*  g_cub_temp = nullptr;
static size_t g_cub_temp_bytes = 0;

void histogram_cub(const uint32_t* d_input, uint32_t* hist, size_t N, int num_bins, cudaStream_t stream, uint32_t* d_input_dup, uint32_t* d_sorted_input) {
    size_t need = 0;
    
    CUDA_CHECK(cub::DeviceHistogram::HistogramEven( nullptr, need, d_input, hist, num_bins + 1, (uint32_t)0, (uint32_t)num_bins, (int)N, stream));
    if (need > g_cub_temp_bytes) {
        if (g_cub_temp) CUDA_CHECK(cudaFree(g_cub_temp));
        CUDA_CHECK(cudaMalloc(&g_cub_temp, need));
        g_cub_temp_bytes = need;
    }
    CUDA_CHECK(cub::DeviceHistogram::HistogramEven(g_cub_temp, need, d_input, hist, num_bins + 1, (uint32_t)0, (uint32_t)num_bins, (int)N, stream));
}


struct Variant {const char* name; HistogramFn fn;} ;
static const Variant histogram_variants[] = {
    {"0. CUB::DeviceHistogram::HistogramEven", histogram_cub},
    {"1. atomic_naive", histogram_naive},
    {"2. atomic_float4", histogram_float4},
    {"3. atomic_subhist", histogram_sub},
    {"4. atomic_warp_aggr", histogram_warp_aggr},
    {"5. atomic_warp_aggr_512", histogram_warp_aggr_512},
    {"6. atomic_warp_aggr_1024", histogram_warp_aggr_1024},
    {"7. radix_sort + without atomic", histogram_with_raidx_sort},
    
};

static void run_once(HistogramFn fn, const uint32_t* d_keys, uint32_t* d_hist, size_t N, int num_bins, uint32_t* d_input_dup = nullptr, uint32_t* d_sorted_input = nullptr)
{
    CUDA_CHECK(cudaMemsetAsync(d_hist, 0, num_bins * sizeof(uint32_t), 0));
    fn(d_keys, d_hist, N, num_bins, 0, d_input_dup, d_sorted_input);
}

static float time_kernel(HistogramFn fn, const uint32_t* d_keys, uint32_t* d_hist, size_t N, int num_bins,
                         int warmup = WARMUP, int iters = ITERS, uint32_t* d_input_dup = nullptr, uint32_t* d_sorted_input = nullptr)
{
    for (int i = 0; i < warmup; i++) run_once(fn, d_keys, d_hist, N, num_bins, d_input_dup, d_sorted_input);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) run_once(fn, d_keys, d_hist, N, num_bins, d_input_dup, d_sorted_input);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}



std::vector<int> make_input(
    size_t n,
    int num_bins,
    Dist dist,
    unsigned seed = 42
) {
    std::mt19937 rng(seed);

    std::uniform_int_distribution<int> uniform(0, num_bins - 1);
    std::uniform_int_distribution<int> few(
        0, std::min(8, num_bins) - 1
    );
    std::uniform_int_distribution<int> other(1, num_bins - 1);
    std::bernoulli_distribution hot(0.9);

    std::vector<int> input(n);

    for (auto& x : input) {
        switch (dist) {
            case Dist::Uniform:
                x = uniform(rng);
                break;
            case Dist::Hot90:
                x = hot(rng) ? 0 : other(rng);
                break;
            case Dist::Sorted:
                x = uniform(rng);
                sort(input.begin(), input.end());
                break;
            case Dist::AllSame:
                x = 0;
                break;  

        }
    }

    return input;
}

int main(int argc, char **argv) {
    

    // const size_t N = (argc > 1) ? strtoull(argv[1], nullptr, 10) : (1ull << 26);
    const size_t N = 2024;
    // const int bin_size[] = {256, 512, 1024, 2048, 4096, 8192, 16384};
    const int bin_size[] = {512, 2048, 5000, 8192};
    
    // const Dist dist_type[] = {Dist::Uniform, Dist::Hot90, Dist::Sorted, Dist::AllSame};
    const Dist dist_type[] = {Dist::Uniform, Dist::Hot90};

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    auto g_num_sms = prop.multiProcessorCount;
    printf("GPU: %s, SMs = %d, N = %zu (%.1f MB)\n\n", prop.name, g_num_sms, N, N * sizeof(uint32_t) / 1e6);

    for(auto bin: bin_size) {
        for(auto dist: dist_type) {
            const char* dist_name = (dist == Dist::Uniform) ? "Uniform" :
                                    (dist == Dist::Hot90) ? "Hot90" :
                                    (dist == Dist::Sorted) ? "Sorted" : "AllSame";
            printf("bin_size = %d, distribution = %s\n", bin, dist_name);

            std::vector<int> h_in = make_input(N, bin, dist);
            std::vector<uint32_t> h_out(bin);

            std::vector<uint32_t> h_ref(bin);
            for(auto v: histogram_variants) {
                uint32_t *d_in, *d_out;
                CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(uint32_t)));
                CUDA_CHECK(cudaMalloc(&d_out, bin * sizeof(uint32_t)));
                CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemset(d_out, 0, bin * sizeof(uint32_t)));

                uint32_t *d_input_dup, *d_sorted_input;
                CUDA_CHECK(cudaMalloc(&d_input_dup, N * sizeof(uint32_t)));
                CUDA_CHECK(cudaMalloc(&d_sorted_input, N * sizeof(uint32_t)));
                CUDA_CHECK(cudaMemcpy(d_input_dup, d_in, N * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_sorted_input, d_in, N * sizeof(uint32_t), cudaMemcpyDeviceToDevice));

                float ms = time_kernel(v.fn, d_in, d_out, N, bin, WARMUP, ITERS, d_input_dup, d_sorted_input);

                CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bin * sizeof(uint32_t), cudaMemcpyDeviceToHost));
                if(v.name == histogram_variants[0].name) {
                    h_ref = h_out;
                }

                long long  max_abs_err = 0;
                for(int i=0; i<bin; i++) {
                    const long long diff = static_cast<long long>(h_ref[i]) - static_cast<long long>(h_out[i]);
                    max_abs_err = std::max(max_abs_err, std::abs(diff));
                }

                float eff_bw_gbps = (N * sizeof(uint32_t) + bin * sizeof(uint32_t)) / (ms / 1e3) / 1e9;

                printf("%-50s %8d %12.3f %12.3f %14d\n", v.name, bin, ms*1e3, eff_bw_gbps, max_abs_err);

                CUDA_CHECK(cudaFree(d_in));
                CUDA_CHECK(cudaFree(d_out));
            }
        }
    }
    




}


