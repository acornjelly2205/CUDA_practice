#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

void __global__ _histogram_naive(const uint32_t* d_input, uint32_t* hist, size_t N, int num_bins)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(&hist[d_input[idx]], 1);
        return ;
    }

    for(int i = idx; i< N; i+= blockDim.x * gridDim.x) {
        atomicAdd(&hist[d_input[i]], 1);
    }

    return;
}


void __global__ _histogram_float4(const uint32_t* d_input, uint32_t* hist, size_t N, int num_bins)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(&hist[d_input[idx]], 1);
        return ;
    }

    if(idx * 4 >= N) {
        return;
    }

    int i = idx << 2;
    for(; i< N; i+= blockDim.x * gridDim.x * 4) {

        uint4 val = reinterpret_cast<const uint4*>(d_input)[i];
        atomicAdd(&hist[d_input[val.x]], 1);
        atomicAdd(&hist[d_input[val.y]], 1);
        atomicAdd(&hist[d_input[val.z]], 1);
        atomicAdd(&hist[d_input[val.w]], 1);
    }

    for(; i< N; i+= blockDim.x * gridDim.x) {
        atomicAdd(&hist[d_input[i]], 1);
    }

    return;
}

void __global__ _histogram_sub(const uint32_t* d_input, uint32_t* hist, size_t N, int num_bins)
{
    
    extern __shared__ uint32_t sub_hist[];

    for(int i = threadIdx.x; i < num_bins; i+= blockDim.x) {
        if(i < num_bins) sub_hist[i] = 0;
    }
    __syncthreads();

    const int size_per_block = (N + gridDim.x - 1) / gridDim.x;
    const int start = blockIdx.x * size_per_block;
    const int end = min(start + size_per_block, (int)N);

    for(int i = start + threadIdx.x; i < end; i+= blockDim.x) {
        if(i < end) atomicAdd(&sub_hist[d_input[i]], 1);
    }
    __syncthreads();

    for(int i = threadIdx.x; i < num_bins; i+= blockDim.x) {
        if(i < num_bins && sub_hist[i] > 0) {
            atomicAdd(&hist[i], sub_hist[i]);
        }
    }

}


void __global__ _histogram_warp_aggregate(const uint32_t* d_input, uint32_t* hist, size_t N, int num_bins)
{

    extern __shared__ uint32_t sub_hist[];

    for(int i = threadIdx.x; i < num_bins; i+= blockDim.x) {
        if(i < num_bins) sub_hist[i] = 0;
    }
    __syncthreads();

    uint32_t val;

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i+= blockDim.x * gridDim.x) {
        val = d_input[i];

        unsigned int mask = __activemask();
        unsigned int maks_same = __match_any_sync(mask, val);   
        
        if(threadIdx.x% 32 == __ffs(maks_same) - 1)
        {
            atomicAdd(&sub_hist[val], __popc(maks_same));
        }
    }

    __syncthreads();

    for(int i = threadIdx.x; i < num_bins; i+= blockDim.x) {
        if(i < num_bins && sub_hist[i] > 0) {
            atomicAdd(&hist[i], sub_hist[i]);
        }
    }

}


// 반드시 한 블록으로 실행.
// d_input과 d_output은 서로 다른 버퍼.
// shared count/start가 int이므로 N <= INT_MAX 전제.
__global__ void radix_sort_pass(const uint32_t* d_input, size_t N, uint32_t* d_output, int shift)
 {
    __shared__ int shared_count[256];
    __shared__ int shared_start[256];

    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        shared_count[i] = 0;
    }
    __syncthreads();

    for (size_t idx = threadIdx.x; idx < N; idx += blockDim.x) {
        unsigned int digit = (d_input[idx] >> shift) & 0xFFu;
        atomicAdd(&shared_count[digit], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        int sum = 0;

        for (int j = 0; j < i; ++j) {
            sum += shared_count[j];
        }

        shared_start[i] = sum;
    }
    __syncthreads();

    for (size_t idx = threadIdx.x; idx < N; idx += blockDim.x) {
        uint32_t val = d_input[idx];
        unsigned int digit = (val >> shift) & 0xFFu;

        size_t inner_index = 0;

        for (size_t i = 0; i < idx; ++i) {
            unsigned int prev_digit = (d_input[i] >> shift) & 0xFFu;

            if (prev_digit == digit) {
                ++inner_index;
            }
        }

        size_t dst =
            static_cast<size_t>(shared_start[digit]) + inner_index;

        d_output[dst] = val;
    }
}


__global__ void histogram_with_sorted_input( const uint32_t* d_input, uint32_t* hist, size_t N, int num_bins)
 {
    if (num_bins <= 0) return;

    size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (
        size_t idx =
            static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        idx < N;
        idx += stride
    ) {
        uint32_t value = d_input[idx];

        if (idx > 0 && d_input[idx - 1] == value) {
            continue;
        }

        if (value >= static_cast<uint32_t>(num_bins)) {
            continue;
        }

        size_t end = idx + 1;
        while (end < N && d_input[end] == value) {
            ++end;
        }

        hist[value] = static_cast<uint32_t>(end - idx);
    }
}