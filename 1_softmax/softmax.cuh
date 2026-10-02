
#pragma once
#include <cmath>
#include "cuda_runtime.h"

__global__ void _softmax_v0(const float* __restrict__ in,
                            float* __restrict__ out,
                            int M, int N)
{
    int r = (blockDim.x * blockIdx.x + threadIdx.x < M)? blockDim.x * blockIdx.x + threadIdx.x : -1;
    auto x = in + r * N;
    auto y = out + r * N;
    float max_val = x[0];
    float m  = 0.0f;

    if (r > -1) 
    {
        for(int i = 0; i < N; i++)
        {
            max_val = fmaxf(max_val, x[i]);
        }

        float m  = 0.0f;

        for(int i =0; i< N; i++)
        {
        m += expf(x[i] - max_val);
        }

        for(int i = 0; i < N; i++)
        {
            y[i] = expf(x[i] - max_val) / m;
        }
    }
    
    return;
}

__global__ void _softmax_v1(const float* __restrict__ in,
                            float* __restrict__ out,
                            int M, int N)
{
    int r = (blockIdx.x < M) ? blockIdx.x : -1;
    
    if(r == -1) return;

    int c_size = (blockDim.x < N) ? N/blockDim.x : (threadIdx.x < N) ? 1 : 0;
    if(threadIdx.x == blockDim.x - 1) c_size += N % blockDim.x;

    auto x = in + r * N + threadIdx.x * (N/blockDim.x);
    auto y = out + r * N + threadIdx.x * (N/blockDim.x);

    __shared__ float max_;
    __shared__ float m_;
    
    float max_tmp = x[0];
    float tmp_m = 0.0f;

    if(threadIdx.x == 0) 
    {
        max_ = x[0];
        m_ = 0.0f;
    }
    __syncthreads();

    for(int i =0; i<c_size; i++) 
    {
        max_tmp = fmaxf(max_tmp, x[i]);
    }

    atomicMax((int*)&max_, __float_as_int(max_tmp));
    __syncthreads();

    for(int i =0; i<c_size; i++) 
    {
        tmp_m += expf(x[i] - max_);
    }

    atomicAdd(&m_, tmp_m);
    __syncthreads();

    for(int i =0; i<c_size; i++) 
    {
        y[i] = expf(x[i] - max_) / m_;
    }

}


__global__ void _softmax_v2(const float* __restrict__ in,
                            float* __restrict__ out,
                            int M, int N)
{
    int r = (blockIdx.x < M) ? blockIdx.x : -1;
    
    if(r == -1) return;

    int c_size = (blockDim.x < N) ? N/blockDim.x : (threadIdx.x < N) ? 1 : 0;
    int lane_id = threadIdx.x % 32;

    auto x = in + r * N + threadIdx.x * (N/blockDim.x);
    auto y = out + r * N + threadIdx.x * (N/blockDim.x);

    float max_tmp = (threadIdx.x < N % blockDim.x)? x[N-1-threadIdx.x]: x[0];
    float tmp_m = (threadIdx.x < N % blockDim.x)? expf(x[N-1-threadIdx.x] - max_tmp): 0.0f;

    for(int i =0; i< c_size; i++)
        max_tmp = fmaxf(max_tmp, x[i]);
    
    for(int offset = 16; offset >0 ; offset >>= 1 ) 
        max_tmp = fmaxf(max_tmp, __shfl_down_sync(0xffffffff, max_tmp, offset));

    max_tmp = __shfl_sync(0xffffffff, max_tmp, 0);

    for(int i =0; i<c_size; i++) 
    {
        tmp_m += expf(x[i] - max_tmp);
    }

    for(int offset = 16; offset > 0; offset >>= 1) 
        tmp_m += __shfl_down_sync(0xffffffff, tmp_m, offset);

    tmp_m = __shfl_sync(0xffffffff, tmp_m, 0);

    for(int i =0; i<c_size; i++) 
    {
        y[i] = expf(x[i] - max_tmp) / tmp_m;
    }

}

__global__ void _softmax_v3(const float* __restrict__ in,
                            float* __restrict__ out,
                            int M, int N)
{
    int r = (blockIdx.x < M) ? blockIdx.x : -1;
    
    if(r == -1) return;

    int c_size = (blockDim.x < N) ? N/blockDim.x : (threadIdx.x < N) ? 1 : 0;
    int lane_id = threadIdx.x % 32;

    auto x = in + r * N + threadIdx.x * (N/blockDim.x);
    auto y = out + r * N + threadIdx.x * (N/blockDim.x);

    float max_tmp = (threadIdx.x < N % blockDim.x)? x[N-1-threadIdx.x]: x[0];
    float tmp_m = (threadIdx.x < N % blockDim.x)? expf(x[N-1-threadIdx.x] - max_tmp): 0.0f;

    for(int i =0; i< c_size/4 ; i++)
    {
        float4 val = *(float4*)&x[i*4];
        max_tmp = fmaxf(max_tmp, fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w)));
    }

    if(c_size % 4 != 0) 
    {
        for(int i = c_size - 1 - c_size % 4; i < c_size; i++)
            max_tmp = fmaxf(max_tmp, x[i]);
    }

    for(int offset = 16; offset >0 ; offset >>= 1 ) 
        max_tmp = fmaxf(max_tmp, __shfl_down_sync(0xffffffff, max_tmp, offset));

    max_tmp = __shfl_sync(0xffffffff, max_tmp, 0);

    for(int i =0; i< c_size/4 ; i++) 
    {
        float4 val = *(float4*)&x[i*4];
        tmp_m += expf(val.x - max_tmp) + expf(val.y - max_tmp) + expf(val.z - max_tmp) + expf(val.w - max_tmp);
    }

    if(c_size % 4 != 0) 
    {
        for(int i = c_size - 1 - c_size % 4; i < c_size; i++)
            tmp_m += expf(x[i] - max_tmp);
    }

    for(int offset = 16; offset > 0; offset >>= 1) 
        tmp_m += __shfl_down_sync(0xffffffff, tmp_m, offset);

    tmp_m = __shfl_sync(0xffffffff, tmp_m, 0);

    for(int i =0; i<c_size; i++) 
    {
        y[i] = expf(x[i] - max_tmp) / tmp_m;
    }

}

__global__ void _softmax_v4(const float* __restrict__ in,
                            float* __restrict__ out,
                            int M, int N)
{
    int r = (blockIdx.x < M) ? blockIdx.x : -1;
    
    if(r == -1) return;

    int c_size = (blockDim.x < N) ? N/blockDim.x : (threadIdx.x < N) ? 1 : 0;
    int lane_id = threadIdx.x % 32;

    auto x = in + r * N + threadIdx.x * (N/blockDim.x);
    auto y = out + r * N + threadIdx.x * (N/blockDim.x);

    float max_tmp = (threadIdx.x < N % blockDim.x)? x[N-1-threadIdx.x]: x[0];
    float tmp_m = (threadIdx.x < N % blockDim.x)? expf(x[N-1-threadIdx.x] - max_tmp): 0.0f;

    __shared__ float max_;
    __shared__ float m_;

    if(threadIdx.x == 0) 
    {
        max_ = x[0];
        m_ = 0.0f;
    }
    __syncthreads();

    for(int i =0; i< c_size/4 ; i++)
    {
        float4 val = *(float4*)&x[i*4];
        max_tmp = fmaxf(max_tmp, fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w)));
    }

    if(c_size % 4 != 0) 
    {
        for(int i = c_size - 1 - c_size % 4; i < c_size; i++)
            max_tmp = fmaxf(max_tmp, x[i]);
    }

    for(int offset = 16; offset >0 ; offset >>= 1 ) 
        max_tmp = fmaxf(max_tmp, __shfl_down_sync(0xffffffff, max_tmp, offset));

    max_tmp = __shfl_sync(0xffffffff, max_tmp, 0);
    
    if(lane_id == 0) atomicMax((int*)&max_, __float_as_int(max_tmp));
    __syncthreads();

    for(int i =0; i< c_size/4 ; i++) 
    {
        float4 val = *(float4*)&x[i*4];
        tmp_m += expf(val.x - max_) + expf(val.y - max_) + expf(val.z - max_) + expf(val.w - max_);
    }

    if(c_size % 4 != 0) 
    {
        for(int i = c_size - 1 - c_size % 4; i < c_size; i++)
            tmp_m += expf(x[i] - max_);
    }

    for(int offset = 16; offset > 0; offset >>= 1) 
        tmp_m += __shfl_down_sync(0xffffffff, tmp_m, offset);

    tmp_m = __shfl_sync(0xffffffff, tmp_m, 0);
    if(lane_id == 0) atomicAdd(&m_, tmp_m);
    __syncthreads();

    for(int i =0; i<c_size; i++) 
    {
        y[i] = expf(x[i] - max_) / m_;
    }

}

__global__ void _softmax_v5(const float* __restrict__ in,
                            float* __restrict__ out,
                            int M, int N)
{
    int r = (blockIdx.x < M) ? blockIdx.x : -1;
    
    if(r == -1) return;

    int c_size = (blockDim.x < N) ? N/blockDim.x : (threadIdx.x < N) ? 1 : 0;
    int lane_id = threadIdx.x % 32;

    auto x = in + r * N + threadIdx.x * (N/blockDim.x);
    auto y = out + r * N + threadIdx.x * (N/blockDim.x);

    float max_tmp = (threadIdx.x < N % blockDim.x)? x[N-1-threadIdx.x]: x[0];
    float tmp_m = (threadIdx.x < N % blockDim.x)? expf(x[N-1-threadIdx.x] - max_tmp): 0.0f;

    __shared__ float max_;
    __shared__ float m_;

    if(threadIdx.x == 0) 
    {
        max_ = x[0];
        m_ = 0.0f;
    }
    __syncthreads();

    for(int i =0; i< c_size/4 ; i++)
    {
        float4 val = *(float4*)&x[i*4];
        float new_max = fmaxf(max_tmp, fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w)));
        tmp_m *= expf(max_tmp - new_max);
        max_tmp = new_max;
        tmp_m += expf(val.x - new_max) + expf(val.y - new_max) + expf(val.z - new_max) + expf(val.w - new_max);
    }

    if(c_size % 4 != 0) 
    {
        float new_max;
        for(int i = c_size - 1 - c_size % 4; i < c_size; i++)
        {
            new_max = fmaxf(max_tmp, x[i]);
            tmp_m *= expf(max_tmp - new_max);
            max_tmp = new_max;
            tmp_m += expf(x[i] - new_max);
        }   
    }

    float warp_max = max_tmp;

    for(int offset = 16; offset >0 ; offset >>= 1 ) 
        warp_max = fmaxf(warp_max, __shfl_down_sync(0xffffffff, max_tmp, offset));

    warp_max = __shfl_sync(0xffffffff, warp_max, 0);
    tmp_m *= expf(max_tmp - warp_max);

    for(int offset = 16; offset > 0; offset >>= 1) 
        tmp_m += __shfl_down_sync(0xffffffff, tmp_m, offset);

    if(lane_id == 0) atomicMax((int*)&max_, __float_as_int(warp_max));
    __syncthreads();
    if(lane_id == 0) tmp_m *= expf(warp_max - max_);    
    if(lane_id == 0) atomicAdd(&m_, tmp_m);
    __syncthreads();

    for(int i =0; i<c_size; i++) 
    {
        y[i] = expf(x[i] - max_) / m_;
    }

}


__global__ void _softmax_v6(const float* __restrict__ in,
                            float* __restrict__ out,
                            int M, int N)
{
    int r = (blockIdx.x < M) ? blockIdx.x : -1;
    
    if(r == -1) return;

    int c_size = (blockDim.x < N) ? N/blockDim.x : (threadIdx.x < N) ? 1 : 0;
    int lane_id = threadIdx.x % 32;

    auto x = in + r * N;
    auto y = out + r * N;

    float max_tmp = (threadIdx.x < N % blockDim.x)? x[N-1-threadIdx.x]: x[0];
    float tmp_m = (threadIdx.x < N % blockDim.x)? expf(x[N-1-threadIdx.x] - max_tmp): 0.0f;

    __shared__ float max_;
    __shared__ float m_;

    if(threadIdx.x == 0) 
    {
        max_ = x[0];
        m_ = 0.0f;
    }
    __syncthreads();

    for(int i =0; i< c_size/4 ; i++)
    {
        float4 val = *(float4*)&x[blockDim.x * 4 * i + threadIdx.x * 4];
        float new_max = fmaxf(max_tmp, fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w)));
        tmp_m *= expf(max_tmp - new_max);
        max_tmp = new_max;
        tmp_m += expf(val.x - new_max) + expf(val.y - new_max) + expf(val.z - new_max) + expf(val.w - new_max);
    }

    if(c_size % 4 != 0) 
    {
        float new_max;
        for(int i = c_size - 1 - c_size % 4; i < c_size; i++)
        {
            new_max = fmaxf(max_tmp, x[i]);
            tmp_m *= expf(max_tmp - new_max);
            max_tmp = new_max;
            tmp_m += expf(x[i] - new_max);
        }   
    }

    float warp_max = max_tmp;

    for(int offset = 16; offset >0 ; offset >>= 1 ) 
        warp_max = fmaxf(warp_max, __shfl_down_sync(0xffffffff, max_tmp, offset));

    warp_max = __shfl_sync(0xffffffff, warp_max, 0);
    tmp_m *= expf(max_tmp - warp_max);

    for(int offset = 16; offset > 0; offset >>= 1) 
        tmp_m += __shfl_down_sync(0xffffffff, tmp_m, offset);

    if(lane_id == 0) atomicMax((int*)&max_, __float_as_int(warp_max));
    __syncthreads();
    if(lane_id == 0) tmp_m *= expf(warp_max - max_);    
    if(lane_id == 0) atomicAdd(&m_, tmp_m);
    __syncthreads();

    for(int i =0; i<c_size; i++) 
    {
        y[blockDim.x * i + threadIdx.x] = expf(x[blockDim.x * i + threadIdx.x] - max_) / m_;
    }

}


__global__ void _softmax_v7(const float* __restrict__ in,
                            float* __restrict__ out,
                            int M, int N)
{
    int r = (blockIdx.x < M) ? blockIdx.x : -1;
    
    if(r == -1) return;

    int c_size = (blockDim.x < N) ? N/blockDim.x : (threadIdx.x < N) ? 1 : 0;
    int lane_id = threadIdx.x % 32;

    auto x = in + r * N;
    auto y = out + r * N;

    float max_tmp = (threadIdx.x < N % blockDim.x)? x[N-1-threadIdx.x]: x[0];
    float tmp_m = (threadIdx.x < N % blockDim.x)? expf(x[N-1-threadIdx.x] - max_tmp): 0.0f;

    __shared__ float max_;
    __shared__ float m_;

    if(threadIdx.x == 0) 
    {
        max_ = x[0];
        m_ = 0.0f;
    }
    __syncthreads();

    for(int i =0; i< c_size/4 ; i++)
    {
        float4 val = *(float4*)&x[blockDim.x * 4 * i + threadIdx.x * 4];
        float new_max = fmaxf(max_tmp, fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w)));
        tmp_m *= expf(max_tmp - new_max);
        max_tmp = new_max;
        tmp_m += expf(val.x - new_max) + expf(val.y - new_max) + expf(val.z - new_max) + expf(val.w - new_max);
    }

    if(c_size % 4 != 0) 
    {
        float new_max;
        for(int i = c_size - c_size % 4; i < c_size; i++)
        {
            new_max = fmaxf(max_tmp, x[i]);
            tmp_m *= expf(max_tmp - new_max);
            max_tmp = new_max;
            tmp_m += expf(x[i] - new_max);
        }   
    }

    float warp_max = max_tmp;

    for(int offset = 16; offset >0 ; offset >>= 1 ) 
        warp_max = fmaxf(warp_max, __shfl_down_sync(0xffffffff, warp_max, offset));

    warp_max = __shfl_sync(0xffffffff, warp_max, 0);
    tmp_m *= expf(max_tmp - warp_max);

    for(int offset = 16; offset > 0; offset >>= 1) 
        tmp_m += __shfl_down_sync(0xffffffff, tmp_m, offset);

    if(lane_id == 0) atomicMax((int*)&max_, __float_as_int(warp_max));
    __syncthreads();
    if(lane_id == 0) tmp_m *= expf(warp_max - max_);    
    if(lane_id == 0) atomicAdd(&m_, tmp_m);
    __syncthreads();

    for(int i =0; i<c_size/4; i++) 
    {

        auto x4 = reinterpret_cast<const float4*>(x + blockDim.x * 4 * i + threadIdx.x * 4);
        auto y4 = reinterpret_cast<float4*>(y + blockDim.x * 4 * i + threadIdx.x * 4);
        y4->x = expf(x4->x - max_) / m_;
        y4->y = expf(x4->y - max_) / m_;
        y4->z = expf(x4->z - max_) / m_;
        y4->w = expf(x4->w - max_) / m_;
    }

    if(c_size % 4 != 0) 
    {
        for(int i = c_size - c_size % 4; i < c_size; i++)
            y[ + threadIdx.x] = expf(x[blockDim.x * i + threadIdx.x] - max_) / m_;
    }

}
