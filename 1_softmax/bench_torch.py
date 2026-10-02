# bench_torch.py
# 실행: python bench_torch.py          (workplace 폴더에서, 1_softmax/softmax.cuh가 있는 위치 기준)
# 필요: pip install ninja

import torch
from torch.utils.cpp_extension import load_inline
import os 

KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
BLOCK = 256                # main.cu의 v6 래퍼와 같은 블록 크기로 맞추기

cuda_src = rf'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "softmax.cuh"

torch::Tensor my_softmax(torch::Tensor x) {{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2 && x.is_contiguous(), "x must be contiguous 2D");
    const int M = x.size(0), N = x.size(1);
    auto y = torch::empty_like(x);
    auto stream = at::cuda::getCurrentCUDAStream();
    // main.cu의 softmax_v7 래퍼와 같은 launch 구성으로 맞추세요
    _softmax_v7<<<M, {BLOCK}, 0, stream>>>(x.data_ptr<float>(), y.data_ptr<float>(), M, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}}
'''
cpp_src = "torch::Tensor my_softmax(torch::Tensor x);"

mod = load_inline(
    name="my_softmax_ext",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["my_softmax"],
    extra_include_paths=[KERNEL_DIR],
    extra_cuda_cflags=["-O3", "-lineinfo"],
    verbose=False,
)


def bench(fn, x, warmup=10, iters=100):
    """CUDA event로 평균 시간(ms) 측정. C++ 하네스와 같은 방식(L2 flush 없음)."""
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(x)
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / iters


def torch_softmax(x):
    return torch.softmax(x, dim=-1)


def main():

    import sys
    EDGE_ONLY = "--edge" in sys.argv
    if EDGE_ONLY:
        # 엣지 케이스 정답 검사 (torch를 정답으로 사용)
        # 커널이 지원하지 않는 N에서 misaligned address 등 CUDA 에러가 나면
        # 이후 CUDA 호출이 전부 실패하므로 맨 마지막에 둠
        print("\n[edge cases]")
        cases = [
            ("N=1000",          lambda: torch.rand(256, 1000, device="cuda") * 20 - 10),
            ("N=4097",          lambda: torch.rand(256, 4097, device="cuda") * 20 - 10),
            ("all negative",    lambda: torch.rand(256, 4096, device="cuda") * 10 - 20),
            ("large [90,110]",  lambda: torch.rand(256, 4096, device="cuda") * 20 + 90),
        ]
        for name, make in cases:
            x = make()
            try:
                out = mod.my_softmax(x)
                torch.cuda.synchronize()
                diff = (out - torch_softmax(x)).abs().max().item()
                ok = "OK" if diff < 1e-5 else "MISMATCH"
                print(f"{name:<16} max_abs_diff = {diff:.3e}  {ok}")
            except RuntimeError as e:
                print(f"{name:<16} ERROR: {str(e).splitlines()[0]}")
                break
    else:
        torch.manual_seed(42)
        M = 65536
        Ns = [1024, 2048, 4096, 8192, 16384]

        print(f"GPU: {torch.cuda.get_device_name()}, M = {M}\n")
        print(f"{'impl':<12} {'N':>7} {'time(us)':>11} {'BW(GB/s)':>10} {'max_abs_diff':>14}")

        for N in Ns:
            x = torch.rand(M, N, device="cuda") * 20 - 10      # [-10, 10)
            ideal_bytes = 2 * x.numel() * x.element_size()       # 읽기 1 + 쓰기 1

            ref = torch_softmax(x)
            mine = mod.my_softmax(x)
            diff = (mine - ref).abs().max().item()
            del ref, mine

            for name, fn in [("torch", torch_softmax), ("mine_v7", mod.my_softmax)]:
                ms = bench(fn, x)
                gbps = ideal_bytes / (ms * 1e-3) / 1e9
                d = diff if name == "mine_v7" else 0.0
                print(f"{name:<12} {N:>7} {ms * 1e3:>11.2f} {gbps:>10.1f} {d:>14.3e}")

            del x
            torch.cuda.empty_cache()

        # 엣지 케이스 정답 검사 (torch를 정답으로 사용)
        # 커널이 지원하지 않는 N에서 misaligned address 등 CUDA 에러가 나면
        # 이후 CUDA 호출이 전부 실패하므로 맨 마지막에 둠
        print("\n[edge cases]")
        cases = [
            ("N=1000",          lambda: torch.rand(256, 1000, device="cuda") * 20 - 10),
            ("N=4097",          lambda: torch.rand(256, 4097, device="cuda") * 20 - 10),
            ("all negative",    lambda: torch.rand(256, 4096, device="cuda") * 10 - 20),
            ("large [90,110]",  lambda: torch.rand(256, 4096, device="cuda") * 20 + 90),
        ]
        for name, make in cases:
            x = make()
            try:
                out = mod.my_softmax(x)
                torch.cuda.synchronize()
                diff = (out - torch_softmax(x)).abs().max().item()
                ok = "OK" if diff < 1e-5 else "MISMATCH"
                print(f"{name:<16} max_abs_diff = {diff:.3e}  {ok}")
            except RuntimeError as e:
                print(f"{name:<16} ERROR: {str(e).splitlines()[0]}")
                break


if __name__ == "__main__":
    main()