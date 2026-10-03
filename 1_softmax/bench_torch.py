# bench_torch.py
# 실행:   python3 bench_torch.py            # 벤치마크 + edge case + CSV/그래프 저장
#         python3 bench_torch.py --edge     # edge case만
#         python3 bench_torch.py --no-plot  # 그래프 없이 CSV만
# 필요:   pip install ninja numpy matplotlib

import argparse
import os

import torch
from torch.utils.cpp_extension import load_inline

KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
BLOCKS = [256, 512, 1024]
PEAK_GBPS = 1008.0  # RTX 4090 이론 DRAM 대역폭

# 헤더 내용을 소스 문자열에 직접 넣어서, softmax.cuh가 바뀌면 load_inline 해시도 바뀌게 함
# (#include로 끌어오면 헤더 변경을 감지 못하고 오래된 빌드를 재사용할 수 있음)
with open(os.path.join(KERNEL_DIR, "softmax.cuh")) as f:
    header_src = f.read()

cuda_src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
''' + header_src + r'''

torch::Tensor my_softmax(torch::Tensor x, int64_t block) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2 && x.is_contiguous(), "x must be contiguous 2D");
    TORCH_CHECK(block >= 32 && block <= 1024 && block % 32 == 0,
                "block must be a multiple of 32 in [32, 1024]");
    const int M = x.size(0), N = x.size(1);
    auto y = torch::empty_like(x);
    auto stream = at::cuda::getCurrentCUDAStream();
    _softmax_v7<<<M, (int)block, 0, stream>>>(x.data_ptr<float>(), y.data_ptr<float>(), M, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
'''
cpp_src = "torch::Tensor my_softmax(torch::Tensor x, int64_t block);"

mod = load_inline(
    name="my_softmax_ext",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["my_softmax"],
    extra_cuda_cflags=["-O3", "-lineinfo"],
    verbose=False,
)


def auto_block(N):
    """스레드당 최소 4개(float4 하나)를 유지하면서 블록을 최대한 키우는 규칙."""
    b = (N // 4) // 32 * 32
    return max(256, min(1024, b))


def torch_softmax(x):
    return torch.softmax(x, dim=-1)


def make_impls():
    impls = [("torch", None, torch_softmax)]
    for b in BLOCKS:
        impls.append((f"v7_b{b}", b, lambda x, b=b: mod.my_softmax(x, b)))
    impls.append(("v7_auto", "auto", lambda x: mod.my_softmax(x, auto_block(x.size(1)))))
    return impls


def bench(fn, x, warmup=10, iters=100):
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


def run_edge_cases(impls):
    # 지원하지 않는 입력에서 CUDA 에러가 나면 이후 CUDA 호출이 전부 실패하므로 맨 마지막에 실행
    print("\n[edge cases]")
    cases = [
        ("N=1000",         lambda: torch.rand(256, 1000, device="cuda") * 20 - 10),
        ("N=4097",         lambda: torch.rand(256, 4097, device="cuda") * 20 - 10),
        ("all negative",   lambda: torch.rand(256, 4096, device="cuda") * 10 - 20),
        ("large [90,110]", lambda: torch.rand(256, 4096, device="cuda") * 20 + 90),
    ]
    for case_name, make in cases:
        x = make()
        ref = torch_softmax(x)
        for name, _, fn in impls:
            if name == "torch":
                continue
            try:
                out = fn(x)
                torch.cuda.synchronize()
                diff = (out - ref).abs().max().item()
                ok = "OK" if diff < 1e-5 else "MISMATCH"
                print(f"{case_name:<16} {name:<10} max_abs_diff = {diff:.3e}  {ok}")
            except RuntimeError as e:
                print(f"{case_name:<16} {name:<10} ERROR: {str(e).splitlines()[0]}")
                return


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {path}")


def save_plot(rows, out_dir, M, gpu):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib을 불러올 수 없어 그래프는 건너뜀 (pip install numpy matplotlib)")
        return

    impl_names = [n for n in dict.fromkeys(r["impl"] for r in rows) if n != "torch"]
    Ns = sorted({r["N"] for r in rows})
    lookup = {(r["impl"], r["N"]): r["speedup"] for r in rows}

    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / len(impl_names)
    max_y = 1.0
    for i, impl in enumerate(impl_names):
        xs = [j + (i - (len(impl_names) - 1) / 2) * width for j in range(len(Ns))]
        ys = [lookup[(impl, n)] for n in Ns]
        max_y = max(max_y, *ys)
        bars = ax.bar(xs, ys, width, label=impl,
                      edgecolor="black" if impl == "v7_auto" else None,
                      linewidth=1.2 if impl == "v7_auto" else 0)
        for bar, y in zip(bars, ys):
            ax.text(bar.get_x() + bar.get_width() / 2, y + 0.01, f"{y:.2f}",
                    ha="center", va="bottom", fontsize=6.5)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="torch.softmax (1.0x)")
    ax.set_xticks(range(len(Ns)))
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_ylim(0, max_y * 1.15)
    ax.set_xlabel("N (row length)")
    ax.set_ylabel("Speedup over torch.softmax (x)")
    ax.set_title(f"Softmax speedup vs. torch (M={M}, {gpu})", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()

    for ext in ("png", "svg"):
        path = os.path.join(out_dir, f"softmax_speedup.{ext}")
        fig.savefig(path, dpi=150)
        print(f"saved: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge", action="store_true", help="edge case만 실행")
    parser.add_argument("--no-plot", action="store_true", help="그래프 저장 생략")
    parser.add_argument("--out", default=os.path.join(KERNEL_DIR, "results"), help="결과 저장 폴더")
    args = parser.parse_args()

    impls = make_impls()
    if args.edge:
        run_edge_cases(impls)
        return

    torch.manual_seed(42)
    M = 65536
    Ns = [1024, 2048, 4096, 8192, 16384]
    gpu = torch.cuda.get_device_name()

    print(f"GPU: {gpu}, M = {M}\n")
    print(f"{'impl':<10} {'block':>6} {'N':>7} {'time(us)':>11} {'effBW(GB/s)':>12} {'max_abs_diff':>14}")

    rows = []
    for N in Ns:
        x = torch.rand(M, N, device="cuda") * 20 - 10
        ideal_bytes = 2 * x.numel() * x.element_size()  # 이상적 트래픽: 읽기 1번 + 쓰기 1번
        ref = torch_softmax(x)

        for name, block, fn in impls:
            out = fn(x)
            diff = 0.0 if name == "torch" else (out - ref).abs().max().item()
            del out

            ms = bench(fn, x)
            gbps = ideal_bytes / (ms * 1e-3) / 1e9
            block_str = "-" if block is None else (f"{auto_block(N)}*" if block == "auto" else str(block))
            print(f"{name:<10} {block_str:>6} {N:>7} {ms * 1e3:>11.2f} {gbps:>12.1f} {diff:>14.3e}")

            rows.append(dict(gpu=gpu, impl=name, block=block_str, M=M, N=N,
                             time_us=round(ms * 1e3, 2), eff_bw_gbps=round(gbps, 1),
                             max_abs_diff=diff))

        del x, ref
        torch.cuda.empty_cache()
        print()

    # torch 시간을 1로 놓은 speedup (값 > 1이면 torch보다 빠름)
    torch_time = {r["N"]: r["time_us"] for r in rows if r["impl"] == "torch"}
    for r in rows:
        r["speedup"] = round(torch_time[r["N"]] / r["time_us"], 3)

    print(f"{'impl':<10} " + " ".join(f"{n:>8}" for n in Ns))
    for name in dict.fromkeys(r["impl"] for r in rows):
        sp = {r["N"]: r["speedup"] for r in rows if r["impl"] == name}
        print(f"{name:<10} " + " ".join(f"{sp[n]:>7.3f}x" for n in Ns))
    print()

    os.makedirs(args.out, exist_ok=True)
    
    if not args.no_plot:
        save_plot(rows, args.out, M, gpu)

    run_edge_cases(impls)


if __name__ == "__main__":
    main()