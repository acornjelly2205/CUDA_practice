# CUDA Practice

Hand-written CUDA kernels, optimized step by step and analyzed with Nsight Compute.
Each kernel starts from a naive implementation and is improved one change at a time.

## Kernels

| # | Operation | Status |
|---|---|---|
| 1 | [Softmax](1_softmax/) | Done | 


## Methodology
- **Performance metric:** effective memory bandwidth (GB/s), compared against the theoretical peak
- **Profiling:** Nsight Compute, comparing each version with the previous one

## Environment
- GPU: NVIDIA GeForce RTX 4090 (sm_89), rented via vast.ai

## Build & Run

```bash
make                  # build all kernels
make 1_softmax        # build a single kernel by folder name
make clean
```

Benchmark against PyTorch (per kernel folder):

```bash
python3 bench_torch.py           # benchmark + edge cases + CSV/plots
python3 bench_torch.py --edge    # edge cases only
```
