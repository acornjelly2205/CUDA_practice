# plot_versions.py
# main.cu가 남긴 CSV로 특정 N에서 버전별 실행 시간을 그림
# 실행: python3 plot_versions.py results/softmax_versions.csv --N 16384
#       python3 plot_versions.py results/softmax_versions.csv --N 16384 --torch-csv results/softmax_bw.csv
# 필요: pip install numpy matplotlib

import argparse
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(path, N):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if int(r["N"]) == N]
    if not rows:
        raise SystemExit(f"{path}에 N={N} 결과가 없습니다")
    for r in rows:
        r["time_us"] = float(r["time_us"])
        r["label"] = re.sub(r"^\d+\.\s*", "", r["variant"])  # "11. v7_..." -> "v7_..."
    return rows


def load_torch_time(path, N):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["impl"] == "torch" and int(r["N"]) == N:
                return float(r["time_us"])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="main.cu가 생성한 CSV")
    parser.add_argument("--N", type=int, default=16384)
    parser.add_argument("--torch-csv", default=None, help="bench_torch.py의 softmax_bw.csv (torch 기준선 표시용)")
    parser.add_argument("--out", default=None, help="저장 경로 (확장자 제외), 기본은 CSV와 같은 폴더")
    args = parser.parse_args()

    rows = load_rows(args.csv, args.N)
    torch_us = load_torch_time(args.torch_csv, args.N)
    base_us = rows[0]["time_us"]                      # 첫 버전(v0) 대비 speedup
    best_us = min(r["time_us"] for r in rows)

    labels = [r["label"] for r in rows]
    times_ms = [r["time_us"] / 1e3 for r in rows]
    colors = ["tab:red" if r["time_us"] == best_us else "tab:blue" for r in rows]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.6))
    ys = list(range(len(rows)))
    ax.barh(ys, times_ms, color=colors)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()                                  # v0이 맨 위

    xmax = max(times_ms)
    for y, r, t in zip(ys, rows, times_ms):
        ax.text(t + xmax * 0.01, y, f"{t:.2f} ms  ({base_us / r['time_us']:.1f}x)",
                va="center", fontsize=7.5)

    if torch_us is not None:
        ax.axvline(torch_us / 1e3, color="black", linestyle="--", linewidth=1,
                   label=f"torch.softmax ({torch_us / 1e3:.2f} ms)")
        ax.legend(fontsize=8, loc="lower right")

    gpu = rows[0].get("gpu", "")
    M = rows[0].get("M", "")
    ax.set_xlim(0, xmax * 1.25)
    ax.set_xlabel("Execution time (ms, lower is better)")
    ax.set_title(f"Softmax versions at N={args.N} (M={M}, {gpu})\n"
                 f"speedup in parentheses is relative to the first version", fontsize=9.5)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.csv)), f"softmax_versions_N{args.N}")
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=150)
        print(f"saved: {out}.{ext}")


if __name__ == "__main__":
    main()