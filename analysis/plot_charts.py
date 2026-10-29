from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from load_data import load_csvs

parser = argparse.ArgumentParser(prog="NCCL-Tests Data Loader")

parser.add_argument("system", help="System name, e.g. aac6 or bluebear")
parser.add_argument("job_id", help="Slurm job ID")

args = parser.parse_args()

system = args.system
job_id = args.job_id


benchmarks, partition, ngpus = load_csvs(system, job_id)

FILE_HEAD = f"out/plots/{system}-{job_id}"

for b in benchmarks:

    name = b.replace("_perf", "")

    Path(FILE_HEAD).mkdir(parents=True, exist_ok=True)

    df = benchmarks[b]

    df = df[
    (df["inplace"] == 0) & (df["size"] > 0)].sort_values("size")

    ax = df.plot(
        x="size",
        y="time",
        marker="o",
        color="tab:blue",
        legend=False
    )

    ax2 = ax.twinx()
    df.plot(
        x="size",
        y="busbw",
        marker="s",
        color="tab:orange",
        legend=False,
        ax=ax2
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Message size (bytes)")
    ax.set_ylabel("Time (µs)", color="tab:blue")
    ax2.set_ylabel("Bus bandwidth (GB/s)", color="tab:orange")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax.set_title(f"{system} {name} latency and bus bandwidth")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # GPU metadata note
    ax.text(
    0.02,
    0.98,
    f"Partition: {partition}\nGPUs: {ngpus}",
    transform=ax.transAxes,
    verticalalignment="top",
    fontsize=8,
    bbox={
        "boxstyle": "round",
        "facecolor": "white",
        "alpha": 0.8,
    },
    )

    plt.savefig(f"{FILE_HEAD}/{system}-{name}.png", dpi=200)
    plt.close()

print(f"{len(benchmarks)} plots saved to {FILE_HEAD}")