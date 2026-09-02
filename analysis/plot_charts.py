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

    name = b.replace("perf", "")

    Path(FILE_HEAD).mkdir(parents=True, exist_ok=True)

    df = benchmarks[b]

    df = df[
    (df["inplace"] == 0) & (df["size"] > 0)].sort_values("size")

    ax = df.plot(
        x="size",
        y="time",
        marker="o",
        legend=False
    )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Message size (bytes)")
    ax.set_ylabel("Time (µs)")
    ax.set_title(f"{system} {name} latency")
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