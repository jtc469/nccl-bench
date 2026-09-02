from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(prog="NCCL-Tests Data Loader")

parser.add_argument("system", help="System name, e.g. aac6 or bluebear")
parser.add_argument("job_id", help="Slurm job ID")

args = parser.parse_args()

system = args.system
job_id = args.job_id

benchmark_names = [
    "all_gather_perf",
    "all_reduce_perf",
    "broadcast_perf",
    "reduce_scatter_perf",
]

FILE_HEAD =  Path("out/raw") / f"{system}-{job_id}"

benchmarks = {
    name: pd.read_csv(f"{FILE_HEAD}/{system}-{name}-{job_id}.csv")
    for name in benchmark_names
}

metadata = pd.read_csv(
    f"{FILE_HEAD}/metadata-{job_id}.csv"
).iloc[0].to_dict()

for name, df in benchmarks.items():
    benchmarks[name] = df.assign(**metadata)

partition = metadata["partition"]
ngpus = metadata["ngpus"]

for b in benchmarks:

    name = b.replace("perf", "")

    Path(f"out/plots/{system}-{job_id}").mkdir(parents=True, exist_ok=True)

    df = benchmarks[b]

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

    plt.savefig(f"out/plots/{system}-{job_id}/{system}-{name}.png", dpi=200)
    plt.close()