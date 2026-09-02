from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

system = "aac6"
job = "19434"

benchmark_names = [
    "all_gather_perf",
    "all_reduce_perf",
    "broadcast_perf",
    "reduce_scatter_perf",
]

FILE_HEAD = "out/raw"

benchmarks = {
    name: pd.read_csv(f"{FILE_HEAD}/{system}-{name}-{job}.csv")
    for name in benchmark_names
}

Path("out/plots").mkdir(parents=True, exist_ok=True)

df = benchmarks["all_reduce_perf"]

ax = df.plot(
    x="size",
    y="time",
    marker="o",
)

ax.set_xscale("log", base=2)
ax.set_xlabel("Message size (bytes)")
ax.set_ylabel("Time (µs)")
ax.set_title("AAC6 all-reduce latency")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("out/plots/test.png", dpi=200)
plt.close()