"""
Combine repeated runs of each config (label) into mean & spread.

Usage: python3 analysis/summarise.py [SYSTEM] [--labels ...] [--exclude JOB_ID ...] [--peak GB/s]

Reads every out/raw/[SYSTEM]-*/ run, out-of-place results only, and writes to out/summary/[SYSTEM]/:
- stats.csv    per (label, benchmark, size): runs, mean/std/min/max of time & busbw
- summary.csv  per (label, benchmark): best busbw & small message latency, % of peak
- [benchmark]-busbw.png / [benchmark]-time.png, one line per label, min-max error bars
"""

from pathlib import Path
import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt

BENCHMARKS = [
    "all_reduce_perf",
    "all_gather_perf",
    "reduce_scatter_perf",
    "broadcast_perf",
]

RAW = Path("out/raw")

parser = argparse.ArgumentParser(prog="NCCL-Tests Multi-run Summary")

parser.add_argument("system", help="System name, e.g. scc or bluebear")
parser.add_argument("--labels", nargs="+", help="Only include these labels (default: all)")
parser.add_argument("--exclude", nargs="+", type=int, default=[], help="Job IDs to leave out, e.g. aborted runs")
parser.add_argument("--peak", type=float, help="Theoretical peak busbw in GB/s (default: read from nvlink-*.txt)")

args = parser.parse_args()

system = args.system
OUT = Path("out/summary") / system


def load_runs():
    frames = []

    for meta_file in sorted(RAW.glob(f"{system}-*/metadata-*.csv")):
        run_dir = meta_file.parent
        meta = pd.read_csv(meta_file).iloc[0].to_dict()
        job_id = int(meta["job_id"])

        # Runs from before labels existed count as baseline
        label = meta.get("label", "baseline")
        if pd.isna(label):
            label = "baseline"

        if job_id in args.exclude or (args.labels and label not in args.labels):
            continue

        for b in BENCHMARKS:
            f = run_dir / f"{system}-{b}-{job_id}.csv"

            if not f.exists():
                print(f"warning: {f} missing, skipping")
                continue

            df = pd.read_csv(f).rename(columns={"cputime": "time"})
            df = df[(df["inplace"] == 0) & (df["size"] > 0)]

            wrong = pd.to_numeric(df["#wrong"], errors="coerce").fillna(0)
            if (wrong > 0).any():
                print(f"warning: {f} failed the correctness check at {(wrong > 0).sum()} sizes")

            frames.append(
                df.assign(label=label, benchmark=b, job_id=job_id)
                [["label", "benchmark", "job_id", "size", "time", "algbw", "busbw"]]
            )

    if not frames:
        raise SystemExit(f"No runs found in {RAW}/{system}-*/")

    return pd.concat(frames, ignore_index=True)


def nvlink_peak():
    # nvidia-smi nvlink --status lists "GPU n: ..." then "Link m: 25 GB/s" per link.
    # Peak is the per-direction total of a GPU's links (slowest GPU if they differ)
    for f in sorted(RAW.glob(f"{system}-*/nvlink-*.txt")):
        per_gpu = []

        for line in f.read_text().splitlines():
            if line.startswith("GPU "):
                per_gpu.append(0.0)

            m = re.search(r"Link \d+: ([\d.]+) GB/s", line)
            if m and per_gpu:
                per_gpu[-1] += float(m.group(1))

        if per_gpu and min(per_gpu) > 0:
            return min(per_gpu), f

    return None, None


def gpu_name():
    for f in sorted(RAW.glob(f"{system}-*/gpus-*.csv")):
        return pd.read_csv(f, skipinitialspace=True)["name"].iloc[0]

    return None


data = load_runs()

if args.peak:
    peak, peak_source = args.peak, "--peak"
else:
    peak, peak_source = nvlink_peak()

gpu = gpu_name()


# Per-size spread across runs
stats = (
    data.groupby(["label", "benchmark", "size"])
    .agg(
        runs=("job_id", "nunique"),
        time_mean=("time", "mean"),
        time_std=("time", "std"),
        time_min=("time", "min"),
        time_max=("time", "max"),
        busbw_mean=("busbw", "mean"),
        busbw_std=("busbw", "std"),
        busbw_min=("busbw", "min"),
        busbw_max=("busbw", "max"),
    )
    .reset_index()
)


# Headline numbers: best mean busbw (large messages), time at the smallest size (latency)
rows = []

for (label, b), g in stats.groupby(["label", "benchmark"]):
    bw = g.loc[g["busbw_mean"].idxmax()]
    lat = g.loc[g["size"].idxmin()]
    jobs = sorted(data.loc[(data["label"] == label) & (data["benchmark"] == b), "job_id"].unique())

    rows.append({
        "label": label,
        "benchmark": b.replace("_perf", ""),
        "runs": len(jobs),
        "busbw_size": int(bw["size"]),
        "busbw_mean": bw["busbw_mean"],
        "busbw_std": bw["busbw_std"],
        "busbw_min": bw["busbw_min"],
        "busbw_max": bw["busbw_max"],
        "pct_peak": 100 * bw["busbw_mean"] / peak if peak else None,
        "lat_size": int(lat["size"]),
        "lat_mean": lat["time_mean"],
        "lat_std": lat["time_std"],
        "lat_min": lat["time_min"],
        "lat_max": lat["time_max"],
        "job_ids": " ".join(str(j) for j in jobs),
    })

summary = pd.DataFrame(rows)

OUT.mkdir(parents=True, exist_ok=True)
stats.to_csv(OUT / "stats.csv", index=False)
summary.to_csv(OUT / "summary.csv", index=False)


# Plots: one line per label, error bars span min to max across runs
for b in data["benchmark"].unique():
    name = b.replace("_perf", "")
    s = stats[stats["benchmark"] == b]
    title = f"{system} {name}" + (f" ({gpu})" if gpu else "")

    for metric, ylabel, logy in [
        ("busbw", "Bus bandwidth (GB/s)", False),
        ("time", "Time (µs)", True),
    ]:
        fig, ax = plt.subplots()

        for label, g in s.groupby("label"):
            g = g.sort_values("size")
            ax.errorbar(
                g["size"],
                g[f"{metric}_mean"],
                yerr=[g[f"{metric}_mean"] - g[f"{metric}_min"], g[f"{metric}_max"] - g[f"{metric}_mean"]],
                marker="o",
                markersize=3,
                capsize=2,
                label=f"{label} (n={g['runs'].max()})",
            )

        if metric == "busbw" and peak:
            ax.axhline(peak, color="grey", linestyle="--", label=f"Peak ({peak:g} GB/s)")

        ax.set_xscale("log", base=2)
        if logy:
            ax.set_yscale("log", base=10)
        ax.set_xlabel("Message size (bytes)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n{'Bus bandwidth' if metric == 'busbw' else 'Time'}, out-of-place", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(OUT / f"{name}-{metric}.png", dpi=200)
        plt.close(fig)


# Print the headline table
pd.set_option("display.width", 200)

print(f"System: {system}" + (f", GPU: {gpu}" if gpu else ""))
print(f"Peak busbw: {peak:g} GB/s (from {peak_source})" if peak else "Peak busbw: unknown, pass --peak")
print()
print(summary.drop(columns="job_ids").round(2).to_string(index=False))
print()

print("Best per benchmark:")
for b, g in summary.groupby("benchmark"):
    bw = g.loc[g["busbw_mean"].idxmax()]
    lat = g.loc[g["lat_mean"].idxmin()]
    print(
        f"  {b:15s} busbw {bw['busbw_mean']:.2f} GB/s ({bw['label']})"
        f"   latency {lat['lat_mean']:.2f} us at {lat['lat_size']} B ({lat['label']})"
    )

print(f"\nWritten to {OUT}/")
