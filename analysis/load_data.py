from pathlib import Path
import pandas as pd

def load_csvs(system, job_id):

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

    return benchmarks, partition, ngpus

