## NCCL-Tests Framework for SCC Connect '26

This is a benchmarking framework for NCCL, built for the SCC Connect '26 Hackathon.

It is built to benchmark cross-gpu communication on NVIDEA (NCCL-Test) and AMD (RCCL-Test) systems.

Testing on:
- AMD Developer Cloud (aac6 & aac7)
- BlueBEAR
- NVIDEA Launchpad instance

### Instructions for use
- Clone the repo
- ./build.sh [SYSTEM]
    - Clones the benchmark source set in `config/[SYSTEM].env` into external/ and builds it
    - NVIDIA systems use the [jtc469/nccl-tests](https://github.com/jtc469/nccl-tests/tree/csv-output) `csv-output` branch, as stock nccl-tests can't write csv output
    - AMD systems use [ROCm/rccl-tests](https://github.com/ROCm/rccl-tests)
    - Extra args are passed to make, e.g. `./build.sh scc NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"`
    - Build details (repo, commit, compiler) are saved to `external/build/build-info.txt`

#### Running with slurm
- ./job.sh [SYSTEM] [LABEL] [extra benchmark args...]
    - LABEL names the config so repeated runs can be grouped (default: baseline)
    - Extra args go to every benchmark and override the defaults in job.sh
    - `NCCL_*` env vars and `LAUNCHER` (a command prefix, e.g. numactl) are passed into the job
- python3 analysis/plot_charts.py [SYSTEM] [JOB_ID] - plots for a single run
- python3 analysis/summarise.py [SYSTEM] - combines all runs, grouped by label
    - Mean/std/min/max across runs, out-of-place only
    - Summary table: best busbw, small message latency, % of NVLink peak (read from `nvlink-*.txt`, or `--peak`)
    - Error bar plots (min-max) per benchmark, one line per label
    - `--labels a b` to compare specific configs, `--exclude JOB_ID` to drop bad runs
    - Writes to `out/summary/[SYSTEM]/`

Example tuning session (aim for 3-5 runs of each config):
```bash
for i in 1 2 3 4 5; do ./job.sh scc; done             # baseline

NCCL_DEBUG=INFO ./job.sh scc debug                    # NCCL version, channels, algo/proto, NVLink path

./job.sh scc big --maxbytes 8G                        # does busbw keep rising past 1G? (check GPU memory)

NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=32 NCCL_BUFFSIZE=8388608 \
    ./job.sh scc bw-tuned --maxbytes 8G               # bandwidth config

NCCL_PROTO=LL NCCL_MAX_NCHANNELS=2 \
    ./job.sh scc lat-tuned --maxbytes 64K --iters 1000 --cudagraph 10   # latency config

LAUNCHER="numactl --cpunodebind=0 --membind=0" ./job.sh scc numa       # pin to the GPUs' NUMA node

python3 analysis/summarise.py scc
```

Each run writes to `out/raw/[SYSTEM]-[JOB_ID]/`:
- `[SYSTEM]-[BENCHMARK]-[JOB_ID].csv` - results, one row per (size, in-place)
- `[SYSTEM]-[BENCHMARK]-[JOB_ID].log` - complete, unedited benchmark output
- `metadata-[JOB_ID].csv` - system, label, node, partition, ngpus
- `run-config-[JOB_ID].txt` - label, launcher, benchmark args and every `NCCL_*` variable set
- `gpus-*`, `clocks-*`, `topo-*`, `nvlink-*` - GPU model, clocks and interconnect topology
- `build-info.txt` - which build produced the results


## NCCL Docs (NVDEA)
https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html

## TODO/Research

#### The Story
Does latency increase with ngpus? -- Why?
Do all tests scale the same way? -- Why/Why not?

#### In-place vs Out-of-place
Decided: the SCC Connect '26 rules say to report out-of-place only, so all analysis uses out-of-place.
- In-place: the input buffer is reused to hold the result, overwriting its original contents.
- Out-of-place: input and output use separate buffers. The original input remains available.

#### Multi-node benchmark
- Allocate multiple nodes.
- Use a test build with MPI support.
- Launch cooperating benchmark processes on those nodes using the cluster's MPI launcher.

Note: just changing --nodes=1 to --nodes=2 will just launch two copies, one on each node
