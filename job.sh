#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --job-name=nccl-bench

set -eo pipefail


# Set up slurm environment
if [[ -z "$SLURM_JOB_ID" ]]; then
    SYSTEM=$1

    source "config/${SYSTEM}.env"
    mkdir -p out

    sbatch \
        "${SBATCH_ARGS[@]}" \
        --export=ALL,SYSTEM="$SYSTEM" \
        "$0"

    exit
fi

source "config/${SYSTEM}.env"

# Only load modules on systems that have them
if command -v module >/dev/null 2>&1; then
    module purge

    for mod in "${MODULES[@]}"; do
        module load "$mod"
    done
fi

# Log metadata (GPU hardware, partitions, ngpus, etc)
# Some systems don't set SLURM_GPUS_ON_NODE, so fall back to the config value
NGPUS=${SLURM_GPUS_ON_NODE:-$NGPUS_FALLBACK}

if [[ -z "$NGPUS" ]]; then
    echo "SLURM_GPUS_ON_NODE is unset; set NGPUS_FALLBACK in config/${SYSTEM}.env"
    exit 1
fi

RUN_DIR="out/raw/${SYSTEM}-${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

METADATA_FILE="${RUN_DIR}/metadata-${SLURM_JOB_ID}.csv"

printf '%s\n' \
    "system,job_id,node,partition,ngpus" \
    "$SYSTEM,$SLURM_JOB_ID,$SLURMD_NODENAME,$SLURM_JOB_PARTITION,$NGPUS" \
    > "$METADATA_FILE"

# Log GPU model & interconnect topology (link count sets the peak busbw)
# Logging failures shouldn't kill the benchmark run, hence || echo
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version --format=csv \
        > "${RUN_DIR}/gpus-${SLURM_JOB_ID}.csv" || echo "nvidia-smi query failed"
    nvidia-smi topo -m \
        > "${RUN_DIR}/topo-${SLURM_JOB_ID}.txt" || echo "nvidia-smi topo failed"
    nvidia-smi nvlink --status \
        > "${RUN_DIR}/nvlink-${SLURM_JOB_ID}.txt" || echo "nvidia-smi nvlink failed"
elif command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showproductname \
        > "${RUN_DIR}/gpus-${SLURM_JOB_ID}.txt" || echo "rocm-smi productname failed"
    rocm-smi --showtopo \
        > "${RUN_DIR}/topo-${SLURM_JOB_ID}.txt" || echo "rocm-smi topo failed"
fi

# Keep a record of which build produced these results
cp external/build/build-info.txt "$RUN_DIR/" || echo "no build-info.txt, was build.sh run?"


# Experiment outline:
#
# - 5 untimed warm-ups
# - 1 correctness check
# - 50 measured runs of 
# ... Repeat for each benchmark
#

BENCHMARKS=(
    all_reduce_perf
    all_gather_perf
    reduce_scatter_perf
    broadcast_perf
)

ARGS=(
    # We start with 8 bytes for small message latency,
    # build up to 1GB to check bandwidth for large messages
    
    # Always check ngpus < gres:gpus in system config
    --minbytes 8 
    --maxbytes 1G 
    --stepfactor 2 
    --ngpus "$NGPUS"

    --warmup_iters 5
    --iters 50
    --check 1
    --datatype float
)




# Complete, unedited output of each benchmark is kept alongside its csv
# (pipefail above makes a failing benchmark still fail the job through tee)
for b in "${BENCHMARKS[@]}"; do
    "./external/build/$b" \
        "${ARGS[@]}" \
        "${OUTPUT_ARGS[@]}" \
        --output_file "$RUN_DIR/${SYSTEM}-${b}-${SLURM_JOB_ID}.csv" \
        2>&1 | tee "$RUN_DIR/${SYSTEM}-${b}-${SLURM_JOB_ID}.log"
done