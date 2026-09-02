#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=out/%x-%j.out
#SBATCH --error=out/%x-%j.err
#SBATCH --job-name=nccl-bench

set -e

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

module purge

for mod in "${MODULES[@]}"; do
    module load "$mod"
done

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
    # All tests are across 2 GPUs
    --minbytes 8 
    --maxbytes 1G 
    --stepfactor 2 
    --ngpus 2

    --warmup_iters 5
    --iters 50
    --check 1
    --datatype float
)



for b in "${BENCHMARKS[@]}"; do
    "./external/build/$b" \
        "${ARGS[@]}" \
        --output_format csv \
        --output_file "out/raw/${SYSTEM}-${b}-${SLURM_JOB_ID}.csv"
done