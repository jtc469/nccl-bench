#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=out/%x-%j.out
#SBATCH --error=out/%x-%j.err
#SBATCH --job-name=nccl-bench

set -e

if [[ -z "$SLURM_JOB_ID" ]]; then
    SYSTEM=$1

    source "config/systems/${SYSTEM}.env"
    mkdir -p out

    sbatch \
        "${SBATCH_ARGS[@]}" \
        --export=ALL,SYSTEM="$SYSTEM" \
        "$0"

    exit
fi

source "config/systems/${SYSTEM}.env"

module purge

for mod in "${MODULES[@]}"; do
    module load "$mod"
done

./external/build/all_reduce_perf \
    -b 8 \
    -e 1G \
    -f 2 \
    -g 2